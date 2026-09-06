from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_bench.backend import SamplingBaseline
from agent_bench.events import RawEventWriter, load_raw_events, normalize_raw_events
from agent_bench.proxy import (
    LoggingProxy,
    ProxyAddress,
    _CaptureServer,
    extract_response_observations,
    redact_json_body,
)


def _captured_server_error(exception: BaseException) -> str:
    """Exercise ThreadingHTTPServer's error boundary without opening a socket."""
    server = object.__new__(_CaptureServer)
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            raise exception
        except BaseException:
            server.handle_error(None, ("127.0.0.1", 0))
    return stderr.getvalue()


def test_capture_proxy_suppresses_only_expected_client_disconnect_tracebacks() -> None:
    for exception in (ConnectionResetError("reset"), BrokenPipeError("broken"), ConnectionAbortedError("aborted")):
        assert _captured_server_error(exception) == ""

    unexpected = _captured_server_error(RuntimeError("unexpected server failure"))
    assert "Exception occurred during processing of request" in unexpected
    assert "RuntimeError: unexpected server failure" in unexpected


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    received_body = b""
    received_authorization = ""

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        type(self).received_body = self.rfile.read(length)
        type(self).received_authorization = self.headers.get("Authorization", "")
        if self.path.endswith("/stream"):
            chunks = (
                b'data: {"choices":[{"delta":{"reasoning_content":"think "},"finish_reason":null}]}\n\n',
                b'data: {"choices":[{"delta":{"reasoning_content":"carefully"},"finish_reason":"stop"}],"usage":{"prompt_tokens":21,"completion_tokens":8}}\n\n',
                b"data: [DONE]\n\n",
            )
            body = b"".join(chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()
            return
        response = {
            "id": "chatcmpl-fixture",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "reasoning_content": "inspect the repository",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 11},
            },
        }
        body = json.dumps(response, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Fixture", "yes")
        self.end_headers()
        self.wfile.write(body)


def _start_upstream() -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _request(proxy: LoggingProxy, path: str, body: bytes) -> tuple[int, bytes, dict[str, str]]:
    connection = http.client.HTTPConnection(proxy.address.host, proxy.address.port, timeout=5)
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer top-secret"},
    )
    response = connection.getresponse()
    response_body = response.read()
    headers = dict(response.getheaders())
    status = response.status
    connection.close()
    return status, response_body, headers


def _proxy_exchange(tmp_path: Path, path: str, request_body: bytes):
    upstream, thread = _start_upstream()
    raw_path = tmp_path / "raw.jsonl"
    writer = RawEventWriter(raw_path, "proxy-test")
    proxy = LoggingProxy(
        upstream=ProxyAddress("127.0.0.1", upstream.server_address[1]),
        bind=ProxyAddress("127.0.0.1", 0),
        events=writer,
        sampling_baseline=SamplingBaseline(),
        intended_seed=1001,
        configured_max_context_tokens=107520,
    )
    try:
        proxy.start()
        response = _request(proxy, path, request_body)
    finally:
        proxy.shutdown()
        writer.seal()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
    return response, load_raw_events(raw_path), raw_path


def test_non_streaming_proxy_preserves_semantics_and_captures_exact_fields(
    tmp_path: Path,
) -> None:
    request = {
        "model": "qwen",
        "messages": [{"role": "user", "content": "inspect"}],
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "seed": 1001,
        "max_completion_tokens": 4000,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "tool_choice": "auto",
        "stream": False,
    }
    request_body = json.dumps(request, separators=(",", ":")).encode()
    (status, response_body, headers), events, raw_path = _proxy_exchange(
        tmp_path, "/v1/chat/completions", request_body
    )

    assert status == 200
    assert headers["X-Fixture"] == "yes"
    assert _UpstreamHandler.received_body == request_body
    assert _UpstreamHandler.received_authorization == "Bearer top-secret"
    request_event = next(event for event in events if event.event_type == "llm_request")
    assert base64.b64decode(request_event.payload["body_base64"]) == request_body
    assert request_event.payload["headers"]["Authorization"] == "[REDACTED]"  # type: ignore[index]
    assert request_event.payload["observed_parameters"]["max_completion_tokens"] == 4000  # type: ignore[index]
    assert request_event.payload["observed_parameters"]["model"] == "qwen"  # type: ignore[index]
    assert request_event.payload["observed_parameters"]["stream"] is False  # type: ignore[index]
    assert request_event.payload["observed_parameters"]["tool_choice"] == "auto"  # type: ignore[index]
    assert request_event.payload["observed_parameters"]["tools"] == request["tools"]  # type: ignore[index]
    assert request_event.payload["parameter_mismatches"] == []
    response_event = next(event for event in events if event.event_type == "llm_response")
    assert base64.b64decode(response_event.payload["body_base64"]) == response_body
    assert response_event.payload["input_tokens"] == 123
    assert response_event.payload["output_tokens"] == 17
    assert response_event.payload["reasoning_tokens"] == 11
    assert response_event.payload["reasoning_content"] == "inspect the repository"
    assert response_event.payload["finish_reason"] == "tool_calls"
    assert response_event.payload["tool_calls"][0]["id"] == "call-1"  # type: ignore[index]

    normalized_path = tmp_path / "normalized.jsonl"
    normalized = normalize_raw_events(raw_path, normalized_path)
    assert [event.event_kind for event in normalized] == ["llm_request", "llm_response"]
    assert normalized[-1].payload["token_source"] == "api_exact"


def test_streaming_proxy_preserves_sse_bytes_and_reconstructs_observations(
    tmp_path: Path,
) -> None:
    request_body = json.dumps(
        {
            "messages": [{"role": "user", "content": "stream"}],
            "temperature": 0.7,
            "stream": True,
        },
        separators=(",", ":"),
    ).encode()
    (status, response_body, _), events, _ = _proxy_exchange(
        tmp_path, "/v1/chat/completions/stream", request_body
    )

    assert status == 200
    chunks = [event for event in events if event.event_type == "proxy_response_chunk"]
    assert b"".join(base64.b64decode(event.payload["chunk_base64"]) for event in chunks) == response_body
    response = next(event for event in events if event.event_type == "llm_response")
    assert response.payload["streaming"] is True
    assert response.payload["reasoning_content"] == "think carefully"
    assert response.payload["finish_reason"] == "stop"
    assert response.payload["input_tokens"] == 21
    assert response.payload["output_tokens"] == 8
    request = next(event for event in events if event.event_type == "llm_request")
    mismatch_names = {item["parameter"] for item in request.payload["parameter_mismatches"]}  # type: ignore[index]
    assert {"temperature", "top_k", "top_p", "min_p", "seed"} == mismatch_names
    assert request.payload["seed_transmission"] == "not_transmitted"


def test_secret_json_fields_are_redacted_before_capture() -> None:
    original = b'{"messages":[],"api_key":"do-not-store","nested":{"token":"hidden"}}'
    captured, changed = redact_json_body(original)
    assert changed
    assert b"do-not-store" not in captured
    assert b"hidden" not in captured
    assert json.loads(captured)["api_key"] == "[REDACTED]"


def test_context_usage_reasoning_and_finish_extraction_is_exact_only() -> None:
    body = b'{"choices":[{"finish_reason":"stop","message":{"reasoning_content":"r"}}],"usage":{"prompt_tokens":9,"completion_tokens":4}}'
    observed = extract_response_observations(body, "application/json")
    assert observed.input_tokens == 9
    assert observed.output_tokens == 4
    assert observed.reasoning_tokens is None
    assert observed.reasoning_content == "r"
    assert observed.finish_reason == "stop"


def test_response_visible_answer_presence_is_explicit_for_final_response_metrics() -> None:
    reasoning_only = extract_response_observations(
        b'{"choices":[{"finish_reason":"length","message":{"content":"","reasoning_content":"r","tool_calls":[]}}]}',
        "application/json",
    )
    visible = extract_response_observations(
        b'{"choices":[{"finish_reason":"stop","message":{"content":"done","reasoning_content":"r"}}]}',
        "application/json",
    )
    unavailable = extract_response_observations(b'{"usage":{"prompt_tokens":1}}', "application/json")
    assert reasoning_only.visible_answer_present is False
    assert reasoning_only.finish_reason == "length"
    assert visible.visible_answer_present is True
    assert unavailable.visible_answer_present is None


def test_owned_proxy_can_reuse_one_fixed_port_sequentially(tmp_path: Path) -> None:
    """A clean owned proxy shutdown must not require an arbitrary delay."""
    upstream, thread = _start_upstream()
    try:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        for index in range(8):
            writer = RawEventWriter(tmp_path / f"raw-{index}.jsonl", f"proxy-{index}")
            proxy = LoggingProxy(
                upstream=ProxyAddress("127.0.0.1", upstream.server_address[1]),
                bind=ProxyAddress("127.0.0.1", port), events=writer,
                sampling_baseline=SamplingBaseline(), intended_seed=1001,
                configured_max_context_tokens=107520,
            )
            try:
                proxy.start()
                assert _request(proxy, "/v1/chat/completions", b'{"messages":[]}')[0] == 200
            finally:
                proxy.shutdown()
                writer.seal()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
