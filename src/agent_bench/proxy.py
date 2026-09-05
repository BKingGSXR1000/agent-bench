"""OpenAI-compatible transparent HTTP proxy with safe exact-byte capture."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agent_bench.backend import SamplingBaseline
from agent_bench.capture import detect_empty_historical_think_blocks
from agent_bench.harness import EventSink

_SECRET_HEADER_NAMES = {
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
    "set-cookie",
}
_SECRET_JSON_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "bearer_token",
    "password",
    "secret",
    "token",
}
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
_GENERATION_PARAMETERS = (
    "model",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "seed",
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
    "reasoning_budget",
    "stop",
    "tools",
    "tool_choice",
    "stream",
)


class ProxyError(RuntimeError):
    """Raised when the capture proxy cannot be started safely."""


@dataclass(frozen=True)
class ProxyAddress:
    host: str
    port: int


@dataclass(frozen=True)
class ResponseObservations:
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    reasoning_content: str | None
    visible_answer_present: bool | None
    finish_reason: str | None
    tool_calls: tuple[object, ...]
    usage: dict[str, object] | None


class LoggingProxy:
    """Owned local proxy forwarding unmodified bodies to one upstream backend."""

    def __init__(
        self,
        *,
        upstream: ProxyAddress,
        bind: ProxyAddress,
        events: EventSink,
        sampling_baseline: SamplingBaseline,
        intended_seed: int,
        configured_max_context_tokens: int,
    ) -> None:
        state = _ProxyState(
            upstream=upstream,
            events=events,
            sampling_baseline=sampling_baseline,
            intended_seed=intended_seed,
            configured_max_context_tokens=configured_max_context_tokens,
        )
        try:
            self._server = _CaptureServer((bind.host, bind.port), state)
        except OSError as exc:
            raise ProxyError(f"cannot bind capture proxy at {bind.host}:{bind.port}: {exc}") from exc
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> ProxyAddress:
        host, port = self._server.server_address[:2]
        return ProxyAddress(str(host), int(port))

    def start(self) -> None:
        if self._thread is not None:
            raise ProxyError("capture proxy is already started")
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="agent-bench-capture-proxy",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Stop only this owned proxy server."""
        if self._thread is None:
            self._server.server_close()
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise ProxyError("owned capture proxy did not stop")
        self._thread = None

    def __enter__(self) -> LoggingProxy:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()


@dataclass
class _ProxyState:
    upstream: ProxyAddress
    events: EventSink
    sampling_baseline: SamplingBaseline
    intended_seed: int
    configured_max_context_tokens: int

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._request_index = 0

    def next_request(self) -> tuple[int, str]:
        with self._lock:
            self._request_index += 1
            index = self._request_index
        return index, f"proxy-request-{index:06d}"


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True
    # The owned proxy must be reusable immediately after its own clean
    # shutdown.  This is SO_REUSEADDR, not SO_REUSEPORT: it does not select or
    # share traffic with an existing listener, which preflight still rejects.
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: _ProxyState) -> None:
        self.state = state
        super().__init__(address, _ProxyHandler)


class _ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _CaptureServer

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._forward()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _forward(self) -> None:
        state = self.server.state
        request_index, request_id = state.next_request()
        started_ns = time.monotonic_ns()
        body = self._read_request_body()
        captured_body, body_redacted = redact_json_body(body)
        inbound_headers = redact_headers(tuple(self.headers.items()))
        parsed_request = _parse_json(body)
        observed_parameters = extract_generation_parameters(parsed_request)
        configured_parameters = {
            "temperature": state.sampling_baseline.temperature,
            "top_k": state.sampling_baseline.top_k,
            "top_p": state.sampling_baseline.top_p,
            "min_p": state.sampling_baseline.min_p,
            "seed": state.intended_seed,
        }
        mismatches = parameter_mismatches(configured_parameters, observed_parameters)
        messages = parsed_request.get("messages") if isinstance(parsed_request, dict) else None
        validation = detect_empty_historical_think_blocks(messages if isinstance(messages, list) else [])
        state.events.emit(
            source="proxy",
            event_type="llm_request",
            payload={
                "request_id": request_id,
                "request_index": request_index,
                "method": self.command,
                "endpoint": self.path,
                "headers": inbound_headers,
                "body_base64": base64.b64encode(captured_body).decode("ascii"),
                "body_sha256": hashlib.sha256(captured_body).hexdigest(),
                "body_redacted": body_redacted,
                "configured_parameters": configured_parameters,
                "observed_parameters": observed_parameters,
                "parameter_mismatches": mismatches,
                "configured_max_context_tokens": state.configured_max_context_tokens,
                "seed_transmission": _seed_status(state.intended_seed, observed_parameters),
            },
        )
        state.events.emit(
            source="proxy",
            event_type="empty_history_think_validation",
            payload={
                **validation.model_dump(mode="json", exclude={"definition_digest"}),
                "request_id": request_id,
                "scope": "request_messages_only",
                "rendered_template_validation": "pending_real_harness_capture",
            },
        )

        forwarded_headers, header_changes = _forward_request_headers(
            tuple(self.headers.items()), state.upstream
        )
        state.events.emit(
            source="proxy",
            event_type="proxy_upstream_request",
            payload={
                "request_id": request_id,
                "method": self.command,
                "endpoint": self.path,
                "body_base64": base64.b64encode(captured_body).decode("ascii"),
                "body_sha256": hashlib.sha256(captured_body).hexdigest(),
                "body_redacted": body_redacted,
                "headers": redact_headers(tuple(forwarded_headers.items())),
                "routing_header_changes": header_changes,
                "body_transformed": False,
            },
        )

        connection = http.client.HTTPConnection(
            state.upstream.host,
            state.upstream.port,
            timeout=3600,
        )
        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=forwarded_headers,
            )
            response = connection.getresponse()
            self._relay_response(response, request_id, started_ns)
        except Exception as exc:
            state.events.emit(
                source="proxy",
                event_type="backend_error",
                payload={
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            error_body = b'{"error":{"message":"upstream backend unavailable"}}\n'
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(error_body)
            self.close_connection = True
        finally:
            connection.close()

    def _relay_response(
        self,
        response: http.client.HTTPResponse,
        request_id: str,
        started_ns: int,
    ) -> None:
        state = self.server.state
        headers = tuple(response.getheaders())
        content_type = response.getheader("Content-Type", "")
        streaming = "text/event-stream" in content_type.lower() or response.chunked
        self.send_response(response.status, response.reason)
        for name, value in headers:
            lowered = name.lower()
            if lowered in _HOP_BY_HOP:
                continue
            if streaming and lowered == "content-length":
                continue
            self.send_header(name, value)
        if streaming:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

        chunks: list[bytes] = []
        chunk_index = 0
        while True:
            chunk = response.read1(8192)
            if not chunk:
                break
            chunk_index += 1
            chunks.append(chunk)
            captured_chunk, redacted = redact_json_or_sse_chunk(chunk)
            state.events.emit(
                source="proxy",
                event_type="proxy_response_chunk",
                payload={
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "chunk_base64": base64.b64encode(captured_chunk).decode("ascii"),
                    "chunk_sha256": hashlib.sha256(captured_chunk).hexdigest(),
                    "chunk_redacted": redacted,
                },
            )
            self.wfile.write(chunk)
            self.wfile.flush()

        body = b"".join(chunks)
        captured_body, body_redacted = redact_json_body(body)
        observations = extract_response_observations(body, content_type)
        payload: dict[str, Any] = {
            "request_id": request_id,
            "response_id": f"{request_id}:response",
            "outcome": "success" if 200 <= response.status < 400 else "error",
            "http_status": response.status,
            "headers": redact_headers(headers),
            "body_base64": base64.b64encode(captured_body).decode("ascii"),
            "body_sha256": hashlib.sha256(captured_body).hexdigest(),
            "body_redacted": body_redacted,
            "streaming": streaming,
            "chunk_count": chunk_index,
            "duration_ns": time.monotonic_ns() - started_ns,
            "finish_reason": observations.finish_reason,
            "reasoning_content": observations.reasoning_content,
            "visible_answer_present": observations.visible_answer_present,
            "tool_calls": list(observations.tool_calls),
            "usage": observations.usage,
            "token_source": "api_exact",
        }
        if observations.input_tokens is not None:
            payload["input_tokens"] = observations.input_tokens
        if observations.output_tokens is not None:
            payload["output_tokens"] = observations.output_tokens
        if observations.reasoning_tokens is not None:
            payload["reasoning_tokens"] = observations.reasoning_tokens
        state.events.emit(source="proxy", event_type="llm_response", payload=payload)

    def _read_request_body(self) -> bytes:
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            return b""
        length = int(length_text)
        return self.rfile.read(length)


def redact_headers(headers: tuple[tuple[str, str], ...]) -> dict[str, str]:
    """Return safe headers with credentials removed before persistence."""
    return {
        name: "[REDACTED]" if name.lower() in _SECRET_HEADER_NAMES else value
        for name, value in headers
    }


def redact_json_body(body: bytes) -> tuple[bytes, bool]:
    """Preserve exact bytes unless recognized structured secret fields exist."""
    parsed = _parse_json(body)
    if parsed is None:
        return body, False
    redacted, changed = _redact_value(parsed)
    if not changed:
        return body, False
    return (
        json.dumps(
            redacted,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"),
        True,
    )


def redact_json_or_sse_chunk(chunk: bytes) -> tuple[bytes, bool]:
    """Redact complete JSON bodies; SSE chunks are retained as observed bytes."""
    stripped = chunk.lstrip()
    if stripped.startswith((b"{", b"[")):
        return redact_json_body(chunk)
    return chunk, False


def extract_generation_parameters(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {name: value[name] for name in _GENERATION_PARAMETERS if name in value}


def parameter_mismatches(
    configured: dict[str, object], observed: dict[str, object]
) -> list[dict[str, object]]:
    """Record fixed sampling and seed omissions/overrides without rejecting profiles."""
    mismatches: list[dict[str, object]] = []
    for name, expected in configured.items():
        if name not in observed:
            mismatches.append(
                {"parameter": name, "expected": expected, "observed": None, "status": "not_transmitted"}
            )
        elif observed[name] != expected:
            mismatches.append(
                {"parameter": name, "expected": expected, "observed": observed[name], "status": "overridden"}
            )
    return mismatches


def extract_response_observations(body: bytes, content_type: str) -> ResponseObservations:
    """Extract only explicit OpenAI-compatible response fields."""
    objects: list[dict[str, object]] = []
    if "text/event-stream" in content_type.lower() or body.lstrip().startswith(b"data:"):
        for line in body.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            parsed = _parse_json(data)
            if isinstance(parsed, dict):
                objects.append(parsed)
    else:
        parsed = _parse_json(body)
        if isinstance(parsed, dict):
            objects.append(parsed)

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    reasoning_parts: list[str] = []
    visible_parts: list[str] = []
    response_message_seen = False
    tool_calls: list[object] = []
    usage_value: dict[str, object] | None = None
    for item in objects:
        usage = item.get("usage")
        if isinstance(usage, dict):
            usage_value = usage
            input_tokens = _exact_int(usage.get("prompt_tokens"), input_tokens)
            output_tokens = _exact_int(usage.get("completion_tokens"), output_tokens)
            details = usage.get("completion_tokens_details")
            if isinstance(details, dict):
                reasoning_tokens = _exact_int(details.get("reasoning_tokens"), reasoning_tokens)
        choices = item.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            candidate_finish = choice.get("finish_reason")
            if isinstance(candidate_finish, str):
                finish_reason = candidate_finish
            message = choice.get("message")
            delta = choice.get("delta")
            for value in (message, delta):
                if not isinstance(value, dict):
                    continue
                response_message_seen = True
                reasoning = value.get("reasoning_content")
                if isinstance(reasoning, str):
                    reasoning_parts.append(reasoning)
                content = value.get("content")
                if isinstance(content, str):
                    visible_parts.append(content)
                calls = value.get("tool_calls")
                if isinstance(calls, list):
                    tool_calls.extend(calls)
    return ResponseObservations(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        reasoning_content="".join(reasoning_parts) or None,
        visible_answer_present=(bool("".join(visible_parts)) if response_message_seen else None),
        finish_reason=finish_reason,
        tool_calls=tuple(tool_calls),
        usage=usage_value,
    )


def _forward_request_headers(
    headers: tuple[tuple[str, str], ...], upstream: ProxyAddress
) -> tuple[dict[str, str], list[dict[str, str]]]:
    forwarded = {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP and name.lower() != "host"
    }
    old_host = next((value for name, value in headers if name.lower() == "host"), "")
    new_host = f"{upstream.host}:{upstream.port}"
    forwarded["Host"] = new_host
    return forwarded, [{"header": "Host", "from": old_host, "to": new_host}]


def _redact_value(value: object) -> tuple[object, bool]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        changed = False
        for key, item in value.items():
            if str(key).lower() in _SECRET_JSON_KEYS:
                result[str(key)] = "[REDACTED]"
                changed = True
            else:
                result[str(key)], child_changed = _redact_value(item)
                changed = changed or child_changed
        return result, changed
    if isinstance(value, list):
        result_list: list[object] = []
        changed = False
        for item in value:
            child, child_changed = _redact_value(item)
            result_list.append(child)
            changed = changed or child_changed
        return result_list, changed
    return value, False


def _parse_json(body: bytes) -> object | None:
    if not body:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _seed_status(intended: int, observed: dict[str, object]) -> str:
    if "seed" not in observed:
        return "not_transmitted"
    return "matched" if observed["seed"] == intended else "overridden"


def _exact_int(value: object, existing: int | None) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return existing
