"""Deterministic reasoning/thinking evidence fixtures; no model is invoked."""

from __future__ import annotations

from pathlib import Path
import stat

from agent_bench.events import RawEventWriter, load_normalized_events, load_raw_events
from agent_bench.hermes_events import normalize_hermes_events
from agent_bench.metrics import _available, _calculate_reasoning_metrics
from agent_bench.opencode_events import normalize_opencode_events
from agent_bench.pi_events import normalize_pi_events
from agent_bench.reasoning_blocks import extract_reasoning_blocks
from agent_bench.reasoning_tokenizer import LlamaTokenizeCounter


def _events(tmp_path: Path, name: str, normalizer: object, native: list[dict[str, object]]):
    raw_path = tmp_path / f"{name}-raw.jsonl"
    writer = RawEventWriter(raw_path, name)
    writer.emit(source="runner", event_type="run_start", payload={"isolated_paths": {"workspace": "/tmp/work"}})
    for item in native:
        writer.emit(source="harness", event_type={"hermes": "hermes_session_message", "opencode": "opencode_event", "pi": "pi_event"}[name], payload={"native_event": item})
    writer.seal()
    normalized = tmp_path / f"{name}-normalized.jsonl"
    normalizer(raw_path, normalized)  # type: ignore[operator]
    return load_normalized_events(normalized)


def test_hermes_reasoning_content_wins_over_duplicate_reasoning_field(tmp_path: Path) -> None:
    events = _events(tmp_path, "hermes", normalize_hermes_events, [{
        "id": "message-1", "role": "assistant", "reasoning_content": "authoritative thought",
        "reasoning": "duplicate fallback thought", "tool_calls": [],
    }])
    blocks = extract_reasoning_blocks(events)
    assert [block.text for block in blocks] == ["authoritative thought"]
    assert next(event for event in events if event.event_kind == "reasoning").payload["source_field"] == "reasoning_content"


def test_opencode_reasoning_preserves_native_start_end_timing(tmp_path: Path) -> None:
    events = _events(tmp_path, "opencode", normalize_opencode_events, [{
        "type": "reasoning", "sessionID": "session", "part": {
            "id": "reason-1", "messageID": "message-1", "text": "inspect", "time": {"start": 1000, "end": 1750},
        },
    }])
    block = extract_reasoning_blocks(events)[0]
    assert block.text == "inspect"
    assert block.timing_provenance == "exact_native_timing_from_normalized_evidence"
    assert block.source_artifact_paths == ("normalized/events.jsonl",)
    assert block.duration_seconds == 0.75


def test_legacy_opencode_normalized_events_recover_exact_timing_from_raw(
    tmp_path: Path,
) -> None:
    """Old normalized streams are enriched in memory without rewriting them."""
    raw_path = tmp_path / "opencode-raw.jsonl"
    durations_ms = (52662, 34861, *([0] * 11))
    with RawEventWriter(raw_path, "legacy-opencode") as writer:
        writer.emit(source="runner", event_type="run_start", payload={})
        for index, duration in enumerate(durations_ms, start=1):
            start = index * 100_000
            writer.emit(
                source="harness",
                event_type="opencode_event",
                payload={"native_event": {
                    "type": "reasoning",
                    "sessionID": "session",
                    "part": {
                        "id": f"reason-{index}",
                        "messageID": f"message-{index}",
                        "text": f"reasoning block {index}",
                        "time": {"start": start, "end": start + duration},
                    },
                }},
            )
    normalized_path = tmp_path / "opencode-normalized.jsonl"
    normalize_opencode_events(raw_path, normalized_path)
    current_events = load_normalized_events(normalized_path)
    legacy_events = tuple(
        event.model_copy(update={"payload": {
            key: value
            for key, value in event.payload.items()
            if key not in {"native_reasoning_start_ms", "native_reasoning_end_ms", "timing_provenance"}
        }})
        for event in current_events
    )

    blocks = extract_reasoning_blocks(legacy_events, raw_events=load_raw_events(raw_path))
    assert len(blocks) == 13
    assert {block.timing_provenance for block in blocks} == {
        "exact_native_timing_recovered_from_historical_raw_evidence"
    }
    assert all(block.source_artifact_paths == ("raw/events.jsonl",) for block in blocks)

    metrics = _calculate_reasoning_metrics(
        legacy_events,
        _available(3176, "tokens", "api_exact"),
        _available(2012, "tokens", "api_exact"),
        raw_events=load_raw_events(raw_path),
    )
    assert metrics.reasoning_block_count.value == 13
    assert metrics.reasoning_time_total_seconds.value == 87.523
    assert metrics.max_continuous_reasoning_time_seconds.value == 52.662
    assert metrics.reasoning_time_total_seconds.provenance.source_methods == (
        "exact_native_timing_recovered_from_historical_raw_evidence",
    )
    assert metrics.reasoning_time_total_seconds.provenance.source_artifact_paths == (
        "raw/events.jsonl",
    )
    unavailable_without_raw = _calculate_reasoning_metrics(
        legacy_events,
        _available(3176, "tokens", "api_exact"),
        _available(2012, "tokens", "api_exact"),
    )
    assert unavailable_without_raw.reasoning_time_total_seconds.availability == "unavailable"
    assert unavailable_without_raw.reasoning_time_total_seconds.unavailable_reason == "source_not_exposed"
    # The timing fallback does not replace existing token values.
    assert metrics.reasoning_tokens_total.value == 3176
    assert metrics.reasoning_tokens_before_first_edit.value == 2012


def test_pi_deltas_and_terminal_copies_reconstruct_one_reasoning_block(tmp_path: Path) -> None:
    events = _events(tmp_path, "pi", normalize_pi_events, [
        {"type": "thinking_start", "responseId": "response-1"},
        {"type": "thinking_delta", "responseId": "response-1", "delta": "inspect "},
        {"type": "thinking_delta", "responseId": "response-1", "delta": "then edit"},
        {"type": "thinking_end", "responseId": "response-1"},
        {"type": "message_end", "message": {"role": "assistant", "responseId": "response-1", "content": [{"type": "thinking", "thinking": "inspect then edit"}], "stopReason": "stop"}},
    ])
    blocks = extract_reasoning_blocks(events)
    assert [block.text for block in blocks] == ["inspect then edit"]


def test_explicit_reasoning_with_zero_usage_is_not_reported_as_measured_zero(tmp_path: Path) -> None:
    events = _events(tmp_path, "hermes", normalize_hermes_events, [{
        "id": "message-1", "role": "assistant", "reasoning_content": "think", "tool_calls": [],
    }])
    metrics = _calculate_reasoning_metrics(
        events, _available(0, "tokens", "api_exact"), _available(0, "tokens", "api_exact")
    )
    assert metrics.reasoning_block_count.value == 1
    assert metrics.reasoning_chars_total.value == 5
    assert metrics.reasoning_tokens_total.availability == "unavailable"
    assert metrics.reasoning_tokens_total.unavailable_reason == "ambiguous_evidence"
    assert metrics.max_continuous_reasoning_tokens.availability == "unavailable"


def test_exact_tokenizer_enables_block_token_boundaries_without_character_estimates(tmp_path: Path) -> None:
    events = _events(tmp_path, "hermes", normalize_hermes_events, [{
        "id": "message-1", "role": "assistant", "reasoning_content": "a long thought",
        "tool_calls": [{"id": "edit-1", "function": {"name": "edit_file", "arguments": "{\"path\": \"a.py\"}"}}],
    }])
    executable, model = tmp_path / "llama-tokenize", tmp_path / "model.gguf"
    executable.write_text("#!/bin/sh\ncat >/dev/null\nprintf 'token count: 7\\n'\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    model.write_bytes(b"fixture-model")
    counter = LlamaTokenizeCounter(executable, model, "a" * 64, "dc72703")
    metrics = _calculate_reasoning_metrics(
        events, _available(0, "tokens", "api_exact"), _available(0, "tokens", "api_exact"), reasoning_tokenizer=counter
    )
    assert metrics.reasoning_chars_total.value == len("a long thought")
    assert metrics.reasoning_tokens_total.value == 7
    assert metrics.reasoning_tokens_before_first_tool.value == 7
    assert metrics.reasoning_tokens_before_first_edit.value == 7
    assert metrics.max_continuous_reasoning_tokens.value == 7
    assert metrics.reasoning_tokens_total.provenance.method == "tokenizer_reconstructed"
