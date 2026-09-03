from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_bench.events import (
    EventStorageError,
    RawEventWriter,
    load_normalized_events,
    load_raw_events,
    normalize_raw_events,
)


def test_raw_jsonl_is_append_ordered_versioned_and_immutable(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw/events.jsonl"
    utc_values = iter(
        datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=index)
        for index in range(3)
    )
    monotonic_values = iter((100, 200, 350))
    writer = RawEventWriter(
        raw_path,
        "event-run",
        task_start_ns=100,
        utc_now=lambda: next(utc_values),
        monotonic_ns=lambda: next(monotonic_values),
    )

    first = writer.emit(
        source="runner",
        event_type="run_start",
        payload={"future_field": {"nested": [1, "two", True]}},
    )
    writer.emit(source="harness", event_type="future_native_event", payload={"x": 3})
    writer.emit(source="runner", event_type="run_end", payload={"outcome": "success"})
    writer.seal()

    events = load_raw_events(raw_path)
    assert [event.sequence for event in events] == [1, 2, 3]
    assert all(event.schema_version == "1.0.0" for event in events)
    assert all(event.timestamp_utc.tzinfo == timezone.utc for event in events)
    assert first.payload["future_field"] == {"nested": [1, "two", True]}
    assert raw_path.read_bytes().count(b"\n") == 3
    with pytest.raises(ValidationError):
        first.sequence = 9  # type: ignore[misc]
    with pytest.raises(EventStorageError, match="sealed"):
        writer.emit(source="runner", event_type="late_event")


def test_raw_parser_rejects_non_increasing_sequence(tmp_path: Path) -> None:
    raw_path = tmp_path / "events.jsonl"
    writer = RawEventWriter(raw_path, "event-run", task_start_ns=0)
    writer.emit(source="runner", event_type="run_start")
    writer.emit(source="runner", event_type="run_end")
    writer.seal()
    lines = raw_path.read_bytes().splitlines(keepends=True)
    raw_path.write_bytes(lines[1] + lines[0])

    with pytest.raises(EventStorageError, match="strictly increase"):
        load_raw_events(raw_path)


def test_normalization_is_byte_deterministic_and_retains_provenance(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    timestamps = iter(
        (
            datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
        )
    )
    monotonic = iter((1_000, 2_000, 3_000))
    with RawEventWriter(
        raw_path,
        "normalize-run",
        task_start_ns=1_000,
        utc_now=lambda: next(timestamps),
        monotonic_ns=lambda: next(monotonic),
    ) as writer:
        first = writer.emit(
            source="runner",
            event_type="run_start",
            payload={"extension": "preserved"},
        )
        writer.emit(
            source="harness",
            event_type="unknown_future_event",
            payload={"native": "retained only in raw"},
        )
        last = writer.emit(
            source="runner",
            event_type="run_end",
            payload={"outcome": "success"},
        )

    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    normalize_raw_events(raw_path, first_output)
    normalize_raw_events(raw_path, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    normalized = load_normalized_events(first_output)
    assert [event.event_kind for event in normalized] == ["run_start", "run_end"]
    assert [event.sequence for event in normalized] == [1, 2]
    assert normalized[0].payload["extension"] == "preserved"
    assert normalized[0].raw_event_refs[0].raw_event_id == first.raw_event_id
    assert normalized[0].raw_event_refs[0].raw_record_digest == first.record_digest
    assert normalized[1].raw_event_refs[0].raw_event_id == last.raw_event_id
    assert all(event.schema_version == "1.0.0" for event in normalized)


def test_event_digest_detects_payload_mutation(tmp_path: Path) -> None:
    raw_path = tmp_path / "events.jsonl"
    with RawEventWriter(raw_path, "digest-run") as writer:
        writer.emit(source="runner", event_type="run_start", payload={"value": 1})
    record = json.loads(raw_path.read_text(encoding="utf-8"))
    record["payload"]["value"] = 2
    raw_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(EventStorageError, match="record_digest"):
        load_raw_events(raw_path)
