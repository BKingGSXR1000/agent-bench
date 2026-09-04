"""Versioned raw and normalized JSONL event records."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_bench.models import Identifier, JsonMapping, Sha256, canonical_sha256

EventSource = Literal[
    "runner",
    "harness",
    "backend",
    "proxy",
    "git",
    "system",
    "stdout",
    "stderr",
    "hardware",
]
EventKind = Literal[
    "run_start",
    "run_end",
    "llm_request",
    "llm_response",
    "reasoning",
    "tool_call_start",
    "tool_call_end",
    "file_read",
    "file_search",
    "file_edit",
    "file_write",
    "shell_command",
    "test_execution",
    "compaction_start",
    "compaction_end",
    "output_truncation",
    "context_overflow",
    "harness_error",
    "backend_error",
    "timeout",
    "process_termination",
]
Confidence = Literal["direct", "parsed", "deterministically_reconstructed"]

NORMALIZER_NAME = "agent-bench-common"
NORMALIZER_VERSION = "1.0.0"

_NORMALIZED_EVENT_TYPES: dict[str, EventKind] = {
    event_kind: event_kind
    for event_kind in (
        "run_start",
        "run_end",
        "llm_request",
        "llm_response",
        "reasoning",
        "tool_call_start",
        "tool_call_end",
        "file_read",
        "file_search",
        "file_edit",
        "file_write",
        "shell_command",
        "test_execution",
        "compaction_start",
        "compaction_end",
        "output_truncation",
        "context_overflow",
        "harness_error",
        "backend_error",
        "timeout",
        "process_termination",
    )
}
NORMALIZER_CONFIGURATION_DIGEST = canonical_sha256(
    {
        "normalizer": NORMALIZER_NAME,
        "version": NORMALIZER_VERSION,
        "event_type_mapping": _NORMALIZED_EVENT_TYPES,
    }
)


class EventStorageError(RuntimeError):
    """Raised when an event stream cannot be safely written or validated."""


class RawEvent(BaseModel):
    """Immutable append-only envelope around one captured source event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    raw_event_id: str = Field(min_length=1)
    run_id: Identifier
    sequence: int = Field(ge=1)
    timestamp_utc: datetime
    elapsed_seconds: float | None = Field(default=None, ge=0)
    elapsed_ns: int | None = Field(default=None, ge=0)
    source: EventSource
    event_type: str = Field(min_length=1)
    payload: JsonMapping = Field(default_factory=dict)
    record_digest: Sha256

    @field_validator("timestamp_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp_utc must include a UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_elapsed_and_digest(self) -> RawEvent:
        if (self.elapsed_seconds is None) != (self.elapsed_ns is None):
            raise ValueError("elapsed_seconds and elapsed_ns must be present together")
        if self.elapsed_ns is not None and self.elapsed_seconds != (
            self.elapsed_ns / 1_000_000_000
        ):
            raise ValueError("elapsed_seconds must exactly represent elapsed_ns")
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"record_digest"})
        )
        if self.record_digest != expected:
            raise ValueError("record_digest does not match raw event content")
        return self

    @classmethod
    def create(cls, **values: object) -> RawEvent:
        """Construct a raw event and bind its digest to canonical content."""
        content = {"schema_version": "1.0.0", **values}
        draft = cls.model_construct(**content, record_digest="0" * 64)
        canonical_content = draft.model_dump(
            mode="json", exclude={"record_digest"}
        )
        return cls.model_validate(
            {
                **canonical_content,
                "record_digest": canonical_sha256(canonical_content),
            }
        )


class RawEventReference(BaseModel):
    """Integrity-bearing provenance link to one raw record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    raw_event_id: str = Field(min_length=1)
    raw_sequence: int = Field(ge=1)
    raw_record_digest: Sha256


class NormalizedEvent(BaseModel):
    """Immutable harness-independent event derived from raw evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    event_id: str = Field(min_length=1)
    run_id: Identifier
    event_kind: EventKind
    sequence: int = Field(ge=1)
    timestamp_utc: datetime
    elapsed_seconds: float | None = Field(default=None, ge=0)
    elapsed_ns: int | None = Field(default=None, ge=0)
    clock_source: Literal["runner_monotonic", "harness_wall_clock"] = (
        "runner_monotonic"
    )
    raw_event_refs: tuple[RawEventReference, ...] = Field(min_length=1)
    normalizer_name: str = Field(default=NORMALIZER_NAME, min_length=1)
    normalizer_version: str = Field(default=NORMALIZER_VERSION, min_length=1)
    normalizer_configuration_digest: Sha256 = NORMALIZER_CONFIGURATION_DIGEST
    confidence: Confidence
    payload: JsonMapping = Field(default_factory=dict)
    event_digest: Sha256

    @field_validator("timestamp_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp_utc must include a UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_elapsed_and_digest(self) -> NormalizedEvent:
        if (self.elapsed_seconds is None) != (self.elapsed_ns is None):
            raise ValueError("elapsed_seconds and elapsed_ns must be present together")
        if self.elapsed_ns is not None and self.elapsed_seconds != (
            self.elapsed_ns / 1_000_000_000
        ):
            raise ValueError("elapsed_seconds must exactly represent elapsed_ns")
        expected = canonical_sha256(
            self.model_dump(mode="json", exclude={"event_digest"})
        )
        if self.event_digest != expected:
            raise ValueError("event_digest does not match normalized event content")
        return self

    @classmethod
    def create(cls, **values: object) -> NormalizedEvent:
        """Construct a normalized event with a canonical integrity digest."""
        content = {"schema_version": "1.0.0", **values}
        draft = cls.model_construct(**content, event_digest="0" * 64)
        canonical_content = draft.model_dump(mode="json", exclude={"event_digest"})
        return cls.model_validate(
            {
                **canonical_content,
                "event_digest": canonical_sha256(canonical_content),
            }
        )


class RawEventWriter:
    """Create one new raw JSONL stream and append sequenced events to it."""

    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        task_start_ns: int | None = None,
        utc_now: Callable[[], datetime] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("xb")
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self.task_start_ns = (
            self._monotonic_ns() if task_start_ns is None else task_start_ns
        )
        self._sequence = 0
        self._closed = False
        self._lock = threading.Lock()

    def reset_task_start(self, task_start_ns: int) -> None:
        """Set the timing origin before the first event is appended."""
        with self._lock:
            if self._closed or self._sequence:
                raise EventStorageError(
                    "task timing origin can only be reset before the first event"
                )
            self.task_start_ns = task_start_ns

    def emit(
        self,
        *,
        source: EventSource,
        event_type: str,
        payload: JsonMapping | None = None,
        timed: bool = True,
    ) -> RawEvent:
        """Append one immutable event with the next capture sequence."""
        with self._lock:
            if self._closed:
                raise EventStorageError("raw event stream is sealed")
            sequence = self._sequence + 1
            elapsed_ns = max(0, self._monotonic_ns() - self.task_start_ns) if timed else None
            event = RawEvent.create(
                raw_event_id=f"{self.run_id}:raw:{sequence:06d}",
                run_id=self.run_id,
                sequence=sequence,
                timestamp_utc=self._utc_now(),
                elapsed_seconds=(
                    elapsed_ns / 1_000_000_000 if elapsed_ns is not None else None
                ),
                elapsed_ns=elapsed_ns,
                source=source,
                event_type=event_type,
                payload=payload or {},
            )
            self._stream.write(_json_line(event))
            self._stream.flush()
            self._sequence = sequence
            return event

    def seal(self) -> None:
        """Flush, sync, and close the stream; later emissions are rejected."""
        with self._lock:
            if self._closed:
                return
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._closed = True

    def __enter__(self) -> RawEventWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.seal()


def load_raw_events(path: Path) -> tuple[RawEvent, ...]:
    """Parse and validate one raw JSONL stream in capture order."""
    records = _load_jsonl(path, RawEvent)
    previous_sequence = 0
    run_id: str | None = None
    for event in records:
        if event.sequence <= previous_sequence:
            raise EventStorageError("raw event sequence numbers must strictly increase")
        if run_id is not None and event.run_id != run_id:
            raise EventStorageError("raw event stream contains multiple run IDs")
        previous_sequence = event.sequence
        run_id = event.run_id
    return records


def load_normalized_events(path: Path) -> tuple[NormalizedEvent, ...]:
    """Parse and validate one normalized JSONL stream."""
    records = _load_jsonl(path, NormalizedEvent)
    for expected, event in enumerate(records, start=1):
        if event.sequence != expected:
            raise EventStorageError("normalized sequence must be contiguous and one-based")
    return records


@dataclass(frozen=True)
class DerivedEvent:
    """One normalized event deterministically derived from one raw record."""

    event_kind: EventKind
    payload: JsonMapping
    confidence: Confidence = "direct"
    timestamp_utc: datetime | None = None
    elapsed_ns: int | None = None
    clock_source: Literal["runner_monotonic", "harness_wall_clock"] = (
        "runner_monotonic"
    )


RawEventTransformer = Callable[[RawEvent], tuple[DerivedEvent, ...]]


def normalize_raw_events(
    raw_path: Path,
    normalized_path: Path,
    *,
    transformer: RawEventTransformer | None = None,
    normalizer_name: str = NORMALIZER_NAME,
    normalizer_version: str = NORMALIZER_VERSION,
    normalizer_configuration_digest: str = NORMALIZER_CONFIGURATION_DIGEST,
) -> tuple[NormalizedEvent, ...]:
    """Deterministically normalize supported raw records into a new JSONL file."""
    raw_events = load_raw_events(raw_path)
    destination = normalized_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized: list[NormalizedEvent] = []
    with destination.open("xb") as stream:
        for raw_event in raw_events:
            if transformer is None:
                event_kind = _NORMALIZED_EVENT_TYPES.get(raw_event.event_type)
                derived = (
                    DerivedEvent(event_kind=event_kind, payload=raw_event.payload),
                ) if event_kind is not None else ()
            else:
                derived = transformer(raw_event)
            for item in derived:
                sequence = len(normalized) + 1
                elapsed_ns = (
                    raw_event.elapsed_ns
                    if item.timestamp_utc is None and item.elapsed_ns is None
                    else item.elapsed_ns
                )
                event = NormalizedEvent.create(
                    event_id=f"{raw_event.run_id}:normalized:{sequence:06d}",
                    run_id=raw_event.run_id,
                    event_kind=item.event_kind,
                    sequence=sequence,
                    timestamp_utc=item.timestamp_utc or raw_event.timestamp_utc,
                    elapsed_seconds=(
                        elapsed_ns / 1_000_000_000
                        if elapsed_ns is not None
                        else None
                    ),
                    elapsed_ns=elapsed_ns,
                    clock_source=item.clock_source,
                    raw_event_refs=(
                        RawEventReference(
                            raw_event_id=raw_event.raw_event_id,
                            raw_sequence=raw_event.sequence,
                            raw_record_digest=raw_event.record_digest,
                        ),
                    ),
                    normalizer_name=normalizer_name,
                    normalizer_version=normalizer_version,
                    normalizer_configuration_digest=normalizer_configuration_digest,
                    confidence=item.confidence,
                    payload=item.payload,
                )
                stream.write(_json_line(event))
                normalized.append(event)
        stream.flush()
        os.fsync(stream.fileno())
    return tuple(normalized)


def _load_jsonl(path: Path, model: type[RawEvent] | type[NormalizedEvent]):
    records = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise EventStorageError(f"cannot read event stream {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            records.append(model.model_validate_json(line))
        except Exception as exc:
            raise EventStorageError(
                f"invalid event at {path}:{line_number}: {exc}"
            ) from exc
    return tuple(records)


def _json_line(record: BaseModel) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
