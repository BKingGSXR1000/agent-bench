"""Immutable timing-semantics analysis derived from sealed run evidence.

This layer deliberately separates execution-clock measurements from a harness
or proxy merely observing an event.  It exists because the two are useful but
must never be compared as though they were the same measurement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_bench.events import RawEvent, load_normalized_events, load_raw_events
from agent_bench.models import Identifier, Sha256, canonical_sha256
from agent_bench.preservation import verify_artifact
from agent_bench.runner import NORMALIZED_EVENTS_PATH, RAW_EVENTS_PATH, RUN_MANIFEST_PATH, RunManifest

TIMING_PROVENANCE_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
TIMING_PROVENANCE_CONFIGURATION = {
    "version": TIMING_PROVENANCE_SCHEMA_VERSION,
    "hermes_native_messages": "sqlite-exported-message-timestamps-v1",
    "execution_timing_rule": "explicit-execution-boundary-only-v1",
}
TIMING_PROVENANCE_CONFIGURATION_DIGEST = canonical_sha256(
    TIMING_PROVENANCE_CONFIGURATION
)


class TimingProvenanceError(RuntimeError):
    """Raised when sealed evidence cannot support this analysis."""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["1.0.0"] = TIMING_PROVENANCE_SCHEMA_VERSION


class TimingValue(_Record):
    value: float | None
    units: Literal["seconds"] = "seconds"
    availability: Literal["available", "unavailable"]
    semantics: Literal[
        "harness_tool_execution_start",
        "harness_tool_execution_end",
        "tool_event_observed",
        "model_tool_call_observed",
    ]
    method: Literal["runner_monotonic", "proxy_observation", "not_available"]
    source_event_ids: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "TimingValue":
        if self.availability == "available":
            if self.value is None or self.unavailable_reason is not None:
                raise ValueError("available timing value requires a value and no reason")
        elif self.value is not None or self.unavailable_reason is None:
            raise ValueError("unavailable timing value requires null and a reason")
        return self


class HermesToolTiming(_Record):
    ordinal: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    native_call_message_id: str | None = None
    native_call_recorded_timestamp_utc: datetime | None = None
    native_result_message_id: str | None = None
    native_result_recorded_timestamp_utc: datetime | None = None
    capture_timestamp_utc: datetime
    capture_elapsed_seconds: float = Field(ge=0)
    normalized_event_id: str
    normalized_timestamp_utc: datetime
    normalized_elapsed_seconds: float = Field(ge=0)
    native_timestamp_provenance: Literal["hermes_sqlite_message_recorded"] = (
        "hermes_sqlite_message_recorded"
    )
    capture_timestamp_provenance: Literal[
        "agent_bench_post_process_sqlite_export"
    ] = "agent_bench_post_process_sqlite_export"
    execution_start: TimingValue
    execution_end: TimingValue
    model_tool_call_observed: TimingValue


class TimingProvenanceAnalysis(_Record):
    analysis_id: str = Field(min_length=1)
    calculator_name: Literal["agent-bench-timing-provenance"] = (
        "agent-bench-timing-provenance"
    )
    calculator_version: Literal["1.0.0"] = "1.0.0"
    calculator_configuration_digest: Sha256
    run_id: Identifier
    harness_id: Literal["hermes"] = "hermes"
    source_artifact_manifest_sha256: Sha256
    source_run_manifest_sha256: Sha256
    source_raw_events_sha256: Sha256
    source_normalized_events_sha256: Sha256
    tools: tuple[HermesToolTiming, ...]
    time_to_first_harness_tool_execution: TimingValue
    time_to_first_harness_edit_execution: TimingValue
    time_to_first_observed_tool_event: TimingValue
    time_to_first_observed_edit_event: TimingValue
    time_to_first_model_tool_call_observed: TimingValue
    time_to_first_model_edit_call_observed: TimingValue
    comparability_rule: Literal[
        "compare_only_identical_timing_semantics_and_methods"
    ] = "compare_only_identical_timing_semantics_and_methods"
    record_digest: Sha256

    @model_validator(mode="after")
    def _digest(self) -> "TimingProvenanceAnalysis":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_digest"}))
        if self.record_digest != expected:
            raise ValueError("record_digest does not match timing provenance analysis")
        return self

    @classmethod
    def create(cls, **values: object) -> "TimingProvenanceAnalysis":
        draft = cls.model_construct(
            schema_version=TIMING_PROVENANCE_SCHEMA_VERSION,
            **values,
            record_digest="0" * 64,
        )
        content = draft.model_dump(mode="json", exclude={"record_digest"})
        return cls.model_validate({**content, "record_digest": canonical_sha256(content)})

    def canonical_json_bytes(self) -> bytes:
        return (
            json.dumps(
                self.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")


def derive_hermes_timing_provenance(artifact_path: Path) -> TimingProvenanceAnalysis:
    """Analyze Hermes SQLite-export evidence without changing its source run."""
    root = artifact_path.expanduser().resolve()
    try:
        artifact = verify_artifact(root)
        manifest = RunManifest.model_validate_json((root / RUN_MANIFEST_PATH).read_bytes())
        raw = load_raw_events(root / RAW_EVENTS_PATH)
        normalized = load_normalized_events(root / NORMALIZED_EVENTS_PATH)
    except Exception as exc:
        raise TimingProvenanceError(f"invalid sealed source artifact: {exc}") from exc
    if manifest.harness_id != "hermes" or artifact.run_id != manifest.run_id:
        raise TimingProvenanceError("source artifact is not a consistent Hermes run")

    normalized_by_raw = {
        item.raw_event_refs[0].raw_event_id: item
        for item in normalized
        if item.raw_event_refs and item.event_kind == "tool_call_start"
    }
    response_by_call: dict[str, object] = {}
    for event in normalized:
        if event.event_kind != "llm_response":
            continue
        for call in _tool_calls(event.payload.get("tool_calls")):
            call_id = call.get("id")
            if isinstance(call_id, str) and call_id and call_id not in response_by_call:
                response_by_call[call_id] = event

    calls: list[tuple[RawEvent, dict[str, object], dict[str, object]]] = []
    results: dict[str, tuple[RawEvent, dict[str, object]]] = {}
    for event in raw:
        if event.event_type != "hermes_session_message":
            continue
        native = event.payload.get("native_event")
        if not isinstance(native, dict):
            continue
        if native.get("role") == "assistant":
            for call in _tool_calls(native.get("tool_calls")):
                call_id, name = _call_identity(call)
                if call_id and name:
                    calls.append((event, native, call))
        elif native.get("role") == "tool":
            call_id = native.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                results.setdefault(call_id, (event, native))

    points: list[HermesToolTiming] = []
    for ordinal, (raw_start, native_start, call) in enumerate(calls, start=1):
        call_id, name = _call_identity(call)
        assert call_id is not None and name is not None
        normalized_start = normalized_by_raw.get(raw_start.raw_event_id)
        if normalized_start is None or normalized_start.elapsed_seconds is None:
            raise TimingProvenanceError(f"missing normalized tool-start evidence for {call_id}")
        result = results.get(call_id)
        response = response_by_call.get(call_id)
        points.append(HermesToolTiming(
            ordinal=ordinal,
            tool_name=name,
            category=_category(name),
            tool_call_id=call_id,
            native_call_message_id=_string(native_start.get("id")),
            native_call_recorded_timestamp_utc=_native_timestamp(native_start.get("timestamp")),
            native_result_message_id=_string(result[1].get("id")) if result else None,
            native_result_recorded_timestamp_utc=_native_timestamp(result[1].get("timestamp")) if result else None,
            capture_timestamp_utc=raw_start.timestamp_utc,
            capture_elapsed_seconds=raw_start.elapsed_ns / 1_000_000_000,
            normalized_event_id=normalized_start.event_id,
            normalized_timestamp_utc=normalized_start.timestamp_utc,
            normalized_elapsed_seconds=normalized_start.elapsed_seconds,
            execution_start=_unavailable("harness_tool_execution_start", "native_execution_timestamp_not_exposed"),
            execution_end=_unavailable("harness_tool_execution_end", "native_execution_timestamp_not_exposed"),
            model_tool_call_observed=(
                _available(response.elapsed_seconds, "model_tool_call_observed", "proxy_observation", response.event_id)
                if response is not None and response.elapsed_seconds is not None
                else _unavailable("model_tool_call_observed", "proxy_model_tool_call_not_observed")
            ),
        ))
    if not points:
        raise TimingProvenanceError("no Hermes native tool calls found")

    first = points[0]
    first_edit = next((point for point in points if point.category in {"edit", "write"}), None)
    return TimingProvenanceAnalysis.create(
        analysis_id=f"{manifest.run_id}-timing-provenance-v1",
        calculator_configuration_digest=TIMING_PROVENANCE_CONFIGURATION_DIGEST,
        run_id=manifest.run_id,
        source_artifact_manifest_sha256=_sha(root / "manifest.json"),
        source_run_manifest_sha256=_sha(root / RUN_MANIFEST_PATH),
        source_raw_events_sha256=_sha(root / RAW_EVENTS_PATH),
        source_normalized_events_sha256=_sha(root / NORMALIZED_EVENTS_PATH),
        tools=tuple(points),
        time_to_first_harness_tool_execution=_unavailable("harness_tool_execution_start", "native_execution_timestamp_not_exposed"),
        time_to_first_harness_edit_execution=_unavailable("harness_tool_execution_start", "native_execution_timestamp_not_exposed"),
        time_to_first_observed_tool_event=_available(first.normalized_elapsed_seconds, "tool_event_observed", "runner_monotonic", first.normalized_event_id),
        time_to_first_observed_edit_event=(
            _available(first_edit.normalized_elapsed_seconds, "tool_event_observed", "runner_monotonic", first_edit.normalized_event_id)
            if first_edit else _unavailable("tool_event_observed", "edit_event_not_observed")
        ),
        time_to_first_model_tool_call_observed=first.model_tool_call_observed,
        time_to_first_model_edit_call_observed=(
            first_edit.model_tool_call_observed if first_edit else _unavailable("model_tool_call_observed", "edit_model_tool_call_not_observed")
        ),
    )


def _tool_calls(value: object) -> tuple[dict[str, object], ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return ()
    return tuple(item for item in value if isinstance(item, dict)) if isinstance(value, list) else ()


def _call_identity(call: dict[str, object]) -> tuple[str | None, str | None]:
    call_id = call.get("id") or call.get("tool_call_id")
    name = call.get("name") or call.get("tool_name")
    function = call.get("function")
    if isinstance(function, dict):
        name = name or function.get("name")
    return _string(call_id), _string(name)


def _category(name: str) -> str:
    return {
        "read_file": "read", "read": "read", "search_files": "search",
        "search": "search", "patch": "edit", "edit_file": "edit",
        "write_file": "write", "terminal": "shell",
    }.get(name, "other")


def _native_timestamp(value: object) -> datetime | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def _string(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _available(
    value: float,
    semantics: Literal[
        "harness_tool_execution_start", "harness_tool_execution_end",
        "tool_event_observed", "model_tool_call_observed",
    ],
    method: Literal["runner_monotonic", "proxy_observation"],
    event_id: str,
) -> TimingValue:
    return TimingValue(value=value, availability="available", semantics=semantics, method=method, source_event_ids=(event_id,))


def _unavailable(
    semantics: Literal[
        "harness_tool_execution_start", "harness_tool_execution_end",
        "tool_event_observed", "model_tool_call_observed",
    ],
    reason: str,
) -> TimingValue:
    return TimingValue(value=None, availability="unavailable", semantics=semantics, method="not_available", unavailable_reason=reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
