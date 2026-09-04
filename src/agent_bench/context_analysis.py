"""Versioned, generic request-purpose and context-overhead analysis.

This module is intentionally independent of harness adapters.  It consumes the
sealed proxy exchange evidence emitted by M5 and produces a new immutable
analysis layer; it never changes raw events or an existing metrics-v1 record.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_bench.events import NormalizedEvent, load_normalized_events
from agent_bench.metrics import _model_exchange_events
from agent_bench.models import Identifier, Sha256, canonical_sha256
from agent_bench.preservation import verify_artifact
from agent_bench.runner import NORMALIZED_EVENTS_PATH, RAW_EVENTS_PATH, RUN_MANIFEST_PATH, RunManifest


CONTEXT_ANALYSIS_SCHEMA_VERSION: Literal["2.0.0"] = "2.0.0"
CONTEXT_ANALYSIS_CONFIGURATION = {
    "version": "2.0.0",
    "purpose_rules": "exact-task-and-title-evidence-v1",
    "decomposition": "exact-tokenizer-only-no-estimates-v1",
    "inference_endpoint_filter": "post-completions-or-responses-v1",
}
CONTEXT_ANALYSIS_CONFIGURATION_DIGEST = canonical_sha256(CONTEXT_ANALYSIS_CONFIGURATION)
Purpose = Literal[
    "task", "title", "planning", "summarization", "compaction",
    "metadata_discovery", "other_internal", "unknown",
]


class ContextAnalysisError(RuntimeError):
    """Raised when sealed evidence cannot support context analysis."""


class AnalysisModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["2.0.0"] = CONTEXT_ANALYSIS_SCHEMA_VERSION


class AnalysisScalar(AnalysisModel):
    value: int | float | None
    units: str = Field(min_length=1)
    availability: Literal["available", "unavailable", "not_applicable"]
    method: Literal["api_exact", "deterministically_calculated", "not_available"]
    source_event_ids: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    @model_validator(mode="after")
    def _valid(self) -> "AnalysisScalar":
        if self.availability == "available":
            if self.value is None or self.unavailable_reason is not None:
                raise ValueError("available value requires a value and no reason")
        elif self.value is not None or self.unavailable_reason is None:
            raise ValueError("unavailable value requires null and a reason")
        return self


class ContextComponent(AnalysisModel):
    name: Literal[
        "system_harness", "tool_schema", "skills_planning", "project_instruction",
        "user_task", "historical_assistant_text", "historical_reasoning",
        "historical_tool_call", "historical_tool_result", "other",
    ]
    tokens: AnalysisScalar


class RequestContextObservation(AnalysisModel):
    model_request_index: int = Field(ge=1)
    captured_http_request_index: int | None = Field(default=None, ge=1)
    request_event_id: str
    response_event_id: str | None = None
    request_body_sha256: Sha256 | None = None
    elapsed_seconds: float | None = Field(default=None, ge=0)
    purpose: Purpose
    purpose_evidence: str
    message_roles: tuple[str, ...]
    messages_sha256: Sha256 | None = None
    tool_schema_sha256: Sha256 | None = None
    input_context_tokens: AnalysisScalar
    output_tokens: AnalysisScalar
    configured_max_context_tokens: AnalysisScalar
    context_utilization_percent: AnalysisScalar
    delta_vs_previous_inference_tokens: AnalysisScalar
    delta_vs_first_task_tokens: AnalysisScalar
    components: tuple[ContextComponent, ...]


class DiagnosticHttpRequest(AnalysisModel):
    captured_http_request_index: int | None = Field(default=None, ge=1)
    request_event_id: str
    method: str | None = None
    endpoint: str | None = None
    purpose: Literal["metadata_discovery"] = "metadata_discovery"


class PromptConfigurationEvidence(AnalysisModel):
    relative_path: str
    sha256: Sha256
    size_bytes: int = Field(ge=0)


class ContextAnalysis(AnalysisModel):
    """Immutable generic analysis sufficient for later task-relative charts."""

    analysis_id: str = Field(min_length=1)
    calculator_name: Literal["agent-bench-context-analysis"] = "agent-bench-context-analysis"
    calculator_version: Literal["2.0.0"] = "2.0.0"
    calculator_configuration_digest: Sha256
    run_id: Identifier
    source_artifact_manifest_sha256: Sha256
    source_run_manifest_sha256: Sha256
    source_raw_events_sha256: Sha256
    source_normalized_events_sha256: Sha256
    prompt_sha256: Sha256
    stable_prompt_configuration: tuple[PromptConfigurationEvidence, ...]
    rendered_prompt_observation: Literal["unavailable_at_proxy_boundary"] = "unavailable_at_proxy_boundary"
    requests: tuple[RequestContextObservation, ...]
    diagnostic_http_requests: tuple[DiagnosticHttpRequest, ...]
    first_task_request_index: int | None = Field(default=None, ge=1)
    initial_task_context_tokens: AnalysisScalar
    initial_task_context_utilization_percent: AnalysisScalar
    auxiliary_requests_before_first_task: AnalysisScalar
    auxiliary_input_tokens_before_first_task: AnalysisScalar
    auxiliary_output_tokens_before_first_task: AnalysisScalar
    non_user_initial_context_tokens: AnalysisScalar
    record_digest: Sha256

    @model_validator(mode="after")
    def _digest(self) -> "ContextAnalysis":
        expected = canonical_sha256(self.model_dump(mode="json", exclude={"record_digest"}))
        if expected != self.record_digest:
            raise ValueError("record_digest does not match context analysis")
        return self

    @classmethod
    def create(cls, **values: object) -> "ContextAnalysis":
        draft = cls.model_construct(schema_version=CONTEXT_ANALYSIS_SCHEMA_VERSION, **values, record_digest="0" * 64)
        content = draft.model_dump(mode="json", exclude={"record_digest"})
        return cls.model_validate({**content, "record_digest": canonical_sha256(content)})

    def canonical_json_bytes(self) -> bytes:
        return (json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def derive_context_analysis(artifact_path: Path) -> ContextAnalysis:
    """Derive generic analysis from a sealed run without modifying it."""
    root = artifact_path.expanduser().resolve()
    try:
        artifact = verify_artifact(root)
        manifest = RunManifest.model_validate_json((root / RUN_MANIFEST_PATH).read_bytes())
        events = load_normalized_events(root / NORMALIZED_EVENTS_PATH)
        prompt = (root / "run" / "prompt.txt").read_bytes()
    except Exception as exc:
        raise ContextAnalysisError(f"invalid sealed source artifact: {exc}") from exc
    if artifact.run_id != manifest.run_id:
        raise ContextAnalysisError("artifact and run manifests have different run IDs")
    prompt_sha = hashlib.sha256(prompt).hexdigest()
    request_events, response_events = _model_exchange_events(events)
    response_by_id = {
        str(item.payload["request_id"]): item for item in response_events
        if isinstance(item.payload.get("request_id"), str)
    }
    points: list[RequestContextObservation] = []
    first_task_index: int | None = None
    first_task_input: AnalysisScalar | None = None
    previous_input: AnalysisScalar | None = None
    for index, request in enumerate(request_events, start=1):
        response = response_by_id.get(str(request.payload.get("request_id", "")))
        decoded = _decode_json_body(request)
        roles, messages_sha, tools_sha = _structure_hashes(decoded)
        purpose, evidence = _classify(decoded, prompt)
        input_tokens = _api_metric(response, "input_tokens", "tokens")
        output_tokens = _api_metric(response, "output_tokens", "tokens")
        maximum = _request_integer_metric(request, "configured_max_context_tokens", "tokens")
        utilization = _ratio(input_tokens, maximum, request.event_id)
        if first_task_index is None and purpose == "task":
            first_task_index, first_task_input = index, input_tokens
        points.append(RequestContextObservation(
            model_request_index=index,
            captured_http_request_index=_positive_int(request.payload.get("request_index")),
            request_event_id=request.event_id,
            response_event_id=response.event_id if response else None,
            request_body_sha256=_sha(request.payload.get("body_sha256")),
            elapsed_seconds=request.elapsed_seconds,
            purpose=purpose,
            purpose_evidence=evidence,
            message_roles=roles,
            messages_sha256=messages_sha,
            tool_schema_sha256=tools_sha,
            input_context_tokens=input_tokens,
            output_tokens=output_tokens,
            configured_max_context_tokens=maximum,
            context_utilization_percent=utilization,
            delta_vs_previous_inference_tokens=_difference(input_tokens, previous_input, request.event_id, "no_previous_inference"),
            delta_vs_first_task_tokens=_difference(input_tokens, first_task_input, request.event_id, "first_task_not_observed"),
            components=tuple(ContextComponent(name=name, tokens=_unavailable("tokens", "exact_component_tokenization_not_available")) for name in (
                "system_harness", "tool_schema", "skills_planning", "project_instruction", "user_task",
                "historical_assistant_text", "historical_reasoning", "historical_tool_call", "historical_tool_result", "other",
            )),
        ))
        previous_input = input_tokens
    # Delta against the first task is meaningful only once the task has been seen.
    if first_task_index is not None:
        first = points[first_task_index - 1].input_context_tokens
        points = [point.model_copy(update={"delta_vs_first_task_tokens": _difference(point.input_context_tokens, first, point.request_event_id, "not_applicable")}) if point.model_request_index >= first_task_index else point for point in points]
    diagnostics = tuple(DiagnosticHttpRequest(
        captured_http_request_index=_positive_int(event.payload.get("request_index")),
        request_event_id=event.event_id, method=_str(event.payload.get("method")), endpoint=_str(event.payload.get("endpoint")),
    ) for event in events if event.event_kind == "llm_request" and event not in request_events)
    auxiliary = [p for p in points if first_task_index is not None and p.model_request_index < first_task_index]
    initial = points[first_task_index - 1] if first_task_index else None
    evidence = _stable_configuration_evidence(root)
    return ContextAnalysis.create(
        analysis_id=f"{manifest.run_id}-context-analysis-v2",
        calculator_configuration_digest=CONTEXT_ANALYSIS_CONFIGURATION_DIGEST,
        run_id=manifest.run_id,
        source_artifact_manifest_sha256=_file_sha(root / "manifest.json"),
        source_run_manifest_sha256=_file_sha(root / RUN_MANIFEST_PATH),
        source_raw_events_sha256=_file_sha(root / RAW_EVENTS_PATH),
        source_normalized_events_sha256=_file_sha(root / NORMALIZED_EVENTS_PATH),
        prompt_sha256=prompt_sha,
        stable_prompt_configuration=evidence,
        requests=tuple(points), diagnostic_http_requests=diagnostics,
        first_task_request_index=first_task_index,
        initial_task_context_tokens=(initial.input_context_tokens if initial else _unavailable("tokens", "task_request_not_observed")),
        initial_task_context_utilization_percent=(initial.context_utilization_percent if initial else _unavailable("percent", "task_request_not_observed")),
        auxiliary_requests_before_first_task=_available(len(auxiliary), "requests", tuple(p.request_event_id for p in auxiliary)),
        auxiliary_input_tokens_before_first_task=_sum(auxiliary, "input_context_tokens"),
        auxiliary_output_tokens_before_first_task=_sum(auxiliary, "output_tokens"),
        non_user_initial_context_tokens=_unavailable("tokens", "exact_user_task_component_tokenization_not_available"),
    )


def _decode_json_body(event: NormalizedEvent) -> dict[str, Any] | None:
    value = event.payload.get("body_base64")
    if not isinstance(value, str): return None
    try:
        parsed = json.loads(base64.b64decode(value, validate=True))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError): return None
    return parsed if isinstance(parsed, dict) else None


def _structure_hashes(body: dict[str, Any] | None) -> tuple[tuple[str, ...], str | None, str | None]:
    if body is None: return (), None, None
    messages = body.get("messages")
    roles = tuple(str(item.get("role")) for item in messages if isinstance(item, dict) and isinstance(item.get("role"), str)) if isinstance(messages, list) else ()
    return roles, canonical_sha256(messages) if isinstance(messages, list) else None, canonical_sha256(body.get("tools")) if "tools" in body else None


def _classify(body: dict[str, Any] | None, prompt: bytes) -> tuple[Purpose, str]:
    if body is None: return "unknown", "request body unavailable or non-JSON"
    serialized = json.dumps(body.get("messages", []), ensure_ascii=False, sort_keys=True)
    if "You are a title generator" in serialized or "Generate a brief title" in serialized or "You name chat sessions" in serialized:
        return "title", "exact title-generator marker occurs in request messages"
    prompt_text = prompt.decode("utf-8")
    if _contains_exact_text(body.get("messages"), prompt_text):
        return "task", "exact preserved run prompt occurs in request messages"
    # Some upstream one-shot CLIs deliberately trim only a terminal line ending
    # while constructing their user message.  This is a deterministic byte-level
    # transport normalization, not semantic matching or a heuristic.
    if prompt_text.endswith(("\n", "\r")) and _contains_exact_text(body.get("messages"), prompt_text.rstrip("\r\n")):
        return "task", "preserved prompt with terminal line ending trimmed by harness transport occurs in request messages"
    return "other_internal", "no deterministic task/title purpose marker"


def _contains_exact_text(value: object, expected: str) -> bool:
    if isinstance(value, str): return value == expected
    if isinstance(value, list): return any(_contains_exact_text(item, expected) for item in value)
    if isinstance(value, dict): return any(_contains_exact_text(item, expected) for item in value.values())
    return False


def _api_metric(response: NormalizedEvent | None, key: str, units: str) -> AnalysisScalar:
    value = response.payload.get(key) if response else None
    if isinstance(value, int) and value >= 0:
        return _available(value, units, (response.event_id,))
    return _unavailable(units, "api_usage_not_exposed")


def _request_integer_metric(request: NormalizedEvent, key: str, units: str) -> AnalysisScalar:
    value = request.payload.get(key)
    if isinstance(value, int) and value > 0: return _available(value, units, (request.event_id,))
    return _unavailable(units, "request_configuration_not_exposed")


def _ratio(left: AnalysisScalar, right: AnalysisScalar, event_id: str) -> AnalysisScalar:
    if left.value is not None and right.value not in (None, 0): return _available(float(left.value) * 100 / float(right.value), "percent", (event_id,))
    return _unavailable("percent", "input_or_maximum_context_unavailable")


def _difference(left: AnalysisScalar, right: AnalysisScalar | None, event_id: str, reason: str) -> AnalysisScalar:
    if right is not None and left.value is not None and right.value is not None: return _available(int(left.value) - int(right.value), "tokens", (event_id,))
    return _unavailable("tokens", reason)


def _sum(items: list[RequestContextObservation], field: str) -> AnalysisScalar:
    values = [getattr(item, field) for item in items]
    if not values: return _available(0, "tokens", ())
    if all(value.value is not None for value in values): return _available(sum(int(value.value) for value in values), "tokens", tuple(item.request_event_id for item in items))
    return _unavailable("tokens", "api_usage_not_exposed")


def _available(value: int | float, units: str, ids: tuple[str, ...]) -> AnalysisScalar:
    return AnalysisScalar(value=value, units=units, availability="available", method="api_exact" if units == "tokens" else "deterministically_calculated", source_event_ids=ids)


def _unavailable(units: str, reason: str) -> AnalysisScalar:
    return AnalysisScalar(value=None, units=units, availability="unavailable", method="not_available", unavailable_reason=reason)


def _stable_configuration_evidence(root: Path) -> tuple[PromptConfigurationEvidence, ...]:
    state = root / "run" / "harness-state"
    if not state.is_dir(): return ()
    # Only stable configuration names, never mutable session/cache/log data.
    candidates = [path for path in state.rglob("*") if path.is_file() and path.name in {"config.yaml", "config.json", "models.json", "opencode.json", "invocation.json"}]
    return tuple(sorted((PromptConfigurationEvidence(relative_path=str(path.relative_to(root)), sha256=_file_sha(path), size_bytes=path.stat().st_size) for path in candidates), key=lambda item: item.relative_path))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _sha(value: object) -> str | None:
    return value if isinstance(value, str) and len(value) == 64 else None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None
