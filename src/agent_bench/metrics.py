"""Deterministic M4 metric calculation over sealed M3 run artifacts."""

from __future__ import annotations

import hashlib
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agent_bench.events import (
    NormalizedEvent,
    RawEvent,
    load_normalized_events,
    load_raw_events,
)
from agent_bench.metric_models import (
    BehaviorMetrics,
    CompactionPoint,
    ContextMetrics,
    ContextRequestPoint,
    DerivedMetrics,
    GitResultMetrics,
    MetricProvenance,
    MetricsInputIdentity,
    RequestTokenUsage,
    RunMetrics,
    ScalarMetric,
    TerminationResult,
    TimingMetrics,
    TokenMetrics,
    ToolCategoryCounts,
)
from agent_bench.models import canonical_sha256
from agent_bench.preservation import (
    GIT_UNTRACKED_NUMSTAT_PATH,
    GIT_TRACKED_NUMSTAT_PATH,
    MANIFEST_PATH,
    GitNumstatRecord,
    verify_artifact,
)
from agent_bench.runner import NORMALIZED_EVENTS_PATH, RAW_EVENTS_PATH, RUN_MANIFEST_PATH, RunManifest

METRICS_CONFIGURATION = {
    "version": "1.0.1",
    "duration_aggregation": "sum_completed_correlated_intervals",
    "duplicate_arguments": "canonical-json-exact-v1",
    "path_normalization": "project-relative-posix-lexical-v1",
    "git_status": "porcelain-v1-no-rename-reclassification",
    "path_classifier": "agent-bench-path-classifier-v1",
    "tool_timing": "explicit-execution-boundary-only-v1",
    "termination_precedence": [
        "precondition_failed",
        "preservation_failed",
        "timeout",
        "process_killed",
        "context_overflow",
        "output_truncation",
        "model_backend_error",
        "harness_crash",
        "invalid_harness_output",
        "no_changes",
        "success",
        "unknown_other",
    ],
}
METRICS_CONFIGURATION_DIGEST = canonical_sha256(METRICS_CONFIGURATION)

_CATEGORIES = ("read", "search", "edit", "write", "test", "shell", "other")
_TOKEN_METHODS = {"backend_exact", "api_exact", "tokenizer_reconstructed"}
_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html",
    ".java", ".js", ".jsx", ".kt", ".php", ".py", ".rb", ".rs", ".scala",
    ".sh", ".swift", ".ts", ".tsx", ".vue",
}
_CONFIG_NAMES = {
    ".editorconfig", ".env", ".gitignore", "cargo.toml", "dockerfile",
    "go.mod", "package.json", "pyproject.toml", "requirements.txt", "tox.ini",
}
_CONFIG_SUFFIXES = {".ini", ".json", ".toml", ".yaml", ".yml"}


class MetricsCalculationError(RuntimeError):
    """Raised when a sealed input artifact cannot be safely measured."""


@dataclass(frozen=True)
class _ToolCall:
    call_id: str
    start: NormalizedEvent
    end: NormalizedEvent | None
    category: str
    outcome: str


@dataclass(frozen=True)
class _GitSummary:
    paths: tuple[str, ...]
    created: int
    deleted: int
    renamed: int
    lines_added: int | None
    lines_deleted: int | None
    binary_files: int | None
    line_reason: str | None


def calculate_run_metrics(artifact_path: Path) -> RunMetrics:
    """Verify and deterministically calculate metrics for one sealed run."""
    root = artifact_path.expanduser().resolve()
    try:
        artifact_manifest = verify_artifact(root)
        run_manifest = RunManifest.model_validate_json(
            (root / RUN_MANIFEST_PATH).read_bytes()
        )
        events = load_normalized_events(root / NORMALIZED_EVENTS_PATH)
        raw_events = load_raw_events(root / RAW_EVENTS_PATH)
    except Exception as exc:
        raise MetricsCalculationError(f"invalid metrics input artifact: {exc}") from exc

    if run_manifest.run_id != artifact_manifest.run_id:
        raise MetricsCalculationError("run and artifact manifests have different run IDs")
    if any(event.run_id != run_manifest.run_id for event in events):
        raise MetricsCalculationError("normalized stream contains a different run ID")
    if any(event.run_id != run_manifest.run_id for event in raw_events):
        raise MetricsCalculationError("raw stream contains a different run ID")
    _validate_event_provenance(raw_events, events)

    diagnostics: list[str] = []
    capabilities = run_manifest.capture_capabilities
    complete_llm_capture = (
        run_manifest.adapter_id == "fake-harness"
        or (
            capabilities is not None
            and capabilities.supports_complete_llm_exchange_capture()
        )
    )
    complete_tool_capture = (
        run_manifest.adapter_id == "fake-harness"
        or (
            capabilities is not None
            and capabilities.tool_calls == "harness_exact"
            and capabilities.tool_results == "harness_exact"
        )
    )
    complete_compaction_capture = (
        run_manifest.adapter_id == "fake-harness"
        or (capabilities is not None and capabilities.compaction_events != "unavailable")
    )
    tools = _correlate_tools(events, diagnostics)
    tool_integrity = _tool_identity_valid(events)
    timing = _calculate_timing(
        run_manifest,
        events,
        tools,
        complete_llm_capture,
        complete_tool_capture,
        tool_integrity,
    )
    tokens, context = _calculate_tokens_and_context(
        events,
        timing.time_to_first_edit_seconds,
        complete_llm_capture,
        diagnostics,
        complete_compaction_capture=complete_compaction_capture,
    )
    behavior = _calculate_behavior(
        events,
        tools,
        complete_llm_capture,
        tool_integrity,
        diagnostics,
        complete_tool_capture=complete_tool_capture,
    )
    derived = _calculate_derived(timing, tokens, behavior)
    git_result, git_summary = _calculate_git(root, diagnostics)
    termination = _classify_termination(
        run_manifest, events, git_summary, git_result, raw_events
    )

    input_identity = MetricsInputIdentity(
        artifact_manifest_sha256=_sha256_file(root / MANIFEST_PATH),
        run_manifest_sha256=_sha256_file(root / RUN_MANIFEST_PATH),
        raw_events_sha256=_sha256_file(root / RAW_EVENTS_PATH),
        normalized_events_sha256=_sha256_file(root / NORMALIZED_EVENTS_PATH),
        source_snapshot_sha256=artifact_manifest.source_snapshot_sha256,
        git_diff_sha256=artifact_manifest.git_diff_sha256,
        git_tracked_numstat_sha256=_sha256_file(root / GIT_TRACKED_NUMSTAT_PATH),
        git_untracked_numstat_sha256=_sha256_file(root / GIT_UNTRACKED_NUMSTAT_PATH),
    )
    return RunMetrics.create(
        metrics_id=f"{run_manifest.run_id}-metrics-v1",
        calculator_configuration_digest=METRICS_CONFIGURATION_DIGEST,
        run_id=run_manifest.run_id,
        input_identity=input_identity,
        timing=timing,
        tokens=tokens,
        context=context,
        behavior=behavior,
        derived=derived,
        git_result=git_result,
        termination=termination,
        validation_status=("valid_with_diagnostics" if diagnostics else "valid"),
        diagnostics=tuple(sorted(set(diagnostics))),
    )


def _calculate_timing(
    manifest: RunManifest,
    events: tuple[NormalizedEvent, ...],
    tools: tuple[_ToolCall, ...],
    complete_llm_capture: bool,
    complete_tool_capture: bool,
    tool_integrity: bool,
) -> TimingMetrics:
    wall = _available(
        manifest.task_elapsed_ns / 1_000_000_000,
        "seconds",
        "manifest_exact",
        artifacts=(RUN_MANIFEST_PATH,),
    )
    requests, responses = _model_exchange_events(events)
    llm_time = _interval_sum(
        requests, responses, "request_id", "seconds", complete_llm_capture
    )
    exact_tool_timing = all(
        _has_exact_tool_execution_timing(call) for call in tools
    )
    if tool_integrity and complete_tool_capture and exact_tool_timing:
        tool_time = _tool_interval_sum(tools, lambda call: True, complete_tool_capture)
        shell_time = _tool_interval_sum(
            tools,
            lambda call: call.category == "shell"
            or (call.category == "test" and call.start.payload.get("uses_shell") is True),
            complete_tool_capture,
        )
    else:
        reason = (
            "invalid_source"
            if not tool_integrity
            else (
                "capture_incomplete"
                if not complete_tool_capture
                else "native_execution_timestamp_not_exposed"
            )
        )
        tool_time = _unavailable("seconds", reason)
        shell_time = _unavailable("seconds", reason)
    exact_starts = [call.start for call in tools if _is_exact_execution_start(call.start)]
    edit_starts = [
        call.start for call in tools
        if (
            call.category in {"edit", "write"}
            and _tool_targets_worktree(call.start)
            and _is_exact_execution_start(call.start)
        )
    ]
    test_starts = [
        call.start for call in tools
        if call.category == "test" and _is_exact_execution_start(call.start)
    ]
    return TimingMetrics(
        wall_time_seconds=wall,
        llm_time_seconds=llm_time,
        tool_execution_time_seconds=tool_time,
        shell_execution_time_seconds=shell_time,
        time_to_first_llm_request_seconds=(
            _first_elapsed(requests, "requests")
            if complete_llm_capture
            else _unavailable("seconds", "capture_incomplete")
        ),
        time_to_first_tool_call_seconds=(
            _first_elapsed(exact_starts, "calls")
            if tool_integrity and complete_tool_capture and exact_starts
            else _unavailable("seconds", "invalid_source" if not tool_integrity else ("capture_incomplete" if not complete_tool_capture else "native_execution_timestamp_not_exposed"))
        ),
        time_to_first_edit_seconds=(
            _first_elapsed(edit_starts, "calls")
            if tool_integrity and complete_tool_capture and edit_starts
            else _unavailable("seconds", "invalid_source" if not tool_integrity else ("capture_incomplete" if not complete_tool_capture else "native_execution_timestamp_not_exposed"))
        ),
        time_to_first_test_command_seconds=(
            _first_elapsed(test_starts, "calls")
            if tool_integrity and complete_tool_capture and test_starts
            else _unavailable("seconds", "invalid_source" if not tool_integrity else ("capture_incomplete" if not complete_tool_capture else "native_execution_timestamp_not_exposed"))
        ),
    )


def _model_exchange_events(
    events: tuple[NormalizedEvent, ...],
) -> tuple[list[NormalizedEvent], list[NormalizedEvent]]:
    """Return completion exchanges, excluding proxy metadata/discovery traffic.

    Raw proxy capture deliberately retains every HTTP request.  Only POST
    completion endpoints are model requests for timing, token, context, and
    request-count metrics.  Events without HTTP endpoint metadata (for example
    deterministic FakeHarness fixtures) remain accepted for backward-compatible
    common-event handling.
    """
    requests: list[NormalizedEvent] = []
    request_ids: set[str] = set()
    for event in events:
        if event.event_kind != "llm_request":
            continue
        endpoint = event.payload.get("endpoint")
        method = event.payload.get("method")
        if isinstance(endpoint, str) and endpoint:
            if method != "POST" or not endpoint.rstrip("/").endswith(
                ("/chat/completions", "/completions", "/responses")
            ):
                continue
        requests.append(event)
        request_id = event.payload.get("request_id")
        if isinstance(request_id, str):
            request_ids.add(request_id)
    responses = [
        event
        for event in events
        if event.event_kind == "llm_response"
        and (
            not isinstance(event.payload.get("request_id"), str)
            or event.payload["request_id"] in request_ids
        )
    ]
    return requests, responses


def _calculate_tokens_and_context(
    events: tuple[NormalizedEvent, ...],
    first_edit: ScalarMetric,
    complete_llm_capture: bool,
    diagnostics: list[str],
    *,
    complete_compaction_capture: bool | None = None,
) -> tuple[TokenMetrics, ContextMetrics]:
    if complete_compaction_capture is None:
        complete_compaction_capture = complete_llm_capture
    requests, responses = _model_exchange_events(events)
    response_by_request: dict[str, NormalizedEvent] = {}
    response_identity_valid = True
    for response in responses:
        request_id = response.payload.get("request_id")
        if isinstance(request_id, str) and request_id not in response_by_request:
            response_by_request[request_id] = response
        elif isinstance(request_id, str):
            diagnostics.append(f"duplicate response correlation ID: {request_id}")
            response_identity_valid = False

    indexed: list[tuple[int, NormalizedEvent]] = []
    seen_indices: set[int] = set()
    seen_request_ids: set[str] = set()
    request_identity_valid = True
    for position, request in enumerate(requests, start=1):
        index = _strict_int(request.payload.get("request_index"))
        if index is None:
            index = position
            request_identity_valid = False
            diagnostics.append(
                f"request index unavailable; stream position used for {request.event_id}"
            )
        if index in seen_indices:
            diagnostics.append(f"duplicate request index: {index}")
            request_identity_valid = False
        seen_indices.add(index)
        request_id = request.payload.get("request_id")
        if not isinstance(request_id, str) or request_id in seen_request_ids:
            diagnostics.append(f"invalid or duplicate request correlation ID: {request.event_id}")
            request_identity_valid = False
        else:
            seen_request_ids.add(request_id)
        indexed.append((index, request))
    indexed.sort(key=lambda item: item[0])

    request_usages: list[RequestTokenUsage] = []
    context_points: list[ContextRequestPoint] = []
    input_values: list[int] = []
    input_source_events: list[NormalizedEvent] = []
    output_values: list[int] = []
    reasoning_values: list[int] = []
    visible_values: list[int] = []
    input_complete = bool(requests) and request_identity_valid and complete_llm_capture
    output_complete = (
        bool(requests)
        and request_identity_valid
        and response_identity_valid
        and complete_llm_capture
        and len(response_by_request) >= len(requests)
    )
    reasoning_complete = output_complete
    visible_complete = output_complete
    previous_context: ScalarMetric | None = None

    for index, request in indexed:
        request_id = request.payload.get("request_id")
        response = response_by_request.get(request_id) if isinstance(request_id, str) else None
        input_metric = _token_metric(request, "context_tokens")
        if input_metric.availability != "available" and response is not None:
            input_metric = _token_metric(response, "input_tokens")
        output_metric = (
            _token_metric(response, "output_tokens")
            if response is not None
            else _unavailable("tokens", "capture_incomplete")
        )
        reasoning_metric = (
            _token_metric(response, "reasoning_tokens")
            if response is not None
            else _unavailable("tokens", "capture_incomplete")
        )
        visible_metric = (
            _token_metric(response, "visible_answer_tokens")
            if response is not None
            else _unavailable("tokens", "capture_incomplete")
        )
        if input_metric.availability == "available":
            input_values.append(int(input_metric.value))
            input_source_events.append(
                response
                if (
                    response is not None
                    and response.event_id in input_metric.provenance.source_event_ids
                )
                else request
            )
        else:
            input_complete = False
        if output_metric.availability == "available":
            output_values.append(int(output_metric.value))
        else:
            output_complete = False
        if reasoning_metric.availability == "available":
            reasoning_values.append(int(reasoning_metric.value))
        else:
            reasoning_complete = False
        if visible_metric.availability == "available":
            visible_values.append(int(visible_metric.value))
        else:
            visible_complete = False
        total_for_request = _sum_metrics(input_metric, output_metric, "tokens")
        request_usages.append(
            RequestTokenUsage(
                request_index=index,
                request_event_id=request.event_id,
                response_event_id=response.event_id if response else None,
                input_tokens=input_metric,
                output_tokens=output_metric,
                total_tokens=total_for_request,
            )
        )

        max_context = _integer_metric(
            request,
            "configured_max_context_tokens",
            "tokens",
            require_positive=True,
        )
        utilization = _ratio_percent(input_metric, max_context)
        if previous_context is None:
            growth = _not_applicable("tokens")
        else:
            growth = _subtract_metrics(input_metric, previous_context, "tokens")
        context_points.append(
            ContextRequestPoint(
                request_index=index,
                request_event_id=request.event_id,
                elapsed_seconds=request.elapsed_seconds,
                context_used_tokens=input_metric,
                context_max_tokens=max_context,
                context_utilization_percent=utilization,
                context_growth_tokens=growth,
            )
        )
        previous_context = input_metric

    input_total = _aggregate_tokens(
        input_values, input_complete, input_source_events
    )
    output_total = _aggregate_tokens(output_values, output_complete, responses)
    reasoning_total = _aggregate_tokens(reasoning_values, reasoning_complete, responses)
    visible_total = _aggregate_tokens(visible_values, visible_complete, responses)
    total_tokens = _sum_metrics(input_total, output_total, "tokens")
    mean = _divide_metrics(total_tokens, len(requests), "tokens/request")
    before, reasoning_before = _tokens_before_edit(
        indexed, response_by_request, first_edit
    )

    complete_context = complete_llm_capture and request_identity_valid and bool(context_points) and all(
        point.context_used_tokens.availability == "available"
        and point.context_utilization_percent.availability == "available"
        for point in context_points
    )
    if complete_context:
        peak_tokens = _available(
            max(int(point.context_used_tokens.value) for point in context_points),
            "tokens",
            "deterministically_calculated",
            events=tuple(point.request_event_id for point in context_points),
        )
        peak_util = _available(
            max(float(point.context_utilization_percent.value) for point in context_points),
            "percent",
            "deterministically_calculated",
            events=tuple(point.request_event_id for point in context_points),
        )
    else:
        peak_tokens = _unavailable("tokens", "source_not_exposed")
        peak_util = _unavailable("percent", "source_not_exposed")
    if len(context_points) < 2:
        net_growth = _not_applicable("tokens")
    else:
        net_growth = _subtract_metrics(
            context_points[-1].context_used_tokens,
            context_points[0].context_used_tokens,
            "tokens",
        )

    compaction_events = [event for event in events if event.event_kind == "compaction_start"]
    compaction_ids = [event.payload.get("compaction_id") for event in compaction_events]
    compaction_identity_valid = all(isinstance(item, str) for item in compaction_ids) and len(set(compaction_ids)) == len(compaction_ids)
    if not compaction_identity_valid:
        diagnostics.append("compaction correlation IDs are missing or duplicated")
    compactions = tuple(
        _compaction_point(position, event)
        for position, event in enumerate(compaction_events, start=1)
    )
    compaction_count = (
        _available(
            len(compactions),
            "compactions",
            "normalized_event_exact",
            events=tuple(event.event_id for event in compaction_events),
        )
        if complete_compaction_capture and compaction_identity_valid
        else _unavailable(
            "compactions",
            "invalid_source" if not compaction_identity_valid else "source_not_exposed",
        )
    )
    if compactions:
        first_compaction = compactions[0]
        first_context = first_compaction.tokens_before_compaction
        first_util = first_compaction.before_utilization_percent
    else:
        first_context = _unavailable("tokens", "event_not_observed")
        first_util = _unavailable("percent", "event_not_observed")

    return (
        TokenMetrics(
            input_tokens_total=input_total,
            output_tokens_total=output_total,
            reasoning_tokens_total=reasoning_total,
            visible_answer_tokens_total=visible_total,
            total_tokens=total_tokens,
            tokens_per_llm_request=tuple(request_usages),
            mean_tokens_per_llm_request=mean,
            tokens_before_first_edit=before,
            reasoning_tokens_before_first_edit=reasoning_before,
        ),
        ContextMetrics(
            context_used_per_request=tuple(context_points),
            peak_context_tokens=peak_tokens,
            peak_context_utilization_percent=peak_util,
            net_context_growth_tokens=net_growth,
            number_of_compactions=compaction_count,
            context_at_first_compaction_tokens=first_context,
            context_utilization_at_first_compaction_percent=first_util,
            compactions=compactions,
        ),
    )


def _calculate_behavior(
    events: tuple[NormalizedEvent, ...],
    tools: tuple[_ToolCall, ...],
    complete_llm_capture: bool,
    tool_integrity: bool,
    diagnostics: list[str],
    *,
    complete_tool_capture: bool | None = None,
) -> BehaviorMetrics:
    if complete_tool_capture is None:
        complete_tool_capture = complete_llm_capture
    requests, responses = _model_exchange_events(events)
    request_ids = [event.payload.get("request_id") for event in requests]
    request_indices = [event.payload.get("request_index") for event in requests]
    requests_valid = (
        all(isinstance(item, str) for item in request_ids)
        and len(set(request_ids)) == len(request_ids)
        and all(_strict_int(item) is not None for item in request_indices)
        and len(set(request_indices)) == len(request_indices)
    )
    llm_requests = (
        _available(len(requests), "requests", "normalized_event_exact", events=tuple(e.event_id for e in requests))
        if complete_llm_capture and requests_valid
        else _unavailable("requests", "capture_incomplete" if not complete_llm_capture else "invalid_source")
    )
    response_ids = [event.payload.get("response_id") for event in responses]
    responses_valid = (
        all(isinstance(item, str) for item in response_ids)
        and len(set(response_ids)) == len(response_ids)
    )
    categories = {name: 0 for name in _CATEGORIES}
    for call in tools:
        categories[call.category] += 1
    tool_event_ids = tuple(call.start.event_id for call in tools)
    count_method = "normalized_event_exact"
    tool_metrics_valid = tool_integrity and complete_tool_capture
    total = (
        _available(len(tools), "calls", count_method, events=tool_event_ids)
        if tool_metrics_valid
        else _unavailable(
            "calls",
            "invalid_source" if not tool_integrity else "capture_incomplete",
            events=tool_event_ids,
        )
    )
    successful = sum(call.outcome == "success" for call in tools)
    failed = sum(call.outcome in {"failure", "timeout"} for call in tools)
    unknown = len(tools) - successful - failed
    first_edits = [call for call in tools if call.category in {"edit", "write"} and _tool_targets_worktree(call.start)]
    before = _calls_before_first_edit(tools, first_edits)
    after = _calls_after_last_edit(tools, first_edits)
    duplicates = _duplicate_calls(tools)
    repeated_shell = _repeated_shell_calls(tools)
    repeated_reads = _repeated_reads(events, tools)
    reasoning_only = _reasoning_only_turns(events)
    shell_calls = sum(
        call.category == "shell"
        or (call.category == "test" and call.start.payload.get("uses_shell") is True)
        for call in tools
    )
    if not complete_llm_capture:
        diagnostics.append("LLM capture completeness is not declared for this adapter")
    if not complete_tool_capture:
        diagnostics.append("tool capture completeness is not declared for this adapter")
    count = lambda value: (
        _available(value, "calls", count_method, events=tool_event_ids)
        if tool_metrics_valid
        else _unavailable(
            "calls",
            "invalid_source" if not tool_integrity else "capture_incomplete",
            events=tool_event_ids,
        )
    )
    return BehaviorMetrics(
        llm_request_count=llm_requests,
        llm_response_count=(
            _available(
                len(responses),
                "responses",
                "normalized_event_exact",
                events=tuple(event.event_id for event in responses),
            )
            if complete_llm_capture and responses_valid
            else _unavailable(
                "responses",
                "capture_incomplete" if not complete_llm_capture else "invalid_source",
            )
        ),
        tool_calls_total=total,
        tool_calls_by_category=ToolCategoryCounts(**categories) if tool_metrics_valid else None,
        tool_calls_by_category_availability="available" if tool_metrics_valid else "unavailable",
        tool_calls_by_category_unavailable_reason=(
            None if tool_metrics_valid
            else "invalid_source" if not tool_integrity else "capture_incomplete"
        ),
        tool_calls_by_category_provenance=MetricProvenance(
            method="normalized_event_exact" if tool_metrics_valid else "not_available",
            source_event_ids=tool_event_ids,
        ),
        tool_calls_successful=count(successful),
        tool_calls_failed=count(failed),
        unknown_outcome_tool_calls=count(unknown),
        read_calls=count(categories["read"]),
        search_calls=count(categories["search"]),
        edit_calls=count(categories["edit"]),
        write_calls=count(categories["write"]),
        shell_calls=count(shell_calls),
        agent_invoked_test_calls=count(categories["test"]),
        calls_before_first_edit=before if tool_metrics_valid else _unavailable("calls", "invalid_source" if not tool_integrity else "capture_incomplete"),
        calls_after_last_edit=after if tool_metrics_valid else _unavailable("calls", "invalid_source" if not tool_integrity else "capture_incomplete"),
        exact_duplicate_tool_calls=duplicates if tool_metrics_valid else _unavailable("calls", "invalid_source" if not tool_integrity else "capture_incomplete"),
        repeated_reads_of_unchanged_files=repeated_reads if tool_metrics_valid else _unavailable("reads", "invalid_source" if not tool_integrity else "capture_incomplete"),
        repeated_identical_shell_commands=repeated_shell if tool_metrics_valid else _unavailable("calls", "invalid_source" if not tool_integrity else "capture_incomplete"),
        turns_with_reasoning_but_no_action=(
            reasoning_only if complete_llm_capture and complete_tool_capture
            else _unavailable("turns", "capture_incomplete")
        ),
    )


def _calculate_derived(
    timing: TimingMetrics, tokens: TokenMetrics, behavior: BehaviorMetrics
) -> DerivedMetrics:
    mutations = (
        int(behavior.edit_calls.value) + int(behavior.write_calls.value)
        if behavior.edit_calls.availability == "available"
        and behavior.write_calls.availability == "available"
        else None
    )
    return DerivedMetrics(
        tokens_per_tool_call=_divide_by_metric(tokens.total_tokens, behavior.tool_calls_total, "tokens/call"),
        tokens_per_edit=_divide_metrics(tokens.total_tokens, mutations, "tokens/mutation call"),
        reads_per_edit=_divide_metrics(behavior.read_calls, mutations, "reads/mutation call"),
        searches_per_edit=_divide_metrics(behavior.search_calls, mutations, "searches/mutation call"),
        seconds_per_edit=_divide_metrics(timing.wall_time_seconds, mutations, "seconds/mutation call"),
        failed_tool_call_rate=_divide_by_metric(behavior.tool_calls_failed, behavior.tool_calls_total, "ratio"),
        reasoning_to_output_ratio=_divide_by_metric(tokens.reasoning_tokens_total, tokens.output_tokens_total, "ratio"),
    )


def _calculate_git(root: Path, diagnostics: list[str]) -> tuple[GitResultMetrics, _GitSummary | None]:
    paths: set[str] = set()
    created: set[str] = set()
    deleted: set[str] = set()
    renamed = 0
    ambiguous = False
    try:
        snapshot_paths = _snapshot_file_paths(root / "source/source.tar")
        for raw_line in (root / "git/status.txt").read_text(
            encoding="utf-8", errors="surrogateescape"
        ).splitlines():
            if len(raw_line) < 4:
                ambiguous = True
                continue
            code = raw_line[:2]
            raw_path = raw_line[3:]
            if code in {"??", "!!"}:
                continue
            if " -> " in raw_path:
                old, new = raw_path.split(" -> ", 1)
                old_path = _canonical_path(old)
                new_path = _canonical_path(new)
                if old_path is None or new_path is None:
                    ambiguous = True
                    continue
                paths.add(new_path)
                renamed += 1
                continue
            path = _decode_status_path(raw_path)
            if path is None:
                ambiguous = True
                continue
            canonical = _canonical_path(path)
            if canonical is None:
                ambiguous = True
                continue
            paths.add(canonical)
            if "A" in code:
                created.add(canonical)
            if "D" in code:
                deleted.add(canonical)
        for inventory_name in ("git/untracked.txt", "git/ignored.txt"):
            for path in _read_inventory(root / inventory_name):
                canonical = _canonical_path(path)
                if canonical is None:
                    ambiguous = True
                elif canonical in snapshot_paths:
                    paths.add(canonical)
                    created.add(canonical)
        tracked_stats = GitNumstatRecord.model_validate_json(
            (root / GIT_TRACKED_NUMSTAT_PATH).read_bytes()
        )
        lines_added, lines_deleted, binary_files, line_error = _sum_numstat(
            tracked_stats
        )
        tracked_added = set(_tracked_added_paths(root / "git/status.txt"))
        extra_created = created - tracked_added
        untracked_stats = GitNumstatRecord.model_validate_json(
            (root / GIT_UNTRACKED_NUMSTAT_PATH).read_bytes()
        )
        entries = {entry.path: entry for entry in untracked_stats.entries}
        tracked_entry_paths = {entry.path for entry in tracked_stats.entries}
        tracked_paths = paths - extra_created
        if not tracked_entry_paths.issubset(tracked_paths):
            line_error = "tracked_numstat_inventory_mismatch"
            lines_added = lines_deleted = binary_files = None
        elif tracked_stats.git_version != untracked_stats.git_version:
            line_error = "numstat_git_version_mismatch"
            lines_added = lines_deleted = binary_files = None
        elif set(entries) != extra_created:
            line_error = "untracked_numstat_inventory_mismatch"
            lines_added = lines_deleted = binary_files = None
        elif line_error is None:
            for entry in entries.values():
                if entry.availability != "available":
                    line_error = entry.unavailable_reason or "untracked_numstat_unavailable"
                    lines_added = lines_deleted = binary_files = None
                    break
                if entry.binary:
                    assert binary_files is not None
                    binary_files += 1
                else:
                    assert lines_added is not None and lines_deleted is not None
                    assert entry.lines_added is not None and entry.lines_deleted is not None
                    lines_added += entry.lines_added
                    lines_deleted += entry.lines_deleted
    except Exception as exc:
        diagnostics.append(f"Git metric evidence could not be parsed: {exc}")
        ambiguous = True
        lines_added = lines_deleted = binary_files = None
        line_error = "ambiguous_evidence"
    if ambiguous:
        unavailable = _unavailable("files", "ambiguous_evidence")
        result = GitResultMetrics(
            files_changed=unavailable,
            files_created=unavailable,
            files_deleted=unavailable,
            files_renamed=unavailable,
            lines_added=_unavailable("lines", "ambiguous_evidence"),
            lines_deleted=_unavailable("lines", "ambiguous_evidence"),
            binary_files_changed=_unavailable("files", "ambiguous_evidence"),
            source_files_changed=unavailable,
            test_files_changed=unavailable,
            configuration_files_changed=unavailable,
        )
        return result, None
    ordered = tuple(sorted(paths))
    evidence = (
        "git/status.txt",
        "git/untracked.txt",
        "git/ignored.txt",
        GIT_TRACKED_NUMSTAT_PATH,
        GIT_UNTRACKED_NUMSTAT_PATH,
        "source/source.tar",
    )
    file_metric = lambda value: _available(value, "files", "git_native", artifacts=evidence)
    if line_error is None:
        line_evidence = (GIT_TRACKED_NUMSTAT_PATH, GIT_UNTRACKED_NUMSTAT_PATH)
        added_metric = _available(lines_added, "lines", "git_native", artifacts=line_evidence)
        deleted_metric = _available(lines_deleted, "lines", "git_native", artifacts=line_evidence)
        binary_metric = _available(binary_files, "files", "git_native", artifacts=line_evidence)
    else:
        added_metric = _unavailable("lines", "ambiguous_evidence", artifacts=("git/diff.patch",))
        deleted_metric = _unavailable("lines", "ambiguous_evidence", artifacts=("git/diff.patch",))
        binary_metric = _unavailable("files", "ambiguous_evidence", artifacts=("git/diff.patch",))
        diagnostics.append(f"Git line metrics unavailable: {line_error}")
    summary = _GitSummary(
        paths=ordered,
        created=len(created),
        deleted=len(deleted),
        renamed=renamed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        binary_files=binary_files,
        line_reason=line_error,
    )
    return GitResultMetrics(
        files_changed=file_metric(len(ordered)),
        files_created=file_metric(len(created)),
        files_deleted=file_metric(len(deleted)),
        files_renamed=file_metric(renamed),
        lines_added=added_metric,
        lines_deleted=deleted_metric,
        binary_files_changed=binary_metric,
        source_files_changed=file_metric(sum(_is_source(path) for path in ordered)),
        test_files_changed=file_metric(sum(_is_test(path) for path in ordered)),
        configuration_files_changed=file_metric(sum(_is_config(path) for path in ordered)),
    ), summary


def _classify_termination(
    manifest: RunManifest,
    events: tuple[NormalizedEvent, ...],
    git_summary: _GitSummary | None,
    git_metrics: GitResultMetrics,
    raw_events: tuple[RawEvent, ...] = (),
) -> TerminationResult:
    by_kind: dict[str, list[NormalizedEvent]] = {}
    for event in events:
        by_kind.setdefault(event.event_kind, []).append(event)
    def result(name: str, reason: str, kinds: tuple[str, ...] = ()) -> TerminationResult:
        evidence = tuple(
            event.event_id for kind in kinds for event in by_kind.get(kind, [])
        )
        return TerminationResult(
            termination_class=name,  # type: ignore[arg-type]
            reason=reason,
            source_event_ids=evidence,
            source_artifact_paths=(RUN_MANIFEST_PATH, MANIFEST_PATH),
        )
    preconditions = [
        event for event in raw_events if event.event_type == "precondition_failed"
    ]
    if preconditions:
        return TerminationResult(
            termination_class="precondition_failed",
            reason="required benchmark precondition failed",
            source_event_ids=tuple(event.raw_event_id for event in preconditions),
            source_artifact_paths=(RAW_EVENTS_PATH, RUN_MANIFEST_PATH),
        )
    preservation_failures = [
        event for event in raw_events if event.event_type == "preservation_failed"
    ]
    if preservation_failures:
        underlying = _classify_termination(
            manifest,
            events,
            git_summary,
            git_metrics,
            tuple(
                event for event in raw_events
                if event.event_type != "preservation_failed"
            ),
        )
        return TerminationResult(
            termination_class="preservation_failed",
            underlying_termination_class=underlying.termination_class,
            reason="required result preservation or verification failed",
            source_event_ids=tuple(
                event.raw_event_id for event in preservation_failures
            ),
            source_artifact_paths=(RAW_EVENTS_PATH,),
        )
    if by_kind.get("timeout") or manifest.observed_execution_outcome == "timeout":
        return result("timeout", "runner task deadline fired", ("timeout", "process_termination"))
    killed = any(
        event.payload.get("status") in {"killed", "sigkill", "terminated"}
        for event in by_kind.get("process_termination", [])
    )
    if killed:
        return result("process_killed", "task process was explicitly killed", ("process_termination",))
    if by_kind.get("context_overflow"):
        return result("context_overflow", "recognized context overflow event", ("context_overflow",))
    if by_kind.get("output_truncation") or manifest.observed_execution_outcome == "output_truncation":
        return result("output_truncation", "recognized task output truncation", ("output_truncation",))
    fatal_backend = [
        event for event in by_kind.get("backend_error", [])
        if event.payload.get("fatal") is True
        or event.payload.get("prevented_ordinary_completion") is True
    ]
    if fatal_backend:
        return result("model_backend_error", "backend error prevented ordinary completion", ("backend_error",))
    invalid_output = [
        event for event in by_kind.get("harness_error", [])
        if event.payload.get("error_type")
        in {"invalid_harness_output", "malformed_harness_response"}
    ]
    fatal_harness = [
        event for event in by_kind.get("harness_error", [])
        if event not in invalid_output
    ]
    if fatal_harness or (
        manifest.observed_execution_outcome == "harness_crash" and not invalid_output
    ):
        return result("harness_crash", "harness ended abnormally", ("harness_error", "process_termination"))
    if invalid_output:
        return result("invalid_harness_output", "required harness output was malformed", ("harness_error",))
    if git_summary is None or git_metrics.files_changed.availability != "available":
        return result("unknown_other", "ordinary completion lacks a complete Git comparison", ("run_end",))
    if len(git_summary.paths) == 0:
        return result("no_changes", "ordinary completion produced no preserved file changes", ("run_end",))
    if manifest.observed_execution_outcome in {"success", "no_changes"}:
        return result("success", "ordinary completion produced preserved file changes", ("run_end",))
    return result("unknown_other", "terminal evidence does not match a defined class", ("run_end",))


def _validate_event_provenance(
    raw_events: tuple[RawEvent, ...],
    normalized_events: tuple[NormalizedEvent, ...],
) -> None:
    raw_by_id = {event.raw_event_id: event for event in raw_events}
    if len(raw_by_id) != len(raw_events):
        raise MetricsCalculationError("raw event IDs are not unique")
    normalized_ids = [event.event_id for event in normalized_events]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise MetricsCalculationError("normalized event IDs are not unique")
    for event in normalized_events:
        for reference in event.raw_event_refs:
            raw = raw_by_id.get(reference.raw_event_id)
            if raw is None:
                raise MetricsCalculationError(
                    f"normalized event {event.event_id} references missing raw evidence"
                )
            if (
                reference.raw_sequence != raw.sequence
                or reference.raw_record_digest != raw.record_digest
            ):
                raise MetricsCalculationError(
                    f"normalized event {event.event_id} has invalid raw provenance"
                )


def _correlate_tools(events: tuple[NormalizedEvent, ...], diagnostics: list[str]) -> tuple[_ToolCall, ...]:
    ends: dict[str, NormalizedEvent] = {}
    for event in events:
        if event.event_kind != "tool_call_end":
            continue
        call_id = event.payload.get("tool_call_id")
        if isinstance(call_id, str) and call_id not in ends:
            ends[call_id] = event
        elif isinstance(call_id, str):
            diagnostics.append(f"duplicate tool end correlation ID: {call_id}")
    calls: list[_ToolCall] = []
    seen: set[str] = set()
    for event in events:
        if event.event_kind != "tool_call_start":
            continue
        call_id = event.payload.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            call_id = event.event_id
            diagnostics.append(f"tool correlation ID unavailable: {event.event_id}")
        if call_id in seen:
            diagnostics.append(f"duplicate tool start correlation ID: {call_id}")
        seen.add(call_id)
        category = event.payload.get("category")
        if category not in _CATEGORIES:
            category = "other"
        end = ends.get(call_id)
        outcome = end.payload.get("outcome") if end else "unknown"
        if outcome not in {"success", "failure", "timeout", "cancelled", "unknown"}:
            outcome = "unknown"
        calls.append(_ToolCall(call_id, event, end, str(category), str(outcome)))
    return tuple(calls)


def _is_exact_execution_start(event: NormalizedEvent) -> bool:
    """Whether a tool-start timestamp explicitly denotes execution start.

    Older generic fixtures predate the provenance field and retain their
    established semantics.  Harness adapters added after timing-provenance-v1
    must declare the field rather than relying on this compatibility default.
    """
    return event.payload.get("timing_semantics", "harness_tool_execution_start") == (
        "harness_tool_execution_start"
    )


def _is_exact_execution_end(event: NormalizedEvent | None) -> bool:
    if event is None:
        return False
    return event.payload.get("timing_semantics", "harness_tool_execution_end") == (
        "harness_tool_execution_end"
    )


def _has_exact_tool_execution_timing(call: _ToolCall) -> bool:
    return _is_exact_execution_start(call.start) and _is_exact_execution_end(call.end)


def _tool_identity_valid(events: tuple[NormalizedEvent, ...]) -> bool:
    starts = [
        event.payload.get("tool_call_id")
        for event in events
        if event.event_kind == "tool_call_start"
    ]
    ends = [
        event.payload.get("tool_call_id")
        for event in events
        if event.event_kind == "tool_call_end"
    ]
    return (
        all(isinstance(item, str) and item for item in starts + ends)
        and len(set(starts)) == len(starts)
        and len(set(ends)) == len(ends)
        and set(ends).issubset(set(starts))
    )


def _interval_sum(
    starts: list[NormalizedEvent],
    ends: list[NormalizedEvent],
    key: str,
    units: str,
    complete_capture: bool,
) -> ScalarMetric:
    if not complete_capture:
        return _unavailable(units, "capture_incomplete")
    start_ids = [event.payload.get(key) for event in starts]
    end_ids = [event.payload.get(key) for event in ends]
    if (
        not all(isinstance(item, str) and item for item in start_ids + end_ids)
        or len(set(start_ids)) != len(start_ids)
        or len(set(end_ids)) != len(end_ids)
    ):
        return _unavailable(units, "invalid_source")
    if not starts:
        if ends:
            return _unavailable(units, "capture_incomplete")
        return (
            _available(0.0, units, "deterministically_calculated")
            if complete_capture
            else _unavailable(units, "source_not_exposed")
        )
    end_by_id = {
        event.payload.get(key): event
        for event in ends
        if isinstance(event.payload.get(key), str)
    }
    duration_ns = 0
    source_ids: list[str] = []
    for start in starts:
        correlation = start.payload.get(key)
        end = end_by_id.get(correlation)
        if start.elapsed_ns is None or end is None or end.elapsed_ns is None or end.elapsed_ns < start.elapsed_ns:
            return _unavailable(units, "capture_incomplete")
        duration_ns += end.elapsed_ns - start.elapsed_ns
        source_ids.extend((start.event_id, end.event_id))
    return _available(duration_ns / 1_000_000_000, units, "deterministically_calculated", events=tuple(source_ids))


def _tool_interval_sum(tools: tuple[_ToolCall, ...], predicate: Any, complete_capture: bool) -> ScalarMetric:
    if not complete_capture:
        return _unavailable("seconds", "capture_incomplete")
    selected = [call for call in tools if predicate(call)]
    if not selected:
        return _available(0.0, "seconds", "deterministically_calculated") if complete_capture else _unavailable("seconds", "source_not_exposed")
    total = 0
    source_ids: list[str] = []
    for call in selected:
        if call.start.elapsed_ns is None or call.end is None or call.end.elapsed_ns is None or call.end.elapsed_ns < call.start.elapsed_ns:
            return _unavailable("seconds", "capture_incomplete")
        total += call.end.elapsed_ns - call.start.elapsed_ns
        source_ids.extend((call.start.event_id, call.end.event_id))
    return _available(total / 1_000_000_000, "seconds", "deterministically_calculated", events=tuple(source_ids))


def _first_elapsed(events: list[NormalizedEvent], noun: str) -> ScalarMetric:
    usable = [event for event in events if event.elapsed_ns is not None]
    if not usable:
        reason = "event_not_observed" if not events else "invalid_source"
        return _unavailable("seconds", reason)
    event = min(usable, key=lambda item: (item.elapsed_ns, item.sequence))
    return _available(event.elapsed_ns / 1_000_000_000, "seconds", "normalized_event_exact", events=(event.event_id,))


def _token_metric(event: NormalizedEvent | None, field: str) -> ScalarMetric:
    if event is None:
        return _unavailable("tokens", "capture_incomplete")
    value = _strict_int(event.payload.get(field))
    method = event.payload.get("token_source")
    if value is None or value < 0:
        return _unavailable("tokens", "source_not_exposed", events=(event.event_id,))
    if method not in _TOKEN_METHODS:
        return _unavailable("tokens", "invalid_source", events=(event.event_id,))
    if method == "tokenizer_reconstructed":
        tokenizer_identity = event.payload.get("tokenizer_identity")
        tokenizer_digest = event.payload.get("tokenizer_digest")
        if (
            not isinstance(tokenizer_identity, str)
            or not isinstance(tokenizer_digest, str)
            or len(tokenizer_digest) != 64
        ):
            return _unavailable("tokens", "invalid_source", events=(event.event_id,))
    return _available(value, "tokens", str(method), events=(event.event_id,), source_methods=(str(method),))


def _integer_metric(event: NormalizedEvent, field: str, units: str, *, require_positive: bool = False) -> ScalarMetric:
    value = _strict_int(event.payload.get(field))
    if value is None or (require_positive and value <= 0) or value < 0:
        return _unavailable(units, "source_not_exposed", events=(event.event_id,))
    return _available(value, units, "normalized_event_exact", events=(event.event_id,))


def _aggregate_tokens(values: list[int], complete: bool, source_events: list[NormalizedEvent]) -> ScalarMetric:
    if not source_events:
        return _unavailable("tokens", "source_not_exposed")
    if not complete or len(values) != len(source_events):
        return _unavailable("tokens", "source_not_exposed", events=tuple(event.event_id for event in source_events))
    methods = tuple(
        sorted(
            {
                str(event.payload["token_source"])
                for event in source_events
                if event.payload.get("token_source") in _TOKEN_METHODS
            }
        )
    )
    return _available(
        sum(values),
        "tokens",
        "deterministically_calculated",
        events=tuple(event.event_id for event in source_events),
        source_methods=methods,
    )


def _tokens_before_edit(
    indexed: list[tuple[int, NormalizedEvent]],
    responses: dict[str, NormalizedEvent],
    first_edit: ScalarMetric,
) -> tuple[ScalarMetric, ScalarMetric]:
    if first_edit.availability != "available":
        return _unavailable("tokens", "event_not_observed"), _unavailable("tokens", "event_not_observed")
    boundary_ns = int(float(first_edit.value) * 1_000_000_000)
    total = 0
    reasoning = 0
    evidence: list[str] = []
    for _, request in indexed:
        request_id = request.payload.get("request_id")
        response = responses.get(request_id) if isinstance(request_id, str) else None
        if response is None or response.elapsed_ns is None or response.elapsed_ns > boundary_ns:
            continue
        input_metric = _token_metric(request, "context_tokens")
        if input_metric.availability != "available":
            input_metric = _token_metric(response, "input_tokens")
        output_metric = _token_metric(response, "output_tokens")
        reasoning_metric = _token_metric(response, "reasoning_tokens")
        if input_metric.availability != "available" or output_metric.availability != "available":
            return _unavailable("tokens", "source_not_exposed"), _unavailable("tokens", "source_not_exposed")
        if reasoning_metric.availability != "available":
            return _available(total + int(input_metric.value) + int(output_metric.value), "tokens", "deterministically_calculated"), _unavailable("tokens", "source_not_exposed")
        total += int(input_metric.value) + int(output_metric.value)
        reasoning += int(reasoning_metric.value)
        evidence.extend((request.event_id, response.event_id))
    return (
        _available(total, "tokens", "deterministically_calculated", events=tuple(evidence)),
        _available(reasoning, "tokens", "deterministically_calculated", events=tuple(evidence)),
    )


def _compaction_point(index: int, event: NormalizedEvent) -> CompactionPoint:
    before = _token_metric(event, "before_context_tokens")
    after = _token_metric(event, "after_context_tokens")
    maximum = _integer_metric(event, "configured_max_context_tokens", "tokens", require_positive=True)
    return CompactionPoint(
        compaction_index=index,
        compaction_event_id=event.event_id,
        elapsed_seconds=event.elapsed_seconds,
        tokens_before_compaction=before,
        tokens_after_compaction=after,
        context_max_tokens=maximum,
        before_utilization_percent=_ratio_percent(before, maximum),
        after_utilization_percent=_ratio_percent(after, maximum),
    )


def _calls_before_first_edit(tools: tuple[_ToolCall, ...], edits: list[_ToolCall]) -> ScalarMetric:
    if not edits:
        return _unavailable("calls", "event_not_observed")
    boundary = min(call.start.elapsed_ns for call in edits if call.start.elapsed_ns is not None)
    return _available(sum(call.start.elapsed_ns is not None and call.start.elapsed_ns < boundary for call in tools), "calls", "deterministically_calculated", events=tuple(call.start.event_id for call in tools))


def _calls_after_last_edit(tools: tuple[_ToolCall, ...], edits: list[_ToolCall]) -> ScalarMetric:
    if not edits:
        return _unavailable("calls", "event_not_observed")
    boundaries: list[int] = []
    for call in edits:
        event = call.end or call.start
        if event.elapsed_ns is None:
            return _unavailable("calls", "invalid_source")
        boundaries.append(event.elapsed_ns)
    boundary = max(boundaries)
    return _available(sum(call.start.elapsed_ns is not None and call.start.elapsed_ns > boundary for call in tools), "calls", "deterministically_calculated", events=tuple(call.start.event_id for call in tools))


def _duplicate_calls(tools: tuple[_ToolCall, ...]) -> ScalarMetric:
    seen: set[str] = set()
    duplicates = 0
    evidence: list[str] = []
    for call in tools:
        payload = call.start.payload
        if "arguments" not in payload or not isinstance(payload.get("tool_name"), str):
            return _unavailable("calls", "source_not_exposed", events=tuple(item.start.event_id for item in tools))
        digest = canonical_sha256({"category": call.category, "tool_name": payload["tool_name"], "arguments": payload["arguments"]})
        if digest in seen:
            duplicates += 1
        seen.add(digest)
        evidence.append(call.start.event_id)
    return _available(duplicates, "calls", "deterministically_calculated", events=tuple(evidence))


def _repeated_shell_calls(tools: tuple[_ToolCall, ...]) -> ScalarMetric:
    selected = [
        call for call in tools
        if call.category == "shell"
        or (call.category == "test" and call.start.payload.get("uses_shell") is True)
    ]
    seen: set[str] = set()
    repeats = 0
    for call in selected:
        payload = call.start.payload
        if "command" not in payload or "working_directory" not in payload:
            return _unavailable("calls", "source_not_exposed", events=tuple(item.start.event_id for item in selected))
        digest = canonical_sha256({"command": payload["command"], "working_directory": payload["working_directory"], "environment": payload.get("environment", {})})
        if digest in seen:
            repeats += 1
        seen.add(digest)
    return _available(repeats, "calls", "deterministically_calculated", events=tuple(item.start.event_id for item in selected))


def _repeated_reads(events: tuple[NormalizedEvent, ...], tools: tuple[_ToolCall, ...]) -> ScalarMetric:
    outcomes = {call.call_id: call.outcome for call in tools}
    last_hash: dict[str, str] = {}
    mutated: set[str] = set()
    repeats = 0
    evidence: list[str] = []
    for event in events:
        if event.event_kind in {"file_edit", "file_write"}:
            call_id = event.payload.get("tool_call_id")
            if isinstance(call_id, str) and outcomes.get(call_id) == "success":
                path = _canonical_path_value(event.payload.get("path"))
                if path is None:
                    return _unavailable("reads", "source_not_exposed")
                mutated.add(path)
        elif event.event_kind == "file_read":
            call_id = event.payload.get("tool_call_id")
            if not isinstance(call_id, str) or outcomes.get(call_id) != "success":
                continue
            path = _canonical_path_value(event.payload.get("path"))
            content_hash = event.payload.get("content_sha256")
            if path is None or not isinstance(content_hash, str):
                return _unavailable("reads", "source_not_exposed")
            if path in last_hash and path not in mutated and last_hash[path] == content_hash:
                repeats += 1
            last_hash[path] = content_hash
            mutated.discard(path)
            evidence.append(event.event_id)
    return _available(repeats, "reads", "deterministically_calculated", events=tuple(evidence))


def _reasoning_only_turns(events: tuple[NormalizedEvent, ...]) -> ScalarMetric:
    reasoning_turns: set[str] = set()
    action_turns: set[str] = set()
    visible_turns: set[str] = set()
    response_visibility: dict[str, bool] = {}
    evidence: list[str] = []
    for event in events:
        turn = event.payload.get("turn_id")
        if event.event_kind == "reasoning":
            if not isinstance(turn, str):
                return _unavailable("turns", "source_not_exposed")
            reasoning_turns.add(turn)
            evidence.append(event.event_id)
        elif event.event_kind == "tool_call_start" and isinstance(turn, str):
            action_turns.add(turn)
        elif event.event_kind == "llm_response" and isinstance(turn, str):
            visible = event.payload.get("visible_answer_present")
            if isinstance(visible, bool):
                response_visibility[turn] = visible
                if visible:
                    visible_turns.add(turn)
    if any(turn not in response_visibility for turn in reasoning_turns):
        return _unavailable("turns", "source_not_exposed", events=tuple(evidence))
    return _available(len(reasoning_turns - action_turns - visible_turns), "turns", "deterministically_calculated", events=tuple(evidence))


def _sum_numstat(
    record: GitNumstatRecord,
) -> tuple[int | None, int | None, int | None, str | None]:
    added = deleted = binary = 0
    seen: set[str] = set()
    for entry in record.entries:
        if entry.path in seen:
            return None, None, None, "duplicate_git_numstat_path"
        seen.add(entry.path)
        if entry.availability != "available":
            return None, None, None, entry.unavailable_reason or "git_numstat_unavailable"
        if entry.binary:
            binary += 1
        else:
            assert entry.lines_added is not None and entry.lines_deleted is not None
            added += entry.lines_added
            deleted += entry.lines_deleted
    return added, deleted, binary, None


def _snapshot_file_paths(path: Path) -> set[str]:
    with tarfile.open(path, mode="r:") as archive:
        return {member.name for member in archive.getmembers() if member.isfile() or member.issym()}


def _read_inventory(path: Path) -> tuple[str, ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="ascii").splitlines())


def _tracked_added_paths(status_path: Path) -> tuple[str, ...]:
    paths: list[str] = []
    for line in status_path.read_text(encoding="utf-8", errors="surrogateescape").splitlines():
        if len(line) >= 4 and line[:2] not in {"??", "!!"} and "A" in line[:2]:
            decoded = _decode_status_path(line[3:])
            if decoded is not None:
                paths.append(decoded)
    return tuple(paths)


def _decode_status_path(value: str) -> str | None:
    if value.startswith('"'):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, str) else None
    return value


def _canonical_path_value(value: object) -> str | None:
    return _canonical_path(value) if isinstance(value, str) else None


def _canonical_path(value: str) -> str | None:
    path = PurePosixPath(value.replace("\\", "/"))
    if not value or path.is_absolute() or ".." in path.parts:
        return None
    parts = tuple(part for part in path.parts if part not in {"", "."})
    return PurePosixPath(*parts).as_posix() if parts else None


def _tool_targets_worktree(event: NormalizedEvent) -> bool:
    path = event.payload.get("path")
    if not isinstance(path, str):
        arguments = event.payload.get("arguments")
        path = arguments.get("path") if isinstance(arguments, dict) else None
    return isinstance(path, str) and _canonical_path(path) is not None


def _is_source(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in _SOURCE_SUFFIXES


def _is_test(path: str) -> bool:
    item = PurePosixPath(path)
    lower_parts = tuple(part.lower() for part in item.parts)
    name = item.name.lower()
    return "tests" in lower_parts or "test" in lower_parts or name.startswith("test_") or name.endswith(("_test.py", ".test.js", ".spec.ts"))


def _is_config(path: str) -> bool:
    item = PurePosixPath(path)
    return item.name.lower() in _CONFIG_NAMES or item.suffix.lower() in _CONFIG_SUFFIXES


def _ratio_percent(numerator: ScalarMetric, denominator: ScalarMetric) -> ScalarMetric:
    if numerator.availability != "available" or denominator.availability != "available":
        return _unavailable("percent", "source_not_exposed")
    if float(denominator.value) <= 0:
        return _unavailable("percent", "invalid_source")
    return _available(100.0 * float(numerator.value) / float(denominator.value), "percent", "deterministically_calculated", events=tuple(sorted(set(numerator.provenance.source_event_ids + denominator.provenance.source_event_ids))))


def _sum_metrics(first: ScalarMetric, second: ScalarMetric, units: str) -> ScalarMetric:
    if first.availability != "available" or second.availability != "available":
        return _unavailable(units, "source_not_exposed")
    return _available(
        float(first.value) + float(second.value)
        if isinstance(first.value, float) or isinstance(second.value, float)
        else int(first.value) + int(second.value),
        units,
        "deterministically_calculated",
        events=tuple(sorted(set(first.provenance.source_event_ids + second.provenance.source_event_ids))),
        source_methods=tuple(sorted(set(first.provenance.source_methods + second.provenance.source_methods))),
    )


def _subtract_metrics(first: ScalarMetric, second: ScalarMetric, units: str) -> ScalarMetric:
    if first.availability != "available" or second.availability != "available":
        return _unavailable(units, "source_not_exposed")
    return _available(float(first.value) - float(second.value) if isinstance(first.value, float) or isinstance(second.value, float) else int(first.value) - int(second.value), units, "deterministically_calculated", events=tuple(sorted(set(first.provenance.source_event_ids + second.provenance.source_event_ids))))


def _divide_by_metric(numerator: ScalarMetric, denominator: ScalarMetric, units: str) -> ScalarMetric:
    if denominator.availability != "available":
        return _unavailable(units, "source_not_exposed")
    return _divide_metrics(numerator, int(denominator.value), units)


def _divide_metrics(numerator: ScalarMetric, denominator: int | None, units: str) -> ScalarMetric:
    if numerator.availability != "available":
        return _unavailable(units, "source_not_exposed")
    if denominator is None:
        return _unavailable(units, "source_not_exposed")
    if denominator == 0:
        return _unavailable(units, "zero_denominator")
    return _available(float(numerator.value) / denominator, units, "deterministically_calculated", events=numerator.provenance.source_event_ids)


def _strict_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _available(
    value: int | float | None,
    units: str,
    method: str,
    *,
    events: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
    source_methods: tuple[str, ...] = (),
) -> ScalarMetric:
    assert value is not None
    return ScalarMetric(
        value=value,
        units=units,
        availability="available",
        provenance=MetricProvenance(
            method=method,  # type: ignore[arg-type]
            source_event_ids=events,
            source_artifact_paths=artifacts,
            source_methods=source_methods,
        ),
    )


def _unavailable(
    units: str,
    reason: str,
    *,
    events: tuple[str, ...] = (),
    artifacts: tuple[str, ...] = (),
) -> ScalarMetric:
    return ScalarMetric(
        value=None,
        units=units,
        availability="unavailable",
        unavailable_reason=reason,  # type: ignore[arg-type]
        provenance=MetricProvenance(
            method="not_available",
            source_event_ids=events,
            source_artifact_paths=artifacts,
        ),
    )


def _not_applicable(units: str) -> ScalarMetric:
    return ScalarMetric(
        value=None,
        units=units,
        availability="not_applicable",
        unavailable_reason="not_applicable",
        provenance=MetricProvenance(method="not_available"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
