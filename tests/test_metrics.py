from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_bench.cli import app
from agent_bench.events import load_normalized_events, load_raw_events
from agent_bench.fake_harness import FakeHarness
from agent_bench.metric_models import RunMetrics
from agent_bench.metrics import (
    _calculate_behavior,
    _calculate_tokens_and_context,
    _classify_termination,
    _correlate_tools,
    _divide_metrics,
    _interval_sum,
    _token_metric,
    _tool_identity_valid,
    _available,
    calculate_run_metrics,
)
from agent_bench.metrics_storage import (
    MetricsStorageError,
    store_metrics_artifact,
    verify_metrics_artifact,
)
from agent_bench.runner import execute_run
from conftest import GitRepositoryFixture, RunFixture


def _run(
    repository: GitRepositoryFixture,
    fixture: RunFixture,
    scenario: str,
):
    return execute_run(
        run_definition=fixture.run_definition,
        prompt_content=fixture.prompt_content,
        adapter=FakeHarness(scenario),  # type: ignore[arg-type]
        adapter_scenario=scenario,
        artifacts_root=repository.artifacts_root,
        worktrees_root=repository.worktrees_root,
        isolation_root=repository.path.parent / "metrics-isolation",
    )


@pytest.fixture
def metrics_run(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
):
    return _run(git_repository, run_fixture, "metrics")


def test_calculates_timing_token_context_and_compaction_metrics(metrics_run: object) -> None:
    metrics = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]

    assert metrics.timing.wall_time_seconds.value > 0
    assert metrics.timing.llm_time_seconds.value >= 0
    assert metrics.timing.tool_execution_time_seconds.value >= 0
    assert metrics.timing.time_to_first_llm_request_seconds.availability == "available"
    assert metrics.timing.time_to_first_edit_seconds.availability == "available"
    assert metrics.timing.time_to_first_test_command_seconds.availability == "available"
    assert metrics.tokens.input_tokens_total.value == 490
    assert metrics.tokens.output_tokens_total.value == 70
    assert metrics.tokens.reasoning_tokens_total.value == 28
    assert metrics.tokens.visible_answer_tokens_total.value == 42
    assert metrics.tokens.total_tokens.value == 560
    assert metrics.tokens.tokens_before_first_edit.value == 330
    assert metrics.tokens.reasoning_tokens_before_first_edit.value == 18
    assert [point.context_used_tokens.value for point in metrics.context.context_used_per_request] == [100, 180, 90, 120]
    assert [point.context_growth_tokens.value for point in metrics.context.context_used_per_request] == [None, 80, -90, 30]
    assert metrics.context.peak_context_tokens.value == 180
    assert metrics.context.peak_context_utilization_percent.value == 18.0
    assert metrics.context.net_context_growth_tokens.value == 20
    assert metrics.context.number_of_compactions.value == 1
    assert metrics.context.context_at_first_compaction_tokens.value == 180
    assert metrics.context.context_utilization_at_first_compaction_percent.value == 18.0
    assert metrics.context.compactions[0].tokens_after_compaction.value == 80


def test_calculates_behavior_duplicates_repeated_reads_and_formulas(metrics_run: object) -> None:
    metrics = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]

    assert metrics.behavior.llm_request_count.value == 4
    assert metrics.behavior.llm_response_count.value == 4
    assert metrics.behavior.tool_calls_total.value == 9
    assert metrics.behavior.tool_calls_by_category.model_dump(exclude={"schema_version"}) == {
        "read": 3,
        "search": 2,
        "edit": 1,
        "write": 0,
        "test": 1,
        "shell": 2,
        "other": 0,
    }
    assert metrics.behavior.tool_calls_successful.value == 9
    assert metrics.behavior.tool_calls_failed.value == 0
    assert metrics.behavior.calls_before_first_edit.value == 2
    assert metrics.behavior.calls_after_last_edit.value == 6
    assert metrics.behavior.exact_duplicate_tool_calls.value == 4
    assert metrics.behavior.repeated_reads_of_unchanged_files.value == 1
    assert metrics.behavior.repeated_identical_shell_commands.value == 1
    assert metrics.behavior.shell_calls.value == 3
    assert metrics.behavior.agent_invoked_test_calls.value == 1
    assert metrics.behavior.turns_with_reasoning_but_no_action.value == 1
    assert metrics.derived.tokens_per_tool_call.value == pytest.approx(560 / 9)
    assert metrics.derived.tokens_per_edit.value == 560.0
    assert metrics.derived.failed_tool_call_rate.value == 0.0
    assert metrics.derived.reasoning_to_output_ratio.value == 0.4


def test_git_metrics_include_preserved_untracked_and_deleted_files(metrics_run: object) -> None:
    metrics = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]

    assert metrics.git_result.files_changed.value == 4
    assert metrics.git_result.files_created.value == 2
    assert metrics.git_result.files_deleted.value == 1
    assert metrics.git_result.files_renamed.value == 0
    assert metrics.git_result.source_files_changed.value == 1
    assert metrics.git_result.test_files_changed.value == 1
    assert metrics.git_result.lines_added.value == 3
    assert metrics.git_result.lines_deleted.value == 2
    assert metrics.git_result.binary_files_changed.value == 1
    assert metrics.termination.termination_class == "success"


def test_no_change_git_and_zero_line_metrics_are_exact(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "no_change")
    metrics = calculate_run_metrics(result.artifact_path)

    assert metrics.git_result.files_changed.value == 0
    assert metrics.git_result.files_created.value == 0
    assert metrics.git_result.files_deleted.value == 0
    assert metrics.git_result.lines_added.value == 0
    assert metrics.git_result.lines_deleted.value == 0
    assert metrics.termination.termination_class == "no_changes"


def test_failed_tool_outcome_counts_are_exact(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    result = _run(git_repository, run_fixture, "failed_tool")
    metrics = calculate_run_metrics(result.artifact_path)

    assert metrics.behavior.tool_calls_total.value == 2
    assert metrics.behavior.tool_calls_successful.value == 1
    assert metrics.behavior.tool_calls_failed.value == 1
    assert metrics.behavior.unknown_outcome_tool_calls.value == 0


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("output_truncation", "output_truncation"),
        ("crash", "harness_crash"),
        ("timeout", "timeout"),
    ],
)
def test_failure_termination_classes_and_unavailable_tokens(
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
    scenario: str,
    expected: str,
) -> None:
    result = _run(git_repository, run_fixture, scenario)
    metrics = calculate_run_metrics(result.artifact_path)

    assert metrics.termination.termination_class == expected
    assert metrics.tokens.input_tokens_total.value is None
    assert metrics.tokens.input_tokens_total.unavailable_reason == "source_not_exposed"


def test_termination_precedence_prefers_timeout_then_context_overflow(metrics_run: object) -> None:
    events = load_normalized_events(metrics_run.normalized_event_path)  # type: ignore[attr-defined]
    exemplar = events[0]
    timeout = exemplar.model_copy(update={"event_kind": "timeout", "event_id": "timeout"})
    overflow = exemplar.model_copy(update={"event_kind": "context_overflow", "event_id": "overflow"})
    truncation = exemplar.model_copy(update={"event_kind": "output_truncation", "event_id": "truncation"})
    metrics = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]
    git_summary = type("Summary", (), {"paths": ("tracked.txt",)})()

    timeout_result = _classify_termination(
        metrics_run.run_manifest,  # type: ignore[attr-defined]
        (overflow, truncation, timeout),
        git_summary,  # type: ignore[arg-type]
        metrics.git_result,
    )
    overflow_result = _classify_termination(
        metrics_run.run_manifest,  # type: ignore[attr-defined]
        (truncation, overflow),
        git_summary,  # type: ignore[arg-type]
        metrics.git_result,
    )

    assert timeout_result.termination_class == "timeout"
    assert overflow_result.termination_class == "context_overflow"


def test_process_backend_invalid_output_and_unknown_termination_classes(metrics_run: object) -> None:
    events = load_normalized_events(metrics_run.normalized_event_path)  # type: ignore[attr-defined]
    exemplar = events[0]
    killed = exemplar.model_copy(
        update={"event_kind": "process_termination", "event_id": "killed", "payload": {"status": "killed"}}
    )
    backend = exemplar.model_copy(
        update={"event_kind": "backend_error", "event_id": "backend", "payload": {"fatal": True}}
    )
    malformed = exemplar.model_copy(
        update={"event_kind": "harness_error", "event_id": "malformed", "payload": {"error_type": "invalid_harness_output"}}
    )
    metrics = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]
    git_summary = type("Summary", (), {"paths": ("tracked.txt",)})()

    process_result = _classify_termination(metrics_run.run_manifest, (killed,), git_summary, metrics.git_result)  # type: ignore[attr-defined,arg-type]
    backend_result = _classify_termination(metrics_run.run_manifest, (backend,), git_summary, metrics.git_result)  # type: ignore[attr-defined,arg-type]
    malformed_result = _classify_termination(metrics_run.run_manifest, (malformed,), git_summary, metrics.git_result)  # type: ignore[attr-defined,arg-type]
    unknown_result = _classify_termination(metrics_run.run_manifest, (), None, metrics.git_result)  # type: ignore[attr-defined]

    assert process_result.termination_class == "process_killed"
    assert backend_result.termination_class == "model_backend_error"
    assert malformed_result.termination_class == "invalid_harness_output"
    assert unknown_result.termination_class == "unknown_other"


def test_overlapping_llm_intervals_are_summed_not_unioned(metrics_run: object) -> None:
    events = load_normalized_events(metrics_run.normalized_event_path)  # type: ignore[attr-defined]
    exemplar = events[0]
    starts = [
        exemplar.model_copy(update={"event_id": "s1", "elapsed_ns": 1_000_000_000, "elapsed_seconds": 1.0, "payload": {"request_id": "a"}}),
        exemplar.model_copy(update={"event_id": "s2", "elapsed_ns": 2_000_000_000, "elapsed_seconds": 2.0, "payload": {"request_id": "b"}}),
    ]
    ends = [
        exemplar.model_copy(update={"event_id": "e1", "elapsed_ns": 4_000_000_000, "elapsed_seconds": 4.0, "payload": {"request_id": "a"}}),
        exemplar.model_copy(update={"event_id": "e2", "elapsed_ns": 5_000_000_000, "elapsed_seconds": 5.0, "payload": {"request_id": "b"}}),
    ]

    metric = _interval_sum(starts, ends, "request_id", "seconds", True)

    assert metric.value == 6.0


def test_incomplete_interval_is_unavailable_not_a_partial_sum(metrics_run: object) -> None:
    events = load_normalized_events(metrics_run.normalized_event_path)  # type: ignore[attr-defined]
    request = next(event for event in events if event.event_kind == "llm_request")

    metric = _interval_sum([request], [], "request_id", "seconds", True)

    assert metric.value is None
    assert metric.unavailable_reason == "capture_incomplete"


def test_missing_context_observation_keeps_point_but_invalidates_aggregates(metrics_run: object) -> None:
    events = list(load_normalized_events(metrics_run.normalized_event_path))  # type: ignore[attr-defined]
    request_position = next(
        index for index, event in enumerate(events) if event.event_kind == "llm_request"
    )
    payload = dict(events[request_position].payload)
    payload.pop("context_tokens")
    events[request_position] = events[request_position].model_copy(update={"payload": payload})
    first_edit = calculate_run_metrics(metrics_run.artifact_path).timing.time_to_first_edit_seconds  # type: ignore[attr-defined]

    tokens, context = _calculate_tokens_and_context(
        tuple(events), first_edit, True, []
    )

    assert len(context.context_used_per_request) == 4
    assert context.context_used_per_request[0].context_used_tokens.value is None
    assert context.context_used_per_request[1].context_used_tokens.value == 180
    assert context.peak_context_tokens.value is None
    assert tokens.input_tokens_total.value is None


def test_api_response_prompt_tokens_feed_context_without_estimation(
    metrics_run: object,
) -> None:
    events = list(load_normalized_events(metrics_run.normalized_event_path))  # type: ignore[attr-defined]
    request_position = next(
        index for index, event in enumerate(events) if event.event_kind == "llm_request"
    )
    request = events[request_position]
    request_payload = dict(request.payload)
    expected = request_payload.pop("context_tokens")
    events[request_position] = request.model_copy(update={"payload": request_payload})
    response_position = next(
        index
        for index, event in enumerate(events)
        if event.event_kind == "llm_response"
        and event.payload.get("request_id") == request.payload.get("request_id")
    )
    response = events[response_position]
    events[response_position] = response.model_copy(
        update={"payload": {**response.payload, "input_tokens": expected}}
    )
    first_edit = calculate_run_metrics(
        metrics_run.artifact_path  # type: ignore[attr-defined]
    ).timing.time_to_first_edit_seconds

    tokens, context = _calculate_tokens_and_context(
        tuple(events), first_edit, True, []
    )

    assert tokens.input_tokens_total.value == 490
    assert context.context_used_per_request[0].context_used_tokens.value == expected
    assert (
        context.context_used_per_request[0].context_used_tokens.provenance.method
        == "api_exact"
    )


def test_missing_context_max_keeps_tokens_but_utilization_is_unavailable(metrics_run: object) -> None:
    events = list(load_normalized_events(metrics_run.normalized_event_path))  # type: ignore[attr-defined]
    request_position = next(
        index for index, event in enumerate(events) if event.event_kind == "llm_request"
    )
    payload = dict(events[request_position].payload)
    payload.pop("configured_max_context_tokens")
    events[request_position] = events[request_position].model_copy(update={"payload": payload})
    first_edit = calculate_run_metrics(metrics_run.artifact_path).timing.time_to_first_edit_seconds  # type: ignore[attr-defined]

    tokens, context = _calculate_tokens_and_context(tuple(events), first_edit, True, [])

    assert tokens.input_tokens_total.value == 490
    assert context.context_used_per_request[0].context_used_tokens.value == 100
    assert context.context_used_per_request[0].context_max_tokens.value is None
    assert context.context_used_per_request[0].context_utilization_percent.value is None
    assert context.peak_context_utilization_percent.value is None


def test_reconstructed_tokens_require_tokenizer_identity(metrics_run: object) -> None:
    events = load_normalized_events(metrics_run.normalized_event_path)  # type: ignore[attr-defined]
    request = next(event for event in events if event.event_kind == "llm_request")
    missing_identity = request.model_copy(
        update={"payload": {**request.payload, "token_source": "tokenizer_reconstructed"}}
    )
    identified = request.model_copy(
        update={
            "payload": {
                **request.payload,
                "token_source": "tokenizer_reconstructed",
                "tokenizer_identity": "qwen-tokenizer-v1",
                "tokenizer_digest": "a" * 64,
            }
        }
    )

    assert _token_metric(missing_identity, "context_tokens").unavailable_reason == "invalid_source"
    available = _token_metric(identified, "context_tokens")
    assert available.value == 100
    assert available.provenance.method == "tokenizer_reconstructed"


def test_duplicate_tool_identity_invalidates_tool_totals(metrics_run: object) -> None:
    events = list(load_normalized_events(metrics_run.normalized_event_path))  # type: ignore[attr-defined]
    start = next(event for event in events if event.event_kind == "tool_call_start")
    duplicated = start.model_copy(update={"event_id": "duplicate-tool-event"})
    corrupted = tuple(events + [duplicated])
    diagnostics: list[str] = []
    tools = _correlate_tools(corrupted, diagnostics)

    behavior = _calculate_behavior(
        corrupted,
        tools,
        True,
        _tool_identity_valid(corrupted),
        diagnostics,
    )

    assert behavior.tool_calls_total.value is None
    assert behavior.tool_calls_total.unavailable_reason == "invalid_source"
    assert behavior.tool_calls_by_category is None


def test_precondition_and_preservation_classes_are_supported_with_precedence(metrics_run: object) -> None:
    normalized = load_normalized_events(metrics_run.normalized_event_path)  # type: ignore[attr-defined]
    raw = load_raw_events(metrics_run.raw_event_path)  # type: ignore[attr-defined]
    metrics = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]
    git_summary = type("Summary", (), {"paths": ("tracked.txt",)})()
    precondition = raw[0].model_copy(
        update={"event_type": "precondition_failed", "raw_event_id": "precondition"}
    )
    preservation = raw[0].model_copy(
        update={"event_type": "preservation_failed", "raw_event_id": "preservation"}
    )

    primary = _classify_termination(
        metrics_run.run_manifest,  # type: ignore[attr-defined]
        normalized,
        git_summary,  # type: ignore[arg-type]
        metrics.git_result,
        (preservation, precondition),
    )
    preserved_failure = _classify_termination(
        metrics_run.run_manifest,  # type: ignore[attr-defined]
        normalized,
        git_summary,  # type: ignore[arg-type]
        metrics.git_result,
        (preservation,),
    )

    assert primary.termination_class == "precondition_failed"
    assert preserved_failure.termination_class == "preservation_failed"
    assert preserved_failure.underlying_termination_class == "success"


def test_zero_denominator_formula_is_unavailable() -> None:
    metric = _divide_metrics(
        _available(10, "tokens", "deterministically_calculated"),
        0,
        "tokens/call",
    )

    assert metric.value is None
    assert metric.unavailable_reason == "zero_denominator"


def test_identical_inputs_produce_identical_metrics_bytes_and_sha(metrics_run: object) -> None:
    first = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]
    second = calculate_run_metrics(metrics_run.artifact_path)  # type: ignore[attr-defined]

    assert first.canonical_json_bytes() == second.canonical_json_bytes()
    assert hashlib.sha256(first.canonical_json_bytes()).hexdigest() == hashlib.sha256(second.canonical_json_bytes()).hexdigest()
    assert first.record_digest == second.record_digest
    assert "calculation_timestamp" not in first.model_dump(mode="json")
    assert RunMetrics.model_validate_json(first.canonical_json_bytes()) == first


def test_metrics_artifact_is_separate_immutable_linked_and_verified(
    metrics_run: object,
    tmp_path: Path,
) -> None:
    source = metrics_run.artifact_path  # type: ignore[attr-defined]
    source_manifest_before = (source / "manifest.json").read_bytes()
    metrics = calculate_run_metrics(source)
    stored = store_metrics_artifact(
        source_artifact=source,
        output_root=tmp_path / "analysis",
        metrics=metrics,
    )

    verified = verify_metrics_artifact(stored.root)
    assert verified.metrics == metrics
    assert verified.manifest.source_artifact_manifest_sha256 == hashlib.sha256(source_manifest_before).hexdigest()
    assert (source / "manifest.json").read_bytes() == source_manifest_before
    with pytest.raises(MetricsStorageError, match="already exists"):
        store_metrics_artifact(
            source_artifact=source,
            output_root=tmp_path / "analysis",
            metrics=metrics,
        )
    (stored.root / "metrics.json").write_bytes(b"{}\n")
    with pytest.raises(MetricsStorageError, match="invalid metrics artifact|checksum"):
        verify_metrics_artifact(stored.root)


def test_metrics_cli_calculate_and_show(metrics_run: object, tmp_path: Path) -> None:
    cli = CliRunner()
    output_root = tmp_path / "cli-analysis"
    calculate = cli.invoke(
        app,
        ["metrics", "calculate", str(metrics_run.artifact_path), str(output_root)],  # type: ignore[attr-defined]
    )
    assert calculate.exit_code == 0, calculate.output
    assert "termination=success" in calculate.output
    metrics_root = output_root / metrics_run.run_manifest.run_id / "metrics-v1"  # type: ignore[attr-defined]

    show = cli.invoke(app, ["metrics", "show", str(metrics_root)])
    assert show.exit_code == 0, show.output
    assert '"metric_spec_version": "1.0.0"' in show.output
    assert '"termination_class": "success"' in show.output
