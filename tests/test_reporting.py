"""Deterministic M9C reporting tests; no harness, model, or GPU is required."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from agent_bench.reporting import (
    SCHEMAS,
    _html_report,
    _append_curves,
    _append_summaries,
    _assert_public_safe,
    _build_database,
    _write_parquet,
    build_report,
    export_public,
    normalized_elapsed_curve,
    quantile_type7,
    verify_report,
    _ingest,
    _comparative_validity,
    concise_chart_series_labels,
    build_unified_report,
)
from agent_bench.executor import ExperimentState, RunProgress
from agent_bench.failure import FailureEnvironmentRecord, preserve_failed_run
from agent_bench.backend import BackendPreflightReport, PreflightCheck, ResolvedBackendInvocation
from agent_bench.capture import fixed_proxy_capture_capabilities


def _run(*, harness: str, task: str, variant: str, repetition: int, value: float, state: str = "completed", evidence: str = "verified") -> dict[str, object]:
    return {
        "experiment_id": "synthetic-v1", "run_id": f"{harness}-{task}-{variant}-{repetition}",
        "harness": harness, "semantic_task": task, "prompt_variant": variant,
        "repetition": repetition, "state": state, "evidence_status": evidence,
        "termination_class": "success" if state == "completed" else None,
        "wall_time_seconds": value, "input_tokens": int(value * 10), "output_tokens": int(value),
        "llm_requests": int(value), "tool_calls": int(value + 1),
        "first_task_context_tokens": int(value * 100), "peak_context_tokens": int(value * 120),
        "context_growth_from_first_task_tokens": int(value * 20), "files_changed": 1,
    }


def test_type7_quantiles_and_n1_are_deterministic() -> None:
    assert quantile_type7([1, 2, 3], .5) == 2
    assert quantile_type7([1, 2, 3], .25) == 1.5
    assert quantile_type7([9], .25) == quantile_type7([9], .75) == 9
    assert quantile_type7([], .5) is None


def test_normalized_curve_uses_task_relative_time_and_does_not_cross_unavailable_gap() -> None:
    points = [
        {"request_index": 1, "is_auxiliary": True, "elapsed_seconds": 1.0, "context_utilization_percent": 1.0},
        {"request_index": 2, "is_auxiliary": False, "elapsed_seconds": 5.0, "context_utilization_percent": 10.0},
        {"request_index": 3, "is_auxiliary": False, "elapsed_seconds": 7.0, "context_utilization_percent": None},
        {"request_index": 4, "is_auxiliary": False, "elapsed_seconds": 9.0, "context_utilization_percent": 30.0},
    ]
    curve = normalized_elapsed_curve(points, first_task_elapsed_seconds=5.0, wall_time_seconds=13.0, grid=(0, 25, 50, 75, 100))
    assert curve[0] == {"x": 0.0, "context_utilization_percent": 10.0, "value_method": "measured"}
    assert curve[1]["value_method"] == "unavailable_gap"
    assert curve[-1]["context_utilization_percent"] == 30.0


def test_concise_chart_labels_use_structured_metadata_and_known_reasoning_mapping() -> None:
    rows = [
        {"run_id": "opaque-xhigh-digest", "harness": "hermes", "harness_profile": "hermes-default-v1", "semantic_task": "entry-category", "prompt_variant": "normal", "repetition": 1},
        {"run_id": "opaque-medium-digest", "harness": "hermes", "harness_profile": "hermes-reasoning-medium-v1", "semantic_task": "entry-category", "prompt_variant": "normal", "repetition": 1},
    ]
    assert concise_chart_series_labels(rows) == {
        "opaque-medium-digest": "medium · entry-category · normal · R001",
        "opaque-xhigh-digest": "xhigh · entry-category · normal · R001",
    }


def test_chart_labels_prefix_multiple_harnesses_and_resolve_collisions_without_run_digest() -> None:
    rows = [
        {"run_id": "opaque-a", "harness": "hermes", "harness_profile": "hermes-default-v1", "semantic_task": "entry-category", "prompt_variant": "normal", "repetition": 1, "seed": 1001},
        {"run_id": "opaque-b", "harness": "hermes", "harness_profile": "hermes-default-v1", "semantic_task": "entry-category", "prompt_variant": "normal", "repetition": 1, "seed": 1002},
        {"run_id": "opaque-c", "harness": "opencode", "harness_profile": "opencode-default-v1", "semantic_task": "entry-category", "prompt_variant": "normal", "repetition": 1, "seed": 1001},
    ]
    labels = concise_chart_series_labels(rows)
    assert labels["opaque-a"] == "Hermes · xhigh · entry-category · normal · R001 · seed 1001"
    assert labels["opaque-b"].endswith("seed 1002")
    assert labels["opaque-c"] == "OpenCode · default · entry-category · normal · R001"
    assert all("opaque-" not in label for label in labels.values())


def test_synthetic_multi_harness_aggregation_and_parquet_duckdb(tmp_path: Path) -> None:
    runs = [
        _run(harness=harness, task=task, variant=variant, repetition=repetition, value=float(index + 1))
        for index, (harness, task, variant, repetition) in enumerate(
            (h, t, v, r)
            for h in ("opencode", "pi", "hermes")
            for t in ("entry-delete", "keyboard-entry")
            for v in ("vague", "normal", "precise")
            for r in (1, 2, 3)
        )
    ]
    runs.append(_run(harness="pi", task="entry-delete", variant="vague", repetition=4, value=99, state="failed", evidence="not_ingested"))
    summaries: list[dict[str, object]] = []
    _append_summaries("synthetic-v1", runs, summaries)
    harness_rows = [row for row in summaries if row["grouping"] == "harness" and row["group_key"] == "opencode" and row["metric_name"] == "wall_time_seconds"]
    assert harness_rows[0]["n_planned"] == 18
    assert harness_rows[0]["n_available"] == 18
    assert any(row["grouping"] == "harness_task_prompt_variant" for row in summaries)
    assert any(row["grouping"] == "repetition" for row in summaries)

    points = [
        {"experiment_id": "synthetic-v1", "run_id": run["run_id"], "request_index": 1, "is_auxiliary": False,
         "elapsed_seconds": 1.0, "task_elapsed_seconds": 0.0, "task_request_index": 1,
         "normalized_elapsed_task_percent": 0.0, "context_utilization_percent": 10.0,
         "context_tokens": 100, "delta_vs_previous_tokens": None, "delta_vs_first_task_tokens": 0,
         "availability": "available", "unavailable_reason": None}
        for run in runs[:3]
    ]
    for run in runs[:3]:
        run["wall_time_seconds"] = 10.0
    curves: list[dict[str, object]] = []
    _append_curves("synthetic-v1", runs[:3], points, curves)
    assert any(row["curve_kind"] == "normalized_elapsed_task_time_aggregate" for row in curves)

    rows = {name: [] for name in SCHEMAS}
    rows["runs"] = runs
    rows["summaries"] = summaries
    rows["context_points"] = points
    rows["curves"] = curves
    parquet = tmp_path / "parquet"
    _write_parquet(parquet, rows)
    assert pq.read_table(parquet / "runs.parquet").num_rows == len(runs)
    assert pq.read_table(parquet / "runs.parquet").schema == SCHEMAS["runs"]
    database = tmp_path / "agent-bench.duckdb"
    _build_database(database, parquet)
    connection = duckdb.connect(str(database), read_only=True)
    try:
        assert connection.execute("select count(*) from all_runs").fetchone() == (len(runs),)
        assert connection.execute("select count(*) from per_harness_metrics").fetchone()[0] > 0
    finally:
        connection.close()
    repeated = tmp_path / "repeated-parquet"
    _write_parquet(repeated, rows)
    assert hashlib.sha256((parquet / "runs.parquet").read_bytes()).hexdigest() == hashlib.sha256((repeated / "runs.parquet").read_bytes()).hexdigest()


def test_public_privacy_audit_rejects_secret_and_personal_path(tmp_path: Path) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text('{"raw_reasoning_included":false}\n', encoding="utf-8")
    _assert_public_safe(tmp_path)
    (tmp_path / "unsafe.json").write_text('{"authorization":"secret"}\n', encoding="utf-8")
    with pytest.raises(Exception, match="privacy"):
        _assert_public_safe(tmp_path)
    (tmp_path / "unsafe.json").unlink()
    pq.write_table(pa.Table.from_pylist([{"detail": "/home/alice/private"}]), tmp_path / "unsafe.parquet")
    with pytest.raises(Exception, match="privacy"):
        _assert_public_safe(tmp_path)


def test_dashboard_html_supports_a_full_matrix_without_claiming_variability() -> None:
    runs = [
        _run(harness=harness, task=task, variant=variant, repetition=repetition, value=1.0)
        for harness in ("opencode", "pi", "hermes")
        for task in ("task-a", "task-b", "task-c", "task-d", "task-e")
        for variant in ("vague", "normal", "precise")
        for repetition in (1, 2, 3)
    ]
    for index, item in enumerate(runs, start=1):
        item.update({"execution_index": index, "canonical_matrix_index": index, "harness_profile": f"{item['harness']}-default-v1", "prompt_id": f"{item['semantic_task']}-{item['prompt_variant']}", "seed": 1000 + index})
    presentation = {
        "generator": {"name": "agent-bench-report", "version": "test", "agent_bench_version": "test"},
        "experiment_id": "synthetic-v1", "definition_digest": "d" * 64, "expansion_digest": "e" * 64,
        "completion": {"total": 135, "completed": 135, "failed": 0, "interrupted": 0, "invalid": 0, "pending": 0, "is_partial": False},
        "definition": {"repetitions": 3, "harnesses": [{"harness_id": h, "display_name": h, "version": "1"} for h in ("opencode", "pi", "hermes")], "profiles": [], "prompts": [], "fixed_environment": {"model": {"name": "Qwen 3.8 27B"}}, "backend_configuration": {"server": {"context_size": 107520}}, "portable_baseline": {}},
        "summary_environment": {}, "runs": runs, "summaries": [], "curves": [], "markers": [], "failures": [], "details": {}, "data_files": [],
    }
    output = _html_report({"experiment_id": "synthetic-v1"}, presentation)
    assert "AGENT BENCH" in output
    assert "Run Explorer" in output
    assert "Pending / Planned" in output
    assert "individual only; no spread" in output
    assert "median + Q1–Q3" in output
    assert "Prompt variant" in output and "Repetition" in output
    assert "opencode-task-a-vague-1" in output
    assert "135" in output
    assert "N/A" not in output


def test_comparison_dashboard_emits_offline_numeric_sorting_for_current_group_rows() -> None:
    presentation = {
        "generator": {"name": "test", "version": "test", "agent_bench_version": "test"},
        "experiment_id": "synthetic-v1", "definition_digest": "d" * 64, "expansion_digest": "e" * 64,
        "completion": {"total": 3, "completed": 3, "failed": 0, "interrupted": 0, "invalid": 0, "pending": 0, "is_partial": False},
        "definition": {"repetitions": 1, "harnesses": [], "profiles": [], "prompts": [], "fixed_environment": {}, "backend_configuration": {}, "portable_baseline": {}},
        "summary_environment": {}, "runs": [], "curves": [], "markers": [], "failures": [], "details": {}, "data_files": [],
        "summaries": [
            {"grouping": "harness", "group_key": "a", "metric_name": "wall_time_seconds", "n_available": 2, "median": 10.5, "q1": -2.0, "q3": None, "minimum": 2, "maximum": 100},
        ],
    }
    output = _html_report({"experiment_id": "synthetic-v1"}, presentation)
    # Headers retain raw values and type metadata, rather than sorting formatted text.
    assert 'class="sortable-header"' in output
    assert 'data-sort-type="${type(c)}"' in output
    assert "numericColumns=new Set" in output
    assert "relative_delta_percent" in output and "absolute_delta" in output
    # The local comparator explicitly handles percentage, decimals, negative values,
    # and unavailable cells; unavailable always ranks after a real value.
    assert "replace(/%$/,'')" in output
    assert "Number.isFinite(parsed)" in output
    assert "if(am!==bm)return am?1:-1" in output
    assert "direction==='asc'?compared:-compared" in output
    # Only rows selected for the active filters/group are passed to the sortable table,
    # and its selected order is restored after a group view redraw.
    assert "comparisonSummaryRows(group)" in output
    assert "id:'comparison-summary'" in output
    assert "tableSorts[id]" in output and "applyTableSort(table,saved.key,saved.direction)" in output
    assert "data-sort-indicator" in output and "'↑':'↓'" in output
    assert "<script src=" not in output and "cdn" not in output.lower()


def test_group_comparison_bars_have_local_numeric_sorting_after_filters() -> None:
    presentation = {
        "generator": {"name": "test", "version": "test", "agent_bench_version": "test"},
        "experiment_id": "synthetic-v1", "definition_digest": "d" * 64, "expansion_digest": "e" * 64,
        "completion": {"total": 3, "completed": 3, "failed": 0, "interrupted": 0, "invalid": 0, "pending": 0, "is_partial": False},
        "definition": {"repetitions": 1, "harnesses": [], "profiles": [], "prompts": [], "fixed_environment": {}, "backend_configuration": {}, "portable_baseline": {}},
        "summary_environment": {}, "curves": [], "markers": [], "failures": [], "details": {}, "data_files": [], "summaries": [],
        "runs": [
            {**_run(harness="hermes", task="entry-category", variant="normal", repetition=1, value=10.0), "harness_profile": "hermes-default-v1", "seed": 1001},
            {**_run(harness="hermes", task="entry-category", variant="normal", repetition=2, value=2.0), "harness_profile": "hermes-reasoning-low-v1", "seed": 1002},
            {**_run(harness="hermes", task="entry-category", variant="normal", repetition=3, value=100.0), "harness_profile": "hermes-reasoning-medium-v1", "seed": 1003},
        ],
    }
    output = _html_report({"experiment_id": "synthetic-v1"}, presentation)
    assert 'data-bar-sort="${esc(id)}"' in output
    assert "Original</option>" in output and "Ascending</option>" in output and "Descending</option>" in output
    # Central aggregate (median) drives the category order, with numeric—not
    # lexical—ordering. Missing values are explicitly final in either order.
    assert "function orderBarRows(rows,order)" in output
    assert "const copy=[...rows]" in output
    assert "if(am!==bm)return am?1:-1" in output
    assert "order==='ascending'?a-b:b-a" in output
    assert 'data-bar-category="${esc(r.group_key)}"' in output
    # The chart summaries are rebuilt from currently filtered completed runs;
    # sorting never mutates the immutable d.runs input.
    assert "completed().filter(matchesFilters)" in output
    assert "barSorts[control.dataset.barSort]=control.value;comparison()" in output
    assert "<script src=" not in output and "cdn" not in output.lower()


def test_variant_comparison_is_metric_selectable_and_uses_only_matched_deterministic_observations() -> None:
    """The ranking view is a metric view, never an efficiency or quality score."""
    def raw_run(profile: str, value: float | None, *, seed: int = 1001) -> dict[str, object]:
        return {
            "experiment_id": f"screen-{profile}", "run_id": f"opaque-{profile}-{seed}",
            "harness": "hermes", "profile": profile,
            "reasoning_setting": {"hermes-default-v1": "xhigh", "hermes-reasoning-low-v1": "low"}[profile],
            "semantic_task": "entry-category", "prompt_variant": "normal", "prompt_sha256": "p" * 64,
            "repetition": 1, "seed": seed,
            "metrics": {
                "timing.wall_time_seconds.value": value,
                "tokens.output_tokens_total.value": 100 if value is not None else None,
                "behavior.requests_before_first_model_tool_call.value": 2,
            },
        }

    presentation = {
        "generator": {"name": "test", "version": "test", "agent_bench_version": "test"},
        "experiment_id": "synthetic-v1", "definition_digest": "d" * 64, "expansion_digest": "e" * 64,
        "completion": {"total": 0, "completed": 0, "failed": 0, "interrupted": 0, "invalid": 0, "pending": 0, "is_partial": False},
        "definition": {"repetitions": 1, "harnesses": [], "profiles": [], "prompts": [], "fixed_environment": {}, "backend_configuration": {}, "portable_baseline": {"subject_id": "pocket-ledger"}},
        "summary_environment": {}, "runs": [], "summaries": [], "curves": [], "markers": [], "failures": [], "details": {}, "data_files": [],
        "matched_comparison": {"raw_runs": [
            raw_run("hermes-default-v1", 10.0), raw_run("hermes-reasoning-low-v1", 8.0),
            raw_run("hermes-default-v1", 12.0, seed=1002), raw_run("hermes-reasoning-low-v1", None, seed=1002),
        ]},
    }
    output = _html_report({"experiment_id": "synthetic-v1"}, presentation)
    assert "Variant Comparison" in output and "not an overall efficiency score or quality ranking" in output
    assert "variantMetricLabels" in output
    assert "Wall time" in output and "Output tokens" in output
    assert "Requests before first model tool" in output
    # The selector is populated from finite values in raw immutable report data;
    # no unavailable or inferred reasoning time/token metric is introduced.
    assert "function variantMetricKeys(rows)" in output
    assert "Object.entries(row.metrics||{})" in output
    assert "reasoning_time_seconds" not in output
    # Reconstructed reasoning-token fields are allowed only when raw metric
    # data contains a finite value; the static label alone does not select it.
    assert "reasoning.reasoning_tokens_total.value" in output
    # Category identity includes harness, profile, and effective setting, while
    # matching requires subject/task/exact prompt hash/repetition/seed.
    assert "function variantDescriptors(rows)" in output
    assert "${titleCase(row.harness)} · ${profileLabel(row.profile)}" in output
    assert "row.prompt_sha256,row.repetition,row.seed" in output
    assert "function variantMatchable(row)" in output
    assert "rows without task, exact prompt SHA-256, repetition, or seed provenance are explicitly excluded" in output
    assert "matched by subject, task, exact prompt SHA-256, repetition, and seed" in output
    assert "[variantSubject(row),row.semantic_task,row.prompt_sha256,row.repetition,row.seed]" in output
    assert "for(const variants of cases.values())" in output
    # Median/Q1/Q3 are calculated only from matched values. Missing metrics are
    # explicit, and both sort directions remain numeric with N/A last.
    assert "median:type7(observed,.5)" in output
    assert "q1:observed.length>1?type7(observed,.25):null" in output
    assert "n_unavailable:total-observed.length" in output
    assert "function orderVariantRows(rows,order)" in output
    assert "order==='lowest'?a-b:b-a" in output
    assert "if(am!==bm)return am?1:-1" in output
    # Delta values are candidate-reference and percentages have a defined zero
    # reference edge case. Rendering works from copied summaries, never sorting
    # the sealed d.matched_comparison.raw_runs source.
    assert "candidate-referenceValue" in output
    assert "referenceValue===0?null" in output
    assert "const copy=[...rows]" in output
    assert "Prompt = All aggregates matched observations within each exact prompt SHA and seed" in output
    assert "<script src=" not in output and "cdn" not in output.lower()


def test_unified_report_combines_verified_roots_into_one_sealed_rich_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The combine path reuses the full M9C artifact layout, not a mini report."""
    from agent_bench import comparison, reporting

    roots = [tmp_path / "old", tmp_path / "screen"]
    for root in roots:
        root.mkdir()
    states = iter([
        ExperimentState(experiment_id="old", definition_digest="a" * 64, expansion_digest="b" * 64, ordering={}, runs=[], updated_at="2026-01-01T00:00:00Z"),
        ExperimentState(experiment_id="screen", definition_digest="c" * 64, expansion_digest="d" * 64, ordering={}, runs=[], updated_at="2026-01-01T00:00:00Z"),
    ])
    identity = {"subject_baseline": "subject", "model": "model", "backend": "backend", "chat_template": "template", "hardware": "hardware", "context_backend_settings": "settings"}
    comparison_inputs = iter([
        {"experiment_id": "old", "root": str(roots[0]), "definition_digest": "a" * 64, "completed_runs": 0, "partial": False, "identity": identity, "rows": []},
        {"experiment_id": "screen", "root": str(roots[1]), "definition_digest": "c" * 64, "completed_runs": 0, "partial": True, "identity": identity, "rows": []},
    ])
    presentation_definition = {
        "definition_available": True, "definition_digest": "fixture", "prompts": [], "profiles": [], "harnesses": [],
        "fixed_environment": {}, "backend_configuration": {}, "portable_baseline": {}, "repetition_indices": [],
    }
    monkeypatch.setattr(reporting, "_load_state", lambda _root: next(states))
    monkeypatch.setattr(reporting, "_load_definition", lambda *_args: ({"fixture": object()}, {"fixed_environment_id": "fixture", "model": "model", "model_sha256": "sha", "backend": "backend", "backend_commit": "commit", "hardware_name": "hardware", "gpu_model": "gpu"}, presentation_definition))
    monkeypatch.setattr(reporting, "_ingest", lambda *_args: ({name: [] for name in SCHEMAS}, []))
    monkeypatch.setattr(comparison, "_read_root", lambda *_args: next(comparison_inputs))

    report = build_unified_report(roots, output=tmp_path / "unified")
    manifest = verify_report(report.root)
    assert manifest["included_run_ids"] == []
    assert (report.root / "report.html").is_file()
    assert (report.root / "comparison.json").is_file()
    assert (report.root / "parquet" / "runs.parquet").is_file()
    presentation = json.loads((report.root / "presentation.json").read_text(encoding="utf-8"))
    assert [item["experiment_id"] for item in presentation["source_experiments"]] == ["old", "screen"]
    assert "Matched seed / paired profile effects" in (report.root / "report.html").read_text(encoding="utf-8")


def test_chart_html_uses_concise_legend_labels_and_offline_series_interaction() -> None:
    run_id = "hermes-hermes-default-v1-entry-category-normal-r001-347450c68e34f414e5c905b6"
    run = _run(harness="hermes", task="entry-category", variant="normal", repetition=1, value=1.0)
    run.update({"run_id": run_id, "harness_profile": "hermes-default-v1", "execution_index": 1, "seed": 1001})
    presentation = {
        "generator": {"name": "test", "version": "test", "agent_bench_version": "test"},
        "experiment_id": "synthetic-v1", "definition_digest": "d" * 64, "expansion_digest": "e" * 64,
        "completion": {"total": 1, "completed": 1, "failed": 0, "interrupted": 0, "invalid": 0, "pending": 0, "is_partial": False},
        "definition": {"repetitions": 1, "harnesses": [], "profiles": [], "prompts": [], "fixed_environment": {}, "backend_configuration": {}, "portable_baseline": {}},
        "summary_environment": {}, "runs": [run], "summaries": [],
        "curves": [{"curve_kind": "absolute_elapsed_task_time", "run_id": run_id, "x": 1.0, "context_utilization_percent": 10.0}],
        "markers": [], "failures": [], "details": {}, "data_files": [],
        "chart_series_labels": concise_chart_series_labels([run]),
    }
    output = _html_report({"experiment_id": "synthetic-v1"}, presentation)
    assert "xhigh · entry-category · normal · R001" in output
    assert f">{run_id}</button>" not in output
    assert "data-series-id" in output and "data-run-id" in output and "aria-controls" in output
    assert 'title="${esc(id)} — full run identity"' in output and "series-hit" in output
    assert "wireChartInteractions" in output and "pointerover" in output and "Escape" in output
    assert "<script src=" not in output and "cdn" not in output.lower()


def test_failed_run_evidence_is_reported_as_verified_infrastructure_record(tmp_path: Path) -> None:
    run_id = "hermes-fixture-normal-r001"
    environment = FailureEnvironmentRecord(
        run_id=run_id, backend_profile_digest="a" * 64,
        preflight=BackendPreflightReport(profile_id="fixture", passed=False,
            primary_failure_class="benchmark_port_in_use", checks=(PreflightCheck(
                check_id="benchmark-port", passed=False,
                failure_class="benchmark_port_in_use", message="active listener",
                evidence={"port": 18080}),)),
        invocation=ResolvedBackendInvocation(profile_id="fixture", run_seed=1001,
            executable=Path("/opt/llama-server"), argv=("/opt/llama-server",),
            working_directory=Path("/opt"), environment={},
            stdout_artifact="failure/stdout.log", stderr_artifact="failure/stderr.log"),
        capture_capabilities=fixed_proxy_capture_capabilities(),
    )
    preserve_failed_run(runs_root=tmp_path / "runs", run_id=run_id,
        failure_class="benchmark_port_in_use", reason="active listener", environment=environment)
    state = ExperimentState(experiment_id="fixture", definition_digest="b" * 64,
        expansion_digest="c" * 64, ordering={}, runs=[RunProgress(run_id=run_id,
        execution_index=1, state="failed", failure_domain="infrastructure_precondition",
        failure_class="benchmark_port_in_use", failure_phase="preflight",
        harness_execution_started=False, llm_request_observed=False,
        preservation_completed=True)], updated_at="2026-09-04T00:00:00Z")
    rows, excluded = _ingest(tmp_path, state, {}, {})
    assert excluded == []
    assert rows["runs"][0]["evidence_status"] == "verified_failed_run_evidence"
    assert rows["runs"][0]["termination_class"] == "benchmark_port_in_use"
    assert rows["failures"][0]["failure_phase"] == "preflight"
    assert rows["failures"][0]["harness_execution_started"] is False


def test_comparative_validity_distinguishes_complete_partial_and_infrastructure() -> None:
    def state(states: list[str]) -> ExperimentState:
        return ExperimentState(experiment_id="fixture", definition_digest="a" * 64, expansion_digest="b" * 64,
            ordering={}, runs=[RunProgress(run_id=f"run-{i}", execution_index=i, state=value) for i, value in enumerate(states, 1)], updated_at="2026-09-05T00:00:00Z")
    clean = state(["completed"] * 135)
    assert _comparative_validity(clean, [{"state": "completed", "evidence_status": "verified"} for _ in clean.runs]) == "complete_valid_for_comparative_interpretation"
    partial = state(["completed", "pending"])
    assert _comparative_validity(partial, [{"state": "completed", "evidence_status": "verified"}, {"state": "pending", "evidence_status": "not_executed"}]) == "partial_but_otherwise_healthy"
    ordinary = state(["completed", "failed"])
    assert _comparative_validity(ordinary, [{"state": "completed", "evidence_status": "verified"}, {"state": "failed", "evidence_status": "verified_failed_run_evidence", "termination_class": "harness_crash"}]) == "complete_with_ordinary_run_failures"
    broken = state(["completed", "failed"])
    assert _comparative_validity(broken, [{"state": "completed", "evidence_status": "verified"}, {"state": "failed", "evidence_status": "verified_failed_run_evidence", "termination_class": "benchmark_port_in_use"}]) == "invalid_for_comparative_interpretation"


def test_report_status_explicitly_identifies_selected_report_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from agent_bench import reporting
    state = ExperimentState(experiment_id="fixture", definition_digest="a" * 64, expansion_digest="b" * 64,
                            ordering={}, runs=[], updated_at="2026-09-05T00:00:00Z")
    monkeypatch.setattr(reporting, "_load_state", lambda _root: state)
    (tmp_path / "report-v1").mkdir()
    (tmp_path / "report-v2").mkdir()
    default = reporting.report_status(tmp_path)
    assert default["selected_report_source"] == "default report-v1"
    assert default["available_report_roots"] == ["report-v1", "report-v2"]
    explicit = reporting.report_status(tmp_path, report_root=tmp_path / "report-v2")
    assert explicit["selected_report_source"] == "explicit --report-root"
    assert explicit["selected_report_root"].endswith("report-v2")


def test_real_smoke_is_reportable_read_only_when_available(tmp_path: Path) -> None:
    """A checkout without local smoke evidence still has fully synthetic coverage."""
    source = Path("runs/m9b-real-smoke-v3")
    if not source.is_dir():
        pytest.skip("authoritative host smoke evidence is not present in this checkout")
    report = build_report(source, output=tmp_path / "report-v1", experiment_definition=Path("experiments/pocket-ledger-v1.yaml"))
    assert verify_report(report.root)["experiment_id"] == "pocket-ledger-v1-qwen38"
    summary = json.loads((report.root / "summary.json").read_text(encoding="utf-8"))
    assert summary["completion"]["total"] == 135
    assert summary["completion"]["completed"] == 1
    assert summary["completion"]["is_partial"] is True
    dashboard = (report.root / "report.html").read_text(encoding="utf-8")
    assert "PARTIAL EXPERIMENT" in dashboard
    assert "hermes-hermes-default-v1-keyboard-entry-vague-r001" in dashboard
    assert "Timing provenance" in dashboard
    assert "<host-path-redacted>" in dashboard
    assert "/home/bking/" not in dashboard
    public = export_public(report.root, tmp_path / "public")
    assert verify_report(public)["experiment_id"] == "pocket-ledger-v1-qwen38"
