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
)


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
