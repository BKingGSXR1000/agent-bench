"""Deterministic, derived-only M9C reporting over sealed experiment evidence.

This module deliberately consumes the small immutable analysis layers rather
than raw proxy bodies.  It never starts a harness, opens a worktree, or mutates
an experiment result.  A report is a new, versioned, rebuildable product.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from agent_bench import __version__
from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.context_storage import verify_context_analysis_artifact
from agent_bench.events import load_normalized_events
from agent_bench.executor import ExperimentState
from agent_bench.matrix import expand_experiment
from agent_bench.metrics_storage import verify_metrics_artifact
from agent_bench.models import RunDefinition, canonical_sha256
from agent_bench.preservation import verify_artifact
from agent_bench.result_store import verify_published_result
from agent_bench.timing_provenance_storage import verify_timing_provenance_artifact

REPORT_SCHEMA_VERSION = "1.0.0"
REPORT_GENERATOR = "agent-bench-report"
REPORT_GENERATOR_VERSION = "1.0.1"
REPORT_DIRECTORY = "report-v1"
PARQUET_DIRECTORY = "parquet"
DATABASE_NAME = "agent-bench.duckdb"
MANIFEST_NAME = "report-manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
SUMMARY_NAME = "summary.json"
ARCHIVAL_MANIFEST_NAME = "raw-archival-manifest.json"
HTML_NAME = "report.html"


class ReportError(RuntimeError):
    """Reporting evidence or derived output is unsafe to use."""


@dataclass(frozen=True)
class ReportBuild:
    root: Path
    manifest: dict[str, Any]


def _schema(*fields: tuple[str, pa.DataType]) -> pa.Schema:
    return pa.schema([pa.field(name, kind, nullable=True) for name, kind in fields])


# Schema and column order are an explicit report-schema-v1 contract.
SCHEMAS: dict[str, pa.Schema] = {
    "experiments": _schema(
        ("experiment_id", pa.string()), ("definition_digest", pa.string()),
        ("expansion_digest", pa.string()), ("identity_version", pa.string()),
        ("planned_runs", pa.int64()), ("completed_runs", pa.int64()),
        ("failed_runs", pa.int64()), ("interrupted_runs", pa.int64()),
        ("invalid_runs", pa.int64()), ("pending_runs", pa.int64()),
        ("is_partial", pa.bool_()), ("report_schema_version", pa.string()),
        ("fixed_environment_id", pa.string()), ("model_sha256", pa.string()),
        ("backend_commit", pa.string()), ("hardware_name", pa.string()),
        ("gpu_model", pa.string()),
    ),
    "runs": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("execution_index", pa.int64()), ("canonical_matrix_index", pa.int64()),
        ("harness", pa.string()), ("harness_profile", pa.string()),
        ("semantic_task", pa.string()), ("prompt_id", pa.string()),
        ("prompt_variant", pa.string()), ("repetition", pa.int64()),
        ("seed", pa.int64()), ("state", pa.string()),
        ("evidence_status", pa.string()), ("evidence_reason", pa.string()),
        ("termination_class", pa.string()), ("wall_time_seconds", pa.float64()),
        ("llm_requests", pa.int64()), ("tool_calls", pa.int64()),
        ("input_tokens", pa.int64()), ("output_tokens", pa.int64()),
        ("total_tokens", pa.int64()), ("first_task_context_tokens", pa.int64()),
        ("first_task_context_utilization_percent", pa.float64()),
        ("peak_context_tokens", pa.int64()),
        ("peak_context_utilization_percent", pa.float64()),
        ("context_growth_from_first_task_tokens", pa.int64()),
        ("files_changed", pa.int64()), ("lines_added", pa.int64()),
        ("lines_deleted", pa.int64()),
    ),
    "metrics": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("metric_group", pa.string()), ("metric_name", pa.string()),
        ("value", pa.float64()), ("units", pa.string()),
        ("availability", pa.string()), ("unavailable_reason", pa.string()),
        ("method", pa.string()),
    ),
    "requests": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("request_index", pa.int64()), ("captured_http_request_index", pa.int64()),
        ("purpose", pa.string()), ("purpose_evidence", pa.string()),
        ("elapsed_seconds", pa.float64()), ("request_body_sha256", pa.string()),
        ("messages_sha256", pa.string()), ("tool_schema_sha256", pa.string()),
        ("input_context_tokens", pa.int64()), ("input_tokens_availability", pa.string()),
        ("output_tokens", pa.int64()), ("output_tokens_availability", pa.string()),
        ("configured_max_context_tokens", pa.int64()),
        ("context_utilization_percent", pa.float64()),
        ("delta_vs_previous_inference_tokens", pa.int64()),
        ("delta_vs_first_task_tokens", pa.int64()),
    ),
    "context_points": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("request_index", pa.int64()), ("task_request_index", pa.int64()),
        ("is_auxiliary", pa.bool_()), ("elapsed_seconds", pa.float64()),
        ("task_elapsed_seconds", pa.float64()),
        ("normalized_elapsed_task_percent", pa.float64()),
        ("context_tokens", pa.int64()), ("context_utilization_percent", pa.float64()),
        ("delta_vs_previous_tokens", pa.int64()), ("delta_vs_first_task_tokens", pa.int64()),
        ("availability", pa.string()), ("unavailable_reason", pa.string()),
    ),
    "tools": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("event_id", pa.string()), ("event_kind", pa.string()),
        ("tool_name", pa.string()), ("category", pa.string()),
        ("outcome", pa.string()), ("elapsed_seconds", pa.float64()),
        ("timing_semantics", pa.string()),
    ),
    "markers": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("marker_kind", pa.string()), ("elapsed_seconds", pa.float64()),
        ("task_elapsed_seconds", pa.float64()), ("timing_semantics", pa.string()),
        ("source_event_id", pa.string()),
    ),
    "timing": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("timing_name", pa.string()), ("value_seconds", pa.float64()),
        ("availability", pa.string()), ("unavailable_reason", pa.string()),
        ("semantics", pa.string()), ("method", pa.string()),
        ("source", pa.string()),
    ),
    "failures": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("state", pa.string()), ("termination_class", pa.string()),
        ("detail", pa.string()), ("evidence_status", pa.string()),
    ),
    "artifacts": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("artifact_manifest_id", pa.string()), ("artifact_manifest_sha256", pa.string()),
        ("metrics_manifest_sha256", pa.string()), ("context_manifest_sha256", pa.string()),
        ("timing_manifest_sha256", pa.string()), ("result_commit", pa.string()),
        ("result_ref", pa.string()), ("source_snapshot_sha256", pa.string()),
        ("capture_capabilities_sha256", pa.string()), ("artifact_relative_path", pa.string()),
    ),
    "git_change_metrics": _schema(
        ("experiment_id", pa.string()), ("run_id", pa.string()),
        ("metric_name", pa.string()), ("value", pa.float64()),
        ("availability", pa.string()), ("unavailable_reason", pa.string()),
    ),
    "curves": _schema(
        ("experiment_id", pa.string()), ("curve_kind", pa.string()),
        ("grouping", pa.string()), ("group_key", pa.string()), ("run_id", pa.string()),
        ("x", pa.float64()), ("context_utilization_percent", pa.float64()),
        ("value_method", pa.string()), ("n_available", pa.int64()),
        ("median", pa.float64()), ("q1", pa.float64()), ("q3", pa.float64()),
        ("termination_class", pa.string()),
    ),
    "summaries": _schema(
        ("experiment_id", pa.string()), ("grouping", pa.string()),
        ("group_key", pa.string()), ("metric_name", pa.string()),
        ("n_planned", pa.int64()), ("n_completed", pa.int64()),
        ("n_successful", pa.int64()), ("n_failed_or_invalid", pa.int64()),
        ("n_available", pa.int64()), ("median", pa.float64()),
        ("q1", pa.float64()), ("q3", pa.float64()), ("iqr", pa.float64()),
        ("minimum", pa.float64()), ("maximum", pa.float64()),
    ),
}


def build_report(
    experiment_output: Path,
    *,
    output: Path | None = None,
    experiment_definition: Path | None = None,
) -> ReportBuild:
    """Build one non-overwriting report from immutable experiment evidence."""
    source = experiment_output.expanduser().resolve()
    state = _load_state(source)
    definition, environment = _load_definition(state, experiment_definition)
    target = (output or source / REPORT_DIRECTORY).expanduser().resolve()
    if target.exists():
        raise ReportError(f"report destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".report-v1.incomplete-", dir=target.parent))
    try:
        rows, excluded = _ingest(source, state, definition, environment)
        _write_parquet(staging / PARQUET_DIRECTORY, rows)
        _build_database(staging / DATABASE_NAME, staging / PARQUET_DIRECTORY)
        summary = _summary(state, rows["runs"], rows["summaries"], excluded, environment)
        _write_json(staging / SUMMARY_NAME, summary)
        _write_json(staging / ARCHIVAL_MANIFEST_NAME, _archival_manifest(state, rows["artifacts"]))
        _write_json(staging / "charts.json", {"schema_version": REPORT_SCHEMA_VERSION, "curves": rows["curves"]})
        (staging / HTML_NAME).write_text(_html_report(summary, rows), encoding="utf-8", newline="\n")
        manifest = _seal_report(staging, state, rows, excluded)
        verify_report(staging)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ReportBuild(root=target, manifest=manifest)


def verify_report(report_root: Path) -> dict[str, Any]:
    """Verify a derived report's own manifest and checksum inventory."""
    root = report_root.expanduser().resolve()
    try:
        manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
        checksums = _read_checksums(root / CHECKSUMS_NAME)
    except Exception as exc:
        raise ReportError(f"invalid report artifact: {exc}") from exc
    if manifest.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ReportError("unsupported report schema version")
    content = {key: value for key, value in manifest.items() if key != "record_digest"}
    if manifest.get("record_digest") != canonical_sha256(content):
        raise ReportError("report manifest digest mismatch")
    if set(checksums) != set(manifest.get("files", {})) | {MANIFEST_NAME}:
        raise ReportError("report checksum inventory does not match manifest")
    for relative, expected in checksums.items():
        path = root / relative
        if not path.is_file() or _sha(path) != expected:
            raise ReportError(f"report checksum mismatch: {relative}")
    return manifest


def report_status(experiment_output: Path) -> dict[str, Any]:
    """Return deterministic completion status and optional report verification."""
    root = experiment_output.expanduser().resolve()
    state = _load_state(root)
    result = _state_counts(state)
    report = root / REPORT_DIRECTORY
    result["report"] = "missing"
    if report.exists():
        try:
            result["report"] = "valid"
            result["report_manifest"] = verify_report(report)
        except ReportError as exc:
            result["report"] = "invalid"
            result["report_error"] = str(exc)
    return result


def export_public(report_root: Path, output: Path) -> Path:
    """Copy the small sanitized report surface into a non-overwriting export."""
    source = report_root.expanduser().resolve()
    verify_report(source)
    target = output.expanduser().resolve()
    if target.exists():
        raise ReportError(f"publication destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".public-report.incomplete-", dir=target.parent))
    try:
        for name in (HTML_NAME, MANIFEST_NAME, CHECKSUMS_NAME, SUMMARY_NAME, ARCHIVAL_MANIFEST_NAME):
            shutil.copy2(source / name, staging / name)
        shutil.copytree(source / PARQUET_DIRECTORY, staging / PARQUET_DIRECTORY)
        # Database files are useful but are not necessary for a GitHub viewer.
        shutil.copy2(source / DATABASE_NAME, staging / DATABASE_NAME)
        _assert_public_safe(staging)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def quantile_type7(values: Iterable[float], probability: float) -> float | None:
    """Hyndman-Fan type 7 quantile, rounded only by serialization callers."""
    ordered = sorted(Decimal(str(value)) for value in values)
    if not ordered:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    if len(ordered) == 1:
        return float(ordered[0])
    h = Decimal(len(ordered) - 1) * Decimal(str(probability))
    lower = int(h)
    fraction = h - lower
    result = ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])
    return float(result)


def normalized_elapsed_curve(
    points: list[dict[str, Any]], *, first_task_elapsed_seconds: float | None,
    wall_time_seconds: float | None, grid: Iterable[int] = range(101),
) -> list[dict[str, Any]]:
    """Resample task-relative context with REPORTING.md's labelled carries.

    M9C defines task-time zero as the first real task inference request.  Thus
    title/auxiliary inference is excluded from the x-axis but retained in the
    separate request and overhead datasets.
    """
    if first_task_elapsed_seconds is None or wall_time_seconds is None:
        return []
    duration = Decimal(str(wall_time_seconds)) - Decimal(str(first_task_elapsed_seconds))
    if duration <= 0:
        return []
    prepared: list[tuple[Decimal, Decimal | None, int]] = []
    for point in points:
        value, elapsed = point.get("context_utilization_percent"), point.get("elapsed_seconds")
        if elapsed is None or point.get("is_auxiliary"):
            continue
        progress = (Decimal(str(elapsed)) - Decimal(str(first_task_elapsed_seconds))) * Decimal("100") / duration
        prepared.append((progress, Decimal(str(value)) if value is not None else None, int(point["request_index"])))
    if not prepared:
        return []
    prepared.sort(key=lambda item: (item[0], item[2]))
    # At a shared coordinate, the later request is the prescribed retained point.
    unique: list[tuple[Decimal, Decimal | None, int]] = []
    for point in prepared:
        if unique and unique[-1][0] == point[0]:
            unique[-1] = point
        else:
            unique.append(point)
    available_indices = [index for index, point in enumerate(unique) if point[1] is not None]
    if not available_indices:
        return []
    first_available, last_available = available_indices[0], available_indices[-1]
    result: list[dict[str, Any]] = []
    for integer in grid:
        x = Decimal(integer)
        if x <= unique[first_available][0]:
            value, method = unique[first_available][1], "boundary_carried"
            if x == unique[first_available][0]: method = "measured"
        elif x >= unique[last_available][0]:
            value, method = unique[last_available][1], "boundary_carried"
            if x == unique[last_available][0]: method = "measured"
        else:
            left, right = next((a, b) for a, b in zip(unique, unique[1:]) if a[0] <= x <= b[0])
            if left[1] is None or right[1] is None:
                result.append({"x": float(x), "context_utilization_percent": None, "value_method": "unavailable_gap"})
                continue
            if x == left[0]: value, method = left[1], "measured"
            elif x == right[0]: value, method = right[1], "measured"
            else:
                value = left[1] + (right[1] - left[1]) * (x - left[0]) / (right[0] - left[0])
                method = "linear_interpolated"
        assert value is not None
        result.append({"x": float(x), "context_utilization_percent": _round6(value), "value_method": method})
    return result


def _load_state(root: Path) -> ExperimentState:
    try:
        return ExperimentState.model_validate_json((root / "experiment-state.json").read_bytes())
    except Exception as exc:
        raise ReportError(f"invalid experiment state: {exc}") from exc


def _load_definition(state: ExperimentState, explicit: Path | None) -> tuple[dict[str, RunDefinition], dict[str, str | None]]:
    candidate = explicit
    if candidate is None:
        conventional = Path.cwd() / "experiments" / f"{state.experiment_id}.yaml"
        candidate = conventional if conventional.is_file() else None
    if candidate is None:
        return {}, {}
    try:
        definition = load_experiment(candidate)
    except ExperimentConfigError as exc:
        raise ReportError(f"invalid supplied experiment definition: {exc}") from exc
    if definition.experiment_id != state.experiment_id or definition.definition_digest != state.definition_digest:
        raise ReportError("experiment definition does not match immutable experiment state")
    environment = {
        "fixed_environment_id": definition.fixed_environment.fixed_environment_id,
        "model": definition.fixed_environment.model.name,
        "model_sha256": definition.fixed_environment.model.sha256,
        "backend": definition.fixed_environment.backend.implementation,
        "backend_commit": definition.fixed_environment.backend.commit,
        "hardware_name": definition.fixed_environment.hardware.name,
        "gpu_model": definition.fixed_environment.hardware.gpu_model,
    }
    return {run.run_id: run for run in expand_experiment(definition)}, environment


def _ingest(source: Path, state: ExperimentState, definitions: dict[str, RunDefinition], environment: dict[str, str | None]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    rows = {name: [] for name in SCHEMAS}
    excluded: list[dict[str, str]] = []
    counts = _state_counts(state)
    rows["experiments"].append({
        "experiment_id": state.experiment_id, "definition_digest": state.definition_digest,
        "expansion_digest": state.expansion_digest,
        "identity_version": (definitions[next(iter(definitions))].identity_version if definitions else None),
        "planned_runs": counts["total"], "completed_runs": counts["completed"],
        "failed_runs": counts["failed"], "interrupted_runs": counts["interrupted"],
        "invalid_runs": counts["invalid"], "pending_runs": counts["pending"],
        "is_partial": counts["completed"] != counts["total"], "report_schema_version": REPORT_SCHEMA_VERSION,
        "fixed_environment_id": environment.get("fixed_environment_id"), "model_sha256": environment.get("model_sha256"),
        "backend_commit": environment.get("backend_commit"), "hardware_name": environment.get("hardware_name"), "gpu_model": environment.get("gpu_model"),
    })
    for progress in sorted(state.runs, key=lambda item: (item.execution_index, item.run_id)):
        definition = definitions.get(progress.run_id)
        identity = _identity(progress.run_id, definition)
        base = {"experiment_id": state.experiment_id, "run_id": progress.run_id, "execution_index": progress.execution_index, **identity}
        if progress.state != "completed":
            rows["runs"].append({**base, "state": progress.state, "evidence_status": "not_ingested", "evidence_reason": progress.detail, **_empty_run_metrics()})
            rows["failures"].append({**base, "state": progress.state, "termination_class": None, "detail": progress.detail, "evidence_status": "not_ingested"})
            continue
        try:
            _ingest_completed(source, state.experiment_id, progress.run_id, base, rows)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            excluded.append({"run_id": progress.run_id, "reason": reason})
            rows["runs"].append({**base, "state": "completed", "evidence_status": "invalid", "evidence_reason": reason, **_empty_run_metrics()})
            rows["failures"].append({**base, "state": "completed", "termination_class": None, "detail": reason, "evidence_status": "invalid"})
    _append_summaries(state.experiment_id, rows["runs"], rows["summaries"])
    _append_curves(state.experiment_id, rows["runs"], rows["context_points"], rows["curves"])
    return rows, excluded


def _identity(run_id: str, definition: RunDefinition | None) -> dict[str, Any]:
    if definition is not None:
        prompt_variant = definition.prompt_id.rsplit("-", 1)[-1]
        return {"canonical_matrix_index": definition.matrix_index, "harness": definition.harness_id,
                "harness_profile": definition.profile_id, "semantic_task": definition.semantic_task_id,
                "prompt_id": definition.prompt_id, "prompt_variant": prompt_variant,
                "repetition": definition.repetition_index, "seed": definition.generation_seed}
    # Existing M9B evidence does not duplicate its definition; this conservative
    # fallback only decodes the checked-in prompt-id naming convention.
    stem = run_id.split("-r", 1)[0]
    parts = stem.split("-")
    prompt_id = "-".join(parts[2:]) if len(parts) > 2 else None
    variant = prompt_id.rsplit("-", 1)[-1] if prompt_id and "-" in prompt_id else None
    task = prompt_id[: -(len(variant) + 1)] if variant and prompt_id else None
    return {"canonical_matrix_index": None, "harness": parts[0] if parts else None,
            "harness_profile": None, "semantic_task": task, "prompt_id": prompt_id,
            "prompt_variant": variant, "repetition": None, "seed": None}


def _ingest_completed(source: Path, experiment_id: str, run_id: str, base: dict[str, Any], rows: dict[str, list[dict[str, Any]]]) -> None:
    artifact_root = source / "artifacts" / run_id
    artifact = verify_artifact(artifact_root)
    verify_published_result(source, artifact)
    metrics_stored = verify_metrics_artifact(source / "analysis" / run_id / "metrics-v1")
    context = verify_context_analysis_artifact(source / "analysis" / run_id / "context-analysis-v2")
    artifact_sha = _sha(artifact_root / "manifest.json")
    if metrics_stored.manifest.source_artifact_manifest_sha256 != artifact_sha:
        raise ReportError("metrics artifact does not link to sealed run manifest")
    if context.source_artifact_manifest_sha256 != artifact_sha:
        raise ReportError("context analysis does not link to sealed run manifest")
    timing_root = source / "analysis" / run_id / "timing-provenance-v1"
    timing = verify_timing_provenance_artifact(timing_root) if timing_root.is_dir() else None
    metric_data = metrics_stored.metrics.model_dump(mode="json")
    context_data = context.model_dump(mode="json")
    run = {**base, "state": "completed", "evidence_status": "verified", "evidence_reason": None,
           "termination_class": metric_data["termination"]["termination_class"], **_run_metrics(metric_data, context_data)}
    rows["runs"].append(run)
    _append_metric_rows(experiment_id, run_id, metric_data, rows["metrics"])
    _append_request_rows(experiment_id, run_id, context_data, rows["requests"], rows["context_points"], run["wall_time_seconds"])
    _append_tool_rows(experiment_id, run_id, artifact_root, rows["tools"])
    _append_marker_rows(experiment_id, run_id, artifact_root, context_data, rows["markers"])
    _append_timing_rows(experiment_id, run_id, metric_data, timing.model_dump(mode="json") if timing else None, rows["timing"])
    _append_git_rows(experiment_id, run_id, metric_data, rows["git_change_metrics"])
    raw_manifest = json.loads((artifact_root / "run" / "manifest.json").read_text(encoding="utf-8"))
    rows["artifacts"].append({
        "experiment_id": experiment_id, "run_id": run_id,
        "artifact_manifest_id": artifact.artifact_manifest_id, "artifact_manifest_sha256": artifact_sha,
        "metrics_manifest_sha256": _sha(source / "analysis" / run_id / "metrics-v1" / "manifest.json"),
        "context_manifest_sha256": _sha(source / "analysis" / run_id / "context-analysis-v2" / "manifest.json"),
        "timing_manifest_sha256": _sha(timing_root / "manifest.json") if timing else None,
        "result_commit": artifact.result_commit, "result_ref": artifact.result_ref,
        "source_snapshot_sha256": artifact.source_snapshot_sha256,
        "capture_capabilities_sha256": canonical_sha256(raw_manifest.get("capture_capabilities", {})),
        "artifact_relative_path": f"artifacts/{run_id}",
    })


def _scalar(value: dict[str, Any] | None) -> tuple[float | int | None, str | None, str | None, str | None, str | None]:
    if not value:
        return None, None, "unavailable", "source_not_exposed", "not_available"
    provenance = value.get("provenance", {})
    return value.get("value"), value.get("units"), value.get("availability"), value.get("unavailable_reason"), provenance.get("method")


def _metric(data: dict[str, Any], group: str, name: str) -> Any:
    return data[group][name]


def _run_metrics(metrics: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    def value(group: str, name: str) -> Any: return _metric(metrics, group, name).get("value")
    initial = context["initial_task_context_tokens"]["value"]
    peak = value("context", "peak_context_tokens")
    return {
        "wall_time_seconds": value("timing", "wall_time_seconds"),
        "llm_requests": value("behavior", "llm_request_count"), "tool_calls": value("behavior", "tool_calls_total"),
        "input_tokens": value("tokens", "input_tokens_total"), "output_tokens": value("tokens", "output_tokens_total"),
        "total_tokens": value("tokens", "total_tokens"), "first_task_context_tokens": initial,
        "first_task_context_utilization_percent": context["initial_task_context_utilization_percent"]["value"],
        "peak_context_tokens": peak, "peak_context_utilization_percent": value("context", "peak_context_utilization_percent"),
        "context_growth_from_first_task_tokens": (peak - initial if isinstance(peak, int) and isinstance(initial, int) else None),
        "files_changed": value("git_result", "files_changed"), "lines_added": value("git_result", "lines_added"),
        "lines_deleted": value("git_result", "lines_deleted"),
    }


def _empty_run_metrics() -> dict[str, Any]:
    return {key: None for key in (
        "termination_class", "wall_time_seconds", "llm_requests", "tool_calls", "input_tokens", "output_tokens", "total_tokens",
        "first_task_context_tokens", "first_task_context_utilization_percent", "peak_context_tokens", "peak_context_utilization_percent",
        "context_growth_from_first_task_tokens", "files_changed", "lines_added", "lines_deleted",
    )}


def _append_metric_rows(experiment_id: str, run_id: str, metrics: dict[str, Any], destination: list[dict[str, Any]]) -> None:
    for group in ("timing", "tokens", "context", "behavior", "derived", "git_result"):
        for name, item in metrics[group].items():
            if not isinstance(item, dict) or "availability" not in item or "value" not in item:
                continue
            value, units, availability, reason, method = _scalar(item)
            destination.append({"experiment_id": experiment_id, "run_id": run_id, "metric_group": group,
                                "metric_name": name, "value": value, "units": units, "availability": availability,
                                "unavailable_reason": reason, "method": method})


def _append_request_rows(experiment_id: str, run_id: str, context: dict[str, Any], requests: list[dict[str, Any]], points: list[dict[str, Any]], wall: float | None) -> None:
    first_index = context.get("first_task_request_index")
    lookup = {item["model_request_index"]: item for item in context["requests"]}
    first_elapsed = lookup.get(first_index, {}).get("elapsed_seconds") if first_index else None
    for request in context["requests"]:
        value = request["input_context_tokens"]
        requests.append({"experiment_id": experiment_id, "run_id": run_id,
            "request_index": request["model_request_index"], "captured_http_request_index": request.get("captured_http_request_index"),
            "purpose": request["purpose"], "purpose_evidence": request["purpose_evidence"], "elapsed_seconds": request.get("elapsed_seconds"),
            "request_body_sha256": request.get("request_body_sha256"), "messages_sha256": request.get("messages_sha256"),
            "tool_schema_sha256": request.get("tool_schema_sha256"), "input_context_tokens": value.get("value"),
            "input_tokens_availability": value.get("availability"), "output_tokens": request["output_tokens"].get("value"),
            "output_tokens_availability": request["output_tokens"].get("availability"),
            "configured_max_context_tokens": request["configured_max_context_tokens"].get("value"),
            "context_utilization_percent": request["context_utilization_percent"].get("value"),
            "delta_vs_previous_inference_tokens": request["delta_vs_previous_inference_tokens"].get("value"),
            "delta_vs_first_task_tokens": request["delta_vs_first_task_tokens"].get("value"),
        })
        auxiliary = first_index is None or request["model_request_index"] < first_index
        elapsed = request.get("elapsed_seconds")
        task_elapsed = elapsed - first_elapsed if not auxiliary and elapsed is not None and first_elapsed is not None else None
        duration = wall - first_elapsed if wall is not None and first_elapsed is not None else None
        progress = (task_elapsed * 100 / duration if task_elapsed is not None and duration and duration > 0 else None)
        points.append({"experiment_id": experiment_id, "run_id": run_id, "request_index": request["model_request_index"],
            "task_request_index": (request["model_request_index"] - first_index + 1 if not auxiliary and first_index else None),
            "is_auxiliary": auxiliary, "elapsed_seconds": elapsed, "task_elapsed_seconds": task_elapsed,
            "normalized_elapsed_task_percent": progress, "context_tokens": value.get("value"),
            "context_utilization_percent": request["context_utilization_percent"].get("value"),
            "delta_vs_previous_tokens": request["delta_vs_previous_inference_tokens"].get("value"),
            "delta_vs_first_task_tokens": request["delta_vs_first_task_tokens"].get("value"),
            "availability": value.get("availability"), "unavailable_reason": value.get("unavailable_reason"),
        })


def _append_tool_rows(experiment_id: str, run_id: str, artifact: Path, destination: list[dict[str, Any]]) -> None:
    for event in load_normalized_events(artifact / "normalized" / "events.jsonl"):
        if event.event_kind not in {"tool_call_start", "tool_call_end", "file_read", "file_search", "file_edit", "file_write", "shell_command", "test_execution"}:
            continue
        payload = event.payload
        destination.append({"experiment_id": experiment_id, "run_id": run_id, "event_id": event.event_id,
            "event_kind": event.event_kind, "tool_name": payload.get("tool_name"),
            "category": payload.get("category") or _category(event.event_kind), "outcome": payload.get("outcome"),
            "elapsed_seconds": event.elapsed_seconds, "timing_semantics": payload.get("timing_semantics"),})


def _category(kind: str) -> str | None:
    return {"file_read": "read", "file_search": "search", "file_edit": "edit", "file_write": "write", "shell_command": "shell", "test_execution": "test"}.get(kind)


def _append_marker_rows(experiment_id: str, run_id: str, artifact: Path, context: dict[str, Any], destination: list[dict[str, Any]]) -> None:
    """Emit only observed markers; execution timing stays a distinct analysis."""
    requests = {item["model_request_index"]: item for item in context["requests"]}
    first_index = context.get("first_task_request_index")
    first_elapsed = requests.get(first_index, {}).get("elapsed_seconds") if first_index else None
    seen: set[str] = set()
    mapping = {
        "llm_request": ("first_model_request", "model_request_observed"),
        "tool_call_start": ("first_tool_event", "tool_event_observed"),
        "tool_call_end": ("first_tool_event", "tool_event_observed"),
        "file_read": ("first_tool_event", "tool_event_observed"),
        "file_search": ("first_tool_event", "tool_event_observed"),
        "file_edit": ("first_edit_event", "tool_event_observed"),
        "file_write": ("first_edit_event", "tool_event_observed"),
        "test_execution": ("first_test_event", "tool_event_observed"),
        "compaction_start": ("compaction_start", "tool_event_observed"),
        "compaction_end": ("compaction_end", "tool_event_observed"),
        "output_truncation": ("output_truncation", "tool_event_observed"),
        "context_overflow": ("context_overflow", "tool_event_observed"),
        "timeout": ("timeout", "tool_event_observed"),
        "run_end": ("run_end", "tool_event_observed"),
    }
    for event in load_normalized_events(artifact / "normalized" / "events.jsonl"):
        mapped = mapping.get(event.event_kind)
        if mapped is None or event.elapsed_seconds is None:
            continue
        kind, semantics = mapped
        if kind.startswith("first_") and kind in seen:
            continue
        seen.add(kind)
        task_elapsed = event.elapsed_seconds - first_elapsed if first_elapsed is not None and event.elapsed_seconds >= first_elapsed else None
        destination.append({"experiment_id": experiment_id, "run_id": run_id, "marker_kind": kind,
                            "elapsed_seconds": event.elapsed_seconds, "task_elapsed_seconds": task_elapsed,
                            "timing_semantics": semantics, "source_event_id": event.event_id})


def _append_timing_rows(experiment_id: str, run_id: str, metrics: dict[str, Any], timing: dict[str, Any] | None, destination: list[dict[str, Any]]) -> None:
    for name, item in metrics["timing"].items():
        if name == "schema_version": continue
        value, _units, availability, reason, method = _scalar(item)
        semantics = "unavailable"
        if name == "time_to_first_llm_request_seconds": semantics = "model_request_observed"
        elif name in {"time_to_first_tool_call_seconds", "time_to_first_edit_seconds", "tool_execution_time_seconds"}: semantics = "harness_tool_execution_start"
        destination.append({"experiment_id": experiment_id, "run_id": run_id, "timing_name": name, "value_seconds": value,
                            "availability": availability, "unavailable_reason": reason, "semantics": semantics,
                            "method": method, "source": "metrics-v1"})
    if timing is None: return
    for name in ("time_to_first_harness_tool_execution", "time_to_first_harness_edit_execution", "time_to_first_observed_tool_event", "time_to_first_observed_edit_event", "time_to_first_model_tool_call_observed", "time_to_first_model_edit_call_observed"):
        item = timing[name]
        destination.append({"experiment_id": experiment_id, "run_id": run_id, "timing_name": name,
                            "value_seconds": item["value"], "availability": item["availability"],
                            "unavailable_reason": item["unavailable_reason"], "semantics": item["semantics"],
                            "method": item["method"], "source": "timing-provenance-v1"})


def _append_git_rows(experiment_id: str, run_id: str, metrics: dict[str, Any], destination: list[dict[str, Any]]) -> None:
    for name, item in metrics["git_result"].items():
        if name == "schema_version": continue
        destination.append({"experiment_id": experiment_id, "run_id": run_id, "metric_name": name,
                            "value": item.get("value"), "availability": item.get("availability"),
                            "unavailable_reason": item.get("unavailable_reason")})


def _append_summaries(experiment_id: str, runs: list[dict[str, Any]], destination: list[dict[str, Any]]) -> None:
    dimensions = {"all": lambda _r: "all", "harness": lambda r: r.get("harness"), "semantic_task": lambda r: r.get("semantic_task"),
                  "prompt_variant": lambda r: r.get("prompt_variant"), "harness_task": lambda r: _joint(r, "harness", "semantic_task"),
                  "harness_prompt_variant": lambda r: _joint(r, "harness", "prompt_variant"),
                  "harness_task_prompt_variant": lambda r: _joint(r, "harness", "semantic_task", "prompt_variant")}
    metrics = ("wall_time_seconds", "input_tokens", "output_tokens", "llm_requests", "tool_calls", "first_task_context_tokens", "peak_context_tokens", "context_growth_from_first_task_tokens", "files_changed")
    for grouping, selector in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            key = selector(run)
            if key is not None: groups[str(key)].append(run)
        for key, members in sorted(groups.items()):
            for metric in metrics:
                values = [float(item[metric]) for item in members if item.get(metric) is not None]
                q1, median, q3 = quantile_type7(values, .25), quantile_type7(values, .5), quantile_type7(values, .75)
                destination.append({"experiment_id": experiment_id, "grouping": grouping, "group_key": key, "metric_name": metric,
                    "n_planned": len(members), "n_completed": sum(item["state"] == "completed" for item in members),
                    "n_successful": sum(item.get("termination_class") == "success" for item in members),
                    "n_failed_or_invalid": sum(item["state"] in {"failed", "invalid", "interrupted"} or item.get("evidence_status") == "invalid" for item in members),
                    "n_available": len(values), "median": median, "q1": q1, "q3": q3,
                    "iqr": (q3 - q1 if q3 is not None and q1 is not None else None),
                    "minimum": min(values) if values else None, "maximum": max(values) if values else None})


def _joint(row: dict[str, Any], *keys: str) -> str | None:
    values = [row.get(key) for key in keys]
    return " × ".join(str(value) for value in values) if all(value is not None for value in values) else None


def _append_curves(experiment_id: str, runs: list[dict[str, Any]], points: list[dict[str, Any]], destination: list[dict[str, Any]]) -> None:
    points_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points: points_by_run[point["run_id"]].append(point)
    for run in runs:
        run_points = sorted(points_by_run.get(run["run_id"], []), key=lambda p: p["request_index"])
        first = next((item["elapsed_seconds"] for item in run_points if not item["is_auxiliary"]), None)
        for point in run_points:
            if point["is_auxiliary"]: continue
            destination.append({"experiment_id": experiment_id, "curve_kind": "absolute_elapsed_task_time", "grouping": "run", "group_key": run["run_id"], "run_id": run["run_id"], "x": point["task_elapsed_seconds"], "context_utilization_percent": point["context_utilization_percent"], "value_method": "measured", "n_available": 1, "median": None, "q1": None, "q3": None, "termination_class": run.get("termination_class")})
            destination.append({"experiment_id": experiment_id, "curve_kind": "request_index", "grouping": "run", "group_key": run["run_id"], "run_id": run["run_id"], "x": float(point["task_request_index"]), "context_utilization_percent": point["context_utilization_percent"], "value_method": "measured", "n_available": 1, "median": None, "q1": None, "q3": None, "termination_class": run.get("termination_class")})
        normalized = normalized_elapsed_curve(run_points, first_task_elapsed_seconds=first, wall_time_seconds=run.get("wall_time_seconds"))
        for point in normalized:
            destination.append({"experiment_id": experiment_id, "curve_kind": "normalized_elapsed_task_time", "grouping": "run", "group_key": run["run_id"], "run_id": run["run_id"], **point, "n_available": 1, "median": None, "q1": None, "q3": None, "termination_class": run.get("termination_class")})
    # Aggregate only like-for-like normalized task-time series, preserving N.
    individual = [row for row in destination if row["curve_kind"] == "normalized_elapsed_task_time"]
    for grouping, key_fn in (("all", lambda _r: "all"), ("harness", lambda r: r.get("harness"))):
        groups: dict[str, set[str]] = defaultdict(set)
        for run in runs:
            key = key_fn(run)
            if key is not None: groups[str(key)].add(run["run_id"])
        for key, run_ids in sorted(groups.items()):
            for index in range(101):
                values = [row["context_utilization_percent"] for row in individual if row["run_id"] in run_ids and row["x"] == float(index) and row["context_utilization_percent"] is not None]
                if not values: continue
                destination.append({"experiment_id": experiment_id, "curve_kind": "normalized_elapsed_task_time_aggregate", "grouping": grouping, "group_key": key, "run_id": None, "x": float(index), "context_utilization_percent": None, "value_method": "aggregate_type7", "n_available": len(values), "median": quantile_type7(values, .5), "q1": quantile_type7(values, .25), "q3": quantile_type7(values, .75), "termination_class": None})


def _write_parquet(root: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, schema in SCHEMAS.items():
        records = [{field.name: row.get(field.name) for field in schema} for row in rows[name]]
        table = pa.Table.from_pylist(records, schema=schema)
        pq.write_table(table, root / f"{name}.parquet", compression="zstd", version="2.6", data_page_version="2.0", use_dictionary=False, write_statistics=True)


def _build_database(path: Path, parquet_root: Path) -> None:
    connection = duckdb.connect(str(path))
    try:
        connection.execute("PRAGMA threads=1")
        for name in sorted(SCHEMAS):
            safe = name.replace("_", "_")
            connection.execute(f"CREATE TABLE {safe} AS SELECT * FROM read_parquet(?)", [str(parquet_root / f"{name}.parquet")])
        connection.execute("CREATE VIEW all_runs AS SELECT * FROM runs")
        connection.execute("CREATE VIEW successful_runs AS SELECT * FROM runs WHERE termination_class = 'success' AND evidence_status = 'verified'")
        connection.execute("CREATE VIEW failed_runs AS SELECT * FROM runs WHERE state IN ('failed','interrupted','invalid') OR evidence_status = 'invalid'")
        connection.execute("CREATE VIEW per_harness_metrics AS SELECT * FROM summaries WHERE grouping = 'harness'")
        connection.execute("CREATE VIEW per_task_metrics AS SELECT * FROM summaries WHERE grouping = 'semantic_task'")
        connection.execute("CREATE VIEW per_prompt_variant_metrics AS SELECT * FROM summaries WHERE grouping = 'prompt_variant'")
        connection.execute("CREATE VIEW per_repetition_metrics AS SELECT repetition, metric_name, median(value) AS median_value FROM (SELECT r.repetition, m.metric_name, m.value FROM runs r JOIN metrics m USING (experiment_id, run_id)) GROUP BY repetition, metric_name")
        connection.execute("CREATE VIEW context_series AS SELECT * FROM context_points")
        connection.execute("CREATE VIEW tool_usage AS SELECT * FROM tools")
        connection.execute("CREATE VIEW git_changes AS SELECT * FROM git_change_metrics")
    finally:
        connection.close()


def _summary(state: ExperimentState, runs: list[dict[str, Any]], summaries: list[dict[str, Any]], excluded: list[dict[str, str]], environment: dict[str, str | None]) -> dict[str, Any]:
    counts = _state_counts(state)
    return {"schema_version": REPORT_SCHEMA_VERSION, "generator": {"name": REPORT_GENERATOR, "version": REPORT_GENERATOR_VERSION, "agent_bench_version": __version__},
            "experiment_id": state.experiment_id, "definition_digest": state.definition_digest, "expansion_digest": state.expansion_digest,
            "fixed_environment": environment,
            "completion": {**counts, "is_partial": counts["completed"] != counts["total"], "label": f"PARTIAL EXPERIMENT — {counts['completed']} / {counts['total']} completed" if counts["completed"] != counts["total"] else "COMPLETE EXPERIMENT"},
            "included_verified_run_ids": [item["run_id"] for item in runs if item["evidence_status"] == "verified"], "excluded_or_invalid_runs": excluded,
            "summary_row_count": len(summaries), "quality_notice": "Execution success is not task-quality or manual-review proof."}


def _archival_manifest(state: ExperimentState, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "1.0.0", "experiment_id": state.experiment_id, "expansion_digest": state.expansion_digest,
            "runs": [{"run_id": item["run_id"], "sealed_artifact_manifest_sha256": item["artifact_manifest_sha256"], "persistent_result_ref": item["result_ref"], "persistent_result_commit": item["result_commit"], "expected_raw_bundle_filename": f"{item['run_id']}-raw-evidence.tar.zst", "expected_raw_bundle_sha256": None, "future_release_locations": []} for item in artifacts]}


def _html_report(summary: dict[str, Any], rows: dict[str, list[dict[str, Any]]]) -> str:
    completion = summary["completion"]
    run_rows = "".join("<tr>" + "".join(f"<td>{html.escape(_display(run.get(key)))}</td>" for key in ("run_id", "harness", "semantic_task", "prompt_variant", "repetition", "state", "evidence_status", "termination_class", "wall_time_seconds", "input_tokens", "peak_context_tokens")) + "</tr>" for run in rows["runs"])
    failures = "".join("<tr>" + "".join(f"<td>{html.escape(_display(row.get(key)))}</td>" for key in ("run_id", "state", "termination_class", "detail")) + "</tr>" for row in rows["failures"]) or "<tr><td colspan='4'>None</td></tr>"
    charts = _svg_charts(rows["curves"], rows["markers"])
    return f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Agent Bench report — {html.escape(summary['experiment_id'])}</title>
<style>body{{font:15px system-ui,sans-serif;margin:2rem;color:#172033;background:#f7f8fa}}main{{max-width:1200px;margin:auto}}.notice{{padding:1rem;background:#fff3cd;border-left:5px solid #b78200;font-weight:700}}section{{background:white;padding:1rem 1.25rem;margin:1rem 0;border:1px solid #d9dde5;border-radius:6px}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:.42rem;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}}th{{background:#f1f4f8}}svg{{width:100%;height:280px;border:1px solid #d9dde5;background:#fff}}.muted{{color:#596579}}code{{word-break:break-all}}</style>
<main><h1>Agent Bench deterministic report</h1><p class="notice">{html.escape(completion['label'])}</p>
<section><h2>Experiment and fixed environment</h2><p>Execution success is a machine fact; it is <strong>not</strong> a task-quality or manual-correctness result. Manual review is outside this report.</p><pre>{html.escape(json.dumps({"completion": {k:v for k,v in completion.items() if k != 'label'}, "fixed_environment": summary["fixed_environment"]}, sort_keys=True, indent=2))}</pre></section>
<section><h2>Runs</h2><table><thead><tr><th>Run</th><th>Harness</th><th>Task</th><th>Prompt</th><th>Rep</th><th>State</th><th>Evidence</th><th>Termination</th><th>Wall s</th><th>Input tokens</th><th>Peak context</th></tr></thead><tbody>{run_rows}</tbody></table></section>
<section><h2>Context visualizations</h2><p class="muted">Task time begins at the first real task inference request. Auxiliary/title requests remain separately counted. Normalized elapsed task time is not semantic completion progress.</p>{charts}</section>
<section><h2>Failures and unavailable evidence</h2><table><thead><tr><th>Run</th><th>State</th><th>Termination</th><th>Detail</th></tr></thead><tbody>{failures}</tbody></table></section>
<section><h2>Derived files</h2><p>Parquet, DuckDB, checksums, the archival manifest, and provenance references accompany this offline report. No raw reasoning text is included.</p></section></main></html>\n"""


def _svg_charts(curves: list[dict[str, Any]], markers: list[dict[str, Any]]) -> str:
    labels = (("absolute_elapsed_task_time", "Context vs absolute elapsed task time", "seconds"), ("normalized_elapsed_task_time", "Context vs normalized elapsed task time", "% elapsed task time"), ("request_index", "Context vs real task request index", "request index"))
    return "".join(f"<h3>{title}</h3><p class='muted'>X: {axis}; Y: context utilization percent. Individual runs only; unavailable points are omitted. Markers are observation timing, never inferred execution timing.</p>{_svg([row for row in curves if row['curve_kind'] == kind], markers if kind == 'absolute_elapsed_task_time' else [])}" for kind, title, axis in labels)


def _svg(rows: list[dict[str, Any]], markers: list[dict[str, Any]]) -> str:
    valid = [row for row in rows if row.get("x") is not None and row.get("context_utilization_percent") is not None]
    if not valid: return "<p class='muted'>No observable context points.</p>"
    max_x = max(float(row["x"]) for row in valid) or 1.0
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid: by_run[str(row["run_id"])].append(row)
    paths = []
    colours = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#b45309")
    for index, (run_id, points) in enumerate(sorted(by_run.items())):
        points.sort(key=lambda row: float(row["x"]))
        coords = " ".join(f"{45 + 735 * float(point['x']) / max_x:.3f},{250 - 210 * min(100, max(0, float(point['context_utilization_percent']))) / 100:.3f}" for point in points)
        paths.append(f"<polyline fill='none' stroke='{colours[index % len(colours)]}' stroke-width='2' points='{coords}'><title>{html.escape(run_id)}</title></polyline>")
    marker_svg = "".join(
        f"<path d='M {45 + 735 * float(marker['task_elapsed_seconds']) / max_x:.3f} 250 l -4 8 h 8 z' fill='#111827'><title>{html.escape(str(marker['marker_kind']))}: {html.escape(str(marker['timing_semantics']))}</title></path>"
        for marker in markers if marker.get("task_elapsed_seconds") is not None and 0 <= float(marker["task_elapsed_seconds"]) <= max_x
    )
    return "<svg viewBox='0 0 820 280' role='img' aria-label='Context utilization'><line x1='45' y1='250' x2='780' y2='250' stroke='#64748b'/><line x1='45' y1='40' x2='45' y2='250' stroke='#64748b'/><text x='8' y='48'>100%</text><text x='15' y='254'>0%</text>" + "".join(paths) + marker_svg + "</svg>"


def _seal_report(root: Path, state: ExperimentState, rows: dict[str, list[dict[str, Any]]], excluded: list[dict[str, str]]) -> dict[str, Any]:
    files = {path.relative_to(root).as_posix(): _sha(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in {MANIFEST_NAME, CHECKSUMS_NAME}}
    manifest = {"schema_version": REPORT_SCHEMA_VERSION, "report_id": f"{state.experiment_id}-report-schema-v1", "generator_name": REPORT_GENERATOR,
                "generator_version": REPORT_GENERATOR_VERSION, "agent_bench_version": __version__, "experiment_id": state.experiment_id,
                "definition_digest": state.definition_digest, "expansion_digest": state.expansion_digest,
                "included_run_ids": [row["run_id"] for row in rows["runs"] if row["evidence_status"] == "verified"],
                "excluded_or_invalid_runs": excluded, "files": files,
                "calculation": {"quantiles": "hyndman-fan-type-7", "normalized_curve": "task-relative-linear-with-labelled-boundary-carry-v1", "raw_reasoning_included": False}}
    manifest["record_digest"] = canonical_sha256(manifest)
    _write_json(root / MANIFEST_NAME, manifest)
    checksums = {**files, MANIFEST_NAME: _sha(root / MANIFEST_NAME)}
    (root / CHECKSUMS_NAME).write_text("".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())), encoding="utf-8", newline="\n")
    return manifest


def _state_counts(state: ExperimentState) -> dict[str, int]:
    counts = Counter(item.state for item in state.runs)
    return {"total": len(state.runs), **{key: counts.get(key, 0) for key in ("completed", "failed", "interrupted", "invalid", "pending", "preflight", "running", "preserving", "analyzing")}}


def _read_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if len(digest) != 64 or relative in entries or relative.startswith("/") or ".." in Path(relative).parts:
            raise ReportError("invalid report checksum listing")
        entries[relative] = digest
    return entries


def _assert_public_safe(root: Path) -> None:
    prohibited = (b"authorization:", b'"authorization"', b"api_key", b"cookie:", b'"cookie"', b"/home/", b"<think>", b"reasoning_content")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix == ".duckdb":
            continue
        if path.suffix == ".parquet":
            table = pq.read_table(path)
            text = "\n".join(
                str(value)
                for field in table.schema
                if pa.types.is_string(field.type)
                for value in table.column(field.name).to_pylist()
                if value is not None
            ).encode("utf-8")
        else:
            text = path.read_bytes()
        if any(value in text.lower() for value in prohibited):
            raise ReportError(f"public export privacy audit failed: {path.name}")


def _display(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _round6(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
