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
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from agent_bench import __version__
from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.context_storage import verify_context_analysis_artifact
from agent_bench.failure import verify_failed_run
from agent_bench.events import load_normalized_events
from agent_bench.executor import ExperimentState
from agent_bench.functional_storage import verify_functional_validation_artifact
from agent_bench.matrix import expand_experiment
from agent_bench.metrics_storage import verify_metrics_artifact
from agent_bench.models import RunDefinition, canonical_sha256
from agent_bench.preservation import verify_artifact
from agent_bench.reasoning_tokenizer import LlamaTokenizeCounter, ReasoningTokenCache
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
PRESENTATION_NAME = "presentation.json"


class ReportError(RuntimeError):
    """Reporting evidence or derived output is unsafe to use."""


@dataclass(frozen=True)
class ReportBuild:
    root: Path
    manifest: dict[str, Any]


def _variant_observation_counts(
    rows: Iterable[dict[str, Any]], *, metric: str, variants: Iterable[str],
    reference: str | None = None,
) -> dict[str, dict[str, int]]:
    """Classify Variant Comparison N semantics without calling absence unavailable.

    The browser applies this same exact-stratum rule after interactive filters.
    This small pure counterpart makes the contract regression-testable: a
    missing peer can exclude a common stratum, but cannot invent a metric
    unavailability for a variant that supplied a finite observation.
    """
    names = tuple(variants)
    if reference is not None and reference not in names:
        raise ReportError("variant comparison reference is not displayed")
    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    missing_provenance = {name: 0 for name in names}
    for row in rows:
        name = str(row.get("profile"))
        if name not in missing_provenance:
            continue
        key = (row.get("subject_id"), row.get("semantic_task"), row.get("prompt_sha256"), row.get("repetition"), row.get("seed"))
        if any(value is None or value == "" for value in key[1:]):
            missing_provenance[name] += 1
            continue
        if name in cells[key]:
            raise ReportError("duplicate profile in Variant Comparison stratum")
        cells[key][name] = row
    result = {name: {"common_matched": 0, "excluded_incomplete": 0, "metric_unavailable": 0, "missing_provenance": missing_provenance[name]} for name in names}
    for cell in cells.values():
        for name in names:
            compared = (reference, name) if reference is not None else names
            complete = all(
                isinstance(cell.get(item, {}).get("metrics", {}).get(metric), (int, float))
                and not isinstance(cell.get(item, {}).get("metrics", {}).get(metric), bool)
                for item in compared
            )
            if complete:
                result[name]["common_matched"] += 1
            else:
                own_metrics = cell.get(name, {}).get("metrics", {})
                if isinstance(own_metrics, dict) and metric in own_metrics and not isinstance(own_metrics[metric], (int, float)):
                    result[name]["metric_unavailable"] += 1
                else:
                    result[name]["excluded_incomplete"] += 1
    return result


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
        ("functional_validation_status", pa.string()),
        ("functional_score_numerator", pa.int64()),
        ("functional_score_denominator", pa.int64()),
        ("functional_score_percent", pa.float64()),
        ("hard_gate_pass", pa.bool_()),
        ("baseline_regression_count", pa.int64()),
        ("baseline_regressions", pa.bool_()),
        ("failed_functional_test_count", pa.int64()),
        ("failed_functional_test_ids", pa.string()),
        ("functional_scenario_id", pa.string()),
        ("functional_tier", pa.string()),
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
        ("failure_class", pa.string()), ("failure_domain", pa.string()),
        ("failure_phase", pa.string()), ("harness_execution_started", pa.bool_()),
        ("llm_request_observed", pa.bool_()), ("preservation_completed", pa.bool_()),
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
    definition, environment, definition_presentation = _load_definition(state, experiment_definition)
    target = (output or source / REPORT_DIRECTORY).expanduser().resolve()
    if target.exists():
        raise ReportError(f"report destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".report-v1.incomplete-", dir=target.parent))
    try:
        rows, excluded = _ingest(source, state, definition, environment)
        presentation = _presentation(
            source, state, rows, summary_environment=environment,
            definition_presentation=definition_presentation,
        )
        _write_parquet(staging / PARQUET_DIRECTORY, rows)
        _build_database(staging / DATABASE_NAME, staging / PARQUET_DIRECTORY)
        summary = _summary(state, rows["runs"], rows["summaries"], excluded, environment)
        _write_json(staging / SUMMARY_NAME, summary)
        _write_json(staging / PRESENTATION_NAME, presentation)
        _write_json(staging / ARCHIVAL_MANIFEST_NAME, _archival_manifest(state, rows["artifacts"]))
        _write_json(staging / "charts.json", {"schema_version": REPORT_SCHEMA_VERSION, "curves": rows["curves"]})
        (staging / HTML_NAME).write_text(_html_report(summary, presentation), encoding="utf-8", newline="\n")
        manifest = _seal_report(staging, state, rows, excluded)
        verify_report(staging)
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ReportBuild(root=target, manifest=manifest)


def build_unified_report(
    experiment_outputs: list[Path], *, output: Path,
    experiment_definitions: list[Path] | None = None,
    reference_profile: str | None = None, include_all_pairs: bool = False,
    reasoning_tokenizer: LlamaTokenizeCounter | None = None,
    reasoning_token_cache: ReasoningTokenCache | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ReportBuild:
    """Build one rich, sealed report from compatible immutable experiment roots.

    Source roots are read only.  Their M9C layers are independently verified
    before their analytical rows are combined; no source report is required.
    """
    if len(experiment_outputs) < 2:
        raise ReportError("a unified report requires at least two experiment roots")
    if experiment_definitions is not None and len(experiment_definitions) != len(experiment_outputs):
        raise ReportError("--experiment-definition must be supplied once per root")
    sources = [root.expanduser().resolve() for root in experiment_outputs]
    if len(set(sources)) != len(sources):
        raise ReportError("experiment roots must be distinct")
    target = output.expanduser().resolve()
    if target.exists():
        raise ReportError(f"report destination already exists: {target}")

    # Comparison's reader has the strict historical-evidence compatibility
    # checks, including report-v1 fallback for roots that predate embedded YAML.
    from agent_bench.comparison import (
        _compatibility,
        _completed_run_total,
        _pairs,
        _read_root,
        _resolve_reasoning_token_cache,
        _summaries,
    )

    cache = _resolve_reasoning_token_cache(
        target, sources, reasoning_tokenizer, reasoning_token_cache,
    )
    total = _completed_run_total(sources) if progress is not None else 0
    completed = 0

    def on_run(run_id: str) -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress(completed, total, run_id)

    comparison_inputs = [
        _read_root(
            root, experiment_definitions[index] if experiment_definitions else None,
            reasoning_tokenizer, cache, on_run if progress is not None else None,
        )
        for index, root in enumerate(sources)
    ]
    compatibility = _compatibility(comparison_inputs)
    blocked = [item["dimension"] for item in compatibility if item["status"] in {"incompatible", "unavailable"}]
    if blocked:
        raise ReportError("incompatible or unavailable unified-report identity: " + ", ".join(blocked))
    comparison_rows = [row for item in comparison_inputs for row in item["rows"]]
    pairs = _pairs(comparison_rows, reference_profile=reference_profile, include_all_pairs=include_all_pairs)
    comparison_payload = {
        "schema_version": "1.0.0", "kind": "agent-bench-matched-comparison-v1",
        "interpretation": (
            "Matched deltas are deterministic efficiency/behavior observations, not quality or overall agent-performance wins. "
            "Prompt variants are strata and are not directly efficiency-comparable unless functional equivalence has been established. "
            "Pocket Ledger is a controlled microbenchmark and should not be interpreted as a complete measure of coding-agent capability."
        ),
        "sources": [{key: value for key, value in item.items() if key != "rows"} for item in comparison_inputs],
        "compatibility": compatibility, "reference_profile": reference_profile,
        "all_pairs_included": include_all_pairs or reference_profile is None,
        "raw_runs": comparison_rows, "matched_seed_comparisons": pairs,
        "aggregated_paired_effects": _summaries(pairs),
    }

    all_rows = {name: [] for name in SCHEMAS}
    excluded: list[dict[str, str]] = []
    source_by_run: dict[str, Path] = {}
    state_runs = []
    definition_presentations: list[dict[str, Any]] = []
    environments: list[dict[str, str | None]] = []
    source_states: list[ExperimentState] = []
    for index, (source, input_data) in enumerate(zip(sources, comparison_inputs, strict=True)):
        state = _load_state(source)
        definitions, environment, definition_presentation = _load_definition(
            state, experiment_definitions[index] if experiment_definitions else None,
        )
        if not definitions:
            raise ReportError(
                f"full rich ingestion needs an immutable definition for {state.experiment_id}; "
                "supply --experiment-definition once per root"
            )
        partial_rows, partial_excluded = _ingest(source, state, definitions, environment)
        for name, values in partial_rows.items():
            if name in {"summaries", "curves"}:
                continue
            all_rows[name].extend(values)
        excluded.extend({"run_id": f"{state.experiment_id}:{item['run_id']}", "reason": item["reason"]} for item in partial_excluded)
        for progress in state.runs:
            state_runs.append(progress.model_copy(update={"execution_index": len(state_runs) + 1}))
        for run_id in (row["run_id"] for row in partial_rows["runs"]):
            if run_id in source_by_run:
                raise ReportError(f"duplicate run ID across experiment roots: {run_id}")
            source_by_run[run_id] = source
        definition_presentations.append(definition_presentation)
        environments.append(environment)
        source_states.append(state)

    identity = canonical_sha256([
        {"experiment_id": state.experiment_id, "definition_digest": state.definition_digest,
         "expansion_digest": state.expansion_digest}
        for state in source_states
    ])
    unified_state = ExperimentState(
        experiment_id=f"unified-{identity[:16]}", definition_digest=identity,
        expansion_digest=canonical_sha256([run.model_dump(mode="json") for run in state_runs]),
        ordering={"mode": "source-root-order", "source_roots": [str(root) for root in sources]},
        runs=state_runs, updated_at="1970-01-01T00:00:00Z",
    )
    _append_summaries(unified_state.experiment_id, all_rows["runs"], all_rows["summaries"])
    _append_curves(unified_state.experiment_id, all_rows["runs"], all_rows["context_points"], all_rows["curves"])
    combined_definition = _combined_definition_presentation(definition_presentations, sources)
    environment = environments[0]
    presentation = _presentation(
        sources[0], unified_state, all_rows, summary_environment=environment,
        definition_presentation=combined_definition, source_by_run=source_by_run,
    )
    presentation["source_experiments"] = [
        {"experiment_id": state.experiment_id, "definition_digest": state.definition_digest,
         "root_name": source.name, "completed_runs": item["completed_runs"], "partial": item["partial"]}
        for source, state, item in zip(sources, source_states, comparison_inputs, strict=True)
    ]
    presentation["matched_comparison"] = comparison_payload
    presentation["data_files"].insert(1, "comparison.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".unified-report-v1.incomplete-", dir=target.parent))
    try:
        _write_parquet(staging / PARQUET_DIRECTORY, all_rows)
        _build_database(staging / DATABASE_NAME, staging / PARQUET_DIRECTORY)
        summary = _summary(unified_state, all_rows["runs"], all_rows["summaries"], excluded, environment)
        summary["source_experiments"] = presentation["source_experiments"]
        summary["matched_pair_count"] = len(pairs)
        _write_json(staging / SUMMARY_NAME, summary)
        _write_json(staging / PRESENTATION_NAME, presentation)
        _write_json(staging / "comparison.json", comparison_payload)
        _write_json(staging / ARCHIVAL_MANIFEST_NAME, _archival_manifest(unified_state, all_rows["artifacts"]))
        _write_json(staging / "charts.json", {"schema_version": REPORT_SCHEMA_VERSION, "curves": all_rows["curves"]})
        (staging / HTML_NAME).write_text(_html_report(summary, presentation), encoding="utf-8", newline="\n")
        manifest = _seal_report(staging, unified_state, all_rows, excluded)
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


def report_status(experiment_output: Path, *, report_root: Path | None = None) -> dict[str, Any]:
    """Return completion plus one explicitly selected derived report's integrity."""
    root = experiment_output.expanduser().resolve()
    state = _load_state(root)
    result = _state_counts(state)
    report = (report_root.expanduser().resolve() if report_root is not None else root / REPORT_DIRECTORY)
    candidates = sorted(path.name for path in root.glob("report-v*") if path.is_dir())
    result["selected_report_root"] = str(report)
    result["selected_report_source"] = "explicit --report-root" if report_root is not None else f"default {REPORT_DIRECTORY}"
    result["available_report_roots"] = candidates
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
        for name in (HTML_NAME, MANIFEST_NAME, CHECKSUMS_NAME, SUMMARY_NAME, ARCHIVAL_MANIFEST_NAME, PRESENTATION_NAME, "charts.json"):
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


def concise_chart_series_labels(runs: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Create collision-safe chart labels from structured run metadata only.

    Opaque run IDs remain provenance metadata, never the normal visible legend.
    """
    rows = sorted((dict(run) for run in runs if isinstance(run.get("run_id"), str)), key=lambda run: str(run["run_id"]))
    multiple_harnesses = len({str(run.get("harness")) for run in rows if run.get("harness")}) > 1
    base: dict[str, str] = {}
    for run in rows:
        profile = _concise_profile_label(str(run.get("harness_profile") or "default"), str(run.get("harness") or ""))
        parts = ([_harness_display(str(run.get("harness")))] if multiple_harnesses and run.get("harness") else [])
        parts += [profile, str(run.get("semantic_task") or "task unavailable"), str(run.get("prompt_variant") or "prompt unavailable")]
        repetition = run.get("repetition")
        parts.append(f"R{int(repetition):03d}" if isinstance(repetition, int) else "R?" )
        base[str(run["run_id"])] = " · ".join(parts)
    return _disambiguate_chart_labels(rows, base)


def _concise_profile_label(profile_id: str, harness: str) -> str:
    known = {
        "hermes-default-v1": "xhigh",
        "hermes-reasoning-medium-v1": "medium",
        "hermes-reasoning-low-v1": "low",
        "hermes-reasoning-off-v1": "off",
    }
    if profile_id in known:
        return known[profile_id]
    prefix = f"{harness}-" if harness else ""
    label = profile_id[len(prefix):] if prefix and profile_id.startswith(prefix) else profile_id
    if label.endswith("-v1"):
        label = label[:-3]
    return label or "default"


def _harness_display(harness: str) -> str:
    return {"hermes": "Hermes", "opencode": "OpenCode", "pi": "Pi"}.get(harness, harness.title())


def _disambiguate_chart_labels(rows: list[dict[str, Any]], labels: dict[str, str]) -> dict[str, str]:
    """Add only structured discriminators to labels that collide."""
    collisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        collisions[labels[str(row["run_id"])]].append(row)
    for label, members in collisions.items():
        if len(members) < 2:
            continue
        for row in members:
            seed = row.get("seed")
            if seed is not None:
                labels[str(row["run_id"])] = f"{label} · seed {seed}"
            elif row.get("execution_index") is not None:
                labels[str(row["run_id"])] = f"{label} · run #{row['execution_index']}"
        secondary: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in members:
            secondary[labels[str(row["run_id"])]].append(row)
        for repeated in secondary.values():
            if len(repeated) > 1:
                for number, row in enumerate(sorted(repeated, key=lambda item: str(item["run_id"])), start=1):
                    labels[str(row["run_id"])] += f" · series {number}"
    return labels


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


def _load_definition(
    state: ExperimentState, explicit: Path | None,
) -> tuple[dict[str, RunDefinition], dict[str, str | None], dict[str, Any]]:
    candidates = [explicit] if explicit is not None else [
        Path.cwd() / "experiments" / f"{state.experiment_id}.yaml",
        *sorted((Path.cwd() / "experiments").glob("*.yaml")),
    ]
    matches: list[tuple[Any, Path]] = []
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        try:
            definition = load_experiment(candidate)
        except ExperimentConfigError as exc:
            if explicit is not None:
                raise ReportError(f"invalid supplied experiment definition: {exc}") from exc
            continue
        if definition.experiment_id == state.experiment_id and definition.definition_digest == state.definition_digest:
            matches.append((definition, candidate))
    if not matches:
        if explicit is not None:
            raise ReportError("experiment definition does not match immutable experiment state")
        return {}, {}, {"definition_available": False}
    definition, candidate = matches[0]
    environment = {
        "fixed_environment_id": definition.fixed_environment.fixed_environment_id,
        "model": definition.fixed_environment.model.name,
        "model_sha256": definition.fixed_environment.model.sha256,
        "backend": definition.fixed_environment.backend.implementation,
        "backend_commit": definition.fixed_environment.backend.commit,
        "hardware_name": definition.fixed_environment.hardware.name,
        "gpu_model": definition.fixed_environment.hardware.gpu_model,
    }
    return (
        {run.run_id: run for run in expand_experiment(definition)},
        environment,
        _definition_presentation(definition, candidate),
    )


def _combined_definition_presentation(
    definitions: list[dict[str, Any]], sources: list[Path],
) -> dict[str, Any]:
    """Merge display-only immutable metadata without inventing a new experiment."""
    first = dict(definitions[0])
    for field, key in (("prompts", "prompt_id"), ("profiles", "profile_id"), ("harnesses", "harness_id")):
        seen: dict[str, dict[str, Any]] = {}
        for definition in definitions:
            for item in definition.get(field, []):
                if isinstance(item, dict) and isinstance(item.get(key), str):
                    prior = seen.setdefault(item[key], item)
                    if prior != item:
                        raise ReportError(f"incompatible {field} metadata for {item[key]}")
        first[field] = [seen[name] for name in sorted(seen)]
    first["definition_available"] = True
    first["definition_source"] = "multiple immutable definitions"
    first["source_definition_digests"] = [definition.get("definition_digest") for definition in definitions]
    first["source_root_names"] = [source.name for source in sources]
    first["repetition_indices"] = sorted({
        repetition for definition in definitions for repetition in definition.get("repetition_indices", [])
        if isinstance(repetition, int)
    })
    first["repetitions"] = len(first["repetition_indices"])
    return first


def _definition_presentation(definition: Any, source: Path) -> dict[str, Any]:
    """Return public, definition-derived display data without host paths.

    This is deliberately a presentation companion rather than a new persisted
    benchmark schema.  It makes the human report useful while the Parquet
    tables retain their compact analytical contract.
    """
    repository_root = source.parent.parent
    profiles: list[dict[str, Any]] = []
    for profile in definition.harness_profiles:
        profile_path = repository_root / "environment" / "harnesses" / profile.profile_id / "profile.yaml"
        materialized: dict[str, Any] = {}
        if profile_path.is_file():
            loaded = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                materialized = _sanitize_public(loaded)
        profiles.append({
            "profile_id": profile.profile_id,
            "harness_id": profile.harness_id,
            "profile_version": profile.profile_version,
            "kind": profile.kind,
            "upstream_defaults_source": profile.upstream_defaults_source,
            "settings": _sanitize_public(profile.settings),
            "deviations": list(profile.deviations),
            "bundle_sha256": profile.bundle_sha256,
            "source": _repository_relative(profile_path, repository_root),
            "source_sha256": _sha(profile_path) if profile_path.is_file() else None,
            "materialized_profile": materialized or None,
        })
    backend_path = repository_root / "environment" / "backend-v1.yaml"
    backend_configuration: dict[str, Any] = {}
    if backend_path.is_file():
        loaded_backend = yaml.safe_load(backend_path.read_text(encoding="utf-8"))
        if isinstance(loaded_backend, dict):
            backend_configuration = _sanitize_public(loaded_backend)
    prompts = [{
        "prompt_id": prompt.prompt_id,
        "semantic_task_id": prompt.semantic_task_id,
        "variant_label": prompt.variant_label,
        "content": prompt.content,
        "encoding": prompt.encoding,
        "byte_length": prompt.byte_length,
        "sha256": prompt.sha256,
        "metadata": _sanitize_public(prompt.metadata),
    } for prompt in definition.prompts]
    return {
        "definition_available": True,
        "definition_source": _repository_relative(source, repository_root),
        "definition_digest": definition.definition_digest,
        "experiment_name": definition.name,
        "experiment_description": definition.description,
        "portable_baseline": definition.portable_baseline.model_dump(
            mode="json", exclude={"definition_digest"},
        ),
        "ordering": definition.ordering.model_dump(mode="json"),
        "run_limits": definition.run_limits.model_dump(mode="json"),
        "repetitions": definition.repetitions,
        "repetition_indices": list(definition.effective_repetition_indices),
        "harnesses": [{
            "harness_id": harness.harness_id,
            "display_name": harness.display_name,
            "version": harness.version,
            "upstream_project": harness.upstream_project,
            "supported_raw_capture_sources": list(harness.supported_raw_capture_sources),
        } for harness in definition.harnesses],
        "profiles": profiles,
        "prompts": prompts,
        "fixed_environment": _sanitize_public(definition.fixed_environment.model_dump(mode="json")),
        "backend_configuration": backend_configuration or None,
        "backend_configuration_source": _repository_relative(backend_path, repository_root) if backend_path.is_file() else None,
        "backend_configuration_sha256": _sha(backend_path) if backend_path.is_file() else None,
    }


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return path.relative_to(repository_root).as_posix()
    except ValueError:
        return path.name


def _sanitize_public(value: Any, key: str = "") -> Any:
    """Remove host locations and credentials from a human-facing derivative."""
    sensitive = {"api_key", "authorization", "token", "secret", "password", "cookie"}
    if key.lower() in sensitive or any(part in key.lower() for part in ("api_key", "secret", "password", "authorization", "cookie")):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item).replace("reasoning_content", "reasoning-content"): _sanitize_public(member, str(item))
            for item, member in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_public(item, key) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and value.startswith("/"):
        return f"<host-path-redacted>/{Path(value).name}"
    if isinstance(value, str):
        # This label can occur in capability/profile metadata.  It is not raw
        # reasoning, but spelling it with an underscore would trip the public
        # export's deliberately conservative raw-content detector.
        return value.replace("reasoning_content", "reasoning-content")
    return value


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
            failed_evidence = None
            if progress.state in {"failed", "interrupted", "invalid"}:
                try:
                    failed_evidence = verify_failed_run(source / "runs" / progress.run_id)
                except Exception as exc:
                    failed_reason = f"{type(exc).__name__}: {exc}"
                else:
                    failed_reason = None
            else:
                failed_reason = None
            if failed_evidence is not None:
                manifest = failed_evidence.manifest
                evidence_status = "verified_failed_run_evidence"
                evidence_reason = manifest.reason
                termination = manifest.failure_class
            elif progress.state == "pending":
                evidence_status, evidence_reason, termination = "not_executed", "planned but not executed", None
            else:
                evidence_status, evidence_reason, termination = "unverified_failure", failed_reason or progress.detail, None
            functional_failure = _functional_empty(
                progress.functional_validation_status or ("unavailable" if definition and definition.functional_scenario else "not_applicable")
            )
            failure_run = {**base, **_empty_run_metrics(), **functional_failure, "state": progress.state,
                           "evidence_status": evidence_status, "evidence_reason": evidence_reason,
                           "termination_class": termination}
            rows["runs"].append(failure_run)
            # Planned/pending rows are not failures.  They remain in the run
            # matrix and get a separate dashboard section.
            if progress.state in {"failed", "interrupted", "invalid"}:
                rows["failures"].append({**base, "state": progress.state, "termination_class": termination,
                    "detail": evidence_reason, "evidence_status": evidence_status,
                    "failure_class": (failed_evidence.manifest.failure_class if failed_evidence else progress.failure_class),
                    "failure_domain": progress.failure_domain, "failure_phase": progress.failure_phase,
                    "harness_execution_started": progress.harness_execution_started,
                    "llm_request_observed": progress.llm_request_observed,
                    "preservation_completed": progress.preservation_completed})
            continue
        try:
            _ingest_completed(source, state.experiment_id, progress.run_id, base, rows, definition)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            excluded.append({"run_id": progress.run_id, "reason": reason})
            rows["runs"].append({**base, "state": "completed", "evidence_status": "invalid", "evidence_reason": reason, **_empty_run_metrics(), **_functional_empty("error" if definition and definition.functional_scenario else "not_applicable")})
            rows["failures"].append({**base, "state": "completed", "termination_class": None, "detail": reason, "evidence_status": "invalid",
                                     "failure_class": None, "failure_domain": None, "failure_phase": None,
                                     "harness_execution_started": None, "llm_request_observed": None, "preservation_completed": None})
    _append_summaries(state.experiment_id, rows["runs"], rows["summaries"])
    _append_curves(state.experiment_id, rows["runs"], rows["context_points"], rows["curves"])
    return rows, excluded


def _identity(run_id: str, definition: RunDefinition | None) -> dict[str, Any]:
    if definition is not None:
        prompt_variant = definition.prompt_id.rsplit("-", 1)[-1]
        functional = definition.functional_scenario
        return {"canonical_matrix_index": definition.matrix_index, "harness": definition.harness_id,
                "harness_profile": definition.profile_id, "semantic_task": definition.semantic_task_id,
                "prompt_id": definition.prompt_id, "prompt_variant": prompt_variant,
                "repetition": definition.repetition_index, "seed": definition.generation_seed,
                "functional_scenario_id": functional.scenario_id if functional else None,
                "functional_tier": functional.tier if functional else None}
    # Existing M9B evidence does not duplicate its definition; this conservative
    # fallback only decodes the checked-in prompt-id naming convention.
    stem = run_id.split("-r", 1)[0]
    parts = stem.split("-")
    prompt_id = "-".join(parts[2:]) if len(parts) > 2 else None
    variant = prompt_id.rsplit("-", 1)[-1] if prompt_id and "-" in prompt_id else None
    task = prompt_id[: -(len(variant) + 1)] if variant and prompt_id else None
    return {"canonical_matrix_index": None, "harness": parts[0] if parts else None,
            "harness_profile": None, "semantic_task": task, "prompt_id": prompt_id,
            "prompt_variant": variant, "repetition": None, "seed": None,
            "functional_scenario_id": None, "functional_tier": None}


def _ingest_completed(source: Path, experiment_id: str, run_id: str, base: dict[str, Any], rows: dict[str, list[dict[str, Any]]], definition: RunDefinition | None) -> None:
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
    functional = _functional_report_fields(source, artifact_root, run_id, definition)
    run = {**base, "state": "completed", "evidence_status": "verified", "evidence_reason": None,
           "termination_class": metric_data["termination"]["termination_class"], **_run_metrics(metric_data, context_data), **functional}
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


def _functional_report_fields(source: Path, artifact_root: Path, run_id: str, definition: RunDefinition | None) -> dict[str, Any]:
    """Read only the verified M13 artifact; legacy rows are not functional FAIL."""
    if definition is None or definition.functional_scenario is None:
        return _functional_empty("not_applicable")
    stored = verify_functional_validation_artifact(source / "analysis" / run_id / "functional-validation-v1")
    artifact_sha = _sha(artifact_root / "manifest.json")
    record = stored.result
    run_manifest_sha = _sha(artifact_root / "run" / "manifest.json")
    if (
        record.run_id != run_id
        or record.source_artifact_manifest_sha256 != artifact_sha
        or record.source_snapshot_sha256 != artifact.source_snapshot_sha256
        or record.source_run_manifest_sha256 != run_manifest_sha
    ):
        raise ReportError("functional validation artifact does not link to the sealed run evidence")
    return _functional_fields_from_record(record)


def _functional_fields_from_record(record: Any) -> dict[str, Any]:
    """Project one verified validator record into compact report fields."""
    result = record.functional_result
    return {
        "functional_validation_status": record.validation_status,
        "functional_score_numerator": record.acceptance_score_numerator,
        "functional_score_denominator": record.acceptance_score_denominator,
        "functional_score_percent": result.score_percent if record.validation_status in {"pass", "fail"} else None,
        "hard_gate_pass": result.hard_gate_pass if record.validation_status in {"pass", "fail"} else None,
        "baseline_regression_count": result.baseline_regression["failed"],
        "baseline_regressions": result.baseline_regression["failed"] > 0,
        "failed_functional_test_count": result.failed_tests,
        "failed_functional_test_ids": ", ".join(test.test_id for test in result.tests if test.outcome == "failed") or None,
        "functional_scenario_id": record.scenario.scenario_id,
        "functional_tier": record.scenario.tier,
    }


def _functional_empty(status: str) -> dict[str, Any]:
    return {
        "functional_validation_status": status,
        "functional_score_numerator": None, "functional_score_denominator": None,
        "functional_score_percent": None, "hard_gate_pass": None,
        "baseline_regression_count": None, "baseline_regressions": None,
        "failed_functional_test_count": None,
        "failed_functional_test_ids": None,
    }


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
    for group in ("timing", "tokens", "reasoning", "context", "behavior", "derived", "git_result"):
        if not isinstance(metrics.get(group), dict):
            continue
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
    dimensions = {"all": lambda _r: "all", "harness": lambda r: r.get("harness"),
                  "harness_profile": lambda r: r.get("harness_profile"), "semantic_task": lambda r: r.get("semantic_task"),
                  "prompt_variant": lambda r: r.get("prompt_variant"), "repetition": lambda r: r.get("repetition"),
                  "harness_task": lambda r: _joint(r, "harness", "semantic_task"),
                  "harness_profile_task": lambda r: _joint(r, "harness_profile", "semantic_task"),
                  "harness_prompt_variant": lambda r: _joint(r, "harness", "prompt_variant"),
                  "harness_profile_prompt_variant": lambda r: _joint(r, "harness_profile", "prompt_variant"),
                  "harness_task_prompt_variant": lambda r: _joint(r, "harness", "semantic_task", "prompt_variant"),
                  "harness_profile_task_prompt_variant": lambda r: _joint(r, "harness_profile", "semantic_task", "prompt_variant")}
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
    for grouping, key_fn in (("all", lambda _r: "all"), ("harness", lambda r: r.get("harness")),
                             ("harness_profile", lambda r: r.get("harness_profile"))):
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
    comparative_validity = _comparative_validity(state, runs)
    return {"schema_version": REPORT_SCHEMA_VERSION, "generator": {"name": REPORT_GENERATOR, "version": REPORT_GENERATOR_VERSION, "agent_bench_version": __version__},
            "experiment_id": state.experiment_id, "definition_digest": state.definition_digest, "expansion_digest": state.expansion_digest,
            "fixed_environment": environment,
            "completion": {**counts, "is_partial": counts["completed"] != counts["total"], "label": f"PARTIAL EXPERIMENT — {counts['completed']} / {counts['total']} completed" if counts["completed"] != counts["total"] else "COMPLETE EXPERIMENT"},
            "included_verified_run_ids": [item["run_id"] for item in runs if item["evidence_status"] == "verified"], "excluded_or_invalid_runs": excluded,
            "summary_row_count": len(summaries), "quality_notice": "Execution success is not task-quality or manual-review proof.",
            "comparative_validity": comparative_validity}


def _archival_manifest(state: ExperimentState, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema_version": "1.0.0", "experiment_id": state.experiment_id, "expansion_digest": state.expansion_digest,
            "runs": [{"run_id": item["run_id"], "sealed_artifact_manifest_sha256": item["artifact_manifest_sha256"], "persistent_result_ref": item["result_ref"], "persistent_result_commit": item["result_commit"], "expected_raw_bundle_filename": f"{item['run_id']}-raw-evidence.tar.zst", "expected_raw_bundle_sha256": None, "future_release_locations": []} for item in artifacts]}


def _presentation(
    source: Path,
    state: ExperimentState,
    rows: dict[str, list[dict[str, Any]]],
    *,
    summary_environment: dict[str, str | None],
    definition_presentation: dict[str, Any],
    source_by_run: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Build the safe, rich payload consumed by the offline dashboard."""
    by_run = {row["run_id"]: row for row in rows["runs"]}
    metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    timing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    requests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    markers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for name, destination in (("metrics", metrics), ("timing", timing), ("requests", requests), ("tools", tools), ("markers", markers)):
        for row in rows[name]:
            destination[row["run_id"]].append(row)
    artifacts = {row["run_id"]: row for row in rows["artifacts"]}
    prompts = {item["prompt_id"]: item for item in definition_presentation.get("prompts", [])}
    details: dict[str, dict[str, Any]] = {}
    for run_id, run in by_run.items():
        detail: dict[str, Any] = {
            "identity": run,
            "prompt": prompts.get(run.get("prompt_id")),
            "metrics": sorted(metrics[run_id], key=lambda item: (item["metric_group"], item["metric_name"])),
            "timing": sorted(timing[run_id], key=lambda item: item["timing_name"]),
            "requests": sorted(requests[run_id], key=lambda item: item["request_index"]),
            "tools": sorted(tools[run_id], key=lambda item: (item.get("elapsed_seconds") is None, item.get("elapsed_seconds") or 0, item["event_id"])),
            "markers": sorted(markers[run_id], key=lambda item: (item.get("elapsed_seconds") is None, item.get("elapsed_seconds") or 0)),
            "artifacts": artifacts.get(run_id),
            "manual_review": "NOT REVIEWED — manual application quality is not an M9C metric.",
            "context_overhead": _context_overhead(requests[run_id]),
            "context_components": {
                "availability": "unavailable",
                "reason": "exact_component_tokenization_not_available",
                "note": "System, harness, tool-schema, skills, and history token components are not heuristically decomposed.",
            },
        }
        if run.get("evidence_status") == "verified":
            artifact_root = (source_by_run or {}).get(run_id, source) / "artifacts" / run_id
            manifest = json.loads((artifact_root / "run" / "manifest.json").read_text(encoding="utf-8"))
            detail["capture_capabilities"] = _sanitize_public(manifest.get("capture_capabilities", {}))
            detail["invocation"] = _captured_invocation(artifact_root, str(run.get("harness") or ""))
        details[run_id] = detail
    counts = _state_counts(state)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generator": {"name": REPORT_GENERATOR, "version": REPORT_GENERATOR_VERSION, "agent_bench_version": __version__},
        "experiment_id": state.experiment_id,
        "definition_digest": state.definition_digest,
        "expansion_digest": state.expansion_digest,
        "completion": {**counts, "is_partial": counts["completed"] != counts["total"]},
        "comparative_validity": _comparative_validity(state, rows["runs"]),
        "summary_environment": summary_environment,
        "definition": definition_presentation,
        "runs": rows["runs"],
        "chart_series_labels": concise_chart_series_labels(rows["runs"]),
        # Empty planned-only aggregates remain in Parquet/DuckDB for a complete
        # matrix record but add no information to the interactive dashboard.
        "summaries": [row for row in rows["summaries"] if row["n_available"] > 0],
        "curves": rows["curves"],
        "markers": rows["markers"],
        "failures": rows["failures"],
        "details": {run_id: detail for run_id, detail in details.items() if detail["identity"].get("evidence_status") == "verified"},
        "data_files": [
            "summary.json", "presentation.json", "charts.json", "raw-archival-manifest.json",
            "agent-bench.duckdb", "parquet/runs.parquet", "parquet/metrics.parquet",
            "parquet/requests.parquet", "parquet/context_points.parquet", "parquet/tools.parquet",
            "parquet/timing.parquet", "parquet/artifacts.parquet", "parquet/summaries.parquet",
            "report-manifest.json", "checksums.sha256",
        ],
    }


def _context_overhead(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only explicit pre-task requests; never infer token components."""
    ordered = sorted(requests, key=lambda item: item["request_index"])
    first_task = next((item["request_index"] for item in ordered if item.get("purpose") == "task"), None)
    auxiliary = [item for item in ordered if first_task is not None and item["request_index"] < first_task]
    return {
        "first_real_task_request_index": first_task,
        "auxiliary_inference_requests_before_first_task": len(auxiliary) if first_task is not None else None,
        "auxiliary_input_tokens": _sum_known(auxiliary, "input_context_tokens"),
        "auxiliary_output_tokens": _sum_known(auxiliary, "output_tokens"),
    }


def _sum_known(items: list[dict[str, Any]], key: str) -> int | None:
    values = [item.get(key) for item in items]
    return sum(value for value in values if isinstance(value, int)) if all(isinstance(value, int) for value in values) else None


def _captured_invocation(artifact_root: Path, harness: str) -> dict[str, Any] | None:
    path = artifact_root / "run" / "harness-state" / harness / "invocation.json"
    if not path.is_file():
        return None
    try:
        invocation = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"availability": "unavailable", "reason": "invalid_sealed_invocation_json"}
    argv = list(invocation.get("argv", []))
    for index, value in enumerate(argv):
        if value in {"--oneshot", "--prompt"} and index + 1 < len(argv):
            argv[index + 1] = "<exact prompt shown above>"
    environment = invocation.get("environment", {})
    return {
        "availability": "available",
        "argv": _sanitize_public(argv),
        "profile_digest": invocation.get("profile_digest"),
        "prompt_delivery": invocation.get("prompt_delivery"),
        "prompt_byte_length": invocation.get("prompt_byte_length"),
        "prompt_sha256": invocation.get("prompt_sha256"),
        "run_seed": invocation.get("run_seed"),
        "environment_keys": sorted(environment) if isinstance(environment, dict) else [],
        "isolation": "HOME/XDG and harness state are isolated; host values are deliberately redacted.",
    }


def _html_report(summary: dict[str, Any], presentation: dict[str, Any]) -> str:
    """Create a dependency-free explorer over deterministic, sanitized data."""
    data = json.dumps(presentation, ensure_ascii=False, sort_keys=True, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(str(summary["experiment_id"]))
    document = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Agent Bench — __TITLE__</title>
  <style>
    :root{--ink:#e7edf6;--muted:#a9b7ca;--bg:#0c1220;--panel:#141d2e;--panel2:#101827;--line:#2c3a50;--accent:#5eead4;--blue:#60a5fa;--warn:#fbbf24;--bad:#fb7185;--good:#86efac}
    *{box-sizing:border-box} html{scroll-behavior:smooth} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    main{max-width:1500px;margin:auto;padding:24px} header{padding:22px 0 14px;border-bottom:1px solid var(--line)} h1{margin:2px 0 4px;font-size:clamp(28px,4vw,46px);letter-spacing:-.04em}.eyebrow{color:var(--accent);font-weight:800;letter-spacing:.14em;font-size:12px}.sub,.muted{color:var(--muted)} .notice{margin:18px 0;padding:13px 16px;border:1px solid #8a6411;background:#2e250f;color:#fde68a;border-radius:10px;font-weight:700}
    nav{display:flex;gap:8px;overflow:auto;padding:14px 0}nav a,.button{white-space:nowrap;color:var(--ink);text-decoration:none;border:1px solid var(--line);background:var(--panel2);padding:6px 9px;border-radius:7px;font-size:12px;cursor:pointer}.button.active{border-color:var(--accent);color:var(--accent)}
    section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}h2{margin:0 0 5px;font-size:23px}h3{margin:16px 0 8px;font-size:17px}.kpis,.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px}.kpi,.card{background:var(--panel2);border:1px solid var(--line);padding:13px;border-radius:9px}.kpi .v{display:block;font-size:25px;font-weight:800}.kpi .l{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.card{cursor:pointer}.card:hover{border-color:var(--blue)}.card h3{margin:0 0 6px}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:2px 7px;font-size:11px;color:var(--muted);margin:2px}.ok{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
    table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{color:#c6d2e2;background:#111a2a;position:sticky;top:0}.sortable-header{display:flex;align-items:center;gap:4px;width:100%;padding:0;border:0;background:transparent;color:inherit;font:inherit;font-weight:700;text-align:left;cursor:pointer}.sortable-header:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.sort-indicator{min-width:1em;color:var(--accent)} .table-wrap{overflow:auto;max-height:440px;border:1px solid var(--line);border-radius:8px} code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0;background:#09101c;padding:12px;border-radius:8px;border:1px solid var(--line)}
    details{border:1px solid var(--line);border-radius:8px;padding:9px 11px;margin:10px 0;background:#101827}summary{cursor:pointer;font-weight:700}.filters{display:flex;align-items:end;flex-wrap:wrap;gap:10px;margin:13px 0}.filters label{display:grid;gap:3px;color:var(--muted);font-size:12px}select{background:#09101c;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px}.mode{display:flex;gap:5px;flex-wrap:wrap}.chart{margin:13px 0;padding:12px;background:#0c1421;border:1px solid var(--line);border-radius:8px}.chart svg{width:100%;min-width:740px;height:auto;display:block}.chart-scroll{overflow-x:auto}.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend-item{border:1px solid transparent;background:transparent;color:inherit;border-radius:5px;padding:3px 5px;text-align:left;cursor:pointer}.legend-item:focus-visible{outline:2px solid var(--accent);outline-offset:2px}.legend-item.series-active{color:var(--ink);border-color:var(--accent);background:#172a36}.legend-item.series-muted{opacity:.4}.series-line{transition:stroke-width .12s,opacity .12s}.series-hit{cursor:pointer;pointer-events:stroke}.chart.series-focused .series-line.series-muted{opacity:.18}.chart.series-focused .series-line.series-active{stroke-width:6}.swatch{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:4px}.bar-chart{display:grid;gap:5px}.bar{display:grid;grid-template-columns:minmax(120px,1fr) 2fr 76px;gap:8px;align-items:center;font-size:12px}.bar i{height:14px;border-radius:4px;background:var(--blue);display:block}.run-detail{min-height:160px}.source{font-size:12px;color:var(--accent)}.empty{color:var(--muted);padding:12px}.two{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}@media(max-width:780px){main{padding:13px}.two{grid-template-columns:1fr}.chart svg{min-width:650px}}
  </style>
</head>
<body><main>
  <header id="overview"><div class="eyebrow">AGENT BENCH</div><h1 id="headline">Benchmark report</h1><div id="header-sub" class="sub"></div><div id="partial" class="notice"></div></header>
  <nav aria-label="Report sections"><a href="#variant-comparison">Variant comparison</a><a id="functional-nav" href="#functional">Functional results</a><a href="#comparison">Matched comparisons</a><a href="#context">Context</a><a href="#prompts">Tasks + prompts</a><a href="#overview">Overview</a><a href="#explorer">Executed runs</a><a href="#provenance">Provenance</a><a href="#data">Data</a></nav>
  <section id="variant-comparison"><h2>Variant Comparison</h2><p class="muted">Select one deterministic metric to compare harness/profile variants. This is not an overall efficiency score or quality ranking. Prompt variants are kept as matched strata; Prompt = All aggregates matched observations within each exact prompt SHA and seed, not pooled raw workloads.</p><div id="variant-controls" class="filters"></div><div id="variant-chart"></div></section>
  <section id="functional" hidden><h2>Functional Results</h2><p class="muted">Correctness evidence is shown separately from reasoning and runtime measurements. A completed benchmark process is not itself a functional pass.</p><div id="functional-results" class="grid"></div></section>
  <section id="comparison"><h2>Matched same-seed comparisons</h2><p class="muted">RAW RUN METRICS and MATCHED SEED / PAIRED PROFILE EFFECTS are distinct. Median/Q1/Q3 are Type 7 summaries only when multiple values are available; N=1 intentionally has no variability claim. These comparisons contain no winner or quality judgement. Prompt variants may produce different implementations and are not directly efficiency-comparable unless functional equivalence has been established.</p><div id="comparison-controls" class="filters"></div><div id="comparison-charts"></div><details><summary>Raw-run deterministic comparison summary table</summary><p class="muted">Click a column heading to sort the currently selected group. Sort numeric metric columns ascending to find the lowest measured time or resource use; sort delta columns in either direction to find the largest measured reduction or increase. These are metric observations, not quality rankings.</p><div id="comparison-table"></div></details><details open><summary>Matched seed / paired profile effects</summary><p class="muted">Every absolute delta is candidate minus reference. Direction labels describe deterministic behavior or resource observations only; they are not quality or overall agent-performance wins.</p><div id="matched-comparison-table"></div><h3>Paired aggregate effects</h3><div id="matched-summary-table"></div></details></section>
  <section id="context"><h2>Context Behavior</h2><p>Context values are observed at the API boundary where available. Task time starts at the first real task inference request; auxiliary inference is retained separately. Marker triangles are observed event timing, not inferred harness execution timing.</p><div id="context-charts"></div></section>
  <section id="prompts"><h2>Task / Prompt Specificity</h2><p>Variants are byte-exact benchmark inputs for the same semantic task, not quality grades.</p><div id="prompt-comparison"></div></section>
  <section id="failures"><h2>Failures / Interruptions</h2><p class="muted">Pending or unexecuted matrix rows are deliberately excluded from this section.</p><div id="failure-table"></div></section>
  <section id="pending"><h2>Pending / Planned</h2><p class="muted">These rows have not executed. They do not represent a benchmark failure or unavailable measurement.</p><div id="pending-table"></div></section>
  <section><h2>Overview / experiment metadata</h2><p>Execution and preservation facts only. <strong>Success is not task correctness</strong>; manual application quality is not reviewed here.</p><div id="kpis" class="kpis"></div><div id="matrix-summary" class="two"></div><details><summary>Fixed model, llama.cpp backend, hardware, server, template, and generation configuration</summary><div id="fixed-config"></div></details></section>
  <section id="explorer"><h2>Executed Runs</h2><p class="muted">Individual evidence remains available below; it is collapsed by default so the report opens on analysis.</p><details><summary id="executed-summary">Show individual runs</summary><div id="filters" class="filters"></div><div id="run-list" class="grid"></div><h3>Selected run detail</h3><div id="run-detail" class="run-detail"></div></details></section>
  <section id="provenance"><h2>Environment + Provenance</h2><div id="provenance-content"></div></section>
  <section id="data"><h2>Raw Tables / Data</h2><p>These are deterministic, sealed derived-report files. Raw proxy bodies, raw reasoning, personal host paths, and secrets are not embedded in this dashboard.</p><div id="data-files"></div></section>
</main><script id="agent-bench-data" type="application/json">__DATA__</script><script>
(() => {
  const d=JSON.parse(document.getElementById('agent-bench-data').textContent), $=id=>document.getElementById(id), missing='Not recorded (no sealed evidence)', esc=v=>String(v??missing).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])), num=v=>v===null||v===undefined?missing:Number(v).toLocaleString(undefined,{maximumFractionDigits:3}), val=v=>v===null||v===undefined?missing:(typeof v==='number'?num(v):esc(v));
  const colors=['#60a5fa','#f472b6','#5eead4','#fbbf24','#c4b5fd','#fb923c']; let selected=null, mode='executed', filters={experiment:'all',harness:'all',profile:'all',reasoning:'all',task:'all',prompt:'all',repetition:'all',seed:'all',functional:'all',hardGate:'all',tier:'all',minimumScore:''}, tableCounter=0; const tableSorts={},barSorts={},variantState={metric:'timing.wall_time_seconds.value',sort:'lowest',reference:'',display:'absolute'};
  const config=d.definition.backend_configuration||d.definition.fixed_environment||{}, profileLabel=id=>({"hermes-default-v1":"xhigh","hermes-reasoning-medium-v1":"medium","hermes-reasoning-low-v1":"low","hermes-reasoning-off-v1":"off"}[id]||id||missing);
  const run=(id)=>d.runs.find(x=>x.run_id===id), completed=()=>d.runs.filter(x=>x.state==='completed'&&x.evidence_status==='verified'), reasoning=r=>{const profile=(d.definition.profiles||[]).find(x=>x.profile_id===r.harness_profile),fields=profile?.settings?.reasoning_request_fields;return fields?.reasoning_effort||profile?.settings?.reasoning||profileLabel(r.harness_profile)},matchesFilters=r=>(filters.experiment==='all'||r.experiment_id===filters.experiment)&&(filters.harness==='all'||r.harness===filters.harness)&&(filters.profile==='all'||r.harness_profile===filters.profile)&&(filters.reasoning==='all'||String(reasoning(r))===filters.reasoning)&&(filters.task==='all'||r.semantic_task===filters.task)&&(filters.prompt==='all'||r.prompt_variant===filters.prompt)&&(filters.repetition==='all'||String(r.repetition)===filters.repetition)&&(filters.seed==='all'||String(r.seed)===filters.seed)&&(filters.functional==='all'||r.functional_validation_status===filters.functional)&&(filters.hardGate==='all'||String(r.hard_gate_pass)===filters.hardGate)&&(filters.tier==='all'||r.functional_tier===filters.tier)&&(!filters.minimumScore||Number(r.functional_score_percent)>=Number(filters.minimumScore));
  function flat(obj,prefix=''){const out=[];if(Array.isArray(obj)){if(obj.every(x=>x===null||typeof x!=='object'))out.push([prefix,obj.join(', ')]);else obj.forEach((x,i)=>out.push(...flat(x,`${prefix}[${i}]`)));}else if(obj&&typeof obj==='object'){Object.keys(obj).sort().forEach(k=>out.push(...flat(obj[k],prefix?prefix+'.'+k:k)));}else out.push([prefix,obj]);return out}
  function kv(obj){const rows=flat(obj).map(([k,v])=>`<tr><th>${esc(k)}</th><td class="mono">${val(v)}</td></tr>`).join('');return `<div class="table-wrap"><table><tbody>${rows||'<tr><td>Not recorded (no sealed evidence)</td></tr>'}</tbody></table></div>`}
  const numericColumns=new Set(['execution_index','canonical_matrix_index','repetition','seed','n_available','n_planned','median','q1','q3','minimum','maximum','value','wall_time_seconds','llm_time_seconds','input_tokens','output_tokens','total_tokens','llm_requests','tool_calls','first_task_context_tokens','peak_context_tokens','context_growth_from_first_task_tokens','reference_value','candidate_value','absolute_delta','relative_delta_percent']);
  function table(items,cols,options={}){if(!items.length)return '<p class="empty">None.</p>';const sortable=options.sortable??false,id=options.id||`table-${++tableCounter}`,type=c=>c[3]||(numericColumns.has(c[0])?'number':'text'),raw=(x,c)=>x[c[0]],sortValue=(x,c)=>{const v=raw(x,c);return v===null||v===undefined?'':String(v)};return `<div class="table-wrap"><table${sortable?` data-sortable-table="${esc(id)}"`:''}><thead><tr>${cols.map(c=>sortable?`<th><button type="button" class="sortable-header" data-sort-key="${esc(c[0])}" data-sort-type="${type(c)}" aria-sort="none">${esc(c[1])}<span class="sort-indicator" data-sort-indicator aria-hidden="true"></span></button></th>`:`<th>${esc(c[1])}</th>`).join('')}</tr></thead><tbody>${items.map((x,index)=>`<tr data-sort-row-index="${index}">${cols.map(c=>`<td data-sort-value="${esc(sortValue(x,c))}" data-sort-missing="${raw(x,c)===null||raw(x,c)===undefined?'true':'false'}">${c[2]?c[2](x):val(raw(x,c))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
  function parseSortNumber(value){const parsed=Number(String(value).trim().replace(/,/g,'').replace(/%$/,''));return Number.isFinite(parsed)?parsed:NaN}
  function applyTableSort(table,key,direction){const headers=[...table.querySelectorAll('.sortable-header')],column=headers.findIndex(h=>h.dataset.sortKey===key);if(column<0)return false;const type=headers[column].dataset.sortType,rows=[...table.tBodies[0].rows];rows.sort((a,b)=>{const av=a.cells[column].dataset.sortValue,bv=b.cells[column].dataset.sortValue,am=a.cells[column].dataset.sortMissing==='true'||(type==='number'&&!Number.isFinite(parseSortNumber(av))),bm=b.cells[column].dataset.sortMissing==='true'||(type==='number'&&!Number.isFinite(parseSortNumber(bv)));if(am!==bm)return am?1:-1;if(am)return Number(a.dataset.sortRowIndex)-Number(b.dataset.sortRowIndex);const compared=type==='number'?parseSortNumber(av)-parseSortNumber(bv):av.localeCompare(bv,undefined,{numeric:true,sensitivity:'base'});return compared===0?Number(a.dataset.sortRowIndex)-Number(b.dataset.sortRowIndex):(direction==='asc'?compared:-compared)});rows.forEach(row=>table.tBodies[0].append(row));headers.forEach(header=>{const active=header.dataset.sortKey===key;header.setAttribute('aria-sort',active?(direction==='asc'?'ascending':'descending'):'none');header.querySelector('[data-sort-indicator]').textContent=active?(direction==='asc'?'↑':'↓'):''});return true}
  function wireSortableTable(root,id){const table=root.querySelector(`[data-sortable-table="${id}"]`);if(!table)return;const saved=tableSorts[id];if(saved&&!applyTableSort(table,saved.key,saved.direction))delete tableSorts[id];table.querySelectorAll('.sortable-header').forEach(header=>header.addEventListener('click',()=>{const previous=tableSorts[id],key=header.dataset.sortKey,direction=previous?.key===key&&previous.direction==='asc'?'desc':'asc';tableSorts[id]={key,direction};applyTableSort(table,key,direction)}))}
  $('headline').textContent=`Agent Bench — ${d.experiment_id}`;
  $('header-sub').innerHTML=`<span class="mono">${esc(d.experiment_id)}</span> · code/report generator <span class="mono">${esc(d.generator.name)} ${esc(d.generator.version)} / Agent Bench ${esc(d.generator.agent_bench_version)}</span><br>Expansion digest <span class="mono">${esc(d.expansion_digest)}</span>`;
  const c=d.completion; $('partial').textContent=c.is_partial?`PARTIAL EXPERIMENT — ${c.completed} / ${c.total} completed; remaining planned rows are not failures.`:`COMPLETE EXPERIMENT — ${c.completed} / ${c.total} completed.`;
  const promptCount=(d.definition.prompts||[]).length, taskCount=new Set((d.definition.prompts||[]).map(x=>x.semantic_task_id)).size, harnessCount=(d.definition.harnesses||[]).length;
  const env=d.definition.fixed_environment||{}, model=env.model||{}, backend=env.backend||{}, hardware=env.hardware||{}, server=(config.server||env.server_parameters||{}), generation=(config.sampling||env.generation||{});
  const repetitionLabel=(d.definition.repetition_indices||[]).length?(d.definition.repetition_indices||[]).map(x=>`R${String(x).padStart(3,'0')}`).join(', '):(d.definition.repetitions??missing); const kpis=[['Completed',`${c.completed}/${c.total}`],['Failed',c.failed],['Interrupted',c.interrupted],['Invalid',c.invalid],['Pending',c.pending],['Harnesses',harnessCount],['Tasks',taskCount],['Prompt variants',promptCount],['Repetitions',repetitionLabel],['Model',model.name||missing],['Backend',backend.implementation||'llama.cpp'],['GPU',hardware.gpu_model||missing],['Context size',server.context_size??generation.context_size??missing]];
  $('kpis').innerHTML=kpis.map(([l,v])=>`<div class="kpi"><span class="v">${val(v)}</span><span class="l">${esc(l)}</span></div>`).join('');
  const done=completed(); $('executed-summary').textContent=`Show individual runs (${done.length})`;
  function renderFunctional(){const rows=d.runs.filter(r=>r.functional_validation_status&&r.functional_validation_status!=='not_applicable');const section=$('functional'),nav=$('functional-nav');if(!rows.length){section.hidden=true;nav.hidden=true;return}section.hidden=false;nav.hidden=false;$('functional-results').innerHTML=rows.map(r=>{const status=String(r.functional_validation_status||'unavailable').toUpperCase(),gate=r.hard_gate_pass===true?'PASS':r.hard_gate_pass===false?'FAIL':'UNAVAILABLE',score=r.functional_score_numerator!==null&&r.functional_score_denominator!==null?`${num(r.functional_score_numerator)} / ${num(r.functional_score_denominator)} (${num(r.functional_score_percent)}%)`:missing;return `<article class="card" data-run="${esc(r.run_id)}"><h3 class="${status==='PASS'?'ok':status==='FAIL'||status==='ERROR'?'bad':'warn'}">Functional ${esc(status)}</h3><p><strong>${esc(r.harness)} · ${esc(r.semantic_task)}</strong><br>${esc(r.functional_scenario_id||missing)} · tier ${esc(r.functional_tier||missing)}</p><p>Score: ${score}<br>Hard gate: <strong>${gate}</strong><br>Baseline regressions: ${num(r.baseline_regression_count)}<br>Failed functional tests: ${num(r.failed_functional_test_count)}</p><p class="mono muted">${esc(r.run_id)}</p></article>`}).join('');$('functional-results').querySelectorAll('[data-run]').forEach(e=>e.onclick=()=>openDetail(e.dataset.run))}renderFunctional();
  const taskNames=[...new Set((d.definition.prompts||[]).map(x=>x.semantic_task_id))], variants=[...new Set((d.definition.prompts||[]).map(x=>x.variant_label))];
  $('matrix-summary').innerHTML=`<div><h3>Matrix dimensions</h3>${kv({harnesses:(d.definition.harnesses||[]).map(x=>`${x.display_name} ${x.version}`).join(', '),profiles:(d.definition.profiles||[]).map(x=>`${x.profile_id} (${x.profile_version})`).join(', '),semantic_tasks:taskNames.join(', '),prompt_variants:variants.join(', '),repetitions:repetitionLabel,repetition_indices:d.definition.repetition_indices||null,ordering:d.definition.ordering?.mode,ordering_seed:d.definition.ordering?.seed,planned_rows:c.total})}</div><div><h3>Immutable baseline and run policy</h3>${kv({portable_baseline:d.definition.portable_baseline,run_limits:d.definition.run_limits,fixed_environment_id:env.fixed_environment_id,definition_source:d.definition.definition_source,backend_configuration_source:d.definition.backend_configuration_source,backend_configuration_sha256:d.definition.backend_configuration_sha256})}</div>`;
  $('fixed-config').innerHTML=`<p class="source">Sources: ${esc(d.definition.definition_source||'unavailable')}; ${esc(d.definition.backend_configuration_source||'unavailable')} (path values redacted).</p><div class="two"><div><h3>Model + template + backend identity</h3>${kv({model:config.model||model,chat_template:config.chat_template||model.gguf_metadata,executable:config.executable||backend,llama_cpp_commit:config.llama_cpp_commit||backend.commit,version:backend.version,hardware:config.gpu||hardware})}</div><div><h3>Server + generation (all configured values)</h3>${kv({server:config.server||env.server_parameters,sampling:config.sampling||env.generation,restart_policy:config.restart_policy||env.restart_policy,readiness_policy:env.readiness_policy,warmup:env.warmup})}</div></div>`;
  const metricFields={wall_time_seconds:'wall_time_seconds',input_tokens:'input_tokens',output_tokens:'output_tokens',llm_requests:'llm_requests',tool_calls:'tool_calls',first_task_context_tokens:'first_task_context_tokens',peak_context_tokens:'peak_context_tokens',context_growth_from_first_task_tokens:'context_growth_from_first_task_tokens',files_changed:'files_changed'};
  function type7(values,p){const ordered=[...values].sort((a,b)=>a-b);if(!ordered.length)return null;if(ordered.length===1)return ordered[0];const h=(ordered.length-1)*p,low=Math.floor(h),fraction=h-low;return ordered[low]+fraction*(ordered[low+1]-ordered[low])}
  function comparisonKey(run,group){const parts={harness:run.harness,harness_profile:run.harness_profile,semantic_task:run.semantic_task,prompt_variant:run.prompt_variant,repetition:run.repetition};if(group==='harness_task')return `${run.harness} · ${run.semantic_task}`;if(group==='harness_profile_task')return `${run.harness_profile} · ${run.semantic_task}`;if(group==='harness_prompt_variant')return `${run.harness} · ${run.prompt_variant}`;if(group==='harness_profile_prompt_variant')return `${run.harness_profile} · ${run.prompt_variant}`;if(group==='harness_profile_task_prompt_variant')return `${run.harness_profile} · ${run.semantic_task} · ${run.prompt_variant}`;return parts[group]}
  function comparisonSummaryRows(group){return Object.entries(metricFields).flatMap(([metric,field])=>{const groups={};completed().filter(matchesFilters).forEach(run=>{const key=comparisonKey(run,group);if(key===null||key===undefined)return;(groups[key]??=[]).push(run)});return Object.entries(groups).map(([key,members])=>{const values=members.map(run=>run[field]).filter(Number.isFinite);return {grouping:group,group_key:key,metric_name:metric,n_planned:members.length,n_completed:members.length,n_available:values.length,median:type7(values,.5),q1:values.length>1?type7(values,.25):null,q3:values.length>1?type7(values,.75):null,minimum:values.length?Math.min(...values):null,maximum:values.length?Math.max(...values):null}})})}
  function groupSummary(group,metric){return comparisonSummaryRows(group).filter(x=>x.metric_name===metric)}
  function orderBarRows(rows,order){const copy=[...rows];if(order==='original')return copy;return copy.sort((left,right)=>{const a=left.median,b=right.median,am=!Number.isFinite(a),bm=!Number.isFinite(b);if(am!==bm)return am?1:-1;if(am)return 0;return order==='ascending'?a-b:b-a})}
  function wireBarSorts(){document.querySelectorAll('[data-bar-sort]').forEach(control=>control.onchange=()=>{barSorts[control.dataset.barSort]=control.value;comparison()})}
  function bars(title,group,metric){const source=groupSummary(group,metric),id=`${group}:${metric}`,order=barSorts[id]||'original',rows=orderBarRows(source,order);if(!rows.length)return `<div class="chart"><h3>${esc(title)}</h3><p class="empty">No completed values after the current filters.</p></div>`;const max=Math.max(...rows.map(x=>x.maximum??x.median??0),1);return `<div class="chart" data-bar-chart="${esc(id)}"><h3>${esc(title)}</h3><label class="muted">Sort order <select data-bar-sort="${esc(id)}"><option value="original"${order==='original'?' selected':''}>Original</option><option value="ascending"${order==='ascending'?' selected':''}>Ascending</option><option value="descending"${order==='descending'?' selected':''}>Descending</option></select></label><div class="bar-chart">${rows.map((r,i)=>`<div class="bar" data-bar-category="${esc(r.group_key)}" data-bar-value="${r.median??''}"><span>${esc(r.group_key)} <span class="muted">N=${r.n_available}</span></span><i style="width:${100*(r.median??0)/max}%;background:${colors[i%colors.length]}"></i><strong>${num(r.median)}</strong><span class="muted">${r.n_available>1?`Q1–Q3 ${num(r.q1)}–${num(r.q3)}`:'individual only; no spread'}</span></div>`).join('')}</div></div>`}
  function matchedComparison(){const comparison=d.matched_comparison;if(!comparison){$('matched-comparison-table').innerHTML='<p class="empty">No matched multi-experiment comparison was requested.</p>';return}$('matched-comparison-table').innerHTML=table((comparison.matched_seed_comparisons||[]).flatMap(pair=>Object.entries(pair.metrics||{}).map(([metric,values])=>({...pair,metric,...values}))),[['reference_profile','Reference profile',row=>`<span title="${esc(row.reference_profile)}">${esc(profileLabel(row.reference_profile))}</span>`],['candidate_profile','Candidate profile',row=>`<span title="${esc(row.candidate_profile)}">${esc(profileLabel(row.candidate_profile))}</span>`],['semantic_task','Task'],['prompt_variant','Prompt variant'],['repetition','Repetition'],['seed','Seed'],['metric','Metric'],['reference_value','Reference value'],['candidate_value','Candidate value'],['absolute_delta','Absolute delta'],['relative_delta_percent','Relative delta %'],['direction','Direction']],{id:'matched-seed',sortable:true});wireSortableTable($('matched-comparison-table'),'matched-seed');$('matched-summary-table').innerHTML=table(comparison.aggregated_paired_effects||[],[['reference_profile','Reference profile',row=>`<span title="${esc(row.reference_profile)}">${esc(profileLabel(row.reference_profile))}</span>`],['candidate_profile','Candidate profile',row=>`<span title="${esc(row.candidate_profile)}">${esc(profileLabel(row.candidate_profile))}</span>`],['view','View'],['group_key','Task / stratum'],['metric','Metric'],['n_matched_pairs','N'],['n_available','Available'],['n_not_available','Unavailable'],['median_paired_delta','Median delta'],['mean_paired_delta','Mean delta']],{id:'matched-summary',sortable:true});wireSortableTable($('matched-summary-table'),'matched-summary')}
  const variantMetricLabels={'timing.wall_time_seconds.value':'Wall time','timing.llm_time_seconds.value':'LLM time','tokens.input_tokens_total.value':'Input tokens','tokens.output_tokens_total.value':'Output tokens','tokens.total_tokens.value':'Total tokens','behavior.llm_request_count.value':'LLM request count','behavior.tool_calls_total.value':'Tool calls','context.peak_context_tokens.value':'Peak context','context.net_context_growth_tokens.value':'Context growth','behavior.calls_before_first_edit.value':'Calls before first edit','behavior.requests_before_first_model_tool_call.value':'Prior responses before first tool-call turn','behavior.output_tokens_before_first_model_tool_call.value':'Output tokens in prior responses before first tool-call turn','behavior.reasoning_only_responses.value':'Reasoning-only response count','behavior.length_finished_responses.value':'Length-finished response count','behavior.length_finished_without_tool_call.value':'Length-finished response count without tool call','behavior.requests_before_first_model_edit_call.value':'Prior responses before first edit-call turn','behavior.output_tokens_before_first_model_edit_call.value':'Output tokens in prior responses before first edit-call turn','reasoning.reasoning_tokens_total.value':'Reasoning tokens total','reasoning.reasoning_tokens_before_first_tool.value':'Reasoning tokens before first tool','reasoning.reasoning_tokens_before_first_edit.value':'Reasoning tokens before first edit','reasoning.max_continuous_reasoning_tokens.value':'Max continuous reasoning tokens','reasoning.reasoning_block_count.value':'Reasoning blocks','reasoning.reasoning_chars_total.value':'Reasoning characters total','reasoning.max_continuous_reasoning_chars.value':'Max continuous reasoning characters','reasoning.reasoning_time_total_seconds.value':'Reasoning time','derived.reasoning_to_output_ratio.value':'Reasoning share of output'};
  const responseBeforeHelp='Zero means the first tool or edit call appeared in the first complete model response; it does not imply zero reasoning or text before that call within the response.';
  const titleCase=value=>String(value||missing).replace(/(^|[-_ ])([a-z])/g,(_,prefix,letter)=>`${prefix}${letter.toUpperCase()}`), variantMetricLabel=key=>variantMetricLabels[key]||key.replace(/\.value$/,'').replace(/[._]/g,' '), variantKey=row=>encodeURIComponent([row.harness,row.profile,row.reasoning_setting||''].join('\u001f')), variantSubject=row=>row.subject_id||row.project||d.definition.portable_baseline?.subject_id||d.definition.portable_baseline?.repository_id||'controlled-subject';
  function variantMatchesFilters(row){const nativeReasoning=row.reasoning_setting||profileLabel(row.profile);return (filters.experiment==='all'||row.experiment_id===filters.experiment)&&(filters.harness==='all'||row.harness===filters.harness)&&(filters.profile==='all'||row.profile===filters.profile)&&(filters.reasoning==='all'||String(nativeReasoning)===filters.reasoning)&&(filters.task==='all'||row.semantic_task===filters.task)&&(filters.prompt==='all'||row.prompt_variant===filters.prompt)&&(filters.repetition==='all'||String(row.repetition)===filters.repetition)&&(filters.seed==='all'||String(row.seed)===filters.seed)&&(filters.functional==='all'||row.functional_validation_status===filters.functional)&&(filters.hardGate==='all'||String(row.hard_gate_pass)===filters.hardGate)&&(filters.tier==='all'||row.functional_tier===filters.tier)&&(!filters.minimumScore||Number(row.functional_score_percent)>=Number(filters.minimumScore))}
  function variantRows(){return (d.matched_comparison?.raw_runs||[]).filter(variantMatchesFilters)}
  function variantDescriptors(rows){const first=new Map;rows.forEach(row=>{const key=variantKey(row);if(!first.has(key))first.set(key,row)});const labels=new Map;first.forEach((row,key)=>labels.set(key,`${titleCase(row.harness)} · ${profileLabel(row.profile)}`));const counts={};[...labels.values()].forEach(label=>counts[label]=(counts[label]||0)+1);return [...first].map(([key,row])=>({key,row,label:counts[labels.get(key)]>1?`${labels.get(key)} · ${row.profile}${row.reasoning_setting?` · ${row.reasoning_setting}`:''}`:labels.get(key)}))}
  function variantMetricKeys(rows){const keys=new Set;rows.forEach(row=>Object.entries(row.metrics||{}).forEach(([key,value])=>{if(Number.isFinite(value))keys.add(key)}));return [...keys].sort((left,right)=>{const a=variantMetricLabels[left]?0:1,b=variantMetricLabels[right]?0:1;return a-b||variantMetricLabel(left).localeCompare(variantMetricLabel(right))})}
  function variantCaseKey(row){return [variantSubject(row),row.semantic_task,row.prompt_sha256,row.repetition,row.seed].join('\u001f')}
  function variantMatchable(row){return Boolean(row.semantic_task&&row.prompt_sha256&&row.repetition!==null&&row.repetition!==undefined&&row.seed!==null&&row.seed!==undefined)}
  function variantSummaries(){const raw=variantRows(),rows=raw.filter(variantMatchable),descriptors=variantDescriptors(raw),keys=descriptors.map(item=>item.key),cases=new Map,missingByVariant=new Map(keys.map(key=>[key,0]));raw.filter(row=>!variantMatchable(row)).forEach(row=>missingByVariant.set(variantKey(row),(missingByVariant.get(variantKey(row))||0)+1));rows.forEach(row=>{const key=variantCaseKey(row),variants=cases.get(key)||new Map;if(!variants.has(variantKey(row)))variants.set(variantKey(row),row);cases.set(key,variants)});const reference=variantState.reference,stats=new Map(keys.map(key=>[key,{values:[],excluded_incomplete:0,metric_unavailable:0}]));for(const variants of cases.values()){for(const key of keys){const row=variants.get(key),candidate=row?.metrics?.[variantState.metric],referenceValue=reference?variants.get(reference)?.metrics?.[variantState.metric]:null,complete=reference?Number.isFinite(candidate)&&Number.isFinite(referenceValue):keys.every(name=>Number.isFinite(variants.get(name)?.metrics?.[variantState.metric])),stat=stats.get(key);if(!complete){if(row&&row.metrics&&Object.prototype.hasOwnProperty.call(row.metrics,variantState.metric)&&!Number.isFinite(candidate))stat.metric_unavailable++;else stat.excluded_incomplete++;continue}if(variantState.display==='delta')stat.values.push(candidate-referenceValue);else if(variantState.display==='relative'){if(referenceValue!==0)stat.values.push(100*(candidate-referenceValue)/referenceValue)}else stat.values.push(candidate)}}return descriptors.map(item=>{const stat=stats.get(item.key),observed=stat.values,missingProvenance=missingByVariant.get(item.key)||0;return {...item,values:observed,median:type7(observed,.5),q1:observed.length>1?type7(observed,.25):null,q3:observed.length>1?type7(observed,.75):null,n_common_matched:observed.length,n_excluded_incomplete:stat.excluded_incomplete,n_metric_unavailable:stat.metric_unavailable,n_missing_provenance:missingProvenance}})}
  function orderVariantRows(rows,order){const copy=[...rows];if(order==='original')return copy;return copy.sort((left,right)=>{const a=left.median,b=right.median,am=!Number.isFinite(a),bm=!Number.isFinite(b);if(am!==bm)return am?1:-1;if(am)return 0;return order==='lowest'?a-b:b-a})}
  function variantProvenance(row){return `Run: ${row.row.run_id}; harness: ${row.row.harness}; profile: ${row.row.profile}; effective setting: ${row.row.reasoning_setting||'not recorded'}; matched by subject, task, exact prompt SHA-256, repetition, and seed.`}
  function renderVariantComparison(){const target=$('variant-chart');if(!target)return;const raw=variantRows();if(!d.matched_comparison?.raw_runs?.length){target.innerHTML='<p class="empty">Variant comparison is available in a unified multi-experiment report with matched raw-run metadata.</p>';return}const descriptors=variantDescriptors(raw);if(variantState.reference&&!descriptors.some(item=>item.key===variantState.reference))variantState.reference='';if((variantState.display==='delta'||variantState.display==='relative')&&!variantState.reference){target.innerHTML='<p class="empty">Select a reference variant before displaying deltas.</p>';return}const rows=orderVariantRows(variantSummaries(),variantState.sort);if(!rows.length){target.innerHTML='<p class="empty">No variant observations match the current shared filters.</p>';return}const maximum=Math.max(...rows.map(row=>Math.abs(row.median||0)),1),isDelta=variantState.display!=='absolute',label=variantMetricLabel(variantState.metric),displayLabel=variantState.display==='absolute'?label:variantState.display==='delta'?`Absolute delta in ${label}`:`Relative delta % in ${label}`,scope=variantState.reference?'exact pairwise candidate/reference matching':'complete common strata across all displayed variants';target.innerHTML=`<div class="chart" data-variant-chart><h3>${esc(displayLabel)} by harness/profile variant</h3><p class="muted">${isDelta?'Reference = 0 '+(variantState.display==='relative'?'%':'')+'. ':''}Bars use ${scope}, matched by subject, task, exact prompt SHA-256, repetition, and seed. ${responseBeforeHelp}</p><div class="bar-chart">${rows.map((row,index)=>`<div class="bar" data-variant-category="${esc(row.key)}" data-variant-value="${row.median??''}" title="${esc(variantProvenance(row))}"><span>${esc(row.label)} <span class="muted">N=${row.n_common_matched} ${variantState.reference?'pairwise matched':'common matched strata'}</span></span><i style="width:${100*Math.abs(row.median||0)/maximum}%;background:${row.median<0?'#f472b6':colors[index%colors.length]}"></i><strong>${num(row.median)}${variantState.display==='relative'&&Number.isFinite(row.median)?'%':''}</strong><span class="muted">${row.n_common_matched>1?`Q1–Q3 ${num(row.q1)}–${num(row.q3)}`:'individual only; no spread'} · ${row.n_excluded_incomplete} excluded: incomplete ${variantState.reference?'pairwise':'across displayed variants'} · ${row.n_metric_unavailable} metric unavailable${row.n_missing_provenance?` · ${row.n_missing_provenance} missing matching provenance`:''}</span></div>`).join('')}</div></div>`}
  function variantFilterOptions(id,label,key,values,render=value=>value){return `<label>${label}<select id="${id}"><option value="all">All</option>${values.map(value=>`<option value="${esc(value)}"${String(filters[key])===String(value)?' selected':''}>${esc(render(value))}</option>`).join('')}</select></label>`}
  function setVariantControls(){const target=$('variant-controls');if(!target)return;const raw=(d.matched_comparison?.raw_runs||[]),metrics=variantMetricKeys(raw),variants=variantDescriptors(raw),subject=raw[0]?variantSubject(raw[0]):'controlled-subject',values=key=>[...new Set(raw.map(row=>row[key]).filter(value=>value!==null&&value!==undefined&&value!==''))];if(!metrics.includes(variantState.metric))variantState.metric=metrics[0]||'';target.innerHTML=raw.length?`<label>Project / subject<select disabled><option>${esc(subject)}</option></select></label>${variantFilterOptions('variant-task','Task','task',values('semantic_task'))}${variantFilterOptions('variant-prompt','Prompt','prompt',values('prompt_variant'))}${variantFilterOptions('variant-repetition','Repetitions','repetition',values('repetition'),value=>`R${String(value).padStart(3,'0')}`)}<label>Metric<select id="variant-metric">${metrics.map(metric=>`<option value="${esc(metric)}"${variantState.metric===metric?' selected':''}>${esc(variantMetricLabel(metric))}</option>`).join('')}</select></label><label>Sort<select id="variant-sort"><option value="lowest"${variantState.sort==='lowest'?' selected':''}>Lowest first</option><option value="highest"${variantState.sort==='highest'?' selected':''}>Highest first</option><option value="original"${variantState.sort==='original'?' selected':''}>Original</option></select></label><label>Reference<select id="variant-reference"><option value="">None</option>${variants.map(item=>`<option value="${esc(item.key)}"${variantState.reference===item.key?' selected':''}>${esc(item.label)}</option>`).join('')}</select></label><label>Display<select id="variant-display"><option value="absolute"${variantState.display==='absolute'?' selected':''}>Absolute value</option><option value="delta"${variantState.display==='delta'?' selected':''}>Absolute delta vs reference</option><option value="relative"${variantState.display==='relative'?' selected':''}>Relative delta % vs reference</option></select></label>`:'<p class="empty">No matched raw-run metadata is available.</p>';[['variant-metric','metric'],['variant-sort','sort'],['variant-reference','reference'],['variant-display','display']].forEach(([id,key])=>{const control=$(id);if(control)control.onchange=()=>{variantState[key]=control.value;renderVariantComparison()}});[['variant-task','task','f-task'],['variant-prompt','prompt','f-prompt'],['variant-repetition','repetition','f-rep']].forEach(([id,key,sharedId])=>{const control=$(id);if(control)control.onchange=()=>{filters[key]=control.value;const shared=$(sharedId);if(shared)shared.value=control.value;renderRuns();comparison();setVariantControls()}});renderVariantComparison()}
  function comparison(){const group=$('comparison-group')?.value||'harness', label={harness:'harness',harness_profile:'harness profile',semantic_task:'semantic task',prompt_variant:'prompt variant',repetition:'repetition',harness_task:'harness × task',harness_profile_task:'profile × task',harness_prompt_variant:'harness × prompt',harness_profile_prompt_variant:'profile × prompt',harness_profile_task_prompt_variant:'profile × task × prompt'}[group];const metricLabels={wall_time_seconds:'Wall time (s)',input_tokens:'Input tokens',output_tokens:'Output tokens',llm_requests:'LLM requests',tool_calls:'Tool calls',first_task_context_tokens:'First task context',peak_context_tokens:'Peak context',context_growth_from_first_task_tokens:'Context growth',files_changed:'Files changed'};$('comparison-charts').innerHTML=Object.entries(metricLabels).map(([m,l])=>bars(`${l} by ${label}`,group,m)).join('');wireBarSorts();const rows=comparisonSummaryRows(group);$('comparison-table').innerHTML=table(rows,[['group_key','Group'],['metric_name','Metric'],['n_available','N'],['median','Median'],['q1','Q1'],['q3','Q3'],['minimum','Min'],['maximum','Max']],{id:'comparison-summary',sortable:true});wireSortableTable($('comparison-table'),'comparison-summary');matchedComparison()}
  $('comparison-controls').innerHTML=`<label>Group comparisons<select id="comparison-group">${[['harness','Harness'],['harness_profile','Harness profile'],['semantic_task','Semantic task'],['prompt_variant','Prompt variant'],['repetition','Repetition'],['harness_task','Harness × task'],['harness_profile_task','Profile × task'],['harness_prompt_variant','Harness × prompt'],['harness_profile_prompt_variant','Profile × prompt'],['harness_profile_task_prompt_variant','Profile × task × prompt']].map(x=>`<option value="${x[0]}">${x[1]}</option>`).join('')}</select></label>`;$('comparison-group').onchange=comparison;comparison();
  function eligible(){return d.runs.filter(r=>{if(mode==='executed'&&!(r.state==='completed'&&r.evidence_status==='verified'))return false;if(mode==='failed'&&!['failed','interrupted','invalid'].includes(r.state)&&r.evidence_status!=='invalid')return false;if(mode==='pending'&&r.state!=='pending')return false;return matchesFilters(r)})}
  const seriesKey=id=>`series-${encodeURIComponent(id)}`,seriesLabel=id=>d.chart_series_labels?.[id]||(()=>{const r=run(id)||{},rep=Number.isInteger(r.repetition)?`R${String(r.repetition).padStart(3,'0')}`:'R?';return [r.harness_profile||'profile',r.semantic_task||'task unavailable',r.prompt_variant||'prompt unavailable',rep].join(' · ')})();
  function wireChartInteractions(root){let pinned=null,hovered=null;root.tabIndex=0;const idFor=target=>target?.closest?.('[data-series-id]')?.dataset.seriesId||null,items=()=>root.querySelectorAll('[data-series-id]');const update=()=>{const active=pinned||hovered;root.classList.toggle('series-focused',!!active);items().forEach(item=>{const match=!!active&&item.dataset.seriesId===active;item.classList.toggle('series-active',match);item.classList.toggle('series-muted',!!active&&!match);if(item.classList.contains('legend-item'))item.setAttribute('aria-pressed',String(pinned===item.dataset.seriesId));});};root.addEventListener('pointerover',event=>{const id=idFor(event.target);if(id&&!pinned){hovered=id;update();}});root.addEventListener('pointerout',event=>{const leaving=idFor(event.target),entering=idFor(event.relatedTarget);if(leaving&&leaving!==entering&&!pinned){hovered=null;update();}});root.addEventListener('click',event=>{if(event.target.closest('[data-clear-series]')){pinned=null;hovered=null;update();return;}const id=idFor(event.target);if(id){pinned=pinned===id?null:id;hovered=null;update();}});root.addEventListener('keydown',event=>{if(event.key==='Escape'){pinned=null;hovered=null;update();root.querySelector('[data-clear-series]')?.focus();}});}
  function chart(kind,title,xlabel){const rows=(d.curves||[]).filter(x=>x.curve_kind===kind&&eligible().some(r=>r.run_id===x.run_id)&&x.context_utilization_percent!==null&&x.context_utilization_percent!==undefined);if(!rows.length)return `<div class="chart"><h3>${title}</h3><p class="empty">No observable context points for the current executed-run filter.</p></div>`;const w=1000,h=440,L=75,R=30,T=35,B=60,maxX=Math.max(...rows.map(x=>x.x),1),sx=x=>L+(w-L-R)*x/maxX,sy=y=>h-B-(h-T-B)*Math.max(0,Math.min(100,y))/100;const by={};rows.forEach(x=>(by[x.run_id]??=[]).push(x));let lines='',legend='';Object.entries(by).sort().forEach(([id,pts],i)=>{pts.sort((a,b)=>a.x-b.x);const col=colors[i%colors.length],key=seriesKey(id),label=seriesLabel(id),points=pts.map(p=>`${sx(p.x).toFixed(2)},${sy(p.context_utilization_percent).toFixed(2)}`).join(' ');lines+=`<polyline id="${esc(key)}-line" class="chart-series series-line" data-series-id="${esc(key)}" data-run-id="${esc(id)}" fill="none" stroke="${col}" stroke-width="3" points="${points}"><title>${esc(id)} — full run identity</title></polyline><polyline class="chart-series series-hit" data-series-id="${esc(key)}" data-run-id="${esc(id)}" fill="none" stroke="transparent" stroke-width="18" points="${points}"><title>${esc(id)} — full run identity</title></polyline>${pts.map(p=>`<circle cx="${sx(p.x)}" cy="${sy(p.context_utilization_percent)}" r="4" fill="${col}"><title>${esc(id)} | ${xlabel}: ${num(p.x)} | context: ${num(p.context_utilization_percent)}%</title></circle>`).join('')}`;legend+=`<button class="legend-item" type="button" data-series-id="${esc(key)}" data-run-id="${esc(id)}" aria-controls="${esc(key)}-line" aria-pressed="false" title="${esc(id)} — full run identity"><i class="swatch" style="background:${col}"></i>${esc(label)}</button>`});let grid='';for(let y=0;y<=100;y+=20)grid+=`<line x1="${L}" y1="${sy(y)}" x2="${w-R}" y2="${sy(y)}" stroke="#2c3a50"/><text x="10" y="${sy(y)+4}" fill="#a9b7ca">${y}%</text>`;for(let x=0;x<=4;x++){const v=maxX*x/4;grid+=`<line x1="${sx(v)}" y1="${T}" x2="${sx(v)}" y2="${h-B}" stroke="#1d2a3c"/><text x="${sx(v)-10}" y="${h-35}" fill="#a9b7ca">${num(v)}</text>`}let ms='';if(kind==='absolute_elapsed_task_time')(d.markers||[]).filter(m=>eligible().some(r=>r.run_id===m.run_id)&&m.task_elapsed_seconds!==null&&m.task_elapsed_seconds<=maxX).forEach(m=>{const x=sx(m.task_elapsed_seconds);ms+=`<path d="M${x} ${h-B+10}l-6 10h12z" fill="#fbbf24"><title>${esc(m.marker_kind)}: ${esc(m.timing_semantics)} at ${num(m.task_elapsed_seconds)} s</title></path>`});const aggregate=(d.curves||[]).filter(x=>x.curve_kind===kind+'_aggregate'&&x.grouping==='harness'&&x.n_available>1&&(filters.harness==='all'||x.group_key===filters.harness));const ag={};aggregate.forEach(x=>(ag[x.group_key]??=[]).push(x));let bands='';Object.entries(ag).forEach(([key,pts],i)=>{pts.sort((a,b)=>a.x-b.x);const col=colors[(i+3)%colors.length],upper=pts.map(p=>`${sx(p.x)},${sy(p.q3)}`).join(' '),lower=[...pts].reverse().map(p=>`${sx(p.x)},${sy(p.q1)}`).join(' '),median=pts.map(p=>`${sx(p.x)},${sy(p.median)}`).join(' ');bands+=`<polygon points="${upper} ${lower}" fill="${col}" opacity=".18"><title>${esc(key)} Type 7 Q1–Q3 band (N=${pts[0].n_available})</title></polygon><polyline fill="none" stroke="${col}" stroke-dasharray="7 5" stroke-width="3" points="${median}"><title>${esc(key)} Type 7 median (N=${pts[0].n_available})</title></polyline>`;legend+=`<span><i class="swatch" style="background:${col}"></i>${esc(key)} median + Q1–Q3 (N=${pts[0].n_available})</span>`});return `<div class="chart" data-interactive-chart><h3>${title}</h3><p class="muted">Y: API-observed context utilization %. X: ${xlabel}. Hover a legend entry or line to inspect one series; click to pin, click again, clear selection, or press Escape to clear. Points and markers have hover details. Median/Q1/Q3 bands appear only for like-for-like normalized series with N&gt;1.</p><button class="button" type="button" data-clear-series>Clear series selection</button><div class="chart-scroll"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}">${grid}<line x1="${L}" y1="${h-B}" x2="${w-R}" y2="${h-B}" stroke="#a9b7ca"/><line x1="${L}" y1="${T}" x2="${L}" y2="${h-B}" stroke="#a9b7ca"/>${bands}${lines}${ms}<text x="${w/2-80}" y="${h-10}" fill="#a9b7ca">${xlabel}</text></svg></div><div class="legend">${legend}${kind==='absolute_elapsed_task_time'?'<span>▲ observed event marker</span>':''}</div></div>`}
  function renderContext(){$('context-charts').innerHTML=chart('absolute_elapsed_task_time','Context utilization vs absolute elapsed task time','task-relative seconds')+chart('normalized_elapsed_task_time','Context utilization vs normalized elapsed task time','normalized elapsed task time (%)')+chart('request_index','Context utilization vs real task request index','real task request index');$('context-charts').querySelectorAll('[data-interactive-chart]').forEach(wireChartInteractions)}
  function promptComparison(task=taskNames[0]){const options=taskNames.map(x=>`<option ${x===task?'selected':''}>${esc(x)}</option>`).join('');const ps=(d.definition.prompts||[]).filter(x=>x.semantic_task_id===task);$('prompt-comparison').innerHTML=`<label>Semantic task <select id="prompt-task">${options}</select></label><div class="grid">${ps.map(p=>`<article class="card"><h3>${esc(p.variant_label)}</h3><p class="mono">${esc(p.prompt_id)}<br>SHA256 ${esc(p.sha256)}<br>${p.byte_length} UTF-8 bytes</p><pre>${esc(p.content)}</pre></article>`).join('')}</div>`;$('prompt-task').onchange=e=>promptComparison(e.target.value)}promptComparison();
  function filterOptions(id,label,values,key){return `<label>${label}<select id="${id}"><option value="all">All</option>${values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}</select></label>`}function setFilters(){const rows=d.runs;$('filters').innerHTML=`<div class="mode">${[['executed','Executed'],['failed','Failed / interrupted / invalid'],['pending','Pending'],['all','All rows']].map(x=>`<button class="button ${mode===x[0]?'active':''}" data-mode="${x[0]}">${x[1]}</button>`).join('')}</div>${filterOptions('f-experiment','Experiment',[...new Set(rows.map(x=>x.experiment_id).filter(Boolean))])}${filterOptions('f-harness','Harness',[...new Set(rows.map(x=>x.harness).filter(Boolean))])}${filterOptions('f-profile','Profile',[...new Set(rows.map(x=>x.harness_profile).filter(Boolean))])}${filterOptions('f-reasoning','Reasoning setting',[...new Set(rows.map(reasoning).filter(Boolean))])}${filterOptions('f-task','Task',[...new Set(rows.map(x=>x.semantic_task).filter(Boolean))])}${filterOptions('f-prompt','Prompt variant',[...new Set(rows.map(x=>x.prompt_variant).filter(Boolean))])}${filterOptions('f-rep','Repetition',[...new Set(rows.map(x=>x.repetition).filter(x=>x!==null&&x!==undefined))])}${filterOptions('f-seed','Seed',[...new Set(rows.map(x=>x.seed).filter(x=>x!==null&&x!==undefined))])}${filterOptions('f-functional','Functional status',['pass','fail','error','unavailable','not_applicable'],'functional')}${filterOptions('f-hard-gate','Hard gate',['true','false'],'hardGate')}${filterOptions('f-tier','Functional tier',['easy','medium','complex'],'tier')}<label>Minimum functional score<input id="f-min-score" type="number" min="0" max="100" value="${esc(filters.minimumScore)}"></label>`;[['f-experiment','experiment'],['f-harness','harness'],['f-profile','profile'],['f-reasoning','reasoning'],['f-task','task'],['f-prompt','prompt'],['f-rep','repetition'],['f-seed','seed'],['f-functional','functional'],['f-hard-gate','hardGate'],['f-tier','tier']].forEach(([id,key])=>{const e=$(id);e.value=filters[key];e.onchange=()=>{filters[key]=e.value;renderRuns();comparison();setVariantControls()}});$('f-min-score').onchange=e=>{filters.minimumScore=e.target.value;renderRuns();comparison();setVariantControls()};$('filters').querySelectorAll('[data-mode]').forEach(e=>e.onclick=()=>{mode=e.dataset.mode;setFilters();renderRuns();comparison();setVariantControls()})}
  function renderRuns(){const rows=eligible();$('run-list').innerHTML=rows.length?rows.map(r=>`<article class="card" data-run="${esc(r.run_id)}"><h3>${esc(r.harness||'planned')} · ${esc(r.semantic_task||missing)}</h3><span class="pill">${esc(r.prompt_variant||missing)} · r${esc(r.repetition??missing)}</span><span class="pill">${esc(r.state)} / ${esc(r.evidence_status)}</span><p>${r.wall_time_seconds!==null?`${num(r.wall_time_seconds)} s · ${num(r.llm_requests)} requests · ${num(r.tool_calls)} tools`:'No execution metrics yet.'}</p><p>Functional: ${esc(r.functional_validation_status)}${r.functional_score_percent!==null&&r.functional_score_percent!==undefined?` · ${num(r.functional_score_percent)}% · hard gate ${esc(r.hard_gate_pass)}`:''}${r.baseline_regression_count!==null&&r.baseline_regression_count!==undefined?` · baseline regressions ${num(r.baseline_regression_count)}`:''}</p><p class="mono muted">#${esc(r.execution_index)} · ${esc(r.run_id)}</p></article>`).join(''):'<p class="empty">No rows match the current filters.</p>';$('run-list').querySelectorAll('[data-run]').forEach(e=>e.onclick=()=>openDetail(e.dataset.run));renderContext()}
  function openDetail(id){selected=id;const x=d.details[id]||{identity:run(id)},r=x.identity,p=x.prompt;const metricGroups={};(x.metrics||[]).forEach(m=>(metricGroups[m.metric_group]??=[]).push(m));const metricHtml=Object.entries(metricGroups).map(([g,items])=>`<details><summary>${esc(g)} metrics</summary>${table(items,[['metric_name','Metric'],['value','Value'],['units','Units'],['availability','Availability'],['unavailable_reason','Unavailable reason'],['method','Method']])}</details>`).join('');const timingHtml=table(x.timing||[],[['timing_name','Timing metric'],['value_seconds','Seconds'],['availability','Availability'],['semantics','Semantics'],['unavailable_reason','Unavailable reason'],['method','Method'],['source','Source']]);const requestHtml=table(x.requests||[],[['request_index','Request'],['purpose','Purpose'],['elapsed_seconds','Elapsed s'],['input_context_tokens','Context tokens'],['context_utilization_percent','Context %'],['output_tokens','Output tokens'],['delta_vs_first_task_tokens','Δ first task'],['request_body_sha256','Request hash'],['purpose_evidence','Classification evidence']]);const calls=(x.tools||[]).filter(t=>t.event_kind==='tool_call_start'||t.event_kind==='tool_call_end'), activity=(x.tools||[]).filter(t=>t.event_kind!=='tool_call_start'&&t.event_kind!=='tool_call_end');const callHtml=table(calls,[['event_kind','Boundary'],['tool_name','Tool'],['category','Category'],['outcome','Outcome'],['elapsed_seconds','Observed s'],['timing_semantics','Timing semantics'],['event_id','Event ID']]);const activityHtml=table(activity,[['event_kind','Activity'],['category','Category'],['outcome','Outcome'],['elapsed_seconds','Observed s'],['timing_semantics','Timing semantics'],['event_id','Event ID']]);$('run-detail').innerHTML=`<article class="card"><h3>${esc(r.run_id)}</h3><div class="two"><div>${kv({execution_index:r.execution_index,canonical_matrix_index:r.canonical_matrix_index,harness:r.harness,harness_profile:r.harness_profile,semantic_task:r.semantic_task,prompt_variant:r.prompt_variant,repetition:r.repetition,seed:r.seed,state:r.state,evidence_status:r.evidence_status,termination_class:r.termination_class})}</div><div>${kv({manual_review:x.manual_review,wall_time_seconds:r.wall_time_seconds,llm_requests:r.llm_requests,tool_calls:r.tool_calls,input_tokens:r.input_tokens,output_tokens:r.output_tokens,total_tokens:r.total_tokens,first_task_context_tokens:r.first_task_context_tokens,first_task_context_utilization_percent:r.first_task_context_utilization_percent,peak_context_tokens:r.peak_context_tokens,peak_context_utilization_percent:r.peak_context_utilization_percent,context_growth_from_first_task_tokens:r.context_growth_from_first_task_tokens,files_changed:r.files_changed,lines_added:r.lines_added,lines_deleted:r.lines_deleted})}</div></div>${p?`<details open><summary>Exact benchmark prompt — ${esc(p.prompt_id)} · SHA256 ${esc(p.sha256)}</summary><pre>${esc(p.content)}</pre></details>`:'<p class="empty">Exact prompt unavailable because the matching immutable definition was not supplied.</p>'}<details><summary>Harness profile + captured invocation</summary>${kv({profile:(d.definition.profiles||[]).find(q=>q.profile_id===r.harness_profile)||null,invocation:x.invocation||null})}</details><details><summary>Timing provenance</summary><p class="muted">Execution timing is unavailable unless native start/end evidence exists. Observed tool-event time and model tool-call emission remain distinct and are never treated as execution start.</p>${timingHtml}</details><details><summary>Context request sequence and auxiliary overhead</summary><p class="muted">System/harness/tool/skills component token attribution is displayed only if exact sealed analysis exposes it; otherwise it remains unavailable. No heuristic decomposition is made.</p>${kv({auxiliary_inference_overhead:x.context_overhead||null,component_attribution:x.context_components||null})}${requestHtml}</details><details><summary>Tool calls</summary>${callHtml}</details><details><summary>File / shell / test activity</summary>${activityHtml}</details><details><summary>Capture capabilities + preservation</summary>${kv({capture_capabilities:x.capture_capabilities||null,artifacts:x.artifacts||null})}</details></article>`;location.hash='explorer'}
  const baseOpenDetail=openDetail;openDetail=id=>{baseOpenDetail(id);const detail=d.details[id];if(!detail?.metrics?.length)return;const groups={};detail.metrics.forEach(m=>(groups[m.metric_group]??=[]).push(m));const extra=Object.entries(groups).map(([group,items])=>`<details><summary>${esc(group)} metrics</summary>${table(items,[['metric_name','Metric'],['value','Value'],['units','Units'],['availability','Availability'],['unavailable_reason','Unavailable reason'],['method','Method']])}</details>`).join('');$('run-detail article').insertAdjacentHTML('beforeend',extra)};
  const problems=d.failures||[];$('failure-table').innerHTML=table(problems,[['run_id','Run'],['state','State'],['failure_class','Failure class'],['failure_domain','Failure domain'],['failure_phase','Failure phase'],['detail','Detail'],['harness_execution_started','Harness began'],['llm_request_observed','LLM observed'],['preservation_completed','Evidence sealed'],['evidence_status','Evidence']]);const pending=d.runs.filter(x=>x.state==='pending');$('pending-table').innerHTML=table(pending,[['execution_index','Execution index'],['canonical_matrix_index','Matrix index'],['harness','Harness'],['harness_profile','Profile'],['semantic_task','Task'],['prompt_variant','Prompt'],['repetition','Rep'],['run_id','Run ID']]);
  $('provenance-content').innerHTML=`<p class="source">Fixed versus matrix configuration sources: ${esc(d.definition.definition_source||'unavailable')} and ${esc(d.definition.backend_configuration_source||'unavailable')}. Capture-capability and preservation rows are per-run sealed evidence.</p><details open><summary>Experiment / result provenance</summary>${kv({experiment_id:d.experiment_id,definition_digest:d.definition_digest,expansion_digest:d.expansion_digest,summary_environment:d.summary_environment,portable_baseline:d.definition.portable_baseline,profiles:(d.definition.profiles||[]).map(x=>({profile_id:x.profile_id,version:x.profile_version,upstream_defaults_source:x.upstream_defaults_source,deviations:x.deviations,source:x.source,source_sha256:x.source_sha256}))})}</details><details><summary>Harness identities</summary>${kv(d.definition.harnesses||[])}</details>`;
  $('data-files').innerHTML=d.data_files.map(f=>`<span class="pill"><a href="${esc(f)}" download>${esc(f)}</a></span>`).join(' ');setFilters();renderRuns();setVariantControls();
})();
</script></body></html>\n"""
    return document.replace("__TITLE__", title).replace("__DATA__", data)


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


def _comparative_validity(state: ExperimentState, runs: list[dict[str, Any]]) -> str:
    """Classify execution-evidence comparability, never task correctness."""
    counts = _state_counts(state)
    infrastructure_classes = {
        "benchmark_port_in_use", "conflicting_gpu_process", "precondition_failed",
        "backend_identity_mismatch", "model_hash_mismatch", "template_hash_mismatch",
        "backend_start_failed", "backend_readiness_failed", "preservation_failed",
    }
    systematic_infrastructure = any(
        row.get("failure_domain") in {"infrastructure_precondition", "backend_lifecycle", "proxy_lifecycle", "preservation"}
        or row.get("failure_class") in infrastructure_classes
        or row.get("termination_class") in infrastructure_classes
        for row in runs
    )
    invalid_evidence = any(row.get("evidence_status") == "invalid" for row in runs)
    unfinished = any(counts[key] for key in ("pending", "preflight", "running", "preserving", "analyzing"))
    ordinary_failure = bool(counts["failed"] or counts["interrupted"] or counts["invalid"])
    verified_completed = sum(
        row.get("state") == "completed" and row.get("evidence_status") == "verified"
        for row in runs
    )
    if systematic_infrastructure or invalid_evidence:
        return "invalid_for_comparative_interpretation"
    if not unfinished and not ordinary_failure and verified_completed == counts["total"]:
        return "complete_valid_for_comparative_interpretation"
    if unfinished and not ordinary_failure:
        return "partial_but_otherwise_healthy"
    if unfinished:
        return "partial_with_ordinary_run_failures"
    return "complete_with_ordinary_run_failures"


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
    return "not_recorded_no_sealed_evidence" if value is None else str(value)


def _round6(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
