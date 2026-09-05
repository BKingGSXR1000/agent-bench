"""Read-only comparison input for the Hermes reasoning screen.

This deliberately does not run a harness or write into a sealed run.  It
joins the existing Hermes-default R001 control evidence with completed rows of
the future reasoning-screen experiment and recalculates only deterministic
metrics in memory from their immutable source artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_bench.context_storage import verify_context_analysis_artifact
from agent_bench.executor import ExperimentState
from agent_bench.metrics import calculate_run_metrics
from agent_bench.metrics_storage import verify_metrics_artifact
from agent_bench.preservation import verify_artifact
from agent_bench.reporting import quantile_type7
from agent_bench.result_store import verify_published_result
from agent_bench.runner import RunManifest


class ReasoningScreenError(RuntimeError):
    """Raised when the requested control or candidate evidence is invalid."""


_CANDIDATE_SETTINGS = {
    "hermes-reasoning-off-v1": "off",
    "hermes-reasoning-low-v1": "low",
    "hermes-reasoning-medium-v1": "medium",
}


def build_reasoning_screen_comparison(
    *, control_root: Path, screen_root: Path,
) -> dict[str, Any]:
    """Return evidence-backed individual rows and Type-7 profile summaries.

    The default control is intentionally selected only from existing completed
    ``hermes-default-v1`` R001 artifacts.  Candidate rows must be completed in
    the screen state.  No absent result is treated as a zero or a failure.
    """
    controls = _control_rows(control_root.expanduser().resolve())
    candidates = _candidate_rows(screen_root.expanduser().resolve())
    if not controls:
        raise ReasoningScreenError("no completed Hermes default R001 control artifacts found")
    if not candidates:
        raise ReasoningScreenError("no completed Hermes reasoning-screen candidate artifacts found")
    rows = sorted(controls + candidates, key=lambda row: (row["profile_id"], row["prompt_id"], row["run_id"]))
    summaries: list[dict[str, Any]] = []
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_profile[row["profile_id"]].append(row)
    for profile_id, members in sorted(by_profile.items()):
        for metric in _SUMMARY_METRICS:
            values = [float(row[metric]) for row in members if row[metric] is not None]
            summaries.append({
                "profile_id": profile_id,
                "reasoning_setting": members[0]["reasoning_setting"],
                "metric_name": metric,
                "n_completed": len(members),
                "n_available": len(values),
                "median": quantile_type7(values, 0.5),
                "q1": quantile_type7(values, 0.25),
                "q3": quantile_type7(values, 0.75),
            })
    return {
        "schema_version": "1.0.0",
        "kind": "hermes-reasoning-screen-derived-comparison",
        "control": {"profile_id": "hermes-default-v1", "repetition": 1, "rerun": False},
        "candidate_profiles": _CANDIDATE_SETTINGS,
        "rows": rows,
        "profile_summaries": summaries,
        "interpretation": "deterministic behavior and resource observations only; no quality conclusion",
    }


_SUMMARY_METRICS = (
    "wall_time_seconds", "llm_requests", "input_tokens", "output_tokens",
    "peak_context_tokens", "tool_calls", "reasoning_only_responses",
    "length_finished_responses", "length_finished_without_tool_call",
    "requests_before_first_model_tool_call", "output_tokens_before_first_model_tool_call",
    "requests_before_first_model_edit_call", "output_tokens_before_first_model_edit_call",
)


def _control_rows(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ReasoningScreenError(f"control output root does not exist: {root}")
    rows: list[dict[str, Any]] = []
    for artifact in sorted((root / "artifacts").glob("hermes-hermes-default-v1-*-r001-*")):
        if not artifact.is_dir():
            continue
        row = _row(root, artifact)
        if row["profile_id"] == "hermes-default-v1":
            rows.append(row)
    return rows


def _candidate_rows(root: Path) -> list[dict[str, Any]]:
    try:
        state = ExperimentState.model_validate_json((root / "experiment-state.json").read_bytes())
    except Exception as exc:
        raise ReasoningScreenError(f"invalid reasoning-screen experiment state: {exc}") from exc
    if state.experiment_id != "pocket-ledger-v1-hermes-reasoning-screen-v1":
        raise ReasoningScreenError("screen root has an unexpected experiment identity")
    completed = {progress.run_id for progress in state.runs if progress.state == "completed"}
    rows: list[dict[str, Any]] = []
    for run_id in sorted(completed):
        artifact = root / "artifacts" / run_id
        row = _row(root, artifact)
        if row["profile_id"] not in _CANDIDATE_SETTINGS:
            raise ReasoningScreenError(f"completed row uses a non-screen profile: {row['profile_id']}")
        rows.append(row)
    return rows


def _row(experiment_root: Path, artifact: Path) -> dict[str, Any]:
    """Validate all three immutable evidence layers and extract scalar facts."""
    sealed = verify_artifact(artifact)
    verify_published_result(experiment_root, sealed)
    manifest = RunManifest.model_validate_json((artifact / "run" / "manifest.json").read_bytes())
    if sealed.run_id != manifest.run_id:
        raise ReasoningScreenError("artifact and run manifest IDs disagree")
    stored = verify_metrics_artifact(experiment_root / "analysis" / manifest.run_id / "metrics-v1")
    context = verify_context_analysis_artifact(experiment_root / "analysis" / manifest.run_id / "context-analysis-v2")
    if stored.manifest.source_artifact_manifest_sha256 != _sha(artifact / "manifest.json"):
        raise ReasoningScreenError("metrics artifact does not reference this sealed source")
    if context.source_artifact_manifest_sha256 != _sha(artifact / "manifest.json"):
        raise ReasoningScreenError("context analysis does not reference this sealed source")
    # The calculator is deterministic and does not write.  This exposes new
    # M11-preparation fields from old control raw evidence without altering its
    # sealed metrics-v1 artifact.
    metrics = calculate_run_metrics(artifact)
    value = lambda group, name: metrics.model_dump(mode="json")[group][name].get("value")
    setting = (
        "default / effective xhigh"
        if manifest.profile_id == "hermes-default-v1"
        else _CANDIDATE_SETTINGS[manifest.profile_id]
    )
    return {
        "run_id": manifest.run_id,
        "profile_id": manifest.profile_id,
        "reasoning_setting": setting,
        "prompt_id": manifest.prompt_id,
        "prompt_sha256": manifest.prompt_sha256,
        "repetition": _repetition(manifest.run_id),
        "wall_time_seconds": value("timing", "wall_time_seconds"),
        "llm_requests": value("behavior", "llm_request_count"),
        "input_tokens": value("tokens", "input_tokens_total"),
        "output_tokens": value("tokens", "output_tokens_total"),
        "peak_context_tokens": value("context", "peak_context_tokens"),
        "tool_calls": value("behavior", "tool_calls_total"),
        "reasoning_only_responses": value("behavior", "reasoning_only_responses"),
        "length_finished_responses": value("behavior", "length_finished_responses"),
        "length_finished_without_tool_call": value("behavior", "length_finished_without_tool_call"),
        "requests_before_first_model_tool_call": value("behavior", "requests_before_first_model_tool_call"),
        "output_tokens_before_first_model_tool_call": value("behavior", "output_tokens_before_first_model_tool_call"),
        "requests_before_first_model_edit_call": value("behavior", "requests_before_first_model_edit_call"),
        "output_tokens_before_first_model_edit_call": value("behavior", "output_tokens_before_first_model_edit_call"),
    }


def _repetition(run_id: str) -> int | None:
    marker = "-r"
    if marker not in run_id:
        return None
    suffix = run_id.split(marker, 1)[1].split("-", 1)[0]
    return int(suffix) if suffix.isdigit() else None


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()
