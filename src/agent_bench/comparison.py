"""Read-only, matched comparisons across sealed experiment outputs.

This module is intentionally a derived-report consumer.  It verifies every
input layer before reading it and writes only a new, non-overwriting report.
It does not make profile-wide raw medians stand in for paired effects.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.executor import ExperimentState
from agent_bench.matrix import expand_experiment
from agent_bench.metrics import calculate_run_metrics
from agent_bench.metrics_storage import verify_metrics_artifact
from agent_bench.models import canonical_sha256
from agent_bench.preservation import verify_artifact
from agent_bench.reporting import quantile_type7
from agent_bench.result_store import verify_published_result
from agent_bench.runner import RunManifest


class ComparisonError(RuntimeError):
    """Evidence cannot safely be included in a matched comparison."""


METRICS = (
    "timing.wall_time_seconds.value",
    "timing.llm_time_seconds.value",
    "tokens.input_tokens_total.value",
    "tokens.output_tokens_total.value",
    "behavior.llm_request_count.value",
    "behavior.tool_calls_total.value",
    "context.peak_context_tokens.value",
    "context.net_context_growth_tokens.value",
    "behavior.calls_before_first_edit.value",
    "behavior.reasoning_only_responses.value",
    "behavior.length_finished_responses.value",
    "behavior.length_finished_without_tool_call.value",
    "behavior.requests_before_first_model_tool_call.value",
    "behavior.output_tokens_before_first_model_tool_call.value",
    "behavior.requests_before_first_model_edit_call.value",
    "behavior.output_tokens_before_first_model_edit_call.value",
)


def build_comparison(
    roots: list[Path], *, output: Path, definitions: list[Path] | None = None,
    reference_profile: str | None = None, include_all_pairs: bool = False,
) -> Path:
    """Build a sealed derived comparison without modifying source roots.

    Definitions provide semantic-task and fixed-environment identity.  They
    are discovered conventionally when possible, otherwise callers must pass
    one definition per root; guessing those identities is unsafe.
    """
    if len(roots) < 2:
        raise ComparisonError("at least two experiment roots are required")
    if definitions is not None and len(definitions) != len(roots):
        raise ComparisonError("--experiment-definition must be supplied once per root")
    sources = [root.expanduser().resolve() for root in roots]
    if len(set(sources)) != len(sources):
        raise ComparisonError("experiment roots must be distinct")
    target = output.expanduser().resolve()
    if target.exists():
        raise ComparisonError(f"comparison destination already exists: {target}")
    inputs = [
        _read_root(root, definitions[index] if definitions else None)
        for index, root in enumerate(sources)
    ]
    compatibility = _compatibility(inputs)
    incompatible = [item["dimension"] for item in compatibility if item["status"] == "incompatible"]
    if incompatible:
        raise ComparisonError("incompatible evidence: " + ", ".join(incompatible))
    rows = [row for item in inputs for row in item["rows"]]
    pairs = _pairs(
        rows, reference_profile=reference_profile,
        include_all_pairs=include_all_pairs,
    )
    summaries = _summaries(pairs)
    payload = {
        "schema_version": "1.0.0",
        "kind": "agent-bench-matched-comparison-v1",
        "interpretation": (
            "RAW VALUES and MATCHED PROFILE EFFECTS are separate. Every absolute "
            "delta is candidate minus reference. Direction labels are deterministic "
            "efficiency/behavior observations, never quality or overall agent-"
            "performance wins. Prompt "
            "variants are strata, not efficiency-comparable performance levels "
            "without functional-equivalence evidence. Pocket Ledger is a controlled "
            "microbenchmark and should not be interpreted as a complete measure of "
            "coding-agent capability."
        ),
        "sources": [{key: value for key, value in item.items() if key != "rows"} for item in inputs],
        "compatibility": compatibility,
        "reference_profile": reference_profile,
        "all_pairs_included": include_all_pairs or reference_profile is None,
        "raw_runs": rows,
        "matched_seed_comparisons": pairs,
        "aggregated_paired_effects": summaries,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".comparison-v1.incomplete-", dir=target.parent))
    try:
        _write_json(staging / "comparison.json", payload)
        (staging / "report.html").write_text(_html(payload), encoding="utf-8", newline="\n")
        manifest = {
            "schema_version": "1.0.0", "comparison_id": "matched-comparison-v1-" + canonical_sha256(payload)[:16],
            "source_roots": [str(root) for root in sources], "files": ["comparison.json", "report.html"],
        }
        manifest["record_digest"] = canonical_sha256(manifest)
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging, manifest["files"] + ["manifest.json"])
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _read_root(root: Path, explicit_definition: Path | None) -> dict[str, Any]:
    try:
        state = ExperimentState.model_validate_json((root / "experiment-state.json").read_bytes())
    except Exception as exc:
        raise ComparisonError(f"invalid experiment state at {root}: {exc}") from exc
    definition_path = explicit_definition
    if definition_path is None:
        candidate = Path.cwd() / "experiments" / f"{state.experiment_id}.yaml"
        definition_path = candidate if candidate.is_file() else None
    if definition_path is None:
        raise ComparisonError(f"no immutable experiment definition available for {state.experiment_id}")
    try:
        definition = load_experiment(definition_path)
    except ExperimentConfigError as exc:
        raise ComparisonError(f"invalid experiment definition for {root}: {exc}") from exc
    if definition.experiment_id != state.experiment_id or definition.definition_digest != state.definition_digest:
        raise ComparisonError(f"experiment definition does not match immutable state for {root}")
    planned = {item.run_id: item for item in expand_experiment(definition)}
    profiles = {item.profile_id: item for item in definition.harness_profiles}
    harnesses = {item.harness_id: item for item in definition.harnesses}
    completed = [progress for progress in state.runs if progress.state == "completed"]
    rows: list[dict[str, Any]] = []
    for progress in completed:
        run = planned.get(progress.run_id)
        if run is None:
            raise ComparisonError(f"completed run is absent from immutable definition: {progress.run_id}")
        artifact = root / "artifacts" / progress.run_id
        sealed = verify_artifact(artifact)
        verify_published_result(root, sealed)
        manifest = RunManifest.model_validate_json((artifact / "run" / "manifest.json").read_bytes())
        if manifest.run_id != run.run_id or manifest.prompt_sha256 != run.prompt_sha256:
            raise ComparisonError(f"sealed run identity disagrees with definition: {run.run_id}")
        stored = verify_metrics_artifact(root / "analysis" / run.run_id / "metrics-v1")
        values, provenance = _metric_values(stored.metrics, artifact)
        rows.append({
            "experiment_id": state.experiment_id, "run_id": run.run_id,
            "harness": run.harness_id, "profile": run.profile_id,
            "harness_version": harnesses[run.harness_id].version,
            "reasoning_setting": _reasoning_setting(profiles[run.profile_id].settings),
            "semantic_task": run.semantic_task_id, "prompt_id": run.prompt_id,
            "prompt_sha256": run.prompt_sha256, "prompt_variant": next(p.variant_label for p in definition.prompts if p.prompt_id == run.prompt_id),
            "repetition": run.repetition_index, "seed": manifest.run_seed if manifest.run_seed is not None else run.generation_seed,
            "metrics": values, "metric_provenance": provenance,
        })
    fixed = definition.fixed_environment
    return {
        "experiment_id": state.experiment_id, "root": str(root),
        "definition_digest": state.definition_digest, "completed_runs": len(rows),
        "partial": any(progress.state not in {"completed", "failed", "invalid"} for progress in state.runs),
        "identity": {
            "subject_baseline": fixed.fixed_environment_id + ":" + (definition.portable_baseline.baseline_commit if definition.portable_baseline else definition.baseline_revision),
            "model": fixed.model.definition_digest, "backend": fixed.backend.definition_digest,
            "chat_template": str(fixed.model.gguf_metadata.get("chat_template_sha256") or fixed.backend.build_metadata.get("template_sha256") or "unavailable"),
            "hardware": fixed.hardware.definition_digest,
            "context_backend_settings": canonical_sha256({"server": fixed.server_parameters, "generation": {key: value for key, value in fixed.generation.model_dump(mode="json", exclude={"definition_digest"}).items() if key != "seed"}}),
        }, "rows": rows,
    }


def _metric_values(metrics: Any, artifact: Path) -> tuple[dict[str, float | int | None], dict[str, str]]:
    dumped = metrics.model_dump(mode="json")
    recalculated: dict[str, Any] | None = None
    values: dict[str, float | int | None] = {}
    provenance: dict[str, str] = {}
    for path in METRICS:
        value = _path(dumped, path)
        if value is None and path.startswith("behavior."):
            recalculated = recalculated or calculate_run_metrics(artifact).model_dump(mode="json")
            value = _path(recalculated, path)
            provenance[path] = "recalculated_from_raw_evidence" if value is not None else "unavailable"
        else:
            provenance[path] = "stored_historic_metrics" if value is not None else "unavailable"
        values[path] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return values, provenance


def _path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict): return None
        current = current.get(part)
    return current


def _reasoning_setting(settings: dict[str, Any]) -> str:
    """Expose native profile wording without asserting cross-harness equivalence."""
    fields = settings.get("reasoning_request_fields")
    if isinstance(fields, dict) and "reasoning_effort" in fields:
        return str(fields["reasoning_effort"])
    reasoning = settings.get("reasoning")
    return str(reasoning) if reasoning is not None else "default/unspecified"


def _compatibility(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for dimension in ("subject_baseline", "model", "backend", "chat_template", "hardware", "context_backend_settings"):
        values = {item["identity"][dimension] for item in inputs}
        result.append({"dimension": dimension, "values": sorted(values), "status": "compatible" if len(values) == 1 else "incompatible"})
    harnesses = sorted({row["harness"] + ":" + row["harness_version"] for item in inputs for row in item["rows"]})
    result.append({"dimension": "harness_identity", "values": harnesses, "status": "stratified"})
    return result


def _pairs(
    rows: list[dict[str, Any]], *, reference_profile: str | None = None,
    include_all_pairs: bool = False,
) -> list[dict[str, Any]]:
    cells: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["harness"], row["semantic_task"], row["prompt_sha256"], row["repetition"], row["seed"])
        if row["profile"] in cells[key]:
            raise ComparisonError(f"duplicate profile in matched cell {key}: {row['profile']}")
        cells[key][row["profile"]] = row
    result: list[dict[str, Any]] = []
    observed_reference = False
    for key, profiles in sorted(cells.items()):
        names = sorted(profiles)
        emitted: set[frozenset[str]] = set()
        if reference_profile is not None and reference_profile in profiles:
            observed_reference = True
            for candidate in names:
                if candidate != reference_profile:
                    result.append(_pair(key, profiles[reference_profile], profiles[candidate], "reference"))
                    emitted.add(frozenset((reference_profile, candidate)))
        if reference_profile is None or include_all_pairs:
            for left_index, reference in enumerate(names):
                for candidate in names[left_index + 1:]:
                    if frozenset((reference, candidate)) not in emitted:
                        result.append(_pair(key, profiles[reference], profiles[candidate], "all_pairs"))
    if reference_profile is not None and not observed_reference:
        raise ComparisonError(f"reference profile was not present in completed evidence: {reference_profile}")
    return result


def _pair(
    key: tuple[Any, ...], reference: dict[str, Any], candidate: dict[str, Any],
    comparison_scope: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name in METRICS:
        left, right = reference["metrics"][name], candidate["metrics"][name]
        if left is None or right is None:
            metrics[name] = {"reference_value": left, "candidate_value": right, "absolute_delta": None, "relative_delta_percent": None, "direction": "not_available"}
            continue
        delta = right - left
        metrics[name] = {"reference_value": left, "candidate_value": right, "absolute_delta": delta, "relative_delta_percent": None if left == 0 else delta / left * 100, "direction": _direction(name, delta)}
    harness, task, prompt_sha, repetition, seed = key
    return {"harness": harness, "semantic_task": task, "prompt_sha256": prompt_sha, "repetition": repetition, "seed": seed, "prompt_variant": reference["prompt_variant"], "reference_profile": reference["profile"], "candidate_profile": candidate["profile"], "comparison_scope": comparison_scope, "delta_definition": "candidate_minus_reference", "metrics": metrics}


def _direction(metric: str, delta: float | int) -> str:
    if delta == 0:
        return "equal"
    lower = delta < 0
    if metric.startswith("timing."):
        return "faster" if lower else "slower"
    if metric.startswith(("tokens.", "behavior.")):
        return "fewer" if lower else "more"
    return "lower" if lower else "higher"


def _summaries(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        profiles = (pair["harness"], pair["reference_profile"], pair["candidate_profile"])
        dimensions = {
            "individual_matched_case": pair["semantic_task"] + ":" + pair["prompt_sha256"] + ":" + str(pair["seed"]),
            "task_prompt_across_seeds": pair["semantic_task"] + ":" + pair["prompt_sha256"],
            "task_across_matched_cases": pair["semantic_task"], "per_seed": str(pair["seed"]), "overall": "all",
        }
        for view, key in dimensions.items(): groups[(pair["comparison_scope"], view, key, *profiles)].append(pair)
    result: list[dict[str, Any]] = []
    for (scope, view, group_key, harness, reference, candidate), members in sorted(groups.items()):
        for metric in METRICS:
            records = [item["metrics"][metric] for item in members]
            deltas = [float(item["absolute_delta"]) for item in records if item["absolute_delta"] is not None]
            directions = defaultdict(int)
            for record in records: directions[record["direction"]] += 1
            result.append({"comparison_scope": scope, "view": view, "group_key": group_key, "harness": harness, "reference_profile": reference, "candidate_profile": candidate, "metric": metric, "n_matched_pairs": len(members), "n_available": len(deltas), "n_not_available": len(members) - len(deltas), "mean_paired_delta": sum(deltas) / len(deltas) if deltas else None, "median_paired_delta": quantile_type7(deltas, 0.5), "q1": quantile_type7(deltas, 0.25), "q3": quantile_type7(deltas, 0.75), "direction_counts": dict(sorted(directions.items())), "not_available_count": directions["not_available"]})
    return result


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_checksums(root: Path, names: list[str]) -> None:
    (root / "checksums.sha256").write_text("".join(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n" for name in sorted(names)), encoding="utf-8", newline="\n")


def _html(payload: dict[str, Any]) -> str:
    data = html.escape(json.dumps(payload, ensure_ascii=False), quote=False)
    default_scope = "reference" if payload["reference_profile"] is not None else "all_pairs"
    def summary_rows(scope: str) -> str:
        return "".join(
        f"<tr><td>{html.escape(str(item['view']))}</td>"
        f"<td>{html.escape(item['reference_profile'])} reference; {html.escape(item['candidate_profile'])} candidate</td>"
        f"<td>{html.escape(item['metric'])}</td><td>{item['n_available']}/{item['n_matched_pairs']}</td>"
        f"<td>{item['median_paired_delta']}</td><td>{html.escape(str(item['direction_counts']))}</td></tr>"
        for item in payload["aggregated_paired_effects"]
        if item["comparison_scope"] == scope and item["view"] == "overall"
        )
    reference_note = (
        "Reference-focused view: every delta is candidate minus "
        + html.escape(str(payload["reference_profile"])) + "."
        if payload["reference_profile"] is not None
        else "All profile-pair view: each row names its explicit reference and candidate."
    )
    header = "<tr><th>View</th><th>Explicit orientation</th><th>Metric</th><th>N</th><th>Median Δ</th><th>Direction counts</th></tr>"
    all_pairs = ""
    if payload["reference_profile"] is not None and payload["all_pairs_included"]:
        all_pairs = "<details><summary>All profile-pair summaries</summary><table>" + header + summary_rows("all_pairs") + "</table></details>"
    return """<!doctype html><meta charset=utf-8><title>Agent Bench matched comparison</title><style>body{font:14px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:.4rem;text-align:left}label{display:inline-grid;margin:.25rem;gap:.2rem}select{min-width:9rem}</style><h1>Agent Bench matched comparison</h1><p>RAW RUN METRICS and MATCHED SEED COMPARISONS are distinct. Direction labels are deterministic efficiency/behavior observations, not quality or overall agent-performance wins. Prompt variants are strata, not direct efficiency comparisons.</p><p>""" + reference_note + """</p><h2>Raw run metrics</h2><div id=filters></div><p id=raw-count></p><table><thead><tr><th>Experiment</th><th>Harness</th><th>Profile</th><th>Reasoning</th><th>Task</th><th>Prompt</th><th>Repetition</th><th>Seed</th></tr></thead><tbody id=raw-runs></tbody></table><h2>Aggregated paired effects</h2><p>Absolute delta = candidate − reference. Relative delta percent is unavailable when reference is zero.</p><table>""" + header + summary_rows(default_scope) + "</table>" + all_pairs + """<script type=application/json id=comparison-data>""" + data + """</script><script>
const d=JSON.parse(document.getElementById('comparison-data').textContent), fields=[['experiment_id','Experiment'],['harness','Harness'],['profile','Profile'],['reasoning_setting','Reasoning setting'],['semantic_task','Task'],['prompt_variant','Prompt variant'],['repetition','Repetition'],['seed','Seed']], selected={};
const esc=x=>String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
for(const [key,label] of fields){const s=document.createElement('select');s.innerHTML='<option value="">All</option>'+[...new Set(d.raw_runs.map(r=>r[key]))].sort().map(x=>'<option>'+esc(x)+'</option>').join('');s.onchange=()=>{selected[key]=s.value;render()};const l=document.createElement('label');l.textContent=label;l.append(s);document.getElementById('filters').append(l)}
function render(){const rows=d.raw_runs.filter(r=>fields.every(([key])=>!selected[key]||String(r[key])===selected[key]));document.getElementById('raw-count').textContent=rows.length+' matching raw runs';document.getElementById('raw-runs').innerHTML=rows.map(r=>'<tr>'+fields.map(([key])=>'<td>'+esc(r[key])+'</td>').join('')+'</tr>').join('')};render();
</script>"""
