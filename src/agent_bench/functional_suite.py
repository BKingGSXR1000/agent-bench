"""Versioned all-scenario functional-suite verification."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from agent_bench.functional import (
    FunctionalScenario,
    FunctionalValidationError,
    load_functional_scenario,
    self_validate,
)
from agent_bench.models import canonical_sha256
from agent_bench.subject import load_frozen_subject, materialize_baseline


class FunctionalSuiteError(RuntimeError):
    """Raised when a suite identity or invariant does not hold."""


class PromptIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sha256: str


class ComplexityInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_module_count: int
    source_line_count: int
    baseline_features: tuple[str, ...]
    conceptual_change_areas: tuple[str, ...]


class SuiteScenario(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str
    tier: Literal["easy", "medium", "complex"]
    scenario_definition: Path
    scenario_definition_sha256: str
    subject_id: str
    baseline_commit: str
    baseline_tree: str
    baseline_bundle_sha256: str
    validator_sha256: str
    prompt_manifest: Path
    prompts: dict[Literal["vague", "normal", "precise"], PromptIdentity]
    expected_fixture_vector_sha256: dict[str, str]
    complexity: ComplexityInventory


class FunctionalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    suite_id: str
    suite_version: str
    scenarios: tuple[SuiteScenario, ...]
    manifest_path: Path = Path()
    manifest_sha256: str = ""


def load_functional_suite(path: Path) -> FunctionalSuite:
    manifest_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FunctionalSuiteError(f"cannot read functional suite {manifest_path}: {exc}") from exc
    try:
        suite = FunctionalSuite.model_validate(raw)
    except ValueError as exc:
        raise FunctionalSuiteError(f"invalid functional suite: {exc}") from exc
    scenarios = tuple(
        item.model_copy(update={
            "scenario_definition": (manifest_path.parent / item.scenario_definition).resolve(),
            "prompt_manifest": (manifest_path.parent / item.prompt_manifest).resolve(),
            "prompts": {name: prompt.model_copy(update={"path": (manifest_path.parent / prompt.path).resolve()}) for name, prompt in item.prompts.items()},
        })
        for item in suite.scenarios
    )
    if len({item.scenario_id for item in scenarios}) != len(scenarios):
        raise FunctionalSuiteError("suite scenario IDs must be unique")
    return suite.model_copy(update={"scenarios": scenarios, "manifest_path": manifest_path, "manifest_sha256": _sha256(manifest_path)})


def self_check_suite(suite: FunctionalSuite, output: Path) -> dict[str, object]:
    """Run all scenario invariants and create one immutable suite result root."""
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FunctionalSuiteError(f"suite output already exists and is immutable: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    rows: list[dict[str, object]] = []
    leakage_checked = 0
    try:
        for entry in suite.scenarios:
            scenario = _verify_entry_identity(entry)
            subject = load_frozen_subject(scenario.subject_root)
            _verify_subject_identity(entry, subject.identity.subject_id, subject.identity.baseline_commit, subject.identity.baseline_tree, subject.identity.baseline_bundle_sha256)
            with tempfile.TemporaryDirectory(prefix="agent-bench-functional-suite-") as temporary:
                baseline = materialize_baseline(subject, Path(temporary) / "baseline")
                _run_visible_baseline_health(scenario.subject_root, baseline)
                _verify_complexity_inventory(entry, baseline)
                _verify_no_evaluator_leak(baseline, scenario)
                leakage_checked += 1
            results = self_validate(scenario, destination / entry.scenario_id)
            _verify_fixture_vectors(entry, scenario)
            _verify_result_schema(results)
            rows.append({"tier": entry.tier, "scenario_id": entry.scenario_id, "status": "PASS", "baseline_health": "PASS", "leakage_check": "PASS", "fixtures": len(results), "fixtures_matched": len(results)})
    except Exception:
        raise
    payload = {
        "schema_version": "1.0.0",
        "suite_id": suite.suite_id,
        "suite_version": suite.suite_version,
        "suite_manifest_sha256": suite.manifest_sha256,
        "scenarios": rows,
        "scenarios_valid": len(rows),
        "scenarios_total": len(suite.scenarios),
        "fixtures_matched": sum(int(row["fixtures_matched"]) for row in rows),
        "fixtures_total": sum(int(row["fixtures"]) for row in rows),
        "leakage_checks_passed": leakage_checked,
        "schema_consistency": "PASS",
        "suite_status": "PASS",
    }
    _write_new_json(destination / "suite-summary.json", payload)
    return payload


def suite_summary_text(payload: dict[str, object]) -> str:
    labels = {"easy": "Easy", "medium": "Medium", "complex": "Complex"}
    lines = [f"{labels[str(row['tier'])]:<8} {str(row['scenario_id']):<32} PASS" for row in payload["scenarios"]]  # type: ignore[index]
    lines.extend((
        "",
        f"Scenarios: {payload['scenarios_valid']}/{payload['scenarios_total']} valid",
        f"Fixtures: {payload['fixtures_matched']}/{payload['fixtures_total']} exact vectors matched",
        "Suite: PASS",
    ))
    return "\n".join(lines)


def _verify_entry_identity(entry: SuiteScenario) -> FunctionalScenario:
    if not entry.scenario_definition.is_file() or _sha256(entry.scenario_definition) != entry.scenario_definition_sha256:
        raise FunctionalSuiteError(f"scenario definition digest mismatch: {entry.scenario_id}")
    scenario = load_functional_scenario(entry.scenario_definition)
    if scenario.scenario_id != entry.scenario_id:
        raise FunctionalSuiteError(f"scenario ID mismatch: {entry.scenario_id}")
    if _sha256(scenario.validator) != entry.validator_sha256:
        raise FunctionalSuiteError(f"validator digest mismatch: {entry.scenario_id}")
    for variant, prompt in entry.prompts.items():
        if not prompt.path.is_file() or _sha256(prompt.path) != prompt.sha256:
            raise FunctionalSuiteError(f"prompt digest mismatch: {entry.scenario_id}/{variant}")
    _verify_prompt_contract(entry)
    return scenario


def _verify_prompt_contract(entry: SuiteScenario) -> None:
    try:
        raw = yaml.safe_load(entry.prompt_manifest.read_text(encoding="utf-8"))
        declared = raw["prompts"]
    except (OSError, KeyError, TypeError) as exc:
        raise FunctionalSuiteError(f"invalid prompt manifest: {entry.scenario_id}") from exc
    if raw.get("scenario_id") != entry.scenario_id or set(declared) != {"vague", "normal", "precise"}:
        raise FunctionalSuiteError(f"prompt scenario contract mismatch: {entry.scenario_id}")
    for variant, expected in entry.prompts.items():
        item = declared[variant]
        if not isinstance(item, dict) or item.get("sha256") != expected.sha256:
            raise FunctionalSuiteError(f"prompt manifest identity mismatch: {entry.scenario_id}/{variant}")
        if (entry.prompt_manifest.parent / str(item.get("path"))).resolve() != expected.path:
            raise FunctionalSuiteError(f"prompt manifest path mismatch: {entry.scenario_id}/{variant}")


def _verify_subject_identity(entry: SuiteScenario, subject_id: str, commit: str, tree: str, bundle_sha: str) -> None:
    if (entry.subject_id, entry.baseline_commit, entry.baseline_tree, entry.baseline_bundle_sha256) != (subject_id, commit, tree, bundle_sha):
        raise FunctionalSuiteError(f"baseline identity mismatch: {entry.scenario_id}")


def _run_visible_baseline_health(subject_root: Path, baseline: Path) -> None:
    try:
        raw = yaml.safe_load((subject_root / "subject.yaml").read_text(encoding="utf-8"))
        command = raw["commands"]["test"]
    except (OSError, KeyError, TypeError) as exc:
        raise FunctionalSuiteError(f"invalid visible baseline command: {subject_root}") from exc
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise FunctionalSuiteError(f"invalid visible baseline command: {subject_root}")
    completed = subprocess.run(command, cwd=baseline, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
    if completed.returncode:
        raise FunctionalSuiteError(f"visible baseline health failed for {subject_root.name}: {completed.stderr.strip() or completed.stdout.strip()}")


def _verify_no_evaluator_leak(baseline: Path, scenario: FunctionalScenario) -> None:
    owned = {_sha256(scenario.validator)}
    for fixture in scenario.self_validation.values():
        for overlay in fixture.overlays:
            owned.update(_sha256(path) for path in overlay.rglob("*") if path.is_file())
    for path in baseline.rglob("*"):
        relative = path.relative_to(baseline)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_file():
            if any(part in {"functional", "references", "validators"} for part in relative.parts):
                raise FunctionalSuiteError(f"evaluator path leaked into subject bundle: {relative}")
            if _sha256(path) in owned:
                raise FunctionalSuiteError(f"evaluator-owned content leaked into subject bundle: {relative}")


def _verify_complexity_inventory(entry: SuiteScenario, baseline: Path) -> None:
    source_files = sorted((baseline / "src").glob("*.js"))
    module_count = len(source_files)
    line_count = sum(len(path.read_text(encoding="utf-8").splitlines()) for path in source_files)
    if (module_count, line_count) != (entry.complexity.source_module_count, entry.complexity.source_line_count):
        raise FunctionalSuiteError(f"complexity inventory mismatch: {entry.scenario_id}")


def _verify_fixture_vectors(entry: SuiteScenario, scenario: FunctionalScenario) -> None:
    actual = {fixture_id: canonical_sha256(fixture.expected.model_dump(mode="json")) for fixture_id, fixture in scenario.self_validation.items()}
    if actual != entry.expected_fixture_vector_sha256:
        raise FunctionalSuiteError(f"expected fixture vector digest mismatch: {entry.scenario_id}")


def _verify_result_schema(results: tuple[object, ...]) -> None:
    expected_count_keys = {"total", "passed", "failed", "unavailable", "error"}
    for result in results:
        value = result
        if getattr(value, "schema_version") != "1.0.0" or getattr(value, "score_denominator") != getattr(value, "total_tests"):
            raise FunctionalSuiteError("functional result schema mismatch")
        if set(getattr(value, "baseline_regression")) != expected_count_keys or not isinstance(getattr(value, "hard_gates"), dict) or not isinstance(getattr(value, "hard_gate_pass"), bool):
            raise FunctionalSuiteError("functional result summary representation mismatch")
        for test in getattr(value, "tests"):
            if test.outcome not in {"passed", "failed", "error", "unavailable", "manual_review_required"}:
                raise FunctionalSuiteError("functional result status vocabulary mismatch")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
