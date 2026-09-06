"""Deterministic, headless functional validation for benchmark scenarios.

Validators deliberately live outside subject workspaces.  A scenario declares
its frozen subject and an expected untouched-baseline result vector; the same
Node validator then evaluates a post-agent workspace.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from agent_bench.models import PortableBaselineIdentity, canonical_sha256
from agent_bench.subject import load_frozen_subject, materialize_baseline


FUNCTIONAL_SCHEMA_VERSION = "1.0.0"


class FunctionalValidationError(RuntimeError):
    """Raised for malformed scenarios or validation infrastructure failures."""


class FunctionalTestOutcome(BaseModel):
    """One deterministic test result; ``unavailable`` is not a failed test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_id: str
    category: Literal["baseline_regression", "feature_requirement", "edge_case"]
    outcome: Literal["passed", "failed", "error", "unavailable"]
    detail: str = ""


class FunctionalValidationResult(BaseModel):
    """Versioned, immutable-on-write functional result artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = FUNCTIONAL_SCHEMA_VERSION
    scenario_id: str
    run_id: str
    validation_mode: Literal["baseline_discrimination", "post_run"]
    validator_version: str
    validator_sha256: str
    baseline_identity: PortableBaselineIdentity
    observed_at_utc: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    unavailable_tests: int
    error_tests: int
    score_numerator: int
    score_denominator: int
    score_percent: float
    baseline_regression: dict[str, int]
    feature_requirements: dict[str, int]
    edge_cases: dict[str, int]
    hard_gate_pass: bool
    hard_gates: dict[str, bool]
    tests: tuple[FunctionalTestOutcome, ...]
    provenance: dict[str, object]


class FunctionalScenario(BaseModel):
    """Checked-in definition for one functional benchmark scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = FUNCTIONAL_SCHEMA_VERSION
    scenario_id: str
    subject_root: Path
    validator: Path
    validator_version: str
    expected_baseline_outcomes: dict[str, Literal["passed", "failed"]]
    hard_gates: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_gates(self) -> "FunctionalScenario":
        expected = set(self.expected_baseline_outcomes)
        if not expected:
            raise ValueError("expected_baseline_outcomes must not be empty")
        missing = sorted({test for tests in self.hard_gates.values() for test in tests} - expected)
        if missing:
            raise ValueError(f"hard gates reference unknown tests: {', '.join(missing)}")
        return self


def load_functional_scenario(path: Path) -> FunctionalScenario:
    """Load a scenario, resolving subject and validator relative to its YAML."""
    definition = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(definition.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise FunctionalValidationError(f"cannot read functional scenario {definition}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FunctionalValidationError("functional scenario must be a mapping")
    try:
        scenario = FunctionalScenario.model_validate(raw)
    except ValueError as exc:
        raise FunctionalValidationError(f"invalid functional scenario: {exc}") from exc
    subject_root = (definition.parent / scenario.subject_root).resolve()
    validator = (definition.parent / scenario.validator).resolve()
    if not validator.is_file():
        raise FunctionalValidationError(f"functional validator is missing: {validator}")
    return scenario.model_copy(update={"subject_root": subject_root, "validator": validator})


def baseline_check(scenario: FunctionalScenario, output: Path) -> FunctionalValidationResult:
    """Prove that the frozen baseline is healthy and discriminates the feature."""
    subject = load_frozen_subject(scenario.subject_root)
    with tempfile.TemporaryDirectory(prefix="agent-bench-functional-") as temporary:
        baseline = materialize_baseline(subject, Path(temporary) / "baseline")
        results, runner_provenance = _run_validator(scenario, baseline)
    observed = {item.test_id: item.outcome for item in results}
    if observed != scenario.expected_baseline_outcomes:
        raise FunctionalValidationError(
            "baseline discrimination vector differs from the recorded expectation: "
            f"expected {scenario.expected_baseline_outcomes}, observed {observed}"
        )
    result = _build_result(
        scenario, subject.identity, f"baseline-{subject.identity.baseline_commit[:12]}",
        "baseline_discrimination", results, runner_provenance,
        {"expected_baseline_outcomes": scenario.expected_baseline_outcomes,
         "expected_baseline_vector_sha256": canonical_sha256(scenario.expected_baseline_outcomes)},
    )
    _write_new_result(output, result)
    return result


def validate_workspace(
    scenario: FunctionalScenario, workspace: Path, run_id: str, output: Path,
) -> FunctionalValidationResult:
    """Validate one post-agent workspace without modifying it."""
    subject = load_frozen_subject(scenario.subject_root)
    root = workspace.expanduser().resolve()
    if not root.is_dir():
        raise FunctionalValidationError(f"workspace is missing: {root}")
    results, runner_provenance = _run_validator(scenario, root)
    result = _build_result(
        scenario, subject.identity, run_id, "post_run", results, runner_provenance,
        {"workspace": str(root), "workspace_source_sha256": _directory_digest(root)},
    )
    _write_new_result(output, result)
    return result


def _run_validator(
    scenario: FunctionalScenario, workspace: Path,
) -> tuple[tuple[FunctionalTestOutcome, ...], dict[str, object]]:
    node = shutil.which("node")
    if node is None:
        return _unavailable_results(scenario, "Node.js executable 'node' is unavailable"), {"runner": "node", "status": "unavailable"}
    command = [node, str(scenario.validator), str(workspace)]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return _error_results(scenario, "validator timed out after 30 seconds"), {"runner": "node", "status": "timeout", "command": command}
    if completed.returncode:
        return _error_results(scenario, completed.stderr.strip() or f"validator exited {completed.returncode}"), {"runner": "node", "status": "error", "command": command, "exit_code": completed.returncode}
    try:
        payload = json.loads(completed.stdout)
        outcomes = tuple(FunctionalTestOutcome.model_validate(item) for item in payload["tests"])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        return _error_results(scenario, f"validator emitted invalid JSON: {exc}"), {"runner": "node", "status": "invalid_output", "command": command}
    expected = set(scenario.expected_baseline_outcomes)
    observed = [item.test_id for item in outcomes]
    if set(observed) != expected or len(observed) != len(set(observed)):
        return _error_results(scenario, "validator test IDs do not match scenario definition"), {"runner": "node", "status": "invalid_test_ids", "command": command}
    return outcomes, {"runner": "node", "status": "completed", "command": command, "validator_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()}


def _unavailable_results(scenario: FunctionalScenario, detail: str) -> tuple[FunctionalTestOutcome, ...]:
    return tuple(FunctionalTestOutcome(test_id=test_id, category=_category(test_id), outcome="unavailable", detail=detail) for test_id in scenario.expected_baseline_outcomes)


def _error_results(scenario: FunctionalScenario, detail: str) -> tuple[FunctionalTestOutcome, ...]:
    return tuple(FunctionalTestOutcome(test_id=test_id, category=_category(test_id), outcome="error", detail=detail) for test_id in scenario.expected_baseline_outcomes)


def _category(test_id: str) -> Literal["baseline_regression", "feature_requirement", "edge_case"]:
    if test_id.startswith("baseline-"):
        return "baseline_regression"
    if test_id.startswith("edge-"):
        return "edge_case"
    return "feature_requirement"


def _build_result(scenario: FunctionalScenario, identity: PortableBaselineIdentity, run_id: str, mode: Literal["baseline_discrimination", "post_run"], tests: tuple[FunctionalTestOutcome, ...], runner_provenance: dict[str, object], extra_provenance: dict[str, object]) -> FunctionalValidationResult:
    counts = {category: _counts(tuple(item for item in tests if item.category == category)) for category in ("baseline_regression", "feature_requirement", "edge_case")}
    passed = sum(item.outcome == "passed" for item in tests)
    failed = sum(item.outcome == "failed" for item in tests)
    unavailable = sum(item.outcome == "unavailable" for item in tests)
    errors = sum(item.outcome == "error" for item in tests)
    gates = {name: all(next(item for item in tests if item.test_id == test_id).outcome == "passed" for test_id in test_ids) for name, test_ids in scenario.hard_gates.items()}
    validator_digest = hashlib.sha256(scenario.validator.read_bytes()).hexdigest()
    return FunctionalValidationResult(
        scenario_id=scenario.scenario_id, run_id=run_id, validation_mode=mode,
        validator_version=scenario.validator_version, validator_sha256=validator_digest,
        baseline_identity=identity, observed_at_utc=datetime.now(timezone.utc).isoformat(),
        total_tests=len(tests), passed_tests=passed, failed_tests=failed,
        unavailable_tests=unavailable, error_tests=errors, score_numerator=passed,
        score_denominator=len(tests), score_percent=round((passed / len(tests)) * 100, 6) if tests else 0.0,
        baseline_regression=counts["baseline_regression"], feature_requirements=counts["feature_requirement"], edge_cases=counts["edge_case"], hard_gate_pass=all(gates.values()), hard_gates=gates, tests=tests,
        provenance={"scenario_definition_sha256": canonical_sha256(_scenario_identity(scenario)), "runner": runner_provenance, **extra_provenance},
    )


def _counts(tests: tuple[FunctionalTestOutcome, ...]) -> dict[str, int]:
    return {"total": len(tests), "passed": sum(item.outcome == "passed" for item in tests), "failed": sum(item.outcome == "failed" for item in tests), "unavailable": sum(item.outcome == "unavailable" for item in tests), "error": sum(item.outcome == "error" for item in tests)}


def _write_new_result(path: Path, result: FunctionalValidationResult) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise FunctionalValidationError(f"validation result already exists and is immutable: {destination}") from exc


def _scenario_identity(scenario: FunctionalScenario) -> dict[str, object]:
    """Hash only portable scenario semantics, never its checkout location."""
    return {
        "schema_version": scenario.schema_version,
        "scenario_id": scenario.scenario_id,
        "validator_version": scenario.validator_version,
        "expected_baseline_outcomes": scenario.expected_baseline_outcomes,
        "hard_gates": scenario.hard_gates,
    }


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] == ".git":
            continue
        if path.is_file():
            digest.update(relative.as_posix().encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
