"""Focused M13 tests: sealed-result functional validation has no live workspace."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_bench.executor as executor_module
import agent_bench.functional as functional_module
from agent_bench.config import load_experiment
from agent_bench.executor import ExperimentExecutor, controlled_dispatch
from agent_bench.functional import FunctionalBaselineHealth, load_functional_scenario
from agent_bench.functional_storage import (
    FunctionalValidationStorageError,
    validate_and_store_functional_artifact,
    verify_functional_validation_artifact,
)
from agent_bench.functional_suite import load_functional_suite
from agent_bench.harness import HarnessExecutionResult, HarnessRunContext
from agent_bench.models import FunctionalScenarioAssociation, PortableBaselineIdentity, RunDefinition, RunLimits
from agent_bench.matrix import expand_experiment
from agent_bench.runner import execute_run
from agent_bench.subject import load_frozen_subject, materialize_baseline


ROOT = Path(__file__).parents[1]
SCENARIO_PATH = ROOT / "functional/scenarios/task-priority-v1.yaml"
SUITE_PATH = ROOT / "functional/suites/taskboard-functional-v1.yaml"


class _OverlayHarness:
    adapter_id = "functional-overlay-fixture"
    adapter_version = "1.0.0"

    def __init__(self, overlays: tuple[Path, ...]) -> None:
        self.overlays = overlays

    def run(self, context: HarnessRunContext) -> HarnessExecutionResult:
        for overlay in self.overlays:
            for source in overlay.rglob("*"):
                if source.is_file():
                    target = context.paths.workspace / source.relative_to(overlay)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        return HarnessExecutionResult(completed_normally=True, session_id="functional-overlay")


def _association() -> FunctionalScenarioAssociation:
    scenario = load_functional_scenario(SCENARIO_PATH)
    suite = load_functional_suite(SUITE_PATH)
    return FunctionalScenarioAssociation(
        scenario_id=scenario.scenario_id, scenario_definition=SCENARIO_PATH,
        scenario_definition_sha256=hashlib.sha256(SCENARIO_PATH.read_bytes()).hexdigest(),
        validator_version=scenario.validator_version,
        validator_sha256=hashlib.sha256(scenario.validator.read_bytes()).hexdigest(),
        prompt_variant="normal", suite_id=suite.suite_id, suite_version=suite.suite_version,
        suite_manifest_sha256=suite.manifest_sha256,
    )


def _run(tmp_path: Path, run_id: str, overlays: tuple[Path, ...], *, experiment_id: str = "functional-fixture"):
    subject = load_frozen_subject(ROOT / "subjects/taskboard-v1")
    baseline = materialize_baseline(subject, tmp_path / f"baseline-{run_id}")
    association = _association()
    definition = RunDefinition(
        run_id=run_id, experiment_id=experiment_id, experiment_matrix_digest="a" * 64,
        identity_version="2.0.0", matrix_index=1, baseline_repository=baseline,
        baseline_revision=subject.identity.baseline_commit, portable_baseline=subject.identity,
        fixed_environment_id="fixture-env", fixed_environment_digest="b" * 64,
        generation_seed=None, generation_seed_control="uncontrollable", harness_id="pi",
        harness_definition_digest="c" * 64, profile_id="fixture-profile",
        profile_definition_digest="d" * 64, prompt_id="fixture-normal",
        prompt_definition_digest="e" * 64, prompt_sha256=hashlib.sha256(b"fixture\n").hexdigest(),
        semantic_task_id=association.scenario_id, functional_scenario=association,
        repetition_index=1, limits=RunLimits(),
    )
    return execute_run(
        run_definition=definition, prompt_content="fixture\n", adapter=_OverlayHarness(overlays),
        artifacts_root=tmp_path / "artifacts", worktrees_root=tmp_path / "worktrees",
        isolation_root=tmp_path / "isolation",
    ), association


def test_functional_validation_is_derived_from_sealed_good_and_bad_results(tmp_path: Path) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH)
    good, association = _run(tmp_path, "functional-good", scenario.self_validation["known-good"].overlays)
    bad, _ = _run(tmp_path, "functional-bad", ())

    good_stored = validate_and_store_functional_artifact(
        source_artifact=good.artifact_path, output_root=tmp_path / "analysis",
        run_id="functional-good", experiment_id="functional-fixture", association=association,
    )
    bad_stored = validate_and_store_functional_artifact(
        source_artifact=bad.artifact_path, output_root=tmp_path / "analysis",
        run_id="functional-bad", experiment_id="functional-fixture", association=association,
    )

    assert verify_functional_validation_artifact(good_stored.root).result.validation_status == "pass"
    assert good_stored.result.acceptance_score_numerator == good_stored.result.acceptance_score_denominator == 16
    assert verify_functional_validation_artifact(bad_stored.root).result.validation_status == "fail"
    assert bad_stored.result.acceptance_score_numerator == 7
    assert (good.artifact_path / "source/source.tar").is_file()  # source remains retryable


def test_validator_error_is_sealed_without_an_acceptance_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH)
    sealed, association = _run(tmp_path, "functional-validator-error", scenario.self_validation["known-good"].overlays)
    monkeypatch.setattr(
        functional_module, "_run_validator",
        lambda current, _workspace: (functional_module._error_results(current, "synthetic validator fault"), {"status": "error"}),
    )
    stored = validate_and_store_functional_artifact(
        source_artifact=sealed.artifact_path, output_root=tmp_path / "analysis",
        run_id="functional-validator-error", experiment_id="functional-fixture", association=association,
    )
    assert stored.result.validation_status == "error"
    assert stored.result.acceptance_score_numerator is None
    assert stored.result.acceptance_score_denominator is None
    assert (sealed.artifact_path / "source/source.tar").is_file()


def _functional_experiment(association: FunctionalScenarioAssociation):
    subject = load_frozen_subject(ROOT / "subjects/taskboard-v1")
    template = load_experiment(ROOT / "experiments/pocket-ledger-v1.yaml")
    prompt = template.prompts[0].model_copy(update={
        "prompt_id": "functional-normal", "semantic_task_id": association.scenario_id,
        "variant_label": "normal", "content": "fixture\n", "byte_length": 8,
        "sha256": hashlib.sha256(b"fixture\n").hexdigest(), "functional_scenario": association,
    })
    return template.model_copy(update={
        "identity_version": "2.0.0", "portable_baseline": subject.identity,
        "baseline_repository": subject.source_directory, "baseline_revision": subject.identity.baseline_commit,
        "prompts": (prompt,), "repetition_indices": (1,), "repetitions": None,
    }), subject


@pytest.mark.parametrize(("fixture", "expected"), [("known-good", "pass"), ("untouched-baseline", "fail")])
def test_controlled_functional_acceptance_completes_for_good_and_bad_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture: str, expected: str,
) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH); association = _association()
    experiment, subject = _functional_experiment(association)
    run = expand_experiment(experiment)[0]
    sealed, _ = _run(tmp_path, run.run_id, scenario.self_validation[fixture].overlays, experiment_id=experiment.experiment_id)
    controlled = SimpleNamespace(
        run=SimpleNamespace(artifact_path=sealed.artifact_path, artifact_manifest=sealed.artifact_manifest),
        metrics=SimpleNamespace(root=tmp_path / "metrics"), context_analysis_path=tmp_path / "context", failed_run=None,
    )
    for name in ("execute_controlled_opencode_run", "execute_controlled_pi_run", "execute_controlled_hermes_run"):
        monkeypatch.setattr(executor_module, name, lambda **_kwargs: controlled)
    monkeypatch.setattr(executor_module, "verify_artifact", lambda *_args, **_kwargs: sealed.artifact_manifest)
    monkeypatch.setattr(executor_module, "verify_metrics_artifact", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(executor_module, "verify_context_analysis_artifact", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(executor_module, "publish_result_ref", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_module, "verify_published_result", lambda *_args, **_kwargs: None)

    state = ExperimentExecutor(experiment, tmp_path / f"out-{fixture}", controlled_dispatch(experiment, subject)).run(limit=1)
    assert state.runs[0].state == "completed"
    assert state.runs[0].functional_validation_status == expected


def test_functional_baseline_failure_stops_before_harness_and_validator_error_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH); association = _association()
    experiment, subject = _functional_experiment(association); run = expand_experiment(experiment)[0]
    health = FunctionalBaselineHealth(
        scenario_id=scenario.scenario_id, baseline_identity=subject.identity, command=("node", "test"),
        status="failed", return_code=1,
    )
    monkeypatch.setattr(executor_module, "baseline_health_check", lambda *_args: health)
    started: list[bool] = []
    monkeypatch.setattr(executor_module, "execute_controlled_hermes_run", lambda **_kwargs: started.append(True))
    precondition_state = ExperimentExecutor(experiment, tmp_path / "out-precondition", controlled_dispatch(experiment, subject)).run(limit=1)
    assert precondition_state.runs[0].state == "failed"
    assert precondition_state.runs[0].failure_class == "functional_baseline_health_failed"
    assert started == []
    assert (tmp_path / "out-precondition/functional-preconditions" / f"{run.run_id}-baseline-health.json").is_file()

    # A post-run validator/storage fault is analysis infrastructure failure,
    # never a fabricated functional score, and the sealed source stays intact.
    monkeypatch.undo()
    sealed, _ = _run(tmp_path, run.run_id, scenario.self_validation["known-good"].overlays, experiment_id=experiment.experiment_id)
    controlled = SimpleNamespace(run=SimpleNamespace(artifact_path=sealed.artifact_path, artifact_manifest=sealed.artifact_manifest), metrics=SimpleNamespace(root=tmp_path / "metrics"), context_analysis_path=tmp_path / "context", failed_run=None)
    monkeypatch.setattr(executor_module, "execute_controlled_hermes_run", lambda **_kwargs: controlled)
    monkeypatch.setattr(executor_module, "verify_artifact", lambda *_args, **_kwargs: sealed.artifact_manifest)
    monkeypatch.setattr(executor_module, "verify_metrics_artifact", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(executor_module, "verify_context_analysis_artifact", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(executor_module, "publish_result_ref", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_module, "verify_published_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor_module, "validate_and_store_functional_artifact", lambda **_kwargs: (_ for _ in ()).throw(FunctionalValidationStorageError("synthetic validator fault")))
    error_state = ExperimentExecutor(experiment, tmp_path / "out-validator-error", controlled_dispatch(experiment, subject)).run(limit=1)
    assert error_state.runs[0].state == "failed"
    assert error_state.runs[0].failure_domain == "analysis"
    assert (sealed.artifact_path / "source/source.tar").is_file()
