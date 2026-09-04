from pathlib import Path
import hashlib
from types import SimpleNamespace

from agent_bench.config import load_experiment
from agent_bench.executor import DispatchOutcome, ExperimentExecutor, load_or_create, status
from agent_bench.matrix import expand_experiment
from agent_bench.models import PortableBaselineIdentity
from agent_bench.models import RunDefinition
from agent_bench.subject import load_frozen_subject, materialize_baseline, verify_materialized_baseline


def test_executor_resume_subset_and_atomic_state(tmp_path: Path, experiment_fixture: object) -> None:
    experiment = load_experiment(experiment_fixture.path)  # type: ignore[attr-defined]
    seen: list[str] = []
    executor = ExperimentExecutor(experiment, tmp_path / "out", lambda run, root: (seen.append(run.run_id) or True))
    planned = executor.plan(); assert len(planned) > 2
    state = executor.run(limit=2); assert len(seen) == 2 and state_path_exists(tmp_path / "out")
    assert (tmp_path / "out" / "executor-events.jsonl").read_text(encoding="utf-8").count('"state":"running"') == 2
    state = executor.run(resume=True, selected={planned[2].run_id}); assert seen[-1] == planned[2].run_id
    assert status(state)["counts"]["completed"] == 3


def test_executor_failure_continues(tmp_path: Path, experiment_fixture: object) -> None:
    experiment = load_experiment(experiment_fixture.path)  # type: ignore[attr-defined]
    calls: list[str] = []
    def dispatch(run: object, root: Path) -> bool:
        calls.append(run.run_id)  # type: ignore[attr-defined]
        return len(calls) != 1
    state = ExperimentExecutor(experiment, tmp_path / "out", dispatch).run(limit=3)
    assert [r.state for r in state.runs[:3]] == ["failed", "completed", "completed"]
    assert state.runs[0].failure_domain == "harness_runtime"
    assert state.runs[0].failure_phase == "running"


def test_controlled_lifecycle_does_not_emit_running_before_preflight_and_breaks_repeated_infra_failure(tmp_path: Path, experiment_fixture: object) -> None:
    experiment = load_experiment(experiment_fixture.path)  # type: ignore[attr-defined]

    class Dispatch:
        reporter = None
        def set_phase_reporter(self, reporter):
            self.reporter = reporter
        def __call__(self, run, root):
            assert self.reporter is not None
            # This is the real-preflight boundary: no running event existed
            # before this callback.
            self.reporter("running")
            return DispatchOutcome(False, "backend endpoint has active listener", "infrastructure_precondition", "benchmark_port_in_use", "preflight", False, False, True)

    root = tmp_path / "out"
    state = ExperimentExecutor(experiment, root, Dispatch()).run(limit=5)
    assert [item.state for item in state.runs[:2]] == ["failed", "failed"]
    assert all(item.state == "pending" for item in state.runs[2:])
    assert state.circuit_breaker is not None
    assert state.circuit_breaker["threshold"] == 2
    events = (root / "executor-events.jsonl").read_text(encoding="utf-8").splitlines()
    first_run_events = [line for line in events if state.runs[0].run_id in line]
    assert '"state":"preflight"' in first_run_events[0]
    assert '"state":"running"' in first_run_events[1]
    assert state.runs[0].failure_phase == "preflight"
    assert state.runs[0].harness_execution_started is False


def test_resume_marks_completed_run_invalid_when_durable_evidence_is_missing(tmp_path: Path, experiment_fixture: object) -> None:
    experiment = load_experiment(experiment_fixture.path)  # type: ignore[attr-defined]
    root = tmp_path / "out"
    calls: list[str] = []
    executor = ExperimentExecutor(experiment, root, lambda run, _root: (calls.append(run.run_id) or True))
    first = executor.run(limit=1)
    assert first.runs[0].state == "completed"
    resumed = executor.run(resume=True, limit=1, selected={first.runs[0].run_id})
    assert resumed.runs[0].state == "invalid"
    assert "integrity" in (resumed.runs[0].detail or "")
    assert len(calls) == 1


def test_controlled_dispatch_publishes_the_sealed_artifact_manifest(tmp_path: Path, preserved_run: object, monkeypatch: object) -> None:
    """Regression: RunExecutionResult has artifact_manifest, not manifest."""
    from agent_bench.executor import controlled_dispatch
    import agent_bench.executor as executor
    from agent_bench.config import load_experiment

    experiment = load_experiment(Path("experiments/pocket-ledger-v1.yaml"))
    subject = load_frozen_subject(Path("subjects/pocket-ledger-v1"))
    run = next(item for item in expand_experiment(experiment) if item.harness_id == "hermes")
    sealed = preserved_run.manifest  # type: ignore[attr-defined]
    execution = SimpleNamespace(artifact_path=tmp_path / "artifact", artifact_manifest=sealed)
    controlled = SimpleNamespace(run=execution, metrics=SimpleNamespace(root=tmp_path / "metrics"), context_analysis_path=tmp_path / "context", failed_run=None)
    monkeypatch.setattr(executor, "execute_controlled_hermes_run", lambda **_kwargs: controlled)
    monkeypatch.setattr(executor, "verify_artifact", lambda *_args, **_kwargs: sealed)
    monkeypatch.setattr(executor, "verify_metrics_artifact", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(executor, "verify_context_analysis_artifact", lambda *_args, **_kwargs: object())
    published: list[object] = []
    monkeypatch.setattr(executor, "publish_result_ref", lambda _root, _baseline, manifest: published.append(manifest))
    monkeypatch.setattr(executor, "verify_published_result", lambda _root, manifest: published.append(manifest))
    assert controlled_dispatch(experiment, subject)(run, tmp_path / "out")
    assert published == [sealed, sealed]


def state_path_exists(root: Path) -> bool: return (root / "experiment-state.json").is_file()


def test_v2_identity_is_portable_across_clone_and_toolchain_roots(
    tmp_path: Path, experiment_fixture: object
) -> None:
    legacy = load_experiment(experiment_fixture.path)  # type: ignore[attr-defined]
    portable = PortableBaselineIdentity(
        subject_id="fixture-subject", subject_version="1.0.0",
        baseline_commit="a" * 40, baseline_tree="b" * 40,
        baseline_bundle_sha256="c" * 64,
    )
    first = legacy.model_copy(update={
        "identity_version": "2.0.0", "portable_baseline": portable,
        "baseline_repository": tmp_path / "one" / "baseline",
        "baseline_revision": "a" * 40,
        "fixed_environment": legacy.fixed_environment.model_copy(update={
            "model": legacy.fixed_environment.model.model_copy(update={"path": tmp_path / "one" / "model.gguf"}),
            "backend": legacy.fixed_environment.backend.model_copy(update={"executable": tmp_path / "one" / "llama-server"}),
        }),
        "harnesses": tuple(item.model_copy(update={"executable": tmp_path / "one" / item.harness_id}) for item in legacy.harnesses),
    })
    second = first.model_copy(update={
        "baseline_repository": tmp_path / "two" / "baseline",
        "fixed_environment": first.fixed_environment.model_copy(update={
            "model": first.fixed_environment.model.model_copy(update={"path": tmp_path / "two" / "model.gguf"}),
            "backend": first.fixed_environment.backend.model_copy(update={"executable": tmp_path / "two" / "llama-server"}),
        }),
        "harnesses": tuple(item.model_copy(update={"executable": tmp_path / "two" / item.harness_id}) for item in first.harnesses),
    })
    assert first.definition_digest == second.definition_digest
    assert first.matrix_digest == second.matrix_digest
    assert [item.run_id for item in expand_experiment(first)] == [item.run_id for item in expand_experiment(second)]
    assert first.definition_digest != legacy.definition_digest


def test_v2_content_changes_change_run_identity(tmp_path: Path, experiment_fixture: object) -> None:
    experiment = load_experiment(experiment_fixture.path)  # type: ignore[attr-defined]
    portable = PortableBaselineIdentity(subject_id="fixture-subject", subject_version="1.0.0", baseline_commit="a" * 40, baseline_tree="b" * 40, baseline_bundle_sha256="c" * 64)
    base = experiment.model_copy(update={"identity_version": "2.0.0", "portable_baseline": portable, "baseline_revision": "a" * 40})
    changed_prompt = base.prompts[0].model_copy(update={"content": "other\n", "byte_length": 6, "sha256": hashlib.sha256(b"other\n").hexdigest()})
    changed = base.model_copy(update={"prompts": (changed_prompt, *base.prompts[1:])})
    assert expand_experiment(base)[0].run_id != expand_experiment(changed)[0].run_id
    changed_bundle = base.model_copy(update={"portable_baseline": portable.model_copy(update={"baseline_bundle_sha256": "d" * 64})})
    assert base.matrix_digest != changed_bundle.matrix_digest


def test_frozen_subject_bundle_materializes_a_fresh_clean_baseline(tmp_path: Path) -> None:
    subject = load_frozen_subject(Path("subjects/pocket-ledger-v1"))
    first = materialize_baseline(subject, tmp_path / "first")
    second = materialize_baseline(subject, tmp_path / "second")
    verify_materialized_baseline(first, subject.identity)
    verify_materialized_baseline(second, subject.identity)
    assert first != second
    assert (first / "README.md").read_bytes() == (second / "README.md").read_bytes()


def test_legacy_v1_run_definition_remains_readable() -> None:
    # Existing sealed M1--M8 payloads retain their v1 schema/identity values.
    from agent_bench.models import RunLimits
    legacy = RunDefinition(
        run_id="pi-profile-prompt-r001-legacy", experiment_id="legacy",
        experiment_matrix_digest="1" * 64, matrix_index=1,
        baseline_repository=Path("/old/clone/baseline"), baseline_revision="HEAD",
        fixed_environment_id="env", fixed_environment_digest="2" * 64,
        generation_seed=1001, generation_seed_control="controlled", harness_id="pi",
        harness_definition_digest="3" * 64, profile_id="profile",
        profile_definition_digest="4" * 64, prompt_id="prompt",
        prompt_definition_digest="5" * 64, prompt_sha256="6" * 64,
        semantic_task_id="task", repetition_index=1, limits=RunLimits(),
    )
    restored = RunDefinition.model_validate_json(
        legacy.model_dump_json(exclude={"definition_digest": True, "limits": {"definition_digest": True}})
    )
    assert restored.identity_version == "1.0.0"
    assert restored.definition_digest == legacy.definition_digest
