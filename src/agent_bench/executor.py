"""Sequential, resumable experiment orchestration (M9B)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agent_bench.context_analysis import derive_context_analysis
from agent_bench.context_storage import (
    store_context_analysis_artifact,
    verify_context_analysis_artifact,
)
from agent_bench.matrix import expand_experiment
from agent_bench.metrics_storage import verify_metrics_artifact
from agent_bench.models import ExperimentDefinition, RunDefinition, canonical_sha256
from agent_bench.opencode_run import execute_controlled_opencode_run
from agent_bench.pi_run import execute_controlled_pi_run
from agent_bench.hermes_run import execute_controlled_hermes_run
from agent_bench.hermes import load_hermes_profile_for_id
from agent_bench.preservation import verify_artifact
from agent_bench.result_store import ResultStoreError, publish_result_ref, verify_published_result
from agent_bench.subject import FrozenSubject, materialize_baseline, verify_materialized_baseline

RunState = Literal[
    "pending", "preflight", "running", "preserving", "analyzing", "completed",
    "failed", "interrupted", "invalid",
]
TERMINAL = {"completed", "failed", "invalid"}
RunPhase = Literal["preflight", "running", "preserving", "analyzing"]
FailureDomain = Literal[
    "infrastructure_precondition", "backend_lifecycle", "proxy_lifecycle",
    "harness_runtime", "preservation", "analysis", "executor",
]


class ExecutorError(RuntimeError):
    """Raised when persisted executor state cannot be used safely."""


class RunProgress(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    execution_index: int
    state: RunState = "pending"
    detail: str | None = None
    failure_domain: FailureDomain | None = None
    failure_class: str | None = None
    failure_phase: RunPhase | None = None
    harness_execution_started: bool | None = None
    llm_request_observed: bool | None = None
    preservation_completed: bool | None = None


class ExperimentState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0.0", "1.1.0"] = "1.1.0"
    experiment_id: str
    definition_digest: str
    expansion_digest: str
    ordering: dict[str, object]
    runs: list[RunProgress]
    interrupted: bool = False
    circuit_breaker: dict[str, object] | None = None
    updated_at: str


@dataclass(frozen=True)
class DispatchOutcome:
    """Structured terminal result for one executor dispatch.

    This is intentionally limited to execution/provenance facts.  It does not
    turn a task-quality result into an executor failure.
    """

    passed: bool
    detail: str | None = None
    failure_domain: FailureDomain | None = None
    failure_class: str | None = None
    failure_phase: RunPhase | None = None
    harness_execution_started: bool | None = None
    llm_request_observed: bool | None = None
    preservation_completed: bool | None = None


Dispatch = Callable[[RunDefinition, Path], bool | DispatchOutcome]
PhaseReporter = Callable[[RunPhase], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def state_path(root: Path) -> Path:
    return root / "experiment-state.json"


def create_state(experiment: ExperimentDefinition) -> ExperimentState:
    runs = expand_experiment(experiment)
    expansion = canonical_sha256([run.model_dump(mode="json") for run in runs])
    return ExperimentState(
        experiment_id=experiment.experiment_id,
        definition_digest=experiment.definition_digest,
        expansion_digest=expansion,
        ordering=experiment.ordering.model_dump(mode="json"),
        runs=[RunProgress(run_id=run.run_id, execution_index=index) for index, run in enumerate(runs, 1)],
        updated_at=_now(),
    )


def write_state(root: Path, state: ExperimentState) -> None:
    """Atomically replace the small mutable progress record."""
    root.mkdir(parents=True, exist_ok=True)
    target = state_path(root)
    payload = state.model_copy(update={"updated_at": _now()}).model_dump(mode="json")
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_executor_event(root: Path, progress: RunProgress) -> None:
    """Append an operational progress event; state remains authoritative."""
    root.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _now(), "run_id": progress.run_id, "execution_index": progress.execution_index, "state": progress.state, "detail": progress.detail}
    with (root / "executor-events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_or_create(root: Path, experiment: ExperimentDefinition, resume: bool) -> ExperimentState:
    target = state_path(root)
    if not target.exists():
        if resume:
            raise ExecutorError("cannot resume: experiment state does not exist")
        state = create_state(experiment)
        write_state(root, state)
        return state
    state = ExperimentState.model_validate_json(target.read_bytes())
    expected = create_state(experiment)
    if state.definition_digest != expected.definition_digest or state.expansion_digest != expected.expansion_digest:
        raise ExecutorError("experiment definition or expansion differs from persisted state")
    if not resume:
        raise ExecutorError("experiment state already exists; use --resume")
    return state


def completed_artifact(root: Path, run_id: str) -> bool:
    artifact = root / "artifacts" / run_id
    try:
        manifest = verify_artifact(artifact)
        verify_metrics_artifact(root / "analysis" / run_id / "metrics-v1")
        verify_context_analysis_artifact(root / "analysis" / run_id / "context-analysis-v2")
        verify_published_result(root, manifest)
    except Exception:
        return False
    return (artifact / "normalized/events.jsonl").is_file()


@dataclass
class ExperimentExecutor:
    experiment: ExperimentDefinition
    output_root: Path
    dispatch: Dispatch

    def plan(self) -> tuple[RunDefinition, ...]:
        return expand_experiment(self.experiment)

    def run(self, *, resume: bool = False, limit: int | None = None, selected: set[str] | None = None) -> ExperimentState:
        if limit is not None and limit < 1:
            raise ExecutorError("limit must be at least one")
        state = load_or_create(self.output_root, self.experiment, resume)
        planned = {run.run_id: run for run in self.plan()}
        count = 0
        consecutive_infrastructure_failure: tuple[FailureDomain, str, int] | None = None
        for progress in state.runs:
            if selected is not None and progress.run_id not in selected:
                continue
            if progress.state == "completed":
                if completed_artifact(self.output_root, progress.run_id):
                    continue
                progress.state, progress.detail = "invalid", "completed-run integrity verification failed"
                write_state(self.output_root, state)
                _append_executor_event(self.output_root, progress)
                continue
            if progress.state in TERMINAL and progress.state != "completed":
                continue
            if limit is not None and count >= limit:
                break
            progress.state, progress.detail = "preflight", None
            progress.failure_domain = progress.failure_class = progress.failure_phase = None
            progress.harness_execution_started = progress.llm_request_observed = progress.preservation_completed = None
            write_state(self.output_root, state)
            _append_executor_event(self.output_root, progress)

            def report_phase(phase: RunPhase) -> None:
                progress.state = phase
                write_state(self.output_root, state)
                _append_executor_event(self.output_root, progress)

            reporter_setter = getattr(self.dispatch, "set_phase_reporter", None)
            controlled_phases = callable(reporter_setter)
            if controlled_phases:
                reporter_setter(report_phase)
            else:
                # Legacy/fake dispatches have no explicit preflight boundary.
                # Real controlled dispatches always report it before this state.
                report_phase("running")
            try:
                dispatched = self.dispatch(planned[progress.run_id], self.output_root)
                outcome = dispatched if isinstance(dispatched, DispatchOutcome) else DispatchOutcome(
                    passed=dispatched,
                    detail=None if dispatched else "run-local failure",
                    failure_domain=None if dispatched else "harness_runtime",
                    failure_phase=None if dispatched else "running",
                )
                progress.state = "completed" if outcome.passed else "failed"
                progress.detail = outcome.detail
                progress.failure_domain = outcome.failure_domain
                progress.failure_class = outcome.failure_class
                progress.failure_phase = outcome.failure_phase
                progress.harness_execution_started = outcome.harness_execution_started
                progress.llm_request_observed = outcome.llm_request_observed
                progress.preservation_completed = outcome.preservation_completed
            except (KeyboardInterrupt, SystemExit):
                progress.state, progress.detail = "interrupted", "executor interrupted"
                state.interrupted = True
                write_state(self.output_root, state)
                _append_executor_event(self.output_root, progress)
                raise
            except Exception as exc:
                active_phase = progress.state
                progress.state = "failed"
                progress.detail = f"{type(exc).__name__}: {exc}"
                progress.failure_domain = "executor"
                progress.failure_class = type(exc).__name__
                progress.failure_phase = active_phase if active_phase in {"preflight", "running", "preserving", "analyzing"} else "preflight"
                progress.harness_execution_started = False
                progress.llm_request_observed = False
                progress.preservation_completed = False
            write_state(self.output_root, state)
            _append_executor_event(self.output_root, progress)
            count += 1
            if (
                progress.state == "failed"
                and progress.failure_domain in {"infrastructure_precondition", "backend_lifecycle", "proxy_lifecycle"}
                and progress.failure_class
            ):
                key = (progress.failure_domain, progress.failure_class)
                previous = consecutive_infrastructure_failure
                consecutive_infrastructure_failure = (
                    key[0], key[1], (previous[2] + 1 if previous and previous[:2] == key else 1)
                )
            else:
                consecutive_infrastructure_failure = None
            if _trip_circuit_breaker(state, progress, consecutive_infrastructure_failure):
                write_state(self.output_root, state)
                _append_executor_event(self.output_root, progress)
                break
        return state


@dataclass
class _ControlledDispatch:
    experiment: ExperimentDefinition
    subject: FrozenSubject
    prompts: dict[str, str]
    phase_reporter: PhaseReporter | None = None

    def set_phase_reporter(self, reporter: PhaseReporter) -> None:
        self.phase_reporter = reporter

    def __call__(self, run: RunDefinition, output_root: Path) -> DispatchOutcome:
        baseline = output_root / "runtime" / "baselines" / run.run_id
        materialize_baseline(self.subject, baseline)
        preserved = False
        try:
            verify_materialized_baseline(baseline, self.subject.identity)
            local_run = run.model_copy(update={"baseline_repository": baseline, "baseline_revision": self.subject.identity.baseline_commit})
            arguments = {"run_definition": local_run, "prompt_content": self.prompts[run.prompt_id], "output_root": output_root, "phase_reporter": self.phase_reporter}
            if run.harness_id == "opencode":
                result = execute_controlled_opencode_run(**arguments)
            elif run.harness_id == "pi":
                result = execute_controlled_pi_run(**arguments)
            elif run.harness_id == "hermes":
                result = execute_controlled_hermes_run(
                    **arguments,
                    hermes_profile=load_hermes_profile_for_id(run.profile_id),
                )
            else:  # pragma: no cover - pydantic restricts harness IDs
                raise ExecutorError(f"unsupported harness {run.harness_id}")
            if result.failed_run is not None:
                manifest = result.failed_run.manifest
                domain, phase = _failure_taxonomy(manifest.failure_class)
                return DispatchOutcome(False, manifest.reason, domain, manifest.failure_class, phase,
                                       False, False, True)
            assert result.run is not None and result.metrics is not None
            verify_artifact(result.run.artifact_path, repository=baseline)
            verify_metrics_artifact(result.metrics.root)
            context_path = getattr(result, "context_analysis_path", None)
            if context_path is None:
                if self.phase_reporter is not None:
                    self.phase_reporter("analyzing")
                context_path = store_context_analysis_artifact(
                    source_artifact=result.run.artifact_path,
                    output_root=output_root / "analysis",
                    analysis=derive_context_analysis(result.run.artifact_path),
                )
            verify_context_analysis_artifact(context_path)
            publish_result_ref(output_root, baseline, result.run.artifact_manifest)
            verify_published_result(output_root, result.run.artifact_manifest)
            preserved = True
            return DispatchOutcome(True, preservation_completed=True)
        except Exception as exc:
            _preserve_result_store_failure(output_root, run.run_id, baseline, exc)
            raise
        finally:
            if preserved and baseline.exists():
                shutil.rmtree(baseline)


def controlled_dispatch(experiment: ExperimentDefinition, subject: FrozenSubject) -> Dispatch:
    """Create M9B dispatch from the existing controlled harness lifecycles."""
    if experiment.identity_version != "2.0.0" or experiment.portable_baseline != subject.identity:
        raise ExecutorError("experiment portable_baseline does not match the frozen subject")
    return _ControlledDispatch(experiment, subject, {prompt.prompt_id: prompt.content for prompt in experiment.prompts})


def _failure_taxonomy(failure_class: str) -> tuple[FailureDomain, RunPhase]:
    if failure_class in {"precondition_failed", "backend_identity_mismatch", "model_hash_mismatch", "template_hash_mismatch", "benchmark_port_in_use", "conflicting_gpu_process"}:
        return "infrastructure_precondition", "preflight"
    if failure_class in {"backend_start_failed", "backend_readiness_failed"}:
        return "backend_lifecycle", "running"
    if failure_class == "preservation_failed":
        return "preservation", "preserving"
    return "executor", "preflight"


def _trip_circuit_breaker(
    state: ExperimentState,
    progress: RunProgress,
    consecutive: tuple[FailureDomain, str, int] | None,
) -> bool:
    """Stop after two identical global infrastructure failures in one invocation."""
    if consecutive is None:
        return False
    domain, failure_class, count = consecutive
    if count < 2:
        return False
    state.circuit_breaker = {
        "schema_version": "1.0.0", "threshold": 2,
        "failure_domain": domain, "failure_class": failure_class,
        "trigger_run_id": progress.run_id, "message": "stopped after two identical infrastructure failures; no later matrix rows were attempted",
    }
    return True


def _preserve_result_store_failure(output_root: Path, run_id: str, baseline: Path, error: Exception) -> None:
    """Leave the source clone intact and write non-overwriting recovery evidence."""
    root = output_root / "preservation-failures"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{run_id}-result-store-failure.json"
    if target.exists():
        return
    payload = {"schema_version": "1.0.0", "run_id": run_id, "timestamp": _now(), "source_repository": str(baseline), "error_type": type(error).__name__, "error": str(error)}
    descriptor, temporary = tempfile.mkstemp(prefix=".failure-", dir=root)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except FileExistsError:
        pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def status(state: ExperimentState) -> dict[str, object]:
    names = ("pending", "preflight", "running", "preserving", "analyzing", "completed", "failed", "interrupted", "invalid")
    counts = {name: sum(run.state == name for run in state.runs) for name in names}
    return {"experiment_id": state.experiment_id, "total": len(state.runs), "counts": counts, "interrupted": state.interrupted, "expansion_digest": state.expansion_digest}
