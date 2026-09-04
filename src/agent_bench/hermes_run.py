"""One controlled M8 Hermes run against the fixed M5 backend."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from agent_bench.backend import BackendLifecycleError, BackendPreflightReport, BackendProfile, BackendReadinessFailed, BackendStartFailed, OwnedBackendProcess, load_backend_profile, preflight_backend, resolve_backend_invocation, seed_for_repetition, start_owned_backend
from agent_bench.failure import FailedRunEvidence, FailureEnvironmentRecord, preserve_failed_run
from agent_bench.hermes import HermesAdapter, HermesProfile, hermes_capture_capabilities, load_hermes_profile
from agent_bench.metrics import calculate_run_metrics
from agent_bench.metrics_storage import StoredMetrics, store_metrics_artifact
from agent_bench.context_storage import store_context_analysis_artifact
from agent_bench.models import RunDefinition
from agent_bench.opencode_run import _BackendProxyTaskService, _backend_paths
from agent_bench.runner import RunExecutionResult, execute_run
from agent_bench.supervisor import initialize_supervisor


@dataclass(frozen=True)
class ControlledHermesResult:
    run: RunExecutionResult | None
    metrics: StoredMetrics | None
    context_analysis_path: Path | None
    failed_run: FailedRunEvidence | None


def execute_controlled_hermes_run(*, run_definition: RunDefinition, prompt_content: str, output_root: Path, backend_profile: BackendProfile | None = None, hermes_profile: HermesProfile | None = None, phase_reporter: Callable[[str], None] | None = None) -> ControlledHermesResult:
    """Execute one M8 run or seal immutable pre-task backend failure evidence."""
    if run_definition.harness_id != "hermes": raise ValueError("controlled Hermes execution requires harness_id=hermes")
    output = output_root.expanduser().resolve()
    supervisor = initialize_supervisor(
        output_root=output,
        run_id=run_definition.run_id,
        argv=("agent-bench", "hermes", "controlled-run", run_definition.run_id),
    )
    try:
        backend = backend_profile or load_backend_profile()
        harness = hermes_profile or load_hermes_profile()
        run_seed = seed_for_repetition(run_definition.repetition_index)
    except Exception as exc:
        supervisor.mark("failure", error=exc)
        raise
    controls = output / "runtime" / "backend-control"; controls.mkdir(parents=True, exist_ok=True)
    control_root = Path(tempfile.mkdtemp(prefix=f"{run_definition.run_id}-", dir=controls)); backend_paths = _backend_paths(control_root)
    for path in (backend_paths.home, backend_paths.xdg_config_home, backend_paths.xdg_cache_home, backend_paths.xdg_data_home, backend_paths.xdg_state_home): path.mkdir(parents=True)
    report = preflight_backend(backend, backend_paths, run_seed=run_seed)
    _write_json(control_root / "profile.json", backend.model_dump(mode="json")); _write_json(control_root / "preflight.json", report.model_dump(mode="json")); _write_json(control_root / "invocation.json", resolve_backend_invocation(backend, backend_paths, run_seed=run_seed).model_dump(mode="json"))
    if not report.passed:
        failed = _preserve_failure(output, run_definition.run_id, backend, backend_paths, report, run_seed); shutil.rmtree(control_root); return ControlledHermesResult(None, None, None, failed)
    if phase_reporter is not None:
        phase_reporter("running")
    owned: OwnedBackendProcess | None = None; service: _BackendProxyTaskService | None = None; failure_class = "backend_start_failed"
    try:
        try: owned = start_owned_backend(backend, backend_paths, report, control_root / "stdout.log", control_root / "stderr.log", run_seed=run_seed)
        except Exception as exc: raise BackendLifecycleError(str(exc)) from exc
        try: startup_ns = owned.wait_until_ready(backend)
        except BackendStartFailed as exc: raise BackendLifecycleError(str(exc)) from exc
        except BackendReadinessFailed as exc: failure_class = "backend_readiness_failed"; raise BackendLifecycleError(str(exc)) from exc
        service = _BackendProxyTaskService(profile=backend, preflight=report, owned=owned, startup_ns=startup_ns, control_root=control_root, run_seed=run_seed)
        result = execute_run(run_definition=run_definition, prompt_content=prompt_content, adapter=HermesAdapter(harness), artifacts_root=output / "artifacts", worktrees_root=output / "worktrees", isolation_root=output / "runtime" / "harness", proxy_endpoint=harness.proxy_base_url, run_seed=run_seed, task_service=service)
        if phase_reporter is not None:
            phase_reporter("analyzing")
        metrics = calculate_run_metrics(result.artifact_path); stored = store_metrics_artifact(source_artifact=result.artifact_path, output_root=output / "analysis", metrics=metrics)
        from agent_bench.context_analysis import derive_context_analysis
        context_analysis = store_context_analysis_artifact(source_artifact=result.artifact_path, output_root=output / "analysis", analysis=derive_context_analysis(result.artifact_path))
        shutil.rmtree(control_root); return ControlledHermesResult(result, stored, context_analysis, None)
    except BackendLifecycleError as exc:
        if owned is not None and owned.process.poll() is None: owned.shutdown(backend.shutdown_grace_seconds)
        failed = preserve_failed_run(runs_root=output / "runs", run_id=run_definition.run_id, failure_class=failure_class, reason=str(exc), environment=_failure_environment(run_definition.run_id, backend, backend_paths, report, run_seed), stdout=(control_root / "stdout.log").read_bytes() if (control_root / "stdout.log").is_file() else b"", stderr=(control_root / "stderr.log").read_bytes() if (control_root / "stderr.log").is_file() else b"")  # type: ignore[arg-type]
        shutil.rmtree(control_root); return ControlledHermesResult(None, None, None, failed)
    finally:
        if service is not None and not service.stopped: service.force_stop()
        elif owned is not None and owned.process.poll() is None: owned.shutdown(backend.shutdown_grace_seconds)


def _failure_environment(run_id: str, profile: BackendProfile, paths: object, report: BackendPreflightReport, run_seed: int) -> FailureEnvironmentRecord:
    return FailureEnvironmentRecord(run_id=run_id, backend_profile_digest=profile.definition_digest, preflight=report, invocation=resolve_backend_invocation(profile, paths, run_seed=run_seed, failure_logs=True), capture_capabilities=hermes_capture_capabilities())  # type: ignore[arg-type]


def _preserve_failure(output: Path, run_id: str, profile: BackendProfile, paths: object, report: BackendPreflightReport, run_seed: int) -> FailedRunEvidence:
    failed = next(check for check in report.checks if not check.passed)
    return preserve_failed_run(runs_root=output / "runs", run_id=run_id, failure_class=report.primary_failure_class or "precondition_failed", reason=failed.message, environment=_failure_environment(run_id, profile, paths, report, run_seed))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8", newline="\n")
