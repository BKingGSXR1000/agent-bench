"""One controlled M6 OpenCode run against the fixed M5 backend."""

from __future__ import annotations

import json
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from agent_bench.backend import (
    BackendLifecycleError,
    BackendPreflightReport,
    BackendProfile,
    BackendReadinessFailed,
    BackendRunPaths,
    BackendStartFailed,
    OwnedBackendProcess,
    collect_nvidia_observations,
    is_port_free,
    load_backend_profile,
    preflight_backend,
    resolve_backend_invocation,
    seed_for_repetition,
    start_owned_backend,
)
from agent_bench.failure import FailedRunEvidence, FailureEnvironmentRecord, preserve_failed_run
from agent_bench.harness import EventSink, RunTaskService
from agent_bench.metrics import calculate_run_metrics
from agent_bench.metrics_storage import StoredMetrics, store_metrics_artifact
from agent_bench.models import RunDefinition
from agent_bench.opencode import (
    OpenCodeAdapter,
    OpenCodeProfile,
    load_opencode_profile,
    opencode_capture_capabilities,
)
from agent_bench.proxy import LoggingProxy, ProxyAddress
from agent_bench.runner import RunExecutionResult, execute_run


@dataclass(frozen=True)
class ControlledOpenCodeResult:
    run: RunExecutionResult | None
    metrics: StoredMetrics | None
    failed_run: FailedRunEvidence | None


class _BackendProxyTaskService(RunTaskService):
    def __init__(
        self,
        *,
        profile: BackendProfile,
        preflight: BackendPreflightReport,
        owned: OwnedBackendProcess,
        startup_ns: int,
        control_root: Path,
        run_seed: int,
    ) -> None:
        self.profile = profile
        self.preflight = preflight
        self.owned = owned
        self.startup_ns = startup_ns
        self.control_root = control_root
        self.run_seed = run_seed
        self.proxy: LoggingProxy | None = None
        self.stopped = False

    def start(self, events: EventSink) -> None:
        self.proxy = LoggingProxy(
            upstream=ProxyAddress(self.profile.server.host, self.profile.server.port),
            bind=ProxyAddress(self.profile.server.host, self.profile.server.proxy_port),
            events=events,
            sampling_baseline=self.profile.sampling,
            intended_seed=self.run_seed,
            configured_max_context_tokens=self.profile.server.context_size,
        )
        self.proxy.start()
        if self.proxy.address != ProxyAddress("127.0.0.1", 18081):
            self.proxy.shutdown()
            raise BackendLifecycleError("capture proxy did not bind the fixed benchmark endpoint")

    def stop(self, events: EventSink) -> tuple[tuple[str, Path], ...]:
        if self.stopped:
            return self._evidence()
        self.stopped = True
        proxy_error: str | None = None
        if self.proxy is not None:
            try:
                self.proxy.shutdown()
            except Exception as exc:
                proxy_error = f"{type(exc).__name__}: {exc}"
                events.emit(
                    source="proxy",
                    event_type="backend_error",
                    payload={"error_type": "proxy_shutdown_failure", "message": proxy_error},
                    timed=False,
                )
        exited_during_task = self.owned.process.poll()
        if exited_during_task is not None:
            events.emit(
                source="backend",
                event_type="backend_error",
                payload={
                    "error_type": "backend_exited_during_harness_run",
                    "exit_code": exited_during_task,
                },
                timed=False,
            )
        exit_code, shutdown_method = self.owned.shutdown(
            self.profile.shutdown_grace_seconds
        )
        try:
            gpus, processes = collect_nvidia_observations()
            gpu_evidence: object = {
                "availability": "available",
                "gpus": [item.model_dump(mode="json") for item in gpus],
                "processes": [item.model_dump(mode="json") for item in processes],
            }
        except Exception as exc:
            gpu_evidence = {
                "availability": "unavailable",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        _write_json(
            self.control_root / "lifecycle.json",
            {
                "schema_version": "1.0.0",
                "startup_ns": self.startup_ns,
                "backend_exit_code_observed_after_task": exited_during_task,
                "backend_shutdown_exit_code": exit_code,
                "backend_shutdown_method": shutdown_method,
                "proxy_shutdown_error": proxy_error,
                "backend_port_released": not _port_has_listener(
                    self.profile.server.host, self.profile.server.port
                ),
                "proxy_port_released": not _port_has_listener(
                    self.profile.server.host, self.profile.server.proxy_port
                ),
                "backend_port_bindable": is_port_free(
                    self.profile.server.host, self.profile.server.port
                ),
                "proxy_port_bindable": is_port_free(
                    self.profile.server.host, self.profile.server.proxy_port
                ),
                "post_shutdown_gpu_observation": gpu_evidence,
            },
        )
        return self._evidence()

    def force_stop(self) -> None:
        if self.proxy is not None:
            try:
                self.proxy.shutdown()
            except Exception:
                pass
        if self.owned.process.poll() is None:
            self.owned.shutdown(self.profile.shutdown_grace_seconds)

    def _evidence(self) -> tuple[tuple[str, Path], ...]:
        mapping = {
            "raw/backend/stdout.log": self.control_root / "stdout.log",
            "raw/backend/stderr.log": self.control_root / "stderr.log",
            "run/backend/profile.json": self.control_root / "profile.json",
            "run/backend/preflight.json": self.control_root / "preflight.json",
            "run/backend/invocation.json": self.control_root / "invocation.json",
            "run/backend/lifecycle.json": self.control_root / "lifecycle.json",
        }
        return tuple(sorted(mapping.items()))


def execute_controlled_opencode_run(
    *,
    run_definition: RunDefinition,
    prompt_content: str,
    output_root: Path,
    backend_profile: BackendProfile | None = None,
    opencode_profile: OpenCodeProfile | None = None,
    phase_reporter: Callable[[str], None] | None = None,
) -> ControlledOpenCodeResult:
    """Execute one M6 run or seal pre-task backend failure evidence."""
    if run_definition.harness_id != "opencode":
        raise ValueError("controlled OpenCode execution requires harness_id=opencode")
    output = output_root.expanduser().resolve()
    backend = backend_profile or load_backend_profile()
    harness = opencode_profile or load_opencode_profile()
    run_seed = seed_for_repetition(run_definition.repetition_index)
    controls = output / "runtime" / "backend-control"
    controls.mkdir(parents=True, exist_ok=True)
    control_root = Path(
        tempfile.mkdtemp(prefix=f"{run_definition.run_id}-", dir=controls)
    )
    backend_paths = _backend_paths(control_root)
    for path in (
        backend_paths.home,
        backend_paths.xdg_config_home,
        backend_paths.xdg_cache_home,
        backend_paths.xdg_data_home,
        backend_paths.xdg_state_home,
    ):
        path.mkdir(parents=True)
    report = preflight_backend(backend, backend_paths, run_seed=run_seed)
    invocation = resolve_backend_invocation(backend, backend_paths, run_seed=run_seed)
    _write_json(control_root / "profile.json", backend.model_dump(mode="json"))
    _write_json(control_root / "preflight.json", report.model_dump(mode="json"))
    _write_json(control_root / "invocation.json", invocation.model_dump(mode="json"))
    if not report.passed:
        failed = _preserve_failure(
            output, run_definition.run_id, backend, backend_paths, report, run_seed
        )
        shutil.rmtree(control_root)
        return ControlledOpenCodeResult(None, None, failed)

    if phase_reporter is not None:
        phase_reporter("running")

    owned: OwnedBackendProcess | None = None
    service: _BackendProxyTaskService | None = None
    failure_class = "backend_start_failed"
    try:
        try:
            owned = start_owned_backend(
                backend,
                backend_paths,
                report,
                control_root / "stdout.log",
                control_root / "stderr.log",
                run_seed=run_seed,
            )
        except Exception as exc:
            raise BackendLifecycleError(str(exc)) from exc
        try:
            startup_ns = owned.wait_until_ready(backend)
        except BackendStartFailed as exc:
            raise BackendLifecycleError(str(exc)) from exc
        except BackendReadinessFailed as exc:
            failure_class = "backend_readiness_failed"
            raise BackendLifecycleError(str(exc)) from exc
        service = _BackendProxyTaskService(
            profile=backend,
            preflight=report,
            owned=owned,
            startup_ns=startup_ns,
            control_root=control_root,
            run_seed=run_seed,
        )
        result = execute_run(
            run_definition=run_definition,
            prompt_content=prompt_content,
            adapter=OpenCodeAdapter(harness),
            artifacts_root=output / "artifacts",
            worktrees_root=output / "worktrees",
            isolation_root=output / "runtime" / "harness",
            proxy_endpoint=harness.proxy_base_url,
            run_seed=run_seed,
            task_service=service,
        )
        if phase_reporter is not None:
            phase_reporter("analyzing")
        metrics = calculate_run_metrics(result.artifact_path)
        stored = store_metrics_artifact(
            source_artifact=result.artifact_path,
            output_root=output / "analysis",
            metrics=metrics,
        )
        shutil.rmtree(control_root)
        return ControlledOpenCodeResult(result, stored, None)
    except BackendLifecycleError as exc:
        if owned is not None and owned.process.poll() is None:
            owned.shutdown(backend.shutdown_grace_seconds)
        failed = preserve_failed_run(
            runs_root=output / "runs",
            run_id=run_definition.run_id,
            failure_class=failure_class,  # type: ignore[arg-type]
            reason=str(exc),
            environment=_failure_environment(
                run_definition.run_id, backend, backend_paths, report, run_seed
            ),
            stdout=(control_root / "stdout.log").read_bytes()
            if (control_root / "stdout.log").is_file()
            else b"",
            stderr=(control_root / "stderr.log").read_bytes()
            if (control_root / "stderr.log").is_file()
            else b"",
        )
        shutil.rmtree(control_root)
        return ControlledOpenCodeResult(None, None, failed)
    finally:
        if service is not None and not service.stopped:
            service.force_stop()
        elif owned is not None and owned.process.poll() is None:
            owned.shutdown(backend.shutdown_grace_seconds)


def _backend_paths(root: Path) -> BackendRunPaths:
    return BackendRunPaths(
        home=root / "home",
        xdg_config_home=root / "xdg-config",
        xdg_cache_home=root / "xdg-cache",
        xdg_data_home=root / "xdg-data",
        xdg_state_home=root / "xdg-state",
    )


def _failure_environment(
    run_id: str,
    profile: BackendProfile,
    paths: BackendRunPaths,
    report: BackendPreflightReport,
    run_seed: int,
) -> FailureEnvironmentRecord:
    return FailureEnvironmentRecord(
        run_id=run_id,
        backend_profile_digest=profile.definition_digest,
        preflight=report,
        invocation=resolve_backend_invocation(
            profile, paths, run_seed=run_seed, failure_logs=True
        ),
        capture_capabilities=opencode_capture_capabilities(),
    )


def _preserve_failure(
    output: Path,
    run_id: str,
    profile: BackendProfile,
    paths: BackendRunPaths,
    report: BackendPreflightReport,
    run_seed: int,
) -> FailedRunEvidence:
    failed = next(check for check in report.checks if not check.passed)
    return preserve_failed_run(
        runs_root=output / "runs",
        run_id=run_id,
        failure_class=report.primary_failure_class or "precondition_failed",
        reason=failed.message,
        environment=_failure_environment(run_id, profile, paths, report, run_seed),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _port_has_listener(host: str, port: int) -> bool:
    """Check listener reachability without conflating it with TCP TIME_WAIT."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.25)
        return connection.connect_ex((host, port)) == 0
