"""Agent Bench command-line interface."""

import json
import shlex
from pathlib import Path
from typing import Annotated

import typer
from pydantic import TypeAdapter

from agent_bench import __version__
from agent_bench.backend import (
    BackendLifecycleError,
    BackendReadinessFailed,
    BackendRunPaths,
    BackendStartFailed,
    load_backend_profile,
    preflight_backend,
    resolve_backend_invocation,
    seed_for_repetition,
    start_owned_backend,
)
from agent_bench.capture import fixed_proxy_capture_capabilities
from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.fake_harness import FAKE_SCENARIOS, FakeHarness
from agent_bench.failure import (
    FailedRunEvidenceError,
    FailureEnvironmentRecord,
    preserve_failed_run,
)
from agent_bench.git import GitOperationError, resolve_baseline
from agent_bench.matrix import expand_experiment, generate_run_definitions
from agent_bench.metrics import MetricsCalculationError, calculate_run_metrics
from agent_bench.metrics_storage import (
    MetricsStorageError,
    store_metrics_artifact,
    verify_metrics_artifact,
)
from agent_bench.models import ExperimentDefinition, Identifier
from agent_bench.opencode import (
    OpenCodeError,
    inspect_opencode_executable,
    load_opencode_profile,
)
from agent_bench.opencode_run import execute_controlled_opencode_run
from agent_bench.preservation import (
    PreservationError,
    restore_artifact,
    verify_artifact,
)
from agent_bench.runner import RunLifecycleError, execute_run

app = typer.Typer(
    add_completion=False,
    help="Deterministic benchmark framework for coding-agent harnesses.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(
    help="Validate and expand experiment definitions without executing runs.",
    no_args_is_help=True,
)
app.add_typer(experiment_app, name="experiment")
git_app = typer.Typer(
    help="Inspect Git baseline identity without creating a run.",
    no_args_is_help=True,
)
app.add_typer(git_app, name="git")
artifact_app = typer.Typer(
    help="Verify or restore an immutable preserved artifact.",
    no_args_is_help=True,
)
app.add_typer(artifact_app, name="artifact")
metrics_app = typer.Typer(
    help="Calculate or inspect deterministic metrics for a preserved run.",
    no_args_is_help=True,
)
app.add_typer(metrics_app, name="metrics")
backend_app = typer.Typer(
    help="Inspect or explicitly probe the fixed benchmark-v1 llama.cpp backend.",
    no_args_is_help=True,
)
app.add_typer(backend_app, name="backend")
opencode_app = typer.Typer(
    help="Inspect or run the pinned M6 OpenCode integration.",
    no_args_is_help=True,
)
app.add_typer(opencode_app, name="opencode")
_RUN_ID_ADAPTER = TypeAdapter(Identifier)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the Agent Bench version and exit."),
    ] = False,
) -> None:
    """Show help or version information."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@experiment_app.command("validate")
def validate_experiment(path: Path) -> None:
    """Validate an experiment and all referenced prompt identities."""
    experiment = _load_or_exit(path)
    run_count = len(generate_run_definitions(experiment))
    typer.echo(
        f"Valid experiment {experiment.experiment_id!r}: {run_count} runs, "
        f"ordering={experiment.ordering.mode}"
    )


@experiment_app.command("expand")
def expand_experiment_command(
    path: Path,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the expanded matrix as JSON."),
    ] = False,
) -> None:
    """Print the deterministic matrix without executing any runs."""
    experiment = _load_or_exit(path)
    runs = expand_experiment(experiment)
    if json_output:
        payload = {
            "schema_version": "1.0.0",
            "experiment_id": experiment.experiment_id,
            "matrix_digest": experiment.matrix_digest,
            "ordering": experiment.ordering.model_dump(mode="json"),
            "runs": [
                {
                    "execution_position": position,
                    **run.model_dump(mode="json"),
                }
                for position, run in enumerate(runs, start=1)
            ],
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    typer.echo("POSITION\tRUN ID\tHARNESS\tPROFILE\tPROMPT\tREPETITION")
    for position, run in enumerate(runs, start=1):
        typer.echo(
            f"{position}\t{run.run_id}\t{run.harness_id}\t{run.profile_id}\t"
            f"{run.prompt_id}\t{run.repetition_index}"
        )


@git_app.command("baseline")
def git_baseline(path: Path, reference: str) -> None:
    """Resolve a Git baseline reference to an immutable commit."""
    try:
        baseline = resolve_baseline(path, reference)
    except GitOperationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"repository={baseline.repository}\n"
        f"reference={baseline.requested_ref}\n"
        f"commit={baseline.commit}"
    )


@artifact_app.command("verify")
def artifact_verify(path: Path) -> None:
    """Verify a preserved result and its reachable Git result ref."""
    try:
        manifest = verify_artifact(path)
    except PreservationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Verified artifact {manifest.run_id!r}: "
        f"snapshot={manifest.source_snapshot_sha256}"
    )


@artifact_app.command("restore")
def artifact_restore(path: Path, destination: Path) -> None:
    """Verify and restore a preserved source snapshot."""
    try:
        manifest = restore_artifact(path, destination)
    except PreservationError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Restored artifact {manifest.run_id!r} to {destination.resolve()}")


@metrics_app.command("calculate")
def metrics_calculate(source_artifact: Path, output_root: Path) -> None:
    """Calculate and seal a separate immutable metrics artifact."""
    try:
        metrics = calculate_run_metrics(source_artifact)
        stored = store_metrics_artifact(
            source_artifact=source_artifact,
            output_root=output_root,
            metrics=metrics,
        )
    except (MetricsCalculationError, MetricsStorageError, PreservationError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"run_id={metrics.run_id}\n"
        f"termination={metrics.termination.termination_class}\n"
        f"metrics_sha256={stored.manifest.metrics_sha256}\n"
        f"metrics_artifact={stored.root}"
    )


@metrics_app.command("show")
def metrics_show(path: Path) -> None:
    """Print validated metrics JSON, calculating in memory for a run artifact."""
    resolved = path.expanduser().resolve()
    try:
        if resolved.name == "metrics.json" or (resolved / "metrics.json").is_file():
            metrics = verify_metrics_artifact(resolved).metrics
        else:
            metrics = calculate_run_metrics(resolved)
    except (MetricsCalculationError, MetricsStorageError, PreservationError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            metrics.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


@backend_app.command("validate")
def backend_validate() -> None:
    """Validate the checked-in fixed profile schema without hashing the model."""
    try:
        profile = load_backend_profile()
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"profile_id={profile.profile_id}\n"
        f"profile_digest={profile.definition_digest}\n"
        f"model_sha256={profile.model.sha256}\n"
        f"template_sha256={profile.chat_template.sha256}\n"
        f"llama_cpp_commit={profile.llama_cpp_commit}"
    )


@backend_app.command("show-command")
def backend_show_command(
    run_root: Annotated[
        Path,
        typer.Option(help="Hypothetical absolute isolated run root used in the environment."),
    ] = Path("/tmp/agent-bench-backend-v1"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the invocation record as JSON."),
    ] = False,
    repetition: Annotated[
        int,
        typer.Option("--repetition", min=1, help="One-based repetition seed source."),
    ] = 1,
) -> None:
    """Show exact argv and allowlisted environment without starting a process."""
    try:
        profile = load_backend_profile()
        paths = _backend_paths(run_root.expanduser().resolve())
        invocation = resolve_backend_invocation(
            profile, paths, run_seed=seed_for_repetition(repetition)
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(
            json.dumps(
                invocation.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    typer.echo("argv=" + shlex.join(invocation.argv))
    typer.echo(
        "environment="
        + json.dumps(invocation.environment, ensure_ascii=False, sort_keys=True)
    )


@backend_app.command("preflight")
def backend_preflight(
    run_id: str,
    runtime_root: Path,
    runs_root: Path,
    repetition: Annotated[
        int,
        typer.Option("--repetition", min=1, help="One-based repetition seed source."),
    ] = 1,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the successful preflight report as JSON."),
    ] = False,
) -> None:
    """Run full preflight and seal immutable evidence when it fails."""
    try:
        profile = load_backend_profile()
        paths = _create_backend_paths(runtime_root, run_id)
        run_seed = seed_for_repetition(repetition)
        report = preflight_backend(profile, paths, run_seed=run_seed)
        if not report.passed:
            evidence = _preserve_preflight_failure(
                run_id, runs_root, profile, paths, report, run_seed
            )
            typer.echo(
                f"preflight=failed\n"
                f"failure_class={report.primary_failure_class}\n"
                f"evidence={evidence.root}"
            )
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (ValueError, BackendLifecycleError, FailedRunEvidenceError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    else:
        typer.echo(
            f"preflight=passed\nprofile_digest={profile.definition_digest}\n"
            f"run_seed={run_seed}"
        )


@backend_app.command("probe")
def backend_probe(
    run_id: str,
    runtime_root: Path,
    runs_root: Path,
    repetition: Annotated[
        int,
        typer.Option("--repetition", min=1, help="One-based repetition seed source."),
    ] = 1,
) -> None:
    """Explicitly start, readiness-check, and stop one owned backend process."""
    profile = load_backend_profile()
    paths = _create_backend_paths(runtime_root, run_id)
    run_seed = seed_for_repetition(repetition)
    report = preflight_backend(profile, paths, run_seed=run_seed)
    if not report.passed:
        evidence = _preserve_preflight_failure(
            run_id, runs_root, profile, paths, report, run_seed
        )
        typer.echo(f"preflight=failed\nevidence={evidence.root}", err=True)
        raise typer.Exit(code=1)
    log_root = runtime_root.expanduser().resolve() / run_id / "backend-logs"
    stdout_path = log_root / "stdout.log"
    stderr_path = log_root / "stderr.log"
    owned = None
    failure_class = None
    try:
        try:
            owned = start_owned_backend(
                profile, paths, report, stdout_path, stderr_path,
                run_seed=run_seed,
            )
        except Exception as exc:
            failure_class = "backend_start_failed"
            raise BackendLifecycleError(str(exc)) from exc
        try:
            startup_ns = owned.wait_until_ready(profile)
        except BackendStartFailed as exc:
            failure_class = "backend_start_failed"
            raise BackendLifecycleError(str(exc)) from exc
        except BackendReadinessFailed as exc:
            failure_class = "backend_readiness_failed"
            raise BackendLifecycleError(str(exc)) from exc
        except Exception as exc:
            failure_class = "backend_readiness_failed"
            raise BackendLifecycleError(str(exc)) from exc
        typer.echo(f"readiness=passed\nstartup_ns={startup_ns}")
    except BackendLifecycleError as exc:
        if owned is not None:
            owned.shutdown(profile.shutdown_grace_seconds)
        environment = _failure_environment(
            run_id, profile, paths, report, run_seed
        )
        evidence = preserve_failed_run(
            runs_root=runs_root,
            run_id=run_id,
            failure_class=failure_class or "backend_start_failed",
            reason=str(exc),
            environment=environment,
            stdout=stdout_path.read_bytes() if stdout_path.is_file() else b"",
            stderr=stderr_path.read_bytes() if stderr_path.is_file() else b"",
        )
        typer.echo(f"Error: {exc}\nevidence={evidence.root}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        if owned is not None and owned.process.poll() is None:
            code, method = owned.shutdown(profile.shutdown_grace_seconds)
            typer.echo(f"shutdown={method}\nexit_code={code}")


@opencode_app.command("inspect")
def opencode_inspect() -> None:
    """Inspect the pinned executable and controlled profile without running a task."""
    try:
        profile = load_opencode_profile()
        observed = inspect_opencode_executable(profile.executable.path)
    except OpenCodeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    matches = observed == profile.executable
    typer.echo(
        f"profile_id={profile.profile_id}\n"
        f"profile_digest={profile.definition_digest}\n"
        f"config_sha256={profile.config_sha256}\n"
        f"executable={observed.path}\n"
        f"version={observed.version}\n"
        f"executable_sha256={observed.sha256}\n"
        f"runtime_identity={observed.runtime_identity}\n"
        f"pinned_identity_match={str(matches).lower()}"
    )
    if not matches:
        raise typer.Exit(code=1)


@opencode_app.command("run")
def opencode_run(
    experiment_path: Path,
    run_id: str,
    output_root: Path,
) -> None:
    """Execute exactly one selected OpenCode run with the fixed M5 backend."""
    experiment = _load_or_exit(experiment_path)
    run_definition = next(
        (run for run in expand_experiment(experiment) if run.run_id == run_id),
        None,
    )
    if run_definition is None:
        typer.echo(f"Error: run ID is not in the expanded experiment: {run_id}", err=True)
        raise typer.Exit(code=1)
    if run_definition.harness_id != "opencode":
        typer.echo("Error: selected run is not an OpenCode run", err=True)
        raise typer.Exit(code=1)
    if run_definition.profile_id != "opencode-default-v1":
        typer.echo("Error: M6 supports only profile opencode-default-v1", err=True)
        raise typer.Exit(code=1)
    prompt = next(
        item for item in experiment.prompts if item.prompt_id == run_definition.prompt_id
    )
    try:
        controlled = execute_controlled_opencode_run(
            run_definition=run_definition,
            prompt_content=prompt.content,
            output_root=output_root,
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if controlled.failed_run is not None:
        typer.echo(
            f"preflight=failed\n"
            f"failure_class={controlled.failed_run.manifest.failure_class}\n"
            f"evidence={controlled.failed_run.root}",
            err=True,
        )
        raise typer.Exit(code=1)
    assert controlled.run is not None and controlled.metrics is not None
    typer.echo(
        f"run_id={controlled.run.run_manifest.run_id}\n"
        f"outcome={controlled.run.run_manifest.observed_execution_outcome}\n"
        f"artifact={controlled.run.artifact_path}\n"
        f"metrics_artifact={controlled.metrics.root}\n"
        f"termination={controlled.metrics.metrics.termination.termination_class}"
    )


@app.command("fake-run")
def fake_run(
    experiment_path: Path,
    run_id: str,
    output_root: Path,
    scenario: Annotated[
        str,
        typer.Option(
            "--scenario",
            help="Deterministic FakeHarness scenario.",
        ),
    ] = "success",
) -> None:
    """Execute exactly one expanded run with the test-only FakeHarness."""
    experiment = _load_or_exit(experiment_path)
    if scenario not in FAKE_SCENARIOS:
        typer.echo(
            "Error: unsupported FakeHarness scenario; choose one of: "
            + ", ".join(FAKE_SCENARIOS),
            err=True,
        )
        raise typer.Exit(code=1)
    run_definition = next(
        (run for run in expand_experiment(experiment) if run.run_id == run_id),
        None,
    )
    if run_definition is None:
        typer.echo(f"Error: run ID is not in the expanded experiment: {run_id}", err=True)
        raise typer.Exit(code=1)
    prompt = next(
        prompt
        for prompt in experiment.prompts
        if prompt.prompt_id == run_definition.prompt_id
    )
    resolved_output = output_root.expanduser().resolve()
    try:
        result = execute_run(
            run_definition=run_definition,
            prompt_content=prompt.content,
            adapter=FakeHarness(scenario),  # type: ignore[arg-type]
            adapter_scenario=scenario,
            artifacts_root=resolved_output / "artifacts",
            worktrees_root=resolved_output / "worktrees",
            isolation_root=resolved_output / "runtime",
        )
    except (RunLifecycleError, GitOperationError, PreservationError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"run_id={result.run_manifest.run_id}\n"
        f"scenario={scenario}\n"
        f"outcome={result.run_manifest.observed_execution_outcome}\n"
        f"raw_events={result.raw_event_path}\n"
        f"normalized_events={result.normalized_event_path}\n"
        f"artifact={result.artifact_path}"
    )


def _backend_paths(root: Path) -> BackendRunPaths:
    return BackendRunPaths(
        home=root / "home",
        xdg_config_home=root / "xdg-config",
        xdg_cache_home=root / "xdg-cache",
        xdg_data_home=root / "xdg-data",
        xdg_state_home=root / "xdg-state",
    )


def _create_backend_paths(runtime_root: Path, run_id: str) -> BackendRunPaths:
    run_id = _RUN_ID_ADAPTER.validate_python(run_id)
    root = runtime_root.expanduser().resolve() / run_id
    if root.exists():
        raise BackendLifecycleError(f"backend runtime already exists: {root}")
    paths = _backend_paths(root)
    for path in (
        paths.home,
        paths.xdg_config_home,
        paths.xdg_cache_home,
        paths.xdg_data_home,
        paths.xdg_state_home,
    ):
        path.mkdir(parents=True)
    return paths


def _failure_environment(
    run_id: str,
    profile: object,
    paths: BackendRunPaths,
    report: object,
    run_seed: int,
) -> FailureEnvironmentRecord:
    from agent_bench.backend import BackendPreflightReport, BackendProfile

    assert isinstance(profile, BackendProfile)
    assert isinstance(report, BackendPreflightReport)
    return FailureEnvironmentRecord(
        run_id=run_id,
        backend_profile_digest=profile.definition_digest,
        preflight=report,
        invocation=resolve_backend_invocation(
            profile, paths, run_seed=run_seed, failure_logs=True
        ),
        capture_capabilities=fixed_proxy_capture_capabilities(),
    )


def _preserve_preflight_failure(
    run_id: str,
    runs_root: Path,
    profile: object,
    paths: BackendRunPaths,
    report: object,
    run_seed: int,
):
    from agent_bench.backend import BackendPreflightReport, BackendProfile

    assert isinstance(profile, BackendProfile)
    assert isinstance(report, BackendPreflightReport)
    failed = next(check for check in report.checks if not check.passed)
    version_check = next(
        (check for check in report.checks if check.check_id == "backend-version"),
        None,
    )
    version_evidence = version_check.evidence if version_check is not None else {}
    return preserve_failed_run(
        runs_root=runs_root,
        run_id=run_id,
        failure_class=report.primary_failure_class or "precondition_failed",
        reason=failed.message,
        environment=_failure_environment(run_id, profile, paths, report, run_seed),
        stdout=str(version_evidence.get("stdout", "")).encode("utf-8"),
        stderr=str(version_evidence.get("stderr", "")).encode("utf-8"),
    )


def _load_or_exit(path: Path) -> ExperimentDefinition:
    try:
        return load_experiment(path)
    except ExperimentConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
