"""Agent Bench command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer

from agent_bench import __version__
from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.fake_harness import FAKE_SCENARIOS, FakeHarness
from agent_bench.git import GitOperationError, resolve_baseline
from agent_bench.matrix import expand_experiment, generate_run_definitions
from agent_bench.metrics import MetricsCalculationError, calculate_run_metrics
from agent_bench.metrics_storage import (
    MetricsStorageError,
    store_metrics_artifact,
    verify_metrics_artifact,
)
from agent_bench.models import ExperimentDefinition
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


def _load_or_exit(path: Path) -> ExperimentDefinition:
    try:
        return load_experiment(path)
    except ExperimentConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
