import json
from pathlib import Path

from typer.testing import CliRunner

from agent_bench import __version__
from agent_bench.cli import app
from conftest import ExperimentFixture, GitRepositoryFixture

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Deterministic benchmark framework" in result.output
    assert "--version" in result.output
    assert "experiment" in result.output
    assert "git" in result.output
    assert "artifact" in result.output


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_cli_validates_experiment(experiment_fixture: ExperimentFixture) -> None:
    result = runner.invoke(
        app, ["experiment", "validate", str(experiment_fixture.path)]
    )

    assert result.exit_code == 0
    assert "Valid experiment 'test-experiment': 24 runs" in result.output


def test_cli_expands_experiment_as_json(
    experiment_fixture: ExperimentFixture,
) -> None:
    result = runner.invoke(
        app,
        ["experiment", "expand", str(experiment_fixture.path), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ordering"]["mode"] == "canonical"
    assert len(payload["runs"]) == 24
    assert payload["runs"][0]["execution_position"] == 1
    assert len({run["run_id"] for run in payload["runs"]}) == 24


def test_cli_reports_invalid_experiment(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment_fixture.path.write_text("prompts: []\n", encoding="utf-8")

    result = runner.invoke(
        app, ["experiment", "validate", str(experiment_fixture.path)]
    )

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "prompts must be a non-empty YAML list" in result.output


def test_checked_in_example_expands() -> None:
    example = (
        Path(__file__).parents[1] / "experiments" / "examples" / "m1-example.yaml"
    )

    result = runner.invoke(app, ["experiment", "expand", str(example), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["runs"]) == 24
    assert payload["ordering"]["mode"] == "interleaved"


def test_cli_resolves_git_baseline(
    git_repository: GitRepositoryFixture,
) -> None:
    result = runner.invoke(
        app, ["git", "baseline", str(git_repository.path), "HEAD"]
    )

    assert result.exit_code == 0
    assert f"repository={git_repository.path.resolve()}" in result.output
    assert "reference=HEAD" in result.output
    assert f"commit={git_repository.baseline_commit}" in result.output


def test_cli_verifies_and_restores_artifact(
    git_repository: GitRepositoryFixture,
    preserved_run: object,
    tmp_path: Path,
) -> None:
    artifact_path = git_repository.artifacts_root / "preserved-run"
    restored = tmp_path / "cli-restored"

    verify_result = runner.invoke(app, ["artifact", "verify", str(artifact_path)])
    restore_result = runner.invoke(
        app, ["artifact", "restore", str(artifact_path), str(restored)]
    )

    assert verify_result.exit_code == 0
    assert "Verified artifact 'preserved-run'" in verify_result.output
    assert restore_result.exit_code == 0
    assert "Restored artifact 'preserved-run'" in restore_result.output
    assert (restored / "tracked.txt").read_text(encoding="utf-8") == "modified\n"
    assert (restored / "ignored/generated.bin").is_file()


def test_cli_reports_artifact_verification_failure(tmp_path: Path) -> None:
    result = runner.invoke(app, ["artifact", "verify", str(tmp_path / "missing")])

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "missing artifact manifest" in result.output
