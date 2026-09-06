from __future__ import annotations

import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from agent_bench.cli import app
from agent_bench.functional import baseline_check, load_functional_scenario, self_validate


ROOT = Path(__file__).parents[1]
SCENARIO_PATH = ROOT / "functional" / "scenarios" / "task-priority-v1.yaml"
MEDIUM_SCENARIO_PATH = ROOT / "functional" / "scenarios" / "combined-filtering-v1.yaml"


def _source_fingerprint(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_task_priority_baseline_discrimination_and_immutable_result(tmp_path: Path) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH)
    output = tmp_path / "baseline.json"
    result = baseline_check(scenario, output)

    assert result.validation_mode == "baseline_discrimination"
    assert result.passed_tests == 7
    assert result.failed_tests == 9
    assert result.hard_gate_pass is False
    assert json.loads(output.read_text(encoding="utf-8"))["scenario_id"] == "task-priority-v1"
    try:
        baseline_check(scenario, output)
    except Exception as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("an existing validation result must never be overwritten")


def test_task_priority_validator_self_check_proves_known_good_and_bad_vectors(tmp_path: Path) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH)
    subject_source = ROOT / "subjects/taskboard-v1/baseline-repo"
    before = _source_fingerprint(subject_source)
    results = {result.run_id.removeprefix("self-"): result for result in self_validate(scenario, tmp_path / "self-check")}

    assert _source_fingerprint(subject_source) == before
    assert not (subject_source / "functional").exists()
    assert results["known-good"].score_numerator == results["known-good"].score_denominator == 16
    assert results["known-good"].hard_gate_pass is True
    assert {test.test_id for test in results["known-bad-persistence"].tests if test.outcome == "failed"} == {"priority-persists"}
    assert results["known-bad-persistence"].hard_gate_pass is False
    assert {test.test_id for test in results["known-bad-regression"].tests if test.outcome == "failed"} == {"baseline-delete"}
    assert results["known-bad-regression"].hard_gates["baseline_regressions"] is False
    assert (tmp_path / "self-check/known-good.json").is_file()


def test_functional_cli_writes_post_run_result(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "result.json"
    result = runner.invoke(app, ["functional", "validate", str(SCENARIO_PATH), str(ROOT / "subjects/taskboard-v1/baseline-repo"), "--run-id", "synthetic-baseline", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["failed_tests"] == 9


def test_functional_self_check_cli_writes_all_fixture_results(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "self-check"
    result = runner.invoke(app, ["functional", "self-check", str(SCENARIO_PATH), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in output.glob("*.json")) == [
        "known-bad-persistence.json", "known-bad-regression.json", "known-good.json", "untouched-baseline.json",
    ]


def test_combined_filtering_self_check_proves_derived_baseline_and_targeted_rejections(tmp_path: Path) -> None:
    scenario = load_functional_scenario(MEDIUM_SCENARIO_PATH)
    source = ROOT / "subjects/taskboard-priority-v1/baseline-repo"
    before = _source_fingerprint(source)
    results = {result.run_id.removeprefix("self-"): result for result in self_validate(scenario, tmp_path / "medium-self-check")}

    assert scenario.baseline_strategy == "derived-priority-baseline"
    assert _source_fingerprint(source) == before
    assert results["untouched-baseline"].score_numerator == 8
    assert results["known-good"].score_numerator == results["known-good"].score_denominator == 30
    assert results["known-good"].hard_gate_pass is True
    assert {item.test_id for item in results["known-bad-or-semantics"].tests if item.outcome == "failed"} == {
        "combine-search-status", "combine-search-priority", "combine-status-priority", "combine-all-filters", "combine-and-not-or",
    }
    assert {item.test_id for item in results["known-bad-no-filter-persistence"].tests if item.outcome == "failed"} == {
        "filter-state-search-persists", "filter-state-status-persists", "filter-state-priority-persists", "filter-state-reload-visible",
    }
    assert results["known-bad-delete-regression"].hard_gates["baseline_regressions"] is False
