"""Focused M14 reporting and planned-matrix checks; never launch a harness."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from agent_bench.cli import app
from agent_bench.comparison import METRICS
from agent_bench.config import load_experiment
from agent_bench.matrix import expand_experiment
from agent_bench.reporting import SCHEMAS, _functional_empty, _functional_fields_from_record, _html_report


ROOT = Path(__file__).parents[1]


def test_planned_functional_matrix_is_81_baseline_homogeneous_runs() -> None:
    rows = []
    reference = load_experiment(ROOT / "experiments/pocket-ledger-v1.yaml")
    for tier in ("easy", "medium", "complex"):
        definition = load_experiment(ROOT / f"experiments/taskboard-functional-{tier}-v1.yaml")
        expanded = expand_experiment(definition)
        assert len(expanded) == 27
        assert {run.functional_scenario.tier for run in expanded} == {tier}
        assert {run.generation_seed for run in expanded} == {1001, 1002, 1003}
        assert {run.harness_id for run in expanded} == {"hermes", "opencode", "pi"}
        assert definition.fixed_environment == reference.fixed_environment
        assert definition.harnesses == reference.harnesses
        assert definition.harness_profiles == reference.harness_profiles
        rows.extend(expanded)
    assert len(rows) == 81
    assert {run.harness_id: sum(item.harness_id == run.harness_id for item in rows) for run in rows} == {"hermes": 27, "opencode": 27, "pi": 27}
    assert all(sum(item.generation_seed == seed for item in rows) == 27 for seed in (1001, 1002, 1003))


def test_functional_plan_is_read_only_and_reports_the_full_matrix() -> None:
    result = CliRunner().invoke(app, ["functional", "plan", str(ROOT / "functional/experiments/taskboard-functional-v1.yaml")])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total_runs"] == 81
    assert payload["by_harness"] == {"hermes": 27, "opencode": 27, "pi": 27}
    assert payload["by_tier"] == {"easy": 27, "medium": 27, "complex": 27}
    assert all(member["definition_digest"] and member["expansion_digest"] for member in payload["members"])
    assert {member["scenario_contract"]["tier"] for member in payload["members"]} == {"easy", "medium", "complex"}


def test_reporting_keeps_legacy_not_applicable_distinct_from_fail_and_validator_faults() -> None:
    fields = {field.name for field in SCHEMAS["runs"]}
    assert {"functional_validation_status", "functional_score_percent", "hard_gate_pass", "baseline_regression_count", "functional_tier", "failed_functional_test_ids"} <= fields
    assert _functional_empty("not_applicable")["functional_validation_status"] == "not_applicable"
    assert _functional_empty("unavailable")["functional_validation_status"] == "unavailable"
    assert _functional_empty("error")["functional_validation_status"] == "error"
    assert "functional.functional_score_percent" in METRICS
    presentation = {"generator": {}, "experiment_id": "fixture", "definition_digest": "d", "expansion_digest": "e", "completion": {}, "definition": {"prompts": [], "profiles": [], "fixed_environment": {}}, "summary_environment": {}, "runs": [], "summaries": [], "curves": [], "markers": [], "failures": [], "details": {}, "data_files": []}
    html = _html_report({"experiment_id": "fixture"}, presentation)
    assert "Functional status" in html and "Hard gate" in html and "Minimum functional score" in html


def test_functional_report_projection_keeps_all_validator_states_distinct() -> None:
    def record(status: str, passed: int, failed: int, baseline_failed: int, hard_gate: bool) -> SimpleNamespace:
        tests = tuple(
            SimpleNamespace(test_id=f"test-{index}", outcome="failed" if index < failed else "passed")
            for index in range(passed + failed)
        )
        return SimpleNamespace(
            validation_status=status,
            acceptance_score_numerator=passed if status in {"pass", "fail"} else None,
            acceptance_score_denominator=passed + failed if status in {"pass", "fail"} else None,
            scenario=SimpleNamespace(scenario_id="task-priority-v1", tier="easy"),
            functional_result=SimpleNamespace(
                score_percent=(100 * passed / (passed + failed)) if passed + failed else 0,
                hard_gate_pass=hard_gate,
                baseline_regression={"failed": baseline_failed},
                failed_tests=failed,
                tests=tests,
            ),
        )

    passed = _functional_fields_from_record(record("pass", 30, 0, 0, True))
    high_score_fail = _functional_fields_from_record(record("fail", 29, 1, 0, False))
    low_score_fail = _functional_fields_from_record(record("fail", 1, 29, 0, False))
    regression_fail = _functional_fields_from_record(record("fail", 29, 1, 1, False))
    error = _functional_fields_from_record(record("error", 0, 0, 0, False))
    unavailable = _functional_fields_from_record(record("unavailable", 0, 0, 0, False))

    assert passed["functional_validation_status"] == "pass" and passed["hard_gate_pass"] is True
    assert high_score_fail["functional_score_percent"] > 90 and high_score_fail["hard_gate_pass"] is False
    assert low_score_fail["functional_score_percent"] < 10 and low_score_fail["failed_functional_test_count"] == 29
    assert regression_fail["baseline_regressions"] is True and regression_fail["baseline_regression_count"] == 1
    assert error["functional_score_percent"] is None and error["functional_validation_status"] == "error"
    assert unavailable["functional_score_percent"] is None and unavailable["functional_validation_status"] == "unavailable"
    assert _functional_empty("not_applicable")["functional_validation_status"] == "not_applicable"
