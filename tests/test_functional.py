from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from agent_bench.cli import app
from agent_bench.functional import baseline_check, load_functional_scenario, validate_workspace


ROOT = Path(__file__).parents[1]
SCENARIO_PATH = ROOT / "functional" / "scenarios" / "task-priority-v1.yaml"


def _priority_solution(source: Path) -> None:
    index = source / "index.html"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            '<label>Status <select id="task-status"><option>Todo</option><option>Doing</option><option>Done</option></select></label>',
            '<label>Status <select id="task-status"><option>Todo</option><option>Doing</option><option>Done</option></select></label><label>Priority <select><option>Low</option><option>Medium</option><option>High</option></select></label>',
        ),
        encoding="utf-8",
    )
    constants = source / "src/constants.js"
    constants.write_text(
        constants.read_text(encoding="utf-8") + "\nexport const PRIORITIES = Object.freeze([\"Low\", \"Medium\", \"High\"]);\nexport function validPriority(value) { return PRIORITIES.includes(value); }\n",
        encoding="utf-8",
    )
    taskboard = source / "src/taskboard.js"
    text = taskboard.read_text(encoding="utf-8")
    text = text.replace('import { TASK_STORAGE_KEY, STATUSES, validStatus } from "./constants.js";', 'import { TASK_STORAGE_KEY, STATUSES, validStatus, validPriority } from "./constants.js";')
    text = text.replace('return { id: value.id, title: value.title, status: validStatus(value.status) ? value.status : "Todo" };', 'return { id: value.id, title: value.title, status: validStatus(value.status) ? value.status : "Todo", priority: validPriority(value.priority) ? value.priority : "Medium" };')
    text = text.replace('return `${task.title} · ${task.status}`;', 'return `${task.title} · ${task.status} · ${task.priority}`;')
    text = text.replace('createTask({ title, status = "Todo" }) {', 'createTask({ title, status = "Todo", priority = "Medium" }) {')
    text = text.replace('if (!cleanTitle || !validStatus(status))', 'if (!cleanTitle || !validStatus(status) || !validPriority(priority))')
    text = text.replace('const task = { id: taskId(), title: cleanTitle, status };', 'const task = { id: taskId(), title: cleanTitle, status, priority };')
    text = text.replace('this.save(); return task;\n  }\n\n  deleteTask', 'if ("priority" in changes) { if (!validPriority(changes.priority)) throw new Error("Invalid task priority."); task.priority = changes.priority; }\n    this.save(); return task;\n  }\n\n  deleteTask')
    taskboard.write_text(text, encoding="utf-8")


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


def test_task_priority_solution_scores_independently_from_hard_gates(tmp_path: Path) -> None:
    scenario = load_functional_scenario(SCENARIO_PATH)
    workspace = tmp_path / "solution"
    shutil.copytree(ROOT / "subjects/taskboard-v1/baseline-repo", workspace)
    _priority_solution(workspace)
    result = validate_workspace(scenario, workspace, "synthetic-priority-run", tmp_path / "solution.json")

    assert result.score_numerator == result.score_denominator == 16
    assert result.score_percent == 100
    assert result.hard_gate_pass is True
    assert result.baseline_regression == {"total": 7, "passed": 7, "failed": 0, "unavailable": 0, "error": 0}


def test_functional_cli_writes_post_run_result(tmp_path: Path) -> None:
    runner = CliRunner()
    output = tmp_path / "result.json"
    result = runner.invoke(app, ["functional", "validate", str(SCENARIO_PATH), str(ROOT / "subjects/taskboard-v1/baseline-repo"), "--run-id", "synthetic-baseline", "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["failed_tests"] == 9
