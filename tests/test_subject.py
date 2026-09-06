from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path

import yaml


ROOT = Path(__file__).parents[1]
SUBJECT = ROOT / "subjects" / "pocket-ledger-v1"
TASKBOARD_SUBJECT = ROOT / "subjects" / "taskboard-v1"
TASKBOARD_PRIORITY_SUBJECT = ROOT / "subjects" / "taskboard-priority-v1"


def test_subject_identity_baseline_and_commands_are_frozen() -> None:
    definition = yaml.safe_load((SUBJECT / "subject.yaml").read_text())
    repository = SUBJECT / definition["baseline_repository"]
    assert definition["subject_id"] == "pocket-ledger-v1"
    bundle = SUBJECT / definition["baseline_bundle"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == definition["baseline_bundle_sha256"]
    with tempfile.TemporaryDirectory() as temporary:
        clone = Path(temporary) / "baseline"
        subprocess.run(["git", "clone", "--quiet", str(bundle), str(clone)], check=True)
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip() == definition["baseline_commit"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=clone, text=True).strip() == definition["baseline_tree"]
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=clone, text=True) == ""
        source_files = {p.relative_to(repository) for p in repository.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        clone_files = {p.relative_to(clone) for p in clone.rglob("*") if p.is_file() and ".git" not in p.parts}
        assert source_files == clone_files
    assert definition["runtime"]["package_manager"] == "none"


def test_prompt_variants_link_to_each_semantic_task_and_stay_outside_baseline() -> None:
    tasks = yaml.safe_load((SUBJECT / "tasks" / "tasks.yaml").read_text())["tasks"]
    manifest = yaml.safe_load((SUBJECT / "prompts" / "manifest.yaml").read_text())["prompts"]
    baseline_text = "\n".join(path.read_text(errors="ignore") for path in (SUBJECT / "baseline-repo").rglob("*") if path.is_file() and ".git" not in path.parts)
    for task in tasks:
        assert task["id"] not in baseline_text
        variants = [SUBJECT / "prompts" / f"{task['id']}-{variant}.txt" for variant in ("vague", "normal", "precise")]
        assert all(path.is_file() and path.read_bytes() for path in variants)
        assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in variants] == [manifest[task["id"]][variant] for variant in ("vague", "normal", "precise")]
        assert len({hashlib.sha256(path.read_bytes()).hexdigest() for path in variants}) == 3
        assert task["acceptance"] not in "\n".join(path.read_text() for path in variants)


def test_baseline_static_check_does_not_mutate_subject() -> None:
    repository = SUBJECT / "baseline-repo"
    subprocess.run(["python3", "tests/test_baseline.py"], cwd=repository, check=True)
    assert not (repository / ".git").exists()


def test_taskboard_frozen_baseline_is_modular_and_has_no_acceptance_suite() -> None:
    definition = yaml.safe_load((TASKBOARD_SUBJECT / "subject.yaml").read_text())
    repository = TASKBOARD_SUBJECT / definition["baseline_repository"]
    bundle = TASKBOARD_SUBJECT / definition["baseline_bundle"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == definition["baseline_bundle_sha256"]
    with tempfile.TemporaryDirectory() as temporary:
        clone = Path(temporary) / "baseline"
        subprocess.run(["git", "clone", "--quiet", str(bundle), str(clone)], check=True)
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip() == definition["baseline_commit"]
        assert subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=clone, text=True).strip() == definition["baseline_tree"]
        subprocess.run(["node", "tests/test_baseline.mjs"], cwd=clone, check=True)
    assert (repository / "src" / "taskboard.js").is_file()
    assert not (repository / "functional").exists()


def test_taskboard_priority_derived_baseline_is_frozen_and_independent() -> None:
    definition = yaml.safe_load((TASKBOARD_PRIORITY_SUBJECT / "subject.yaml").read_text())
    bundle = TASKBOARD_PRIORITY_SUBJECT / definition["baseline_bundle"]
    assert definition["derived_from"]["easy_scenario_id"] == "task-priority-v1"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == definition["baseline_bundle_sha256"]
    with tempfile.TemporaryDirectory() as temporary:
        clone = Path(temporary) / "baseline"
        subprocess.run(["git", "clone", "--quiet", str(bundle), str(clone)], check=True)
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip() == definition["baseline_commit"]
        subprocess.run(["node", "tests/test_baseline.mjs"], cwd=clone, check=True)
    assert not (TASKBOARD_PRIORITY_SUBJECT / "baseline-repo" / "functional").exists()
