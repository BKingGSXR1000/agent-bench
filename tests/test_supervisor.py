from __future__ import annotations

import json

import pytest

from agent_bench.supervisor import SupervisorError, run_startup_diagnostic


def test_dry_supervisor_creates_fsynced_startup_and_completion_records(tmp_path):
    evidence = run_startup_diagnostic(output_root=tmp_path / "output", run_id="supervisor-dry", argv=("agent-bench", "dry"))
    records = [json.loads(path.read_text()) for path in sorted(evidence.root.glob("*.json"))]
    assert [record["stage"] for record in records] == ["initialized", "dry_startup_complete"]
    assert all(record["supervisor_pid"] > 0 for record in records)
    assert records[0]["intended_output_root"] == str((tmp_path / "output").resolve())
    assert not (tmp_path / "output" / "runtime").exists()


def test_dry_supervisor_preserves_early_exception_and_nonzero_is_observable(tmp_path):
    def fail() -> None:
        raise ImportError("pinned runtime missing")

    with pytest.raises(SupervisorError, match="pinned runtime missing"):
        run_startup_diagnostic(output_root=tmp_path / "output", run_id="supervisor-failure", argv=("agent-bench", "dry"), initialize=fail)
    root = tmp_path / "output" / "supervisor" / "supervisor-failure"
    failure = json.loads((root / "02-failure.json").read_text())
    assert failure["failure_type"] == "ImportError"
    assert "pinned runtime missing" in (root / "stderr.log").read_text()


def test_supervisor_never_silently_overwrites_prior_evidence(tmp_path):
    run_startup_diagnostic(output_root=tmp_path, run_id="same-run", argv=("a",))
    with pytest.raises(SupervisorError, match="already exists"):
        run_startup_diagnostic(output_root=tmp_path, run_id="same-run", argv=("a",))
