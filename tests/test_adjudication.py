from __future__ import annotations

from pathlib import Path

from agent_bench.adjudication import active_adjudication, append_adjudication, load_adjudications
from agent_bench.preservation import preserve_isolated_operation
from conftest import GitRepositoryFixture


def test_manual_adjudication_is_append_only_and_separate_from_preserved_result(
    tmp_path: Path, git_repository: GitRepositoryFixture,
) -> None:
    preserved = preserve_isolated_operation(
        repository=git_repository.path, baseline_ref="HEAD", run_id="manual-run", experiment_id="fixture",
        artifacts_root=tmp_path / "artifacts", worktrees_root=tmp_path / "worktrees",
        operation=lambda root: (root / "index.html").write_text("ok", encoding="utf-8"),
    )
    output = tmp_path / "output"
    before = (preserved.artifact_path / "manifest.json").read_bytes()
    marked = append_adjudication(experiment_output=output, artifact_root=preserved.artifact_path, run_id="manual-run", decision="pass")
    revoked = append_adjudication(experiment_output=output, artifact_root=preserved.artifact_path, run_id="manual-run", decision="revoked")
    assert (marked.revision, revoked.revision) == (1, 2)
    assert [item.decision for item in load_adjudications(output, "manual-run")] == ["pass", "revoked"]
    assert active_adjudication(output, "manual-run") is None
    assert (preserved.artifact_path / "manifest.json").read_bytes() == before
