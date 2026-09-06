"""Local static-result launcher tests; no harness, model, or workload is run."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import threading
from copy import deepcopy
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import yaml

from agent_bench.config import load_experiment
from agent_bench.executor import ExperimentState, RunProgress, create_state
from agent_bench.matrix import expand_experiment
from agent_bench.preservation import preserve_isolated_operation
from agent_bench.report_server import ReportServer, ReportServerError
from agent_bench.reporting import _seal_report
from conftest import ExperimentFixture, GitRepositoryFixture


ROOT = Path(__file__).resolve().parents[1]


def _request(url: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object] | str, str]:
    request = Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(request, timeout=3) as response:  # noqa: S310 - test-only loopback server
            body = response.read().decode()
            return response.status, json.loads(body) if "application/json" in response.headers["Content-Type"] else body, response.headers["Content-Type"]
    except HTTPError as exc:
        body = exc.read().decode()
        content_type = exc.headers["Content-Type"]
        return exc.code, json.loads(body) if "application/json" in content_type else body, content_type


def _operation(worktree: Path, *, index: bool = True) -> None:
    if index:
        (worktree / "index.html").write_text("<h1>exact preserved app</h1><script src='app.js'></script>", encoding="utf-8")
    (worktree / "app.js").write_text("window.preservedApp = true;", encoding="utf-8")
    (worktree / "package.json").write_text('{"scripts":{"start":"touch must-never-run"}}', encoding="utf-8")


def _fixture_root(tmp_path: Path, repository: GitRepositoryFixture, *, run_ids: tuple[str, ...] = ("pass-run", "functional-fail-run"), index: bool = True) -> tuple[Path, Path]:
    output = tmp_path / "experiment-output"
    artifacts = output / "artifacts"
    for run_id in run_ids:
        preserved = preserve_isolated_operation(
            repository=repository.path,
            baseline_ref="HEAD",
            run_id=run_id,
            experiment_id="launcher-fixture",
            artifacts_root=tmp_path / "source-artifacts",
            worktrees_root=tmp_path / "worktrees",
            operation=lambda root, supported=index: _operation(root, index=supported),
        )
        shutil.copytree(preserved.artifact_path, artifacts / run_id)
    state = ExperimentState(
        experiment_id="launcher-fixture", definition_digest="a" * 64, expansion_digest="b" * 64,
        ordering={}, runs=[
            RunProgress(run_id="pass-run", execution_index=1, state="completed", functional_validation_status="pass"),
            RunProgress(run_id="functional-fail-run", execution_index=2, state="completed", functional_validation_status="fail"),
        ][:len(run_ids)], updated_at="2026-09-06T00:00:00Z",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "experiment-state.json").write_text(state.model_dump_json(), encoding="utf-8")
    report = tmp_path / "report"
    report.mkdir()
    (report / "report.html").write_text("<h1>sealed report</h1>", encoding="utf-8")
    (report / "presentation.json").write_text("{}", encoding="utf-8")
    _seal_report(report, state, {"runs": [{"run_id": item.run_id, "evidence_status": "verified"} for item in state.runs]}, [])
    return report, output


def _server(
    report: Path,
    output: Path,
    definitions: tuple[Path, ...] = (),
) -> tuple[ReportServer, threading.Thread, str]:
    try:
        server = ReportServer(("127.0.0.1", 0), report, output, experiment_definitions=definitions)
    except PermissionError:
        pytest.skip("the restricted test sandbox forbids loopback sockets; host CI exercises this test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def _frozen_baseline_fixture(
    tmp_path: Path,
    repository: GitRepositoryFixture,
    experiment_fixture: ExperimentFixture,
    *,
    result_has_index: bool = True,
    baseline_has_index: bool = True,
) -> tuple[Path, Path, Path, str]:
    """Build a temporary portable subject and one sealed run using it."""
    subject_root = tmp_path / "subjects" / "baseline-subject"
    source = subject_root / "baseline-repo"
    shutil.copytree(repository.path, source)
    if baseline_has_index:
        (source / "index.html").write_text("<h1>exact frozen baseline</h1><script src='app.js'></script>", encoding="utf-8")
    (source / "app.js").write_text("window.frozenBaseline = true;", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run([
        "git", "-C", str(source), "-c", "user.name=Fixture", "-c", "user.email=fixture@invalid",
        "commit", "-m", "static baseline",
    ], check=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True).strip()
    bundle = subject_root / "baseline.bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(source), "bundle", "create", str(bundle), "HEAD"], check=True)
    identity = {
        "subject_id": "baseline-subject-v1",
        "subject_version": "1.0.0",
        "baseline_commit": commit,
        "baseline_tree": tree,
        "baseline_bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
    }
    (subject_root / "subject.yaml").write_text(yaml.safe_dump({
        "schema_version": "1.0.0", **identity,
        "baseline_repository": "baseline-repo", "baseline_bundle": "baseline.bundle",
    }, sort_keys=False), encoding="utf-8")
    data = deepcopy(experiment_fixture.data)
    data.update({
        "identity_version": "2.0.0",
        "experiment_id": "baseline-launch-fixture",
        "baseline_repository": str(source),
        "baseline_revision": commit,
        "portable_baseline": identity,
    })
    definition_path = tmp_path / "baseline-launch-fixture.yaml"
    definition_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    definition = load_experiment(definition_path)
    run = expand_experiment(definition)[0]

    def preserve_operation(root: Path) -> None:
        if not result_has_index:
            (root / "index.html").unlink()
        _operation(root, index=result_has_index)

    preserved = preserve_isolated_operation(
        repository=source,
        baseline_ref=commit,
        run_id=run.run_id,
        experiment_id=definition.experiment_id,
        artifacts_root=tmp_path / "source-artifacts",
        worktrees_root=tmp_path / "worktrees",
        operation=preserve_operation,
    )
    output = tmp_path / "experiment-output"
    shutil.copytree(preserved.artifact_path, output / "artifacts" / run.run_id)
    state = create_state(definition)
    state.runs[0] = state.runs[0].model_copy(update={"state": "completed"})
    output.mkdir(exist_ok=True)
    (output / "experiment-state.json").write_text(state.model_dump_json(), encoding="utf-8")
    report = tmp_path / "report"
    report.mkdir()
    (report / "report.html").write_text("<h1>sealed report</h1>", encoding="utf-8")
    (report / "presentation.json").write_text("{}", encoding="utf-8")
    _seal_report(report, state, {"runs": [{"run_id": run.run_id, "evidence_status": "verified"}]}, [])
    return report, output, definition_path, run.run_id


def test_report_server_is_loopback_verified_and_launches_fresh_exact_static_results(tmp_path: Path, git_repository: GitRepositoryFixture) -> None:
    report, output = _fixture_root(tmp_path, git_repository)
    try:
        ReportServer(("0.0.0.0", 0), report, output)
    except ValueError as exc:
        assert "127.0.0.1" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-loopback bind unexpectedly accepted")
    server, thread, base = _server(report, output)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert server.server_address[1] != 0
        status, report_html, mime = _request(base + "/")
        assert status == 200 and "sealed report" in str(report_html) and mime.startswith("text/html")
        status, first, _ = _request(base + "/api/launch-result", {"run_id": "pass-run"})
        assert status == 200 and isinstance(first, dict) and first["fresh"] is True
        status, second, _ = _request(base + "/api/launch-result", {"run_id": "pass-run"})
        assert status == 200 and isinstance(second, dict)
        status, failed, _ = _request(base + "/api/launch-result", {"run_id": "functional-fail-run"})
        assert status == 200 and isinstance(failed, dict)
        assert first["url"] != second["url"] != failed["url"]
        app_status, app, app_mime = _request(str(first["url"]))
        assert app_status == 200 and "exact preserved app" in str(app) and app_mime.startswith("text/html")
        assert _request(str(first["url"]) + "../../etc/passwd")[0] == 404
        assert not (tmp_path / "must-never-run").exists()
        assert (output / "artifacts" / "pass-run" / "manifest.json").is_file()
        copies = [destination for _child, _thread, destination in server._children]
        assert len(copies) == 3 and len(set(copies)) == 3 and all((copy / "index.html").is_file() for copy in copies)
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)
    assert not server.launch_root.exists()


def test_report_server_rejects_identity_mismatch_unknown_or_unsupported_runs(tmp_path: Path, git_repository: GitRepositoryFixture) -> None:
    report, output = _fixture_root(tmp_path, git_repository, run_ids=("pass-run",), index=False)
    server, thread, base = _server(report, output)
    try:
        status, unknown, _ = _request(base + "/api/launch-result", {"run_id": "pending-run"})
        assert status == 400 and isinstance(unknown, dict) and "unknown" in str(unknown["error"])
        status, unsupported, _ = _request(base + "/api/launch-result", {"run_id": "pass-run"})
        assert status == 422 and isinstance(unsupported, dict)
        assert unsupported["error"] == "Runnable result unavailable: no supported static web entry point"
        assert _request(base + "/api/launch-result", {"run_id": "x" * 40_000})[0] == 400
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)

    state = json.loads((output / "experiment-state.json").read_text(encoding="utf-8"))
    state["experiment_id"] = "different-experiment"
    (output / "experiment-state.json").write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ReportServerError, match="identities disagree"):
        ReportServer(("127.0.0.1", 0), report, output)


def test_report_server_verifies_the_sealed_report_before_binding(tmp_path: Path, git_repository: GitRepositoryFixture) -> None:
    report, output = _fixture_root(tmp_path, git_repository, run_ids=("pass-run",))
    (report / "report.html").write_text("tampered", encoding="utf-8")

    with pytest.raises(ReportServerError, match="cannot serve report"):
        ReportServer(("127.0.0.1", 0), report, output)


def test_report_server_launches_exact_fresh_portable_baselines(
    tmp_path: Path,
    git_repository: GitRepositoryFixture,
    experiment_fixture: ExperimentFixture,
) -> None:
    report, output, definition, run_id = _frozen_baseline_fixture(
        tmp_path, git_repository, experiment_fixture,
    )
    server, thread, base = _server(report, output, (definition,))
    try:
        # A mutable subject checkout is not the launch source: the verified
        # portable bundle is, so this dirty change must be absent in the app.
        subject_source = tmp_path / "subjects" / "baseline-subject" / "baseline-repo"
        (subject_source / "index.html").write_text("<h1>mutable checkout</h1>", encoding="utf-8")
        status, before, _ = _request(base + "/api/launch-baseline", {"run_id": run_id})
        assert status == 200 and isinstance(before, dict)
        assert before["kind"] == "baseline" and before["fresh"] is True
        status, second_before, _ = _request(base + "/api/launch-baseline", {"run_id": run_id})
        assert status == 200 and isinstance(second_before, dict)
        status, after, _ = _request(base + "/api/launch-result", {"run_id": run_id})
        assert status == 200 and isinstance(after, dict) and after["kind"] == "result"
        assert before["url"] != second_before["url"] != after["url"]
        app_status, baseline_html, _ = _request(str(before["url"]))
        assert app_status == 200
        assert "exact frozen baseline" in str(baseline_html)
        assert "mutable checkout" not in str(baseline_html)
        copies = [destination for _child, _thread, destination in server._children]
        baseline_copies = [copy for copy in copies if copy.name.startswith("baseline-")]
        assert len(baseline_copies) == 2
        assert all(not (copy / ".git").exists() for copy in baseline_copies)
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)


def test_report_server_reports_independent_baseline_unavailability(
    tmp_path: Path,
    git_repository: GitRepositoryFixture,
) -> None:
    report, output = _fixture_root(tmp_path, git_repository, run_ids=("pass-run",))
    server, thread, base = _server(report, output)
    try:
        status, baseline, _ = _request(base + "/api/launch-baseline", {"run_id": "pass-run"})
        assert status == 422 and isinstance(baseline, dict)
        assert "exact immutable experiment definition is unavailable" in str(baseline["error"])
        status, result, _ = _request(base + "/api/launch-result", {"run_id": "pass-run"})
        assert status == 200 and isinstance(result, dict)
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)


def test_report_server_appends_and_revokes_manual_adjudication(
    tmp_path: Path, git_repository: GitRepositoryFixture,
) -> None:
    report, output = _fixture_root(tmp_path, git_repository, run_ids=("pass-run",))
    server, thread, base = _server(report, output)
    try:
        status, marked, _ = _request(base + "/api/adjudicate", {"run_id": "pass-run", "decision": "pass"})
        assert status == 200 and isinstance(marked, dict)
        assert marked["effective_functional_status"] == "pass"
        assert (output / "adjudications" / "pass-run" / "revision-001.json").is_file()
        status, undone, _ = _request(base + "/api/adjudicate/undo", {"run_id": "pass-run"})
        assert status == 200 and isinstance(undone, dict) and undone["decision"] == "revoked"
        assert (output / "adjudications" / "pass-run" / "revision-002.json").is_file()
        status, rejected, _ = _request(base + "/api/adjudicate", {"run_id": "unknown", "decision": "pass"})
        assert status == 400 and isinstance(rejected, dict)
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)


def test_report_server_keeps_a_valid_baseline_available_when_result_is_not_static(
    tmp_path: Path,
    git_repository: GitRepositoryFixture,
    experiment_fixture: ExperimentFixture,
) -> None:
    report, output, definition, run_id = _frozen_baseline_fixture(
        tmp_path, git_repository, experiment_fixture, result_has_index=False,
    )
    server, thread, base = _server(report, output, (definition,))
    try:
        status, result, _ = _request(base + "/api/launch-result", {"run_id": run_id})
        assert status == 422 and isinstance(result, dict)
        status, baseline, _ = _request(base + "/api/launch-baseline", {"run_id": run_id})
        assert status == 200 and isinstance(baseline, dict)
        assert "exact frozen baseline" in str(_request(str(baseline["url"]))[1])
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)


def test_report_server_reports_an_unsupported_baseline_independently(
    tmp_path: Path,
    git_repository: GitRepositoryFixture,
    experiment_fixture: ExperimentFixture,
) -> None:
    report, output, definition, run_id = _frozen_baseline_fixture(
        tmp_path, git_repository, experiment_fixture, baseline_has_index=False,
    )
    server, thread, base = _server(report, output, (definition,))
    try:
        status, baseline, _ = _request(base + "/api/launch-baseline", {"run_id": run_id})
        assert status == 422 and isinstance(baseline, dict)
        assert baseline["error"] == "Baseline unavailable: no supported static web entry point"
        assert _request(base + "/api/launch-result", {"run_id": run_id})[0] == 200
    finally:
        server.shutdown()
        server.close()
        thread.join(timeout=2)


@pytest.mark.parametrize(
    ("definition_name", "subject_id"),
    [
        ("taskboard-functional-easy-v1.yaml", "taskboard-v1"),
        ("taskboard-functional-medium-v1.yaml", "taskboard-priority-v1"),
        ("taskboard-functional-complex-v1.yaml", "taskboard-filtering-v1"),
        ("pocket-ledger-v1.yaml", "pocket-ledger-v1"),
    ],
)
def test_builtin_experiments_resolve_their_own_portable_baselines(
    definition_name: str,
    subject_id: str,
) -> None:
    definition = load_experiment(ROOT / "experiments" / definition_name)
    run = expand_experiment(definition)[0]
    subject = ReportServer._verified_subject(
        definition, run, definition.portable_baseline.baseline_commit,
    )
    assert subject.identity.subject_id == subject_id
    assert subject.identity.baseline_commit == definition.portable_baseline.baseline_commit
