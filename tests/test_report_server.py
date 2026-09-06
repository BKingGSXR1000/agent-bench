"""Local static-result launcher tests; no harness, model, or workload is run."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from agent_bench.executor import ExperimentState, RunProgress
from agent_bench.preservation import preserve_isolated_operation
from agent_bench.report_server import ReportServer, ReportServerError
from agent_bench.reporting import _seal_report
from conftest import GitRepositoryFixture


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
    _seal_report(report, state, {"runs": [{"run_id": item.run_id, "evidence_status": "verified"} for item in state.runs]}, [])
    return report, output


def _server(report: Path, output: Path) -> tuple[ReportServer, threading.Thread, str]:
    try:
        server = ReportServer(("127.0.0.1", 0), report, output)
    except PermissionError:
        pytest.skip("the restricted test sandbox forbids loopback sockets; host CI exercises this test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


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
