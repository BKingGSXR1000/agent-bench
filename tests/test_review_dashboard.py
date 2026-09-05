"""Local M10 dashboard tests; they use no browser, harness, model, or GPU."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import agent_bench.review_dashboard as dashboard
import pytest


def _request(url: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object] | str]:
    request = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"} if payload is not None else {})
    try:
        with urlopen(request, timeout=3) as response:  # noqa: S310 - test-only loopback server
            data = response.read().decode()
            return response.status, json.loads(data) if "application/json" in response.headers["Content-Type"] else data
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def test_dashboard_is_loopback_blind_and_prepares_resettable_copy(monkeypatch: object, tmp_path: Path) -> None:
    row = {"run_id": "canonical-secret-run", "blind_review_id": "blind-0123456789ab", "blind_sort": "x",
           "semantic_task": "entry-delete", "state": "completed", "reviewed": False,
           "priority_flags": ("no_agent_invoked_test",), "harness": "never-exposed"}
    monkeypatch.setattr(dashboard, "review_queue", lambda *_args: [row])

    def fake_prepare(_output: Path, run_id: str, destination: Path, _subject: Path) -> dict[str, object]:
        assert run_id == "canonical-secret-run"
        destination.mkdir(parents=True)
        (destination / "review-fixture.html").write_text("<script>localStorage.clear();sessionStorage.clear()</script>", encoding="utf-8")
        (destination / "index.html").write_text("fixture app", encoding="utf-8")
        return {}

    monkeypatch.setattr(dashboard, "prepare_review_copy", fake_prepare)
    try:
        server = dashboard.ReviewDashboardServer(("127.0.0.1", 0), tmp_path, Path("experiments/pocket-ledger-v1.yaml"), Path("subjects/pocket-ledger-v1"))
    except PermissionError:
        pytest.skip("the restricted test sandbox forbids loopback sockets; host CI exercises this test")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        status, next_payload = _request(base + "/api/next")
        assert status == 200
        assert isinstance(next_payload, dict)
        assert next_payload["blind_review_id"] == row["blind_review_id"]
        serialized = json.dumps(next_payload)
        assert "canonical-secret-run" not in serialized and "never-exposed" not in serialized
        assert next_payload["semantic_task"] == "entry-delete"
        assert "€1,825.00" in json.dumps(next_payload, ensure_ascii=False)
        status, prepared = _request(base + "/api/prepare", {"blind_review_id": row["blind_review_id"]})
        assert status == 200 and isinstance(prepared, dict)
        app_status, fixture = _request(base + str(prepared["app_url"]))
        assert app_status == 200 and "sessionStorage.clear" in str(fixture) and "localStorage.clear" in str(fixture)
        status, hidden = _request(base + "/api/reveal?blind_review_id=" + row["blind_review_id"])
        assert status == 403 and isinstance(hidden, dict)
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_scripts_cover_all_five_tasks_and_loopback_is_enforced() -> None:
    assert set(dashboard._STEPS) == {"entry-delete", "entry-filter", "entry-category", "monthly-summary", "keyboard-entry"}
    for task in dashboard._STEPS:
        rendered = dashboard.human_steps(task)
        assert rendered and all(item["action"] and item["expected"] for item in rendered)
    protocol, _digest = dashboard.load_protocol(Path("subjects/pocket-ledger-v1"))
    for task in protocol["tasks"].values():
        for item in dashboard.human_criteria(task["criteria"]):
            assert item["criterion_id"] in dashboard._CRITERION_DETAILS
            assert item["action"] and item["expected"]
    for item in dashboard.human_criteria(protocol["common_regression_criteria"]):
        assert item["criterion_id"] in dashboard._CRITERION_DETAILS
    try:
        dashboard.ReviewDashboardServer(("0.0.0.0", 0), Path("."), Path("."), Path("."))
    except ValueError as exc:
        assert "127.0.0.1" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("non-loopback bind unexpectedly accepted")
