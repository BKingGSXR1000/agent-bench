"""Local-only serving of sealed reports and disposable static result copies."""

from __future__ import annotations

import json
import mimetypes
import shutil
import tempfile
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from agent_bench.executor import ExperimentState
from agent_bench.preservation import PreservationError, restore_artifact, verify_artifact
from agent_bench.reporting import ReportError, verify_report


class ReportServerError(RuntimeError):
    """The local report launcher cannot safely serve the requested evidence."""


class ResultUnavailableError(ReportServerError):
    """A verified result does not meet the static-web v1 launch contract."""


@dataclass(frozen=True)
class _LaunchableResult:
    run_id: str
    artifact_root: Path


class StaticResultServer(ThreadingHTTPServer):
    """One isolated origin serving one disposable restored static result."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], root: Path) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("result server must bind only to 127.0.0.1")
        self.root = root.resolve()
        super().__init__(address, _StaticResultHandler)


class _StaticResultHandler(BaseHTTPRequestHandler):
    server: StaticResultServer

    def log_message(self, _format: str, *_args: object) -> None:
        """Result browsing is not benchmark telemetry."""

    def do_GET(self) -> None:  # noqa: N802
        relative = _safe_relative_path(urlparse(self.path).path, default="index.html")
        if relative is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        target = (self.server.root / relative).resolve()
        if not _within(target, self.server.root) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send(HTTPStatus.OK, target.read_bytes(), _mime_type(target))

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)


class ReportServer(ThreadingHTTPServer):
    """Serve one verified report and create fresh static-result origins on demand."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], report_root: Path, experiment_output: Path) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("report server must bind only to 127.0.0.1")
        self.report_root = report_root.expanduser().resolve()
        self.experiment_output = experiment_output.expanduser().resolve()
        self.launch_root = Path(tempfile.mkdtemp(prefix="agent-bench-report-launch-"))
        self._children: list[tuple[StaticResultServer, threading.Thread, Path]] = []
        self._children_lock = threading.Lock()
        try:
            self._launchable = self._validate_identity()
            super().__init__(address, ReportServerHandler)
        except Exception:
            shutil.rmtree(self.launch_root, ignore_errors=True)
            raise

    def _validate_identity(self) -> dict[str, _LaunchableResult]:
        try:
            report = verify_report(self.report_root)
            state = ExperimentState.model_validate_json(
                (self.experiment_output / "experiment-state.json").read_bytes()
            )
        except (ReportError, OSError, ValueError) as exc:
            raise ReportServerError(f"cannot serve report: {exc}") from exc
        if report.get("experiment_id") != state.experiment_id:
            raise ReportServerError("report and experiment output identities disagree")
        for field in ("definition_digest", "expansion_digest"):
            if report.get(field) != getattr(state, field):
                raise ReportServerError("report and experiment output identities disagree")
        completed = {item.run_id for item in state.runs if item.state == "completed"}
        results: dict[str, _LaunchableResult] = {}
        run_ids = report.get("included_run_ids")
        if not isinstance(run_ids, list) or any(not isinstance(item, str) for item in run_ids):
            raise ReportServerError("report has invalid included run identities")
        for run_id in run_ids:
            if run_id not in completed:
                raise ReportServerError("report and experiment output run identities disagree")
            artifact_root = self.experiment_output / "artifacts" / run_id
            try:
                artifact = verify_artifact(artifact_root)
            except PreservationError as exc:
                raise ReportServerError("report contains an unverifiable preserved artifact") from exc
            if artifact.run_id != run_id or artifact.experiment_id != state.experiment_id:
                raise ReportServerError("report and experiment output artifact identities disagree")
            results[run_id] = _LaunchableResult(run_id=run_id, artifact_root=artifact_root)
        return results

    def launch(self, run_id: str) -> dict[str, Any]:
        result = self._launchable.get(run_id)
        if result is None:
            raise ReportServerError("unknown or non-preserved report run")
        destination = Path(tempfile.mkdtemp(prefix="result-", dir=self.launch_root))
        try:
            restore_artifact(result.artifact_root, destination)
            if not (destination / "index.html").is_file():
                raise ResultUnavailableError("no supported static web entry point")
            child = StaticResultServer(("127.0.0.1", 0), destination)
            thread = threading.Thread(target=child.serve_forever, daemon=True)
            thread.start()
        except ResultUnavailableError:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        except PreservationError as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise ReportServerError("Artifact verification failed") from exc
        except OSError as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise ReportServerError("could not prepare runnable result") from exc
        with self._children_lock:
            self._children.append((child, thread, destination))
        return {"run_id": run_id, "url": f"http://127.0.0.1:{child.server_address[1]}/", "fresh": True}

    def close(self) -> None:
        """Close child origins and delete only disposable restored copies."""
        with self._children_lock:
            children = self._children
            self._children = []
        for child, thread, _destination in children:
            child.shutdown()
            child.server_close()
            thread.join(timeout=2)
        shutil.rmtree(self.launch_root, ignore_errors=True)
        self.server_close()


class ReportServerHandler(BaseHTTPRequestHandler):
    server: ReportServer

    def log_message(self, _format: str, *_args: object) -> None:
        """Normal local browsing is not terminal telemetry."""

    def do_GET(self) -> None:  # noqa: N802
        relative = _safe_relative_path(urlparse(self.path).path, default="report.html")
        if relative is None:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        target = (self.server.report_root / relative).resolve()
        if not _within(target, self.server.report_root) or not target.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._send(HTTPStatus.OK, target.read_bytes(), _mime_type(target))

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/launch-result":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            payload = self._request_json()
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ReportServerError("run_id is required")
            self._json(HTTPStatus.OK, self.server.launch(run_id))
        except ResultUnavailableError as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": f"Runnable result unavailable: {exc}"})
        except ReportServerError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "request must contain a bounded JSON object"})

    def _request_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 32_768:
            raise ValueError("invalid request length")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body is not an object")
        return value

    def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _safe_relative_path(path: str, *, default: str) -> Path | None:
    decoded = unquote(path)
    relative = decoded.lstrip("/") or default
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    return candidate


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _mime_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"
