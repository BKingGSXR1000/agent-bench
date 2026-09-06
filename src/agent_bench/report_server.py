"""Local-only serving of sealed reports and disposable static application copies."""

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
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import unquote, urlparse

from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.adjudication import AdjudicationError, append_adjudication
from agent_bench.executor import ExperimentState
from agent_bench.matrix import expand_experiment
from agent_bench.models import ExperimentDefinition, RunDefinition
from agent_bench.preservation import PreservationError, restore_artifact, verify_artifact
from agent_bench.reporting import ReportError, verify_report
from agent_bench.subject import FrozenSubject, SubjectError, load_frozen_subject, materialize_baseline


class ReportServerError(RuntimeError):
    """The local report launcher cannot safely serve the requested evidence."""


class ResultUnavailableError(ReportServerError):
    """A verified result does not meet the static-web v1 launch contract."""


class BaselineUnavailableError(ReportServerError):
    """A selected run's exact frozen baseline cannot be safely launched."""


@dataclass(frozen=True)
class _LaunchableResult:
    run_id: str
    artifact_root: Path


@dataclass(frozen=True)
class _LaunchableBaseline:
    run_id: str
    subject: FrozenSubject


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

    def __init__(
        self,
        address: tuple[str, int],
        report_root: Path,
        experiment_output: Path | Sequence[Path],
        *,
        experiment_definitions: Sequence[Path] | None = None,
    ) -> None:
        if address[0] != "127.0.0.1":
            raise ValueError("report server must bind only to 127.0.0.1")
        self.report_root = report_root.expanduser().resolve()
        outputs = (experiment_output,) if isinstance(experiment_output, Path) else tuple(experiment_output)
        if not outputs:
            raise ReportServerError("at least one experiment output is required")
        self.experiment_outputs = tuple(output.expanduser().resolve() for output in outputs)
        self.experiment_output = self.experiment_outputs[0]
        self.experiment_definitions = tuple(path.expanduser().resolve() for path in (experiment_definitions or ()))
        if self.experiment_definitions and len(self.experiment_definitions) != len(self.experiment_outputs):
            raise ReportServerError("supply one immutable experiment definition per experiment output")
        self.launch_root = Path(tempfile.mkdtemp(prefix="agent-bench-report-launch-"))
        self._children: list[tuple[StaticResultServer, threading.Thread, Path]] = []
        self._children_lock = threading.Lock()
        try:
            self._launchable, self._baselines, self._adjudication_outputs = self._validate_identity()
            super().__init__(address, ReportServerHandler)
        except Exception:
            shutil.rmtree(self.launch_root, ignore_errors=True)
            raise

    def _validate_identity(self) -> tuple[
        dict[str, _LaunchableResult],
        dict[str, _LaunchableBaseline | BaselineUnavailableError],
    ]:
        try:
            report = verify_report(self.report_root)
            presentation = json.loads((self.report_root / "presentation.json").read_text(encoding="utf-8"))
            states = {
                state.experiment_id: (state, output)
                for output in self.experiment_outputs
                for state in [ExperimentState.model_validate_json((output / "experiment-state.json").read_bytes())]
            }
        except (ReportError, OSError, ValueError) as exc:
            raise ReportServerError(f"cannot serve report: {exc}") from exc
        if len(states) != len(self.experiment_outputs):
            raise ReportServerError("experiment outputs must have distinct experiment identities")
        source_experiments = presentation.get("source_experiments")
        if isinstance(source_experiments, list):
            expected = {
                item.get("experiment_id"): item
                for item in source_experiments
                if isinstance(item, dict) and isinstance(item.get("experiment_id"), str)
            }
            if set(expected) != set(states):
                raise ReportServerError("report and experiment output identities disagree")
            for experiment_id, item in expected.items():
                state, _output = states[experiment_id]
                if (
                    item.get("definition_digest") != state.definition_digest
                    or item.get("expansion_digest") != state.expansion_digest
                ):
                    raise ReportServerError("report and experiment output identities disagree")
        elif len(states) == 1:
            state, _output = next(iter(states.values()))
            if report.get("experiment_id") != state.experiment_id:
                raise ReportServerError("report and experiment output identities disagree")
            for field in ("definition_digest", "expansion_digest"):
                if report.get(field) != getattr(state, field):
                    raise ReportServerError("report and experiment output identities disagree")
        else:
            raise ReportServerError("unified report is missing source experiment identities")

        try:
            definitions = self._load_definitions(states)
            baseline_definition_error: ReportServerError | None = None
        except ReportServerError as exc:
            # Historical reports may still open preserved results. Baseline
            # launch remains fail-closed with this specific sealed-evidence error.
            definitions = {}
            baseline_definition_error = exc
        run_definitions = {
            run.run_id: (definition, run)
            for definition in definitions.values()
            for run in expand_experiment(definition)
        }
        results: dict[str, _LaunchableResult] = {}
        baselines: dict[str, _LaunchableBaseline | BaselineUnavailableError] = {}
        adjudication_outputs: dict[str, Path] = {}
        run_ids = report.get("included_run_ids")
        if not isinstance(run_ids, list) or any(not isinstance(item, str) for item in run_ids):
            raise ReportServerError("report has invalid included run identities")
        for run_id in run_ids:
            owners = [
                (state, output) for state, output in states.values()
                if run_id in {item.run_id for item in state.runs if item.state == "completed"}
            ]
            if len(owners) != 1:
                raise ReportServerError("report and experiment output run identities disagree")
            state, output = owners[0]
            artifact_root = output / "artifacts" / run_id
            try:
                artifact = verify_artifact(artifact_root)
            except PreservationError as exc:
                raise ReportServerError("report contains an unverifiable preserved artifact") from exc
            if artifact.run_id != run_id or artifact.experiment_id != state.experiment_id:
                raise ReportServerError("report and experiment output artifact identities disagree")
            results[run_id] = _LaunchableResult(run_id=run_id, artifact_root=artifact_root)
            adjudication_outputs[run_id] = output
            resolved = run_definitions.get(run_id)
            if baseline_definition_error is not None:
                baselines[run_id] = BaselineUnavailableError(str(baseline_definition_error))
            elif resolved is None:
                baselines[run_id] = BaselineUnavailableError("report run is absent from its immutable experiment definition")
            else:
                definition, run = resolved
                try:
                    baselines[run_id] = _LaunchableBaseline(
                        run_id=run_id,
                        subject=self._verified_subject(definition, run, artifact.baseline_commit),
                    )
                except ReportServerError as exc:
                    baselines[run_id] = BaselineUnavailableError(str(exc))
        return results, baselines, adjudication_outputs

    def _load_definitions(
        self,
        states: Mapping[str, tuple[ExperimentState, Path]],
    ) -> dict[str, ExperimentDefinition]:
        supplied = list(self.experiment_definitions)
        candidates = supplied or sorted((Path.cwd() / "experiments").glob("*.yaml"))
        definitions: dict[str, ExperimentDefinition] = {}
        for state in states.values():
            match: ExperimentDefinition | None = None
            for candidate in candidates:
                try:
                    definition = load_experiment(candidate)
                except ExperimentConfigError:
                    continue
                if (
                    definition.experiment_id == state[0].experiment_id
                    and definition.definition_digest == state[0].definition_digest
                ):
                    if match is not None and match != definition:
                        raise ReportServerError("multiple incompatible immutable experiment definitions match a report")
                    match = definition
            if match is None:
                raise ReportServerError("exact immutable experiment definition is unavailable for baseline launch")
            definitions[match.experiment_id] = match
        return definitions

    @staticmethod
    def _verified_subject(
        definition: ExperimentDefinition,
        run: RunDefinition,
        artifact_baseline_commit: str,
    ) -> FrozenSubject:
        identity = definition.portable_baseline
        if identity is None or run.portable_baseline != identity:
            raise ReportServerError("run lacks a matching portable baseline identity")
        if run.baseline_revision != identity.baseline_commit or artifact_baseline_commit != identity.baseline_commit:
            raise ReportServerError("run baseline commit disagrees with its immutable definition")
        try:
            subject = load_frozen_subject(definition.baseline_repository.parent)
        except SubjectError as exc:
            raise ReportServerError(f"cannot verify frozen baseline subject: {exc}") from exc
        if subject.identity != identity or subject.source_directory != definition.baseline_repository:
            raise ReportServerError("frozen subject does not match the experiment portable baseline")
        return subject

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
        return {"run_id": run_id, "kind": "result", "url": f"http://127.0.0.1:{child.server_address[1]}/", "fresh": True}

    def launch_baseline(self, run_id: str) -> dict[str, Any]:
        baseline = self._baselines.get(run_id)
        if baseline is None:
            raise ReportServerError("unknown or non-preserved report run")
        if isinstance(baseline, BaselineUnavailableError):
            raise baseline
        destination = Path(tempfile.mkdtemp(prefix="baseline-", dir=self.launch_root))
        try:
            destination.rmdir()
            materialize_baseline(baseline.subject, destination, verify_tracked_source=False)
            shutil.rmtree(destination / ".git")
            if not (destination / "index.html").is_file():
                raise BaselineUnavailableError("no supported static web entry point")
            child = StaticResultServer(("127.0.0.1", 0), destination)
            thread = threading.Thread(target=child.serve_forever, daemon=True)
            thread.start()
        except BaselineUnavailableError:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        except (OSError, SubjectError) as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise BaselineUnavailableError(f"exact frozen baseline could not be materialized: {exc}") from exc
        with self._children_lock:
            self._children.append((child, thread, destination))
        return {"run_id": run_id, "kind": "baseline", "url": f"http://127.0.0.1:{child.server_address[1]}/", "fresh": True}

    def adjudicate(self, run_id: str, decision: str) -> dict[str, Any]:
        result = self._launchable.get(run_id)
        output = self._adjudication_outputs.get(run_id)
        if result is None or output is None:
            raise ReportServerError("unknown, pending, or non-preserved report run")
        if decision not in {"pass", "revoked"}:
            raise ReportServerError("decision must be pass or revoked")
        try:
            record = append_adjudication(
                experiment_output=output, artifact_root=result.artifact_root, run_id=run_id, decision=decision,
            )
        except (AdjudicationError, PreservationError) as exc:
            raise ReportServerError(f"manual adjudication was not recorded: {exc}") from exc
        return {
            "run_id": run_id, "decision": record.decision, "revision": record.revision,
            "effective_functional_status": "pass" if record.decision == "pass" else None,
            "provenance": "manual human verification" if record.decision == "pass" else None,
        }

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
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/launch-result", "/api/launch-baseline", "/api/adjudicate", "/api/adjudicate/undo"}:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            payload = self._request_json()
            run_id = payload.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ReportServerError("run_id is required")
            if endpoint.startswith("/api/adjudicate"):
                decision = "revoked" if endpoint.endswith("/undo") else payload.get("decision")
                self._json(HTTPStatus.OK, self.server.adjudicate(run_id, decision if isinstance(decision, str) else ""))
            else:
                launch = self.server.launch_baseline if endpoint == "/api/launch-baseline" else self.server.launch
                self._json(HTTPStatus.OK, launch(run_id))
        except ResultUnavailableError as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": f"Runnable result unavailable: {exc}"})
        except BaselineUnavailableError as exc:
            self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": f"Baseline unavailable: {exc}"})
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
