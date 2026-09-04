"""Pinned Hermes 0.21.0 profile materialization and process-backed adapter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import Field, model_validator

from agent_bench.capture import CaptureCapabilities
from agent_bench.harness import HarnessExecutionResult, HarnessRunContext
from agent_bench.hermes_events import normalize_hermes_events
from agent_bench.models import Identifier, PersistedModel, Sha256, canonical_sha256

DEFAULT_HERMES_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "environment"
    / "harnesses"
    / "hermes-default-v1"
    / "profile.yaml"
)


class HermesError(RuntimeError):
    """Raised when the benchmark-managed Hermes toolchain cannot run safely."""


class HermesRuntime(PersistedModel):
    node_path: Path
    node_sha256: Sha256
    node_version: Literal["v26.8.1"]
    python_path: Path
    python_sha256: Sha256
    python_version: Literal["Python 3.11.16"]
    entrypoint_path: Path
    entrypoint_sha256: Sha256
    version_output: Literal["0.21.0"]
    source_root: Path
    source_tree_sha256: Sha256
    source_tree_records: int = Field(ge=1)
    environment_root: Path
    environment_tree_sha256: Sha256
    environment_tree_records: int = Field(ge=1)
    python_root: Path
    python_tree_sha256: Sha256
    python_tree_records: int = Field(ge=1)
    uv_lock_sha256: Sha256
    pyproject_sha256: Sha256

    @model_validator(mode="after")
    def require_absolute_paths(self) -> HermesRuntime:
        for name in (
            "node_path", "python_path", "entrypoint_path", "source_root", "environment_root", "python_root"
        ):
            if not getattr(self, name).is_absolute():
                raise ValueError(f"{name} must be absolute")
        return self


class HermesProfile(PersistedModel):
    profile_id: Literal["hermes-default-v1"] = "hermes-default-v1"
    profile_version: Literal["1.0.0"] = "1.0.0"
    profile_path: Path
    toolchain: HermesRuntime
    config_file: Path
    config_sha256: Sha256
    proxy_base_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/v1$")
    provider_id: Identifier
    model_id: Identifier
    invocation: dict[str, object]
    deviations: tuple[str, ...]

    @model_validator(mode="after")
    def require_absolute_paths(self) -> HermesProfile:
        if not self.profile_path.is_absolute() or not self.config_file.is_absolute():
            raise ValueError("Hermes profile paths must be absolute")
        return self


def load_hermes_profile(path: Path = DEFAULT_HERMES_PROFILE_PATH) -> HermesProfile:
    """Load the controlled profile and its separately pinned toolchain identity."""
    profile_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HermesError(f"cannot load Hermes profile {profile_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HermesError("Hermes profile must be a YAML mapping")
    root = profile_path.parent
    identity = raw.pop("toolchain_identity_file", None)
    if not isinstance(identity, str):
        raise HermesError("Hermes profile toolchain_identity_file must be a string")
    raw["profile_path"] = profile_path
    raw["config_file"] = (root / str(raw.get("config_file", ""))).resolve()
    raw["toolchain"] = _load_toolchain((root / identity).resolve())
    try:
        profile = HermesProfile.model_validate(raw)
    except ValueError as exc:
        raise HermesError(f"invalid Hermes profile: {exc}") from exc
    _verify_file(profile.config_file, profile.config_sha256)
    return profile


def _load_toolchain(identity_path: Path) -> HermesRuntime:
    try:
        raw = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HermesError(f"cannot load Hermes toolchain identity {identity_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HermesError("Hermes toolchain identity must be a JSON object")
    try:
        runtime = raw["runtime"]
        installation = raw["installation"]
        source = raw["source"]
        python = raw["python"]
        if not all(isinstance(value, dict) for value in (runtime, installation, source, python)):
            raise KeyError("runtime/installation/source/python")
        return HermesRuntime.model_validate(
            {
                "node_path": _resolve_toolchain_path(raw["node"]["executable_path"]),
                "node_sha256": raw["node"]["executable_sha256"],
                "node_version": raw["node"]["version"],
                "python_path": _resolve_toolchain_path(python["executable_path"]),
                "python_sha256": python["executable_sha256"],
                "python_version": python["version"],
                "entrypoint_path": _resolve_toolchain_path(runtime["entrypoint_path"]),
                "entrypoint_sha256": runtime["entrypoint_sha256"],
                "version_output": runtime["version_output"],
                "source_root": _resolve_toolchain_path(source["source_root"]),
                "source_tree_sha256": source["tree_sha256"],
                "source_tree_records": source["tree_records"],
                "environment_root": _resolve_toolchain_path(installation["venv_root"]),
                "environment_tree_sha256": installation["venv_tree_sha256"],
                "environment_tree_records": installation["venv_tree_records"],
                "python_root": _resolve_toolchain_path(python["root"]),
                "python_tree_sha256": python["tree_sha256"],
                "python_tree_records": python["tree_records"],
                "uv_lock_sha256": source["uv_lock_sha256"],
                "pyproject_sha256": source["pyproject_sha256"],
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HermesError(f"invalid Hermes toolchain identity: {exc}") from exc


def _resolve_toolchain_path(value: object) -> Path:
    if not isinstance(value, str):
        raise HermesError("toolchain path must be a string")
    path = Path(value)
    if not path.is_absolute():
        return (Path(__file__).resolve().parents[2] / path).resolve()
    if "toolchains" in path.parts:
        return Path(__file__).resolve().parents[2] / "toolchains" / Path(*path.parts[path.parts.index("toolchains") + 1:])
    return path


def inspect_hermes_toolchain(
    toolchain: HermesRuntime,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> HermesRuntime:
    """Verify every benchmark-owned runtime boundary before a task starts."""
    _verify_file(toolchain.node_path, toolchain.node_sha256)
    _verify_file(toolchain.python_path, toolchain.python_sha256, allow_symlink=True)
    _verify_file(toolchain.entrypoint_path, toolchain.entrypoint_sha256)
    _verify_file(toolchain.source_root / "uv.lock", toolchain.uv_lock_sha256)
    _verify_file(toolchain.source_root / "pyproject.toml", toolchain.pyproject_sha256)
    for root, expected, records, label in (
        (toolchain.source_root, toolchain.source_tree_sha256, toolchain.source_tree_records, "source"),
        (toolchain.environment_root, toolchain.environment_tree_sha256, toolchain.environment_tree_records, "environment"),
        (toolchain.python_root, toolchain.python_tree_sha256, toolchain.python_tree_records, "python"),
    ):
        observed, count = _tree_digest(root, exclude_generated=(label == "source"))
        if observed != expected or count != records:
            raise HermesError(f"Hermes {label} tree differs from the pinned toolchain")
    with tempfile.TemporaryDirectory(prefix="agent-bench-hermes-inspect-") as temporary:
        root = Path(temporary)
        environment = _base_environment(root / "home", root / "config", root / "cache", root / "data", root / "state", root / "hermes")
        for value in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "HERMES_HOME"):
            Path(environment[value]).mkdir(parents=True, exist_ok=True)
        result = run_command([str(toolchain.entrypoint_path), "--version"], cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
    version = _parse_version(result.stdout.decode("utf-8", errors="replace"))
    if result.returncode != 0 or version != toolchain.version_output:
        raise HermesError(f"pinned Hermes version inspection failed: code={result.returncode}, version={version!r}")
    python = run_command([str(toolchain.python_path), "--version"], cwd=toolchain.environment_root, env=_base_environment(Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp")), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
    version_text = (python.stdout + python.stderr).decode("utf-8", errors="replace").strip()
    if python.returncode != 0 or version_text != toolchain.python_version:
        raise HermesError("pinned Hermes Python runtime version differs from identity")
    node = run_command([str(toolchain.node_path), "--version"], cwd=toolchain.environment_root, env=_base_environment(Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp")), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30)
    node_version = (node.stdout + node.stderr).decode("utf-8", errors="replace").strip()
    if node.returncode != 0 or node_version != toolchain.node_version:
        raise HermesError("pinned Hermes Node runtime version differs from identity")
    return toolchain


def materialize_hermes_profile(profile: HermesProfile, context: HarnessRunContext) -> Path:
    """Copy only the versioned source profile into a new Hermes home."""
    _verify_file(profile.config_file, profile.config_sha256)
    home = context.paths.harness_state / "hermes-home"
    if home.exists():
        raise HermesError(f"Hermes run home already exists: {home}")
    home.mkdir(parents=True)
    shutil.copyfile(profile.config_file, home / "config.yaml")
    return home


def hermes_environment(context: HarnessRunContext, hermes_home: Path) -> dict[str, str]:
    """Build Hermes's complete explicit isolated child environment."""
    environment = _base_environment(context.paths.home, context.paths.xdg_config_home, context.paths.xdg_cache_home, context.paths.xdg_data_home, context.paths.xdg_state_home, hermes_home)
    environment.update({
        "AGENT_BENCH_HARNESS_STATE": str(context.paths.harness_state),
        "PYTHONNOUSERSITE": "1",
        # Hermes file and terminal tools intentionally use this explicit
        # workspace anchor.  Process cwd alone is not sufficient for oneshot
        # sessions, whose default terminal anchor is the isolated Hermes home.
        "TERMINAL_CWD": str(context.paths.workspace),
    })
    return environment


def build_hermes_command(profile: HermesProfile, context: HarnessRunContext, usage_file: Path) -> tuple[str, ...]:
    """Build an argv-element-exact upstream Hermes oneshot invocation."""
    return (str(profile.toolchain.entrypoint_path), "--model", profile.model_id, "--provider", profile.provider_id, "--usage-file", str(usage_file), "--oneshot", context.prompt_content)


def hermes_capture_capabilities() -> CaptureCapabilities:
    """Declare the exact proxy and Hermes SQLite/usage evidence boundaries."""
    return CaptureCapabilities(
        capability_id="hermes-0.21.0-proxy-v1", backend_id="llamacpp-qwen38-agent-bench-v1", harness_id="hermes",
        raw_request_payload="proxy_exact", raw_response_payload="proxy_exact", request_generation_parameters="proxy_exact",
        input_token_usage="api_exact", output_token_usage="api_exact", reasoning_content="harness_exact",
        reasoning_token_count="harness_exact", context_token_count="api_exact", finish_reason="harness_exact",
        tool_calls="harness_exact", tool_results="harness_exact", compaction_events="harness_exact",
        session_identity="harness_exact", serialized_prompt_history_validation="harness_exact",
        empty_historical_think_block_detection="proxy_exact",
        notes=(
            "Proxy request and response bodies remain authoritative for model traffic and parameters.",
            "Hermes's isolated SQLite session provides persisted user/assistant/tool records, including reasoning_content when exposed.",
            "The oneshot usage file provides session identity and aggregate native usage; post-Jinja prompt bytes remain unavailable.",
        ),
    )


class HermesAdapter:
    """Run one fresh Hermes 0.21.0 oneshot session without personal state."""

    adapter_id = "hermes"
    adapter_version = "1.0.0"
    capture_capabilities = hermes_capture_capabilities()

    def __init__(self, profile: HermesProfile | None = None, *, popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen, verify_toolchain: bool = True) -> None:
        self.profile = profile or load_hermes_profile()
        self._popen = popen
        self._verify_toolchain = verify_toolchain

    def normalize_events(self, raw_path: Path, normalized_path: Path) -> None:
        normalize_hermes_events(raw_path, normalized_path)

    def run(self, context: HarnessRunContext) -> HarnessExecutionResult:
        if context.proxy_endpoint is None:
            raise HermesError("Hermes requires an Agent Bench proxy endpoint")
        if self._verify_toolchain:
            inspect_hermes_toolchain(self.profile.toolchain)
        hermes_home = materialize_hermes_profile(self.profile, context)
        environment = hermes_environment(context, hermes_home)
        evidence_root = context.paths.harness_state / "hermes"
        evidence_root.mkdir()
        prompt_bytes = context.prompt_content.encode("utf-8")
        prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        (evidence_root / "prompt-transport.bin").write_bytes(prompt_bytes)
        usage_file = evidence_root / "usage.json"
        argv = build_hermes_command(self.profile, context, usage_file)
        _write_json(evidence_root / "invocation.json", {"schema_version": "1.0.0", "argv": list(argv), "prompt_sha256": prompt_sha256, "prompt_byte_length": len(prompt_bytes), "prompt_delivery": "argv_element_exact_utf8", "working_directory": str(context.paths.workspace), "environment": environment, "profile_digest": self.profile.definition_digest, "run_seed": context.run_seed})
        context.events.emit(source="harness", event_type="hermes_start", payload={"profile_id": self.profile.profile_id, "model": self.profile.model_id, "proxy_endpoint": context.proxy_endpoint, "prompt_sha256": prompt_sha256, "prompt_delivery": "argv_element_exact_utf8", "fresh_session": True, "continued_session": False, "environment": environment, "argv": list(argv)})
        stdout_path, stderr_path = evidence_root / "stdout.log", evidence_root / "stderr.log"
        process = self._popen(list(argv), cwd=context.paths.workspace, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, start_new_session=True)
        if process.stdout is None or process.stderr is None:
            raise HermesError("Hermes process pipes were not created")
        errors: list[BaseException] = []
        stdout_thread = threading.Thread(target=self._copy_stream, args=(process.stdout, stdout_path, errors), name="agent-bench-hermes-stdout")
        stderr_thread = threading.Thread(target=self._copy_stream, args=(process.stderr, stderr_path, errors), name="agent-bench-hermes-stderr")
        stdout_thread.start(); stderr_thread.start()
        cancelled = False
        while process.poll() is None:
            if context.cancellation.wait(0.05):
                cancelled = True; _terminate_owned_process(process); break
        return_code = process.wait()
        stdout_thread.join(timeout=5); stderr_thread.join(timeout=5)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise HermesError("Hermes output reader did not finish")
        if errors:
            raise HermesError(f"Hermes output capture failed: {errors[0]}")
        usage = _load_json(usage_file)
        _write_json(evidence_root / "usage-observed.json", usage)
        session_id = usage.get("session_id") if isinstance(usage.get("session_id"), str) else None
        records = _export_session_records(hermes_home / "state.db", session_id)
        _write_json(evidence_root / "session-records.json", records)
        _emit_native_records(context, records, prompt_sha256)
        output_truncated = any(row.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"} for row in records.get("messages", []) if isinstance(row, dict))
        context.events.emit(source="harness", event_type="hermes_end", payload={"return_code": return_code, "cancelled": cancelled, "session_id": session_id, "usage": usage, "output_truncated": output_truncated})
        return HarnessExecutionResult(completed_normally=return_code == 0 and not cancelled, output_truncated=output_truncated, exit_code=return_code, session_id=session_id, evidence_files=self._evidence_files(context, evidence_root))

    @staticmethod
    def _copy_stream(source: object, destination: Path, errors: list[BaseException]) -> None:
        try:
            with destination.open("xb") as output:
                while chunk := source.read(64 * 1024): output.write(chunk)  # type: ignore[attr-defined]
                output.flush(); os.fsync(output.fileno())
        except BaseException as exc:
            errors.append(exc)

    @staticmethod
    def _evidence_files(context: HarnessRunContext, evidence_root: Path) -> tuple[tuple[str, Path], ...]:
        records: dict[str, Path] = {
            "raw/hermes/stdout.log": evidence_root / "stdout.log", "raw/hermes/stderr.log": evidence_root / "stderr.log",
            "raw/hermes/prompt-transport.bin": evidence_root / "prompt-transport.bin", "raw/hermes/invocation.json": evidence_root / "invocation.json",
            "raw/hermes/usage.json": evidence_root / "usage.json", "raw/hermes/usage-observed.json": evidence_root / "usage-observed.json",
            "raw/hermes/session-records.json": evidence_root / "session-records.json",
        }
        for source in sorted((context.paths.harness_state / "hermes-home").rglob("*")):
            if source.is_file() and not source.is_symlink(): records[f"run/hermes/home/{source.relative_to(context.paths.harness_state / 'hermes-home').as_posix()}"] = source
        for label, root in (("home", context.paths.home), ("config", context.paths.xdg_config_home), ("cache", context.paths.xdg_cache_home), ("data", context.paths.xdg_data_home), ("state", context.paths.xdg_state_home)):
            for source in sorted(root.rglob("*")):
                if source.is_file() and not source.is_symlink(): records[f"run/hermes/{label}/{source.relative_to(root).as_posix()}"] = source
        return tuple(sorted(records.items()))


def _emit_native_records(context: HarnessRunContext, records: dict[str, object], prompt_sha256: str) -> None:
    messages = records.get("messages")
    rows = messages if isinstance(messages, list) else []
    user_messages: list[str] = []
    for row in rows:
        if not isinstance(row, dict): continue
        context.events.emit(source="harness", event_type="hermes_session_message", payload={"native_event": row})
        if row.get("role") == "user": user_messages.extend(_message_texts(row.get("content")))
        if row.get("compacted") in {True, 1}:
            context.events.emit(source="harness", event_type="hermes_session_compaction", payload={"native_event": row})
    exact = any(hashlib.sha256(value.encode("utf-8")).hexdigest() == prompt_sha256 for value in user_messages)
    context.events.emit(source="harness", event_type="hermes_prompt_validation", payload={"expected_prompt_sha256": prompt_sha256, "exact_prompt_found": exact, "candidate_user_text_count": len(user_messages), "session_message_count": len(rows), "method": "hermes_sqlite_session_exact_utf8"})
    if not exact:
        context.events.emit(source="harness", event_type="harness_error", payload={"error_type": "invalid_harness_output", "message": "Hermes session did not contain exact prompt bytes"})


def _export_session_records(database: Path, session_id: str | None) -> dict[str, object]:
    if not database.is_file():
        return {"database_present": False, "sessions": [], "messages": []}
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "sessions" not in tables or "messages" not in tables:
            return {"database_present": True, "sessions": [], "messages": [], "schema_error": "required tables unavailable"}
        where, params = ("WHERE id = ?", (session_id,)) if session_id else ("", ())
        sessions = [_row_to_json(row) for row in connection.execute(f"SELECT * FROM sessions {where} ORDER BY started_at, id", params)]
        ids = [row.get("id") for row in sessions if isinstance(row.get("id"), str)]
        if not ids:
            return {"database_present": True, "sessions": sessions, "messages": []}
        placeholders = ",".join("?" for _ in ids)
        messages = [_row_to_json(row) for row in connection.execute(f"SELECT * FROM messages WHERE session_id IN ({placeholders}) ORDER BY id", ids)]
        return {"database_present": True, "sessions": sessions, "messages": messages}
    except sqlite3.Error as exc:
        return {"database_present": True, "sessions": [], "messages": [], "schema_error": str(exc)}
    finally:
        try: connection.close()  # type: ignore[has-type]
        except Exception: pass


def _row_to_json(row: sqlite3.Row) -> dict[str, object]:
    return {key: _json_value(row[key]) for key in row.keys()}


def _json_value(value: object) -> object:
    if isinstance(value, bytes): return {"base64": __import__("base64").b64encode(value).decode("ascii")}
    return value


def _message_texts(value: object) -> list[str]:
    if isinstance(value, str):
        try: value = json.loads(value)
        except json.JSONDecodeError: return [value]
    if isinstance(value, str): return [value]
    if isinstance(value, list): return [item["text"] for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)]
    if isinstance(value, dict):
        content = value.get("text") or value.get("content")
        return _message_texts(content) if content is not None else []
    return []


def _base_environment(home: Path, config: Path, cache: Path, data: Path, state: Path, hermes_home: Path) -> dict[str, str]:
    node_bin = Path(__file__).resolve().parents[2] / "toolchains" / "node" / "26.8.1" / "bin"
    return {"HOME": str(home), "XDG_CONFIG_HOME": str(config), "XDG_CACHE_HOME": str(cache), "XDG_DATA_HOME": str(data), "XDG_STATE_HOME": str(state), "HERMES_HOME": str(hermes_home), "PATH": f"{node_bin}:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "TERM": "dumb", "CI": "true", "NO_COLOR": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def _tree_digest(root: Path, *, exclude_generated: bool = False) -> tuple[str, int]:
    if not root.is_dir(): raise HermesError(f"pinned Hermes tree is missing: {root}")
    digest = hashlib.sha256(); records = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if exclude_generated and ("__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}):
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink(): record = b"L\0" + relative + b"\0" + os.readlink(path).encode("utf-8") + b"\n"
        elif path.is_file(): record = b"F\0" + relative + b"\0" + _sha256_file(path).encode("ascii") + b"\n"
        else: continue
        digest.update(record); records += 1
    return digest.hexdigest(), records


def _verify_file(path: Path, expected: str, *, allow_symlink: bool = False) -> None:
    if not path.is_file() or (path.is_symlink() and not allow_symlink): raise HermesError(f"pinned file is missing or invalid: {path}")
    if _sha256_file(path) != expected: raise HermesError(f"pinned file SHA256 differs: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _parse_version(value: str) -> str:
    for line in value.splitlines():
        if line.startswith("Hermes Agent v"): return line.split("v", 1)[1].split(" ", 1)[0]
    return ""


def _load_json(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {"parse_error": "usage JSON is not an object"}
    except (OSError, json.JSONDecodeError) as exc:
        return {"parse_error": str(exc)}


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> None:
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: return
        process.wait(timeout=5)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8", newline="\n")
