"""Pinned Pi 0.84.4 profile materialization and process-backed adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from agent_bench.capture import CaptureCapabilities
from agent_bench.harness import HarnessExecutionResult, HarnessRunContext
from agent_bench.models import Identifier, PersistedModel, Sha256
from agent_bench.pi_events import normalize_pi_events

DEFAULT_PI_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "environment"
    / "harnesses"
    / "pi-default-v1"
    / "profile.yaml"
)


class PiError(RuntimeError):
    """Raised when the benchmark-managed Pi toolchain cannot run safely."""


class PiNodeRuntime(PersistedModel):
    path: Path
    size_bytes: int = Field(ge=1)
    sha256: Sha256
    version: str = Field(min_length=1)


class PiToolchain(PersistedModel):
    identity_path: Path
    package_name: Literal["@earendil-works/pi-coding-agent"]
    package_version: Literal["0.84.4"]
    package_integrity: str = Field(min_length=1)
    installation_root: Path
    package_lock_sha256: Sha256
    node_modules_tree_sha256: Sha256
    node_modules_tree_records: int = Field(ge=1)
    node: PiNodeRuntime
    entrypoint_path: Path
    entrypoint_size_bytes: int = Field(ge=1)
    entrypoint_sha256: Sha256
    version_output: Literal["0.84.4"]
    cli_help_sha256: Sha256

    @model_validator(mode="after")
    def require_absolute_paths(self) -> PiToolchain:
        for field_name in ("identity_path", "installation_root", "entrypoint_path"):
            if not getattr(self, field_name).is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        if not self.node.path.is_absolute():
            raise ValueError("node.path must be absolute")
        return self


class PiInvocationPolicy(PersistedModel):
    output_mode: Literal["json"] = "json"
    non_interactive: Literal[True] = True
    project_trust: Literal["upstream_fresh_default"] = "upstream_fresh_default"
    startup_network: Literal["disabled"] = "disabled"
    telemetry: Literal["disabled"] = "disabled"
    prompt_delivery: Literal["positional_exact_utf8"] = "positional_exact_utf8"
    thinking_level: Literal["upstream_default_medium"] = "upstream_default_medium"


class PiProfile(PersistedModel):
    profile_id: Literal["pi-default-v1"] = "pi-default-v1"
    profile_version: Literal["1.0.1"] = "1.0.1"
    profile_path: Path
    toolchain: PiToolchain
    models_file: Path
    models_sha256: Sha256
    settings_file: Path
    settings_sha256: Sha256
    proxy_base_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/v1$")
    provider_id: Identifier
    model_id: Identifier
    invocation: PiInvocationPolicy
    deviations: tuple[str, ...]

    @model_validator(mode="after")
    def require_absolute_paths(self) -> PiProfile:
        for field_name in ("profile_path", "models_file", "settings_file"):
            if not getattr(self, field_name).is_absolute():
                raise ValueError(f"{field_name} must be absolute")
        return self


def load_pi_profile(path: Path = DEFAULT_PI_PROFILE_PATH) -> PiProfile:
    """Load the checked-in Pi profile and verify immutable source files."""
    profile_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PiError(f"cannot load Pi profile {profile_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PiError("Pi profile must be a YAML mapping")
    root = profile_path.parent
    identity_value = raw.pop("toolchain_identity_file", None)
    if not isinstance(identity_value, str):
        raise PiError("Pi profile toolchain_identity_file must be a string")
    identity_path = (root / identity_value).resolve()
    raw["profile_path"] = profile_path
    raw["models_file"] = (root / str(raw.get("models_file", ""))).resolve()
    raw["settings_file"] = (root / str(raw.get("settings_file", ""))).resolve()
    raw["toolchain"] = _load_toolchain(identity_path)
    try:
        profile = PiProfile.model_validate(raw)
    except ValueError as exc:
        raise PiError(f"invalid Pi profile: {exc}") from exc
    _verify_file(profile.models_file, profile.models_sha256)
    _verify_file(profile.settings_file, profile.settings_sha256)
    return profile


def _load_toolchain(identity_path: Path) -> PiToolchain:
    try:
        raw = json.loads(identity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PiError(f"cannot load Pi toolchain identity {identity_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PiError("Pi toolchain identity must be a JSON object")
    package = raw.get("package")
    installation = raw.get("installation")
    runtime = raw.get("runtime")
    if not all(isinstance(value, dict) for value in (package, installation, runtime)):
        raise PiError("Pi toolchain identity lacks package, installation, or runtime data")
    try:
        return PiToolchain.model_validate(
            {
                "identity_path": identity_path,
                "package_name": package["name"],
                "package_version": package["version"],
                "package_integrity": package["registry_integrity"],
                "installation_root": _resolve_toolchain_path(installation["root"]),
                "package_lock_sha256": installation["package_lock_sha256"],
                "node_modules_tree_sha256": installation["node_modules_tree_sha256"],
                "node_modules_tree_records": installation["node_modules_tree_records"],
                "node": {
                    "path": _resolve_toolchain_path(runtime["node_path"]),
                    "size_bytes": runtime["node_size_bytes"],
                    "sha256": runtime["node_sha256"],
                    "version": installation["node_version"],
                },
                "entrypoint_path": _resolve_toolchain_path(runtime["entrypoint_path"]),
                "entrypoint_size_bytes": runtime["entrypoint_size_bytes"],
                "entrypoint_sha256": runtime["entrypoint_sha256"],
                "version_output": runtime["version_output"],
                "cli_help_sha256": runtime["cli_help_sha256"],
            }
        )
    except (KeyError, ValueError) as exc:
        raise PiError(f"invalid Pi toolchain identity: {exc}") from exc


def _resolve_toolchain_path(value: object) -> Path:
    if not isinstance(value, str):
        raise PiError("toolchain path must be a string")
    path = Path(value)
    if not path.is_absolute():
        return (Path(__file__).resolve().parents[2] / path).resolve()
    if "toolchains" in path.parts:
        return Path(__file__).resolve().parents[2] / "toolchains" / Path(*path.parts[path.parts.index("toolchains") + 1:])
    return path


def inspect_pi_toolchain(
    toolchain: PiToolchain,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> PiToolchain:
    """Verify pinned Node, entrypoint, package lock, installed graph, and version."""
    _verify_file(toolchain.installation_root / "package-lock.json", toolchain.package_lock_sha256)
    _verify_file(toolchain.entrypoint_path, toolchain.entrypoint_sha256, toolchain.entrypoint_size_bytes)
    _verify_file(toolchain.node.path, toolchain.node.sha256, toolchain.node.size_bytes)
    observed_tree, observed_records = _node_modules_tree_digest(toolchain.installation_root / "node_modules")
    if observed_tree != toolchain.node_modules_tree_sha256 or observed_records != toolchain.node_modules_tree_records:
        raise PiError("installed Pi node_modules identity differs from the pinned toolchain")
    with tempfile.TemporaryDirectory(prefix="agent-bench-pi-inspect-") as root_text:
        root = Path(root_text)
        environment = _base_environment(root / "home", root / "config", root / "cache", root / "data", root / "state")
        for value in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
            Path(environment[value]).mkdir(parents=True, exist_ok=True)
        result = run_command(
            [str(toolchain.node.path), str(toolchain.entrypoint_path), "--version"],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    version = result.stdout.decode("utf-8", errors="replace").strip()
    if result.returncode != 0 or version != toolchain.version_output:
        raise PiError(
            f"pinned Pi version inspection failed: code={result.returncode}, output={version!r}"
        )
    node_result = run_command(
        [str(toolchain.node.path), "--version"],
        cwd=toolchain.installation_root,
        env=_base_environment(Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp"), Path("/tmp")),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    node_version = node_result.stdout.decode("utf-8", errors="replace").strip()
    if node_result.returncode != 0 or node_version != toolchain.node.version:
        raise PiError("pinned Node runtime version differs from the toolchain identity")
    return toolchain


def materialize_pi_profile(profile: PiProfile, context: HarnessRunContext) -> Path:
    """Copy the immutable controlled profile into a fresh isolated Pi config root."""
    _verify_file(profile.models_file, profile.models_sha256)
    _verify_file(profile.settings_file, profile.settings_sha256)
    agent_dir = context.paths.xdg_config_home / "pi" / "agent"
    if agent_dir.exists():
        raise PiError(f"Pi run config already exists: {agent_dir}")
    agent_dir.mkdir(parents=True)
    models_destination = agent_dir / "models.json"
    settings_destination = agent_dir / "settings.json"
    shutil.copyfile(profile.models_file, models_destination)
    shutil.copyfile(profile.settings_file, settings_destination)
    endpoint = context.proxy_endpoint or profile.proxy_base_url
    if endpoint != profile.proxy_base_url:
        models = json.loads(models_destination.read_text(encoding="utf-8"))
        models["providers"][profile.provider_id]["baseUrl"] = endpoint
        models_destination.write_text(json.dumps(models, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    _verify_file(profile.models_file, profile.models_sha256)
    _verify_file(profile.settings_file, profile.settings_sha256)
    return agent_dir


def pi_environment(context: HarnessRunContext, agent_dir: Path) -> dict[str, str]:
    """Build Pi's complete explicit isolated child environment."""
    environment = _base_environment(
        context.paths.home,
        context.paths.xdg_config_home,
        context.paths.xdg_cache_home,
        context.paths.xdg_data_home,
        context.paths.xdg_state_home,
    )
    environment.update(
        {
            "PI_CODING_AGENT_DIR": str(agent_dir),
            "PI_CODING_AGENT_SESSION_DIR": str(context.paths.xdg_data_home / "pi" / "sessions"),
            "PI_TELEMETRY": "0",
            "AGENT_BENCH_HARNESS_STATE": str(context.paths.harness_state),
        }
    )
    return environment


def build_pi_command(profile: PiProfile, context: HarnessRunContext) -> tuple[str, ...]:
    """Build the native JSON/non-interactive argv, including exact positional prompt bytes."""
    return (
        str(profile.toolchain.node.path),
        str(profile.toolchain.entrypoint_path),
        "--provider", profile.provider_id,
        "--model", profile.model_id,
        "--mode", "json",
        "--print",
        "--offline",
        "--",
        context.prompt_content,
    )


def pi_capture_capabilities() -> CaptureCapabilities:
    """Declare proxy and Pi 0.84.4 native JSON capture boundaries."""
    return CaptureCapabilities(
        capability_id="pi-0.84.4-proxy-v1",
        backend_id="llamacpp-qwen38-agent-bench-v1",
        harness_id="pi",
        raw_request_payload="proxy_exact",
        raw_response_payload="proxy_exact",
        request_generation_parameters="proxy_exact",
        input_token_usage="api_exact",
        output_token_usage="api_exact",
        reasoning_content="harness_exact",
        reasoning_token_count="api_exact",
        context_token_count="api_exact",
        finish_reason="harness_exact",
        tool_calls="harness_exact",
        tool_results="harness_exact",
        compaction_events="harness_exact",
        session_identity="harness_exact",
        serialized_prompt_history_validation="harness_exact",
        empty_historical_think_block_detection="proxy_exact",
        notes=(
            "The transparent proxy is authoritative for complete LLM request and response bodies.",
            "Pi JSON message_end records provide final reasoning, usage, and finish state; usage is only api_exact when supplied by llama.cpp.",
            "Pi JSON exposes tool execution and compaction lifecycle events.",
            "Session JSONL confirms exact prompt bytes, while empty-think detection remains request-message-only because post-Jinja prompts are unavailable.",
        ),
    )


class PiAdapter:
    """Run one fresh Pi session using the pinned Node and Pi 0.84.4 entrypoint."""

    adapter_id = "pi"
    adapter_version = "1.0.1"
    capture_capabilities = pi_capture_capabilities()

    def __init__(
        self,
        profile: PiProfile | None = None,
        *,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        verify_toolchain: bool = True,
    ) -> None:
        self.profile = profile or load_pi_profile()
        self._popen = popen
        self._verify_toolchain = verify_toolchain

    def normalize_events(self, raw_path: Path, normalized_path: Path) -> None:
        normalize_pi_events(raw_path, normalized_path)

    def run(self, context: HarnessRunContext) -> HarnessExecutionResult:
        if context.proxy_endpoint is None:
            raise PiError("Pi requires an Agent Bench proxy endpoint")
        if self._verify_toolchain:
            inspect_pi_toolchain(self.profile.toolchain)
        agent_dir = materialize_pi_profile(self.profile, context)
        environment = pi_environment(context, agent_dir)
        argv = build_pi_command(self.profile, context)
        prompt_bytes = context.prompt_content.encode("utf-8")
        evidence_root = context.paths.harness_state / "pi"
        evidence_root.mkdir()
        (evidence_root / "prompt-transport.bin").write_bytes(prompt_bytes)
        stdout_path = evidence_root / "stdout.jsonl"
        stderr_path = evidence_root / "stderr.log"
        _write_json(evidence_root / "invocation.json", {
            "schema_version": "1.0.0", "argv": list(argv), "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
            "prompt_byte_length": len(prompt_bytes), "prompt_delivery": "positional_exact_utf8", "working_directory": str(context.paths.workspace),
            "environment": environment, "profile_digest": self.profile.definition_digest, "run_seed": context.run_seed,
        })
        context.events.emit(source="harness", event_type="pi_start", payload={
            "profile_id": self.profile.profile_id, "model": self.profile.model_id, "proxy_endpoint": context.proxy_endpoint,
            "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(), "prompt_delivery": "positional_exact_utf8",
            "fresh_session": True, "continued_session": False, "environment": environment, "argv": list(argv),
        })
        process = self._popen(list(argv), cwd=context.paths.workspace, env=environment, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, start_new_session=True)
        if process.stdout is None or process.stderr is None:
            raise PiError("Pi process pipes were not created")
        session_ids: set[str] = set()
        output_truncated = threading.Event()
        errors: list[BaseException] = []
        stdout_thread = threading.Thread(target=self._capture_stdout, args=(process.stdout, stdout_path, context, session_ids, output_truncated, errors), name="agent-bench-pi-stdout")
        stderr_thread = threading.Thread(target=self._capture_stderr, args=(process.stderr, stderr_path, errors), name="agent-bench-pi-stderr")
        stdout_thread.start(); stderr_thread.start()
        cancelled = False
        while process.poll() is None:
            if context.cancellation.wait(0.05):
                cancelled = True; _terminate_owned_process(process); break
        return_code = process.wait()
        stdout_thread.join(timeout=5); stderr_thread.join(timeout=5)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise PiError("Pi output reader did not finish")
        if errors:
            raise PiError(f"Pi output capture failed: {errors[0]}")
        session_id = next(iter(session_ids)) if len(session_ids) == 1 else None
        validation = self._validate_session(
            context.paths.xdg_data_home / "pi" / "sessions",
            context,
            session_id,
            hashlib.sha256(prompt_bytes).hexdigest(),
        )
        if validation == "mismatch":
            context.events.emit(source="harness", event_type="harness_error", payload={"error_type": "invalid_harness_output", "message": "Pi session did not contain exact prompt bytes"})
        if return_code != 0 and not cancelled:
            context.events.emit(source="harness", event_type="harness_error", payload={"error_type": "pi_nonzero_exit", "exit_code": return_code})
        _write_json(evidence_root / "result.json", {"schema_version": "1.0.0", "session_id": session_id, "exit_code": return_code,
            "termination_signal": -return_code if return_code < 0 else None, "cancelled_by_runner": cancelled,
            "output_truncated": output_truncated.is_set(), "prompt_session_validation": validation})
        return HarnessExecutionResult(completed_normally=return_code == 0 and not cancelled, output_truncated=output_truncated.is_set(),
            exit_code=return_code, termination_signal=-return_code if return_code < 0 else None, session_id=session_id,
            evidence_files=self._evidence_files(context, evidence_root))

    def _capture_stdout(self, source: object, destination: Path, context: HarnessRunContext, session_ids: set[str], output_truncated: threading.Event, errors: list[BaseException]) -> None:
        try:
            with destination.open("xb") as output:
                index = 0
                while line := source.readline():  # type: ignore[attr-defined]
                    output.write(line); output.flush(); index += 1
                    payload: dict[str, object] = {"stream_index": index, "line_base64": base64.b64encode(line).decode("ascii"), "line_sha256": hashlib.sha256(line).hexdigest()}
                    try: native = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        context.events.emit(source="stdout", event_type="pi_unparsed_stdout", payload=payload); continue
                    payload["native_event"] = native
                    if isinstance(native, dict):
                        if native.get("type") == "session" and isinstance(native.get("id"), str): session_ids.add(native["id"])
                        if _native_output_truncated(native): output_truncated.set()
                    context.events.emit(source="harness", event_type="pi_event", payload=payload)
                output.flush(); os.fsync(output.fileno())
        except BaseException as exc: errors.append(exc)

    def _capture_stderr(self, source: object, destination: Path, errors: list[BaseException]) -> None:
        try:
            with destination.open("xb") as output:
                while chunk := source.read(64 * 1024): output.write(chunk)  # type: ignore[attr-defined]
                output.flush(); os.fsync(output.fileno())
        except BaseException as exc: errors.append(exc)

    def _validate_session(self, session_dir: Path, context: HarnessRunContext, session_id: str | None, prompt_sha: str) -> str:
        matches: list[str] = []
        files = sorted(session_dir.glob("*.jsonl")) if session_dir.is_dir() else []
        for file in files:
            for line in file.read_text(encoding="utf-8").splitlines():
                try: entry = json.loads(line)
                except json.JSONDecodeError: continue
                if not isinstance(entry, dict) or entry.get("type") != "message": continue
                message = entry.get("message")
                if isinstance(message, dict) and message.get("role") == "user": matches.extend(_message_texts(message))
        exact = any(hashlib.sha256(value.encode("utf-8")).hexdigest() == prompt_sha for value in matches)
        context.events.emit(source="harness", event_type="pi_prompt_validation", payload={"session_id": session_id, "expected_prompt_sha256": prompt_sha,
            "exact_prompt_found": exact, "candidate_user_text_count": len(matches), "session_file_count": len(files), "method": "pi_session_jsonl_exact_utf8"})
        return "exact_match" if exact else "mismatch"

    def _evidence_files(self, context: HarnessRunContext, evidence_root: Path) -> tuple[tuple[str, Path], ...]:
        records = {"raw/pi/stdout.jsonl": evidence_root / "stdout.jsonl", "raw/pi/stderr.log": evidence_root / "stderr.log", "raw/pi/prompt-transport.bin": evidence_root / "prompt-transport.bin", "raw/pi/invocation.json": evidence_root / "invocation.json", "raw/pi/result.json": evidence_root / "result.json"}
        for label, root in (("home", context.paths.home), ("config", context.paths.xdg_config_home), ("cache", context.paths.xdg_cache_home), ("data", context.paths.xdg_data_home), ("state", context.paths.xdg_state_home)):
            for source in sorted(root.rglob("*")):
                if source.is_file() and not source.is_symlink(): records[f"run/pi/{label}/{source.relative_to(root).as_posix()}"] = source
        return tuple(sorted(records.items()))


def _base_environment(home: Path, config: Path, cache: Path, data: Path, state: Path) -> dict[str, str]:
    return {"HOME": str(home), "XDG_CONFIG_HOME": str(config), "XDG_CACHE_HOME": str(cache), "XDG_DATA_HOME": str(data), "XDG_STATE_HOME": str(state),
            "PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TZ": "UTC", "TERM": "dumb", "CI": "true", "NO_COLOR": "1"}


def _node_modules_tree_digest(root: Path) -> tuple[str, int]:
    if not root.is_dir(): raise PiError(f"Pi node_modules is missing: {root}")
    digest = hashlib.sha256(); records = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink(): record = b"L\0" + relative + b"\0" + os.readlink(path).encode("utf-8") + b"\n"
        elif path.is_file(): record = b"F\0" + relative + b"\0" + _sha256_file(path).encode("ascii") + b"\n"
        else: continue
        digest.update(record); records += 1
    return digest.hexdigest(), records


def _verify_file(path: Path, expected: str, size: int | None = None) -> None:
    if not path.is_file() or path.is_symlink(): raise PiError(f"pinned file is missing or invalid: {path}")
    if size is not None and path.stat().st_size != size: raise PiError(f"pinned file size differs: {path}")
    if _sha256_file(path) != expected: raise PiError(f"pinned file SHA256 differs: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def _message_texts(message: dict[str, object]) -> list[str]:
    content = message.get("content")
    if isinstance(content, str): return [content]
    if not isinstance(content, list): return []
    return [item["text"] for item in content if isinstance(item, dict) and isinstance(item.get("text"), str)]


def _native_output_truncated(native: dict[str, object]) -> bool:
    message = native.get("message")
    return native.get("type") == "message_end" and isinstance(message, dict) and message.get("stopReason") in {"length", "max_tokens", "max_output_tokens"}


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
