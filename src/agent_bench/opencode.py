"""Pinned OpenCode profile materialization and process-backed adapter."""

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
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from agent_bench.capture import CaptureCapabilities
from agent_bench.harness import HarnessExecutionResult, HarnessRunContext
from agent_bench.models import Identifier, PersistedModel, Sha256
from agent_bench.opencode_events import normalize_opencode_events

DEFAULT_OPENCODE_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "environment"
    / "harnesses"
    / "opencode-default-v1"
    / "profile.yaml"
)
BENCHMARK_OPENCODE_EXECUTABLE = (
    Path(__file__).resolve().parents[2]
    / "toolchains"
    / "opencode"
    / "1.18.25"
    / "bin"
    / "opencode"
)


class OpenCodeError(RuntimeError):
    """Raised when the pinned OpenCode adapter cannot run safely."""


class OpenCodeExecutable(PersistedModel):
    path: Path
    size_bytes: int = Field(ge=1)
    sha256: Sha256
    version: str = Field(min_length=1)
    runtime_identity: str = Field(min_length=1)


class OpenCodeInvocationPolicy(PersistedModel):
    pure: Literal[True] = True
    format: Literal["json"] = "json"
    thinking: Literal[True] = True
    auto_approve: Literal[True] = True
    prompt_delivery: Literal["stdin_exact_utf8"] = "stdin_exact_utf8"


class OpenCodeProfile(PersistedModel):
    profile_id: Literal["opencode-default-v1"] = "opencode-default-v1"
    profile_version: Literal["1.0.2"] = "1.0.2"
    profile_path: Path
    config_file: Path
    config_sha256: Sha256
    proxy_base_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/v1$")
    provider_id: Identifier
    model_id: Identifier
    executable: OpenCodeExecutable
    invocation: OpenCodeInvocationPolicy
    deviations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_paths(self) -> OpenCodeProfile:
        if not self.profile_path.is_absolute() or not self.config_file.is_absolute():
            raise ValueError("OpenCode profile paths must be absolute")
        if self.executable.path != BENCHMARK_OPENCODE_EXECUTABLE:
            raise ValueError("OpenCode profile must use the benchmark-managed executable")
        return self

    @property
    def model_name(self) -> str:
        return f"{self.provider_id}/{self.model_id}"


def load_opencode_profile(
    path: Path = DEFAULT_OPENCODE_PROFILE_PATH,
) -> OpenCodeProfile:
    """Load and validate the checked-in controlled OpenCode profile."""
    profile_path = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OpenCodeError(f"cannot load OpenCode profile {profile_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise OpenCodeError("OpenCode profile must be a YAML mapping")
    config_value = raw.get("config_file")
    if not isinstance(config_value, str):
        raise OpenCodeError("OpenCode profile config_file must be a string")
    raw["profile_path"] = profile_path
    raw["config_file"] = (profile_path.parent / config_value).resolve()
    executable = raw.get("executable")
    if isinstance(executable, dict) and isinstance(executable.get("path"), str):
        executable["path"] = _resolve_benchmark_path(executable["path"], profile_path.parent)
    try:
        profile = OpenCodeProfile.model_validate(raw)
    except ValueError as exc:
        raise OpenCodeError(f"invalid OpenCode profile: {exc}") from exc
    _verify_file(profile.config_file, profile.config_sha256)
    return profile


def _resolve_benchmark_path(value: str, base: Path) -> Path:
    """Resolve a checked-in layout path without retaining an old clone root."""
    path = Path(value)
    if not path.is_absolute():
        return (base / path).resolve()
    if "toolchains" in path.parts:
        return Path(__file__).resolve().parents[2] / "toolchains" / Path(*path.parts[path.parts.index("toolchains") + 1:])
    return path


def inspect_opencode_executable(path: Path) -> OpenCodeExecutable:
    """Inspect a binary using an isolated temporary HOME/XDG environment."""
    executable = path.expanduser().resolve()
    if not executable.is_file() or executable.is_symlink() or not os.access(executable, os.X_OK):
        raise OpenCodeError(f"OpenCode executable is missing or invalid: {executable}")
    with tempfile.TemporaryDirectory(prefix="agent-bench-opencode-inspect-") as root_text:
        root = Path(root_text)
        environment = _base_environment(
            home=root / "home",
            config=root / "config",
            cache=root / "cache",
            data=root / "data",
            state=root / "state",
        )
        for value in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME"):
            Path(environment[value]).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [str(executable), "--version"],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=30,
        )
    if result.returncode != 0 or not result.stdout.strip():
        raise OpenCodeError(
            f"OpenCode version inspection failed with code {result.returncode}: {result.stderr.strip()}"
        )
    return OpenCodeExecutable(
        path=executable,
        size_bytes=executable.stat().st_size,
        sha256=_sha256_file(executable),
        version=result.stdout.strip(),
        runtime_identity=_runtime_identity(executable),
    )


def verify_opencode_toolchain(profile: OpenCodeProfile) -> OpenCodeExecutable:
    """Fail before a task when the sole benchmark-managed binary drifts or vanishes."""
    observed = inspect_opencode_executable(profile.executable.path)
    if observed != profile.executable:
        raise OpenCodeError("benchmark-managed OpenCode identity differs from the pinned profile")
    return observed


def materialize_opencode_profile(
    profile: OpenCodeProfile,
    context: HarnessRunContext,
) -> Path:
    """Copy the immutable source config into the fresh run config directory."""
    _verify_file(profile.config_file, profile.config_sha256)
    destination_dir = context.paths.xdg_config_home / "opencode"
    if destination_dir.exists():
        raise OpenCodeError(f"OpenCode run config already exists: {destination_dir}")
    destination_dir.mkdir(parents=True)
    destination = destination_dir / "opencode.json"
    shutil.copyfile(profile.config_file, destination)
    endpoint = context.proxy_endpoint or profile.proxy_base_url
    if endpoint != profile.proxy_base_url:
        config = json.loads(destination.read_text(encoding="utf-8"))
        config["provider"][profile.provider_id]["options"]["baseURL"] = endpoint
        destination.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    _verify_file(profile.config_file, profile.config_sha256)
    return destination


def opencode_environment(
    context: HarnessRunContext,
    config_path: Path,
) -> dict[str, str]:
    """Build the complete allowlisted environment for the OpenCode child."""
    environment = _base_environment(
        home=context.paths.home,
        config=context.paths.xdg_config_home,
        cache=context.paths.xdg_cache_home,
        data=context.paths.xdg_data_home,
        state=context.paths.xdg_state_home,
    )
    environment.update(
        {
            "OPENCODE_CONFIG": str(config_path),
            "OPENCODE_DISABLE_AUTOUPDATE": "true",
            "AGENT_BENCH_HARNESS_STATE": str(context.paths.harness_state),
        }
    )
    return environment


def build_opencode_command(
    profile: OpenCodeProfile,
    context: HarnessRunContext,
) -> tuple[str, ...]:
    """Build argv; the exact prompt is supplied separately on standard input."""
    return (
        str(profile.executable.path),
        "--pure",
        "run",
        "--format",
        "json",
        "--thinking",
        "--auto",
        "--model",
        profile.model_name,
        "--dir",
        str(context.paths.workspace),
    )


def opencode_capture_capabilities() -> CaptureCapabilities:
    """Declare only observations provided by proxy plus OpenCode 1.18.25 JSON."""
    return CaptureCapabilities(
        capability_id="opencode-1.18.25-proxy-v1",
        backend_id="llamacpp-qwen38-agent-bench-v1",
        harness_id="opencode",
        raw_request_payload="proxy_exact",
        raw_response_payload="proxy_exact",
        request_generation_parameters="proxy_exact",
        input_token_usage="api_exact",
        output_token_usage="api_exact",
        reasoning_content="proxy_exact",
        reasoning_token_count="api_exact",
        context_token_count="api_exact",
        finish_reason="proxy_exact",
        tool_calls="harness_exact",
        tool_results="harness_exact",
        compaction_events="unavailable",
        session_identity="harness_exact",
        serialized_prompt_history_validation="unavailable",
        empty_historical_think_block_detection="proxy_exact",
        notes=(
            "LLM exchanges are authoritative at the Agent Bench proxy boundary.",
            "Completed/error OpenCode JSON tool parts expose native input, result, and start/end milliseconds.",
            "OpenCode run JSON does not expose complete compaction events.",
            "Request messages can be checked for empty historical think blocks; llama.cpp's rendered Jinja prompt remains unavailable.",
        ),
    )


class OpenCodeAdapter:
    """Run one fresh OpenCode session and capture its official JSON event stream."""

    adapter_id = "opencode"
    adapter_version = "1.0.2"
    capture_capabilities = opencode_capture_capabilities()

    def __init__(
        self,
        profile: OpenCodeProfile | None = None,
        *,
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        run_command: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        verify_executable: bool = True,
    ) -> None:
        self.profile = profile or load_opencode_profile()
        self._popen = popen
        self._run_command = run_command
        self._verify_executable = verify_executable

    def normalize_events(self, raw_path: Path, normalized_path: Path) -> None:
        normalize_opencode_events(raw_path, normalized_path)

    def run(self, context: HarnessRunContext) -> HarnessExecutionResult:
        if context.proxy_endpoint is None:
            raise OpenCodeError("OpenCode requires an Agent Bench proxy endpoint")
        if self._verify_executable:
            verify_opencode_toolchain(self.profile)
        config_path = materialize_opencode_profile(self.profile, context)
        environment = opencode_environment(context, config_path)
        argv = build_opencode_command(self.profile, context)
        prompt_bytes = context.prompt_content.encode("utf-8")
        prompt_sha = hashlib.sha256(prompt_bytes).hexdigest()
        evidence_root = context.paths.harness_state / "opencode"
        evidence_root.mkdir()
        stdin_prompt_path = evidence_root / "stdin-prompt.bin"
        stdin_prompt_path.write_bytes(prompt_bytes)
        stdout_path = evidence_root / "stdout.jsonl"
        stderr_path = evidence_root / "stderr.log"
        _write_json(
            evidence_root / "invocation.json",
            {
                "schema_version": "1.0.0",
                "argv": list(argv),
                "stdin_policy": "exact_prompt_utf8_then_eof",
                "prompt_sha256": prompt_sha,
                "prompt_byte_length": len(prompt_bytes),
                "working_directory": str(context.paths.workspace),
                "environment": environment,
                "profile_digest": self.profile.definition_digest,
                "run_seed": context.run_seed,
            },
        )
        context.events.emit(
            source="harness",
            event_type="opencode_start",
            payload={
                "profile_id": self.profile.profile_id,
                "model": self.profile.model_name,
                "proxy_endpoint": context.proxy_endpoint,
                "prompt_sha256": prompt_sha,
                "prompt_delivery": "stdin_exact_utf8_then_eof",
                "fresh_session": True,
                "continued_session": False,
                "environment": environment,
                "argv": list(argv),
            },
        )
        process = self._popen(
            list(argv),
            cwd=context.paths.workspace,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=True,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise OpenCodeError("OpenCode process pipes were not created")
        try:
            process.stdin.write(prompt_bytes)
            process.stdin.close()
        except BaseException:
            _terminate_owned_process(process)
            raise
        session_ids: set[str] = set()
        output_truncated = threading.Event()
        reader_errors: list[BaseException] = []
        stdout_thread = threading.Thread(
            target=self._capture_stdout,
            args=(process.stdout, stdout_path, context, session_ids, output_truncated, reader_errors),
            name="agent-bench-opencode-stdout",
        )
        stderr_thread = threading.Thread(
            target=self._capture_stderr,
            args=(process.stderr, stderr_path, reader_errors),
            name="agent-bench-opencode-stderr",
        )
        stdout_thread.start()
        stderr_thread.start()
        cancelled = False
        while process.poll() is None:
            if context.cancellation.wait(0.05):
                cancelled = True
                _terminate_owned_process(process)
                break
        return_code = process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        if stdout_thread.is_alive() or stderr_thread.is_alive():
            raise OpenCodeError("OpenCode output reader did not finish")
        if reader_errors:
            raise OpenCodeError(f"OpenCode output capture failed: {reader_errors[0]}")
        session_id = next(iter(session_ids)) if len(session_ids) == 1 else None
        export_path = evidence_root / "session-export.json"
        prompt_validated = self._export_session(
            session_id, export_path, environment, context, prompt_sha
        )
        if prompt_validated == "mismatch":
            context.events.emit(
                source="harness",
                event_type="harness_error",
                payload={
                    "error_type": "invalid_harness_output",
                    "message": "OpenCode session export did not contain the exact prompt bytes",
                    "expected_prompt_sha256": prompt_sha,
                },
            )
        if len(session_ids) > 1:
            context.events.emit(
                source="harness",
                event_type="harness_error",
                payload={
                    "error_type": "multiple_opencode_sessions",
                    "session_ids": sorted(session_ids),
                },
            )
        if return_code != 0 and not cancelled:
            context.events.emit(
                source="harness",
                event_type="harness_error",
                payload={
                    "error_type": "opencode_nonzero_exit",
                    "exit_code": return_code,
                },
            )
        result_record = {
            "schema_version": "1.0.0",
            "session_id": session_id,
            "exit_code": return_code,
            "termination_signal": -return_code if return_code < 0 else None,
            "cancelled_by_runner": cancelled,
            "output_truncated": output_truncated.is_set(),
            "prompt_export_validation": prompt_validated,
        }
        _write_json(evidence_root / "result.json", result_record)
        evidence = self._evidence_files(context, evidence_root, export_path)
        return HarnessExecutionResult(
            completed_normally=return_code == 0 and not cancelled,
            output_truncated=output_truncated.is_set(),
            exit_code=return_code,
            termination_signal=-return_code if return_code < 0 else None,
            session_id=session_id,
            evidence_files=evidence,
        )

    def _capture_stdout(
        self,
        source: object,
        destination: Path,
        context: HarnessRunContext,
        session_ids: set[str],
        output_truncated: threading.Event,
        errors: list[BaseException],
    ) -> None:
        try:
            with destination.open("xb") as output:
                index = 0
                while True:
                    line = source.readline()  # type: ignore[attr-defined]
                    if not line:
                        break
                    output.write(line)
                    output.flush()
                    index += 1
                    payload: dict[str, object] = {
                        "stream_index": index,
                        "line_base64": base64.b64encode(line).decode("ascii"),
                        "line_sha256": hashlib.sha256(line).hexdigest(),
                    }
                    try:
                        native = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        context.events.emit(
                            source="stdout",
                            event_type="opencode_unparsed_stdout",
                            payload=payload,
                        )
                        continue
                    payload["native_event"] = native
                    if isinstance(native, dict):
                        session_id = native.get("sessionID")
                        if isinstance(session_id, str) and session_id:
                            session_ids.add(session_id)
                        if _native_output_truncated(native):
                            output_truncated.set()
                    context.events.emit(
                        source="harness",
                        event_type="opencode_event",
                        payload=payload,
                    )
                os.fsync(output.fileno())
        except BaseException as exc:
            errors.append(exc)

    def _capture_stderr(
        self,
        source: object,
        destination: Path,
        errors: list[BaseException],
    ) -> None:
        try:
            with destination.open("xb") as output:
                while True:
                    chunk = source.read(64 * 1024)  # type: ignore[attr-defined]
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except BaseException as exc:
            errors.append(exc)

    def _export_session(
        self,
        session_id: str | None,
        destination: Path,
        environment: Mapping[str, str],
        context: HarnessRunContext,
        prompt_sha: str,
    ) -> str:
        if session_id is None:
            return "unavailable_no_unique_session_id"
        try:
            result = self._run_command(
                [str(self.profile.executable.path), "--pure", "export", session_id],
                cwd=context.paths.workspace,
                env=dict(environment),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
        except Exception as exc:
            context.events.emit(
                source="harness",
                event_type="opencode_session_export_error",
                payload={"error_type": type(exc).__name__, "message": str(exc)},
            )
            return "unavailable_export_failed"
        destination.write_bytes(result.stdout)
        if result.returncode != 0:
            return "unavailable_export_failed"
        try:
            exported = json.loads(result.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "unavailable_invalid_export"
        matches = _prompt_texts(exported)
        exact = any(
            hashlib.sha256(text.encode("utf-8")).hexdigest() == prompt_sha
            for text in matches
        )
        context.events.emit(
            source="harness",
            event_type="opencode_prompt_validation",
            payload={
                "session_id": session_id,
                "expected_prompt_sha256": prompt_sha,
                "exact_prompt_found": exact,
                "candidate_user_text_count": len(matches),
                "method": "opencode_session_export_exact_utf8",
            },
        )
        return "exact_match" if exact else "mismatch"

    def _evidence_files(
        self,
        context: HarnessRunContext,
        evidence_root: Path,
        export_path: Path,
    ) -> tuple[tuple[str, Path], ...]:
        records: dict[str, Path] = {
            "raw/opencode/stdout.jsonl": evidence_root / "stdout.jsonl",
            "raw/opencode/stderr.log": evidence_root / "stderr.log",
            "raw/opencode/stdin-prompt.bin": evidence_root / "stdin-prompt.bin",
        }
        if export_path.is_file():
            records["raw/opencode/session-export.json"] = export_path
        roots = (
            ("home", context.paths.home),
            ("config", context.paths.xdg_config_home),
            ("cache", context.paths.xdg_cache_home),
            ("data", context.paths.xdg_data_home),
            ("state", context.paths.xdg_state_home),
        )
        for label, root in roots:
            for source in sorted(root.rglob("*")):
                if source.is_file() and not source.is_symlink():
                    relative = source.relative_to(root).as_posix()
                    records[f"run/opencode/{label}/{relative}"] = source
        return tuple(sorted(records.items()))


def _base_environment(
    *, home: Path, config: Path, cache: Path, data: Path, state: Path
) -> dict[str, str]:
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(config),
        "XDG_CACHE_HOME": str(cache),
        "XDG_DATA_HOME": str(data),
        "XDG_STATE_HOME": str(state),
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "TERM": "dumb",
        "CI": "true",
        "NO_COLOR": "1",
    }


def _terminate_owned_process(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5)


def _native_output_truncated(native: dict[str, object]) -> bool:
    if native.get("type") == "step_finish":
        part = native.get("part")
        return isinstance(part, dict) and part.get("reason") in {
            "length",
            "max_tokens",
            "max_output_tokens",
        }
    if native.get("type") == "error":
        error = native.get("error")
        return isinstance(error, dict) and error.get("name") == "MessageOutputLengthError"
    return False


def _prompt_texts(value: object) -> tuple[str, ...]:
    results: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, dict):
            info = item.get("info")
            if isinstance(info, dict) and info.get("role") == "user":
                parts = item.get("parts")
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text = part.get("text")
                            if isinstance(text, str):
                                results.append(text)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(results)


def _runtime_identity(executable: Path) -> str:
    root = executable.parent.parent
    versions: list[str] = ["self-contained Bun ELF"]
    for package in (
        root / "node_modules/@opencode-ai/sdk/package.json",
        root / "node_modules/@opencode-ai/plugin/package.json",
    ):
        try:
            data = json.loads(package.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name, version = data.get("name"), data.get("version")
        if isinstance(name, str) and isinstance(version, str):
            versions.append(f"{name} {version}")
    return "; ".join(versions)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _verify_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise OpenCodeError(f"required OpenCode profile file is missing: {path}")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise OpenCodeError(
            f"OpenCode profile file SHA256 mismatch: expected {expected_sha256}, observed {observed}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
