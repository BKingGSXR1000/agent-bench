"""Fixed llama.cpp backend profile, preflight, and owned process lifecycle."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, JsonValue, field_validator, model_validator

from agent_bench.models import Identifier, JsonMapping, PersistedModel, Sha256
from agent_bench.reasoning_tokenizer import LlamaTokenizeCounter, ReasoningTokenizerError

DEFAULT_BACKEND_PROFILE_PATH = (
    Path(__file__).resolve().parents[2] / "environment" / "backend-v1.yaml"
)

FailureClass = Literal[
    "precondition_failed",
    "backend_start_failed",
    "backend_readiness_failed",
    "backend_identity_mismatch",
    "model_hash_mismatch",
    "template_hash_mismatch",
    "benchmark_port_in_use",
    "conflicting_gpu_process",
    "preservation_failed",
]


class BackendConfigurationError(ValueError):
    """Raised when the fixed backend profile cannot be loaded safely."""


class BackendLifecycleError(RuntimeError):
    """Raised for startup, readiness, or owned-process lifecycle failures."""


class BackendStartFailed(BackendLifecycleError):
    """Raised when an owned backend exits before it ever becomes ready."""


class BackendReadinessFailed(BackendLifecycleError):
    """Raised when an owned backend remains alive but misses its readiness deadline."""


class PinnedFile(PersistedModel):
    """Expected identity of a deployment file."""

    path: Path
    size_bytes: int = Field(ge=0)
    sha256: Sha256


class SamplingBaseline(PersistedModel):
    """Fixed backend sampling defaults, distinct from observed requests."""

    temperature: float = 1.0
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0


class ServerConfiguration(PersistedModel):
    """Fixed server-start behavior for benchmark v1."""

    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=18080, ge=1, le=65535)
    proxy_port: int = Field(default=18081, ge=1, le=65535)
    context_size: Literal[107520] = 107520
    parallel: Literal[1] = 1
    batch_size: Literal[128] = 128
    ubatch_size: Literal[128] = 128
    split_mode: Literal["none"] = "none"
    main_gpu: Literal[0] = 0
    gpu_layers: Literal["all"] = "all"
    fit: Literal["off"] = "off"
    flash_attention: Literal["on"] = "on"
    cache_type_k: Literal["q8_0"] = "q8_0"
    cache_type_v: Literal["q8_0"] = "q8_0"
    context_shift: Literal[False] = False
    speculative_type: Literal["none"] = "none"
    prompt_cache: Literal[True] = True
    continuous_batching: Literal[True] = True
    jinja: Literal[True] = True
    reasoning: Literal["on"] = "on"
    reasoning_format: Literal["deepseek"] = "deepseek"
    built_in_warmup: Literal[True] = True
    synthetic_warmup: Literal[False] = False
    metrics_endpoint: Literal[True] = True
    slots_endpoint: Literal[True] = True

    @model_validator(mode="after")
    def require_distinct_ports(self) -> ServerConfiguration:
        if self.port == self.proxy_port:
            raise ValueError("backend and capture proxy ports must differ")
        return self


class GpuPreconditionPolicy(PersistedModel):
    """Conservative policy for exclusive benchmark use of the RTX 3090."""

    expected_uuid: str = Field(min_length=1)
    expected_name: Literal["NVIDIA GeForce RTX 3090"] = "NVIDIA GeForce RTX 3090"
    minimum_total_vram_mib: Literal[24576] = 24576
    maximum_exempt_desktop_process_vram_mib: Literal[512] = 512
    exempt_desktop_process_markers: tuple[str, ...] = (
        "Xorg",
        "Xwayland",
        "gnome-shell",
        "gnome-remote-desktop-daemon",
        "kwin",
        "plasmashell",
        "code --type=gpu-process",
    )


class BackendProfile(PersistedModel):
    """One fixed, versioned Agent Bench v1 backend deployment."""

    profile_id: Identifier
    model_name: Literal["Qwen 3.8 27B"] = "Qwen 3.8 27B"
    quantization: Literal["UD-Q4_K_XL"] = "UD-Q4_K_XL"
    model: PinnedFile
    chat_template: PinnedFile
    executable: PinnedFile
    # This is deliberately an external deployment dependency.  The benchmark
    # repository records its identity, but never contains the binary or GGUF.
    reasoning_tokenizer: PinnedFile | None = None
    llama_cpp_repository: Path
    llama_cpp_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    version_output_contains: tuple[str, ...] = Field(min_length=1)
    local_libraries: tuple[PinnedFile, ...]
    gpu: GpuPreconditionPolicy
    server: ServerConfiguration = Field(default_factory=ServerConfiguration)
    sampling: SamplingBaseline = Field(default_factory=SamplingBaseline)
    restart_policy: Literal["per_run"] = "per_run"
    readiness_endpoint: Literal["/health"] = "/health"
    readiness_timeout_seconds: float = Field(default=900.0, gt=0)
    shutdown_grace_seconds: float = Field(default=10.0, gt=0)


def reasoning_tokenizer_from_profile(profile: BackendProfile) -> LlamaTokenizeCounter | None:
    """Build the configured exact counter after preflight has pinned its files.

    ``None`` remains valid for legacy/general profiles.  Functional profiles
    opt in through ``reasoning_tokenizer`` and therefore never silently fall
    back to character counts or token estimates.
    """
    configured = profile.reasoning_tokenizer
    if configured is None:
        return None
    try:
        return LlamaTokenizeCounter(
            executable=configured.path,
            model=profile.model.path,
            model_sha256=profile.model.sha256,
            llama_cpp_commit=profile.llama_cpp_commit,
        )
    except ReasoningTokenizerError as exc:
        raise BackendLifecycleError(
            f"configured reasoning tokenizer is unavailable: {exc}"
        ) from exc


class BackendRunPaths(PersistedModel):
    """Fresh per-run HOME/XDG directories used by the backend."""

    home: Path
    xdg_config_home: Path
    xdg_cache_home: Path
    xdg_data_home: Path
    xdg_state_home: Path

    @field_validator(
        "home",
        "xdg_config_home",
        "xdg_cache_home",
        "xdg_data_home",
        "xdg_state_home",
    )
    @classmethod
    def require_absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("backend isolation paths must be absolute")
        return value


class ResolvedBackendInvocation(PersistedModel):
    """Authoritative process creation inputs for one backend instance."""

    profile_id: Identifier
    run_seed: int | None = Field(default=None, ge=0)
    executable: Path
    argv: tuple[str, ...]
    working_directory: Path
    environment: dict[str, str]
    stdin_policy: Literal["closed"] = "closed"
    stdout_artifact: Literal["failure/stdout.log", "raw/backend/stdout.log"]
    stderr_artifact: Literal["failure/stderr.log", "raw/backend/stderr.log"]


class GpuObservation(PersistedModel):
    index: int = Field(ge=0)
    uuid: str = Field(min_length=1)
    name: str = Field(min_length=1)
    memory_total_mib: int = Field(ge=0)
    memory_used_mib: int = Field(ge=0)
    memory_free_mib: int | None = Field(default=None, ge=0)
    utilization_percent: int | None = Field(default=None, ge=0)
    temperature_celsius: int | None = None


class GpuProcessObservation(PersistedModel):
    gpu_uuid: str = Field(min_length=1)
    pid: int = Field(ge=1)
    process_name: str = Field(min_length=1)
    used_gpu_memory_mib: int | None = Field(default=None, ge=0)


class PreflightCheck(PersistedModel):
    """One deterministic preflight decision and its safe evidence."""

    check_id: Identifier
    passed: bool
    failure_class: FailureClass | None = None
    message: str = Field(min_length=1)
    evidence: JsonMapping = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_failure_class(self) -> PreflightCheck:
        if self.passed == (self.failure_class is not None):
            raise ValueError("only failed checks define a failure_class")
        return self


class BackendPreflightReport(PersistedModel):
    """Immutable result of checking the fixed backend before process startup."""

    profile_id: Identifier
    passed: bool
    primary_failure_class: FailureClass | None = None
    checks: tuple[PreflightCheck, ...]
    gpu_observations: tuple[GpuObservation, ...] = ()
    gpu_processes: tuple[GpuProcessObservation, ...] = ()
    gpu_observed_at_utc: datetime | None = None
    gpu_observation_source: Literal[
        "live_nvidia_smi", "injected_test_fixture", "unavailable"
    ] = "unavailable"
    gpu_observation_commands: tuple[tuple[str, ...], ...] = ()

    @field_validator("gpu_observed_at_utc")
    @classmethod
    def require_gpu_observation_utc(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("GPU observation timestamp must include an offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_outcome(self) -> BackendPreflightReport:
        failures = [check for check in self.checks if not check.passed]
        if self.passed != (not failures):
            raise ValueError("preflight outcome disagrees with check results")
        if self.passed != (self.primary_failure_class is None):
            raise ValueError("only failed preflight defines a primary failure class")
        return self


class BackendEndpointObservation(PersistedModel):
    """Exact diagnostic response from a pinned llama-server endpoint."""

    endpoint: Literal["/metrics", "/slots"]
    observed_at_utc: datetime
    http_status: int = Field(ge=100, le=599)
    content_type: str
    body_base64: str
    body_sha256: Sha256
    parsed_body: JsonValue | None = None
    provenance: Literal["llama_server_endpoint_exact"] = (
        "llama_server_endpoint_exact"
    )

    @field_validator("observed_at_utc")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("backend observation timestamp must include an offset")
        return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[tuple[str, ...], Mapping[str, str] | None, Path | None], CommandResult]
FileHasher = Callable[[Path], str]
GpuCollector = Callable[[], tuple[tuple[GpuObservation, ...], tuple[GpuProcessObservation, ...]]]
PortChecker = Callable[[str, int], bool]


def load_backend_profile(path: Path = DEFAULT_BACKEND_PROFILE_PATH) -> BackendProfile:
    """Load the checked-in fixed backend profile."""
    configured = path.expanduser().resolve()
    try:
        raw = yaml.safe_load(configured.read_text(encoding="utf-8"))
        return BackendProfile.model_validate(raw)
    except Exception as exc:
        raise BackendConfigurationError(
            f"cannot load backend profile {configured}: {exc}"
        ) from exc


def seed_for_repetition(repetition_index: int) -> int:
    """Map a one-based repetition to the fixed comparable generation seed."""
    if repetition_index < 1:
        raise ValueError("repetition_index must be one-based")
    return 1000 + repetition_index


def resolve_backend_invocation(
    profile: BackendProfile,
    paths: BackendRunPaths,
    *,
    run_seed: int,
    failure_logs: bool = False,
) -> ResolvedBackendInvocation:
    """Produce the exact argv/environment passed to the pinned llama-server."""
    server = profile.server
    argv = (
        str(profile.executable.path),
        "--model", str(profile.model.path),
        "--ctx-size", str(server.context_size),
        "--parallel", str(server.parallel),
        "--batch-size", str(server.batch_size),
        "--ubatch-size", str(server.ubatch_size),
        "--split-mode", server.split_mode,
        "--main-gpu", str(server.main_gpu),
        "--n-gpu-layers", server.gpu_layers,
        "--fit", server.fit,
        "--flash-attn", server.flash_attention,
        "--cache-type-k", server.cache_type_k,
        "--cache-type-v", server.cache_type_v,
        "--no-context-shift",
        "--spec-type", server.speculative_type,
        "--cache-prompt",
        "--cont-batching",
        "--jinja",
        "--chat-template-file", str(profile.chat_template.path),
        "--reasoning", server.reasoning,
        "--reasoning-format", server.reasoning_format,
        "--temp", _number(profile.sampling.temperature),
        "--top-k", str(profile.sampling.top_k),
        "--top-p", _number(profile.sampling.top_p),
        "--min-p", _number(profile.sampling.min_p),
        "--seed", str(run_seed),
        "--warmup",
        "--host", server.host,
        "--port", str(server.port),
        "--metrics",
        "--slots",
        "--no-webui",
    )
    log_prefix = "failure" if failure_logs else "raw/backend"
    return ResolvedBackendInvocation(
        profile_id=profile.profile_id,
        run_seed=run_seed,
        executable=profile.executable.path,
        argv=argv,
        working_directory=profile.llama_cpp_repository,
        environment=backend_environment(profile, paths),
        stdout_artifact=f"{log_prefix}/stdout.log",
        stderr_artifact=f"{log_prefix}/stderr.log",
    )


def backend_environment(
    profile: BackendProfile,
    paths: BackendRunPaths,
) -> dict[str, str]:
    """Build the allowlisted backend environment without login-shell inheritance."""
    return {
        "HOME": str(paths.home),
        "XDG_CONFIG_HOME": str(paths.xdg_config_home),
        "XDG_CACHE_HOME": str(paths.xdg_cache_home),
        "XDG_DATA_HOME": str(paths.xdg_data_home),
        "XDG_STATE_HOME": str(paths.xdg_state_home),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": profile.gpu.expected_uuid,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }


def preflight_backend(
    profile: BackendProfile,
    paths: BackendRunPaths,
    *,
    run_seed: int,
    hash_file: FileHasher | None = None,
    run_command: CommandRunner | None = None,
    collect_gpus: GpuCollector | None = None,
    port_is_free: PortChecker | None = None,
) -> BackendPreflightReport:
    """Run deterministic checks without starting or stopping llama-server."""
    hasher = hash_file or sha256_file
    command = run_command or _run_command
    gpu_collector = collect_gpus or collect_nvidia_observations
    port_checker = port_is_free or is_port_free
    checks: list[PreflightCheck] = []

    # Port availability is deliberately the first check.  The model hash is
    # expensive (about 18 GiB) and a live foreign listener is an immediate,
    # deterministic reason not to do that work.  A clean port does *not*
    # relax any later identity/GPU checks: a run cannot proceed until every
    # check below has passed.
    port_ok = port_checker(profile.server.host, profile.server.port)
    _add_check(
        checks,
        "benchmark-port",
        port_ok,
        "benchmark_port_in_use",
        "benchmark backend port has no active listener"
        if port_ok else "benchmark backend port has an active listener",
        {"host": profile.server.host, "port": profile.server.port,
         "probe": "active_listener_connect"},
    )
    proxy_port_ok = port_checker(profile.server.host, profile.server.proxy_port)
    _add_check(
        checks,
        "capture-proxy-port",
        proxy_port_ok,
        "benchmark_port_in_use",
        "benchmark capture-proxy port has no active listener"
        if proxy_port_ok else "benchmark capture-proxy port has an active listener",
        {"host": profile.server.host, "port": profile.server.proxy_port,
         "probe": "active_listener_connect"},
    )
    if not port_ok or not proxy_port_ok:
        failures = [check for check in checks if not check.passed]
        return BackendPreflightReport(
            profile_id=profile.profile_id,
            passed=False,
            primary_failure_class=failures[0].failure_class,
            checks=tuple(checks),
        )

    _check_pinned_file(
        checks, "model-file", profile.model, "model_hash_mismatch", hasher
    )
    _check_pinned_file(
        checks, "chat-template", profile.chat_template, "template_hash_mismatch", hasher
    )
    _check_pinned_file(
        checks, "llama-server", profile.executable, "backend_identity_mismatch", hasher,
        require_executable=True,
    )
    if profile.reasoning_tokenizer is not None:
        _check_pinned_file(
            checks, "llama-tokenize", profile.reasoning_tokenizer,
            "backend_identity_mismatch", hasher, require_executable=True,
        )
    for index, library in enumerate(profile.local_libraries, start=1):
        _check_pinned_file(
            checks,
            f"backend-library-{index}",
            library,
            "backend_identity_mismatch",
            hasher,
        )

    invocation = resolve_backend_invocation(profile, paths, run_seed=run_seed)
    template_tokens = [
        invocation.argv[index + 1]
        for index, token in enumerate(invocation.argv[:-1])
        if token == "--chat-template-file"
    ]
    expected_template = str(profile.chat_template.path)
    _add_check(
        checks,
        "resolved-template",
        template_tokens == [expected_template],
        "precondition_failed",
        "resolved argv references exactly the pinned chat template"
        if template_tokens == [expected_template]
        else "resolved argv does not reference exactly the pinned chat template",
        {"observed": template_tokens, "expected": expected_template},
    )

    version = command((str(profile.executable.path), "--version"), invocation.environment, profile.llama_cpp_repository)
    version_text = version.stdout + version.stderr
    version_ok = version.returncode == 0 and all(
        marker in version_text for marker in profile.version_output_contains
    )
    _add_check(
        checks,
        "backend-version",
        version_ok,
        "backend_identity_mismatch",
        "llama-server version/build identity matches"
        if version_ok else "llama-server version/build identity mismatch",
        {
            "returncode": version.returncode,
            "stdout": version.stdout,
            "stderr": version.stderr,
        },
    )

    commit = command(
        ("git", "-C", str(profile.llama_cpp_repository), "rev-parse", "HEAD"),
        None,
        None,
    )
    observed_commit = commit.stdout.strip()
    commit_ok = commit.returncode == 0 and observed_commit == profile.llama_cpp_commit
    _add_check(
        checks,
        "backend-source-commit",
        commit_ok,
        "backend_identity_mismatch",
        "llama.cpp source commit matches" if commit_ok else "llama.cpp source commit mismatch",
        {"observed": observed_commit, "expected": profile.llama_cpp_commit},
    )
    dirty = command(
        ("git", "-C", str(profile.llama_cpp_repository), "status", "--porcelain"),
        None,
        None,
    )
    source_clean = dirty.returncode == 0 and not dirty.stdout.strip()
    _add_check(
        checks,
        "backend-source-clean",
        source_clean,
        "backend_identity_mismatch",
        "llama.cpp source working tree is clean"
        if source_clean else "llama.cpp source working tree has local modifications",
        {"status_porcelain": dirty.stdout, "returncode": dirty.returncode},
    )

    linked = command(
        ("ldd", str(profile.executable.path)),
        invocation.environment,
        profile.llama_cpp_repository,
    )
    observed_local_libraries = _local_ldd_paths(
        linked.stdout, profile.llama_cpp_repository
    )
    expected_local_libraries = {
        str(library.path.resolve()) for library in profile.local_libraries
    }
    linkage_ok = (
        linked.returncode == 0
        and observed_local_libraries == expected_local_libraries
    )
    _add_check(
        checks,
        "backend-library-linkage",
        linkage_ok,
        "backend_identity_mismatch",
        "llama-server resolves exactly the pinned local llama/ggml libraries"
        if linkage_ok else "llama-server local library linkage differs from the pinned set",
        {
            "expected": sorted(expected_local_libraries),
            "observed": sorted(observed_local_libraries),
            "returncode": linked.returncode,
            "stderr": linked.stderr,
        },
    )

    all_paths = (
        paths.home,
        paths.xdg_config_home,
        paths.xdg_cache_home,
        paths.xdg_data_home,
        paths.xdg_state_home,
    )
    paths_ok = len(set(all_paths)) == len(all_paths) and all(
        path.is_absolute() and path.is_dir() and not path.is_symlink()
        for path in all_paths
    )
    _add_check(
        checks,
        "isolated-home-xdg",
        paths_ok,
        "precondition_failed",
        "fresh HOME/XDG paths are valid" if paths_ok else "HOME/XDG paths are invalid",
        {"paths": [str(path) for path in all_paths]},
    )

    try:
        gpus, processes = gpu_collector()
        gpu_observed_at_utc = datetime.now(timezone.utc)
        gpu_observation_source = (
            "live_nvidia_smi" if collect_gpus is None else "injected_test_fixture"
        )
        gpu_checks = evaluate_gpu_preconditions(profile.gpu, gpus, processes)
        checks.extend(gpu_checks)
    except Exception as exc:
        gpus, processes = (), ()
        gpu_observed_at_utc = None
        gpu_observation_source = "unavailable"
        _add_check(
            checks,
            "gpu-observation",
            False,
            "precondition_failed",
            "could not collect deterministic GPU precondition evidence",
            {
                "availability": "unavailable",
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )

    failures = [check for check in checks if not check.passed]
    return BackendPreflightReport(
        profile_id=profile.profile_id,
        passed=not failures,
        primary_failure_class=(failures[0].failure_class if failures else None),
        checks=tuple(checks),
        gpu_observations=gpus,
        gpu_processes=processes,
        gpu_observed_at_utc=gpu_observed_at_utc,
        gpu_observation_source=gpu_observation_source,
        gpu_observation_commands=(
            _GPU_INVENTORY_COMMAND,
            _GPU_PROCESS_COMMAND,
        ) if collect_gpus is None else (),
    )


def evaluate_gpu_preconditions(
    policy: GpuPreconditionPolicy,
    gpus: tuple[GpuObservation, ...],
    processes: tuple[GpuProcessObservation, ...],
) -> tuple[PreflightCheck, ...]:
    """Validate only the pinned RTX 3090; activity on other GPUs is irrelevant."""
    target = next((gpu for gpu in gpus if gpu.uuid == policy.expected_uuid), None)
    identity_ok = target is not None and target.name == policy.expected_name
    checks: list[PreflightCheck] = []
    _add_check(
        checks,
        "benchmark-gpu-identity",
        identity_ok,
        "precondition_failed",
        "pinned RTX 3090 UUID and model are present"
        if identity_ok else "pinned RTX 3090 UUID/model is not present",
        {"expected_uuid": policy.expected_uuid, "expected_name": policy.expected_name},
    )
    if target is None:
        return tuple(checks)

    target_processes = tuple(
        process for process in processes if process.gpu_uuid == policy.expected_uuid
    )
    blockers = tuple(
        process
        for process in target_processes
        if not _is_exempt_desktop_process(process, policy)
    )
    _add_check(
        checks,
        "benchmark-gpu-processes",
        not blockers,
        "conflicting_gpu_process",
        "no conflicting compute process was reported on the benchmark RTX 3090"
        if not blockers else "conflicting compute process is using the benchmark RTX 3090",
        {
            "blocking_processes": [
                process.model_dump(mode="json") for process in blockers
            ],
            "other_gpu_processes_ignored": [
                process.model_dump(mode="json")
                for process in processes
                if process.gpu_uuid != policy.expected_uuid
            ],
        },
    )
    vram_ok = target.memory_total_mib >= policy.minimum_total_vram_mib
    _add_check(
        checks,
        "benchmark-gpu-memory-observation",
        vram_ok,
        "precondition_failed",
        "benchmark RTX 3090 total/used/free VRAM recorded; backend loading decides capacity"
        if vram_ok else "benchmark RTX 3090 does not match the fixed total VRAM capacity",
        {
            "memory_total_mib": target.memory_total_mib,
            "memory_used_mib": target.memory_used_mib,
            "memory_free_mib": target.memory_free_mib,
            "minimum_total_mib": policy.minimum_total_vram_mib,
            "available_vram_threshold_mib": None,
        },
    )
    return tuple(checks)


def parse_gpu_inventory(text: str) -> tuple[GpuObservation, ...]:
    """Parse the fixed no-header nvidia-smi GPU query format."""
    records: list[GpuObservation] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 8:
            raise ValueError("unexpected nvidia-smi GPU row")
        records.append(
            GpuObservation(
                index=int(row[0]),
                uuid=row[1].strip(),
                name=row[2].strip(),
                memory_total_mib=int(row[3]),
                memory_used_mib=int(row[4]),
                memory_free_mib=int(row[5]),
                utilization_percent=_optional_int(row[6]),
                temperature_celsius=_optional_int(row[7]),
            )
        )
    return tuple(records)


def parse_gpu_processes(text: str) -> tuple[GpuProcessObservation, ...]:
    """Parse the fixed no-header nvidia-smi compute-process query format."""
    records: list[GpuProcessObservation] = []
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if not row:
            continue
        if len(row) < 4:
            raise ValueError("unexpected nvidia-smi process row")
        records.append(
            GpuProcessObservation(
                gpu_uuid=row[0].strip(),
                pid=int(row[1]),
                process_name=",".join(row[2:-1]).strip(),
                used_gpu_memory_mib=_optional_int(row[-1]),
            )
        )
    return tuple(records)


_GPU_INVENTORY_COMMAND = (
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
    "--format=csv,noheader,nounits",
)
_GPU_PROCESS_COMMAND = (
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
)


def collect_nvidia_observations(
) -> tuple[tuple[GpuObservation, ...], tuple[GpuProcessObservation, ...]]:
    """Collect exact GPU/process observations without changing GPU state."""
    gpu = _run_command(
        _GPU_INVENTORY_COMMAND,
        None,
        None,
    )
    if gpu.returncode != 0:
        raise BackendLifecycleError(gpu.stderr.strip() or "nvidia-smi GPU query failed")
    process = _run_command(
        _GPU_PROCESS_COMMAND,
        None,
        None,
    )
    if process.returncode != 0:
        raise BackendLifecycleError(
            process.stderr.strip() or "nvidia-smi process query failed"
        )
    return parse_gpu_inventory(gpu.stdout), parse_gpu_processes(process.stdout)


def is_port_free(host: str, port: int) -> bool:
    """Return whether a TCP listener is active, without binding or disturbing it.

    This intentionally answers the benchmark question ("would a foreign or
    stale server receive traffic on this endpoint?") rather than asking whether
    a plain test socket may bind.  The latter falsely reports Linux ``TIME_WAIT``
    as an occupied port after a clean shutdown.  The pinned llama.cpp server
    sets ``SO_REUSEADDR`` itself, so it can safely rebind after that state; this
    function never enables ``SO_REUSEPORT`` and never kills or touches a peer.
    A startup race remains a normal, explicit backend-start failure.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
        candidate.settimeout(0.2)
        return candidate.connect_ex((host, port)) != 0


def observe_backend_endpoint(
    profile: BackendProfile,
    endpoint: Literal["/metrics", "/slots"],
    *,
    request: Callable[[str, float], tuple[int, str, bytes]] | None = None,
    observed_at_utc: datetime | None = None,
) -> BackendEndpointObservation:
    """Read an enabled diagnostic endpoint without estimating its contents."""
    if endpoint == "/metrics" and not profile.server.metrics_endpoint:
        raise BackendLifecycleError("metrics endpoint is not enabled")
    if endpoint == "/slots" and not profile.server.slots_endpoint:
        raise BackendLifecycleError("slots endpoint is not enabled")
    fetch = request or _diagnostic_request
    url = f"http://{profile.server.host}:{profile.server.port}{endpoint}"
    status, content_type, body = fetch(url, 5.0)
    parsed: JsonValue | None = None
    if "json" in content_type.lower():
        try:
            value = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            value = None
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            parsed = value
    return BackendEndpointObservation(
        endpoint=endpoint,
        observed_at_utc=observed_at_utc or datetime.now(timezone.utc),
        http_status=status,
        content_type=content_type,
        body_base64=base64.b64encode(body).decode("ascii"),
        body_sha256=hashlib.sha256(body).hexdigest(),
        parsed_body=parsed,
    )


@dataclass
class OwnedBackendProcess:
    """A llama-server process created and exclusively owned by Agent Bench."""

    process: subprocess.Popen[bytes]
    stdout_stream: object
    stderr_stream: object
    started_monotonic_ns: int
    invocation: ResolvedBackendInvocation

    def wait_until_ready(
        self,
        profile: BackendProfile,
        *,
        request: Callable[[str, float], tuple[int, bytes]] | None = None,
        poll_seconds: float = 0.25,
    ) -> int:
        """Wait for `/health`; return startup duration after built-in warmup."""
        probe = request or _health_request
        deadline = time.monotonic() + profile.readiness_timeout_seconds
        url = f"http://{profile.server.host}:{profile.server.port}{profile.readiness_endpoint}"
        while time.monotonic() < deadline:
            exit_code = self.process.poll()
            if exit_code is not None:
                raise BackendStartFailed(
                    f"owned llama-server exited before readiness with code {exit_code}"
                )
            try:
                status, body = probe(url, min(2.0, profile.readiness_timeout_seconds))
                parsed = json.loads(body) if body else {}
                if status == 200 and parsed.get("status") == "ok":
                    return time.monotonic_ns() - self.started_monotonic_ns
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(poll_seconds)
        raise BackendReadinessFailed("owned llama-server readiness deadline expired")

    def shutdown(self, grace_seconds: float) -> tuple[int | None, str]:
        """Stop only this Agent Bench-owned child, escalating if it ignores TERM."""
        if self.process.poll() is not None:
            code = self.process.returncode
            self._close_logs()
            return code, "already_exited"
        self.process.terminate()
        try:
            code = self.process.wait(timeout=grace_seconds)
            method = "terminate"
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait(timeout=grace_seconds)
            method = "kill_after_owned_process_timeout"
        self._close_logs()
        return code, method

    def _close_logs(self) -> None:
        for stream in (self.stdout_stream, self.stderr_stream):
            close = getattr(stream, "close", None)
            if close is not None:
                close()


def start_owned_backend(
    profile: BackendProfile,
    paths: BackendRunPaths,
    preflight: BackendPreflightReport,
    stdout_path: Path,
    stderr_path: Path,
    *,
    run_seed: int,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> OwnedBackendProcess:
    """Start one dedicated backend only after a successful preflight."""
    if not preflight.passed:
        raise BackendLifecycleError("cannot start backend after failed preflight")
    invocation = resolve_backend_invocation(profile, paths, run_seed=run_seed)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_stream = stdout_path.open("xb")
    try:
        stderr_stream = stderr_path.open("xb")
    except Exception:
        stdout_stream.close()
        raise
    try:
        process = popen(
            list(invocation.argv),
            cwd=invocation.working_directory,
            env=invocation.environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            shell=False,
        )
    except Exception:
        stdout_stream.close()
        stderr_stream.close()
        raise
    return OwnedBackendProcess(
        process=process,
        stdout_stream=stdout_stream,
        stderr_stream=stderr_stream,
        started_monotonic_ns=time.monotonic_ns(),
        invocation=invocation,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_pinned_file(
    checks: list[PreflightCheck],
    check_id: str,
    expected: PinnedFile,
    mismatch_class: FailureClass,
    hasher: FileHasher,
    *,
    require_executable: bool = False,
) -> None:
    path = expected.path
    exists = path.is_file() and not path.is_symlink()
    if require_executable:
        exists = exists and os.access(path, os.X_OK)
    if not exists:
        _add_check(
            checks, check_id, False, "precondition_failed",
            f"required file is missing or invalid: {path}", {"path": str(path)}
        )
        return
    observed_size = path.stat().st_size
    if observed_size != expected.size_bytes:
        _add_check(
            checks, check_id, False, mismatch_class,
            f"file size mismatch: {path}",
            {"path": str(path), "expected_size": expected.size_bytes, "observed_size": observed_size},
        )
        return
    observed_sha = hasher(path)
    matches = observed_sha == expected.sha256
    _add_check(
        checks, check_id, matches, mismatch_class,
        f"file identity matches: {path}" if matches else f"file SHA256 mismatch: {path}",
        {"path": str(path), "size_bytes": observed_size, "expected_sha256": expected.sha256, "observed_sha256": observed_sha},
    )


def _add_check(
    checks: list[PreflightCheck],
    check_id: str,
    passed: bool,
    failure_class: FailureClass,
    message: str,
    evidence: dict[str, JsonValue],
) -> None:
    checks.append(
        PreflightCheck(
            check_id=check_id,
            passed=passed,
            failure_class=None if passed else failure_class,
            message=message,
            evidence=evidence,
        )
    )


def _is_exempt_desktop_process(
    process: GpuProcessObservation,
    policy: GpuPreconditionPolicy,
) -> bool:
    used = process.used_gpu_memory_mib
    return (
        used is not None
        and used <= policy.maximum_exempt_desktop_process_vram_mib
        and any(marker in process.process_name for marker in policy.exempt_desktop_process_markers)
    )


def _run_command(
    argv: tuple[str, ...],
    environment: Mapping[str, str] | None,
    cwd: Path | None,
) -> CommandResult:
    result = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(environment) if environment is not None else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return CommandResult(result.returncode, result.stdout, result.stderr)


def _health_request(url: str, timeout: float) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _diagnostic_request(url: str, timeout: float) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return (
                response.status,
                response.headers.get("Content-Type", ""),
                response.read(),
            )
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), exc.read()


def _optional_int(value: str) -> int | None:
    stripped = value.strip()
    if stripped in {"", "N/A", "[Not Supported]"}:
        return None
    return int(stripped)


def _local_ldd_paths(text: str, repository: Path) -> set[str]:
    root = repository.resolve()
    result: set[str] = set()
    for line in text.splitlines():
        fields = line.strip().split()
        candidate: str | None = None
        if "=>" in fields:
            index = fields.index("=>")
            if index + 1 < len(fields) and fields[index + 1].startswith("/"):
                candidate = fields[index + 1]
        elif fields and fields[0].startswith("/"):
            candidate = fields[0]
        if candidate is None:
            continue
        resolved = Path(candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        result.add(str(resolved))
    return result


def _number(value: float) -> str:
    return f"{value:.1f}" if value.is_integer() else str(value)
