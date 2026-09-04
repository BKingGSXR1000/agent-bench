"""Verification of local payloads against tracked, portable identity manifests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from agent_bench.backend import load_backend_profile
from agent_bench.hermes import inspect_hermes_toolchain, load_hermes_profile
from agent_bench.opencode import load_opencode_profile, verify_opencode_toolchain
from agent_bench.pi import inspect_pi_toolchain, load_pi_profile


class ToolchainVerificationError(RuntimeError):
    """A required benchmark-managed payload is absent or has drifted."""


def verify_toolchains() -> dict[str, dict[str, str]]:
    """Verify every pinned local executable/payload without starting a server."""
    result: dict[str, dict[str, str]] = {}
    checks = {
        "OpenCode 1.18.25": _verify_opencode,
        "Pi 0.84.4": _verify_pi,
        "Node 26.8.1": _verify_node,
        "Hermes 0.21.0": _verify_hermes,
        "llama.cpp": _verify_backend,
        "Qwen GGUF": _verify_model,
    }
    for name, check in checks.items():
        try:
            detail = check()
        except Exception as exc:
            result[name] = {"status": "MISSING_OR_DRIFTED", "detail": f"{type(exc).__name__}: {exc}"}
        else:
            result[name] = {"status": "OK", "detail": detail}
    return result


def _verify_opencode() -> str:
    observed = verify_opencode_toolchain(load_opencode_profile())
    return f"{observed.version} {observed.sha256}"


def _verify_pi() -> str:
    profile = load_pi_profile()
    inspect_pi_toolchain(profile.toolchain)
    return f"{profile.toolchain.package_version} {profile.toolchain.node_modules_tree_sha256}"


def _verify_node() -> str:
    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root / "toolchains/node/26.8.1/identity.json").read_text(encoding="utf-8"))
    path = _resolve_toolchain_path(raw["path"])
    _verify_file(path, raw["sha256"], int(raw["size_bytes"]), executable=True)
    version = subprocess.run([str(path), "--version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=30).stdout.strip()
    if version != raw["version"]:
        raise ToolchainVerificationError(f"Node version mismatch: expected {raw['version']}, observed {version!r}")
    return f"{version} {raw['sha256']}"


def _verify_hermes() -> str:
    profile = load_hermes_profile()
    inspect_hermes_toolchain(profile.toolchain)
    return f"{profile.toolchain.version_output} {profile.toolchain.source_tree_sha256}"


def _verify_backend() -> str:
    profile = load_backend_profile()
    _verify_file(profile.executable.path, profile.executable.sha256, profile.executable.size_bytes, executable=True)
    for library in profile.local_libraries:
        _verify_file(library.path, library.sha256, library.size_bytes)
    _verify_file(profile.chat_template.path, profile.chat_template.sha256, profile.chat_template.size_bytes)
    return f"{profile.llama_cpp_commit} {profile.executable.sha256}"


def _verify_model() -> str:
    profile = load_backend_profile()
    _verify_file(profile.model.path, profile.model.sha256, profile.model.size_bytes)
    return f"{profile.model.path.name} {profile.model.sha256}"


def _verify_file(path: Path, expected_sha256: str, expected_size: int, *, executable: bool = False) -> None:
    if not path.is_file():
        raise ToolchainVerificationError(f"required payload is missing: {path}")
    if path.stat().st_size != expected_size:
        raise ToolchainVerificationError(f"payload size differs: {path}")
    if executable and (path.is_symlink() or not os.access(path, os.X_OK)):
        raise ToolchainVerificationError(f"required executable is invalid: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise ToolchainVerificationError(f"payload SHA256 differs: {path}")


def _resolve_toolchain_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ToolchainVerificationError("toolchain path must be a string")
    path = Path(value)
    if not path.is_absolute():
        return (Path(__file__).resolve().parents[2] / path).resolve()
    if "toolchains" in path.parts:
        return Path(__file__).resolve().parents[2] / "toolchains" / Path(*path.parts[path.parts.index("toolchains") + 1:])
    return path
