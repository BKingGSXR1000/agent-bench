"""Explicit, hash-checked materialization of benchmark-managed payloads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from agent_bench.backend import load_backend_profile
from agent_bench.toolchains import ToolchainVerificationError, _verify_file


class BootstrapError(RuntimeError):
    """A pinned payload could not be downloaded or materialized safely."""


_COMPONENTS = ("opencode", "node", "pi", "hermes", "llama-cpp", "qwen")


def install_toolchains(
    components: tuple[str, ...] = (), *, include_model: bool = False, model_destination: Path | None = None
) -> dict[str, str]:
    """Install only requested pinned payloads; never consult PATH or user homes."""
    chosen = components or ("opencode", "node", "pi", "hermes", "llama-cpp")
    unknown = set(chosen).difference(_COMPONENTS)
    if unknown:
        raise BootstrapError(f"unknown component(s): {', '.join(sorted(unknown))}")
    if include_model and "qwen" not in chosen:
        chosen = (*chosen, "qwen")
    result: dict[str, str] = {}
    for component in chosen:
        if component == "opencode":
            _install_opencode(); result[component] = "INSTALLED_OR_VERIFIED"
        elif component == "node":
            _install_node(); result[component] = "INSTALLED_OR_VERIFIED"
        elif component == "pi":
            _install_pi(); result[component] = "INSTALLED_OR_VERIFIED"
        elif component == "qwen":
            if not include_model:
                result[component] = "SKIPPED_REQUIRES_INCLUDE_MODEL"
            else:
                _install_model(model_destination); result[component] = "INSTALLED_OR_VERIFIED"
        else:
            result[component] = "MANUAL_REQUIRED_SEE_DOCS"
    return result


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _identity(relative: str) -> dict[str, object]:
    return json.loads((_root() / relative).read_text(encoding="utf-8"))


def _install_opencode() -> None:
    raw = _identity("toolchains/opencode/1.18.25/identity.json")
    target = _root() / "toolchains/opencode/1.18.25/bin/opencode"
    if _matches(target, raw):
        return
    asset = raw["public_source"]["linux_x64_asset"]  # type: ignore[index]
    with tempfile.TemporaryDirectory(prefix="agent-bench-opencode-") as directory:
        archive = Path(directory) / asset["name"]
        _download(asset["url"], archive, asset["sha256"])  # type: ignore[index]
        with tarfile.open(archive, "r:gz") as bundle:
            member = next((item for item in bundle.getmembers() if item.isfile() and item.name.rstrip("/").endswith("opencode")), None)
            if member is None:
                raise BootstrapError("OpenCode release archive has no executable named opencode")
            source = bundle.extractfile(member)
            if source is None:
                raise BootstrapError("could not read OpenCode executable from release archive")
            target.parent.mkdir(parents=True, exist_ok=True)
            staged = target.with_name(".opencode.staged")
            with staged.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            os.chmod(staged, 0o755)
            _verify_file(staged, str(raw["sha256"]), int(raw["size_bytes"]), executable=True)
            os.replace(staged, target)
    _verify_file(target, str(raw["sha256"]), int(raw["size_bytes"]), executable=True)


def _install_node() -> None:
    raw = _identity("toolchains/node/26.8.1/identity.json")
    root = _root() / "toolchains/node/26.8.1"
    target = root / "bin/node"
    npm_cli = root / "lib/node_modules/npm/bin/npm-cli.js"
    if _matches(target, raw) and npm_cli.is_file():
        return
    asset = raw["public_source"]["linux_x64_archive"]  # type: ignore[index]
    with tempfile.TemporaryDirectory(prefix="agent-bench-node-") as directory:
        archive = Path(directory) / asset["name"]
        _download(asset["url"], archive, asset["sha256"])  # type: ignore[index]
        with tarfile.open(archive, "r:xz") as bundle:
            names = [item for item in bundle.getmembers() if item.isfile()]
            prefix = asset["top_level_directory"] + "/"  # type: ignore[index]
            for member in names:
                if not member.name.startswith(prefix):
                    raise BootstrapError("unexpected member in Node release archive")
                destination = root / member.name.removeprefix(prefix)
                source = bundle.extractfile(member)
                if source is None:
                    raise BootstrapError(f"could not read {member.name} from Node archive")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                os.chmod(destination, member.mode)
    _verify_file(target, str(raw["sha256"]), int(raw["size_bytes"]), executable=True)
    if not npm_cli.is_file():
        raise BootstrapError("official Node archive did not provide its bundled npm CLI")


def _install_pi() -> None:
    _install_node()
    raw = _identity("toolchains/pi/0.84.4/identity.json")
    root = _root() / "toolchains/pi/0.84.4"
    entrypoint = root / "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
    if _matches_named(entrypoint, raw["runtime"], "entrypoint"):  # type: ignore[arg-type]
        return
    lock = root / "package-lock.json"
    package = root / "package.json"
    if not lock.is_file() or not package.is_file():
        raise BootstrapError("Pi's checked-in package.json and package-lock.json are required")
    npm_cli = _root() / "toolchains/node/26.8.1/lib/node_modules/npm/bin/npm-cli.js"
    node = _root() / "toolchains/node/26.8.1/bin/node"
    cache = root / ".npm-cache"
    environment = {"HOME": str(root / ".npm-home"), "NPM_CONFIG_CACHE": str(cache), "NPM_CONFIG_AUDIT": "false", "NPM_CONFIG_FUND": "false", "PATH": os.environ.get("PATH", "")}
    completed = subprocess.run([str(node), str(npm_cli), "ci", "--prefix", str(root), "--ignore-scripts"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, env=environment)
    if completed.returncode:
        raise BootstrapError(f"pinned Pi npm ci failed: {completed.stderr.strip() or completed.stdout.strip()}")
    _verify_file(entrypoint, str(raw["runtime"]["entrypoint_sha256"]), int(raw["runtime"]["entrypoint_size_bytes"]))  # type: ignore[index]


def _install_model(destination: Path | None) -> None:
    profile = load_backend_profile()
    target = destination.expanduser().resolve() if destination else profile.model.path
    if target.is_file():
        _verify_file(target, profile.model.sha256, profile.model.size_bytes)
        return
    source = _identity("environment/model-v1.json")["public_source"]
    target.parent.mkdir(parents=True, exist_ok=True)
    _download(str(source["url"]), target, profile.model.sha256)
    _verify_file(target, profile.model.sha256, profile.model.size_bytes)


def _matches(path: Path, identity: dict[str, object]) -> bool:
    try:
        _verify_file(path, str(identity["sha256"]), int(identity["size_bytes"]), executable=path.name in {"node", "opencode"})
    except ToolchainVerificationError:
        return False
    return True


def _matches_named(path: Path, identity: dict[str, object], prefix: str) -> bool:
    try:
        _verify_file(path, str(identity[f"{prefix}_sha256"]), int(identity[f"{prefix}_size_bytes"]))
    except ToolchainVerificationError:
        return False
    return True


def _download(url: object, target: Path, expected_sha256: object) -> None:
    digest = hashlib.sha256()
    temporary = target.with_name(f".{target.name}.part")
    try:
        with urlopen(str(url), timeout=60) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != str(expected_sha256):
            raise BootstrapError(f"download SHA256 differs for {url}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
