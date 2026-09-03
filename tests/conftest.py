from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_bench.preservation import PreservedRun, preserve_isolated_operation


@dataclass(frozen=True)
class ExperimentFixture:
    path: Path
    data: dict[str, Any]
    prompt_bytes: dict[str, bytes]

    def write(self, data: dict[str, Any]) -> Path:
        self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return self.path


@dataclass(frozen=True)
class GitRepositoryFixture:
    path: Path
    artifacts_root: Path
    worktrees_root: Path
    baseline_commit: str

    def git(self, *arguments: str, check: bool = True) -> str:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
        result = subprocess.run(
            ["git", "-C", str(self.path), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            text=True,
        )
        if check and result.returncode != 0:
            raise AssertionError(result.stderr)
        return result.stdout


@pytest.fixture
def experiment_fixture(tmp_path: Path) -> ExperimentFixture:
    prompt_bytes = {
        "vague": b"Build the thing.  \r\n",
        "normal": b"Build a useful task tracker.\n",
        "precise": b"Build a task tracker.\nAdd tests.\n",
    }
    prompt_entries: list[dict[str, Any]] = []
    for variant, content in prompt_bytes.items():
        prompt_path = tmp_path / f"{variant}.txt"
        prompt_path.write_bytes(content)
        prompt_entries.append(
            {
                "prompt_id": f"task-{variant}",
                "semantic_task_id": "task",
                "variant_label": variant,
                "path": prompt_path.name,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )

    data: dict[str, Any] = {
        "schema_version": "1.0.0",
        "experiment_id": "test-experiment",
        "name": "Test experiment",
        "created_at": "2026-09-03T10:00:00Z",
        "baseline_repository": "subject",
        "baseline_revision": "0123456789abcdef",
        "fixed_environment": {
            "fixed_environment_id": "fixed-v1",
            "model": {
                "model_identity_id": "qwen-v1",
                "name": "Qwen 3.8 27B",
                "quantization": "Q4_K_M",
                "path": "models/qwen.gguf",
                "filename": "qwen.gguf",
                "file_size_bytes": 1234,
                "sha256": "1" * 64,
            },
            "backend": {
                "backend_identity_id": "llamacpp-v1",
                "implementation": "llama.cpp",
                "executable": "bin/llama-server",
                "version": "test-build",
                "commit": "abc123",
            },
            "hardware": {
                "hardware_identity_id": "hardware-v1",
                "name": "Test fixed host",
                "cpu_architecture": "x86_64",
                "gpu_model": "Test GPU",
                "gpu_count": 1,
            },
            "server_parameters": {"port": 8080},
            "generation": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "seed": 7,
                "max_output_tokens": 1024,
                "context_size": 8192,
            },
            "restart_policy": "per_run",
            "readiness_policy": "fixture-v1",
        },
        "harnesses": [
            {
                "harness_id": "pi",
                "display_name": "Pi",
                "version": "1.0",
                "executable": "pi",
                "upstream_project": "pi.example",
            },
            {
                "harness_id": "opencode",
                "display_name": "OpenCode",
                "version": "1.0",
                "executable": "opencode",
                "upstream_project": "opencode.example",
            },
            {
                "harness_id": "hermes",
                "display_name": "Hermes",
                "version": "1.0",
                "executable": "hermes",
                "upstream_project": "hermes.example",
            },
        ],
        "harness_profiles": [
            {
                "profile_id": "pi-default",
                "harness_id": "pi",
                "profile_version": "1.0",
                "kind": "controlled_default",
                "upstream_defaults_source": "pi-defaults",
            },
            {
                "profile_id": "opencode-high",
                "harness_id": "opencode",
                "profile_version": "1.0",
                "kind": "benchmark_specific",
                "upstream_defaults_source": "opencode-defaults",
                "deviations": ["reasoning=high"],
                "settings": {"reasoning": "high"},
            },
            {
                "profile_id": "hermes-default",
                "harness_id": "hermes",
                "profile_version": "1.0",
                "kind": "controlled_default",
                "upstream_defaults_source": "hermes-defaults",
            },
            {
                "profile_id": "opencode-default",
                "harness_id": "opencode",
                "profile_version": "1.0",
                "kind": "controlled_default",
                "upstream_defaults_source": "opencode-defaults",
            },
        ],
        "prompts": prompt_entries,
        "repetitions": 2,
        "ordering": {"mode": "canonical"},
    }
    path = tmp_path / "experiment.yaml"
    fixture = ExperimentFixture(path=path, data=data, prompt_bytes=prompt_bytes)
    fixture.write(data)
    return fixture


@pytest.fixture
def git_repository(tmp_path: Path) -> GitRepositoryFixture:
    repository = tmp_path / "baseline"
    repository.mkdir()
    artifacts_root = tmp_path / "artifacts"
    worktrees_root = tmp_path / "worktrees"
    fixture = GitRepositoryFixture(
        path=repository,
        artifacts_root=artifacts_root,
        worktrees_root=worktrees_root,
        baseline_commit="",
    )
    fixture.git("init")
    (repository / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (repository / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    (repository / "delete-me.txt").write_text("delete me\n", encoding="utf-8")
    fixture.git("add", ".")
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@invalid",
        "GIT_COMMITTER_NAME": "Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    }
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "baseline"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    commit = fixture.git("rev-parse", "HEAD").strip()
    return GitRepositoryFixture(
        path=repository,
        artifacts_root=artifacts_root,
        worktrees_root=worktrees_root,
        baseline_commit=commit,
    )


def full_result_operation(worktree: Path) -> None:
    (worktree / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (worktree / "delete-me.txt").unlink()
    (worktree / "new.txt").write_text("untracked\n", encoding="utf-8")

    files = {
        "ignored/generated.bin": b"ignored but required\n",
        "__pycache__/module.pyc": b"ephemeral bytecode\n",
        ".pytest_cache/state": b"ephemeral pytest state\n",
        "node_modules/pkg/index.js": b"dependency\n",
        "dist/app.js": b"distribution\n",
        "build/output.bin": b"build output\n",
        "vendor/library.txt": b"vendored\n",
        "generated/asset.txt": b"generated asset\n",
    }
    for relative_path, content in files.items():
        path = worktree / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


@pytest.fixture
def preserved_run(git_repository: GitRepositoryFixture) -> PreservedRun:
    return preserve_isolated_operation(
        repository=git_repository.path,
        baseline_ref="HEAD",
        run_id="preserved-run",
        experiment_id="test-experiment",
        artifacts_root=git_repository.artifacts_root,
        worktrees_root=git_repository.worktrees_root,
        operation=full_result_operation,
    )
