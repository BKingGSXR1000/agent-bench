from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml


@dataclass(frozen=True)
class ExperimentFixture:
    path: Path
    data: dict[str, Any]
    prompt_bytes: dict[str, bytes]

    def write(self, data: dict[str, Any]) -> Path:
        self.path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return self.path


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
