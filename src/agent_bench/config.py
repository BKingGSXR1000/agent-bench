"""Load and validate Agent Bench experiment YAML files."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agent_bench.models import ExperimentDefinition

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ExperimentConfigError(ValueError):
    """Raised when an experiment file or referenced prompt is invalid."""


def load_experiment(path: Path) -> ExperimentDefinition:
    """Load one YAML experiment and resolve its byte-exact prompt files."""
    experiment_path = path.expanduser().resolve()
    try:
        yaml_bytes = experiment_path.read_bytes()
    except OSError as exc:
        raise ExperimentConfigError(
            f"cannot read experiment file {experiment_path}: {exc}"
        ) from exc

    try:
        yaml_text = yaml_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExperimentConfigError(
            f"experiment file is not valid UTF-8: {experiment_path}"
        ) from exc

    try:
        raw = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(
            f"invalid YAML in experiment file {experiment_path}: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ExperimentConfigError(
            f"experiment file must contain a YAML mapping: {experiment_path}"
        )

    data: dict[str, Any] = deepcopy(raw)
    configured_repository = data.get("baseline_repository")
    if isinstance(configured_repository, str) and configured_repository:
        repository_path = Path(configured_repository).expanduser()
        if not repository_path.is_absolute():
            repository_path = experiment_path.parent / repository_path
        data["baseline_repository"] = repository_path.resolve()
    data["prompts"] = _load_prompts(data.get("prompts"), experiment_path.parent)

    try:
        return ExperimentDefinition.model_validate(data)
    except ValidationError as exc:
        raise ExperimentConfigError(
            f"invalid experiment configuration in {experiment_path}:\n{exc}"
        ) from exc


def _load_prompts(value: object, base_directory: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigError("prompts must be a non-empty YAML list")

    loaded: list[dict[str, Any]] = []
    for index, prompt_value in enumerate(value, start=1):
        if not isinstance(prompt_value, dict):
            raise ExperimentConfigError(f"prompt {index} must be a YAML mapping")
        prompt: dict[str, Any] = deepcopy(prompt_value)
        if "content" in prompt or "byte_length" in prompt:
            raise ExperimentConfigError(
                f"prompt {index} content must live in a separate UTF-8 file"
            )

        configured_path = prompt.get("path")
        if not isinstance(configured_path, str) or not configured_path:
            raise ExperimentConfigError(f"prompt {index} requires a non-empty path")
        prompt_path = Path(configured_path).expanduser()
        if not prompt_path.is_absolute():
            prompt_path = base_directory / prompt_path
        prompt_path = prompt_path.resolve()

        try:
            content_bytes = prompt_path.read_bytes()
        except OSError as exc:
            raise ExperimentConfigError(
                f"cannot read prompt file {prompt_path}: {exc}"
            ) from exc
        if not prompt_path.is_file():
            raise ExperimentConfigError(f"prompt path is not a file: {prompt_path}")
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExperimentConfigError(
                f"prompt file is not valid UTF-8: {prompt_path}"
            ) from exc

        calculated_sha256 = hashlib.sha256(content_bytes).hexdigest()
        configured_sha256 = prompt.get("sha256")
        if configured_sha256 is not None:
            if not isinstance(configured_sha256, str) or not _SHA256_PATTERN.fullmatch(
                configured_sha256
            ):
                raise ExperimentConfigError(
                    f"prompt {index} sha256 must be 64 lowercase hexadecimal characters"
                )
            if configured_sha256 != calculated_sha256:
                prompt_id = prompt.get("prompt_id", index)
                raise ExperimentConfigError(
                    f"prompt {prompt_id!r} SHA256 mismatch: expected "
                    f"{configured_sha256}, calculated {calculated_sha256}"
                )

        prompt.update(
            {
                "path": prompt_path,
                "content": content,
                "byte_length": len(content_bytes),
                "sha256": calculated_sha256,
            }
        )
        loaded.append(prompt)
    return loaded
