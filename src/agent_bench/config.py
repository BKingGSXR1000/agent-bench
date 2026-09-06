"""Load and validate Agent Bench experiment YAML files."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agent_bench.functional import load_functional_scenario
from agent_bench.functional_suite import load_functional_suite
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
        experiment = ExperimentDefinition.model_validate(data)
    except ValidationError as exc:
        raise ExperimentConfigError(
            f"invalid experiment configuration in {experiment_path}:\n{exc}"
        ) from exc
    _validate_functional_associations(experiment)
    return experiment


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
        _load_functional_association(prompt, base_directory, index)
        loaded.append(prompt)
    return loaded


def _load_functional_association(prompt: dict[str, Any], base_directory: Path, index: int) -> None:
    """Resolve and pin one optional evaluator scenario contract."""
    value = prompt.get("functional_scenario")
    if value is None:
        return
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"prompt {index} functional_scenario must be a mapping")
    raw = deepcopy(value)
    definition = raw.get("scenario_definition")
    if not isinstance(definition, str) or not definition:
        raise ExperimentConfigError(f"prompt {index} functional_scenario requires scenario_definition")
    scenario_path = (base_directory / definition).resolve() if not Path(definition).is_absolute() else Path(definition).resolve()
    try:
        scenario = load_functional_scenario(scenario_path)
    except Exception as exc:
        raise ExperimentConfigError(f"prompt {index} functional scenario is invalid: {exc}") from exc
    if raw.get("scenario_id") != scenario.scenario_id:
        raise ExperimentConfigError(f"prompt {index} functional scenario_id does not match scenario definition")
    if raw.get("prompt_variant") != prompt.get("variant_label"):
        raise ExperimentConfigError(f"prompt {index} functional prompt_variant must match variant_label")
    raw["scenario_definition"] = scenario_path
    observed_identity = {
        "scenario_definition_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
        "validator_version": scenario.validator_version,
        "validator_sha256": hashlib.sha256(scenario.validator.read_bytes()).hexdigest(),
    }
    for field, observed in observed_identity.items():
        configured = raw.get(field)
        if configured is not None and configured != observed:
            raise ExperimentConfigError(f"prompt {index} functional {field} does not match checked-in evaluator content")
        raw[field] = observed
    suite_path = raw.pop("suite_manifest", None)
    if suite_path is None:
        raise ExperimentConfigError(f"prompt {index} functional_scenario requires suite_manifest to bind its prompt contract")
    if suite_path is not None:
        if not isinstance(suite_path, str) or not suite_path:
            raise ExperimentConfigError(f"prompt {index} suite_manifest must be a non-empty path")
        resolved_suite = (base_directory / suite_path).resolve() if not Path(suite_path).is_absolute() else Path(suite_path).resolve()
        try:
            suite = load_functional_suite(resolved_suite)
            entry = next(item for item in suite.scenarios if item.scenario_id == scenario.scenario_id)
        except Exception as exc:
            raise ExperimentConfigError(f"prompt {index} functional suite contract is invalid: {exc}") from exc
        if entry.scenario_definition_sha256 != raw["scenario_definition_sha256"] or entry.validator_sha256 != raw["validator_sha256"]:
            raise ExperimentConfigError(f"prompt {index} functional scenario does not match suite contract")
        expected_prompt = entry.prompts[raw["prompt_variant"]]
        if expected_prompt.sha256 != prompt["sha256"]:
            raise ExperimentConfigError(f"prompt {index} bytes do not match the suite prompt contract")
        raw.update({"suite_id": suite.suite_id, "suite_version": suite.suite_version, "suite_manifest_sha256": suite.manifest_sha256, "tier": entry.tier})
    prompt["functional_scenario"] = raw


def _validate_functional_associations(experiment: ExperimentDefinition) -> None:
    """Fail closed when a configured functional task disagrees with its baseline."""
    for prompt in experiment.prompts:
        association = prompt.functional_scenario
        if association is None:
            continue
        if prompt.semantic_task_id != association.scenario_id:
            raise ExperimentConfigError(
                f"prompt {prompt.prompt_id} semantic_task_id must equal functional scenario_id"
            )
        scenario = load_functional_scenario(association.scenario_definition)
        if experiment.portable_baseline is None:
            raise ExperimentConfigError("functional scenarios require identity_version 2.0.0 portable_baseline")
        from agent_bench.subject import load_frozen_subject
        subject = load_frozen_subject(scenario.subject_root)
        if experiment.portable_baseline != subject.identity:
            raise ExperimentConfigError(
                f"functional scenario {association.scenario_id} baseline does not match experiment portable_baseline"
            )
