from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from agent_bench.config import ExperimentConfigError, load_experiment
from agent_bench.models import SUPPORTED_HARNESS_IDS
from conftest import ExperimentFixture


@pytest.mark.parametrize(
    ("path", "definition_digest", "matrix_digest", "first_run_id"),
    [
        (
            "experiments/pocket-ledger-v1.yaml",
            "2dea097c2a8142076d11d00b88eb17dbe5192134d4cd140a19e933c2732ff82a",
            "4955ce2a5556e402080455511b2f3aa0ecef134308751f1640b80f4ca71ebab1",
            "hermes-hermes-default-v1-keyboard-entry-vague-r001-e041841d2df985953b43d6c4",
        ),
        (
            "experiments/pocket-ledger-v1-hermes-reasoning-screen-v1.yaml",
            "e98c1eb6396269877628f11f428da2bc76cee131349f6a03e553c9be809443c3",
            "f82cc608539cece72e10f5c16b1c2411efce29ebacd4367ddf2e6df3ec28179c",
            "hermes-hermes-reasoning-low-v1-entry-category-normal-r001-52f596c916666fa34177a588",
        ),
    ],
)
def test_existing_count_based_experiments_retain_sealed_identities(
    path: str, definition_digest: str, matrix_digest: str, first_run_id: str,
) -> None:
    """Adding explicit indices must not change existing count-based evidence."""
    from agent_bench.matrix import expand_experiment

    experiment = load_experiment(Path(path))
    assert experiment.definition_digest == definition_digest
    assert experiment.matrix_digest == matrix_digest
    assert expand_experiment(experiment)[0].run_id == first_run_id


def test_loads_valid_experiment_configuration(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)

    assert experiment.schema_version == "1.0.0"
    assert experiment.fixed_environment.model.name == "Qwen 3.8 27B"
    assert experiment.fixed_environment.backend.implementation == "llama.cpp"
    assert experiment.fixed_environment.hardware.gpu_count == 1
    assert len(experiment.harnesses) == 3
    assert len(experiment.harness_profiles) == 4
    assert len(experiment.prompts) == 3
    assert experiment.prompts[0].path.is_absolute()
    assert experiment.baseline_repository.is_absolute()
    assert experiment.definition_digest
    assert experiment.matrix_digest


def test_persisted_models_are_frozen(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)

    with pytest.raises(ValidationError):
        experiment.repetitions = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(repetitions=0), "greater than or equal to 1"),
        (lambda data: data.update(repetitions=2, repetition_indices=[2, 3]), "exactly one of repetitions or repetition_indices"),
        (lambda data: (data.pop("repetitions"), data.update(repetition_indices=[2, 2])), "must be unique"),
        (lambda data: (data.pop("repetitions"), data.update(repetition_indices=[0])), "must be positive"),
        (lambda data: (data.pop("repetitions"), data.update(repetition_indices=[-1])), "must be positive"),
        (lambda data: (data.pop("repetitions"), data.update(repetition_indices=["2"])), "valid integer"),
        (
            lambda data: data["harness_profiles"][0].update(
                harness_id="opencode"
            ),
            "missing profiles for: pi",
        ),
        (
            lambda data: data["harness_profiles"].append(
                copy.deepcopy(data["harness_profiles"][0])
            ),
            "duplicate profile_id",
        ),
        (lambda data: data.update(unexpected=True), "Extra inputs are not permitted"),
        (
            lambda data: data["fixed_environment"]["model"].update(
                name="Another model"
            ),
            "benchmark v1 model name",
        ),
    ],
)
def test_rejects_invalid_experiment_definitions(
    experiment_fixture: ExperimentFixture,
    mutation: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    data = copy.deepcopy(experiment_fixture.data)
    mutation(data)
    experiment_fixture.write(data)

    with pytest.raises(ExperimentConfigError, match=message):
        load_experiment(experiment_fixture.path)


def test_exact_prompt_bytes_are_loaded_without_normalization(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    prompt = next(item for item in experiment.prompts if item.variant_label == "vague")
    exact_bytes = experiment_fixture.prompt_bytes["vague"]

    assert prompt.content.encode("utf-8") == exact_bytes
    assert prompt.byte_length == len(exact_bytes)
    assert prompt.sha256 == hashlib.sha256(exact_bytes).hexdigest()
    assert "  \r\n" in prompt.content


def test_prompt_hash_is_optional_and_calculated(
    experiment_fixture: ExperimentFixture,
) -> None:
    data = copy.deepcopy(experiment_fixture.data)
    del data["prompts"][0]["sha256"]
    experiment_fixture.write(data)

    experiment = load_experiment(experiment_fixture.path)

    assert experiment.prompts[0].sha256 == hashlib.sha256(
        experiment_fixture.prompt_bytes["vague"]
    ).hexdigest()


def test_rejects_prompt_hash_mismatch(
    experiment_fixture: ExperimentFixture,
) -> None:
    data = copy.deepcopy(experiment_fixture.data)
    data["prompts"][0]["sha256"] = "f" * 64
    experiment_fixture.write(data)

    with pytest.raises(ExperimentConfigError, match="SHA256 mismatch"):
        load_experiment(experiment_fixture.path)


def test_rejects_inline_prompt_content(
    experiment_fixture: ExperimentFixture,
) -> None:
    data = copy.deepcopy(experiment_fixture.data)
    data["prompts"][0]["content"] = "inline is forbidden"
    experiment_fixture.write(data)

    with pytest.raises(ExperimentConfigError, match="separate UTF-8 file"):
        load_experiment(experiment_fixture.path)


def test_supports_exactly_the_initial_harness_ids(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    assert {harness.harness_id for harness in experiment.harnesses} == (
        SUPPORTED_HARNESS_IDS
    )

    data = copy.deepcopy(experiment_fixture.data)
    data["harnesses"][0]["harness_id"] = "unsupported"
    experiment_fixture.write(data)
    with pytest.raises(ExperimentConfigError, match="opencode.*pi.*hermes"):
        load_experiment(experiment_fixture.path)


def test_rejects_unsupported_schema_version(
    experiment_fixture: ExperimentFixture,
) -> None:
    data = copy.deepcopy(experiment_fixture.data)
    data["schema_version"] = "2.0.0"
    experiment_fixture.write(data)

    with pytest.raises(ExperimentConfigError, match="1.0.0"):
        load_experiment(experiment_fixture.path)


def test_missing_prompt_file_is_clear(
    experiment_fixture: ExperimentFixture,
) -> None:
    data = copy.deepcopy(experiment_fixture.data)
    data["prompts"][0]["path"] = "missing.txt"
    experiment_fixture.write(data)

    with pytest.raises(ExperimentConfigError, match="cannot read prompt file"):
        load_experiment(experiment_fixture.path)
