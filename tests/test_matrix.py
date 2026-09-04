from __future__ import annotations

from collections import Counter, defaultdict

from agent_bench.config import load_experiment
from agent_bench.matrix import (
    expand_experiment,
    generate_run_definitions,
    order_run_definitions,
)
from agent_bench.models import ExecutionOrdering
from conftest import ExperimentFixture


def test_matrix_is_complete_cartesian_product(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    runs = generate_run_definitions(experiment)

    assert len(runs) == 4 * 3 * 2
    combinations = {
        (run.harness_id, run.profile_id, run.prompt_id, run.repetition_index)
        for run in runs
    }
    assert len(combinations) == len(runs)


def test_fixed_environment_is_not_a_matrix_dimension(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    runs = generate_run_definitions(experiment)

    assert len({run.fixed_environment_id for run in runs}) == 1
    assert len({run.fixed_environment_digest for run in runs}) == 1
    assert len(runs) == (
        len(experiment.harness_profiles)
        * len(experiment.prompts)
        * experiment.repetitions
    )


def test_repetitions_are_one_based_for_every_matrix_cell(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    repetitions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for run in generate_run_definitions(experiment):
        repetitions[(run.profile_id, run.prompt_id)].append(run.repetition_index)

    assert repetitions
    assert all(indices == [1, 2] for indices in repetitions.values())


def test_generation_seed_is_deterministic_by_repetition_across_harnesses(
    experiment_fixture: ExperimentFixture,
) -> None:
    runs = generate_run_definitions(load_experiment(experiment_fixture.path))

    assert {run.generation_seed for run in runs if run.repetition_index == 1} == {1001}
    assert {run.generation_seed for run in runs if run.repetition_index == 2} == {1002}


def test_run_ids_are_unique_human_readable_and_deterministic(
    experiment_fixture: ExperimentFixture,
) -> None:
    first = generate_run_definitions(load_experiment(experiment_fixture.path))
    second = generate_run_definitions(load_experiment(experiment_fixture.path))

    assert [run.run_id for run in first] == [run.run_id for run in second]
    assert len({run.run_id for run in first}) == len(first)
    assert all(run.harness_id in run.run_id for run in first)
    assert all(run.prompt_id in run.run_id for run in first)


def test_canonical_order_is_lexicographic_then_repetition(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    runs = expand_experiment(experiment)
    keys = [
        (run.harness_id, run.profile_id, run.prompt_id, run.repetition_index)
        for run in runs
    ]

    assert keys == sorted(keys)
    assert [run.matrix_index for run in runs] == list(range(1, len(runs) + 1))


def test_seeded_shuffle_is_reproducible(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path).model_copy(
        update={"ordering": ExecutionOrdering(mode="shuffled", seed=11)}
    )

    first = expand_experiment(experiment)
    second = expand_experiment(experiment)

    assert [run.run_id for run in first] == [run.run_id for run in second]


def test_different_shuffle_seeds_change_only_order(
    experiment_fixture: ExperimentFixture,
) -> None:
    base = load_experiment(experiment_fixture.path)
    first_experiment = base.model_copy(
        update={"ordering": ExecutionOrdering(mode="shuffled", seed=11)}
    )
    second_experiment = base.model_copy(
        update={"ordering": ExecutionOrdering(mode="shuffled", seed=12)}
    )
    first = expand_experiment(first_experiment)
    second = expand_experiment(second_experiment)

    assert [run.run_id for run in first] != [run.run_id for run in second]
    assert {run.run_id for run in first} == {run.run_id for run in second}


def test_ordering_does_not_change_intrinsic_run_definitions(
    experiment_fixture: ExperimentFixture,
) -> None:
    base = load_experiment(experiment_fixture.path)
    canonical = generate_run_definitions(base)
    shuffled_experiment = base.model_copy(
        update={"ordering": ExecutionOrdering(mode="shuffled", seed=99)}
    )
    shuffled_intrinsic = generate_run_definitions(shuffled_experiment)

    assert canonical == shuffled_intrinsic
    assert [run.run_id for run in canonical] != [
        run.run_id
        for run in order_run_definitions(shuffled_intrinsic, shuffled_experiment)
    ]


def test_seeded_interleaving_is_reproducible_and_round_robins_harnesses(
    experiment_fixture: ExperimentFixture,
) -> None:
    base = load_experiment(experiment_fixture.path)
    experiment = base.model_copy(
        update={"ordering": ExecutionOrdering(mode="interleaved", seed=21)}
    )

    first = expand_experiment(experiment)
    second = expand_experiment(experiment)

    assert first == second
    assert len({run.harness_id for run in first[:3]}) == 3
    assert Counter(run.harness_id for run in first) == {
        "hermes": 6,
        "opencode": 12,
        "pi": 6,
    }


def test_multiple_profiles_expand_only_with_their_own_harness(
    experiment_fixture: ExperimentFixture,
) -> None:
    experiment = load_experiment(experiment_fixture.path)
    runs = generate_run_definitions(experiment)

    opencode_profiles = {
        run.profile_id for run in runs if run.harness_id == "opencode"
    }
    assert opencode_profiles == {"opencode-default", "opencode-high"}
    assert all(
        run.harness_id == next(
            profile.harness_id
            for profile in experiment.harness_profiles
            if profile.profile_id == run.profile_id
        )
        for run in runs
    )
