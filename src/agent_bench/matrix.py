"""Deterministic experiment-matrix generation and execution ordering."""

from __future__ import annotations

import hashlib
import re
from collections import deque

from agent_bench.backend import seed_for_repetition
from agent_bench.models import (
    ExperimentDefinition,
    HarnessDefinition,
    HarnessProfile,
    PromptDefinition,
    RunDefinition,
    canonical_sha256,
)

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def generate_run_definitions(
    experiment: ExperimentDefinition,
) -> tuple[RunDefinition, ...]:
    """Generate intrinsic run definitions in canonical matrix order."""
    harnesses = {harness.harness_id: harness for harness in experiment.harnesses}
    profiles = sorted(
        experiment.harness_profiles,
        key=lambda profile: (profile.harness_id, profile.profile_id),
    )
    prompts = sorted(experiment.prompts, key=lambda prompt: prompt.prompt_id)

    runs: list[RunDefinition] = []
    matrix_index = 0
    for profile in profiles:
        harness = harnesses[profile.harness_id]
        for prompt in prompts:
            for repetition_index in experiment.effective_repetition_indices:
                matrix_index += 1
                runs.append(
                    _make_run_definition(
                        experiment=experiment,
                        harness=harness,
                        profile=profile,
                        prompt=prompt,
                        repetition_index=repetition_index,
                        matrix_index=matrix_index,
                    )
                )

    run_ids = [run.run_id for run in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("deterministic run ID collision detected")
    return tuple(runs)


def order_run_definitions(
    runs: tuple[RunDefinition, ...],
    experiment: ExperimentDefinition,
) -> tuple[RunDefinition, ...]:
    """Apply execution ordering without changing any run definition."""
    ordering = experiment.ordering
    if ordering.mode == "canonical":
        return runs

    if ordering.seed is None:
        raise ValueError(f"{ordering.mode} ordering requires a seed")
    if ordering.mode == "shuffled":
        return tuple(
            sorted(
                runs,
                key=lambda run: (_seeded_key(ordering.seed, run.run_id), run.run_id),
            )
        )

    buckets: dict[str, deque[RunDefinition]] = {}
    for run in runs:
        buckets.setdefault(run.harness_id, deque()).append(run)
    for harness_id, bucket in buckets.items():
        buckets[harness_id] = deque(
            sorted(
                bucket,
                key=lambda run: (_seeded_key(ordering.seed, run.run_id), run.run_id),
            )
        )
    harness_order = sorted(
        buckets,
        key=lambda harness_id: (
            _seeded_key(ordering.seed, f"harness:{harness_id}"),
            harness_id,
        ),
    )

    interleaved: list[RunDefinition] = []
    while any(buckets.values()):
        for harness_id in harness_order:
            if buckets[harness_id]:
                interleaved.append(buckets[harness_id].popleft())
    return tuple(interleaved)


def expand_experiment(
    experiment: ExperimentDefinition,
) -> tuple[RunDefinition, ...]:
    """Generate runs and return them in configured execution order."""
    canonical_runs = generate_run_definitions(experiment)
    return order_run_definitions(canonical_runs, experiment)


def _make_run_definition(
    *,
    experiment: ExperimentDefinition,
    harness: HarnessDefinition,
    profile: HarnessProfile,
    prompt: PromptDefinition,
    repetition_index: int,
    matrix_index: int,
) -> RunDefinition:
    intrinsic_identity = {
        "schema_version": "1.0.0",
        "identity_version": experiment.identity_version,
        "experiment_id": experiment.experiment_id,
        "experiment_matrix_digest": experiment.matrix_digest,
        "fixed_environment_id": experiment.fixed_environment.fixed_environment_id,
        "fixed_environment_digest": experiment.fixed_environment.definition_digest,
        "generation_seed": (
            seed_for_repetition(repetition_index)
            if experiment.fixed_environment.generation.seed_control == "controlled"
            else None
        ),
        "generation_seed_control": (
            experiment.fixed_environment.generation.seed_control
        ),
        "harness_id": harness.harness_id,
        "harness_definition_digest": harness.definition_digest,
        "profile_id": profile.profile_id,
        "profile_definition_digest": profile.definition_digest,
        "prompt_id": prompt.prompt_id,
        "prompt_definition_digest": prompt.definition_digest,
        "prompt_sha256": prompt.sha256,
        "semantic_task_id": prompt.semantic_task_id,
        "repetition_index": repetition_index,
        "limits": experiment.run_limits.model_dump(
            mode="json", exclude={"definition_digest"}
        ),
    }
    if prompt.functional_scenario is not None:
        intrinsic_identity["functional_scenario"] = prompt.functional_scenario._definition_identity()
    if experiment.identity_version == "2.0.0":
        # The checkout location is intentionally operational only.  A fresh
        # clone may materialize this same bundle anywhere and retain run IDs.
        assert experiment.portable_baseline is not None
        intrinsic_identity["portable_baseline"] = experiment.portable_baseline.model_dump(
            mode="json", exclude={"definition_digest"}
        )
    else:
        # Preserve M1's historical, path-bearing identity for sealed evidence.
        intrinsic_identity["baseline_repository"] = str(experiment.baseline_repository)
        intrinsic_identity["baseline_revision"] = experiment.baseline_revision
    identity_digest = canonical_sha256(intrinsic_identity)
    run_id = (
        f"{_slug(harness.harness_id)}-{_slug(profile.profile_id)}-"
        f"{_slug(prompt.prompt_id)}-r{repetition_index:03d}-{identity_digest[:24]}"
    )
    run_payload = dict(intrinsic_identity)
    # The resolved scenario path is required by the executor, but never enters
    # the portable run ID: ``intrinsic_identity`` above carries only its pinned
    # digest/identity payload.
    if prompt.functional_scenario is not None:
        run_payload["functional_scenario"] = prompt.functional_scenario
    # These values are required to execute a run locally, but v2 intentionally
    # leaves them outside the intrinsic ID/digest.
    if experiment.identity_version == "2.0.0":
        run_payload["baseline_repository"] = str(experiment.baseline_repository)
        run_payload["baseline_revision"] = experiment.baseline_revision
    return RunDefinition(
        run_id=run_id,
        matrix_index=matrix_index,
        **run_payload,
    )


def _seeded_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-._")
    return slug[:32] or "item"
