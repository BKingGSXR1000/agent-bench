"""Persisted configuration models for Agent Bench experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    field_validator,
    model_validator,
)

SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
SUPPORTED_HARNESS_IDS = frozenset({"opencode", "pi", "hermes"})

Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SchemaVersion = Literal["1.0.0"]
IdentityVersion = Literal["1.0.0", "2.0.0"]
HarnessId = Literal["opencode", "pi", "hermes"]
JsonMapping = dict[str, JsonValue]


def canonical_sha256(value: object) -> str:
    """Return a SHA256 digest of a canonical JSON-compatible value."""
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PersistedModel(BaseModel):
    """Base for immutable persisted definitions with strict field names."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: SchemaVersion = SCHEMA_VERSION

    def _definition_identity(self) -> object:
        return self.model_dump(
            mode="json",
            exclude={"definition_digest"},
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def definition_digest(self) -> str:
        """Digest the definition using deterministic JSON serialization."""
        return canonical_sha256(self._definition_identity())


class ModelIdentity(PersistedModel):
    """Expected identity of the fixed benchmark GGUF."""

    model_identity_id: Identifier
    name: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    path: Path
    filename: str = Field(min_length=1)
    file_size_bytes: int | None = Field(default=None, ge=0)
    sha256: Sha256
    gguf_metadata: JsonMapping = Field(default_factory=dict)
    identity_source: str = Field(default="configured", min_length=1)

    @field_validator("name")
    @classmethod
    def validate_benchmark_model(cls, value: str) -> str:
        if value != "Qwen 3.8 27B":
            raise ValueError("benchmark v1 model name must be 'Qwen 3.8 27B'")
        return value

    @field_validator("quantization")
    @classmethod
    def validate_q4_quantization(cls, value: str) -> str:
        if not value.upper().startswith("Q4"):
            raise ValueError("benchmark v1 quantization must be a Q4 GGUF variant")
        return value

    def _definition_identity(self) -> object:
        data = super()._definition_identity()
        assert isinstance(data, dict)
        data.pop("path", None)
        return data


class BackendIdentity(PersistedModel):
    """Configured identity of the fixed llama.cpp backend."""

    backend_identity_id: Identifier
    implementation: Literal["llama.cpp"] = "llama.cpp"
    executable: Path
    executable_sha256: Sha256 | None = None
    version: str = Field(min_length=1)
    commit: str | None = None
    build_metadata: JsonMapping = Field(default_factory=dict)
    invocation_template_version: str = Field(default="1.0.0", min_length=1)

    def _definition_identity(self) -> object:
        """The executable location is host evidence, not benchmark identity."""
        data = super()._definition_identity()
        assert isinstance(data, dict)
        data.pop("executable", None)
        return data


class HardwareIdentity(PersistedModel):
    """Expected fixed hardware profile, not a dynamic observation."""

    hardware_identity_id: Identifier
    name: str = Field(min_length=1)
    cpu_architecture: str = Field(min_length=1)
    gpu_model: str = Field(min_length=1)
    gpu_count: int = Field(ge=1)
    gpu_uuids: tuple[str, ...] = ()
    memory_bytes: int | None = Field(default=None, ge=1)
    vram_bytes: int | None = Field(default=None, ge=1)
    preconditions: JsonMapping = Field(default_factory=dict)


class GenerationConfiguration(PersistedModel):
    """Fixed per-request generation configuration."""

    temperature: float = Field(ge=0)
    top_p: float = Field(ge=0, le=1)
    top_k: int = Field(ge=0)
    min_p: float = Field(ge=0, le=1)
    seed: int | None = None
    seed_control: Literal["controlled", "uncontrollable"] = "controlled"
    max_output_tokens: int = Field(ge=1)
    context_size: int = Field(ge=1)
    stop_sequences: tuple[str, ...] = ()
    reasoning: JsonMapping = Field(default_factory=dict)
    other_parameters: JsonMapping = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_seed_control(self) -> GenerationConfiguration:
        if self.seed_control == "uncontrollable" and self.seed is not None:
            raise ValueError("uncontrollable generation cannot define a seed")
        return self


class WarmupPolicy(PersistedModel):
    """Configured warmup identity; execution belongs to a later milestone."""

    mode: Literal["disabled", "enabled"] = "disabled"
    request: str | None = None
    repetitions: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_enabled_policy(self) -> WarmupPolicy:
        if self.mode == "enabled" and (self.request is None or self.repetitions < 1):
            raise ValueError(
                "enabled warmup requires a request and at least one repetition"
            )
        if self.mode == "disabled" and (
            self.request is not None or self.repetitions != 0
        ):
            raise ValueError(
                "disabled warmup cannot define a request or repetitions"
            )
        return self


class FixedEnvironment(PersistedModel):
    """The non-varying model, backend, hardware, and inference settings."""

    fixed_environment_id: Identifier
    model: ModelIdentity
    backend: BackendIdentity
    hardware: HardwareIdentity
    server_parameters: JsonMapping
    generation: GenerationConfiguration
    restart_policy: Literal["per_run"] = "per_run"
    readiness_policy: str = Field(min_length=1)
    warmup: WarmupPolicy = Field(default_factory=WarmupPolicy)
    environment_allowlist: tuple[str, ...] = ()

    def _definition_identity(self) -> object:
        """Compose fixed-environment identity from semantic/content identities."""
        data = super()._definition_identity()
        assert isinstance(data, dict)
        data["model"] = self.model._definition_identity()
        data["backend"] = self.backend._definition_identity()
        return data


class PromptDefinition(PersistedModel):
    """A byte-exact UTF-8 prompt loaded from a separate file."""

    prompt_id: Identifier
    semantic_task_id: Identifier
    variant_label: Identifier
    path: Path
    encoding: Literal["utf-8"] = "utf-8"
    content: str
    byte_length: int = Field(ge=0)
    sha256: Sha256
    metadata: JsonMapping = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content_identity(self) -> PromptDefinition:
        try:
            content_bytes = self.content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("prompt content is not valid UTF-8") from exc
        if len(content_bytes) != self.byte_length:
            raise ValueError("prompt byte_length does not match its exact content")
        if hashlib.sha256(content_bytes).hexdigest() != self.sha256:
            raise ValueError("prompt sha256 does not match its exact content")
        return self

    def _definition_identity(self) -> object:
        return {
            "schema_version": self.schema_version,
            "prompt_id": self.prompt_id,
            "semantic_task_id": self.semantic_task_id,
            "variant_label": self.variant_label,
            "encoding": self.encoding,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }


class HarnessDefinition(PersistedModel):
    """Identity of a supported harness release without adapter behavior."""

    harness_id: HarnessId
    display_name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    executable: Path
    upstream_project: str = Field(min_length=1)
    supported_raw_capture_sources: tuple[str, ...] = ()

    def _definition_identity(self) -> object:
        """A pinned release identity, rather than its local executable path."""
        data = super()._definition_identity()
        assert isinstance(data, dict)
        data.pop("executable", None)
        return data


class HarnessProfile(PersistedModel):
    """Clean, versioned settings for one harness identity."""

    profile_id: Identifier
    harness_id: HarnessId
    profile_version: str = Field(min_length=1)
    kind: Literal["controlled_default", "benchmark_specific"]
    upstream_defaults_source: str = Field(min_length=1)
    deviations: tuple[str, ...] = ()
    settings: JsonMapping = Field(default_factory=dict)
    bundle_path: Path | None = None
    bundle_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_bundle_identity(self) -> HarnessProfile:
        if (self.bundle_path is None) != (self.bundle_sha256 is None):
            raise ValueError("bundle_path and bundle_sha256 must be supplied together")
        return self

    def _definition_identity(self) -> object:
        """Profile bytes and semantics identify a profile; its checkout path does not."""
        data = super()._definition_identity()
        assert isinstance(data, dict)
        data.pop("bundle_path", None)
        return data


class ExecutionOrdering(PersistedModel):
    """Ordering applied after intrinsic matrix identities are generated."""

    mode: Literal["canonical", "shuffled", "interleaved"] = "canonical"
    seed: int | None = None

    @model_validator(mode="after")
    def validate_seed(self) -> ExecutionOrdering:
        if self.mode == "canonical" and self.seed is not None:
            raise ValueError("canonical ordering must not define a seed")
        if self.mode != "canonical" and self.seed is None:
            raise ValueError(f"{self.mode} ordering requires a seed")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def algorithm(self) -> str:
        if self.mode == "canonical":
            return "lexicographic-v1"
        if self.mode == "shuffled":
            return "seeded-sha256-sort-v1"
        return "seeded-harness-round-robin-v1"


class RunLimits(PersistedModel):
    """Generic task limits that apply independently of a harness."""

    wall_timeout_seconds: float = Field(default=300.0, gt=0)


class PortableBaselineIdentity(PersistedModel):
    """Versioned, path-independent identity of a frozen benchmark subject."""

    subject_id: Identifier
    subject_version: str = Field(min_length=1)
    baseline_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    baseline_tree: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    baseline_bundle_sha256: Sha256


def _portable_baseline_payload(identity: PortableBaselineIdentity) -> dict[str, object]:
    """Return only portable baseline fields, never a materialization path."""
    return identity.model_dump(mode="json", exclude={"definition_digest"})


def _contains_absolute_path_string(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith("/")
    if isinstance(value, dict):
        return any(_contains_absolute_path_string(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_absolute_path_string(item) for item in value)
    return False


class ExperimentDefinition(PersistedModel):
    """One validated benchmark-v1 experiment matrix definition."""

    experiment_id: Identifier
    name: str = Field(min_length=1)
    description: str | None = None
    created_at: datetime
    # 1.0.0 retains M1's historical identity algorithm.  2.0.0 is mandatory
    # for published/future matrices and excludes host-local paths.
    identity_version: IdentityVersion = "1.0.0"
    baseline_repository: Path
    baseline_revision: str = Field(min_length=1)
    portable_baseline: PortableBaselineIdentity | None = None
    fixed_environment: FixedEnvironment
    harnesses: tuple[HarnessDefinition, ...] = Field(min_length=1)
    harness_profiles: tuple[HarnessProfile, ...] = Field(min_length=1)
    prompts: tuple[PromptDefinition, ...] = Field(min_length=1)
    repetitions: int = Field(ge=1)
    ordering: ExecutionOrdering = Field(default_factory=ExecutionOrdering)
    run_limits: RunLimits = Field(default_factory=RunLimits)

    @field_validator("created_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_relationships(self) -> ExperimentDefinition:
        if self.identity_version == "2.0.0" and self.portable_baseline is None:
            raise ValueError("identity_version 2.0.0 requires portable_baseline")
        if self.identity_version == "2.0.0":
            # Typed paths are excluded in favour of content identities. Do not
            # permit an untyped settings map to reintroduce host locality.
            values = [("fixed_environment.server_parameters", self.fixed_environment.server_parameters)]
            values.extend((f"harness_profile[{profile.profile_id}].settings", profile.settings) for profile in self.harness_profiles)
            for label, value in values:
                if _contains_absolute_path_string(value):
                    raise ValueError(f"identity_version 2.0.0 forbids local path strings in {label}")
        self._require_unique(
            (harness.harness_id for harness in self.harnesses), "harness_id"
        )
        self._require_unique(
            (profile.profile_id for profile in self.harness_profiles), "profile_id"
        )
        self._require_unique(
            (prompt.prompt_id for prompt in self.prompts), "prompt_id"
        )

        harness_ids = {harness.harness_id for harness in self.harnesses}
        profile_harness_ids = {
            profile.harness_id for profile in self.harness_profiles
        }
        unknown = profile_harness_ids - harness_ids
        if unknown:
            raise ValueError(
                "harness profiles reference undefined harnesses: "
                + ", ".join(sorted(unknown))
            )
        missing = harness_ids - profile_harness_ids
        if missing:
            raise ValueError(
                "every harness requires at least one profile; missing profiles for: "
                + ", ".join(sorted(missing))
            )
        return self

    def _definition_identity(self) -> object:
        data = super()._definition_identity()
        assert isinstance(data, dict)
        if self.identity_version == "2.0.0":
            data.pop("baseline_repository", None)
            data.pop("baseline_revision", None)
            # Nested model_dump() deliberately preserves raw local paths for
            # forensic manifests.  Definition v2 instead composes identities.
            data["fixed_environment"] = {
                "fixed_environment_id": self.fixed_environment.fixed_environment_id,
                "definition_digest": self.fixed_environment.definition_digest,
            }
            data["harnesses"] = [
                {"harness_id": item.harness_id, "definition_digest": item.definition_digest}
                for item in sorted(self.harnesses, key=lambda item: item.harness_id)
            ]
            data["harness_profiles"] = [
                {"profile_id": item.profile_id, "definition_digest": item.definition_digest}
                for item in sorted(self.harness_profiles, key=lambda item: item.profile_id)
            ]
            data["prompts"] = [item._definition_identity() for item in sorted(self.prompts, key=lambda item: item.prompt_id)]
        return data

    @staticmethod
    def _require_unique(values: Iterable[str], label: str) -> None:
        items = list(values)
        duplicates = sorted({item for item in items if items.count(item) > 1})
        if duplicates:
            raise ValueError(f"duplicate {label} values: {', '.join(duplicates)}")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matrix_digest(self) -> str:
        """Digest intrinsic matrix inputs while excluding execution ordering."""
        harnesses = sorted(
            (
                {
                    "harness_id": harness.harness_id,
                    "definition_digest": harness.definition_digest,
                }
                for harness in self.harnesses
            ),
            key=lambda item: item["harness_id"],
        )
        profiles = sorted(
            (
                {
                    "profile_id": profile.profile_id,
                    "harness_id": profile.harness_id,
                    "definition_digest": profile.definition_digest,
                }
                for profile in self.harness_profiles
            ),
            key=lambda item: (item["harness_id"], item["profile_id"]),
        )
        prompts = sorted(
            (
                {
                    "prompt_id": prompt.prompt_id,
                    "semantic_task_id": prompt.semantic_task_id,
                    "definition_digest": prompt.definition_digest,
                    "sha256": prompt.sha256,
                }
                for prompt in self.prompts
            ),
            key=lambda item: item["prompt_id"],
        )
        payload: dict[str, object] = {
                "schema_version": self.schema_version,
                "identity_version": self.identity_version,
                "experiment_id": self.experiment_id,
                "fixed_environment_id": self.fixed_environment.fixed_environment_id,
                "fixed_environment_digest": self.fixed_environment.definition_digest,
                "harnesses": harnesses,
                "harness_profiles": profiles,
                "prompts": prompts,
                "repetitions": self.repetitions,
                "run_limits": self.run_limits.model_dump(
                    mode="json", exclude={"definition_digest"}
                ),
            }
        if self.identity_version == "2.0.0":
            assert self.portable_baseline is not None
            payload["portable_baseline"] = _portable_baseline_payload(
                self.portable_baseline
            )
        else:
            # This preserves the identity of sealed M1-era definitions.
            payload["baseline_repository"] = str(self.baseline_repository)
            payload["baseline_revision"] = self.baseline_revision
        return canonical_sha256(payload)


class RunDefinition(PersistedModel):
    """Intrinsic identity of one harness/profile/prompt/repetition matrix row."""

    run_id: Identifier
    experiment_id: Identifier
    experiment_matrix_digest: Sha256
    identity_version: IdentityVersion = "1.0.0"
    matrix_index: int = Field(ge=1)
    baseline_repository: Path
    baseline_revision: str = Field(min_length=1)
    portable_baseline: PortableBaselineIdentity | None = None
    fixed_environment_id: Identifier
    fixed_environment_digest: Sha256
    generation_seed: int | None
    generation_seed_control: Literal["controlled", "uncontrollable"]
    harness_id: HarnessId
    harness_definition_digest: Sha256
    profile_id: Identifier
    profile_definition_digest: Sha256
    prompt_id: Identifier
    prompt_definition_digest: Sha256
    prompt_sha256: Sha256
    semantic_task_id: Identifier
    repetition_index: int = Field(ge=1)
    limits: RunLimits

    @model_validator(mode="after")
    def validate_portable_identity(self) -> RunDefinition:
        if self.identity_version == "2.0.0" and self.portable_baseline is None:
            raise ValueError("identity_version 2.0.0 requires portable_baseline")
        return self

    def _definition_identity(self) -> object:
        data = super()._definition_identity()
        assert isinstance(data, dict)
        if self.identity_version == "2.0.0":
            data.pop("baseline_repository", None)
            data.pop("baseline_revision", None)
        return data
