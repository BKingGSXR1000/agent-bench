"""Durable exact-token reconstruction cache coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path

from agent_bench.comparison import _RunProgress, _completed_run_total
from agent_bench.executor import ExperimentState, RunProgress
from agent_bench.reasoning_tokenizer import ReasoningTokenCache


@dataclass
class _ExactCounter:
    model_sha256: str = "a" * 64
    tokenizer_digest: str = "b" * 64
    tokenizer_identity: str = "llama-tokenize:commit-a:no-bos-stdin-show-count-v1"
    calls: list[str] = field(default_factory=list)

    def count(self, text: str) -> int:
        self.calls.append(text)
        return {"first block": 11, "changed block": 13}[text]


def _evidence() -> dict[str, str]:
    return {
        "artifact_manifest_sha256": "c" * 64,
        "normalized_events_sha256": "d" * 64,
        "raw_events_sha256": "e" * 64,
        "run_id": "fixture-run",
    }


def test_cached_exact_count_matches_uncached_and_avoids_second_invocation(tmp_path: Path) -> None:
    counter = _ExactCounter()
    cache = ReasoningTokenCache(tmp_path / "cache")

    uncached = counter.count("first block")
    first_cached = cache.count("first block", source_evidence_identity=_evidence(), tokenizer=counter)  # type: ignore[arg-type]
    second_cached = cache.count("first block", source_evidence_identity=_evidence(), tokenizer=counter)  # type: ignore[arg-type]

    assert (uncached, first_cached, second_cached) == (11, 11, 11)
    assert counter.calls == ["first block", "first block"]


def test_changed_reasoning_text_invalidates_cache(tmp_path: Path) -> None:
    counter = _ExactCounter()
    cache = ReasoningTokenCache(tmp_path / "cache")

    cache.count("first block", source_evidence_identity=_evidence(), tokenizer=counter)  # type: ignore[arg-type]
    assert cache.count("changed block", source_evidence_identity=_evidence(), tokenizer=counter) == 13  # type: ignore[arg-type]
    assert counter.calls == ["first block", "changed block"]


def test_changed_sealed_evidence_identity_invalidates_cache(tmp_path: Path) -> None:
    counter = _ExactCounter()
    cache = ReasoningTokenCache(tmp_path / "cache")

    cache.count("first block", source_evidence_identity=_evidence(), tokenizer=counter)  # type: ignore[arg-type]
    changed_evidence = {**_evidence(), "raw_events_sha256": "f" * 64}
    assert cache.count("first block", source_evidence_identity=changed_evidence, tokenizer=counter) == 11  # type: ignore[arg-type]

    assert counter.calls == ["first block", "first block"]


def test_changed_model_or_tokenizer_identity_invalidates_cache(tmp_path: Path) -> None:
    cache = ReasoningTokenCache(tmp_path / "cache")
    original = _ExactCounter()
    changed_model = _ExactCounter(model_sha256="f" * 64)
    changed_tokenizer = _ExactCounter(
        tokenizer_digest="0" * 64,
        tokenizer_identity="llama-tokenize:commit-b:no-bos-stdin-show-count-v1",
    )

    for counter in (original, changed_model, changed_tokenizer):
        assert cache.count("first block", source_evidence_identity=_evidence(), tokenizer=counter) == 11  # type: ignore[arg-type]

    assert original.calls == ["first block"]
    assert changed_model.calls == ["first block"]
    assert changed_tokenizer.calls == ["first block"]


def test_partial_cache_file_is_not_a_valid_hit(tmp_path: Path) -> None:
    counter = _ExactCounter()
    cache = ReasoningTokenCache(tmp_path / "cache")
    key = {
        "schema_version": "1.0.0",
        "source_evidence_identity": _evidence(),
        "reasoning_text_sha256": hashlib.sha256(b"first block").hexdigest(),
        "tokenizer_executable_sha256": counter.tokenizer_digest,
        "model_sha256": counter.model_sha256,
        "tokenizer_identity": counter.tokenizer_identity,
    }
    path = cache._entry_path(key)
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":"1.0.0","key":', encoding="utf-8")

    assert cache.count("first block", source_evidence_identity=_evidence(), tokenizer=counter) == 11  # type: ignore[arg-type]
    assert counter.calls == ["first block"]


def test_completed_run_progress_excludes_failed_and_pending_runs(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    state = ExperimentState(
        experiment_id="fixture", definition_digest="a" * 64, expansion_digest="b" * 64,
        ordering={}, updated_at="2026-01-01T00:00:00Z",
        runs=[
            RunProgress(run_id="completed", execution_index=1, state="completed"),
            RunProgress(run_id="failed", execution_index=2, state="failed"),
            RunProgress(run_id="pending", execution_index=3, state="pending"),
        ],
    )
    (root / "experiment-state.json").write_text(state.model_dump_json(), encoding="utf-8")
    observed: list[tuple[int, int, str]] = []
    tracker = _RunProgress(_completed_run_total([root]), lambda *values: observed.append(values))

    tracker.advance("completed")

    assert observed == [(1, 1, "completed")]
