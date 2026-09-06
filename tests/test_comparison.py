from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_bench.comparison import (
    METRICS,
    ComparisonError,
    _compatibility,
    _metric_values,
    _pairs,
    _summaries,
    build_comparison,
)
from agent_bench.reasoning_template import verify_reasoning_template


def _row(profile: str, *, prompt: str = "a" * 64, seed: int = 1001, value: int | None = 10) -> dict[str, object]:
    return {
        "harness": "hermes", "harness_version": "0.21.0", "profile": profile, "reasoning_setting": profile, "semantic_task": "entry-category",
        "prompt_sha256": prompt, "prompt_variant": "normal", "repetition": seed - 1000,
        "seed": seed, "metrics": {
            "timing.wall_time_seconds.value": value,
            "timing.llm_time_seconds.value": value,
            "tokens.input_tokens_total.value": value,
            "tokens.output_tokens_total.value": value,
            "behavior.llm_request_count.value": value,
            "behavior.tool_calls_total.value": value,
            "context.peak_context_tokens.value": value,
            "context.net_context_growth_tokens.value": value,
            "behavior.calls_before_first_edit.value": value,
            "behavior.reasoning_only_responses.value": value,
            "behavior.length_finished_responses.value": value,
            "behavior.length_finished_without_tool_call.value": value,
            "behavior.requests_before_first_model_tool_call.value": value,
            "behavior.output_tokens_before_first_model_tool_call.value": value,
            "behavior.requests_before_first_model_edit_call.value": value,
            "behavior.output_tokens_before_first_model_edit_call.value": value,
        },
    }


def _metric_dump(value: int | float | None) -> dict[str, object]:
    """Minimal metric-value projection used by read-only compatibility tests."""
    result: dict[str, object] = {}
    for path in METRICS:
        current = result
        for part in path.split(".")[:-1]:
            current = current.setdefault(part, {})  # type: ignore[assignment]
        current[path.rsplit(".", 1)[-1]] = value
    return result


def test_historical_reasoning_metrics_are_reconstructed_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An old metrics-v1 shape gains only an in-memory comparison projection."""
    from agent_bench import comparison

    historical = _metric_dump(5)
    historical.pop("reasoning")
    historical["derived"]["reasoning_to_output_ratio"]["value"] = None  # type: ignore[index]
    before = json.dumps(historical, sort_keys=True)
    artifact = tmp_path / "sealed-artifact"
    artifact.mkdir()
    marker = artifact / "immutable-evidence"
    marker.write_text("unchanged", encoding="utf-8")
    calls: list[Path] = []
    monkeypatch.setattr(
        comparison,
        "calculate_run_metrics",
        lambda path: calls.append(path) or SimpleNamespace(model_dump=lambda **_kwargs: _metric_dump(17)),
    )

    values, provenance = _metric_values(SimpleNamespace(model_dump=lambda **_kwargs: historical), artifact)

    assert calls == [artifact]
    assert values["reasoning.reasoning_tokens_total.value"] == 17
    assert values["reasoning.reasoning_time_total_seconds.value"] == 17
    assert values["derived.reasoning_to_output_ratio.value"] == 17
    assert provenance["reasoning.reasoning_tokens_total.value"] == "recalculated_from_raw_evidence"
    assert provenance["derived.reasoning_to_output_ratio.value"] == "recalculated_from_raw_evidence"
    assert json.dumps(historical, sort_keys=True) == before
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_stored_reasoning_values_remain_authoritative(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from agent_bench import comparison

    stored = _metric_dump(41)
    monkeypatch.setattr(
        comparison, "calculate_run_metrics",
        lambda _artifact: pytest.fail("complete stored reasoning metrics must not be recalculated"),
    )

    values, provenance = _metric_values(SimpleNamespace(model_dump=lambda **_kwargs: stored), tmp_path)

    assert values["reasoning.reasoning_tokens_total.value"] == 41
    assert values["derived.reasoning_to_output_ratio.value"] == 41
    assert provenance["reasoning.reasoning_tokens_total.value"] == "stored_historic_metrics"
    assert provenance["derived.reasoning_to_output_ratio.value"] == "stored_historic_metrics"


def test_unavailable_historical_reasoning_evidence_is_not_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from agent_bench import comparison

    historical = _metric_dump(5)
    historical.pop("reasoning")
    historical["derived"]["reasoning_to_output_ratio"]["value"] = None  # type: ignore[index]
    recalculated = _metric_dump(5)
    recalculated["reasoning"] = {
        path.split(".")[1]: {"value": None}
        for path in METRICS if path.startswith("reasoning.")
    }
    recalculated["derived"]["reasoning_to_output_ratio"]["value"] = None  # type: ignore[index]
    monkeypatch.setattr(
        comparison,
        "calculate_run_metrics",
        lambda _artifact: SimpleNamespace(model_dump=lambda **_kwargs: recalculated),
    )

    values, provenance = _metric_values(SimpleNamespace(model_dump=lambda **_kwargs: historical), tmp_path)

    assert values["reasoning.reasoning_tokens_before_first_tool.value"] is None
    assert values["reasoning.reasoning_time_total_seconds.value"] is None
    assert values["derived.reasoning_to_output_ratio.value"] is None
    assert provenance["reasoning.reasoning_tokens_before_first_tool.value"] == "unavailable"
    assert provenance["derived.reasoning_to_output_ratio.value"] == "unavailable"


def test_reference_pairs_have_explicit_candidate_minus_reference_signs_and_na() -> None:
    rows = [
        _row("xhigh", seed=1001, value=10), _row("low", seed=1001, value=8),
        _row("xhigh", seed=1002, value=20), _row("low", seed=1002, value=None),
        _row("medium", seed=1001, prompt="b" * 64, value=1),
    ]
    pairs = _pairs(rows, reference_profile="xhigh")
    # xhigh/low match only inside seed 1001 and seed 1002; medium has no
    # same-prompt partner and cannot become an accidental cross-prompt pair.
    assert len(pairs) == 2
    first = next(pair for pair in pairs if pair["seed"] == 1001)
    metric = first["metrics"]["timing.wall_time_seconds.value"]
    assert first["reference_profile"] == "xhigh"
    assert first["candidate_profile"] == "low"
    assert first["delta_definition"] == "candidate_minus_reference"
    assert metric == {
        "reference_value": 10, "candidate_value": 8, "absolute_delta": -2,
        "relative_delta_percent": -20.0, "direction": "faster",
    }
    assert first["metrics"]["tokens.input_tokens_total.value"]["direction"] == "fewer"
    assert first["metrics"]["context.peak_context_tokens.value"]["direction"] == "lower"
    second = next(pair for pair in pairs if pair["seed"] == 1002)
    assert second["metrics"]["timing.wall_time_seconds.value"]["direction"] == "not_available"
    overall = next(item for item in _summaries(pairs) if item["view"] == "overall" and item["metric"] == "timing.wall_time_seconds.value")
    assert overall["n_matched_pairs"] == 2
    assert overall["n_available"] == 1
    assert overall["not_available_count"] == 1
    assert overall["direction_counts"] == {"faster": 1, "not_available": 1}


def test_matching_requires_exact_prompt_sha_repetition_and_seed_without_replacement() -> None:
    reference = _row("xhigh", prompt="a" * 64, seed=1001, value=10)
    exact = _row("low", prompt="a" * 64, seed=1001, value=8)
    prompt_mismatch = _row("medium", prompt="b" * 64, seed=1001, value=7)
    seed_mismatch = _row("high", prompt="a" * 64, seed=1002, value=6)
    missing_partner = _row("minimal", prompt="a" * 64, seed=1003, value=5)
    # Keep repetition fixed so each exclusion demonstrates its own key field.
    for row in (reference, exact, prompt_mismatch, seed_mismatch, missing_partner):
        row["repetition"] = 1

    pairs = _pairs(
        [reference, exact, prompt_mismatch, seed_mismatch, missing_partner],
        reference_profile="xhigh",
    )

    assert len(pairs) == 1
    assert pairs[0]["candidate_profile"] == "low"
    assert pairs[0]["prompt_sha256"] == "a" * 64
    assert pairs[0]["seed"] == 1001
    # The cross-root controlled subject/context guard is independent of the
    # row-level stratum and rejects unlike benchmark subjects before pairing.
    identity = {
        "subject_baseline": "pocket-ledger:baseline-a", "model": "model",
        "backend": "backend", "chat_template": "template", "hardware": "hardware",
        "context_backend_settings": "settings",
    }
    subject = next(item for item in _compatibility([
        {"root": "one", "identity": identity, "rows": []},
        {"root": "two", "identity": {**identity, "subject_baseline": "other:baseline-a"}, "rows": []},
    ]) if item["dimension"] == "subject_baseline")
    assert subject["status"] == "incompatible"


def test_pair_zero_reference_and_equal_direction_are_explicit() -> None:
    zero = _pairs([_row("xhigh", value=0), _row("low", value=2)], reference_profile="xhigh")[0]
    metric = zero["metrics"]["behavior.tool_calls_total.value"]
    assert metric["absolute_delta"] == 2
    assert metric["relative_delta_percent"] is None
    assert metric["direction"] == "more"
    equal = _pairs([_row("xhigh", value=5), _row("low", value=5)], reference_profile="xhigh")[0]
    assert equal["metrics"]["timing.wall_time_seconds.value"]["direction"] == "equal"


def test_reference_mode_can_include_all_pairs_without_duplicate_reference_pairs() -> None:
    rows = [_row("xhigh", value=10), _row("low", value=8), _row("medium", value=9)]
    pairs = _pairs(rows, reference_profile="xhigh", include_all_pairs=True)
    assert {(pair["reference_profile"], pair["candidate_profile"]) for pair in pairs} == {
        ("xhigh", "low"), ("xhigh", "medium"), ("low", "medium"),
    }


def test_reasoning_template_preflight_is_read_only_and_has_four_unique_renders() -> None:
    report = verify_reasoning_template()
    assert report["diagnostic_only"] is True
    assert report["model_inference"] is False
    assert len(set(report["rendered_prompt_sha256"].values())) == 4
    assert all(all(check.values()) for check in report["checks"].values())


def test_combining_compatible_partial_roots_writes_only_new_derived_output(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent_bench import comparison
    identity = {"subject_baseline": "subject", "model": "model", "backend": "backend", "chat_template": "template", "hardware": "hardware", "context_backend_settings": "settings"}
    supplied = iter((
        {"experiment_id": "one", "root": "one", "definition_digest": "a", "completed_runs": 1, "partial": True, "identity": identity, "rows": [_row("low", value=8)]},
        {"experiment_id": "two", "root": "two", "definition_digest": "b", "completed_runs": 1, "partial": False, "identity": identity, "rows": [_row("xhigh", value=10)]},
    ))
    monkeypatch.setattr(comparison, "_read_root", lambda *_args: next(supplied))
    destination = build_comparison(
        [tmp_path / "one", tmp_path / "two"], output=tmp_path / "derived",
        reference_profile="xhigh", include_all_pairs=True,
    )
    assert (destination / "comparison.json").is_file()
    report = (destination / "comparison.json").read_text(encoding="utf-8")
    assert '"reference_profile": "xhigh"' in report
    assert "win_count" not in report and "loss_count" not in report
    assert "candidate_minus_reference" in report
    html = (destination / "report.html").read_text(encoding="utf-8")
    assert "quality or overall agent-performance wins" in html
    assert "All profile-pair summaries" in html
    assert not (tmp_path / "one").exists() and not (tmp_path / "two").exists()


def test_incompatible_roots_are_rejected_without_output(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from agent_bench import comparison
    base = {"subject_baseline": "subject", "model": "model", "backend": "backend", "chat_template": "template", "hardware": "hardware", "context_backend_settings": "settings"}
    changed = {**base, "model": "other-model"}
    supplied = iter((
        {"experiment_id": "one", "root": "one", "definition_digest": "a", "completed_runs": 1, "partial": False, "identity": base, "rows": [_row("low")]},
        {"experiment_id": "two", "root": "two", "definition_digest": "b", "completed_runs": 1, "partial": False, "identity": changed, "rows": [_row("xhigh")]},
    ))
    monkeypatch.setattr(comparison, "_read_root", lambda *_args: next(supplied))
    with pytest.raises(ComparisonError, match="model"):
        build_comparison([tmp_path / "one", tmp_path / "two"], output=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_legacy_root_without_embedded_definition_uses_verified_report_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A pre-embedded-definition root is read, never backfilled or modified."""
    from agent_bench import comparison

    fixture = Path(__file__).parent / "fixtures" / "legacy-experiment-root"
    root = tmp_path / "legacy-root"
    root.mkdir()
    (root / "experiment-state.json").write_bytes((fixture / "experiment-state.json").read_bytes())
    report = root / "report-v1"
    report.mkdir()
    (report / "presentation.json").write_bytes((fixture / "report-v1" / "presentation.json").read_bytes())
    presentation = json.loads((report / "presentation.json").read_text(encoding="utf-8"))
    state = json.loads((root / "experiment-state.json").read_text(encoding="utf-8"))
    monkeypatch.chdir(tmp_path)  # There is intentionally no checked-in mapping here.
    monkeypatch.setattr(
        comparison, "verify_report",
        lambda _report: {"experiment_id": state["experiment_id"], "definition_digest": state["definition_digest"]},
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        comparison, "_read_legacy_root",
        lambda seen_root, seen_state, seen_presentation: captured.update(
            root=seen_root, state=seen_state, presentation=seen_presentation,
        ) or {"legacy": True},
    )

    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert comparison._read_root(root, None) == {"legacy": True}
    assert captured["root"] == root
    assert captured["presentation"] == presentation
    assert {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    assert not (root / "definition.yaml").exists()


def test_definition_discovery_matches_immutable_identity_not_legacy_root_filename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from agent_bench import comparison
    from agent_bench.executor import ExperimentState

    state = ExperimentState.model_validate({
        "experiment_id": "pocket-ledger-v1-qwen38", "definition_digest": "expected",
        "expansion_digest": "expansion", "ordering": {}, "runs": [],
        "updated_at": "2026-01-01T00:00:00Z",
    })
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    legacy_named_yaml = experiments / "pocket-ledger-v1.yaml"
    legacy_named_yaml.write_text("fixture: ignored by mocked loader\n", encoding="utf-8")
    resolved = SimpleNamespace(experiment_id="pocket-ledger-v1-qwen38", definition_digest="expected")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(comparison, "load_experiment", lambda path: resolved if path == legacy_named_yaml else pytest.fail(str(path)))

    assert comparison._resolve_definition(state, None) is resolved


def test_unavailable_identity_is_reported_not_assumed_compatible() -> None:
    identity = {
        "subject_baseline": "subject", "model": "model", "backend": "backend",
        "chat_template": "template", "hardware": "hardware", "context_backend_settings": "settings",
    }
    report = _compatibility([
        {"root": "old", "identity": {**identity, "hardware": None}, "rows": []},
        {"root": "new", "identity": identity, "rows": []},
    ])
    hardware = next(item for item in report if item["dimension"] == "hardware")
    assert hardware == {
        "dimension": "hardware", "values": ["hardware"],
        "missing_from_roots": ["old"], "status": "unavailable",
    }
