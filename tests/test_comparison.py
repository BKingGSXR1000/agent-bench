from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_bench.comparison import ComparisonError, _compatibility, _pairs, _summaries, build_comparison
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
