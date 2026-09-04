from __future__ import annotations

from agent_bench.context_analysis import _classify, _contains_exact_text


def test_request_purpose_uses_only_exact_prompt_or_explicit_title_markers() -> None:
    prompt = b"Change README.md exactly.\n"
    assert _classify({"messages": [{"role": "user", "content": "Change README.md exactly."}]}, prompt) == (
        "task", "preserved prompt with terminal line ending trimmed by harness transport occurs in request messages",
    )
    assert _classify({"messages": [{"role": "system", "content": "You name chat sessions."}]}, prompt)[0] == "title"
    assert _classify({"messages": [{"role": "user", "content": "Do a similar README task"}]}, prompt)[0] == "other_internal"


def test_exact_prompt_search_does_not_perform_substring_matching() -> None:
    assert _contains_exact_text({"content": "exact"}, "exact")
    assert not _contains_exact_text({"content": "not-exact"}, "exact")
