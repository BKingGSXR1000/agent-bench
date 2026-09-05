"""Read-only preflight for the pinned Qwen reasoning chat template."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from agent_bench.backend import BackendProfile, load_backend_profile


class ReasoningTemplateError(RuntimeError):
    """The pinned template cannot prove the requested reasoning contract."""


_XHIGH = (
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer."
)
_LOW = (
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration."
)


def verify_reasoning_template(profile: BackendProfile | None = None) -> dict[str, Any]:
    """Verify template branches without starting a server or model inference.

    The diagnostic's representative render has no tools or prior assistant
    turns, exactly isolating the template's reasoning branch and generation
    prefix.  The renderer mirrors that documented, bounded template path; it
    never calls a model or a live backend endpoint.
    """
    configured = profile or load_backend_profile()
    template_path = configured.chat_template.path.expanduser().resolve()
    try:
        source = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReasoningTemplateError(f"cannot read pinned chat template: {exc}") from exc
    observed_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if observed_sha != configured.chat_template.sha256:
        raise ReasoningTemplateError("pinned chat template SHA256 mismatch")
    required = (
        "reasoning_effort|default('xhigh')", "enable_thinking is undefined or enable_thinking is true",
        _XHIGH, _LOW, "<think>\\n\\n</think>", "<think>\\n",
    )
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise ReasoningTemplateError("pinned template lacks required reasoning branch: " + repr(missing))
    rendered = {
        "default": _representative_render(instruction=_XHIGH, thinking=True),
        "medium": _representative_render(instruction=None, thinking=True),
        "low": _representative_render(instruction=_LOW, thinking=True),
        "none_off": _representative_render(instruction=None, thinking=False),
    }
    checks = {
        "default": {"instruction_present": _XHIGH in rendered["default"], "open_think": rendered["default"].endswith("<think>\n")},
        "medium": {"no_extra_reasoning_instruction": _XHIGH not in rendered["medium"] and _LOW not in rendered["medium"], "open_think": rendered["medium"].endswith("<think>\n")},
        "low": {"brief_focused_instruction_present": _LOW in rendered["low"], "open_think": rendered["low"].endswith("<think>\n")},
        "none_off": {"no_reasoning_instruction": _XHIGH not in rendered["none_off"] and _LOW not in rendered["none_off"], "closed_empty_think": rendered["none_off"].endswith("<think>\n\n</think>\n\n")},
    }
    if not all(all(value.values()) for value in checks.values()):
        raise ReasoningTemplateError("reasoning template preflight checks failed")
    sha256s = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in rendered.items()}
    if len(set(sha256s.values())) != 4:
        raise ReasoningTemplateError("reasoning template renders do not differ as expected")
    return {"schema_version": "1.0.0", "kind": "reasoning-template-preflight", "diagnostic_only": True, "model_inference": False, "template_path": str(template_path), "template_sha256": observed_sha, "checks": checks, "rendered_prompt_sha256": sha256s}


def _representative_render(*, instruction: str | None, thinking: bool) -> str:
    # This is the no-tools, single-user-message path of the pinned template.
    system = "" if instruction is None else "<|im_start|>system\n" + instruction + "<|im_end|>\n"
    prefix = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    return system + "<|im_start|>user\nTemplate preflight probe.<|im_end|>\n<|im_start|>assistant\n" + prefix
