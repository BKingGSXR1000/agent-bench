"""Explicit exact-tokenizer integration for captured reasoning text.

The counter is opt-in because loading a GGUF is an external operation.  Its
identity is carried with every reconstructed metric; callers must provide the
sealed model SHA-256 rather than treating a local filename as model identity.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ReasoningTokenizerError(RuntimeError):
    """The requested exact tokenizer could not produce a reproducible count."""


_TOKEN_COUNT_PATTERN = re.compile(rb"(?:token count|tokens)\s*[:=]\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class LlamaTokenizeCounter:
    executable: Path
    model: Path
    model_sha256: str
    llama_cpp_commit: str

    def __post_init__(self) -> None:
        if len(self.model_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.model_sha256):
            raise ReasoningTokenizerError("model_sha256 must be a lowercase SHA-256")
        if not self.executable.is_file() or not self.model.is_file():
            raise ReasoningTokenizerError("llama-tokenize executable or pinned model is unavailable")

    @property
    def tokenizer_identity(self) -> str:
        return f"llama-tokenize:{self.llama_cpp_commit}:no-bos-stdin-show-count-v1"

    @property
    def tokenizer_digest(self) -> str:
        return hashlib.sha256(self.executable.read_bytes()).hexdigest()

    def count(self, text: str) -> int:
        """Count with the pinned GGUF tokenizer, never from character length."""
        try:
            encoded_text = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ReasoningTokenizerError(f"reasoning text cannot be encoded as UTF-8: {exc}") from exc
        try:
            result = subprocess.run(
                [str(self.executable), "--model", str(self.model), "--stdin", "--show-count", "--no-bos"],
                input=encoded_text,
                capture_output=True,
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ReasoningTokenizerError(f"llama-tokenize did not complete: {exc}") from exc
        if result.returncode != 0:
            raise ReasoningTokenizerError("llama-tokenize returned a non-zero status")
        matches = _TOKEN_COUNT_PATTERN.findall(result.stdout)
        if len(matches) != 1:
            raise ReasoningTokenizerError("llama-tokenize did not emit one parseable token count")
        return int(matches[0])
