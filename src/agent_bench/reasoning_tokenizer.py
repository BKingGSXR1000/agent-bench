"""Explicit exact-tokenizer integration for captured reasoning text.

The counter is opt-in because loading a GGUF is an external operation.  Its
identity is carried with every reconstructed metric; callers must provide the
sealed model SHA-256 rather than treating a local filename as model identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ReasoningTokenizerError(RuntimeError):
    """The requested exact tokenizer could not produce a reproducible count."""


_TOKEN_COUNT_PATTERN = re.compile(rb"(?:token count|tokens)\s*[:=]\s*(\d+)", re.IGNORECASE)
_CACHE_SCHEMA_VERSION = "1.0.0"


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

    def identity_record(self) -> dict[str, str]:
        """The complete reproducible identity persisted in metric provenance."""
        return {
            "executable_sha256": self.tokenizer_digest,
            "llama_cpp_commit": self.llama_cpp_commit,
            "model_sha256": self.model_sha256,
            "invocation": "--model <GGUF> --stdin --show-count --no-bos",
            "tokenizer_identity": self.tokenizer_identity,
        }

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


@dataclass(frozen=True)
class ReasoningTokenCache:
    """Durable exact-count cache kept separate from sealed run evidence.

    Each cache record repeats its full identity, so a filename collision, stale
    record, or interrupted write can never be interpreted as a valid hit.
    """

    root: Path

    def count(
        self,
        text: str,
        *,
        source_evidence_identity: Mapping[str, str],
        tokenizer: LlamaTokenizeCounter,
    ) -> int:
        try:
            text_bytes = text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ReasoningTokenizerError(f"reasoning text cannot be encoded as UTF-8: {exc}") from exc
        key = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "source_evidence_identity": dict(sorted(source_evidence_identity.items())),
            "reasoning_text_sha256": hashlib.sha256(text_bytes).hexdigest(),
            "tokenizer_executable_sha256": tokenizer.tokenizer_digest,
            "model_sha256": tokenizer.model_sha256,
            "tokenizer_identity": tokenizer.tokenizer_identity,
        }
        path = self._entry_path(key)
        cached = self._read(path, key)
        if cached is not None:
            return cached
        count = tokenizer.count(text)
        self._write(path, {"schema_version": _CACHE_SCHEMA_VERSION, "key": key, "token_count": count})
        return count

    def _entry_path(self, key: dict[str, object]) -> Path:
        digest = _canonical_sha256(key)
        return self.root / digest[:2] / f"{digest}.json"

    @staticmethod
    def _read(path: Path, key: dict[str, object]) -> int | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        count = value.get("token_count") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != _CACHE_SCHEMA_VERSION
            or value.get("key") != key
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            return None
        return count

    @staticmethod
    def _write(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
