"""Byte-safe exact-tokenizer regression coverage."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_bench.reasoning_tokenizer import LlamaTokenizeCounter, ReasoningTokenizerError


def _counter(tmp_path: Path) -> LlamaTokenizeCounter:
    executable = tmp_path / "llama-tokenize"
    model = tmp_path / "model.gguf"
    executable.write_bytes(b"fixture executable")
    model.write_bytes(b"fixture model")
    return LlamaTokenizeCounter(executable, model, "a" * 64, "dc72703")


def test_count_parses_one_ascii_count_from_binary_stdout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    counter = _counter(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        assert command == [
            str(counter.executable), "--model", str(counter.model), "--stdin", "--show-count", "--no-bos",
        ]
        assert kwargs == {"input": b"small reasoning", "capture_output": True, "check": False, "timeout": 120}
        return subprocess.CompletedProcess(command, 0, stdout=b"token count: 17\n", stderr=b"")

    monkeypatch.setattr("agent_bench.reasoning_tokenizer.subprocess.run", fake_run)

    assert counter.count("small reasoning") == 17


def test_count_ignores_invalid_utf8_bytes_around_ascii_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    counter = _counter(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=b"\xfftokens = 19\xfe", stderr=b"\x80")

    monkeypatch.setattr("agent_bench.reasoning_tokenizer.subprocess.run", fake_run)

    assert counter.count("small reasoning") == 19


@pytest.mark.parametrize("stdout", [b"no count here", b"token count: 3; tokens = 4"])
def test_count_rejects_missing_or_multiple_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stdout: bytes,
) -> None:
    counter = _counter(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr("agent_bench.reasoning_tokenizer.subprocess.run", fake_run)

    with pytest.raises(ReasoningTokenizerError, match="one parseable token count"):
        counter.count("small reasoning")


def test_count_rejects_nonzero_tokenizer_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    counter = _counter(tmp_path)

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 1, stdout=b"token count: 17", stderr=b"failure")

    monkeypatch.setattr("agent_bench.reasoning_tokenizer.subprocess.run", fake_run)

    with pytest.raises(ReasoningTokenizerError, match="non-zero status"):
        counter.count("small reasoning")


def test_count_rejects_unencodable_reasoning_text(tmp_path: Path) -> None:
    with pytest.raises(ReasoningTokenizerError, match="cannot be encoded as UTF-8"):
        _counter(tmp_path).count("lone surrogate: \ud800")
