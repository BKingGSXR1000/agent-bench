from __future__ import annotations

from typer.testing import CliRunner
from types import SimpleNamespace

from agent_bench import cli
from agent_bench.cli import app
from agent_bench import toolchains
from agent_bench import bootstrap
from agent_bench.hermes import _resolve_toolchain_path as hermes_path
from agent_bench.pi import _resolve_toolchain_path as pi_path


def test_toolchain_verify_reports_each_required_boundary(monkeypatch: object) -> None:
    monkeypatch.setattr(toolchains, "_verify_opencode", lambda: "opencode")  # type: ignore[attr-defined]
    monkeypatch.setattr(toolchains, "_verify_pi", lambda: "pi")  # type: ignore[attr-defined]
    monkeypatch.setattr(toolchains, "_verify_node", lambda: "node")  # type: ignore[attr-defined]
    monkeypatch.setattr(toolchains, "_verify_hermes", lambda: "hermes")  # type: ignore[attr-defined]
    monkeypatch.setattr(toolchains, "_verify_backend", lambda: "backend")  # type: ignore[attr-defined]
    monkeypatch.setattr(toolchains, "_verify_model", lambda: "model")  # type: ignore[attr-defined]
    report = toolchains.verify_toolchains()
    assert {value["status"] for value in report.values()} == {"OK"}
    assert set(report) == {"OpenCode 1.18.25", "Pi 0.84.4", "Node 26.8.1", "Hermes 0.21.0", "llama.cpp", "Qwen GGUF"}


def test_toolchain_verify_reports_missing_or_drifted_payload(monkeypatch: object) -> None:
    monkeypatch.setattr(toolchains, "_verify_opencode", lambda: (_ for _ in ()).throw(toolchains.ToolchainVerificationError("missing")))  # type: ignore[attr-defined]
    report = toolchains.verify_toolchains()
    assert report["OpenCode 1.18.25"]["status"] == "MISSING_OR_DRIFTED"
    assert "missing" in report["OpenCode 1.18.25"]["detail"]


def test_toolchains_cli_is_machine_readable(monkeypatch: object) -> None:
    monkeypatch.setattr("agent_bench.cli.verify_toolchains", lambda: {"Pi 0.84.4": {"status": "OK", "detail": "pinned"}})
    result = CliRunner().invoke(app, ["toolchains", "verify", "--json-output"])
    assert result.exit_code == 0
    assert '"status": "OK"' in result.output


def test_toolchains_install_reports_explicit_manual_components(monkeypatch: object) -> None:
    monkeypatch.setattr("agent_bench.cli.install_toolchains", lambda components, include_model, model_destination: {"hermes": "MANUAL_REQUIRED_SEE_DOCS"})
    result = CliRunner().invoke(app, ["toolchains", "install", "--component", "hermes"])
    assert result.exit_code == 0
    assert "MANUAL_REQUIRED_SEE_DOCS" in result.output


def test_bootstrap_rejects_unknown_component() -> None:
    import pytest
    with pytest.raises(bootstrap.BootstrapError, match="unknown component"):
        bootstrap.install_toolchains(("not-a-toolchain",))


def test_bootstrap_model_is_never_selected_without_explicit_option(monkeypatch: object) -> None:
    calls: list[str] = []
    monkeypatch.setattr(bootstrap, "_install_opencode", lambda: calls.append("opencode"))
    monkeypatch.setattr(bootstrap, "_install_node", lambda: calls.append("node"))
    result = bootstrap.install_toolchains(("qwen",), include_model=False)
    assert result == {"qwen": "SKIPPED_REQUIRES_INCLUDE_MODEL"}
    assert calls == []


def test_legacy_local_manifest_path_maps_only_to_current_benchmark_layout() -> None:
    old = "/home/someone/agent-bench/toolchains/pi/0.84.4/node_modules/.bin/pi"
    resolved = pi_path(old)
    assert str(resolved).endswith("toolchains/pi/0.84.4/node_modules/.bin/pi")
    assert "/home/someone" not in str(resolved)
    assert hermes_path("/home/someone/agent-bench/toolchains/hermes/0.21.0/venv/bin/hermes").name == "hermes"


def test_global_backend_preflight_is_recorded_before_dispatch(tmp_path, monkeypatch: object) -> None:
    report = SimpleNamespace(passed=False, model_dump=lambda mode: {"passed": False, "checks": []})
    monkeypatch.setattr(cli, "load_backend_profile", lambda: object())  # type: ignore[attr-defined]
    monkeypatch.setattr(cli, "preflight_backend", lambda profile, paths, run_seed: report)  # type: ignore[attr-defined]
    observed = cli._experiment_backend_preflight(tmp_path / "out")
    assert observed is report
    assert '"passed":false' in (tmp_path / "out" / "global-preflight.json").read_text(encoding="utf-8")
