from __future__ import annotations

import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_bench.backend import (
    BackendLifecycleError,
    BackendStartFailed,
    BackendProfile,
    BackendRunPaths,
    CommandResult,
    GpuObservation,
    GpuPreconditionPolicy,
    GpuProcessObservation,
    PinnedFile,
    SamplingBaseline,
    ServerConfiguration,
    evaluate_gpu_preconditions,
    is_port_free,
    load_backend_profile,
    observe_backend_endpoint,
    parse_gpu_inventory,
    parse_gpu_processes,
    preflight_backend,
    resolve_backend_invocation,
    seed_for_repetition,
    start_owned_backend,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(tmp_path: Path) -> BackendProfile:
    model = tmp_path / "model.gguf"
    template = tmp_path / "template.jinja"
    executable = tmp_path / "llama-server"
    repository = tmp_path / "llama.cpp"
    library = repository / "build" / "bin" / "libllama.so"
    model.write_bytes(b"small fixture model")
    template.write_text("fixture {{ messages }}\n", encoding="utf-8")
    executable.write_bytes(b"fixture executable")
    executable.chmod(0o755)
    repository.mkdir()
    library.parent.mkdir(parents=True)
    library.write_bytes(b"fixture library")
    return BackendProfile(
        profile_id="fixture-backend-v1",
        model= PinnedFile(path=model, size_bytes=model.stat().st_size, sha256=_sha(model)),
        chat_template=PinnedFile(path=template, size_bytes=template.stat().st_size, sha256=_sha(template)),
        executable=PinnedFile(path=executable, size_bytes=executable.stat().st_size, sha256=_sha(executable)),
        llama_cpp_repository=repository,
        llama_cpp_commit="a" * 40,
        version_output_contains=("build 10517", "commit fixture"),
        local_libraries=(PinnedFile(path=library, size_bytes=library.stat().st_size, sha256=_sha(library)),),
        gpu=GpuPreconditionPolicy(expected_uuid="GPU-RTX3090"),
        server=ServerConfiguration(port=18080, proxy_port=18081),
        sampling=SamplingBaseline(),
        readiness_timeout_seconds=0.02,
        shutdown_grace_seconds=0.01,
    )


def _paths(tmp_path: Path) -> BackendRunPaths:
    root = tmp_path / "runtime"
    values = {}
    for name in ("home", "xdg_config_home", "xdg_cache_home", "xdg_data_home", "xdg_state_home"):
        path = root / name
        path.mkdir(parents=True)
        values[name] = path
    return BackendRunPaths(**values)


def _gpus(*, used: int = 500) -> tuple[tuple[GpuObservation, ...], tuple[GpuProcessObservation, ...]]:
    return (
        (
            GpuObservation(
                index=0,
                uuid="GPU-RTX3090",
                name="NVIDIA GeForce RTX 3090",
                memory_total_mib=24576,
                memory_used_mib=used,
                memory_free_mib=24576 - used,
                utilization_percent=0,
                temperature_celsius=40,
            ),
            GpuObservation(
                index=1,
                uuid="GPU-V100",
                name="Tesla V100-PCIE-32GB",
                memory_total_mib=32768,
                memory_used_mib=20000,
                memory_free_mib=12768,
                utilization_percent=95,
                temperature_celsius=70,
            ),
        ),
        (),
    )


def _command(argv: tuple[str, ...], env: object, cwd: object) -> CommandResult:
    if "--version" in argv:
        return CommandResult(0, "version build 10517 commit fixture\n", "")
    if "rev-parse" in argv:
        return CommandResult(0, "a" * 40 + "\n", "")
    if "status" in argv:
        return CommandResult(0, "", "")
    if argv[0] == "ldd":
        assert isinstance(cwd, Path)
        library = cwd / "build" / "bin" / "libllama.so"
        return CommandResult(0, f"libllama.so => {library} (0x1)\n", "")
    raise AssertionError(argv)


def test_checked_in_profile_has_pinned_real_identities() -> None:
    profile = load_backend_profile()

    assert profile.model.size_bytes == 17_923_394_624
    assert profile.model.sha256 == "bee238bbeb3dc0a34bde4d0dedbaee1f98c009e8bb4226f03070054c12fb1372"
    assert profile.chat_template.sha256 == "2d59a4438d68dc818c5a75db4edcf4c588e0976b113c5c87def7fc9c1168e955"
    assert profile.executable.sha256 == "92a71ff10ed10f9a24d5af934770f86e1ac6ef0dfccb7d5612f73a2670bb123b"
    assert profile.llama_cpp_commit == "dc72703fc69698b1ea68ece8d2dd8a96e6a4e1fe"
    assert profile.gpu.expected_uuid == "GPU-63f9c2ad-4dbc-962b-b314-a652bf28fc0d"
    assert profile.chat_template.path.read_bytes()
    assert _sha(profile.chat_template.path) == profile.chat_template.sha256


def test_profile_serialization_and_resolved_argv_are_deterministic(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    paths = _paths(tmp_path)
    first = resolve_backend_invocation(profile, paths, run_seed=1001)
    second = resolve_backend_invocation(profile, paths, run_seed=1001)

    assert first == second
    assert first.argv[0] == str(profile.executable.path)
    assert first.argv[first.argv.index("--ctx-size") + 1] == "107520"
    assert first.argv[first.argv.index("--chat-template-file") + 1] == str(profile.chat_template.path)
    assert first.argv[first.argv.index("--temp") + 1] == "1.0"
    assert first.argv[first.argv.index("--min-p") + 1] == "0.0"
    assert first.argv[first.argv.index("--seed") + 1] == "1001"
    assert first.run_seed == 1001
    assert "--reasoning-effort" not in first.argv
    assert "--reasoning-budget" not in first.argv
    assert "--no-reasoning-preserve" not in first.argv
    assert "--predict" not in first.argv
    assert first.environment["CUDA_VISIBLE_DEVICES"] == "GPU-RTX3090"
    assert set(first.environment) == {
        "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
        "XDG_STATE_HOME", "CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES",
        "LANG", "LC_ALL", "TZ",
    }
    json.dumps(first.model_dump(mode="json"), sort_keys=True)


def test_seed_policy_is_repetition_based_and_cross_harness_stable() -> None:
    assert [seed_for_repetition(index) for index in (1, 2, 3)] == [1001, 1002, 1003]
    with pytest.raises(ValueError, match="one-based"):
        seed_for_repetition(0)


def test_resolved_server_argv_changes_only_seed_between_repetitions(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    paths = _paths(tmp_path)
    first = resolve_backend_invocation(profile, paths, run_seed=1001)
    second = resolve_backend_invocation(profile, paths, run_seed=1002)
    seed_position = first.argv.index("--seed") + 1

    assert second.argv.index("--seed") + 1 == seed_position
    assert first.argv[:seed_position] == second.argv[:seed_position]
    assert first.argv[seed_position] == "1001"
    assert second.argv[seed_position] == "1002"
    assert first.argv[seed_position + 1 :] == second.argv[seed_position + 1 :]


def test_preflight_passes_with_exact_fixture_identity(tmp_path: Path) -> None:
    report = preflight_backend(
        _profile(tmp_path),
        _paths(tmp_path),
        run_seed=1001,
        run_command=_command,
        collect_gpus=_gpus,
        port_is_free=lambda host, port: True,
    )

    assert report.passed
    assert report.primary_failure_class is None
    assert all(check.passed for check in report.checks)


@pytest.mark.parametrize(
    ("field", "expected_class"),
    [("model", "model_hash_mismatch"), ("chat_template", "template_hash_mismatch")],
)
def test_preflight_hash_mismatch_is_classified(
    tmp_path: Path, field: str, expected_class: str
) -> None:
    profile = _profile(tmp_path)
    bad = getattr(profile, field).model_copy(update={"sha256": "0" * 64})
    profile = profile.model_copy(update={field: bad})

    report = preflight_backend(
        profile,
        _paths(tmp_path),
        run_seed=1001,
        run_command=_command,
        collect_gpus=_gpus,
        port_is_free=lambda host, port: True,
    )

    assert not report.passed
    assert any(check.failure_class == expected_class for check in report.checks)


def test_executable_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    profile = profile.model_copy(
        update={"executable": profile.executable.model_copy(update={"sha256": "0" * 64})}
    )
    report = preflight_backend(
        profile, _paths(tmp_path), run_seed=1001, run_command=_command, collect_gpus=_gpus,
        port_is_free=lambda host, port: True,
    )
    assert any(check.failure_class == "backend_identity_mismatch" for check in report.checks)


def test_gpu_parsers_and_v100_only_workload_do_not_block() -> None:
    gpus = parse_gpu_inventory(
        "0, GPU-RTX3090, NVIDIA GeForce RTX 3090, 24576, 500, 24076, 0, 40\n"
        "1, GPU-V100, Tesla V100-PCIE-32GB, 32768, 24000, 8768, 99, 70\n"
    )
    processes = parse_gpu_processes("GPU-V100, 55, python vllm, 23000\n")
    checks = evaluate_gpu_preconditions(
        GpuPreconditionPolicy(expected_uuid="GPU-RTX3090"), gpus, processes
    )
    assert all(check.passed for check in checks)
    memory = next(
        check for check in checks
        if check.check_id == "benchmark-gpu-memory-observation"
    )
    assert memory.evidence["memory_used_mib"] == 500
    assert memory.evidence["memory_free_mib"] == 24076
    assert memory.evidence["available_vram_threshold_mib"] is None


def test_desktop_baseline_over_1024_mib_does_not_block_backend_loading() -> None:
    gpus, _ = _gpus(used=1500)
    processes = (
        GpuProcessObservation(
            gpu_uuid="GPU-RTX3090",
            pid=10,
            process_name="/usr/share/code/code --type=gpu-process",
            used_gpu_memory_mib=200,
        ),
    )

    checks = evaluate_gpu_preconditions(
        GpuPreconditionPolicy(expected_uuid="GPU-RTX3090"), gpus, processes
    )

    assert all(check.passed for check in checks)


def test_unavailable_gpu_observation_fails_closed(tmp_path: Path) -> None:
    def unavailable():
        raise PermissionError("GPU process telemetry is restricted")

    report = preflight_backend(
        _profile(tmp_path),
        _paths(tmp_path),
        run_seed=1001,
        run_command=_command,
        collect_gpus=unavailable,
        port_is_free=lambda host, port: True,
    )

    check = next(item for item in report.checks if item.check_id == "gpu-observation")
    assert not report.passed
    assert check.failure_class == "precondition_failed"
    assert check.evidence["availability"] == "unavailable"
    assert report.gpu_observations == ()
    assert report.gpu_processes == ()
    assert report.gpu_observed_at_utc is None
    assert report.gpu_observation_source == "unavailable"


def test_rtx3090_compute_workload_blocks_but_small_desktop_use_is_exempt() -> None:
    gpus, _ = _gpus(used=700)
    processes = (
        GpuProcessObservation(
            gpu_uuid="GPU-RTX3090", pid=10,
            process_name="/usr/share/code/code --type=gpu-process",
            used_gpu_memory_mib=200,
        ),
        GpuProcessObservation(
            gpu_uuid="GPU-RTX3090", pid=11, process_name="VLLM::EngineCore",
            used_gpu_memory_mib=20_000,
        ),
    )
    checks = evaluate_gpu_preconditions(
        GpuPreconditionPolicy(expected_uuid="GPU-RTX3090"), gpus, processes
    )
    process_check = next(check for check in checks if check.check_id == "benchmark-gpu-processes")
    assert not process_check.passed
    assert process_check.failure_class == "conflicting_gpu_process"
    assert process_check.evidence["blocking_processes"][0]["pid"] == 11  # type: ignore[index]


def test_occupied_benchmark_port_is_detected() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not is_port_free("127.0.0.1", port)


def test_preflight_classifies_occupied_benchmark_port(tmp_path: Path) -> None:
    report = preflight_backend(
        _profile(tmp_path), _paths(tmp_path), run_seed=1001, run_command=_command,
        collect_gpus=_gpus, port_is_free=lambda host, port: False,
    )
    check = next(item for item in report.checks if item.check_id == "benchmark-port")
    assert not check.passed
    assert check.failure_class == "benchmark_port_in_use"
    proxy_check = next(
        item for item in report.checks if item.check_id == "capture-proxy-port"
    )
    assert not proxy_check.passed
    assert proxy_check.failure_class == "benchmark_port_in_use"


def test_backend_endpoint_observation_preserves_exact_slots_body(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    body = b'[{"id":0,"n_ctx":107520,"n_past":321}]'
    observed = observe_backend_endpoint(
        profile,
        "/slots",
        request=lambda url, timeout: (200, "application/json", body),
        observed_at_utc=datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc),
    )

    assert observed.http_status == 200
    assert observed.parsed_body == [{"id": 0, "n_ctx": 107520, "n_past": 321}]
    assert observed.body_sha256 == hashlib.sha256(body).hexdigest()


class _FakeProcess:
    def __init__(self, *, running: bool = True) -> None:
        self.returncode = None if running else 3
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


def test_owned_backend_shutdown_never_touches_unrelated_process(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    paths = _paths(tmp_path)
    report = preflight_backend(
        profile, paths, run_seed=1001, run_command=_command, collect_gpus=_gpus,
        port_is_free=lambda host, port: True,
    )
    owned_process = _FakeProcess()
    unrelated = _FakeProcess()
    started = start_owned_backend(
        profile,
        paths,
        report,
        tmp_path / "logs" / "stdout.log",
        tmp_path / "logs" / "stderr.log",
        run_seed=1001,
        popen=lambda *args, **kwargs: owned_process,  # type: ignore[arg-type]
    )
    code, method = started.shutdown(0.01)
    assert (code, method) == (0, "terminate")
    assert owned_process.terminate_calls == 1
    assert unrelated.terminate_calls == unrelated.kill_calls == 0


def test_startup_failure_and_readiness_timeout_are_observable(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    paths = _paths(tmp_path)
    report = preflight_backend(
        profile, paths, run_seed=1001, run_command=_command, collect_gpus=_gpus,
        port_is_free=lambda host, port: True,
    )
    with pytest.raises(OSError, match="start failed"):
        start_owned_backend(
            profile, paths, report,
            tmp_path / "failed" / "stdout.log", tmp_path / "failed" / "stderr.log",
            run_seed=1001,
            popen=lambda *args, **kwargs: (_ for _ in ()).throw(OSError("start failed")),
        )
    started = start_owned_backend(
        profile, paths, report,
        tmp_path / "timeout" / "stdout.log", tmp_path / "timeout" / "stderr.log",
        run_seed=1001,
        popen=lambda *args, **kwargs: _FakeProcess(),  # type: ignore[arg-type]
    )
    with pytest.raises(BackendLifecycleError, match="deadline"):
        started.wait_until_ready(
            profile,
            request=lambda url, timeout: (_ for _ in ()).throw(OSError("not ready")),
            poll_seconds=0.001,
        )
    started.shutdown(0.01)

    exited = start_owned_backend(
        profile, paths, report,
        tmp_path / "early-exit" / "stdout.log",
        tmp_path / "early-exit" / "stderr.log",
        run_seed=1001,
        popen=lambda *args, **kwargs: _FakeProcess(running=False),  # type: ignore[arg-type]
    )
    with pytest.raises(BackendStartFailed, match="before readiness"):
        exited.wait_until_ready(profile, poll_seconds=0.001)
    exited.shutdown(0.01)
