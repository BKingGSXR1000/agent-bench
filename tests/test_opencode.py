from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from pathlib import Path

import pytest

from agent_bench.events import RawEventWriter, load_normalized_events, load_raw_events
from agent_bench.harness import HarnessRunContext, HarnessRunPaths
from agent_bench.metrics import calculate_run_metrics
from agent_bench.models import RunLimits
from agent_bench.opencode import (
    BENCHMARK_OPENCODE_EXECUTABLE,
    OpenCodeAdapter,
    OpenCodeError,
    OpenCodeExecutable,
    OpenCodeProfile,
    build_opencode_command,
    inspect_opencode_executable,
    load_opencode_profile,
    materialize_opencode_profile,
    opencode_capture_capabilities,
    opencode_environment,
    verify_opencode_toolchain,
)
from agent_bench.opencode_events import is_test_command, normalize_opencode_events
from agent_bench.opencode_run import _port_has_listener
from agent_bench.runner import execute_run
from conftest import GitRepositoryFixture, RunFixture


def _make_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_opencode(path: Path, *, exit_code: int = 0, wait: bool = False) -> Path:
    behavior = "time.sleep(30)" if wait else ""
    return _make_executable(
        path,
        f"""#!/usr/bin/python3
import hashlib
import json
import os
import pathlib
import sys
import time

if '--version' in sys.argv:
    print('fixture-1.0')
    raise SystemExit(0)
data = pathlib.Path(os.environ['XDG_DATA_HOME']) / 'opencode'
data.mkdir(parents=True, exist_ok=True)
if 'export' in sys.argv:
    prompt = (data / 'prompt.txt').read_text(encoding='utf-8')
    print(json.dumps({{'messages':[{{'info':{{'role':'user'}},'parts':[{{'type':'text','text':prompt}}]}}]}}))
    raise SystemExit(0)
prompt = sys.stdin.read()
(data / 'prompt.txt').write_text(prompt, encoding='utf-8')
home = os.environ['HOME']
pathlib.Path(home, '.opencode-home-marker').write_text('isolated', encoding='utf-8')
session = 'ses_' + hashlib.sha256(home.encode()).hexdigest()[:12]
now = int(time.time() * 1000)
def emit(kind, part):
    print(json.dumps({{'type':kind,'timestamp':int(time.time()*1000),'sessionID':session,'part':part}}, separators=(',',':')), flush=True)
emit('step_start', {{'id':'step-1','sessionID':session,'messageID':'msg-1','type':'step-start'}})
emit('reasoning', {{'id':'reason-1','sessionID':session,'messageID':'msg-1','type':'reasoning','text':'inspect then edit','time':{{'start':now,'end':now+1}}}})
emit('tool_use', {{'id':'part-read','sessionID':session,'messageID':'msg-1','type':'tool','callID':'call-read','tool':'read','state':{{'status':'completed','input':{{'filePath':'tracked.txt'}},'output':'baseline','title':'read','metadata':{{}},'time':{{'start':now+2,'end':now+3}}}}}})
workspace = pathlib.Path(sys.argv[sys.argv.index('--dir') + 1])
(workspace / 'tracked.txt').write_text('changed by OpenCode fixture\\n', encoding='utf-8')
emit('tool_use', {{'id':'part-edit','sessionID':session,'messageID':'msg-1','type':'tool','callID':'call-edit','tool':'edit','state':{{'status':'completed','input':{{'filePath':'tracked.txt','oldString':'baseline','newString':'changed'}},'output':'done','title':'edit','metadata':{{}},'time':{{'start':now+4,'end':now+5}}}}}})
emit('tool_use', {{'id':'part-test','sessionID':session,'messageID':'msg-1','type':'tool','callID':'call-test','tool':'bash','state':{{'status':'completed','input':{{'command':'pytest -q'}},'output':'1 passed','title':'test','metadata':{{}},'time':{{'start':now+6,'end':now+7}}}}}})
emit('step_finish', {{'id':'step-end','sessionID':session,'messageID':'msg-1','type':'step-finish','reason':'stop','cost':0,'tokens':{{'input':10,'output':4,'reasoning':2,'cache':{{'read':0,'write':0}}}}}})
sys.stderr.write('fixture stderr\\n')
{behavior}
raise SystemExit({exit_code})
""",
    )


def _profile_for(executable: Path):
    profile = load_opencode_profile()
    identity = OpenCodeExecutable(
        path=executable,
        size_bytes=executable.stat().st_size,
        sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        version="fixture-1.0",
        runtime_identity="self-contained Bun ELF",
    )
    return profile.model_copy(update={"executable": identity})


def _context(tmp_path: Path, run_fixture: RunFixture) -> tuple[HarnessRunContext, RawEventWriter]:
    paths = HarnessRunPaths(
        workspace=tmp_path / "workspace",
        home=tmp_path / "home",
        xdg_config_home=tmp_path / "config",
        xdg_cache_home=tmp_path / "cache",
        xdg_data_home=tmp_path / "data",
        xdg_state_home=tmp_path / "state",
        harness_state=tmp_path / "harness",
    )
    for value in paths.__dict__.values():
        value.mkdir(parents=True)
    writer = RawEventWriter(tmp_path / "raw.jsonl", "opencode-context")
    context = HarnessRunContext(
        run_definition=run_fixture.run_definition,
        paths=paths,
        prompt_content=run_fixture.prompt_content,
        events=writer,
        limits=RunLimits(wall_timeout_seconds=5),
        cancellation=__import__("threading").Event(),
        proxy_endpoint="http://127.0.0.1:18081/v1",
        run_seed=1001,
    )
    return context, writer


def test_executable_identity_parsing_uses_isolated_environment(tmp_path: Path) -> None:
    executable = _make_executable(
        tmp_path / "opencode",
        "#!/bin/sh\nprintf 'fixture-1.0\\n'\n",
    )
    identity = inspect_opencode_executable(executable)

    assert identity.path == executable
    assert identity.version == "fixture-1.0"
    assert identity.sha256 == hashlib.sha256(executable.read_bytes()).hexdigest()
    assert identity.size_bytes == executable.stat().st_size


def test_default_profile_uses_only_the_benchmark_managed_executable(
    tmp_path: Path, run_fixture: RunFixture
) -> None:
    profile = load_opencode_profile()
    context, writer = _context(tmp_path, run_fixture)
    writer.seal()

    assert profile.executable.path == BENCHMARK_OPENCODE_EXECUTABLE
    assert profile.executable.path != Path("/home/bking/.opencode/bin/opencode")
    assert verify_opencode_toolchain(profile) == profile.executable
    assert build_opencode_command(profile, context)[0] == str(BENCHMARK_OPENCODE_EXECUTABLE)
    assert "/home/bking/.opencode/bin" not in opencode_environment(
        context, tmp_path / "opencode.json"
    ).values()
    altered = profile.model_dump(mode="python", exclude_computed_fields=True)
    altered["executable"]["path"] = Path("/home/bking/.opencode/bin/opencode")
    with pytest.raises(ValueError, match="benchmark-managed executable"):
        OpenCodeProfile.model_validate(altered)


def test_benchmark_managed_executable_missing_or_drifted_fails_preflight() -> None:
    profile = load_opencode_profile()
    missing = profile.model_copy(
        update={"executable": profile.executable.model_copy(update={"path": Path("/tmp/missing-agent-bench-opencode")})}
    )
    with pytest.raises(OpenCodeError, match="missing or invalid"):
        verify_opencode_toolchain(missing)
    drifted = profile.model_copy(
        update={"executable": profile.executable.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(OpenCodeError, match="identity differs"):
        verify_opencode_toolchain(drifted)


def test_profile_materialization_command_environment_and_prompt_bytes(
    tmp_path: Path, run_fixture: RunFixture
) -> None:
    context, writer = _context(tmp_path, run_fixture)
    profile = load_opencode_profile()
    source_before = profile.config_file.read_bytes()
    config = materialize_opencode_profile(profile, context)
    environment = opencode_environment(context, config)
    command = build_opencode_command(profile, context)
    writer.seal()

    assert profile.config_file.read_bytes() == source_before
    assert json.loads(config.read_text())["provider"]["agent-bench"]["options"]["baseURL"] == "http://127.0.0.1:18081/v1"
    assert command[:3] == (str(profile.executable.path), "--pure", "run")
    assert "--format" in command and "json" in command and "--auto" in command
    assert run_fixture.prompt_content not in command
    assert profile.invocation.prompt_delivery == "stdin_exact_utf8"
    assert environment["HOME"] == str(context.paths.home)
    assert environment["OPENCODE_CONFIG"] == str(config)
    assert "OPENAI_API_KEY" not in environment
    assert "OPENCODE_DISABLE_PROJECT_CONFIG" not in environment


def test_profile_materialization_is_fresh_and_cannot_share_state(
    tmp_path: Path, run_fixture: RunFixture
) -> None:
    first, first_writer = _context(tmp_path / "one", run_fixture)
    second, second_writer = _context(tmp_path / "two", run_fixture)
    profile = load_opencode_profile()
    materialize_opencode_profile(profile, first)
    marker = first.paths.xdg_data_home / "opencode/session-marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("private", encoding="utf-8")
    materialize_opencode_profile(profile, second)
    first_writer.seal()
    second_writer.seal()

    assert not (second.paths.xdg_data_home / "opencode/session-marker").exists()
    assert first.paths.home != second.paths.home


def test_native_event_normalization_classifies_tools_failures_files_shell_and_tests(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    writer = RawEventWriter(raw_path, "normalize-opencode")
    writer.emit(source="runner", event_type="run_start", payload={"isolated_paths": {"workspace": "/tmp/work"}})
    now = int(time_now_ms())
    for call_id, tool, status, inputs in (
        ("read-1", "read", "completed", {"filePath": "a.txt"}),
        ("edit-1", "edit", "error", {"filePath": "/tmp/work/a.txt"}),
        ("bash-1", "bash", "completed", {"command": "pytest -q"}),
    ):
        state = {"status": status, "input": inputs, "time": {"start": now, "end": now + 1}}
        if status == "completed":
            state.update({"output": "ok", "title": "done", "metadata": {}})
        else:
            state["error"] = "failed"
        writer.emit(
            source="harness",
            event_type="opencode_event",
            payload={"native_event": {"type": "tool_use", "sessionID": "ses-1", "part": {"id": call_id, "messageID": "msg-1", "callID": call_id, "tool": tool, "type": "tool", "state": state}}},
        )
    writer.emit(source="runner", event_type="run_end", payload={"observed_execution_outcome": "success"})
    writer.seal()
    normalized_path = tmp_path / "normalized.jsonl"
    normalize_opencode_events(raw_path, normalized_path)
    events = load_normalized_events(normalized_path)

    starts = [event for event in events if event.event_kind == "tool_call_start"]
    ends = [event for event in events if event.event_kind == "tool_call_end"]
    assert [event.payload["category"] for event in starts] == ["read", "edit", "test"]
    assert starts[1].payload["path"] == "a.txt"
    assert [event.payload["outcome"] for event in ends] == ["success", "failure", "success"]
    assert {event.event_kind for event in events} >= {"file_read", "file_edit", "test_execution"}
    assert all(event.normalizer_name == "agent-bench-opencode" for event in events)
    assert starts[0].clock_source == "harness_wall_clock"
    assert is_test_command("python -m pytest tests")
    assert not is_test_command("python app.py")


def test_adapter_run_captures_native_output_session_and_artifact_metrics(
    tmp_path: Path,
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    executable = _fake_opencode(tmp_path / "opencode")
    definition = run_fixture.run_definition.model_copy(
        update={
            "run_id": "opencode-fixture-success",
            "harness_id": "opencode",
            "profile_id": "opencode-default-v1",
            "limits": RunLimits(wall_timeout_seconds=5),
        }
    )
    result = execute_run(
        run_definition=definition,
        prompt_content=run_fixture.prompt_content,
        adapter=OpenCodeAdapter(_profile_for(executable), verify_executable=False),
        artifacts_root=git_repository.artifacts_root,
        worktrees_root=git_repository.worktrees_root,
        isolation_root=tmp_path / "isolation",
        proxy_endpoint="http://127.0.0.1:18081/v1",
        run_seed=1001,
    )
    metrics = calculate_run_metrics(result.artifact_path)
    raw = load_raw_events(result.raw_event_path)
    normalized = load_normalized_events(result.normalized_event_path)

    assert result.run_manifest.observed_execution_outcome == "success"
    assert result.run_manifest.run_seed == 1001
    assert result.run_manifest.proxy_endpoint == "http://127.0.0.1:18081/v1"
    assert (result.artifact_path / "run/prompt.txt").read_bytes() == run_fixture.prompt_content.encode("utf-8")
    assert (result.artifact_path / "raw/opencode/stdout.jsonl").is_file()
    assert (result.artifact_path / "raw/opencode/stderr.log").read_text() == "fixture stderr\n"
    assert (result.artifact_path / "raw/opencode/stdin-prompt.bin").read_bytes() == run_fixture.prompt_content.encode("utf-8")
    assert (result.artifact_path / "raw/opencode/session-export.json").is_file()
    assert (result.artifact_path / "run/opencode/home/.opencode-home-marker").is_file()
    assert (result.artifact_path / "run/opencode/data/opencode/prompt.txt").is_file()
    prompt_validation = next(event for event in raw if event.event_type == "opencode_prompt_validation")
    assert prompt_validation.payload["exact_prompt_found"] is True
    assert {event.event_kind for event in normalized} >= {"reasoning", "file_read", "file_edit", "test_execution"}
    assert metrics.behavior.tool_calls_total.value == 3
    assert metrics.behavior.edit_calls.value == 1
    assert metrics.behavior.agent_invoked_test_calls.value == 1
    assert metrics.timing.time_to_first_tool_call_seconds.availability == "available"
    assert metrics.timing.time_to_first_edit_seconds.availability == "available"
    assert metrics.git_result.files_changed.value == 1
    assert metrics.termination.termination_class == "success"
    assert opencode_capture_capabilities().session_identity == "harness_exact"
    assert opencode_capture_capabilities().compaction_events == "unavailable"


def test_adapter_nonzero_exit_and_timeout_are_preserved(
    tmp_path: Path,
    git_repository: GitRepositoryFixture,
    run_fixture: RunFixture,
) -> None:
    for label, executable, timeout, expected in (
        ("crash", _fake_opencode(tmp_path / "crash-opencode", exit_code=7), 5, "harness_crash"),
        ("timeout", _fake_opencode(tmp_path / "wait-opencode", wait=True), 0.05, "timeout"),
    ):
        definition = run_fixture.run_definition.model_copy(
            update={
                "run_id": f"opencode-fixture-{label}",
                "harness_id": "opencode",
                "profile_id": "opencode-default-v1",
                "limits": RunLimits(wall_timeout_seconds=timeout),
            }
        )
        result = execute_run(
            run_definition=definition,
            prompt_content=run_fixture.prompt_content,
            adapter=OpenCodeAdapter(_profile_for(executable), verify_executable=False),
            artifacts_root=git_repository.artifacts_root,
            worktrees_root=git_repository.worktrees_root,
            isolation_root=tmp_path / f"{label}-isolation",
            proxy_endpoint="http://127.0.0.1:18081/v1",
            run_seed=1001,
        )
        assert result.run_manifest.observed_execution_outcome == expected
        assert result.artifact_path.is_dir()


def test_listener_probe_reports_listener_state() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert _port_has_listener("127.0.0.1", port)
    assert not _port_has_listener("127.0.0.1", port)


def time_now_ms() -> int:
    import time

    return int(time.time() * 1000) + 10
