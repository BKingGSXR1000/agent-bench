from __future__ import annotations

import hashlib
import json
import stat
import threading
from pathlib import Path

import pytest

from agent_bench.events import RawEventWriter, load_normalized_events, load_raw_events
from agent_bench.harness import HarnessRunContext, HarnessRunPaths
from agent_bench.models import RunLimits
from agent_bench.pi import (
    PiAdapter,
    PiError,
    PiNodeRuntime,
    build_pi_command,
    inspect_pi_toolchain,
    load_pi_profile,
    materialize_pi_profile,
    pi_capture_capabilities,
    pi_environment,
)
from agent_bench.pi_events import is_test_command, normalize_pi_events
from agent_bench.runner import execute_run
from conftest import GitRepositoryFixture, RunFixture


EXACT_PROMPT = "Inspect README.md, change the single line `status: pending` to `status: complete`, and make no other source changes.\n"
EXACT_PROMPT_SHA256 = "03b18403ef4a275d88d1dbaaa9f92f0935a5c38631afa3bcf3c3fbe1526de67f"


def _make_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_node(path: Path, *, exit_code: int = 0, wait: bool = False) -> Path:
    delay = "time.sleep(30)" if wait else ""
    return _make_executable(path, f'''#!/usr/bin/python3
import json, os, pathlib, sys, time
if '--version' in sys.argv:
    print('v26.8.1')
    raise SystemExit(0)
prompt = sys.argv[-1]
agent = pathlib.Path(os.environ['PI_CODING_AGENT_DIR'])
sessions = pathlib.Path(os.environ['PI_CODING_AGENT_SESSION_DIR'])
sessions.mkdir(parents=True, exist_ok=True)
session_id = 'pi-fixture-session'
(sessions / 'fixture.jsonl').write_text('\\n'.join([
    json.dumps({{'type':'session','version':3,'id':session_id,'timestamp':'2026-09-04T00:00:00.000Z','cwd':os.getcwd()}}),
    json.dumps({{'type':'message','id':'user-1','parentId':None,'timestamp':'2026-09-04T00:00:00.001Z','message':{{'role':'user','content':[{{'type':'text','text':prompt}}]}}}}),
]) + '\\n', encoding='utf-8')
pathlib.Path(os.environ['HOME'], '.pi-home-marker').write_text('isolated', encoding='utf-8')
workspace = pathlib.Path(os.getcwd())
now = 1788500000000
def emit(value): print(json.dumps(value, separators=(',',':')), flush=True)
emit({{'type':'session','version':3,'id':session_id,'timestamp':'2026-09-04T00:00:00.000Z','cwd':os.getcwd()}})
emit({{'type':'agent_start'}})
emit({{'type':'message_end','message':{{'role':'assistant','content':[{{'type':'thinking','thinking':'inspect then edit','thinkingSignature':'reasoning_content'}}],'responseId':'resp-1','stopReason':'stop'}}}})
emit({{'type':'tool_execution_start','toolCallId':'read-1','toolName':'read','args':{{'path':str(workspace / 'README.md')}}}})
emit({{'type':'tool_execution_end','toolCallId':'read-1','toolName':'read','result':{{'content':'status: pending'}},'isError':False}})
emit({{'type':'tool_execution_start','toolCallId':'edit-1','toolName':'edit','args':{{'path':str(workspace / 'README.md'),'edits':[]}}}})
(workspace / 'README.md').write_text('status: complete\\n', encoding='utf-8')
emit({{'type':'tool_execution_end','toolCallId':'edit-1','toolName':'edit','result':{{'content':'ok'}},'isError':False}})
emit({{'type':'tool_execution_start','toolCallId':'test-1','toolName':'bash','args':{{'command':'pytest -q'}}}})
emit({{'type':'tool_execution_end','toolCallId':'test-1','toolName':'bash','result':{{'content':'1 passed'}},'isError':False}})
emit({{'type':'compaction_start','reason':'fixture'}})
emit({{'type':'compaction_end','reason':'fixture'}})
emit({{'type':'agent_end','messages':[]}})
sys.stderr.write('pi fixture stderr\\n')
{delay}
raise SystemExit({exit_code})
''')


def _profile_for(node: Path):
    profile = load_pi_profile()
    toolchain = profile.toolchain.model_copy(update={
        "node": PiNodeRuntime(path=node, size_bytes=node.stat().st_size, sha256=hashlib.sha256(node.read_bytes()).hexdigest(), version="v26.8.1"),
        "entrypoint_path": node,
        "entrypoint_size_bytes": node.stat().st_size,
        "entrypoint_sha256": hashlib.sha256(node.read_bytes()).hexdigest(),
    })
    return profile.model_copy(update={"toolchain": toolchain})


def _context(tmp_path: Path, run_fixture: RunFixture) -> tuple[HarnessRunContext, RawEventWriter]:
    paths = HarnessRunPaths(workspace=tmp_path / "workspace", home=tmp_path / "home", xdg_config_home=tmp_path / "config", xdg_cache_home=tmp_path / "cache", xdg_data_home=tmp_path / "data", xdg_state_home=tmp_path / "state", harness_state=tmp_path / "harness")
    for path in paths.__dict__.values(): path.mkdir(parents=True)
    writer = RawEventWriter(tmp_path / "raw.jsonl", "pi-context")
    context = HarnessRunContext(run_definition=run_fixture.run_definition, paths=paths, prompt_content=EXACT_PROMPT, events=writer, limits=RunLimits(wall_timeout_seconds=5), cancellation=threading.Event(), proxy_endpoint="http://127.0.0.1:18081/v1", run_seed=1001)
    return context, writer


def test_pinned_toolchain_uses_explicit_node_not_restricted_path() -> None:
    profile = load_pi_profile()
    inspect_pi_toolchain(profile.toolchain)
    assert profile.toolchain.node.path == Path('/home/bking/AI/agent-bench/toolchains/node/26.8.1/bin/node')
    assert profile.toolchain.node.version == 'v26.8.1'
    assert profile.toolchain.node.path != Path('/usr/bin/node')
    assert profile.toolchain.entrypoint_path.name == 'cli.js'
    with pytest.raises(PiError, match='SHA256'):
        inspect_pi_toolchain(profile.toolchain.model_copy(update={"node": profile.toolchain.node.model_copy(update={"sha256": "0" * 64})}))


def test_profile_materialization_command_environment_and_exact_prompt(tmp_path: Path, run_fixture: RunFixture) -> None:
    context, writer = _context(tmp_path, run_fixture)
    profile = load_pi_profile(); source = profile.models_file.read_bytes()
    agent_dir = materialize_pi_profile(profile, context)
    environment = pi_environment(context, agent_dir)
    command = build_pi_command(profile, context)
    writer.seal()
    assert profile.models_file.read_bytes() == source
    assert json.loads((agent_dir / 'models.json').read_text())['providers']['agent-bench']['baseUrl'] == 'http://127.0.0.1:18081/v1'
    assert command[:2] == (str(profile.toolchain.node.path), str(profile.toolchain.entrypoint_path))
    assert command[-2:] == ('--', EXACT_PROMPT)
    assert '--offline' in command
    assert not {'--no-extensions', '--no-skills', '--no-prompt-templates', '--no-themes'} & set(command)
    assert hashlib.sha256(command[-1].encode()).hexdigest() == EXACT_PROMPT_SHA256
    assert environment['PATH'] == '/usr/local/bin:/usr/bin:/bin'
    assert environment['PI_CODING_AGENT_DIR'] == str(agent_dir)
    assert environment['PI_CODING_AGENT_SESSION_DIR'].startswith(str(context.paths.xdg_data_home))
    assert 'PI_OFFLINE' not in environment


def test_profile_isolation_does_not_share_pi_state(tmp_path: Path, run_fixture: RunFixture) -> None:
    first, first_writer = _context(tmp_path / 'one', run_fixture); second, second_writer = _context(tmp_path / 'two', run_fixture)
    profile = load_pi_profile(); first_agent = materialize_pi_profile(profile, first)
    (first_agent / 'sessions').mkdir(); (first_agent / 'sessions' / 'marker').write_text('private')
    second_agent = materialize_pi_profile(profile, second); first_writer.seal(); second_writer.seal()
    assert not (second_agent / 'sessions' / 'marker').exists()
    assert first.paths.home != second.paths.home


def test_native_event_normalization_tracks_tools_reasoning_compaction_and_paths(tmp_path: Path) -> None:
    raw_path = tmp_path / 'raw.jsonl'; writer = RawEventWriter(raw_path, 'pi-normalize')
    writer.emit(source='runner', event_type='run_start', payload={'isolated_paths': {'workspace': '/tmp/work'}})
    for event in (
        {'type':'message_end','message':{'role':'assistant','content':[{'type':'thinking','thinking':'reasoning'}], 'stopReason':'stop'}},
        {'type':'tool_execution_start','toolCallId':'read','toolName':'read','args':{'path':'/tmp/work/a.txt'}},
        {'type':'tool_execution_end','toolCallId':'read','toolName':'read','result':{'value':'x'},'isError':False},
        {'type':'tool_execution_start','toolCallId':'edit','toolName':'edit','args':{'path':'/tmp/work/a.txt'}},
        {'type':'tool_execution_end','toolCallId':'edit','toolName':'edit','result':{},'isError':True},
        {'type':'tool_execution_start','toolCallId':'test','toolName':'bash','args':{'command':'pytest -q'}},
        {'type':'tool_execution_end','toolCallId':'test','toolName':'bash','result':{},'isError':False},
        {'type':'compaction_start'}, {'type':'compaction_end'},
    ):
        writer.emit(source='harness', event_type='pi_event', payload={'native_event': event})
    writer.emit(source='runner', event_type='run_end', payload={'observed_execution_outcome':'success'}); writer.seal()
    normalized_path = tmp_path / 'normalized.jsonl'; normalize_pi_events(raw_path, normalized_path); events = load_normalized_events(normalized_path)
    starts = [event for event in events if event.event_kind == 'tool_call_start']; ends = [event for event in events if event.event_kind == 'tool_call_end']
    assert [event.payload['category'] for event in starts] == ['read', 'edit', 'test']
    assert starts[0].payload['path'] == 'a.txt'; assert [event.payload['outcome'] for event in ends] == ['success', 'failure', 'success']
    assert {event.event_kind for event in events} >= {'reasoning', 'file_read', 'file_edit', 'test_execution', 'compaction_start', 'compaction_end'}
    assert is_test_command('python -m pytest tests') and not is_test_command('python app.py')


def test_adapter_preserves_native_session_exact_prompt_and_metrics(tmp_path: Path, git_repository: GitRepositoryFixture, run_fixture: RunFixture) -> None:
    node = _fake_node(tmp_path / 'node')
    definition = run_fixture.run_definition.model_copy(update={'run_id':'pi-fixture-success','harness_id':'pi','profile_id':'pi-default-v1','limits':RunLimits(wall_timeout_seconds=5), 'prompt_sha256': EXACT_PROMPT_SHA256})
    result = execute_run(run_definition=definition, prompt_content=EXACT_PROMPT, adapter=PiAdapter(_profile_for(node), verify_toolchain=False), artifacts_root=git_repository.artifacts_root, worktrees_root=git_repository.worktrees_root, isolation_root=tmp_path / 'isolation', proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001)
    raw, normalized = load_raw_events(result.raw_event_path), load_normalized_events(result.normalized_event_path)
    assert result.run_manifest.observed_execution_outcome == 'success'
    assert (result.artifact_path / 'raw/pi/prompt-transport.bin').read_bytes() == EXACT_PROMPT.encode()
    assert (result.artifact_path / 'raw/pi/stdout.jsonl').is_file()
    assert (result.artifact_path / 'raw/pi/stderr.log').read_text() == 'pi fixture stderr\n'
    assert (result.artifact_path / 'run/pi/data/pi/sessions/fixture.jsonl').is_file()
    validation = next(event for event in raw if event.event_type == 'pi_prompt_validation')
    assert validation.payload['exact_prompt_found'] is True
    assert {event.event_kind for event in normalized} >= {'reasoning','file_read','file_edit','test_execution','compaction_start','compaction_end'}
    assert pi_capture_capabilities().session_identity == 'harness_exact'
    assert pi_capture_capabilities().compaction_events == 'harness_exact'


def test_adapter_crash_and_timeout_are_preserved(tmp_path: Path, git_repository: GitRepositoryFixture, run_fixture: RunFixture) -> None:
    for label, node, timeout, expected in (('crash', _fake_node(tmp_path / 'crash-node', exit_code=7), 5, 'harness_crash'), ('timeout', _fake_node(tmp_path / 'timeout-node', wait=True), 0.05, 'timeout')):
        definition = run_fixture.run_definition.model_copy(update={'run_id':f'pi-fixture-{label}','harness_id':'pi','profile_id':'pi-default-v1','limits':RunLimits(wall_timeout_seconds=timeout), 'prompt_sha256': EXACT_PROMPT_SHA256})
        result = execute_run(run_definition=definition, prompt_content=EXACT_PROMPT, adapter=PiAdapter(_profile_for(node), verify_toolchain=False), artifacts_root=git_repository.artifacts_root, worktrees_root=git_repository.worktrees_root, isolation_root=tmp_path / f'{label}-isolation', proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001)
        assert result.run_manifest.observed_execution_outcome == expected
