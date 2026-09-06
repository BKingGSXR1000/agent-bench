from __future__ import annotations

import hashlib
import json
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_bench.events import RawEventWriter, load_normalized_events, load_raw_events
from agent_bench.harness import HarnessRunContext, HarnessRunPaths
from agent_bench.hermes import HermesAdapter, HermesError, HermesRuntime, build_hermes_command, hermes_capture_capabilities, hermes_environment, inspect_hermes_toolchain, load_hermes_profile, load_hermes_profile_for_id, materialize_hermes_profile
from agent_bench.hermes_events import is_test_command, normalize_hermes_events
from agent_bench.metrics import calculate_run_metrics
from agent_bench.models import RunLimits
from agent_bench.runner import _evidence_mapping, execute_run
from agent_bench.timing_provenance import derive_hermes_timing_provenance
from conftest import GitRepositoryFixture, RunFixture

EXACT_PROMPT = "Inspect README.md, change the single line `status: pending` to `status: complete`, and make no other source changes.\n"
EXACT_PROMPT_SHA256 = "03b18403ef4a275d88d1dbaaa9f92f0935a5c38631afa3bcf3c3fbe1526de67f"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USAGE_BYTES = b'{"session_id":"hermes-fixture-session","input_tokens":10,"output_tokens":4,"reasoning_tokens":2,"api_calls":2,"completed":true}'


def _make_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _fake_hermes(
    path: Path,
    *,
    exit_code: int = 0,
    wait: bool = False,
    usage_bytes: bytes | None = DEFAULT_USAGE_BYTES,
    final_unexecuted_tool: bool = False,
) -> Path:
    delay = "time.sleep(30)" if wait else ""
    write_usage = usage_bytes is not None
    usage_writer = (
        "usage_path.parent.mkdir(parents=True, exist_ok=True); "
        f"usage_path.write_bytes({usage_bytes!r})"
        if write_usage
        else ""
    )
    return _make_executable(path, f'''#!/usr/bin/python3
import json, os, pathlib, sqlite3, sys, time
if '--version' in sys.argv:
    print('Hermes Agent v0.21.0 (fixture)'); raise SystemExit(0)
prompt = sys.argv[sys.argv.index('--oneshot') + 1]
usage_path = pathlib.Path(sys.argv[sys.argv.index('--usage-file') + 1])
home = pathlib.Path(os.environ['HERMES_HOME']); home.mkdir(parents=True, exist_ok=True)
(pathlib.Path(os.environ['HOME']) / '.home-marker').write_text('isolated', encoding='utf-8')
db = sqlite3.connect(home / 'state.db')
db.execute('CREATE TABLE sessions (id TEXT, started_at REAL, cwd TEXT)')
db.execute('CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, effect_disposition TEXT, timestamp REAL, token_count INTEGER, finish_reason TEXT, reasoning TEXT, reasoning_content TEXT, compacted INTEGER)')
db.execute('INSERT INTO sessions VALUES (?,?,?)', ('hermes-fixture-session', 1.0, str(pathlib.Path.cwd())))
db.execute('INSERT INTO messages (session_id,role,content,timestamp) VALUES (?,?,?,?)', ('hermes-fixture-session','user',prompt,1.0))
calls=json.dumps([{{'id':'read-1','function':{{'name':'read_file','arguments':json.dumps({{'path':str(pathlib.Path.cwd() / 'README.md')}})}}}},{{'id':'edit-1','function':{{'name':'edit_file','arguments':json.dumps({{'path':str(pathlib.Path.cwd() / 'README.md')}})}}}},{{'id':'test-1','function':{{'name':'terminal','arguments':json.dumps({{'command':'pytest -q'}})}}}}])
db.execute('INSERT INTO messages (session_id,role,content,tool_calls,reasoning_content,timestamp) VALUES (?,?,?,?,?,?)', ('hermes-fixture-session','assistant','',calls,'inspect then edit',2.0))
for call_id, name in [('read-1','read_file'),('edit-1','edit_file'),('test-1','terminal')]: db.execute('INSERT INTO messages (session_id,role,content,tool_call_id,tool_name,effect_disposition,timestamp) VALUES (?,?,?,?,?,?,?)', ('hermes-fixture-session','tool','ok',call_id,name,'success',3.0))
if {final_unexecuted_tool!r}:
    db.execute('INSERT INTO messages (session_id,role,content,tool_calls,finish_reason,timestamp) VALUES (?,?,?,?,?,?)', ('hermes-fixture-session','assistant','last visible output',json.dumps([{{'id':'unexecuted-1','function':{{'name':'patch','arguments':json.dumps({{'path':'README.md'}})}}}}]),'tool_calls',4.0))
db.execute('INSERT INTO messages (session_id,role,content,compacted,timestamp) VALUES (?,?,?,?,?)', ('hermes-fixture-session','assistant','summary',1,4.0))
db.commit(); db.close()
pathlib.Path('README.md').write_text('status: complete\\n', encoding='utf-8')
{usage_writer}
print('done'); sys.stderr.write('hermes fixture stderr\\n')
{delay}
raise SystemExit({exit_code})
''')


def _profile_for(executable: Path):
    profile = load_hermes_profile()
    runtime = profile.toolchain.model_copy(update={"python_path": executable, "python_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "entrypoint_path": executable, "entrypoint_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()})
    return profile.model_copy(update={"toolchain": runtime})


def _context(tmp_path: Path, run_fixture: RunFixture) -> tuple[HarnessRunContext, RawEventWriter]:
    paths = HarnessRunPaths(workspace=tmp_path / 'workspace', home=tmp_path / 'home', xdg_config_home=tmp_path / 'config', xdg_cache_home=tmp_path / 'cache', xdg_data_home=tmp_path / 'data', xdg_state_home=tmp_path / 'state', harness_state=tmp_path / 'harness')
    for path in paths.__dict__.values(): path.mkdir(parents=True)
    writer = RawEventWriter(tmp_path / 'raw.jsonl', 'hermes-context')
    return HarnessRunContext(run_definition=run_fixture.run_definition, paths=paths, prompt_content=EXACT_PROMPT, events=writer, limits=RunLimits(wall_timeout_seconds=5), cancellation=threading.Event(), proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001), writer


def test_benchmark_managed_toolchain_is_absolute_and_rejects_drift() -> None:
    profile = load_hermes_profile(); inspect_hermes_toolchain(profile.toolchain)
    assert profile.toolchain.entrypoint_path == REPOSITORY_ROOT / "toolchains/hermes/0.21.0/venv/bin/hermes"
    assert '/home/bking/.hermes' not in str(profile.toolchain.entrypoint_path)
    assert '/home/bking/.hermes' not in str(profile.toolchain.python_path)
    assert profile.toolchain.node_path == REPOSITORY_ROOT / "toolchains/node/26.8.1/bin/node"
    with pytest.raises(HermesError, match='SHA256'):
        inspect_hermes_toolchain(profile.toolchain.model_copy(update={'entrypoint_sha256': '0' * 64}))


def test_profile_environment_command_and_exact_prompt(tmp_path: Path, run_fixture: RunFixture) -> None:
    context, writer = _context(tmp_path, run_fixture); profile = load_hermes_profile(); source = profile.config_file.read_bytes()
    home = materialize_hermes_profile(profile, context); environment = hermes_environment(context, home); command = build_hermes_command(profile, context, tmp_path / 'usage.json'); writer.seal()
    assert (home / 'config.yaml').read_bytes() == source
    assert command[-2:] == ('--oneshot', EXACT_PROMPT)
    assert hashlib.sha256(command[-1].encode()).hexdigest() == EXACT_PROMPT_SHA256
    assert environment['HERMES_HOME'] == str(home)
    assert environment['PATH'] == f"{profile.toolchain.node_path.parent}:/usr/local/bin:/usr/bin:/bin"
    assert environment['TERMINAL_CWD'] == str(context.paths.workspace)
    assert environment['PYTHONNOUSERSITE'] == '1'
    assert '--ignore-rules' not in command and '--safe-mode' not in command


def test_reasoning_screen_profiles_change_only_the_pinned_request_field(
    tmp_path: Path, run_fixture: RunFixture
) -> None:
    expected = {
        "hermes-reasoning-off-v1": ("off", {"reasoning_effort": "none"}),
        "hermes-reasoning-low-v1": ("low", {"reasoning_effort": "low"}),
        "hermes-reasoning-medium-v1": ("medium", {"reasoning_effort": "medium"}),
    }
    context, writer = _context(tmp_path / "default", run_fixture)
    default = load_hermes_profile()
    default_command = build_hermes_command(default, context, tmp_path / "default-usage.json")
    for profile_id, (setting, request_fields) in expected.items():
        profile = load_hermes_profile_for_id(profile_id)
        profile_context, profile_writer = _context(tmp_path / profile_id, run_fixture)
        copied_home = materialize_hermes_profile(profile, profile_context)
        assert profile.reasoning.setting == setting
        assert profile.reasoning.request_fields == request_fields
        assert profile.toolchain == default.toolchain
        assert build_hermes_command(profile, context, tmp_path / f"{profile_id}-usage.json")[1:5] == default_command[1:5]
        assert "reasoning_effort" in (copied_home / "config.yaml").read_text(encoding="utf-8")
        profile_writer.seal()
    writer.seal()


def test_unknown_hermes_profile_has_no_fallback() -> None:
    with pytest.raises(HermesError, match="cannot load Hermes profile"):
        load_hermes_profile_for_id("hermes-not-installed-v1")
    with pytest.raises(HermesError, match="cannot load Hermes profile"):
        load_hermes_profile_for_id("hermes-reasoning-high-v1")


def test_profile_isolation_does_not_share_hermes_state(tmp_path: Path, run_fixture: RunFixture) -> None:
    first, first_writer = _context(tmp_path / 'one', run_fixture); second, second_writer = _context(tmp_path / 'two', run_fixture); profile = load_hermes_profile()
    one = materialize_hermes_profile(profile, first); (one / 'state.db').write_text('private')
    two = materialize_hermes_profile(profile, second); first_writer.seal(); second_writer.seal()
    assert not (two / 'state.db').exists(); assert one != two


def test_native_normalization_tracks_reasoning_tools_compaction_and_paths(tmp_path: Path) -> None:
    raw_path = tmp_path / 'raw.jsonl'; writer = RawEventWriter(raw_path, 'hermes-normalize')
    writer.emit(source='runner', event_type='run_start', payload={'isolated_paths': {'workspace': '/tmp/work'}})
    assistant = {'role':'assistant','reasoning_content':'reasoning','tool_calls':json.dumps([{'id':'read','function':{'name':'read_file','arguments':json.dumps({'path':'/tmp/work/a.txt'})}},{'id':'edit','function':{'name':'edit_file','arguments':json.dumps({'path':'/tmp/work/a.txt'})}},{'id':'test','function':{'name':'terminal','arguments':json.dumps({'command':'pytest -q'})}}])}
    writer.emit(source='harness', event_type='hermes_session_message', payload={'native_event': assistant})
    for call_id in ('read','edit','test'): writer.emit(source='harness', event_type='hermes_session_message', payload={'native_event': {'role':'tool','tool_call_id':call_id,'content':'ok','effect_disposition':'success'}})
    writer.emit(source='harness', event_type='hermes_session_compaction', payload={'native_event': {'compacted': True}}); writer.seal()
    normalized_path = tmp_path / 'normalized.jsonl'; normalize_hermes_events(raw_path, normalized_path); events = load_normalized_events(normalized_path)
    starts = [event for event in events if event.event_kind == 'tool_call_start']
    assert [event.payload['category'] for event in starts] == ['read','edit','test']
    assert starts[0].payload['path'] == 'a.txt'
    assert {event.event_kind for event in events} >= {'reasoning','file_read','file_edit','test_execution','compaction_end'}
    assert is_test_command('python -m pytest tests') and not is_test_command('python app.py')


def test_native_normalization_recognizes_upstream_patch_and_search_files(tmp_path: Path) -> None:
    raw_path = tmp_path / 'raw.jsonl'; writer = RawEventWriter(raw_path, 'hermes-tool-names')
    writer.emit(source='runner', event_type='run_start', payload={'isolated_paths': {'workspace': '/tmp/work'}})
    calls = [
        {'id': 'search', 'function': {'name': 'search_files', 'arguments': json.dumps({'pattern': 'README.md', 'target': 'files'})}},
        {'id': 'patch', 'function': {'name': 'patch', 'arguments': json.dumps({'path': '/tmp/work/README.md', 'old_string': 'pending', 'new_string': 'complete'})}},
    ]
    writer.emit(source='harness', event_type='hermes_session_message', payload={'native_event': {'role': 'assistant', 'timestamp': 123.5, 'tool_calls': json.dumps(calls)}})
    writer.emit(source='harness', event_type='hermes_session_message', payload={'native_event': {'role':'tool', 'tool_call_id':'search', 'tool_name':'search_files', 'content':'ok', 'timestamp':124.0}})
    writer.seal()
    normalized_path = tmp_path / 'normalized.jsonl'; normalize_hermes_events(raw_path, normalized_path)
    starts = [event for event in load_normalized_events(normalized_path) if event.event_kind == 'tool_call_start']
    assert [event.payload['category'] for event in starts] == ['search']
    intents = [event for event in load_normalized_events(normalized_path) if event.event_kind == 'model_tool_call_observed']
    assert [event.payload['tool_name'] for event in intents] == ['search_files', 'patch']
    assert starts[0].payload['timing_semantics'] == 'tool_execution_inferred_from_result_then_exported'
    assert starts[0].payload['native_message_timestamp_seconds'] == 124.0


def test_adapter_preserves_native_session_prompt_and_events(tmp_path: Path, git_repository: GitRepositoryFixture, run_fixture: RunFixture) -> None:
    emitted_usage = b'{\n  "completed": true,\n  "session_id": "hermes-fixture-session"\n}\n'
    executable = _fake_hermes(tmp_path / 'hermes', usage_bytes=emitted_usage)
    definition = run_fixture.run_definition.model_copy(update={'run_id':'hermes-fixture-success','harness_id':'hermes','profile_id':'hermes-default-v1','limits':RunLimits(wall_timeout_seconds=5),'prompt_sha256':EXACT_PROMPT_SHA256})
    result = execute_run(run_definition=definition, prompt_content=EXACT_PROMPT, adapter=HermesAdapter(_profile_for(executable), verify_toolchain=False), artifacts_root=git_repository.artifacts_root, worktrees_root=git_repository.worktrees_root, isolation_root=tmp_path / 'isolation', proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001)
    raw, normalized = load_raw_events(result.raw_event_path), load_normalized_events(result.normalized_event_path)
    assert result.run_manifest.observed_execution_outcome == 'success'
    assert (result.artifact_path / 'raw/hermes/prompt-transport.bin').read_bytes() == EXACT_PROMPT.encode()
    assert (result.artifact_path / 'raw/hermes/usage.json').read_bytes() == emitted_usage
    assert (result.artifact_path / 'raw/hermes/usage-observed.json').is_file()
    assert 'raw/hermes/usage.json' in result.run_manifest.harness_evidence_paths
    assert set(result.run_manifest.harness_evidence_paths) <= {
        path.relative_to(result.artifact_path).as_posix()
        for path in result.artifact_path.rglob('*') if path.is_file()
    }
    assert (result.artifact_path / 'run/hermes/home/state.db').is_file()
    validation = next(event for event in raw if event.event_type == 'hermes_prompt_validation')
    assert validation.payload['exact_prompt_found'] is True
    assert {event.event_kind for event in normalized} >= {'reasoning','file_read','file_edit','test_execution','compaction_end'}
    assert hermes_capture_capabilities().session_identity == 'harness_exact'
    metrics = calculate_run_metrics(result.artifact_path)
    assert metrics.behavior.tool_calls_total.value == 3
    assert metrics.behavior.tool_calls_successful.value == 3


def test_usage_evidence_is_stable_after_live_file_is_removed(
    tmp_path: Path, run_fixture: RunFixture
) -> None:
    emitted_usage = b'{\n "session_id": "hermes-fixture-session"\n}\n'
    context, writer = _context(tmp_path, run_fixture)
    result = HermesAdapter(
        _profile_for(_fake_hermes(tmp_path / 'hermes', usage_bytes=emitted_usage)),
        verify_toolchain=False,
    ).run(context)
    writer.seal()

    evidence_root = context.paths.harness_state / 'hermes'
    captured = evidence_root / 'usage-captured.json'
    assert captured.read_bytes() == emitted_usage
    assert not (evidence_root / 'usage.json').exists()
    declared = dict(result.evidence_files)
    assert declared['raw/hermes/usage.json'] == captured

    runtime = SimpleNamespace(harness_state=context.paths.harness_state)
    first = _evidence_mapping(runtime, result, ())
    second = _evidence_mapping(runtime, result, ())
    assert first == second
    assert first['raw/hermes/usage.json'] == captured


@pytest.mark.parametrize(
    ('usage_bytes', 'message'),
    ((None, 'cannot read Hermes usage file'), (b'{', 'invalid Hermes usage JSON')),
)
def test_adapter_fails_closed_when_usage_cannot_be_captured(
    tmp_path: Path, run_fixture: RunFixture, usage_bytes: bytes | None, message: str
) -> None:
    context, writer = _context(tmp_path, run_fixture)
    adapter = HermesAdapter(
        _profile_for(_fake_hermes(tmp_path / 'hermes', usage_bytes=usage_bytes)),
        verify_toolchain=False,
    )
    with pytest.raises(HermesError, match=message):
        adapter.run(context)
    writer.seal()


def test_hermes_sqlite_export_timestamps_are_not_execution_timing(tmp_path: Path, git_repository: GitRepositoryFixture, run_fixture: RunFixture) -> None:
    executable = _fake_hermes(tmp_path / 'hermes')
    definition = run_fixture.run_definition.model_copy(update={'run_id':'hermes-fixture-timing','harness_id':'hermes','profile_id':'hermes-default-v1','limits':RunLimits(wall_timeout_seconds=5),'prompt_sha256':EXACT_PROMPT_SHA256})
    result = execute_run(run_definition=definition, prompt_content=EXACT_PROMPT, adapter=HermesAdapter(_profile_for(executable), verify_toolchain=False), artifacts_root=git_repository.artifacts_root, worktrees_root=git_repository.worktrees_root, isolation_root=tmp_path / 'timing-isolation', proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001)
    analysis = derive_hermes_timing_provenance(result.artifact_path)
    assert analysis.time_to_first_harness_tool_execution.availability == 'unavailable'
    assert analysis.time_to_first_harness_tool_execution.unavailable_reason == 'native_execution_timestamp_not_exposed'
    assert analysis.time_to_first_observed_tool_event.availability == 'available'
    assert analysis.tools[0].native_call_recorded_timestamp_utc is not None
    assert analysis.tools[0].capture_elapsed_seconds >= analysis.tools[0].normalized_elapsed_seconds
    metrics = calculate_run_metrics(result.artifact_path)
    assert metrics.timing.time_to_first_tool_call_seconds.availability == 'unavailable'
    assert metrics.timing.time_to_first_tool_call_seconds.unavailable_reason == 'native_execution_timestamp_not_exposed'


def test_adapter_crash_and_timeout_are_preserved(tmp_path: Path, git_repository: GitRepositoryFixture, run_fixture: RunFixture) -> None:
    for label, executable, timeout, expected in (('crash', _fake_hermes(tmp_path / 'crash', exit_code=7), 5, 'harness_crash'), ('timeout', _fake_hermes(tmp_path / 'timeout', wait=True), 0.05, 'timeout')):
        definition = run_fixture.run_definition.model_copy(update={'run_id':f'hermes-fixture-{label}','harness_id':'hermes','profile_id':'hermes-default-v1','limits':RunLimits(wall_timeout_seconds=timeout),'prompt_sha256':EXACT_PROMPT_SHA256})
        result = execute_run(run_definition=definition, prompt_content=EXACT_PROMPT, adapter=HermesAdapter(_profile_for(executable), verify_toolchain=False), artifacts_root=git_repository.artifacts_root, worktrees_root=git_repository.worktrees_root, isolation_root=tmp_path / f'{label}-isolation', proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001)
        assert result.run_manifest.observed_execution_outcome == expected


def test_timeout_extracts_completed_tools_but_not_final_unexecuted_intent(tmp_path: Path, git_repository: GitRepositoryFixture, run_fixture: RunFixture) -> None:
    executable = _fake_hermes(
        tmp_path / 'timeout-session', wait=True, usage_bytes=None,
        final_unexecuted_tool=True,
    )
    definition = run_fixture.run_definition.model_copy(update={
        'run_id': 'hermes-fixture-timeout-session', 'harness_id': 'hermes',
        'profile_id': 'hermes-default-v1', 'limits': RunLimits(wall_timeout_seconds=.5),
        'prompt_sha256': EXACT_PROMPT_SHA256,
    })
    result = execute_run(run_definition=definition, prompt_content=EXACT_PROMPT,
        adapter=HermesAdapter(_profile_for(executable), verify_toolchain=False),
        artifacts_root=git_repository.artifacts_root, worktrees_root=git_repository.worktrees_root,
        isolation_root=tmp_path / 'timeout-isolation', proxy_endpoint='http://127.0.0.1:18081/v1', run_seed=1001)
    raw = load_raw_events(result.raw_event_path)
    events = load_normalized_events(result.normalized_event_path)
    metrics = calculate_run_metrics(result.artifact_path)
    assert result.run_manifest.observed_execution_outcome == 'timeout'
    assert next(event for event in raw if event.event_type == 'hermes_capture_status').payload['tool_capture_complete'] is True
    assert len([event for event in events if event.event_kind == 'tool_call_start']) == 3
    assert len([event for event in events if event.event_kind == 'tool_call_end']) == 3
    assert any(event.event_kind == 'model_tool_call_observed' and event.payload['tool_call_id'] == 'unexecuted-1' for event in events)
    assert not any(event.event_kind == 'tool_call_start' and event.payload['tool_call_id'] == 'unexecuted-1' for event in events)
    assert metrics.behavior.tool_calls_total.value == 3
    assert metrics.termination.termination_class == 'timeout'
