import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from tools import run_city_pretrain_supplement as runner


def options(tmp_path, replace=False):
    old, city = tmp_path / 'old', tmp_path / 'supplement'
    old.mkdir()
    city.mkdir()
    return SimpleNamespace(city_dir=city, previous_city_dir=old,
        previous_world_model_dir=old / 'world_model_ingo_profile',
        previous_supervisor_pid=123, replace_waiting_supervisor=replace,
        source_dir=Path('/frozen/source'), device='cuda:0', poll_seconds=.01,
        gpu_memory_fraction=1.0, gpu_admission_min_free_mib=8192.,
        activation_offload=False, tail_execution_batch=None, resume_existing_world_model=False)


def watcher(args, state='S'):
    return dict(pid=123, ppid=1, started_ticks=900, state=state,
        argv=['python', '/frozen/tools/run_city_pretrain.py',
              '--city-dir', str(args.previous_city_dir)])


def write_waiting(args):
    status = dict(pid=123, phase='waiting_for_world_model_gpu', child_pids={},
                  admission=dict(ready=False))
    (args.previous_city_dir / 'supervisor_status.json').write_text(json.dumps(status))
    return status


def test_commands_keep_checked_migration_and_ingo_profile(tmp_path):
    args = options(tmp_path)
    plan = runner.commands(args, dict(status='complete', entries=[
        dict(split='train'), dict(split='train'), dict(split='validation')]))
    assert plan['migrate'][2] == '/frozen/source/tools/migrate_pretrain_corpus.py'
    assert plan['migrate'][plan['migrate'].index('--source-dir') + 1] == str(args.previous_world_model_dir)
    fm = plan['world_model']
    for name, expected in {'--batch-size': '64', '--precision': 'bf16',
        '--gpu-memory-fraction': '1.0', '--compile-blocks': 'on',
        '--expected-train-episodes': '2', '--expected-validation-episodes': '1',
        '--accumulation-steps': '1', '--activation-checkpointing': 'on', '--codec-dir': str(args.previous_city_dir / 'codec')}.items():
        assert fm[fm.index(name) + 1] == expected
    assert '--activation-offload' not in fm
    assert '--tail-execution-batch' not in fm
    assert not any('run_online' in word for word in fm)


def test_commands_reject_incomplete_or_single_partition_cache(tmp_path):
    args = options(tmp_path)
    with pytest.raises(ValueError, match='complete cache'):
        runner.commands(args, dict(status='building', entries=[]))
    with pytest.raises(ValueError, match='independent'):
        runner.commands(args, dict(status='complete', entries=[dict(split='train')]))


def test_previous_activity_finds_descendants_and_detached_training(tmp_path):
    args = options(tmp_path)
    table = {123: watcher(args), 200: dict(pid=200, ppid=123, argv=['worker']),
        201: dict(pid=201, ppid=200, argv=['worker-child']),
        300: dict(pid=300, ppid=1, argv=['python', 'run_complete_pretrain.py',
            '--output-dir', str(args.previous_world_model_dir)]),
        400: dict(pid=400, ppid=1, argv=['python', 'run_complete_pretrain.py',
            '--output-dir', '/different/city/world_model'])}
    assert runner.previous_activity(args, table) == [200, 201, 300]


def test_hidden_process_argv_keeps_parent_identity_for_safety(tmp_path, monkeypatch):
    process = tmp_path / '99'
    process.mkdir()
    fields = ['S', '123'] + ['0'] * 17 + ['9876']
    (process / 'stat').write_text('99 (process with spaces) ' + ' '.join(fields))
    original = Path.read_bytes
    def read(path):
        if path == process / 'cmdline':
            raise PermissionError('another user hides command arguments')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    row = runner.process_snapshot(99, proc_root=tmp_path)
    assert row['ppid'] == 123 and row['started_ticks'] == 9876
    assert row['argv'] == []


def test_cache_producer_exit_is_reported_instead_of_waiting_forever(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.cache_builder_pid = 700
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: None)
    with pytest.raises(RuntimeError, match='cache builder exited'):
        runner.supervise(args)
    status = json.loads((args.city_dir / 'supervisor_status.json').read_text())
    assert status['phase'] == 'failed'
    assert 'cache builder exited' in status['error']


def test_complete_cache_remains_valid_after_producer_exit(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.cache_builder_pid = 700
    (args.city_dir / 'cache').mkdir()
    (args.city_dir / 'cache/cache_manifest.json').write_text(json.dumps(dict(status='complete')))
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: pytest.fail('complete cache needs no producer'))
    assert runner.checked_cache(args) == dict(status='complete')


@pytest.mark.parametrize('script,flag,value', [
    ('build_pretrain_cache.py', '--output-dir', 'cache'),
    ('build_city_supplement_cache.py', '--city-dir', ''),
])
def test_cache_producer_identity_supports_direct_and_incremental_builder(tmp_path, monkeypatch, script, flag, value):
    args = options(tmp_path)
    args.cache_builder_pid = 700
    row = dict(pid=700, argv=['python', '/frozen/tools/' + script,
                             flag, str(args.city_dir / value)])
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: row)
    assert runner.checked_cache(args) is None
    row['argv'][-1] = '/another/city'
    with pytest.raises(RuntimeError, match='changed identity'):
        runner.checked_cache(args)


def test_active_old_training_is_never_signaled(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    write_waiting(args)
    table = {123: watcher(args), 200: dict(pid=200, ppid=123, argv=['training'])}
    monkeypatch.setattr(runner, 'process_table', lambda: table)
    monkeypatch.setattr(runner.os, 'kill', lambda *a: pytest.fail('active old training must not be signaled'))
    result = runner.previous_ready(args)
    assert not result['ready']
    assert result['active_pids'] == [200]


def test_coulommiers_waits_even_if_empty_without_replacement_flag(tmp_path, monkeypatch):
    args = options(tmp_path)
    write_waiting(args)
    monkeypatch.setattr(runner, 'process_table', lambda: {123: watcher(args)})
    monkeypatch.setattr(runner.os, 'kill', lambda *a: pytest.fail('replacement was not authorized'))
    assert runner.previous_ready(args)['reason'] == 'previous_supervisor_running'


def test_live_pid_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    row = watcher(args)
    row['argv'] = ['unrelated']
    monkeypatch.setattr(runner, 'process_table', lambda: {123: row})
    monkeypatch.setattr(runner.os, 'kill', lambda *a: pytest.fail('must not signal reused PID'))
    with pytest.raises(RuntimeError, match='different live process'):
        runner.previous_ready(args)


def test_stopped_watcher_rechecked_for_child_and_resumed_if_race(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    status = write_waiting(args)
    events = []
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: watcher(args, 'T'))
    monkeypatch.setattr(runner, 'process_table', lambda: {
        123: watcher(args, 'T'), 200: dict(pid=200, ppid=123, argv=['training'])})
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: events.append((pid, sig)))
    assert not runner.replace_empty_waiter(args, watcher(args), status)
    assert events == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]
    assert not (args.city_dir / 'previous_handoff.json').exists()


def test_waiter_with_successful_admission_is_not_frozen_during_launch(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    status = write_waiting(args)
    status['admission']['ready'] = True
    monkeypatch.setattr(runner.os, 'kill', lambda *a: pytest.fail('watcher is about to launch'))
    assert not runner.replace_empty_waiter(args, watcher(args), status)


def test_tmux_automatically_continued_waiter_with_open_gpu_is_left_intact(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    status = write_waiting(args)
    events = []
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: watcher(args, 'S'))
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(runner, 'process_table', lambda: {123: watcher(args)})
    monkeypatch.setattr(runner, 'gpu_admission', lambda *a, **kw: dict(ready=True))
    assert not runner.replace_empty_waiter(args, watcher(args), status, timeout=0.)
    assert events == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]
    assert not (args.city_dir / 'previous_handoff.json').exists()


def test_tmux_auto_continued_empty_waiter_requires_live_closed_gpu_then_proven_exit(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    status = write_waiting(args)
    events = []
    def snapshot(pid):
        return None if (123, signal.SIGTERM) in events else watcher(args)
    monkeypatch.setattr(runner, 'process_snapshot', snapshot)
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(runner, 'process_table', lambda: {})
    admission = dict(ready=False, free_mib=1701., required_mib=8192.)
    monkeypatch.setattr(runner, 'gpu_admission', lambda *a, **kw: admission)
    assert runner.replace_empty_waiter(args, watcher(args), status, timeout=0.)
    assert events == [(123, signal.SIGSTOP), (123, signal.SIGTERM), (123, signal.SIGCONT)]
    record = json.loads((args.city_dir / 'previous_handoff.json').read_text())
    assert record['handoff_method'] == 'live_gpu_blocked_empty_waiter'
    assert record['live_admission'] == admission


def test_tmux_auto_continued_waiter_rechecks_children_after_live_gpu_check(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    status = write_waiting(args)
    events = []
    tables = iter([{}, {200: dict(pid=200, ppid=123, argv=['training'])}])
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: watcher(args))
    monkeypatch.setattr(runner, 'process_table', lambda: next(tables))
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: events.append((pid, sig)))
    monkeypatch.setattr(runner, 'gpu_admission', lambda *a, **kw: dict(ready=False))
    assert not runner.replace_empty_waiter(args, watcher(args), status, timeout=0.)
    assert events == [(123, signal.SIGSTOP), (123, signal.SIGCONT)]
    assert not (args.city_dir / 'previous_handoff.json').exists()


def test_empty_waiter_handoff_records_exit_and_can_resume(tmp_path, monkeypatch):
    args = options(tmp_path, replace=True)
    status = write_waiting(args)
    calls = iter([watcher(args), watcher(args, 'T'), None])
    events = []
    monkeypatch.setattr(runner, 'process_snapshot', lambda pid: next(calls))
    monkeypatch.setattr(runner, 'process_table', lambda: {})
    monkeypatch.setattr(runner.os, 'kill', lambda pid, sig: events.append((pid, sig)))
    assert runner.replace_empty_waiter(args, watcher(args), status)
    assert events == [(123, signal.SIGSTOP), (123, signal.SIGTERM), (123, signal.SIGCONT)]
    record = json.loads((args.city_dir / 'previous_handoff.json').read_text())
    assert record['previous_started_ticks'] == 900
    result = runner.previous_ready(args)
    assert result['ready'] and not result['source_completed']


def test_previous_completed_summary_is_required_without_handoff(tmp_path, monkeypatch):
    args = options(tmp_path)
    monkeypatch.setattr(runner, 'process_table', lambda: {})
    with pytest.raises(RuntimeError, match='without completion'):
        runner.previous_ready(args)
    args.previous_world_model_dir.mkdir()
    (args.previous_world_model_dir / 'pretrain_summary.json').write_text(json.dumps(dict(status='pretrained')))
    assert runner.previous_ready(args)['source_completed']


def test_failed_world_model_records_quality_failure_without_retry(tmp_path, monkeypatch):
    args = options(tmp_path)
    (args.city_dir / 'cache').mkdir()
    (args.city_dir / 'cache/cache_manifest.json').write_text(json.dumps(dict(status='complete',
        entries=[dict(split='train'), dict(split='validation')])))
    (args.city_dir / 'world_model').mkdir()
    (args.city_dir / 'world_model/corpus_migration.json').write_text('{}')
    (args.city_dir / 'world_model.status.json').write_text(json.dumps(dict(
        phase='corpus_codec_revalidation', codec_ready=False)))
    launched = []
    def popen(command, **kwargs):
        launched.append(command)
        kwargs['stdout'].write(b'ValueError: codec quality gate has not passed\n')
        return SimpleNamespace(pid=555, returncode=1, poll=lambda: 1)
    monkeypatch.setattr(runner, 'previous_ready', lambda a: dict(ready=True))
    monkeypatch.setattr(runner, 'gpu_admission', lambda *a, **k: dict(ready=True))
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='world_model process 555 exited 1'):
        runner.supervise(args)
    result = json.loads((args.city_dir / 'supervisor_status.json').read_text())
    assert result['phase'] == 'failed'
    assert not result['world_model_status']['codec_ready']
    assert len(launched) == 1


def test_oom_retries_preserving_checkpoint_and_then_completes(tmp_path, monkeypatch):
    args = options(tmp_path)
    (args.city_dir / 'cache').mkdir()
    (args.city_dir / 'cache/cache_manifest.json').write_text(json.dumps(dict(status='complete',
        entries=[dict(split='train'), dict(split='validation')])))
    output = args.city_dir / 'world_model'
    (output / 'checkpoints').mkdir(parents=True)
    (output / 'corpus_migration.json').write_text('{}')
    checkpoint = output / 'checkpoints/latest.pt'
    checkpoint.write_bytes(b'committed model optimizer RNG sampler')
    launched, margins = [], []
    def popen(command, **kwargs):
        launched.append(command)
        code = 1 if len(launched) == 1 else 0
        if code:
            kwargs['stdout'].write(b'torch.OutOfMemoryError: CUDA out of memory\n')
        else:
            (output / 'pretrain_summary.json').write_text(json.dumps(dict(status='pretrained')))
        return SimpleNamespace(pid=500 + len(launched), returncode=code, poll=lambda: code)
    def admission(*a, **k):
        margins.append(k['margin_mib'])
        return dict(ready=True)
    monkeypatch.setattr(runner, 'previous_ready', lambda a: dict(ready=True))
    monkeypatch.setattr(runner, 'gpu_admission', admission)
    monkeypatch.setattr(runner.subprocess, 'Popen', popen)
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    runner.supervise(args)
    assert launched[0] == launched[1]
    assert margins == [8192, 9216]
    assert checkpoint.read_bytes() == b'committed model optimizer RNG sampler'
    result = json.loads((args.city_dir / 'supervisor_status.json').read_text())
    assert result['phase'] == 'complete' and result['oom_retries'] == 1


def test_optional_memory_features_only_when_requested(tmp_path):
    args = options(tmp_path)
    args.gpu_memory_fraction = .75
    args.activation_offload = True
    args.tail_execution_batch = 32
    fm = runner.commands(args, dict(status='complete', entries=[
        dict(split='train'), dict(split='validation')]))['world_model']
    assert '--activation-offload' in fm
    assert fm[fm.index('--tail-execution-batch') + 1] == '32'
    assert fm[fm.index('--gpu-memory-fraction') + 1] == '0.75'


def test_startup_admission_does_not_reserve_entire_allocator_capacity(tmp_path, monkeypatch):
    args = options(tmp_path)
    calls = []
    def admit(device, fraction, *, margin_mib):
        calls.append((device, fraction, margin_mib))
        return dict(ready=True, free_mib=12000, total_mib=24576,
                    required_mib=margin_mib, allocator_fraction=fraction)
    monkeypatch.setattr(runner, 'gpu_admission', admit)
    result = runner.training_admission(args)
    assert calls == [('cuda:0', 0., 8192.)]
    assert result['ready'] and result['allocator_fraction'] == 1.
    assert result['required_mib'] < result['total_mib']


def test_resume_skips_old_handoff_only_after_existing_checkpoint_validation(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.resume_existing_world_model = True
    output = args.city_dir / 'world_model'
    output.mkdir()
    (output / 'corpus_migration.json').write_text('{}')
    (output / 'pretrain_summary.json').write_text(json.dumps(dict(status='pretrained')))
    monkeypatch.setattr(runner, 'checked_cache', lambda a: dict(status='complete', entries=[
        dict(split='train'), dict(split='validation')]))
    monkeypatch.setattr(runner, 'previous_ready', lambda a: pytest.fail('old handoff must not run'))
    verified = []
    monkeypatch.setattr(runner, 'validate_existing_world_model',
                        lambda a: verified.append(a) or dict(ready=True))
    runner.supervise(args)
    assert verified == [args]
    assert json.loads((args.city_dir / 'supervisor_status.json').read_text())['phase'] == 'complete'


def test_resume_validation_failure_never_launches_training(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.resume_existing_world_model = True
    monkeypatch.setattr(runner, 'checked_cache', lambda a: dict(status='complete', entries=[
        dict(split='train'), dict(split='validation')]))
    def reject(a):
        raise ValueError('corpus continuation latest updates moved backwards')
    monkeypatch.setattr(runner, 'validate_existing_world_model', reject)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **kw: pytest.fail('invalid resume launched'))
    with pytest.raises(ValueError, match='moved backwards'):
        runner.supervise(args)
