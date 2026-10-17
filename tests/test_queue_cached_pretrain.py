import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from tools import queue_cached_pretrain as queue


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


@pytest.fixture
def setup(tmp_path):
    manifest = dict(status='complete', source_storage='compact_only_v1', contract={'city': 'ingo'},
                    entries=[{'split': 'train'}, {'split': 'validation'}], train_episodes=1,
                    validation_episodes=1, plan_identity='conversion')
    manifest['identity'] = queue.identity(manifest)
    cache = write(tmp_path / 'manifest.json', manifest)
    report = dict(status='passed', city='ingo', manifest_identity=manifest['identity'],
                  manifest_sha256=hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
                  **{key: True for key in queue.FLAGS})
    verification = write(tmp_path / 'verification.json', report)
    completion = tmp_path / 'train.status.json'
    run = dict(city='ingo', command=[sys.executable, '-c', 'pass'], cwd=str(tmp_path),
               cache_manifest=cache, cache_verification=verification,
               expected_train_episodes=1, expected_validation_episodes=1,
               expected_plan_identity='conversion', completion_status_path=str(completion))
    cleanup = write(tmp_path / 'cleanup.json', dict(stage='eligible_cleanup_complete'))
    plan = dict(schema_version=1, cleanup_status_path=cleanup, runs=[run],
                blocking_queue_status_paths=[], gpu=dict(index=1, required_mib=100),
                disk=dict(path=str(tmp_path), min_free_gb=0), poll_seconds=1)
    return plan, run, tmp_path


def test_cache_identity_and_verification_binding(setup):
    _, run, _ = setup
    assert queue.validate_cache(run)
    report = queue.read(run['cache_verification'])
    report['manifest_sha256'] = 'wrong'
    write(Path(run['cache_verification']), report)
    with pytest.raises(ValueError, match='proof mismatch'):
        queue.validate_cache(run)


@pytest.mark.parametrize('mutation', ['count', 'identity', 'raw', 'flag', 'plan'])
def test_cache_rejects_invalid_contracts(setup, mutation):
    _, run, _ = setup
    manifest = queue.read(run['cache_manifest'])
    if mutation == 'count':
        run['expected_train_episodes'] = 2
    elif mutation == 'identity':
        manifest['identity'] = 'wrong'
        write(Path(run['cache_manifest']), manifest)
    elif mutation == 'raw':
        manifest['source_storage'] = 'raw_required'
        write(Path(run['cache_manifest']), manifest)
    elif mutation == 'flag':
        report = queue.read(run['cache_verification'])
        report[queue.FLAGS[0]] = False
        write(Path(run['cache_verification']), report)
    else:
        run['expected_plan_identity'] = 'different'
    with pytest.raises(ValueError):
        queue.validate_cache(run)


def test_cleanup_and_blocking_queue_must_complete(setup):
    plan, _, tmp = setup
    write(Path(plan['cleanup_status_path']), {'stage': 'deleting'})
    assert queue.prerequisites(plan) == 'waiting_for_cleanup'
    write(Path(plan['cleanup_status_path']), {'stage': 'eligible_cleanup_complete'})
    blocker = tmp / 'blocker.json'
    plan['blocking_queue_status_paths'] = [str(blocker)]
    assert 'readable_prerequisites' in queue.prerequisites(plan)
    for records in ({}, {'a': {'status': 'running'}}, [{'status': 'queued'}]):
        write(blocker, {'runs': records})
        assert 'waiting_for_queue' in queue.prerequisites(plan)
    write(blocker, {'runs': {'a': {'status': 'completed'}, 'b': {'status': 'stopped'}}})
    with patch.object(queue.subprocess, 'check_output', side_effect=['', '500']):
        assert queue.prerequisites(plan) is None


def test_gpu_and_disk_admission(setup):
    plan, _, _ = setup
    with patch.object(queue.subprocess, 'check_output', return_value='1234\n'):
        assert queue.prerequisites(plan) == 'waiting_for_gpu_processes'
    with patch.object(queue.subprocess, 'check_output', side_effect=['', '99']):
        assert queue.prerequisites(plan) == 'waiting_for_gpu_memory'
    with patch.object(queue.subprocess, 'check_output', side_effect=FileNotFoundError()):
        assert 'readable_prerequisites' in queue.prerequisites(plan)
    plan['disk']['min_free_gb'] = 10**9
    assert queue.prerequisites(plan) == 'waiting_for_disk'


def test_runner_requires_exit_zero_and_completion_proof(setup):
    _, run, tmp = setup
    spec = dict(run=run, result_path=str(tmp / 'result.json'), log_path=str(tmp / 'log'), plan_identity='test')
    spec_path = Path(write(tmp / 'spec.json', spec))
    assert queue.execute_child(spec_path) == 1  # A successful command alone is not enough.
    write(Path(run['completion_status_path']), {'phase': 'pretraining_completed'})
    assert queue.execute_child(spec_path) == 0
    result = queue.read(spec['result_path'])
    assert result['exit_code'] == 0 and result['status'] == 'completed'
    spec['run']['command'] = [sys.executable, '-c', 'raise SystemExit(3)']
    write(spec_path, spec)
    assert queue.execute_child(spec_path) == 1
    assert queue.read(spec['result_path'])['exit_code'] == 3


def test_restart_does_not_relaunch_live_child(setup):
    plan, _, tmp = setup
    output = tmp / 'queue'
    output.mkdir()
    plan_path = write(tmp / 'plan.json', plan)
    write(output / 'ingo.result.json', dict(status='running', pid=os.getpid(),
        process_identity=queue.process_identity(os.getpid()), plan_identity=queue.identity(plan)))
    with patch.object(queue.subprocess, 'Popen') as launch, patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            queue.run_queue(plan_path, output)
        launch.assert_not_called()
    assert queue.read(output / 'status.json')['status'] == 'running'


def test_dead_runner_stops_queue_without_retry(setup):
    plan, _, tmp = setup
    output = tmp / 'queue'
    output.mkdir()
    plan_path = write(tmp / 'plan.json', plan)
    write(output / 'ingo.result.json', dict(status='running', pid=999999999,
        process_identity='missing', plan_identity=queue.identity(plan)))
    with patch.object(queue.subprocess, 'Popen') as launch:
        assert queue.run_queue(plan_path, output) == 1
        launch.assert_not_called()
    assert queue.read(output / 'status.json')['status'] == 'failed'


def test_completed_restart_and_no_stale_status_launch(setup):
    plan, run, tmp = setup
    output = tmp / 'queue'
    output.mkdir()
    plan_path = write(tmp / 'plan.json', plan)
    write(Path(run['completion_status_path']), {'phase': 'pretraining_completed'})
    with patch.object(queue, 'prerequisites', return_value=None):
        assert queue.run_queue(plan_path, output) == 1
    write(output / 'ingo.result.json', dict(status='completed', exit_code=0, plan_identity=queue.identity(plan)))
    with patch.object(queue.subprocess, 'Popen') as launch:
        assert queue.run_queue(plan_path, output) == 0
        launch.assert_not_called()


def test_plan_freeze_and_lock_exclusion(setup):
    plan, _, tmp = setup
    output = tmp / 'queue'
    output.mkdir()
    plan_path = write(tmp / 'plan.json', plan)
    write(output / 'plan.json', {'different': True})
    with pytest.raises(ValueError, match='plan changed'):
        queue.run_queue(plan_path, output)
    with (output / 'queue.lock').open('a') as lock:
        queue.fcntl.flock(lock, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.run_queue(plan_path, output)


def test_serial_launch_with_durable_completion(setup):
    plan, run, tmp = setup
    second = dict(run, city='second', completion_status_path=str(tmp / 'second.status.json'))
    plan['runs'].append(second)
    plan_path = write(tmp / 'plan.json', plan)
    output = tmp / 'queue'
    launched = []
    class Child:
        pid = 123
        def poll(self):
            return 0
    def launch(argv, **kwargs):
        spec = queue.read(argv[-1])
        city = spec['run']['city']
        if launched:
            assert queue.read(output / f'{launched[-1]}.result.json')['status'] == 'completed'
        launched.append(city)
        write(Path(spec['run']['completion_status_path']), {'phase': 'pretraining_completed'})
        write(Path(spec['result_path']), dict(status='completed', exit_code=0, plan_identity=queue.identity(plan)))
        return Child()
    with patch.object(queue, 'prerequisites', return_value=None), patch.object(queue, 'validate_cache', return_value='verified'), patch.object(queue.subprocess, 'Popen', side_effect=launch), patch.object(queue.time, 'sleep'):
        assert queue.run_queue(plan_path, output) == 0
    assert launched == ['ingo', 'second']


def test_parallel_launch_and_unready_city_does_not_block(setup):
    plan, run, tmp = setup
    plan['allow_gpu_sharing'] = True
    plan['runs'] += [dict(run, city='second', completion_status_path=str(tmp / 'second.status.json')),
                     dict(run, city='third', completion_status_path=str(tmp / 'third.status.json'))]
    plan_path = write(tmp / 'plan.json', plan)
    output = tmp / 'queue'
    launched = []
    class Child:
        pid = 123
        def poll(self):
            return None
    def launch(argv, **kwargs):
        spec = queue.read(argv[-1])
        launched.append(spec['run']['city'])
        return Child()
    def verify(run):
        if run['city'] == 'ingo':
            raise FileNotFoundError()
        return 'verified'
    with patch.object(queue, 'prerequisites', return_value=None), patch.object(queue, 'validate_cache', side_effect=verify), patch.object(queue.subprocess, 'Popen', side_effect=launch), patch.object(queue.subprocess, 'check_output', return_value=''), patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            queue.run_queue(plan_path, output)
    assert launched == ['second', 'third']
    states = queue.read(output / 'status.json')['runs']
    assert states['ingo']['reason'] == 'waiting_for_verified_city_cache'
    assert states['second']['status'] == states['third']['status'] == 'running'


def test_pending_reservation_prevents_oversubscription(setup):
    plan, run, _ = setup
    plan['allow_gpu_sharing'] = True
    runs = {'ingo': {'status': 'running'}}
    with patch.object(queue.subprocess, 'check_output', return_value=''):
        reserved = queue.reserved_memory(plan, runs, 1)
    assert reserved == 100
    with patch.object(queue.subprocess, 'check_output', return_value='199'):
        assert queue.prerequisites(plan, run, reserved) == 'waiting_for_gpu_memory'
    with patch.object(queue.subprocess, 'check_output', return_value='200'):
        assert queue.prerequisites(plan, run, reserved) is None
    runs['ingo'].update(child_pid=123, child_process_identity='current')
    with patch.object(queue.subprocess, 'check_output', return_value='123, 60'), patch.object(queue, 'process_identity', return_value='current'):
        assert queue.reserved_memory(plan, runs, 1) == 40
    with patch.object(queue.subprocess, 'check_output', return_value='123, 60'), patch.object(queue, 'process_identity', return_value='reused'):
        assert queue.reserved_memory(plan, runs, 1) == 100
    assert queue.reserved_memory(plan, runs, 0) == 0


def test_gpu_override_and_sharing(setup):
    plan, run, _ = setup
    plan['allow_gpu_sharing'] = True
    run['gpu'] = dict(index=0, required_mib=200, allocator_fraction=.5)
    queue.validate_plan(plan)
    with patch.object(queue.subprocess, 'check_output', return_value='199') as query:
        assert queue.prerequisites(plan, run) == 'waiting_for_gpu_memory'
        assert query.call_args.args[0][2] == '0'
    run['gpu']['allocator_fraction'] = 1.1
    with pytest.raises(ValueError):
        queue.validate_plan(plan)


def test_priority_release_only_matching_stopped_controller(setup):
    plan, _, _ = setup
    plan['priority_held_controllers'] = [dict(pid=123, process_identity='current')]
    with patch.object(Path, 'read_text', return_value='123 (controller) T 0'), patch.object(queue, 'process_alive', return_value=True), patch.object(queue.os, 'kill') as kill:
        assert queue.resume_held_controllers(plan)[0]['action'] == 'resumed'
        kill.assert_called_once_with(123, queue.signal.SIGCONT)
    with patch.object(Path, 'read_text', return_value='123 (controller) T 0'), patch.object(queue, 'process_alive', return_value=False), patch.object(queue.os, 'kill') as kill:
        queue.resume_held_controllers(plan)
        kill.assert_not_called()
    with patch.object(Path, 'read_text', return_value='123 (controller) S 0'), patch.object(queue, 'process_alive', return_value=True), patch.object(queue.os, 'kill') as kill:
        queue.resume_held_controllers(plan)
        kill.assert_not_called()


def test_failed_run_releases_controllers(setup):
    plan, run, tmp = setup
    write(Path(run['completion_status_path']), {'phase': 'stale'})
    plan_path = write(tmp / 'plan.json', plan)
    with patch.object(queue, 'prerequisites', return_value=None), patch.object(queue, 'resume_held_controllers', return_value=[]) as resume:
        assert queue.run_queue(plan_path, tmp / 'queue') == 1
        resume.assert_called_once_with(plan)


def test_only_lock_owner_releases_controllers_on_fatal(setup):
    plan, _, tmp = setup
    plan_path = write(tmp / 'plan.json', plan)
    output = tmp / 'queue'
    output.mkdir()
    with (output / 'queue.lock').open('a') as lock:
        queue.fcntl.flock(lock, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
        with patch.object(queue, 'resume_held_controllers') as resume:
            with pytest.raises(BlockingIOError):
                queue.run_queue(plan_path, output)
            resume.assert_not_called()
    with patch.object(queue, 'prerequisites', side_effect=RuntimeError('unexpected fault')), patch.object(queue, 'resume_held_controllers', return_value=[]) as resume:
        with pytest.raises(RuntimeError, match='unexpected fault'):
            queue.run_queue(plan_path, output)
        resume.assert_called_once_with(plan)


def test_orphan_live_child_keeps_reservation_and_priority(setup):
    plan, _, tmp = setup
    output = tmp / 'queue'
    output.mkdir()
    plan_path = write(tmp / 'plan.json', plan)
    write(output / 'ingo.result.json', dict(status='running', pid=999999999,
        process_identity='missing', child_pid=os.getpid(),
        child_process_identity=queue.process_identity(os.getpid()), plan_identity=queue.identity(plan)))
    with patch.object(queue.subprocess, 'Popen') as launch, patch.object(queue, 'resume_held_controllers') as resume, patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            queue.run_queue(plan_path, output)
        launch.assert_not_called()
        resume.assert_not_called()
    assert queue.read(output / 'status.json')['runs']['ingo']['status'] == 'running'


@pytest.mark.parametrize('restart', [False, True])
def test_adopted_live_runner_is_monitored_without_launch(setup, restart):
    plan, run, tmp = setup
    receipt = tmp / 'previous.result.json'
    run.update(adopted_result_path=str(receipt), adopted_plan_identity='previous-plan')
    write(receipt, dict(status='running', pid=os.getpid(),
        process_identity=queue.process_identity(os.getpid()), plan_identity='previous-plan'))
    original = receipt.read_bytes()
    plan_path = write(tmp / 'plan.json', plan)
    output = tmp / 'queue'
    if restart:
        output.mkdir()
        write(output / 'plan.json', plan)
    with patch.object(queue.subprocess, 'Popen') as launch, patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            queue.run_queue(plan_path, output)
        launch.assert_not_called()
    assert queue.read(output / 'status.json')['runs']['ingo']['status'] == 'running'
    assert receipt.read_bytes() == original


@pytest.mark.parametrize('adopted', [False, True])
def test_foreign_receipt_identity_rejected(setup, adopted):
    plan, run, tmp = setup
    output = tmp / 'queue'
    output.mkdir()
    receipt = output / 'ingo.result.json'
    if adopted:
        receipt = tmp / 'adopted.json'
        run.update(adopted_result_path=str(receipt), adopted_plan_identity='old-plan')
    write(receipt, dict(status='running', plan_identity='foreign'))
    plan_path = write(tmp / 'plan.json', plan)
    with patch.object(queue.subprocess, 'Popen') as launch:
        with pytest.raises(ValueError, match='another plan'):
            queue.run_queue(plan_path, output)
        launch.assert_not_called()


def test_missing_adopted_receipt_never_launches(setup):
    plan, run, tmp = setup
    run.update(adopted_result_path=str(tmp / 'missing.json'), adopted_plan_identity='old-plan')
    plan_path = write(tmp / 'plan.json', plan)
    with patch.object(queue.subprocess, 'Popen') as launch, patch.object(queue.time, 'sleep', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            queue.run_queue(plan_path, tmp / 'queue')
        launch.assert_not_called()
    record = queue.read(tmp / 'queue/status.json')['runs']['ingo']
    assert record['launch_pending'] and record['reason'].startswith('adopted_receipt_missing')


@pytest.mark.parametrize('state', ['completed', 'failed'])
def test_adopted_terminal_result(setup, state):
    plan, run, tmp = setup
    receipt = tmp / 'previous.result.json'
    run.update(adopted_result_path=str(receipt), adopted_plan_identity='old-plan')
    write(receipt, dict(status=state, exit_code=0 if state == 'completed' else 1,
                        plan_identity='old-plan'))
    write(Path(run['completion_status_path']), {'phase': 'pretraining_completed'})
    plan_path = write(tmp / 'plan.json', plan)
    with patch.object(queue.subprocess, 'Popen') as launch:
        assert queue.run_queue(plan_path, tmp / 'queue') == int(state == 'failed')
        launch.assert_not_called()


@pytest.mark.parametrize('fields', [
    {'adopted_result_path': '/tmp/receipt'},
    {'adopted_plan_identity': 'old'},
    {'adopted_result_path': 'relative', 'adopted_plan_identity': 'old'},
    {'adopted_result_path': '/tmp/receipt', 'adopted_plan_identity': ''},
])
def test_adoption_requires_absolute_path_and_paired_identity(setup, fields):
    plan, run, _ = setup
    run.update(fields)
    with pytest.raises(ValueError, match='adopted'):
        queue.validate_plan(plan)
