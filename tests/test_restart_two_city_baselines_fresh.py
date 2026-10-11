"""Fresh restart transformations and process isolation, with no live signals."""
import copy
import json
from pathlib import Path
import signal

import pytest

from tools import restart_two_city_baselines_fresh as restart


def batch(tmp_path):
    source = tmp_path / 'old/source'
    entries, jobs = [], {}
    for city in restart.queue.CITIES:
        for method in restart.queue.LEARNED:
            run_id = f'{city}_{method}'
            output = str(tmp_path / 'old' / city / method)
            command = ['python', '-u', str(source / 'tools' / ('train_toy_actor_critic.py' if method == 'online_ac' else 'train_toy_baseline.py')), '--config',
                str(source / 'configs' / f'{city}.toml'), '--output-dir', output,
                '--episodes', '5000', '--global-reward-only', '--reward-objective',
                'observed_mean', '--sp-prior-weight', '0.0', '--evaluation-argmax',
                '--resume', '--skip-initial-evaluation', '--skip-reference-baselines']
            entries.append(dict(id=run_id, city=city, method=method, command=command,
                output_dir=output, episode_budget=5000, required_gpu_mib=2048,
                state=f'original/{run_id}.json', log=f'original/{run_id}.log',
                continuation_episode=123))
            jobs[run_id] = dict(status='running', attempt=4, required_gpu_mib=4096,
                observed_peak_mib=3000, oom_retries=2, pid=100+len(entries),
                process_starttime='123', next_eligible=100, resume=True,
                memory_calibration_episode=123, continuation_episode=123)
    entries.append(dict(id='coulommiers_shortest_ref', city='coulommiers',
                        method='shortest_ref', episode_budget=6, output_dir='old/ref',
                        command=['python', '-u', 'ref.py']))
    jobs['coulommiers_shortest_ref'] = dict(status='completed', pid=None)
    return dict(runs=entries, source_snapshot=str(source), input_hashes_by_city={'test':'abc'},
                min_free_gb=20, poll_seconds=5, gpu_headroom_mib=512), dict(jobs=jobs, controller_pid=9)


def test_exactly_sixteen_fresh_jobs_and_mixed_budgets(tmp_path):
    old, ledger = batch(tmp_path)
    before, before_state = copy.deepcopy(old), copy.deepcopy(ledger)
    root = tmp_path / 'new'
    manifest, state, archived = restart.build_manifest(old, ledger, root)
    assert old == before and ledger == before_state
    assert len(archived) == 16
    assert manifest['total_training_episodes'] == 80000
    assert manifest['episode_budget'] is None
    assert manifest['episode_budget_by_city']['ingolstadt_custom'] == [5000]
    assert manifest['episode_budget_by_city']['coulommiers'] == [2500]
    assert manifest['min_free_gb'] == 20
    assert manifest['launch_status'] == 'stopped'
    for prior, entry in zip(old['runs'], manifest['runs']):
        job = state['jobs'][entry['id']]
        if restart.selected(entry):
            assert entry['episode_budget'] == 2500
            assert restart.flag_value(entry['command'], '--episodes') == '2500'
            assert restart.flag_value(entry['command'], '--output-dir') == entry['output_dir']
            assert restart.flag_value(entry['command'], '--config').startswith(str(root / 'source'))
            assert '--resume' not in entry['command']
            assert '--skip-initial-evaluation' not in entry['command']
            assert '--skip-reference-baselines' in entry['command']
            assert 'continuation_episode' not in entry
            assert job == dict(status='queued', attempt=0, required_gpu_mib=4096,
                               observed_peak_mib=0., oom_retries=0, pid=None, next_eligible=0.)
        else:
            assert entry == prior
            assert job == ledger['jobs'][entry['id']]


@pytest.mark.parametrize('mutation', ['reward', 'missing', 'prior', 'script'])
def test_rejects_wrong_experiment(tmp_path, mutation):
    manifest, ledger = batch(tmp_path)
    selected = next(e for e in manifest['runs'] if restart.selected(e))
    command = selected['command']
    if mutation == 'reward':
        command[command.index('--reward-objective') + 1] = 'other'
    elif mutation == 'missing':
        manifest['runs'].remove(selected)
    elif mutation == 'prior':
        command[command.index('--sp-prior-weight') + 1] = '0.1'
    else:
        command[2] = '/unrelated.py'
    with pytest.raises(ValueError):
        restart.build_manifest(manifest, ledger, tmp_path / 'new')


def test_empty_outputs_reject_links_and_checkpoints(tmp_path):
    old, ledger = batch(tmp_path)
    root = tmp_path / 'new'
    manifest, _, _ = restart.build_manifest(old, ledger, root)
    output = Path(next(e for e in manifest['runs'] if restart.selected(e))['output_dir'])
    output.mkdir(parents=True)
    restart.ensure_empty_outputs(root, manifest)
    (output / 'latest.pt').write_bytes(b'old')
    with pytest.raises(ValueError, match='not empty'):
        restart.ensure_empty_outputs(root, manifest)
    (output / 'latest.pt').unlink()
    output.rmdir()
    target = tmp_path / 'existing'
    target.mkdir()
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match='linked'):
        restart.ensure_empty_outputs(root, manifest)


def test_source_hashes_and_symlinks(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'train.py').write_bytes(b'original source')
    hashes = {'train.py': restart.digest(source / 'train.py')}
    restart.verify_source(source, hashes)
    (source / 'train.py').write_bytes(b'mutated source')
    with pytest.raises(ValueError, match='hash mismatch'):
        restart.verify_source(source, hashes)
    (source / 'train.py').write_bytes(b'original source')
    (source / 'link').symlink_to(source / 'train.py')
    with pytest.raises(ValueError, match='symlinks'):
        restart.verify_source(source, hashes)


def test_stop_only_selected_owned_workers(tmp_path, monkeypatch):
    manifest, ledger = batch(tmp_path)
    alive = {s['pid'] for s in ledger['jobs'].values() if s['pid'] is not None}
    monkeypatch.setattr(restart.queue, 'owned_alive', lambda s, e: s.get('pid') in alive)
    signals = []
    def send(pid, sig, predicate):
        assert predicate()
        signals.append((pid, sig))
        alive.discard(pid)
    monkeypatch.setattr(restart, 'checked_signal', send)
    restart.stop_selected(manifest['runs'], ledger['jobs'], grace_seconds=0)
    expected = {ledger['jobs'][e['id']]['pid'] for e in manifest['runs'] if restart.selected(e)}
    assert {pid for pid, _ in signals} == expected
    assert all(sig == signal.SIGTERM for _, sig in signals)
    assert alive == {ledger['jobs'][e['id']]['pid'] for e in manifest['runs']
                     if e['city'] == 'ingolstadt_custom'}


def test_pid_identity_mismatch_prevents_all_worker_signals(tmp_path, monkeypatch):
    manifest, ledger = batch(tmp_path)
    monkeypatch.setattr(restart.queue, 'owned_alive', lambda s, e: False)
    monkeypatch.setattr(restart.queue, 'process_identity', lambda pid: 'different-starttime')
    monkeypatch.setattr(restart, 'checked_signal', lambda *args: pytest.fail('unexpected signal'))
    with pytest.raises(RuntimeError, match='identity mismatch'):
        restart.stop_selected(manifest['runs'], ledger['jobs'])


def test_reapply_rejected_without_signals(tmp_path, monkeypatch):
    (tmp_path / 'restart_plan.json').write_text(json.dumps({'status':'applied'}))
    monkeypatch.setattr(restart, 'checked_signal', lambda *args: pytest.fail('unexpected signal'))
    with pytest.raises(ValueError, match='already applied'):
        restart.apply(tmp_path)


def test_untracked_worker_in_freeze_window_blocks_handoff(tmp_path, monkeypatch):
    manifest, ledger = batch(tmp_path)
    entry = next(e for e in manifest['runs'] if restart.selected(e))
    unknown_pid = 9876
    monkeypatch.setattr(Path, 'iterdir', lambda self: iter([Path(f'/proc/{unknown_pid}')]))
    monkeypatch.setattr(restart.queue, 'process_argv', lambda pid: entry['command'])
    with pytest.raises(RuntimeError, match='untracked selected worker'):
        restart.verify_worker_inventory(manifest['runs'], ledger['jobs'])


def test_rechecks_identity_before_sending(monkeypatch):
    monkeypatch.setattr(restart.os, 'kill',
                        lambda *args: pytest.fail('stale identity must not receive signal'))
    with pytest.raises(RuntimeError, match='identity changed'):
        restart.checked_signal(9999, signal.SIGSTOP, lambda: False)


def test_failed_selected_older_source_and_default_argmax(tmp_path):
    old, ledger = batch(tmp_path)
    entry = next(e for e in old['runs'] if e['id'] == 'coulommiers_mappo')
    entry['command'][2] = '/historical/source/tools/train_toy_baseline.py'
    entry['command'][entry['command'].index('--config') + 1] = '/historical/source/configs/coulommiers.toml'
    entry['command'].remove('--evaluation-argmax')
    ledger['jobs'][entry['id']]['status'] = 'failed'
    manifest, state, _ = restart.build_manifest(old, ledger, tmp_path / 'new')
    fresh = next(e for e in manifest['runs'] if e['id'] == entry['id'])
    assert fresh['command'][2] == str(tmp_path / 'new/source/tools/train_toy_baseline.py')
    assert '--evaluation-argmax' in fresh['command']
    assert state['jobs'][entry['id']]['status'] == 'queued'


def test_sampling_flags_normalized_to_argmax(tmp_path):
    old, ledger = batch(tmp_path)
    entry = next(e for e in old['runs'] if restart.selected(e))
    entry['command'] += ['--evaluation-sampled', '--evaluation-temperature', '0.2']
    manifest, _, _ = restart.build_manifest(old, ledger, tmp_path / 'new')
    fresh = next(e for e in manifest['runs'] if e['id'] == entry['id'])
    assert fresh['command'].count('--evaluation-argmax') == 1
    assert '--evaluation-temperature' not in fresh['command']
    assert '--evaluation-sampled' not in fresh['command']
