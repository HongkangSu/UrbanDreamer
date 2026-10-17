"""Extend only the existing route baseline batch, preserving its checkpoint identities."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue

BATCH = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
FILES = ('tools/train_toy_actor_critic.py', 'tools/train_toy_baseline.py',
         'urban_dreamer/learning/entropy_schedule.py')


def stop_owned(pid, identity, expected):
    if queue.process_identity(pid) is None:
        return
    assert queue.process_identity(pid) == identity
    assert expected in queue.process_argv(pid)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while queue.process_identity(pid) == identity and time.monotonic() < deadline:
        time.sleep(.1)
    if queue.process_identity(pid) == identity:
        assert expected in queue.process_argv(pid)
        os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 3
    while queue.process_identity(pid) == identity and time.monotonic() < deadline:
        time.sleep(.1)
    assert queue.process_identity(pid) != identity


def main():
    manifest = json.loads((BATCH / 'manifest.json').read_text())
    assert manifest['episode_budget'] == 1000
    source = Path(manifest['source_snapshot'])
    hashes = json.loads((BATCH / 'source_hashes.json').read_text())
    assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == digest
               for p, digest in hashes.items())
    ledger = json.loads((BATCH / 'queue_state.json').read_text())
    controller = ledger['controller_pid']
    identity = queue.process_identity(controller)
    assert identity and str(BATCH) in queue.process_argv(controller)
    assert str(source / 'tools/queued_city_baselines.py') in queue.process_argv(controller)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup = BATCH / 'budget_extensions' / stamp
    backup.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json', 'queue_state.json', 'source_hashes.json',
                 'runtime_estimates_current.json', 'admission_overrides.json'):
        shutil.copy2(BATCH / name, backup / name)
    for relative in FILES:
        target = backup / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    stop_owned(controller, identity, str(BATCH))
    ledger = json.loads((BATCH / 'queue_state.json').read_text())
    stopped = []
    for entry in manifest['runs']:
        state = ledger['jobs'][entry['id']]
        if queue.owned_alive(state, entry):
            assert os.getpgid(state['pid']) == state['pid']
            stopped.append(dict(id=entry['id'], pid=state['pid'], starttime=state['process_starttime']))
            os.kill(state['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while any(queue.owned_alive(ledger['jobs'][e['id']], e) for e in manifest['runs']) and time.monotonic() < deadline:
        time.sleep(.1)
    for entry in manifest['runs']:
        state = ledger['jobs'][entry['id']]
        if queue.owned_alive(state, entry):
            os.killpg(state['pid'], signal.SIGKILL)
    time.sleep(.2)
    assert not any(queue.owned_alive(ledger['jobs'][e['id']], e) for e in manifest['runs'])
    cursors = {}
    old_forecast = json.loads((backup / 'runtime_estimates_current.json').read_text())
    previous = {row['id']: row for row in old_forecast['jobs']}
    estimates = []
    for entry in manifest['runs']:
        if entry['method'].endswith('_ref'):
            assert ledger['jobs'][entry['id']]['status'] == 'completed'
            continue
        run_id = entry['id']
        boundary = queue.durable_episode(entry)
        assert 0 < boundary <= 1000 and (Path(entry['output_dir']) / 'latest.pt').is_file()
        cursors[run_id] = boundary
        state = ledger['jobs'][run_id]
        prior = previous[run_id]
        finish = datetime.fromisoformat(state.get('completed_at', queue.now()))
        elapsed = (finish - datetime.fromisoformat(state['started_at'])).total_seconds() / 3600
        rate = (boundary - prior['resume_episode']) / elapsed
        estimates.append(dict(id=run_id, city=entry['city'], method=entry['method'],
            episode=boundary, episodes_per_hour=rate,
            remaining_hours_at_observed_rate=(2000-boundary)/rate))
        if entry['method'] == 'online_ac':
            checkpoint_backup = backup / 'checkpoints' / run_id
            checkpoint_backup.mkdir(parents=True)
            for name in ('latest.pt', 'best.pt'):
                os.link(Path(entry['output_dir']) / name, checkpoint_backup / name)
        command = entry['command']
        assert command[command.index('--episodes') + 1] == '1000'
        command[command.index('--episodes') + 1] = '2000'
        assert '--entropy-schedule-episodes' not in command
        command.extend(['--entropy-schedule-episodes', '1000'])
        entry['episode_budget'] = 2000
        state.update(status='queued_recovery', pid=None, process_starttime=None, next_eligible=0.,
                     budget_extension=True)
        for key in ('completed_at', 'exit_code', 'failed_at', 'last_error'):
            state.pop(key, None)
        queue.write_json(Path(entry['output_dir']) / 'budget_extension.json', dict(
            updated_at=queue.now(), restored_episode=boundary, episode_budget=2000,
            entropy_schedule_episodes=1000, backup=str(backup),
            config_identity='original frozen config retained'))
    assert len(cursors) == 27
    changes = {}
    for relative in FILES:
        shutil.copy2(ROOT / relative, source / relative)
        digest = hashlib.sha256((source / relative).read_bytes()).hexdigest()
        changes[relative] = dict(before=hashes[relative], after=digest)
        hashes[relative] = digest
    assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == digest
               for p, digest in hashes.items())
    session = f'ud_route_2000_{stamp}'
    manifest.update(episode_budget=2000, total_training_episodes=54000,
        entropy_schedule_episodes=1000, budget_extended_at=queue.now(),
        budget_extension_backup=str(backup), controller_session=session)
    ledger['controller_pid'] = None
    for name, value in [('source_hashes.json', hashes), ('manifest.json', manifest), ('queue_state.json', ledger)]:
        queue.write_json(BATCH / name, value)
    if (BATCH / 'final_status.json').exists():
        shutil.move(BATCH / 'final_status.json', backup / 'final_status.json')
    cities = {}
    for city in queue.CITIES:
        rows = [row for row in estimates if row['city'] == city]
        ac = next(row for row in rows if row['method'] == 'online_ac')
        slowest = max(rows, key=lambda row: row['remaining_hours_at_observed_rate'])
        cities[city] = dict(online_ac_remaining_hours=round(ac['remaining_hours_at_observed_rate'], 1),
            slowest_method=slowest['method'],
            all_methods_remaining_hours=round(slowest['remaining_hours_at_observed_rate'], 1))
    queue.write_json(BATCH / 'runtime_estimates_current.json', dict(updated_at=queue.now(),
        episode_budget=2000, basis='Measured durable episode throughput since cache16384 restart, including periodic evaluation and checkpoint overhead. Original parallel load; estimates may change after restart. Final holdouts add time.',
        cities=cities, jobs=estimates))
    record = dict(updated_at=queue.now(), original_episode_budget=1000, episode_budget=2000,
        entropy_schedule_episodes=1000, stopped_workers=stopped, resume_episodes=cursors,
        updated_source_files=changes, unchanged_config_and_input_hashes=True,
        source_files_verified=len(hashes), session=session, backup=str(backup))
    queue.write_json(BATCH / 'budget_extension_2000.json', record)
    command = [manifest['runs'][0]['command'][0], '-u', str(source / 'tools/queued_city_baselines.py'),
               'run', '--output-root', str(BATCH)]
    shell = shlex.join(command) + ' >> ' + shlex.quote(str(BATCH / 'controller.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(source), shell], check=True)
    print(json.dumps(dict(status='extended_and_controller_launched', episode_budget=2000,
        training_jobs=len(cursors), stopped_workers=len(stopped), resume_episodes=cursors,
        remaining_hours=cities, session=session)), flush=True)


if __name__ == '__main__':
    main()
