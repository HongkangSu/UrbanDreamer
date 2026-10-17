"""Deploy a larger route cache and resume this batch's 27 trainers."""
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
from tmp.extend_route_baselines_2000_20261014 import BATCH, stop_owned


def main():
    manifest = json.loads((BATCH / 'manifest.json').read_text())
    source = Path(manifest['source_snapshot'])
    hashes = json.loads((BATCH / 'source_hashes.json').read_text())
    assert all(hashlib.sha256((source / path).read_bytes()).hexdigest() == digest
               for path, digest in hashes.items())
    assert manifest['episode_budget'] == 2000 and manifest['route_candidate_cache_capacity'] == 16384
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    backup = BATCH / 'cache_restarts' / f'cache200000_{stamp}'
    backup.mkdir(parents=True, exist_ok=False)
    for name in ('manifest.json', 'queue_state.json', 'source_hashes.json', 'runtime_estimates_current.json'):
        shutil.copy2(BATCH / name, backup / name)
    relative = 'urban_dreamer/traffic/route_candidates.py'
    shutil.copy2(source / relative, backup / 'route_candidates.py')
    ledger = json.loads((BATCH / 'queue_state.json').read_text())
    controller = ledger['controller_pid']
    identity = queue.process_identity(controller)
    assert identity and str(source / 'tools/queued_city_baselines.py') in queue.process_argv(controller)
    stop_owned(controller, identity, str(BATCH))
    ledger = json.loads((BATCH / 'queue_state.json').read_text())
    stopped = []
    for entry in manifest['runs']:
        state = ledger['jobs'][entry['id']]
        if queue.owned_alive(state, entry):
            assert os.getpgid(state['pid']) == state['pid']
            stopped.append(dict(id=entry['id'], pid=state['pid'], identity=state['process_starttime']))
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
    for entry in manifest['runs']:
        state = ledger['jobs'][entry['id']]
        if entry['method'].endswith('_ref'):
            assert state['status'] == 'completed'
            continue
        assert entry['episode_budget'] == 2000
        command = entry['command']
        assert command[command.index('--episodes') + 1] == '2000'
        assert command[command.index('--entropy-schedule-episodes') + 1] == '1000'
        boundary = queue.durable_episode(entry)
        assert 0 < boundary < 2000
        cursors[entry['id']] = boundary
        if entry['method'] == 'online_ac':
            target = backup / entry['id']
            target.mkdir()
            for name in ('latest.pt', 'best.pt'):
                os.link(Path(entry['output_dir']) / name, target / name)
        state.update(status='queued_recovery', pid=None, process_starttime=None,
                     next_eligible=0., cache_capacity_restart=True)
        for key in ('completed_at', 'exit_code', 'failed_at', 'last_error'):
            state.pop(key, None)
        entry['route_candidate_cache_capacity'] = 200000
        queue.write_json(Path(entry['output_dir']) / 'cache_runtime.json', dict(
            updated_at=queue.now(), cache_capacity=200000, restored_episode=boundary,
            retained_across_episodes=True, retained_across_process_restart=False,
            target_episodes=2000, entropy_schedule_episodes=1000))
    assert len(cursors) == 27
    original_hash = hashes[relative]
    shutil.copy2(ROOT / relative, source / relative)
    hashes[relative] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
    assert '_route_candidate_cache_limit = 200000' in (source / relative).read_text()
    assert all(hashlib.sha256((source / path).read_bytes()).hexdigest() == digest
               for path, digest in hashes.items())
    session = f'ud_route_cache200000_{stamp}'
    manifest.update(route_candidate_cache_capacity=200000, cache_restarted_at=queue.now(),
                    controller_session=session)
    ledger['controller_pid'] = None
    for name, value in [('source_hashes.json', hashes), ('manifest.json', manifest), ('queue_state.json', ledger)]:
        queue.write_json(BATCH / name, value)
    forecast = json.loads((BATCH / 'runtime_estimates_current.json').read_text())
    forecast.update(cache_capacity=200000, needs_recalibration=True,
        timing_basis_cache_capacity=16384,
        cache_note='Existing estimates measured under cache16384. Recalibrate after fresh cache200000 workers warm across several episodes.')
    queue.write_json(BATCH / 'runtime_estimates_current.json', forecast)
    queue.write_json(BATCH / 'cache_restart_200000.json', dict(updated_at=queue.now(),
        original_capacity=16384, capacity=200000, episode_budget=2000,
        entropy_schedule_episodes=1000, stopped_workers=stopped, resume_episodes=cursors,
        backup=str(backup), session=session, source_files_verified=len(hashes),
        updated_file=relative, original_sha256=original_hash, updated_sha256=hashes[relative],
        frozen_config_and_inputs_unchanged=True,
        note='Cache capacity changes memoization only; deterministic candidate algorithm and learner state are unchanged. Cold cache on restart.'))
    command = [manifest['runs'][0]['command'][0], '-u', str(source / 'tools/queued_city_baselines.py'),
               'run', '--output-root', str(BATCH)]
    shell = shlex.join(command) + ' >> ' + shlex.quote(str(BATCH / 'controller.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(source), shell], check=True)
    print(json.dumps(dict(status='cache200000_deployed_controller_launched', resumed_jobs=27,
        capacity=200000, resume_episodes=cursors, session=session)), flush=True)


if __name__ == '__main__':
    main()
