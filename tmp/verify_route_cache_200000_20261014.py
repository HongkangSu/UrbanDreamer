"""Audit live resume arguments, restored cursors, source identities and progress."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue
from tmp.extend_route_baselines_2000_20261014 import BATCH

manifest = json.loads((BATCH / 'manifest.json').read_text())
extension = json.loads((BATCH / 'cache_restart_200000.json').read_text())
old_hashes = json.loads((Path(extension['backup']) / 'source_hashes.json').read_text())
hashes = json.loads((BATCH / 'source_hashes.json').read_text())
source = Path(manifest['source_snapshot'])
assert manifest['episode_budget'] == 2000 and manifest['total_training_episodes'] == 54000
assert manifest['route_candidate_cache_capacity'] == 200000
assert '_route_candidate_cache_limit = 200000' in (source / 'urban_dreamer/traffic/route_candidates.py').read_text()
assert all(hashlib.sha256((source / p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
assert all(digest == hashes[p] for p, digest in old_hashes.items() if p.startswith(('configs/', 'networks/')))


def sample():
    ledger = json.loads((BATCH / 'queue_state.json').read_text())
    rows = []
    for entry in manifest['runs']:
        state = ledger['jobs'][entry['id']]
        if entry['method'].endswith('_ref'):
            assert state['status'] == 'completed'
            continue
        assert state['status'] == 'running' and queue.owned_alive(state, entry), entry['id']
        argv = queue.process_argv(state['pid'])
        assert argv[argv.index('--episodes') + 1] == '2000'
        assert argv[argv.index('--entropy-schedule-episodes') + 1] == '1000'
        assert '--resume' in argv and state['resume']
        runtime = json.loads((Path(entry['output_dir']) / 'evaluation_runtime.json').read_text())
        expected = extension['resume_episodes'][entry['id']]
        assert runtime['restored_episode'] == expected, (entry['id'], runtime, expected)
        fields = (Path('/proc') / str(state['pid']) / 'stat').read_text().rsplit(')', 1)[1].split()
        episode = queue.durable_episode(entry)
        assert episode >= expected
        rows.append(dict(id=entry['id'], city=entry['city'], method=entry['method'],
            pid=state['pid'], identity=state['process_starttime'], status=state['status'],
            live=True, restored_episode=expected, committed_episode=episode,
            episode_budget=2000, entropy_schedule_episodes=1000,
            cpu_ticks=int(fields[11])+int(fields[12]), oom_retries=state['oom_retries']))
    assert len(rows) == 27
    return rows


before = sample()
time.sleep(20)
after = sample()
for first, last in zip(before, after):
    assert first['pid'] == last['pid'] and first['identity'] == last['identity']
    assert last['cpu_ticks'] > first['cpu_ticks'], last['id']
    last['cpu_ticks_added'] = last['cpu_ticks'] - first['cpu_ticks']
queue.write_json(BATCH / 'cache_restart_200000_verification.json', dict(updated_at=queue.now(),
    training_jobs=27, live=27, cache_capacity=200000, source_files_verified=len(hashes),
    frozen_config_and_input_hashes_preserved=True, all_resume_cursors_verified=True,
    all_live_stop_budgets_verified=True, all_live_entropy_horizons_verified=True,
    all_workers_cpu_progress_verified=True, sample_seconds=20,
    oom_retries=sum(row['oom_retries'] for row in after), rows=after))
queue.write_json(BATCH / 'monitoring_current.json', dict(updated_at=queue.now(), episode_budget=2000,
    jobs=[dict(id=row['id'], pid=row['pid'], live=True, episode=row['committed_episode'],
               ticks=row['cpu_ticks'], status=row['status'], oom=row['oom_retries']) for row in after]))
print(json.dumps(dict(status='verified', live=27, source_files_verified=len(hashes),
    all_restored_and_advancing=True, oom_retries=sum(row['oom_retries'] for row in after),
    online_ac=[row for row in after if row['method']=='online_ac'])), flush=True)
