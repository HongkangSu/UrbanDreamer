"""Deploy collector-reference validation and recover only the stopped MAPPO job."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue
from tmp.extend_route_baselines_2000_20261014 import BATCH, stop_owned

manifest = json.loads((BATCH/'manifest.json').read_text())
ledger = json.loads((BATCH/'queue_state.json').read_text())
source = Path(manifest['source_snapshot'])
relative = 'urban_dreamer/learning/baselines/ppo.py'
hashes = json.loads((BATCH/'source_hashes.json').read_text())
assert all(hashlib.sha256((source/p).read_bytes()).hexdigest() == digest for p,digest in hashes.items())
run_id = 'montereau_fault_yonne_mappo'
entry = next(e for e in manifest['runs'] if e['id'] == run_id)
assert ledger['jobs'][run_id]['status'] == 'failed'
assert queue.durable_episode(entry) == 742
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
backup = BATCH/'numerical_recoveries'/f'mappo_collector_reference_{stamp}'
backup.mkdir(parents=True, exist_ok=False)
for name in ('manifest.json','source_hashes.json','queue_state.json'):
    shutil.copy2(BATCH/name, backup/name)
shutil.copy2(source/relative, backup/'ppo.py')
output = Path(entry['output_dir'])
for name in ('latest.pt','best.pt','initial.pt'):
    os.link(output/name, backup/name)
failed_archive = output/'episodes/train/episode_000743.npz'
if failed_archive.exists():
    os.link(failed_archive, backup/failed_archive.name)
workers = {e['id']:dict(pid=ledger['jobs'][e['id']]['pid'],
                       identity=ledger['jobs'][e['id']]['process_starttime'])
           for e in manifest['runs'] if queue.owned_alive(ledger['jobs'][e['id']],e)}
pid = ledger['controller_pid']
identity = queue.process_identity(pid)
assert identity and str(source/'tools/queued_city_baselines.py') in queue.process_argv(pid)
stop_owned(pid, identity, str(BATCH))
ledger = json.loads((BATCH/'queue_state.json').read_text())
assert ledger['jobs'][run_id]['status'] == 'failed'
before_hash = hashes[relative]
shutil.copy2(ROOT/relative, source/relative)
hashes[relative] = hashlib.sha256((source/relative).read_bytes()).hexdigest()
assert all(hashlib.sha256((source/p).read_bytes()).hexdigest() == digest for p,digest in hashes.items())
ledger['jobs'][run_id].update(status='queued_recovery', pid=None, process_starttime=None,
    next_eligible=0., collector_reference_recovery=True)
for key in ('failed_at','exit_code','last_error'):
    ledger['jobs'][run_id].pop(key,None)
ledger['controller_pid'] = None
session = f'ud_route_collector_reference_{stamp}'
manifest['controller_session'] = session
for name,value in (('manifest.json',manifest),('source_hashes.json',hashes),('queue_state.json',ledger)):
    queue.write_json(BATCH/name,value)
assert all(queue.process_identity(v['pid']) in (v['identity'],None) for v in workers.values())
queue.write_json(BATCH/'mappo_collector_reference_recovery.json',dict(updated_at=queue.now(),
    id=run_id, restored_episode=742, episode_budget=2000, entropy_schedule_episodes=1000,
    backup=str(backup), source_file=relative, original_sha256=before_hash,
    updated_sha256=hashes[relative], source_files_verified=len(hashes),
    unchanged_actor_checkpoint_schema=True, unchanged_behavior_guard_threshold=True,
    unchanged_collection_device=True, retained_collected_log_probabilities=True,
    other_workers_not_interrupted=workers, session=session,
    validation='85 focused PPO/score-gauge/collection tests passed, including actual failed-input CUDA test',
    reason='Validate actual CPU collector distribution rather than equating CUDA batch512 floating-point results to collection probabilities.'))
command=[manifest['runs'][0]['command'][0],'-u',str(source/'tools/queued_city_baselines.py'),
         'run','--output-root',str(BATCH)]
shell=shlex.join(command)+' >> '+shlex.quote(str(BATCH/'controller.log'))+' 2>&1'
subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(source),shell],check=True)
print(json.dumps(dict(status='recovering_only_montereau_mappo',restored_episode=742,
    preserved_other_worker_count=len(workers),session=session)),flush=True)
