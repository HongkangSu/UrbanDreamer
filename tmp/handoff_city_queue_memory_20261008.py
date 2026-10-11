"""Transfer the queue controller while keeping its PTY and training children alive."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT/'outputs/city_baselines_noincident_prior_penalty_5000ep_20261007'
spec = importlib.util.spec_from_file_location('queue_revision', ROOT/'tools/queued_city_baselines.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
manifest = json.loads((BATCH/'manifest.json').read_text())
ledger = json.loads((BATCH/'queue_state.json').read_text())
old_pid = ledger['controller_pid']
old_identity = queue.process_identity(old_pid)
old_source = Path(manifest['source_snapshot'])
assert old_identity and str(BATCH) in queue.process_argv(old_pid)
assert any(arg.endswith('/tools/queued_city_baselines.py') for arg in queue.process_argv(old_pid))
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
revision = BATCH/'controller_revisions'/stamp
revision.mkdir(parents=True)
new_source = revision/'source'

def copy_immutable(src, dst):
    if Path(src) == old_source/'tools/queued_city_baselines.py':
        return shutil.copy2(ROOT/'tools/queued_city_baselines.py', dst)
    os.link(src, dst)
    return dst

shutil.copytree(old_source, new_source, copy_function=copy_immutable,
                ignore=shutil.ignore_patterns('__pycache__'), symlinks=True)
new_session = 'ud_queue_mem_'+stamp
record = dict(created_at=queue.now(), old_pid=old_pid, old_process_starttime=old_identity,
              old_session=manifest['controller_session'], new_session=new_session,
              revision_source=str(new_source),
              controller_sha256=hashlib.sha256((new_source/'tools/queued_city_baselines.py').read_bytes()).hexdigest(),
              note='Old controller is stopped, retaining the PTY for its original workers. Training source is unchanged.')
queue.write_json(revision/'handoff.json', record)
# Stop only this owner, preserving its PTY and all running trainer processes.
assert queue.process_identity(old_pid) == old_identity
os.kill(old_pid, signal.SIGSTOP)
for _ in range(50):
    fields = (Path('/proc')/str(old_pid)/'stat').read_text().split(') ', 1)[1].split()
    if fields[0] == 'T':
        break
    time.sleep(.02)
else:
    os.kill(old_pid, signal.SIGCONT)
    raise RuntimeError('Controller did not stop')
ledger = json.loads((BATCH/'queue_state.json').read_text())
entries = {e['id']:e for e in manifest['runs']}
original_workers = {k:s.copy() for k,s in ledger['jobs'].items()
                    if s['status']=='running' and queue.owned_alive(s, entries[k])}
for filename in ('queue_state.json', 'status.json', 'manifest.json'):
    shutil.copy2(BATCH/filename, revision/('before_'+filename))
lock_backup = revision/'previous_queue.lock'
os.replace(BATCH/'queue.lock', lock_backup)
record.update(original_worker_pids={k:s['pid'] for k,s in original_workers.items()},
              old_lock=str(lock_backup))
queue.write_json(revision/'handoff.json', record)
command = [sys.executable, '-u', str(new_source/'tools/queued_city_baselines.py'),
           'run', '--output-root', str(BATCH)]
shell = shlex.join(command)+' >> '+shlex.quote(str(BATCH/'controller.log'))+' 2>&1'
try:
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', new_session,
                    '-c', str(new_source), shell], check=True)
except BaseException:
    if (BATCH/'queue.lock').exists():
        os.replace(BATCH/'queue.lock', revision/'failed_new_queue.lock')
    os.replace(lock_backup, BATCH/'queue.lock')
    os.kill(old_pid, signal.SIGCONT)
    raise
new_pid = None
for _ in range(60):
    current = json.loads((BATCH/'queue_state.json').read_text())
    candidate = current.get('controller_pid')
    if candidate != old_pid and queue.process_identity(candidate):
        new_pid = candidate
        break
    time.sleep(.5)
if new_pid is None:
    raise RuntimeError('New controller has not published state; original owner remains safely stopped')
manifest['controller_session'] = new_session
manifest['memory_estimates'] = 'Initial estimates until 50 committed episodes; then sampled peak *1.15 +256MiB, quantized256, method floors and 1GiB device headroom; OOM budgets never shrink.'
manifest.setdefault('controller_revisions', []).append(dict(record, new_pid=new_pid))
queue.write_json(BATCH/'manifest.json', manifest)
time.sleep(12)
current = json.loads((BATCH/'status.json').read_text())
preserved = {k:queue.owned_alive(s, entries[k]) for k,s in original_workers.items()}
record.update(new_pid=new_pid, verified_at=queue.now(), preserved_workers=preserved,
              counts=current['counts'], new_running=[r['id'] for r in current['runs']
                         if r['status']=='running' and r['id'] not in original_workers])
queue.write_json(revision/'handoff.json', record)
assert all(preserved.values()), preserved
assert current['controller_pid']==new_pid
print(json.dumps(dict(revision=str(revision), **record), indent=2))
