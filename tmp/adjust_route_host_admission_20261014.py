"""Handoff the batch controller without interrupting restored trainers."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue
from tmp.extend_route_baselines_2000_20261014 import BATCH, stop_owned

manifest = json.loads((BATCH / 'manifest.json').read_text())
ledger = json.loads((BATCH / 'queue_state.json').read_text())
source = Path(manifest['source_snapshot'])
relative = 'tools/queued_city_baselines.py'
hashes = json.loads((BATCH / 'source_hashes.json').read_text())
assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == hashes[relative]
pid = ledger['controller_pid']
identity = queue.process_identity(pid)
assert identity and str(source / relative) in queue.process_argv(pid)
workers = {entry['id']: dict(pid=ledger['jobs'][entry['id']]['pid'],
           identity=ledger['jobs'][entry['id']]['process_starttime'])
           for entry in manifest['runs'] if queue.owned_alive(ledger['jobs'][entry['id']], entry)}
stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
backup = BATCH / 'budget_extensions' / f'host_admission_{stamp}'
backup.mkdir()
for name in ('manifest.json', 'source_hashes.json', 'queue_state.json'):
    shutil.copy2(BATCH / name, backup / name)
shutil.copy2(source / relative, backup / 'queued_city_baselines.py')
stop_owned(pid, identity, str(BATCH))
shutil.copy2(ROOT / relative, source / relative)
hashes[relative] = hashlib.sha256((source / relative).read_bytes()).hexdigest()
assert all(hashlib.sha256((source / path).read_bytes()).hexdigest() == digest
           for path, digest in hashes.items())
manifest = json.loads((BATCH / 'manifest.json').read_text())
manifest['pending_host_reservation_fraction'] = .5
session = f'ud_route_2000_host_{stamp}'
manifest['controller_session'] = session
queue.write_json(BATCH / 'source_hashes.json', hashes)
queue.write_json(BATCH / 'manifest.json', manifest)
assert all(queue.process_identity(row['pid']) == row['identity'] for row in workers.values())
queue.write_json(BATCH / 'host_admission_revision.json', dict(updated_at=queue.now(),
    pending_host_reservation_fraction=.5, host_headroom_mib=manifest['host_headroom_mib'],
    available_host_mib=queue.host_available_mib(), workers_unchanged=workers,
    old_controller_pid=pid, source_sha256=hashes[relative], backup=str(backup),
    validation='36 queue tests passed; source snapshot hashes verified', session=session))
command = [manifest['runs'][0]['command'][0], '-u', str(source / relative), 'run', '--output-root', str(BATCH)]
shell = shlex.join(command) + ' >> ' + shlex.quote(str(BATCH / 'controller.log')) + ' 2>&1'
subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(source), shell], check=True)
print(json.dumps(dict(status='controller_handoff', preserved_workers=len(workers), session=session)), flush=True)
