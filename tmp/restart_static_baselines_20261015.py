"""Fresh, frozen five-method restart after the second static repair gate."""
from argparse import Namespace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue

METHODS = ('ippo', 'mappo', 'dreamer', 'mamba', 'mbpo')
REVIEW = ROOT / 'outputs/baseline_static_round2_20261015'
OLD = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'


def prepare():
    gate = json.loads((REVIEW / 'validation_gate.json').read_text())
    assert gate['passed'] and gate['smoke_methods'] == sorted(METHODS)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = ROOT / 'outputs' / ('city_baselines_static_round2_2000_' + stamp)
    queue.prepare(Namespace(output_root=root, episodes=2000,
        routing_mode='current_to_dest', methods=METHODS, poll_seconds=5.,
        gpu_headroom_mib=1024, min_free_gb=25.))
    manifest = json.loads((root / 'manifest.json').read_text())
    ledger = json.loads((root / 'queue_state.json').read_text())
    prior = json.loads((OLD / 'queue_state.json').read_text())['jobs']
    manifest['runs'] = [entry for entry in manifest['runs'] if entry['method'] in METHODS]
    manifest['runs'].sort(key=lambda e: (METHODS.index(e['method']), queue.CITIES.index(e['city'])))
    ledger['jobs'] = {entry['id']: ledger['jobs'][entry['id']] for entry in manifest['runs']}
    for entry in manifest['runs']:
        entry['command'] += ['--entropy-schedule-episodes', '1000']
        # Retain measured predecessor reservations for unchanged large-batch
        # learners. MBPO starts conservatively and can be calibrated after its
        # new bounded actor backward pass completes real updates.
        previous = prior[entry['id']]
        if entry['method'] == 'mbpo':
            budget = 4096 if entry['city'] == 'ingolstadt_custom' else 8192
        else:
            budget = 256 * math.ceil(max(1024, previous['observed_peak_mib'] * 1.25 + 512) / 256)
        entry['required_gpu_mib'] = budget
        ledger['jobs'][entry['id']]['required_gpu_mib'] = budget
    storage = Path('/home/wangwenrui/private/UrbanDreamer') / root.name
    storage.mkdir(parents=True, exist_ok=False)
    manifest.update(restart_reason='User requested second static repair and restart of all five baselines',
        predecessor_batch=str(OLD), validation_gate=str(REVIEW / 'validation_gate.json'),
        checkpoint_restart='fresh; incompatible prior protocols are not resumed',
        entropy_schedule_episodes=1000, archive_storage_root=str(storage),
        compaction_workers=4, prior_reference_batch=str(OLD),
        resource_note='Conservative measured predecessor reservations; OOM retry and local/archive disk guards enabled')
    queue.write_json(root / 'manifest.json', manifest)
    queue.write_json(root / 'queue_state.json', ledger)
    queue.write_json(REVIEW / 'restart.json', dict(root=str(root), storage=str(storage), status='prepared'))
    print(json.dumps(dict(prepared=str(root), training_jobs=len(manifest['runs']))))


def launch():
    record = json.loads((REVIEW / 'restart.json').read_text())
    assert record['status'] == 'prepared'
    root = Path(record['root'])
    queue.launch(root)
    source = root / 'source'
    session = 'ud_static_offload_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    command = [sys.executable, '-u', str(source / 'tools/offload_episode_archives.py'),
        '--root', str(root), '--storage-root', record['storage'], '--workers', '2', '--interval', '15']
    shell = shlex.join(command) + ' >> ' + shlex.quote(str(root / 'offload.log')) + ' 2>&1'
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', session, '-c', str(source), shell], check=True)
    record.update(status='launched', offload_session=session, launched_at=queue.now())
    queue.write_json(REVIEW / 'restart.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    {'prepare': prepare, 'launch': launch}[sys.argv[1]]()
