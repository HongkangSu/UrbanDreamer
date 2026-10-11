#!/usr/bin/env python3
"""Deploy verified throughput changes to explicitly selected dataset runs."""
from __future__ import annotations

import argparse
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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compact_episode_archives import committed_episode
from tools.run_city_baseline_dataset import atomic_json, report
from tools.stop_city_dataset import command, signal_owned

OVERLAY = (
    'urban_dreamer/learning/observations.py', 'urban_dreamer/traffic/topology.py',
    'urban_dreamer/models/policy.py', 'urban_dreamer/learning/baselines/common.py',
    'urban_dreamer/learning/baselines/dreamer.py', 'urban_dreamer/learning/baselines/mbpo.py',
    'urban_dreamer/learning/baselines/gnac.py', 'tools/train_toy_actor_critic.py',
    'tools/train_toy_baseline.py',
    'urban_dreamer/learning/baselines/collection_policy.py',
    'urban_dreamer/learning/baselines/ppo.py',
    'urban_dreamer/learning/baselines/iql.py',
    'urban_dreamer/learning/baselines/mamba.py',
    'urban_dreamer/learning/baselines/__init__.py',
    'urban_dreamer/learning/baselines/mamba_core/__init__.py',
    'urban_dreamer/learning/baselines/mamba_core/config.py',
    'urban_dreamer/learning/baselines/mamba_core/networks.py',
    'urban_dreamer/learning/baselines/mamba_core/LICENSE',
    'urban_dreamer/learning/baselines/mamba_core/README.md',
    'tools/run_city_baseline_dataset.py',
)


def argument(argv, flag):
    encoded = flag.encode()
    try:
        return argv[argv.index(encoded) + 1].decode()
    except (ValueError, IndexError):
        return None


def process_inventory(root, entries):
    selected = {entry['id'] for entry in entries}
    outputs = {entry['output_dir'] for entry in entries}
    workers, supervisors, compactors = [], [], []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal() or int(path.name) == os.getpid():
            continue
        pid, argv = int(path.name), command(int(path.name))
        if argument(argv, '--output-dir') in outputs:
            workers.append(pid)
        elif argument(argv, '--output-root') == str(root) and argument(argv, '--run-id') in selected:
            supervisors.append(pid)
        elif argument(argv, '--root') == str(root) and any(
                arg.endswith(b'/compact_episode_archives.py') for arg in argv):
            compactors.append(pid)
    return workers, supervisors, compactors


def deployment_command(entry, source, device=None, collection_device=None):
    result = list(entry['command'])
    result[2] = str(source / 'tools' / Path(result[2]).name)
    if device is not None:
        result[result.index('--device') + 1] = device
    if collection_device is not None:
        if '--collection-device' in result:
            result[result.index('--collection-device') + 1] = collection_device
        else:
            result += ['--collection-device', collection_device]
    # Keep original --config and output paths: resolved config is a resume contract.
    return result


def preserve_file(source, target, *, link=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if link:
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def committed_metrics(text, cursor):
    """Discard replayed progress after backup, tolerating an interrupted final write."""
    lines = text.splitlines(keepends=True)
    retained = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith('\n'):
                break
            raise
        if int(record.get('episode', 0)) <= cursor:
            retained.append(line.rstrip('\n') + '\n')
    return ''.join(retained)


def prepare_source(root, destination, verified_hashes):
    hashes = json.loads(verified_hashes.read_text())
    for relative in OVERLAY:
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != hashes[relative]:
            raise ValueError(f'verified source changed: {relative}')
    shutil.copytree(root, destination, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for relative in OVERLAY:
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination / relative)
    return {relative: hashlib.sha256((destination / relative).read_bytes()).hexdigest()
            for relative in OVERLAY}


def stop_processes(pids, root):
    for pid in pids:
        signal_owned(pid, signal.SIGTERM, root)
        signal_owned(pid, signal.SIGCONT, root)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(command(pid) for pid in pids):
        time.sleep(.2)
    for pid in pids:
        signal_owned(pid, signal.SIGKILL, root)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(command(pid) for pid in pids):
        time.sleep(.1)
    if any(command(pid) for pid in pids):
        raise RuntimeError(f'old processes did not exit: {pids}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-id', nargs='+', required=True)
    parser.add_argument('--verified-hashes', type=Path, required=True)
    parser.add_argument('--device', nargs='*', default=[], help='Explicit run_id=cuda:N mappings')
    parser.add_argument('--collection-device', nargs='*', default=[], help='Explicit run_id=cpu/cuda:N inference mappings')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--prepared', type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    selected = set(args.run_id)
    entries = [entry for entry in manifest['runs'] if entry['id'] in selected]
    if len(entries) != len(selected):
        raise ValueError('unknown or duplicate run IDs')
    devices = dict(item.split('=', 1) for item in args.device)
    collection_devices = dict(item.split('=', 1) for item in args.collection_device)
    if not devices.keys() <= selected:
        raise ValueError('device mapping must refer to selected runs')
    if not collection_devices.keys() <= selected:
        raise ValueError('collection device mapping must refer to selected runs')
    if args.prepared:
        deployment = args.prepared.resolve()
        preparation = json.loads((deployment / 'preparation.json').read_text())
        if preparation['root'] != str(root) or preparation['run_ids'] != sorted(selected):
            raise ValueError('prepared deployment scope mismatch')
        if preparation['devices'] != devices:
            raise ValueError('prepared device mapping mismatch')
        if preparation.get('collection_devices', {}) != collection_devices:
            raise ValueError('prepared collection device mapping mismatch')
    else:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        deployment = root / 'restarts' / f'throughput_{stamp}'
        deployment.mkdir(parents=True)
        source_hashes = prepare_source(Path(manifest['source_snapshot']), deployment / 'source', args.verified_hashes)
        preparation = dict(root=str(root), run_ids=sorted(selected), devices=devices,
                           collection_devices=collection_devices,
                           source_hashes=source_hashes, created_at=datetime.now(timezone.utc).isoformat())
        atomic_json(deployment / 'preparation.json', preparation)
    source = deployment / 'source'
    for relative, expected in preparation['source_hashes'].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'prepared source changed: {relative}')
    if args.prepare_only:
        print(json.dumps({'prepared': str(deployment), 'runs': len(entries)}), flush=True)
        return
    if (deployment / 'restart.json').exists():
        raise ValueError('deployment already attempted; inspect restart.json before retrying')

    untouched = {entry['id']: json.loads(Path(entry['state']).read_text()).get('pid')
                 for entry in manifest['runs'] if entry['id'] not in selected}
    workers, supervisors, compactors = process_inventory(root, entries)
    if len(workers) != len(entries) or len(supervisors) != len(entries):
        raise RuntimeError(f'expected one worker/supervisor per selected run: {workers}, {supervisors}')
    ledger = dict(started_at=datetime.now(timezone.utc).isoformat(), status='preserving',
        preparation=preparation, old_workers=workers, old_supervisors=supervisors,
        compactors=compactors, untouched_worker_pids=untouched, runs=[])
    atomic_json(deployment / 'restart.json', ledger)
    try:
        for pid in supervisors + compactors + workers:
            if not signal_owned(pid, signal.SIGSTOP, root):
                raise RuntimeError(f'process vanished before preservation: {pid}')
        preserve_file(manifest_path, deployment / 'manifest_before.json')
        with (deployment / 'archive_inventory.jsonl').open('w') as inventory:
            for entry in entries:
                output = Path(entry['output_dir'])
                checkpoint = output / ('progress.json' if entry['method'] == 'random' else 'latest.pt')
                if not checkpoint.is_file():
                    raise RuntimeError(f'no durable checkpoint for {entry["id"]}')
                cursor = committed_episode(entry)
                if cursor < 0:
                    raise RuntimeError(f'no durable checkpoint for {entry["id"]}')
                archives = sorted((output / 'episodes/train').glob('episode_*.npz'))
                backup = deployment / 'preserved' / entry['city'] / entry['method']
                for path in archives:
                    target = backup / 'episodes/train' / path.name
                    preserve_file(path, target, link=True)
                    with zipfile.ZipFile(target) as archive:
                        members = [(info.filename, info.file_size, info.CRC) for info in archive.infolist()]
                    inventory.write(json.dumps(dict(run=entry['id'], episode=int(path.stem.split('_')[-1]),
                        backup=str(target), bytes=target.stat().st_size, members=members)) + '\n')
                for path in output.iterdir():
                    if path.is_file() and path.suffix in {'.pt', '.json', '.jsonl'}:
                        preserve_file(path, backup / path.name, link=path.suffix == '.pt')
                preserve_file(Path(entry['state']), backup / 'process_before.json')
                preserve_file(Path(entry['log']), backup / 'trainer_before.log')
                if (backup / 'metrics.jsonl').exists():
                    (backup / 'metrics_committed.jsonl').write_text(
                        committed_metrics((backup / 'metrics.jsonl').read_text(), cursor))
                ledger['runs'].append(dict(id=entry['id'], archived_episodes=len(archives),
                    checkpoint_episode=cursor, resume_episode=cursor + 1,
                    prior_device=entry['device'], device=devices.get(entry['id'], entry['device']),
                    collection_device=collection_devices.get(entry['id'], entry['device']),
                    archive_backup=str(backup / 'episodes/train')))
                print(json.dumps(ledger['runs'][-1]), flush=True)
            inventory.flush()
            os.fsync(inventory.fileno())
        atomic_json(deployment / 'restart.json', ledger)
    except BaseException:
        for pid in compactors + workers + supervisors:
            signal_owned(pid, signal.SIGCONT, root)
        ledger['status'] = 'preservation_failed_original_resumed'
        atomic_json(deployment / 'restart.json', ledger)
        raise

    try:
        stop_processes(workers, root)
        stop_processes(supervisors, root)
    except BaseException:
        for pid in compactors + supervisors:
            signal_owned(pid, signal.SIGCONT, root)
        raise
    for entry in entries:
        subprocess.run(['/usr/bin/tmux', 'kill-session', '-t', entry['tmux_session']],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cursor = next(row['checkpoint_episode'] for row in ledger['runs'] if row['id'] == entry['id'])
        output = Path(entry['output_dir'])
        # New-device provenance differs, and all runs must replay from durable progress.
        # Originals remain in the immutable backup; move superseded active files aside.
        quarantined = []
        for path in sorted((output / 'episodes/train').glob('episode_*.npz')):
            if int(path.stem.split('_')[-1]) > cursor:
                target = deployment / 'uncommitted' / entry['city'] / entry['method'] / path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, target)
                quarantined.append(path.name)
        metric_path = output / 'metrics.jsonl'
        if metric_path.exists():
            temporary = metric_path.with_name('.metrics.restart.tmp')
            shutil.copyfile(deployment / 'preserved' / entry['city'] / entry['method'] /
                            'metrics_committed.jsonl', temporary)
            os.replace(temporary, metric_path)
        next(row for row in ledger['runs'] if row['id'] == entry['id'])['replayed_archive_files'] = quarantined
        entry['command'] = deployment_command(entry, source, devices.get(entry['id']), collection_devices.get(entry['id']))
        entry['device'] = devices.get(entry['id'], entry['device'])
        if entry['id'] in collection_devices:
            entry['collection_device'] = collection_devices[entry['id']]
        entry['source_snapshot'] = str(source)
        entry['restart_record'] = str(deployment / 'restart.json')
        entry['tmux_session'] = f"uddata_restart_{deployment.name.removeprefix('throughput_')}_{entry['id']}"
    manifest.setdefault('source_revisions', []).append(dict(source=str(source), run_ids=sorted(selected),
        overlay_hashes=preparation['source_hashes'], restart_record=str(deployment / 'restart.json')))
    atomic_json(manifest_path, manifest)
    ledger['status'] = 'launching'
    atomic_json(deployment / 'restart.json', ledger)
    try:
        for entry in entries:
            cmd = [entry['command'][0], '-u', str(source / 'tools/run_city_baseline_dataset.py'),
                   'supervise', '--output-root', str(root), '--run-id', entry['id']]
            subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', entry['tmux_session'],
                            '-c', str(source), shlex.join(cmd)], check=True)
    finally:
        for pid in compactors:
            signal_owned(pid, signal.SIGCONT, root)
    ledger['status'] = 'launched'
    ledger['launched_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(deployment / 'restart.json', ledger)
    report(argparse.Namespace(output_root=root))
    print(json.dumps({'deployment': str(deployment), 'selected_runs': len(entries), 'status': 'launched'}), flush=True)


if __name__ == '__main__':
    main()
