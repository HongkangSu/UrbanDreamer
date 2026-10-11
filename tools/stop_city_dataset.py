#!/usr/bin/env python3
"""Preserve an archive inventory and stop only processes owned by one dataset manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.compact_episode_archives import committed_episode


def command(pid):
    try:
        return [arg for arg in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if arg]
    except (FileNotFoundError, PermissionError):
        return []


def signal_owned(pid, signum, root):
    encoded = str(root).encode()
    if any(argument == encoded or argument.startswith(encoded + b'/') for argument in command(pid)):
        try:
            os.kill(pid, signum)
            return True
        except ProcessLookupError:
            pass
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    timestamp = datetime.now(timezone.utc).isoformat()
    workers = []
    for entry in manifest['runs']:
        state = json.loads(Path(entry['state']).read_text())
        pid = state.get('pid')
        if pid and str(entry['output_dir']).encode() in command(pid):
            workers.append(pid)
    controllers = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        pid = int(path.name)
        argv = command(pid)
        if (pid != os.getpid() and pid not in workers
                and any(arg == str(root).encode() or arg.startswith(str(root).encode() + b'/') for arg in argv)
                and any(arg.endswith((b'/run_city_baseline_dataset.py', b'/compact_episode_archives.py')) for arg in argv)):
            controllers.append(pid)
    for pid in controllers + workers:
        signal_owned(pid, signal.SIGSTOP, root)
    print(json.dumps({'phase': 'frozen', 'workers': workers, 'controllers': controllers}), flush=True)

    inventories, errors = [], []
    with (root / 'preserved_episode_inventory.jsonl').open('w') as inventory:
        for entry in manifest['runs']:
            files = sorted((Path(entry['output_dir']) / 'episodes' / 'train').glob('episode_*.npz'))
            total_bytes = 0
            for path in files:
                try:
                    with zipfile.ZipFile(path) as archive:
                        members = [dict(name=item.filename, bytes=item.file_size, crc32=item.CRC)
                                   for item in archive.infolist()]
                        if not {'summary_json.npy', 'decisions_json.npy', 'flow.npy'} <= {m['name'] for m in members}:
                            raise ValueError('required archive members missing')
                    size = path.stat().st_size
                    total_bytes += size
                    inventory.write(json.dumps(dict(path=str(path.relative_to(root)), bytes=size, members=members)) + '\n')
                except Exception as error:
                    errors.append({'path': str(path), 'error': repr(error)})
            inventories.append(dict(city=entry['city'], method=entry['method'],
                episodes=len(files), archive_bytes=total_bytes,
                latest_archived=int(files[-1].stem.split('_')[-1]) if files else 0,
                checkpoint_episode=committed_episode(entry)))
            print(json.dumps(inventories[-1]), flush=True)
        inventory.flush()
        os.fsync(inventory.fileno())
    summary = dict(stopped_at=timestamp, root=str(root), total_archived=sum(r['episodes'] for r in inventories),
        runs=inventories, inventory_errors=errors, workers=workers, controllers=controllers,
        preservation='All completed NPZ archives, checkpoints, source snapshot, metrics and plots retained in place. In-flight incomplete episodes are not fabricated.',
        inventory_integrity='ZIP member CRC/length inventory; existing writers already use atomic replacement')
    path = root / 'preservation_and_stop.json'
    with path.open('w') as stream:
        json.dump(summary, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    for entry in manifest['runs']:
        archive_directory = Path(entry['output_dir']) / 'episodes' / 'train'
        if not archive_directory.is_dir():
            continue  # A newly started run may not yet have saved its first episode.
        directory = os.open(str(archive_directory), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    # Controllers remain frozen until workers have exited, preventing retries.
    for pid in workers:
        signal_owned(pid, signal.SIGTERM, root)
        signal_owned(pid, signal.SIGCONT, root)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and any(command(pid) for pid in workers):
        time.sleep(.2)
    for pid in workers:
        signal_owned(pid, signal.SIGKILL, root)
    for pid in controllers:
        signal_owned(pid, signal.SIGTERM, root)
        signal_owned(pid, signal.SIGCONT, root)
    time.sleep(1)
    for pid in controllers:
        signal_owned(pid, signal.SIGKILL, root)
    for entry in manifest['runs']:
        state_path = Path(entry['state'])
        state = json.loads(state_path.read_text())
        if state.get('status') != 'completed':
            state.update(status='stopped_by_user', stopped_at=timestamp)
        state_path.write_text(json.dumps(state, indent=2) + '\n')
        subprocess.run(['/usr/bin/tmux', 'kill-session', '-t', entry['tmux_session']],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    summary['remaining_owned_processes'] = [pid for pid in workers + controllers if command(pid)]
    path.write_text(json.dumps(summary, indent=2) + '\n')
    (root / 'STOPPED_BY_USER').write_text(timestamp + '\n')
    print(json.dumps({'phase': 'stopped', 'episodes_preserved': summary['total_archived'],
        'remaining_processes': summary['remaining_owned_processes'], 'inventory_errors': errors}), flush=True)
    if summary['remaining_owned_processes'] or errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
