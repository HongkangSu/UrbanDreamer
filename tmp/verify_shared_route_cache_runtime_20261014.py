"""Record actual shared database use by each resumed parent and its workers."""
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'


def opened_cache(pid, directory):
    paths = set()
    try:
        for entry in (Path('/proc') / str(pid) / 'fd').iterdir():
            try:
                path = os.readlink(entry)
                if path.startswith(str(directory)) and path.endswith('.sqlite3'):
                    paths.add(path)
            except OSError:
                pass
    except FileNotFoundError:
        pass
    return sorted(paths)


def check():
    rows = []
    for city in ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne'):
        folder = BATCH / city
        status = json.loads((folder / 'status.json').read_text())
        row = {k: status.get(k) for k in ('pid', 'status', 'restored_episode', 'completed_episodes')}
        row['city'] = city
        config = tomllib.loads((folder / 'online/run_config.toml').read_text())
        row.update(route_mode=config['simulation']['routing_mode'],
                   episode_budget=config['run']['online_training_episodes'],
                   precision=config['resources']['world_model_imagination_precision'],
                   rollouts=config['resources']['imagination_rollouts_per_pass'],
                   passes=config['resources']['imagination_passes_per_episode'])
        assert (row['route_mode'], row['episode_budget'], row['precision'], row['rollouts'], row['passes']) == (
            'current_to_dest', 1000, 'fp16', 8, 2)
        directory = folder / 'shared_route_cache_200000'
        pid = status.get('pid')
        if status.get('shared_route_cache_capacity') == 200000 and pid:
            row['parent_open_databases'] = opened_cache(pid, directory)
            workers = []
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                    command = (proc / 'cmdline').read_bytes()
                    if fields[0] != 'Z' and int(fields[1]) == pid and b'spawn_main' in command:
                        workers.append(dict(pid=int(proc.name), databases=opened_cache(proc.name, directory)))
                except FileNotFoundError:
                    pass
            row['workers'] = workers
            stores = list(directory.glob('*.sqlite3'))
            row['cache_stores'] = []
            for path in stores:
                with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
                    count = connection.execute('SELECT entries FROM size WHERE id=1').fetchone()[0]
                    actual = connection.execute('SELECT COUNT(*) FROM routes').fetchone()[0]
                assert count == actual and actual <= 200000
                row['cache_stores'].append(dict(path=str(path), entries=count, capacity=200000))
            row['sharing_verified'] = bool(len(stores) == 1 and row['parent_open_databases'] == [str(stores[0])]
                and len(workers) == 8 and all(w['databases'] == [str(stores[0])] for w in workers))
            if row['sharing_verified'] and status['completed_episodes'] > status['restored_episode']:
                progress = json.loads((folder / 'online/pilot_progress.json').read_text())
                first = progress['online'][status['restored_episode']]
                assert first['optimizer_steps'] == dict(actor=2, event_critic=2, latent_critic=2)
                row.update(resumed_episode_verified=True, resumed_episode_seconds=first['seconds'],
                           optimizer_steps=first['optimizer_steps'])
        rows.append(row)
    report = dict(checked_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        all_three_sharing_verified=all(row.get('sharing_verified') for row in rows),
        all_three_resumed_episode_verified=all(row.get('resumed_episode_verified') for row in rows), cities=rows)
    path = BATCH / 'shared_cache_runtime_verification.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(path)
    return report


if __name__ == '__main__':
    while True:
        report = check()
        if '--watch' not in sys.argv or report['all_three_resumed_episode_verified']:
            print(json.dumps(report, indent=2), flush=True)
            break
        if any(row['status'] == 'failed' for row in report['cities']):
            print(json.dumps(report, indent=2), flush=True)
            raise RuntimeError('resumed online runner failed; inspect its shared-cache log')
        time.sleep(15)
