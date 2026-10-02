#!/usr/bin/env python3
"""Launch and supervise independent city baseline jobs with real-episode archives."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CITIES = ('ingolstadt_custom', 'gretz_armainvilliers', 'fontenay_en_parisis')
METHODS = ('online_ac', 'gnac', 'dreamer', 'mbpo', 'random')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def snapshot(root):
    destination = root / 'source'
    files = {}
    for directory in ('urban_dreamer', 'tools', 'configs'):
        for source in sorted((ROOT / directory).rglob('*')):
            if source.is_file() and source.suffix in {'.py', '.toml'} and '__pycache__' not in source.parts:
                relative = source.relative_to(ROOT)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    (destination / 'networks').symlink_to(ROOT / 'networks', target_is_directory=True)
    atomic_json(root / 'source_hashes.json', files)
    return destination


def build_runs(root, source, args, stamp):
    runs, gpu = [], 0
    for city in args.cities:
        for method in args.methods:
            device = 'cpu'
            if method in {'gnac', 'dreamer', 'mbpo'} and args.devices:
                device = args.devices[gpu % len(args.devices)]
                gpu += 1
            tool = ('collect_random_episodes.py' if method == 'random' else
                    'train_toy_actor_critic.py' if method == 'online_ac' else 'train_toy_baseline.py')
            output = root / city / method
            command = [sys.executable, '-u', str(source / 'tools' / tool),
                '--config', str(source / 'configs' / f'{city}.toml'), '--output-dir', str(output),
                '--episodes', str(args.episodes), '--seed', str(args.seed), '--device', device,
                '--threads', '1', '--session', 'event', '--backend', 'libsumo', '--step-seconds', '1']
            if method != 'random':
                command += ['--save-episodes', '--checkpoint-every', str(args.checkpoint_every),
                    '--eval-every', str(args.eval_every), '--eval-seeds', str(args.eval_seeds),
                    '--skip-reference-baselines', '--skip-initial-evaluation']
                if getattr(args, 'sp_prior_weight', 0.) > 0:
                    command += ['--sp-prior-weight', str(args.sp_prior_weight),
                        '--sp-prior-temperature', str(args.sp_prior_temperature),
                        '--sp-prior-hold-episodes', str(args.sp_prior_hold_episodes),
                        '--sp-prior-off-episode', str(args.sp_prior_off_episode)]
            if method not in {'random', 'online_ac'}:
                command += ['--method', method]
            name = city + '_' + method
            runs.append(dict(id=name, city=city, method=method, device=device, command=command,
                output_dir=str(output), log=str(root / 'logs' / f'{name}.log'),
                state=str(root / 'processes' / f'{name}.json'),
                tmux_session=f'uddata_{stamp}_{name}', launched=False))
    return runs


def launch(args):
    root = args.output_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite existing dataset launch')
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < args.min_free_gb * 1024 ** 3:
        raise RuntimeError('insufficient free disk for collection')
    for directory in ('logs', 'processes'):
        (root / directory).mkdir()
    source = snapshot(root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    runs = build_runs(root, source, args, stamp)
    manifest = dict(created_at=now(), episode_budget=args.episodes,
        total_training_episodes=args.episodes * len(runs), seed=args.seed,
        session_protocol='event', action_semantics='branch-only', backend='libsumo', step_seconds=1.,
        evaluation_seeds=[args.seed + 10000 + i for i in range(args.eval_seeds)],
        holdout_seeds=[args.seed + 20000 + i for i in range(args.eval_seeds)],
        evaluation_archived_as_training=False, source_snapshot=str(source),
        min_free_gb=args.min_free_gb, max_restarts=args.max_restarts,
        policy_prior=dict(enabled=args.sp_prior_weight > 0,
            initial_weight=args.sp_prior_weight, temperature_seconds=args.sp_prior_temperature,
            hold_episodes=args.sp_prior_hold_episodes, off_episode=args.sp_prior_off_episode,
            decay='cosine', schedule_clock='real_training_episode',
            cost='junction_time + candidate_edge_free_flow_time + downstream_remaining_time',
            bias='-weight * (cost - minimum_legal_cost) / temperature_seconds',
            evaluation='scheduled_prior and zero_prior on identical fixed demand seeds',
            best_selection='zero_prior' if args.sp_prior_weight > 0 else 'deterministic',
            random_control='uniform legal actions; no prior',
            mbpo_synthetic_buffer='discard old-weight rows whenever scheduled weight changes'),
        notes=['Independent 5000-episode budgets, no existing experiment overwritten',
               'All archives are real SUMO transitions, including zero and censored episodes',
               'event branch-only action protocol must be matched by downstream world-model usage'], runs=runs)
    manifest_path = root / 'manifest.json'
    atomic_json(manifest_path, manifest)
    for entry in runs:
        command = [sys.executable, '-u', str(source / 'tools' / Path(__file__).name),
                   'supervise', '--output-root', str(root), '--run-id', entry['id']]
        subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', entry['tmux_session'],
                        '-c', str(source), shlex.join(command)], check=True)
        entry['launched'] = True
        atomic_json(manifest_path, manifest)
        print(json.dumps({'launched': entry['id'], 'device': entry['device']}), flush=True)
    watch_command = [sys.executable, '-u', str(source / 'tools' / Path(__file__).name),
                     'watch', '--output-root', str(root)]
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', f'uddata_monitor_{stamp}',
                    '-c', str(source), shlex.join(watch_command)], check=True)
    compact_command = [sys.executable, '-u', str(source / 'tools' / 'compact_episode_archives.py'),
                       '--root', str(root), '--workers', '4']
    compact_log = root / 'compaction.log'
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', f'uddata_compact_{stamp}',
        '-c', str(source), shlex.join(compact_command) + ' > ' + shlex.quote(str(compact_log)) + ' 2>&1'], check=True)
    print(json.dumps({'manifest': str(manifest_path), 'parallel_jobs': len(runs)}), flush=True)


def supervise(args):
    root = args.output_root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    entry = next(run for run in manifest['runs'] if run['id'] == args.run_id)
    state_path = Path(entry['state'])
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    output = Path(entry['output_dir'])
    checkpoint = output / ('progress.json' if entry['method'] == 'random' else 'latest.pt')
    for attempt in range(manifest['max_restarts'] + 1):
        if shutil.disk_usage(root).free < manifest['min_free_gb'] * 1024 ** 3:
            atomic_json(state_path, dict(status='blocked_disk', timestamp=now(), attempt=attempt))
            return 2
        command = entry['command'] + (['--resume'] if checkpoint.exists() else [])
        with Path(entry['log']).open('a') as log:
            log.write(f'\nSupervisor attempt {attempt} at {now()}\n')
            log.flush()
            process = subprocess.Popen(command, cwd=manifest['source_snapshot'], env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            atomic_json(state_path, dict(status='running', pid=process.pid, timestamp=now(), attempt=attempt))
            while process.poll() is None:
                time.sleep(15)
                if shutil.disk_usage(root).free < manifest['min_free_gb'] * 1024 ** 3:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    atomic_json(state_path, dict(status='blocked_disk', pid=process.pid,
                        timestamp=now(), attempt=attempt, exit_code=process.returncode))
                    return 2
        if process.returncode == 0:
            count = len(list((output / 'episodes' / 'train').glob('episode_*.npz')))
            complete = count == manifest['episode_budget']
            atomic_json(state_path, dict(status='completed' if complete else 'incomplete_archives',
                timestamp=now(), attempt=attempt, exit_code=0, archived_episodes=count))
            return 0 if complete else 3
        atomic_json(state_path, dict(status='retrying' if attempt < manifest['max_restarts'] and checkpoint.exists() else 'failed',
            timestamp=now(), attempt=attempt, exit_code=process.returncode))
        if not checkpoint.exists():
            return process.returncode
    return process.returncode


def report(args):
    root = args.output_root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    runs = []
    for entry in manifest['runs']:
        state_path = Path(entry['state'])
        state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'starting'}
        archives = list((Path(entry['output_dir']) / 'episodes' / 'train').glob('episode_*.npz'))
        sizes = sum(path.stat().st_size for path in archives)
        row = dict(city=entry['city'], method=entry['method'], device=entry['device'], **state,
                   archive_count=len(archives), archive_bytes=sizes,
                   latest_archived=max((int(path.stem.rsplit('_', 1)[1]) for path in archives), default=0))
        runs.append(row)
    payload = dict(timestamp=now(), episode_budget=manifest['episode_budget'],
        target_episodes=manifest['total_training_episodes'],
        archived_episodes=sum(row['archive_count'] for row in runs),
        free_disk_gb=shutil.disk_usage(root).free / 1024 ** 3, runs=runs)
    atomic_json(root / 'status.json', payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'supervise', 'status', 'watch'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--eval-seeds', type=int, default=3)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--min-free-gb', type=float, default=10.)
    parser.add_argument('--max-restarts', type=int, default=3)
    parser.add_argument('--cities', nargs='+', choices=CITIES, default=list(CITIES))
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    parser.add_argument('--devices', nargs='*', default=['cuda:0', 'cuda:1'])
    parser.add_argument('--sp-prior-weight', type=float, default=0.)
    parser.add_argument('--sp-prior-temperature', type=float, default=30.)
    parser.add_argument('--sp-prior-hold-episodes', type=int, default=200)
    parser.add_argument('--sp-prior-off-episode', type=int, default=1000)
    args = parser.parse_args(argv)
    if min(args.episodes, args.eval_every, args.eval_seeds, args.checkpoint_every) < 1 or args.min_free_gb <= 0 or args.max_restarts < 0:
        parser.error('invalid collection budgets')
    if args.mode == 'launch':
        import math
        if (not math.isfinite(args.sp_prior_weight) or args.sp_prior_weight < 0
                or not math.isfinite(args.sp_prior_temperature) or args.sp_prior_temperature <= 0
                or args.sp_prior_hold_episodes < 0 or args.sp_prior_off_episode <= args.sp_prior_hold_episodes):
            parser.error('invalid shortest-path prior schedule')
    if args.mode == 'launch':
        launch(args)
    elif args.mode == 'supervise':
        return supervise(args)
    elif args.mode == 'status':
        print(json.dumps(report(args), indent=2))
    else:
        while True:
            status = report(args)
            if all(row['status'] in {'completed', 'failed', 'blocked_disk', 'incomplete_archives', 'stopped_by_user'} for row in status['runs']):
                atomic_json(args.output_root.resolve() / 'final_status.json', status)
                return
            time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
