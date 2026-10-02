#!/usr/bin/env python3
"""Launch matched online AC and approximate DTA-SO experiments in separate tmux sessions."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CITIES = ('ingolstadt_custom', 'gretz_armainvilliers', 'fontenay_en_parisis')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--eval-seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.episodes, args.eval_every, args.eval_seeds) < 1:
        parser.error('experiment budgets must be positive')
    directory = args.output_root.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('refusing to overwrite an existing comparison')
    subprocess.run(['/usr/bin/tmux', 'list-sessions'], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)  # A server need not exist yet.
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        reward_contract='global_physical_individual_timeout_v1',
        reward='shared -active_vehicle_seconds/(N*60); individual -5000 only for the unfinished vehicle',
        selection_score='global_reward_sum + vehicle_timeout_penalty_sum; diagnostic ranking only',
        demand='independent fixed-rate OD Poisson processes',
        training_policy='categorical sampling for AC; seeded OD path-fraction allocation for DTA-SO',
        evaluation_policy='argmax AC; reproducible seeded allocation DTA-SO',
        episode_budget=args.episodes, eval_every=args.eval_every,
        evaluation_seeds=[args.seed + 10000 + i for i in range(args.eval_seeds)],
        holdout_seeds=[args.seed + 20000 + i for i in range(args.eval_seeds)],
        seed=args.seed, backend='libsumo', session='event', step_seconds=1.,
        cpu_threads_per_process=1, world_model=False,
        convergence_note='Inspect fixed-demand-sample evaluation and completion; fraction changes alone do not prove optimum.',
        runs=[])
    for city in CITIES:
        config_path = ROOT / 'configs' / f'{city}.toml'
        config = tomllib.loads(config_path.read_text())
        duration = config['simulation']['demand_duration_seconds']
        with (ROOT / 'networks' / city / 'agents.csv').open() as stream:
            expected = sum(1 for _ in csv.DictReader(stream))
        for method in ('online_ac', 'dta_so'):
            output = directory / city / method
            run_name = f'{city}_{method}'
            session = 'udcmp_' + run_name + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            command = [sys.executable, '-u', str(ROOT / 'tools' /
                ('train_toy_actor_critic.py' if method == 'online_ac' else 'train_toy_baseline.py'))]
            if method == 'dta_so':
                command += ['--method', 'dta']
            command += ['--config', str(config_path), '--output-dir', str(output),
                '--episodes', str(args.episodes), '--eval-every', str(args.eval_every),
                '--eval-seeds', str(args.eval_seeds), '--seed', str(args.seed), '--device', 'cpu',
                '--threads', '1', '--session', 'event', '--backend', 'libsumo',
                '--step-seconds', '1', '--skip-reference-baselines']
            manifest['runs'].append(dict(city=city, method=method, output_dir=str(output),
                tmux_session=session, command=command, expected_vehicle_count=expected,
                demand_duration_seconds=duration,
                log=str(directory / 'logs' / f'{run_name}.log'),
                pid_file=str(directory / 'processes' / f'{run_name}.pid'),
                exit_file=str(directory / 'processes' / f'{run_name}.exit'),
                launcher=str(directory / 'launchers' / f'{run_name}.sh'), launched=False))
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'processes', 'launchers'):
        (directory / name).mkdir()
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    for run in manifest['runs']:
        script = '\n'.join([
            '#!/bin/bash', 'set -uo pipefail', 'cd ' + shlex.quote(str(ROOT)),
            'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
            shlex.join(run['command']) + ' > ' + shlex.quote(run['log']) + ' 2>&1 &',
            'training_pid=$!', 'printf "%s\\n" "$training_pid" > ' + shlex.quote(run['pid_file']),
            'wait "$training_pid"', 'training_status=$?',
            'printf "%s\\n" "$training_status" > ' + shlex.quote(run['exit_file']),
            'exit "$training_status"', ''])
        Path(run['launcher']).write_text(script)
        subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', run['tmux_session'],
                        '-c', str(ROOT), shlex.join(['/bin/bash', run['launcher']])], check=True)
        run['launched'] = True
        run['launched_at'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(dict(city=run['city'], method=run['method'], session=run['tmux_session'])), flush=True)
    print(json.dumps(dict(manifest=str(path), launched=len(manifest['runs']))), flush=True)


if __name__ == '__main__':
    main()
