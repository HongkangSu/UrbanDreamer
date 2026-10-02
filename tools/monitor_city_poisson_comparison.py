#!/usr/bin/env python3
"""Read city comparison progress, process evidence and paired demand hashes; export curves."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time


def mean(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
    return statistics.mean(values) if values else None


def read_records(path):
    records = []
    if not path.exists():
        return records
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # An append in progress is not a complete observation.
            row = json.loads(line)
            records.append({k: v for k, v in row.items() if k not in {'choices', 'branch_choices'}})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--watch', action='store_true', help='Refresh until all six training processes have exited')
    parser.add_argument('--interval-seconds', type=int, default=300)
    args = parser.parse_args()
    manifest = json.loads((args.root / 'manifest.json').read_text())
    if args.interval_seconds < 1:
        parser.error('refresh interval must be positive')
    if args.watch:
        command = [sys.executable, str(Path(__file__).resolve()), '--root', str(args.root.resolve())]
        if args.plot:
            command.append('--plot')
        while True:
            with (args.root / 'monitor.log').open('a') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if all(Path(entry['exit_file']).exists() for entry in manifest['runs']):
                (args.root / 'final_status.json').write_text((args.root / 'status.json').read_text())
                return
            time.sleep(args.interval_seconds)
    runs, records_by_run = [], {}
    for entry in manifest['runs']:
        records = read_records(Path(entry['output_dir']) / 'metrics.jsonl')
        records_by_run[(entry['city'], entry['method'])] = records
        pid_path = Path(entry['pid_file'])
        pid = int(pid_path.read_text()) if pid_path.exists() else None
        cmd = Path(f'/proc/{pid}/cmdline') if pid else None
        # Missing /proc entries may mean sandbox PID isolation, not termination.
        alive = (entry['output_dir'].encode() in cmd.read_bytes() if cmd and cmd.exists() else None)
        exit_path = Path(entry['exit_file'])
        exit_code = int(exit_path.read_text()) if exit_path.exists() else None
        if exit_code is not None:
            alive = False
        train = [r for r in records if r['phase'] == 'train']
        baseline = [r for r in records if r['phase'] == 'baseline' and r['policy'] == 'deterministic']
        evaluation = [r for r in records if r['phase'] == 'evaluation']
        last = train[-1] if train else None
        expected_seeds = set(manifest['evaluation_seeds'])
        evaluated_episodes = {r['episode'] for r in evaluation}
        complete_evaluations = [episode for episode in evaluated_episodes
            if {r['seed'] for r in evaluation if r['episode'] == episode} >= expected_seeds]
        latest_eval_episode = max(complete_evaluations, default=None)
        latest_eval = [r for r in evaluation if r['episode'] == latest_eval_episode]
        keys = ('episode', 'vehicle_count', 'completion_rate', 'mean_travel_seconds', 'mean_exposure_seconds',
                'global_reward_sum', 'vehicle_timeout_penalty_sum', 'selection_score',
                'wall_seconds', 'actor_gradient_norm', 'global_explained_variance',
                'mc_gap', 'path_fraction_change')
        runs.append(dict(city=entry['city'], method=entry['method'], pid=pid, running=alive,
            exit_code=exit_code, last_train={k: last[k] for k in keys if k in last} if last else None,
            initial_evaluation_runs=len(baseline), initial_evaluation_travel=mean(baseline, 'mean_travel_seconds'),
            initial_evaluation_completion=mean(baseline, 'completion_rate'),
            training_episodes=len(train), latest_evaluation_episode=latest_eval_episode,
            latest_evaluation_runs=len(latest_eval), latest_evaluation_travel=mean(latest_eval, 'mean_travel_seconds'),
            latest_evaluation_completion=mean(latest_eval, 'completion_rate'),
            latest_evaluation_reward=mean(latest_eval, 'global_reward_sum'),
            latest_evaluation_individual_penalty=mean(latest_eval, 'vehicle_timeout_penalty_sum'),
            latest_evaluation_selection_score=mean(latest_eval, 'selection_score'),
            latest_evaluation_exposure=mean(latest_eval, 'mean_exposure_seconds'),
            last20_train_travel=mean(train[-20:], 'mean_travel_seconds'),
            last20_train_exposure=mean(train[-20:], 'mean_exposure_seconds'),
            last20_train_completion=mean(train[-20:], 'completion_rate'),
            last20_episode_wall_seconds=mean(train[-20:], 'wall_seconds'),
            log=entry['log'], output_dir=entry['output_dir']))
    paired = []
    for city in sorted({r['city'] for r in runs}):
        indexed = []
        for method in ('online_ac', 'dta_so'):
            indexed.append({(r['phase'], r['episode'], r['policy'], r['seed']):
                r.get('demand_sha256', r.get('demand_realization_sha256'))
                for r in records_by_run.get((city, method), [])})
        shared = indexed[0].keys() & indexed[1].keys()
        mismatches = [key for key in shared if indexed[0][key] != indexed[1][key] or indexed[0][key] is None]
        paired.append(dict(city=city, matched_records=len(shared), demand_mismatches=mismatches))
        if mismatches:
            raise AssertionError(f'methods did not use identical demand realizations: {city}: {mismatches}')
    report = dict(timestamp=datetime.now(timezone.utc).isoformat(), episode_budget=manifest['episode_budget'],
                  runs=runs, paired_demand_checks=paired)
    (args.root / 'status.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if args.plot and any(records_by_run.values()):
        import os
        os.environ.setdefault('MPLCONFIGDIR', str((args.root / '.matplotlib_cache').resolve()))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cities = sorted({r['city'] for r in runs})
        panels = [('mean_exposure_seconds', 'Mean active time per vehicle (seconds)'),
                  ('global_reward_sum', 'Shared traffic reward')]
        if manifest.get('reward_contract') == 'global_physical_individual_timeout_v1':
            panels.append(('selection_score', 'Selection score (diagnostic only)'))
        panels.append(('completion_rate', 'Completion fraction'))
        fig, axes = plt.subplots(len(cities), len(panels), figsize=(5 * len(panels), 4 * len(cities)), squeeze=False)
        for row_index, city in enumerate(cities):
            for method, color in [('online_ac', '#2463a5'), ('dta_so', '#cf6a27')]:
                records = records_by_run[(city, method)]
                train = [r for r in records if r['phase'] == 'train']
                for column, (key, title) in enumerate(panels):
                    axes[row_index, column].plot([r['episode'] for r in train],
                        [mean(train[max(0, i - 19):i + 1], key) for i in range(len(train))],
                        color=color, label=method + ' train (20-episode mean)')
                    eval_rows = [r for r in records if r['phase'] in {'baseline', 'evaluation'} and r['policy'] == 'deterministic']
                    episodes = sorted(episode for episode in {r['episode'] for r in eval_rows}
                        if {r['seed'] for r in eval_rows if r['episode'] == episode} >= expected_seeds)
                    axes[row_index, column].plot(episodes,
                        [mean([r for r in eval_rows if r['episode'] == episode], key) for episode in episodes],
                        marker='o', linestyle='--', color=color, label=method + ' fixed-seed evaluation')
                    axes[row_index, column].set(title=city + '\n' + title, xlabel='Training episode')
                    axes[row_index, column].grid(alpha=.2)
                    axes[row_index, column].legend(fontsize=7)
        fig.suptitle('Fixed-rate Poisson demand: online AC vs approximate DTA-SO')
        fig.tight_layout()
        fig.savefig(args.root / 'learning_curves.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    main()
