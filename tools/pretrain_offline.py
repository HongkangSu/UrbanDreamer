"""Historical pilot helpers; execution requires the validated codec/cache staged runner.

Split/sampling helpers remain useful for diagnostics. The old executable pilot
cannot establish codec readiness and is disabled before reading or writing data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.replay import ReplayBuffer


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def select_split(rows, duplicate_groups, *, seed, train_per_stratum, validation_per_stratum):
    """Reserve demand identities before selecting root/method-balanced files."""
    rng = random.Random(seed)
    strata = {}
    for row in rows:
        strata.setdefault((row['root'], row['method']), []).append(row)
    for entries in strata.values():
        entries.sort(key=lambda row: row['path'])
    identities = sorted({row['demand_group'] for row in rows})
    rng.shuffle(identities)
    holdout = set(identities[:max(validation_per_stratum, round(len(identities) * .15))])
    for key in sorted(strata, key=lambda key: (len(strata[key]), key)):
        candidates = sorted({row['demand_group'] for row in strata[key]})
        rng.shuffle(candidates)
        missing = validation_per_stratum - len(holdout.intersection(candidates))
        if missing > 0:
            holdout.update([group for group in candidates if group not in holdout][:missing])
    trajectory = {path: group['trajectory_hash'] for group in duplicate_groups for path in group['files']}
    selected, used = {'train': [], 'validation': []}, set()
    for split, count in (('validation', validation_per_stratum), ('train', train_per_stratum)):
        for key in sorted(strata):
            candidates = [row for row in strata[key] if (row['demand_group'] in holdout) == (split == 'validation')]
            rng.shuffle(candidates)
            chosen = []
            for row in candidates:
                identity = trajectory.get(row['path'], row['path'])
                if identity not in used:
                    chosen.append(row)
                    used.add(identity)
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f'insufficient distinct trajectories for {split}: {key}')
            selected[split].extend(chosen)
    train_groups = {row['demand_group'] for row in selected['train']}
    validation_groups = {row['demand_group'] for row in selected['validation']}
    if train_groups & validation_groups:
        raise AssertionError('realized-demand leakage in train/validation split')
    return dict(seed=seed, holdout_groups=sorted(holdout), train=selected['train'],
                validation=selected['validation'], train_demand_groups=len(train_groups),
                validation_demand_groups=len(validation_groups))


def sampled_schedule(dataset, *, starts_per_episode, batch_size, seed):
    rng = np.random.default_rng(seed)
    schedule = []
    for index in rng.permutation(len(dataset.manifest['entries'])):
        count = max(1, dataset.manifest['entries'][int(index)]['windows'] - dataset.sequence_length + 1)
        starts = np.unique(np.rint(np.linspace(0, count - 1, min(starts_per_episode, count))).astype(int))
        rng.shuffle(starts)
        for offset in range(0, len(starts), batch_size):
            schedule.append(dict(episode=int(index), starts=starts[offset:offset + batch_size].tolist()))
    return schedule


def scheduled_batches(dataset, schedule):
    current, replay = None, None
    for selection in schedule:
        index = selection['episode']
        if index != current:
            episode = dataset._episode(index)
            replay = ReplayBuffer(dataset.sequence_length, max_episodes=1)
            replay.append_episode(episode['transitions'], total_travel_minutes=episode['total_travel_minutes'],
                                  total_vehicle_count=episode['total_vehicle_count'])
            current = index
        yield replay._batch_from_picks([(0, start) for start in selection['starts']])


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    updates = [row for row in records if row['phase'] == 'update']
    evaluations = [row for row in records if row['phase'] == 'validation']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot([r['update'] for r in updates], [r['world_model_loss'] for r in updates], alpha=.65)
    axes[0, 0].set(title='Training objective', xlabel='Optimizer updates', ylabel='World-model loss', yscale='log')
    panels = ((axes[0, 1], 'speed_mae_mps', 'Occupied-road speed MAE', 'm/s'),
              (axes[1, 0], 'volume_mae_vehicles', 'Occupied-road volume MAE', 'vehicles'),
              (axes[1, 1], 'active_fraction_mae', 'Active-fraction MAE', 'fraction'))
    for axis, metric, title, unit in panels:
        for horizon in (30, 60, 90):
            prefix = '' if metric == 'active_fraction_mae' else 'occupied_road_'
            key = f'rollout_{horizon}s_{prefix}{metric}'
            axis.plot([r['update'] for r in evaluations], [r[key] for r in evaluations], marker='o', label=f'{horizon}s model')
        persistence_key = 'persistence_30s_' + ('' if metric == 'active_fraction_mae' else 'occupied_road_') + metric
        axis.axhline(evaluations[0][persistence_key], color='black', linestyle='--', label='30s persistence')
        axis.set(title=title, xlabel='Optimizer updates', ylabel=unit)
        axis.legend(fontsize=8)
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--census-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-per-stratum', type=int, default=8)
    parser.add_argument('--validation-per-stratum', type=int, default=2)
    parser.add_argument('--starts-per-episode', type=int, default=8)
    parser.add_argument('--validation-starts', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--evaluate-every', type=int, default=80)
    parser.add_argument('--timeout-seconds', type=int, default=7200)
    parser.parse_args(argv)
    parser.error('Historical pilot cannot run FM before independent raw/cohort codec validation. '
                 'Use: python tools/train_codec_ready.py --pilot-dir PILOT --output-dir CODEC; '
                 'python tools/build_pretrain_cache.py --config CONFIG --census-dir CENSUS '
                 '--pilot-dir PILOT --output-dir CACHE; python tools/run_complete_pretrain.py '
                 '--config CONFIG --codec-dir CODEC --cache-manifest CACHE/cache_manifest.json '
                 '--output-dir RUN. No data, collection or training has started.')


if __name__ == '__main__':
    main()
