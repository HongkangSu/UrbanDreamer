#!/usr/bin/env python3
"""Plot a live snapshot of city baseline training and complete fixed-seed evaluations."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS = ('online_ac', 'gnac', 'dreamer', 'mbpo', 'ippo', 'mappo', 'iql', 'mamba', 'random')
LABELS = {'online_ac': 'Online AC', 'gnac': 'GNAC', 'dreamer': 'Dreamer', 'mbpo': 'MBPO',
          'ippo': 'IPPO', 'mappo': 'MAPPO', 'iql': 'IQL', 'mamba': 'MAMBA', 'random': 'Random'}
COLORS = {'online_ac': '#2463a5', 'gnac': '#22825c', 'dreamer': '#df8b26', 'mbpo': '#8a4ead',
          'ippo': '#c75c80', 'mappo': '#aa6b27', 'iql': '#33a4af', 'mamba': '#a53232', 'random': '#777777'}
REFERENCE_LABELS = {'shortest': 'Free-flow shortest path', 'dta_so': 'DTA-SO (1 iteration)'}
REFERENCE_COLORS = {'shortest': '#34495e', 'dta_so': '#c0392b'}
CITIES = {'ingolstadt_custom': 'Ingolstadt', 'gretz_armainvilliers': 'Gretz', 'fontenay_en_parisis': 'Fontenay'}
METRICS = (('completion_rate', 'Vehicle completion (higher is better)', '%', 100.),
           ('mean_exposure_seconds', 'Observed active time / vehicle', 'minutes', 1/60),
           ('mean_travel_seconds', 'Travel time of arrived vehicles only', 'minutes', 1/60))
SCALARS = ('completion_rate', 'mean_exposure_seconds', 'mean_travel_seconds',
           'global_reward_sum', 'vehicle_timeout_penalty_sum', 'selection_score',
           'vehicle_count', 'wall_seconds', 'actor_loss', 'global_critic_loss',
           'latent_critic_loss', 'world_model_loss', 'model_nll', 'branch_entropy',
           'sp_prior_weight', 'sp_prior_temperature')


def read_records(path):
    """Ignore a still-being-written final line; last occurrence wins on checkpoint replay."""
    unique = {}
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):
                    break
                row = json.loads(line)
                key = tuple(row[name] for name in ('phase', 'episode', 'policy', 'seed'))
                unique[key] = {name: value for name, value in row.items()
                               if name not in {'choices', 'branch_choices'}}
    return sorted(unique.values(), key=lambda row: (row['episode'], row['phase'], row['seed']))


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def average(rows, key):
    values = [row[key] for row in rows if finite(row.get(key))]
    return float(np.mean(values)) if values else None


def complete_evaluations(records, seeds, *, policy='deterministic'):
    groups = {}
    for row in records:
        if row['phase'] == 'evaluation' and row['policy'] == policy and row['seed'] in seeds:
            groups.setdefault(row['episode'], {})[row['seed']] = row
    result = []
    for episode, by_seed in sorted(groups.items()):
        if set(by_seed) != set(seeds):
            continue
        row = {'episode': episode, 'seed_count': len(seeds)}
        for key in SCALARS:
            values = [by_seed[seed].get(key) for seed in seeds]
            # Undefined arrival time at any seed remains a gap, not an optimistic
            # average over whichever seeds happened to finish some vehicles.
            valid = all(finite(value) for value in values)
            row[key] = float(np.mean(values)) if valid else None
            row[key + '_std'] = float(np.std(values)) if valid else None
        result.append(row)
    return result


def rolling(values, window):
    result = []
    for index in range(len(values)):
        part = np.asarray(values[max(0, index-window+1):index+1], dtype=float)
        result.append(float(part[np.isfinite(part)].mean()) if np.isfinite(part).any() else np.nan)
    return np.asarray(result)


def load_reference_results(path, cities, seeds):
    """Require the same complete demand seed set for every city/reference pair."""
    payload = json.loads(path.read_text())
    expected_seeds = set(seeds)
    declared = payload.get('evaluation_seeds', [])
    if set(declared) != expected_seeds or len(declared) != len(expected_seeds):
        raise ValueError('Reference evaluation seeds must exactly match the manifest')
    expected_runs = {(city, method) for city in cities for method in REFERENCE_LABELS}
    result = {}
    for run in payload.get('runs', []):
        key = (run['city'], run['method'])
        if key not in expected_runs or key in result:
            raise ValueError(f'Unexpected or duplicate reference run: {key}')
        rows = run.get('evaluations', [])
        actual = [row['seed'] for row in rows]
        if set(actual) != expected_seeds or len(actual) != len(expected_seeds):
            raise ValueError(f'Incomplete or duplicate reference evaluation seeds: {key}')
        if run.get('training_episodes') != (1 if run['method'] == 'dta_so' else 0):
            raise ValueError(f'Unexpected reference training budget: {key}')
        normalized = [dict(row, phase='evaluation', policy='reference', episode=0) for row in rows]
        result[key] = complete_evaluations(normalized, seeds, policy='reference')[0]
    if set(result) != expected_runs:
        raise ValueError('Reference results must include both baselines for every city')
    return result


def reference_results_path(root, manifest, override=None):
    if override is not None:
        return override.resolve()
    configured = manifest.get('reference_results')
    if not configured:
        return None
    path = Path(configured)
    return (path if path.is_absolute() else root / path).resolve()


def evaluation_settings(manifest):
    """Read the declared protocol; legacy manifests retain their actual argmax label."""
    prior_enabled = manifest.get('policy_prior', {}).get('enabled', False)
    policy = manifest.get('evaluation_policy')
    if not policy:
        policy = 'deterministic_no_prior' if prior_enabled else 'deterministic'
    if policy.startswith('sampled_temperature_'):
        label = 'sampled, T=' + policy.removeprefix('sampled_temperature_').removesuffix('_no_prior')
    elif policy.startswith('zero_prior_sampled_temperature_'):
        label = 'sampled, T=' + policy.removeprefix('zero_prior_sampled_temperature_')
    else:
        label = 'argmax'
    if prior_enabled or policy.endswith('_no_prior') or policy.startswith('zero_prior_'):
        label += ', prior=0'
    return policy, label


def save_figure(fig, output, stem):
    fig.savefig(output / f'{stem}.png', dpi=170, facecolor='white')
    fig.savefig(output / f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def plot_grid(data, cities, output, timestamp, *, evaluation=False, limit=None, window=20,
              metrics=METRICS, stem='training_curves', evaluation_label='argmax',
              references=None, seed_count=3):
    fig, axes = plt.subplots(len(cities), len(metrics), figsize=(17, 3.7 * len(cities)), squeeze=False)
    fig.subplots_adjust(left=.065, right=.985, top=.865, bottom=.155, wspace=.25, hspace=.52)
    for row_index, city in enumerate(cities):
        missing = []
        methods = tuple(method for method in METHODS
                        if any((city, method) in data for city in cities)
                        and (not evaluation or method != 'random'))
        for method in methods:
            series = data.get((city, method), {}).get('evaluation' if evaluation else 'train', [])
            if limit is not None:
                series = [row for row in series if row['episode'] <= limit]
            if not series:
                missing.append(LABELS[method])
                continue
            x = [row['episode'] for row in series]
            for column, (key, title, unit, factor) in enumerate(metrics):
                ax = axes[row_index, column]
                y = np.asarray([row[key] * factor if finite(row.get(key)) else np.nan for row in series])
                if evaluation:
                    if not np.isfinite(y).any():
                        continue
                    deviation = np.asarray([row[key + '_std'] * factor if finite(row.get(key + '_std')) else np.nan for row in series])
                    ax.errorbar(x, y, yerr=deviation, color=COLORS[method], marker='o',
                                markersize=4, linewidth=1.6, capsize=2, label=LABELS[method])
                else:
                    ax.plot(x, y, color=COLORS[method], alpha=.12, linewidth=.7)
                    ax.plot(x, rolling(y, window), color=COLORS[method], linewidth=2,
                            linestyle='--' if method == 'random' else '-', label=LABELS[method])
        for column, (key, title, unit, factor) in enumerate(metrics):
            ax = axes[row_index, column]
            ax.set_title(CITIES.get(city, city) + '\n' + title, fontsize=10.5, pad=8)
            ax.set_xlabel('Training episode' if not evaluation else 'Evaluated checkpoint episode')
            ax.set_ylabel(unit)
            ax.grid(alpha=.22)
            ax.spines[['top', 'right']].set_visible(False)
            if key == 'completion_rate':
                ax.set_ylim(-2, 102)
            elif key in {'mean_exposure_seconds', 'mean_travel_seconds'}:
                ax.set_ylim(bottom=0)
            if limit is not None:
                ax.set_xlim(0, limit)
            else:
                ax.set_xlim(left=0)
            if evaluation and references:
                # axhline/axhspan use axes-relative x coordinates, so reference
                # policies cannot stretch the learned-checkpoint episode axis.
                for method, label in REFERENCE_LABELS.items():
                    reference = references[(city, method)]
                    if finite(reference.get(key)):
                        value = reference[key] * factor
                        deviation = reference[key + '_std'] * factor
                        ax.axhline(value, color=REFERENCE_COLORS[method], linestyle='--',
                                   linewidth=1.7, label=label)
                        ax.axhspan(value - deviation, value + deviation,
                                   color=REFERENCE_COLORS[method], alpha=.07)
        if evaluation and missing:
            axes[row_index, 0].text(.02, .05, 'No complete evaluation: ' + ', '.join(missing),
                transform=axes[row_index, 0].transAxes, fontsize=7.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    title = (f'Fixed-demand evaluation ({evaluation_label}): mean +/- 1 SD across {seed_count} seeds' if evaluation else
             f'Training curves: trailing mean of up to {window} episodes')
    if limit is not None:
        title += f' | first {limit} episodes'
    fig.suptitle(title, fontsize=16, y=.975)
    fig.text(.5, .925, f'Live snapshot {timestamp} | 5,000 episodes planned per city/method',
             ha='center', fontsize=10, color='#555555')
    handles = [Line2D([0], [0], color=COLORS[m], linewidth=2,
                     linestyle='--' if m == 'random' else '-', label=LABELS[m]) for m in methods]
    if evaluation and references:
        handles.extend(Line2D([0], [0], color=REFERENCE_COLORS[m], linewidth=1.7,
                              linestyle='--', label=label) for m, label in REFERENCE_LABELS.items())
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .062), ncol=len(handles), frameon=False)
    note = ('Only complete evaluation seed sets are plotted. SD is demand-seed variation, not training-run uncertainty.'
            if evaluation else 'Faint lines: raw episodes. Bold lines: trailing mean. Random is a fixed policy, not a learning algorithm.')
    fig.text(.5, .039, note, ha='center', fontsize=9, color='#555555')
    fig.text(.5, .015, 'Observed active time is censored at the deadline. Arrival-only travel time must be read together with completion.',
             ha='center', fontsize=9, color='#555555')
    save_figure(fig, output, stem)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--reference-results', type=Path,
                        help='Complete fixed-demand shortest-path and one-iteration DTA-SO evaluations')
    args = parser.parse_args(argv)
    if args.window < 1:
        parser.error('window must be positive')
    root = args.root.resolve()
    output = (args.output_dir or root / 'plots').resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    prior_enabled = manifest.get('policy_prior', {}).get('enabled', False)
    evaluation_policy, evaluation_label = evaluation_settings(manifest)
    data, summaries, flat_records = {}, [], []
    cities = list(dict.fromkeys(run['city'] for run in manifest['runs']))
    reference_path = reference_results_path(root, manifest, args.reference_results)
    references = (load_reference_results(reference_path, cities, manifest['evaluation_seeds'])
                  if reference_path else None)
    for run in manifest['runs']:
        records = read_records(Path(run['output_dir']) / 'metrics.jsonl')
        train = [row for row in records if row['phase'] == 'train']
        evaluations = complete_evaluations(records, manifest['evaluation_seeds'], policy=evaluation_policy)
        data[(run['city'], run['method'])] = {'train': train, 'evaluation': evaluations,
            'scheduled_evaluation': complete_evaluations(records, manifest['evaluation_seeds'])}
        state = json.loads(Path(run['state']).read_text())
        archives = list((Path(run['output_dir']) / 'episodes' / 'train').glob('episode_*.npz'))
        summaries.append(dict(city=run['city'], method=run['method'], process_status=state['status'],
            restart_count=state.get('attempt', 0), completed_training_episodes=train[-1]['episode'] if train else 0,
            archived_episodes=len(archives),
            recent_episode_count=len(train[-args.window:]),
            first_completion=average(train[:args.window], 'completion_rate'),
            recent={key: average(train[-args.window:], key) for key in SCALARS},
            latest_complete_evaluation=evaluations[-1] if evaluations else None))
        flat_records.extend(dict(city=run['city'], method=run['method'],
            **{key: row.get(key) for key in ('phase', 'episode', 'policy', 'seed', *SCALARS)}) for row in records)
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    summary = dict(timestamp=timestamp, smoothing_window=args.window, evaluation_policy=evaluation_policy, runs=summaries,
                   total_archived=sum(row['archived_episodes'] for row in summaries))
    if references is not None:
        summary['reference_results'] = str(reference_path)
        summary['reference_baselines'] = [dict(city=city, method=method, **values)
                                          for (city, method), values in references.items()]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    with (output / 'metrics_snapshot.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['city', 'method', 'phase', 'episode', 'policy', 'seed', *SCALARS])
        writer.writeheader()
        writer.writerows(flat_records)
    plot_grid(data, cities, output, timestamp, window=args.window)
    plot_grid(data, cities, output, timestamp, limit=100, window=args.window, stem='training_first100')
    plot_grid(data, cities, output, timestamp, evaluation=True, stem='evaluation_curves',
              evaluation_label=evaluation_label,
              references=references, seed_count=len(manifest['evaluation_seeds']))
    if prior_enabled:
        scheduled = {key: dict(value, evaluation=value['scheduled_evaluation']) for key, value in data.items()}
        plot_grid(scheduled, cities, output, timestamp, evaluation=True,
                  stem='evaluation_scheduled_prior', evaluation_label='scheduled prior',
                  references=references, seed_count=len(manifest['evaluation_seeds']))
    rewards = (('global_reward_sum', 'Shared physical traffic reward', 'reward', 1.),)
    if manifest.get('reward_mode') != 'global_only':
        rewards += (('selection_score', 'Selection score incl. individual timeouts',
                     'score / 1000 (diagnostic)', .001),)
    plot_grid(data, cities, output, timestamp, window=args.window, metrics=rewards, stem='reward_curves')
    print(json.dumps(dict(output_dir=str(output), timestamp=timestamp, total_archived=summary['total_archived'],
        runs=[dict(city=r['city'], method=r['method'], trained=r['completed_training_episodes'],
            archived=r['archived_episodes'], completion=r['recent']['completion_rate'],
            exposure_minutes=r['recent']['mean_exposure_seconds']/60 if r['recent']['mean_exposure_seconds'] is not None else None,
            evaluation_episode=r['latest_complete_evaluation']['episode'] if r['latest_complete_evaluation'] else None)
            for r in summaries]), indent=2))


if __name__ == '__main__':
    main()
