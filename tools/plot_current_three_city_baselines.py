#!/usr/bin/env python3
"""Freeze manifest-selected live baselines and plot comparable per-city traffic scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS = ('online_ac', 'gnac', 'dreamer', 'mbpo', 'ippo', 'mappo', 'iql', 'mamba')
LABELS = dict(zip(METHODS, ('Online AC', 'GNAC', 'Dreamer', 'MBPO', 'IPPO', 'MAPPO', 'IQL', 'MAMBA')))
COLORS = dict(zip(METHODS, ('#2463a5', '#22825c', '#df8b26', '#8a4ead', '#c75c80', '#aa6b27', '#33a4af', '#a53232')))
CITIES = {'ingolstadt_custom': 'Ingolstadt', 'coulommiers': 'Coulommiers', 'montereau_fault_yonne': 'Montereau-Fault-Yonne'}
NUMERIC_FIELDS = ('completion_rate', 'observed_mean_seconds', 'cumulative_vehicle_seconds',
                  'exposure_seconds', 'started_vehicle_count', 'vehicle_count', 'never_departed',
                  'global_reward_sum', 'vehicle_timeout_penalty_sum', 'selection_score')
DEFINITIONS = {
    'canonical_reward_minutes': '-ATT_T / 60, with ATT_T in seconds; higher is better.',
    'ATT_T': 'Valid observed_mean_seconds, otherwise cumulative_vehicle_seconds (or exposure_seconds) / actual started vehicles. Started = started_vehicle_count or vehicle_count - never_departed. Missing denominator stays undefined; mean_exposure_seconds alone is never used.',
    'completion_rate': 'Logged completion_rate; plotted as percent of generated demand.',
    'objectives': 'Historical objectives are mixed. Dashed / triangles = legacy or unmarked historical objective; solid / circles = explicitly observed_mean; dotted / crosses = mixed or unknown explicit objective. The common traffic score is reconstructed, not the original timeout-weighted training loss.',
    'training': 'Trailing mean resets at objective changes, episode gaps, and undefined values; raw episodes shown faintly.',
    'evaluation': 'Only deterministic evaluations with every expected unique seed; mean +/- population SD. SD measures demand-seed variation, not training-run uncertainty. Undefined metric at any seed leaves that metric undefined.',
    'sources': 'Only output_dir paths in the frozen manifest; no external references or stopped ablation runs. Fixed shortest-path references are excluded from training plots.',
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def canonical_att(row):
    observed = row.get('observed_mean_seconds')
    if finite(observed) and observed >= 0:
        return float(observed), 'observed_mean_seconds'
    cumulative = next((row[k] for k in ('cumulative_vehicle_seconds', 'exposure_seconds')
                       if finite(row.get(k)) and row[k] >= 0), None)
    started = row.get('started_vehicle_count')
    if not (finite(started) and started >= 0):
        count, never = row.get('vehicle_count'), row.get('never_departed')
        started = count - never if finite(count) and finite(never) and 0 <= never <= count else None
    if cumulative is not None and finite(started) and started > 0:
        return float(cumulative / started), 'cumulative_seconds / started_vehicles'
    return None, 'undefined'


def normalize(row):
    att, source = canonical_att(row)
    objective = row.get('reward_objective')
    if objective is None:
        objective = 'legacy_unmarked'
    result = {key: row.get(key) for key in ('phase', 'episode', 'policy', 'seed')}
    result.update({key: float(row[key]) if finite(row.get(key)) else None for key in NUMERIC_FIELDS})
    result.update(objective=objective, att_seconds=att, att_source=source,
                  canonical_reward_minutes=-att / 60 if att is not None else None)
    return result


def read_snapshot(path):
    """Bound the read to opening file size; retain only complete lines, last replay wins."""
    unique = {}
    quality = {'path': str(path), 'exists': path.exists(), 'duplicate_rows': 0,
               'partial_lines_ignored': 0, 'malformed_lines_ignored': 0, 'complete_lines': 0,
               'snapshot_bytes': 0, 'read_started_at': datetime.now(timezone.utc).isoformat()}
    digest = hashlib.sha256()
    if path.exists():
        with path.open('rb') as stream:
            remaining = os.fstat(stream.fileno()).st_size
            quality['snapshot_bytes'] = remaining
            while remaining:
                line = stream.readline(remaining)
                if not line:
                    break
                remaining -= len(line)
                if not line.endswith(b'\n'):
                    quality['partial_lines_ignored'] += 1
                    break
                digest.update(line)
                quality['complete_lines'] += 1
                try:
                    row = json.loads(line)
                    key = tuple(row[name] for name in ('phase', 'episode', 'policy', 'seed'))
                    if not isinstance(row['episode'], int) or not isinstance(row['seed'], int):
                        raise ValueError('Invalid episode/seed')
                    normalized = normalize(row)
                    if key in unique:
                        quality['duplicate_rows'] += 1
                    unique[key] = normalized
                except (ValueError, TypeError, KeyError):
                    quality['malformed_lines_ignored'] += 1
    quality['complete_content_sha256'] = digest.hexdigest()
    quality['read_finished_at'] = datetime.now(timezone.utc).isoformat()
    return sorted(unique.values(), key=lambda r: (r['episode'], r['phase'], r['policy'], r['seed'])), quality


def complete_evaluations(records, seeds):
    groups = {}
    for row in records:
        if row['phase'] == 'evaluation' and row['policy'] == 'deterministic' and row['seed'] in seeds:
            groups.setdefault(row['episode'], {})[row['seed']] = row
    result = []
    for episode, group in sorted(groups.items()):
        if set(group) != set(seeds):
            continue
        objectives = {row['objective'] for row in group.values()}
        out = {'phase': 'evaluation', 'episode': episode, 'seed_count': len(seeds),
               'objective': next(iter(objectives)) if len(objectives) == 1 else 'mixed'}
        for metric in ('canonical_reward_minutes', 'completion_rate'):
            values = [group[seed][metric] for seed in seeds]
            valid = all(finite(value) for value in values)
            out[metric] = float(np.mean(values)) if valid else None
            out[metric + '_std'] = float(np.std(values)) if valid else None
        result.append(out)
    return result, len(groups) - len(result)


def segments(rows, *, max_gap=1):
    chunks = []
    for row in rows:
        if not chunks or row['objective'] != chunks[-1][-1]['objective'] or row['episode'] - chunks[-1][-1]['episode'] != max_gap:
            chunks.append([])
        chunks[-1].append(row)
    return chunks


def smooth_segments(rows, metric, window):
    """Preserve unknown values as gaps and never smooth across episode/objective gaps."""
    result = []
    for chunk in segments(rows):
        history = []
        for row in chunk:
            value = row.get(metric)
            if not finite(value):
                history = []
                result.append(None)
            else:
                history.append(value)
                history = history[-window:]
                result.append(float(np.mean(history)))
    return result


def style(objective):
    if objective == 'observed_mean':
        return '-', 'o'
    if objective in ('legacy_exposure', 'legacy_unmarked'):
        return '--', '^'
    return ':', 'x'


def plot_city(city, runs, output, timestamp, window, eval_every, seeds):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(top=.85, bottom=.30, hspace=.33, wspace=.2)
    metrics = ('canonical_reward_minutes', 'completion_rate')
    for row_index, phase in enumerate(('train', 'evaluation')):
        unavailable = []
        for method in METHODS:
            rows = runs.get(method, {}).get(phase, [])
            if not rows:
                unavailable.append(LABELS[method])
                continue
            for chunk in segments(rows, max_gap=1 if phase == 'train' else eval_every):
                line, marker = style(chunk[0]['objective'])
                x = [r['episode'] for r in chunk]
                for col, metric in enumerate(metrics):
                    ax = axes[row_index, col]
                    scale = 100 if metric == 'completion_rate' else 1
                    y = [r[metric] * scale if finite(r.get(metric)) else np.nan for r in chunk]
                    if phase == 'train':
                        ax.plot(x, y, color=COLORS[method], alpha=.12, linewidth=.55)
                        smoothed = [r[metric + '_smoothed'] * scale if finite(r.get(metric + '_smoothed')) else np.nan for r in chunk]
                        ax.plot(x, smoothed, color=COLORS[method], linestyle=line, linewidth=1.8,
                                marker=marker if len(chunk) == 1 else None, markersize=3)
                    else:
                        sd = [r[metric + '_std'] * scale if finite(r.get(metric + '_std')) else np.nan for r in chunk]
                        if any(np.isfinite(y)):
                            ax.errorbar(x, y, yerr=sd, color=COLORS[method], linestyle=line,
                                        marker=marker, markersize=3.5, linewidth=1.2, capsize=2)
        for col, metric in enumerate(metrics):
            ax = axes[row_index, col]
            ax.set_title(('Training: ' if phase == 'train' else f'Deterministic evaluation ({len(seeds)} seeds): ') +
                         ('canonical reward' if col == 0 else 'completion'))
            ax.set_xlabel('Training episode' if phase == 'train' else 'Evaluated checkpoint episode')
            ax.set_ylabel('-ATT_T (minutes; higher is better)' if col == 0 else 'Completion (%)')
            ax.set_xlim(left=0)
            if col == 1:
                ax.set_ylim(-2, 102)
            ax.grid(alpha=.22)
            ax.spines[['top', 'right']].set_visible(False)
            if unavailable:
                ax.text(.015, .025, 'No ' + ('training data' if phase == 'train' else 'complete evaluation') + ':\n' + ', '.join(unavailable),
                        transform=ax.transAxes, fontsize=7, bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none'})
    fig.suptitle(CITIES[city] + ' — current baseline training and evaluation', fontsize=17, y=.965)
    fig.text(.5, .91, f'Snapshot {timestamp} | trailing window {window}; reset at objective changes and episode gaps', ha='center', fontsize=10)
    handles = [Line2D([0], [0], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in METHODS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .185), ncol=8, frameon=False)
    notes = [
        'Dashed / triangles: legacy or unmarked objective. Solid / circles: observed_mean. Dotted / crosses: mixed/unknown objective.',
        'Mixed historical objectives: reconstructed shared traffic score, not the original timeout-weighted training loss.',
        'ATT_T = observed_mean_seconds, otherwise cumulative vehicle seconds / actual started vehicles; reward = -ATT_T / 60.',
        'Started = started_vehicle_count or vehicle_count - never_departed. Missing accounting stays a gap. Completion uses all demand.',
        f'Evaluation: seeds {", ".join(map(str, seeds))}; mean ± SD; complete seed sets only. Faint training lines: raw episodes.',
        'Only manifest-selected run directories. Fixed shortest-path reference is not a trained baseline and is omitted.',
    ]
    for index, note in enumerate(notes):
        fig.text(.5, .167 - index * .022, note, ha='center', fontsize=8, color='#444444')
    for extension in ('png', 'pdf'):
        fig.savefig(output / f'{city}_current_baselines.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ['city', 'method', 'episode'])
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=[10042, 10043, 10044])
    args = parser.parse_args(argv)
    if args.window < 1 or len(set(args.eval_seeds)) != len(args.eval_seeds) or len(args.eval_seeds) != 3:
        parser.error('window must be positive; evaluation requires three distinct seeds')
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    timestamp = datetime.now(timezone.utc).isoformat()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.snapshot.json').write_bytes(manifest_bytes)
    data = {city: {} for city in CITIES}
    summary, records_csv, trace_csv = [], [], []
    selected = set()
    for run in manifest['runs']:
        city, method = run['city'], run['method']
        if city not in CITIES or method not in METHODS:
            continue
        if (city, method) in selected:
            raise ValueError(f'Duplicate manifest city/method: {city}/{method}')
        selected.add((city, method))
        directory = Path(run['output_dir'])
        # Manifest output_dir is authoritative; relative paths follow invocation cwd.
        records, quality = read_snapshot(directory / 'metrics.jsonl')
        train = [row for row in records if row['phase'] == 'train']
        evaluations, incomplete = complete_evaluations(records, args.eval_seeds)
        for metric in ('canonical_reward_minutes', 'completion_rate'):
            for row, value in zip(train, smooth_segments(train, metric, args.window)):
                row[metric + '_smoothed'] = value
        data[city][method] = {'train': train, 'evaluation': evaluations}
        records_csv.extend(dict(city=city, method=method, **row) for row in records)
        trace_csv.extend(dict(city=city, method=method, **row) for row in train + evaluations)
        quality.update(undefined_att_rows=sum(row['att_seconds'] is None for row in records),
                       incomplete_evaluation_groups=incomplete,
                       missing_training_episodes=(max(r['episode'] for r in train) - len({r['episode'] for r in train})) if train else 0)
        latest = train[-20:]
        averages = {}
        for metric in ('canonical_reward_minutes', 'completion_rate'):
            values = [row[metric] for row in latest if finite(row.get(metric))]
            averages[metric] = float(np.mean(values)) if values else None
            averages[metric + '_valid_count'] = len(values)
        summary.append(dict(city=city, method=method, output_dir=str(directory.resolve()),
                            training_rows=len(train), last_training_episode=train[-1]['episode'] if train else None,
                            complete_evaluations=len(evaluations), last_evaluation_episode=evaluations[-1]['episode'] if evaluations else None,
                            latest20_rows=len(latest), latest20_averages=averages,
                            objective_counts={obj: sum(r['objective'] == obj for r in train) for obj in sorted({r['objective'] for r in train})},
                            data_quality=quality))
    payload = dict(snapshot_started_at=timestamp, snapshot_finished_at=datetime.now(timezone.utc).isoformat(),
                   manifest=str(args.manifest.resolve()), manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   window=args.window, evaluation_seeds=args.eval_seeds, definitions=DEFINITIONS, runs=summary)
    (output / 'snapshot_summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    write_csv(output / 'records.snapshot.csv', records_csv)
    write_csv(output / 'plotted_traces.csv', trace_csv)
    write_csv(output / 'run_summary.csv', [dict(city=r['city'], method=r['method'], training_rows=r['training_rows'],
              last_training_episode=r['last_training_episode'], complete_evaluations=r['complete_evaluations'],
              **{'latest20_' + key: value for key, value in r['latest20_averages'].items()},
              **{key: value for key, value in r['data_quality'].items() if isinstance(value, (int, bool))}) for r in summary])
    for city in CITIES:
        plot_city(city, data[city], output, timestamp, args.window, manifest.get('eval_every', 100), args.eval_seeds)
    print(json.dumps({'output_dir': str(output), 'runs': len(summary), 'snapshot_started_at': timestamp}))


if __name__ == '__main__':
    main()
