#!/usr/bin/env python3
"""Compute observed decision-source edge revisits from archived real training episodes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.plot_city_baseline_dataset import CITIES, COLORS, LABELS, METHODS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def event_revisits(decisions):
    """Count observed source visits, independently per vehicle within one episode."""
    seen = {}
    repeated_vehicles = set()
    revisits = 0
    for decision in decisions:
        vehicle, source = str(decision['vehicle_id']), int(decision['current_edge'])
        visited = seen.setdefault(vehicle, set())
        if source in visited:
            revisits += 1
            repeated_vehicles.add(vehicle)
        visited.add(source)
    events, vehicles = len(decisions), len(seen)
    return dict(decision_events=events, repeated_source_events=revisits,
        decision_vehicles=vehicles, revisiting_vehicles=len(repeated_vehicles),
        event_revisit_ratio=revisits/events if events else None,
        vehicle_revisit_ratio=len(repeated_vehicles)/vehicles if vehicles else None)


def read_json_member(archive, name):
    with archive.open(name) as stream:
        return json.loads(np.lib.format.read_array(stream, allow_pickle=False).tobytes())


def analyze_job(job):
    path, city, method, cached = job
    with zipfile.ZipFile(path) as archive:
        # Repacking changes file mtime/size but not member CRC/size. These keys
        # also invalidate a cached statistic if a checkpoint replay replaces data.
        signature = [[archive.getinfo(name).CRC, archive.getinfo(name).file_size]
                     for name in ('summary_json.npy', 'decisions_json.npy')]
        if cached is not None and cached['signature'] == signature:
            return cached
        summary = read_json_member(archive, 'summary_json.npy')
        metadata = summary['metadata']
        if (summary['source'] != 'real_sumo' or metadata['phase'] != 'train'
                or metadata['city'] != city or metadata['method'] != method):
            raise ValueError(f'unexpected training archive provenance: {path}')
        decisions = read_json_member(archive, 'decisions_json.npy')
    result = event_revisits(decisions)
    if result['decision_events'] != summary['metrics']['event_count']:
        raise ValueError(f'event count mismatch: {path}')
    return dict(path=str(path), city=city, method=method, episode=metadata['episode'],
        population=metadata['total_vehicle_count'], completion_rate=summary['metrics']['completion_rate'],
        session_protocol=metadata['session_protocol'], signature=signature, **result)


def pooled(rows):
    events = sum(row['decision_events'] for row in rows)
    vehicles = sum(row['decision_vehicles'] for row in rows)
    population = sum(row['population'] for row in rows)
    repeated_events = sum(row['repeated_source_events'] for row in rows)
    repeated_vehicles = sum(row['revisiting_vehicles'] for row in rows)
    return dict(episodes=len(rows), decision_events=events, decision_vehicles=vehicles,
        event_revisit_ratio=repeated_events/events if events else None,
        vehicle_revisit_ratio=repeated_vehicles/vehicles if vehicles else None,
        decision_vehicle_coverage=vehicles/population if population else None,
        mean_completion_rate=float(np.mean([row['completion_rate'] for row in rows])) if rows else None)


def draw(rows, cities, output, timestamp, window, *, limit=None):
    fig, axes = plt.subplots(len(cities), 3, figsize=(17, 3.7*len(cities)), squeeze=False)
    fig.subplots_adjust(left=.065, right=.985, top=.85, bottom=.16, hspace=.53, wspace=.25)
    metrics = (('event_revisit_ratio', 'Repeated source-edge visits / decision events'),
               ('vehicle_revisit_ratio', 'Vehicles with revisit / vehicles with decisions'),
               ('completion_rate', 'Vehicle completion (all realized vehicles)'))
    for ri, city in enumerate(cities):
        for method in METHODS:
            series = [row for row in rows if row['city'] == city and row['method'] == method
                      and (limit is None or row['episode'] <= limit)]
            series.sort(key=lambda row: row['episode'])
            if not series:
                continue
            x = [row['episode'] for row in series]
            smooth = [pooled(series[max(0, index-window+1):index+1]) for index in range(len(series))]
            for ci, (key, _) in enumerate(metrics):
                y = [100*row[key] if row[key] is not None else np.nan for row in series]
                skey = 'mean_completion_rate' if key == 'completion_rate' else key
                sy = [100*row[skey] if row[skey] is not None else np.nan for row in smooth]
                axes[ri, ci].plot(x, y, color=COLORS[method], alpha=.12, linewidth=.6)
                axes[ri, ci].plot(x, sy, color=COLORS[method], linewidth=2,
                    linestyle='--' if method == 'random' else '-')
        for ci, (_, title) in enumerate(metrics):
            ax = axes[ri, ci]
            ax.set_title(CITIES.get(city, city) + '\n' + title, fontsize=10)
            ax.set_xlabel('Archived training episode')
            ax.set_ylabel('%')
            ax.set_ylim(-2, 102)
            ax.set_xlim(left=0, right=limit)
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
    title = 'Observed fork-decision road revisits during training'
    if limit is not None:
        title += f' | first {limit} episodes'
    fig.suptitle(title, fontsize=16, y=.975)
    fig.text(.5, .925, f'Archive snapshot: {timestamp} | directed source edges, counted per vehicle and episode',
             ha='center', fontsize=10, color='#555555')
    handles = [Line2D([0], [0], color=COLORS[m], label=LABELS[m], linewidth=2,
                     linestyle='--' if m == 'random' else '-') for m in METHODS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .062), ncol=5, frameon=False)
    fig.text(.5, .04, f'Faint: per episode. Bold: pooled counts over up to {window} episodes (completion: episode mean).',
             ha='center', fontsize=9, color='#555555')
    fig.text(.5, .015, 'Not a full-route revisit rate: non-fork roads and visits skipped by 1-second observations are absent. No decisions = undefined rate.',
             ha='center', fontsize=9, color='#555555')
    stem = 'revisit_first100' if limit is not None else 'revisit_curves'
    fig.savefig(output / f'{stem}.png', dpi=170, facecolor='white')
    fig.savefig(output / f'{stem}.pdf', facecolor='white')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--window', type=int, default=20)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.window < 1:
        parser.error('workers and window must be positive')
    root = args.root.resolve()
    output = (args.output_dir or root / 'plots' / 'revisits').resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / 'cache.jsonl'
    cache = {}
    if cache_path.exists():
        with cache_path.open() as stream:
            for line in stream:
                if line.endswith('\n'):
                    row = json.loads(line)
                    cache[row['path']] = row
    manifest = json.loads((root / 'manifest.json').read_text())
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    jobs = []
    for run in manifest['runs']:
        for path in sorted((Path(run['output_dir']) / 'episodes' / 'train').glob('episode_*.npz')):
            jobs.append((path, run['city'], run['method'], cache.get(str(path))))
    rows = []
    print(json.dumps({'snapshot': timestamp, 'archives': len(jobs), 'workers': args.workers}), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor, cache_path.open('a') as stream:
        for count, row in enumerate(executor.map(analyze_job, jobs, chunksize=8), 1):
            rows.append(row)
            old = cache.get(row['path'])
            if old != row:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            if count % 100 == 0 or count == len(jobs):
                stream.flush()
                print(json.dumps({'processed': count, 'total': len(jobs)}), flush=True)
    fields = ['city', 'method', 'episode', 'population', 'completion_rate', 'decision_events',
              'repeated_source_events', 'decision_vehicles', 'revisiting_vehicles',
              'event_revisit_ratio', 'vehicle_revisit_ratio', 'session_protocol']
    with (output / 'episode_revisits.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    runs = []
    for run in manifest['runs']:
        series = sorted((r for r in rows if r['city'] == run['city'] and r['method'] == run['method']),
                        key=lambda r: r['episode'])
        runs.append(dict(city=run['city'], method=run['method'], episode_count=len(series),
            latest_episode=series[-1]['episode'] if series else 0,
            first=pooled(series[:args.window]), recent=pooled(series[-args.window:]), overall=pooled(series)))
    summary = dict(snapshot=timestamp, window=args.window, archives=len(rows), runs=runs,
        event_definition='sum_v(number of observed source-edge events - unique observed source edges) / all decision events',
        vehicle_definition='vehicles with any repeated decision source / vehicles with >=1 decision',
        limitation='Observed directed fork-decision source roads only; not full actual route histories or a guaranteed lower bound on full-route revisit ratio.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    cities = list(dict.fromkeys(run['city'] for run in manifest['runs']))
    draw(rows, cities, output, timestamp, args.window)
    draw(rows, cities, output, timestamp, args.window, limit=100)
    print(json.dumps({'output': str(output), 'archives': len(rows), 'runs': runs}, indent=2), flush=True)


if __name__ == '__main__':
    main()
