#!/usr/bin/env python3
"""Matched-demand archive analysis of revisits, persistent stalls and route warnings."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_no_revisit_archives import read_json

CITIES = ('ingolstadt_custom', 'gretz_armainvilliers', 'fontenay_en_parisis')
CAPACITIES = {}
LATE_EPISODES = {'ingolstadt_custom': range(1171, 1181),
                 'gretz_armainvilliers': range(740, 750),
                 'fontenay_en_parisis': range(140, 150)}


def cohorts(city):
    return [('random', 'random_control', range(1, 21)),
            ('online_ac', 'early_ac', range(101, 111)),
            ('online_ac', 'late_ac', LATE_EPISODES[city])]


def capacities(city):
    """Read the small leading capacity array, avoiding the huge dense adjacency."""
    if city not in CAPACITIES:
        lines, capture = [], False
        with (ROOT / 'networks' / city / 'edge_registry.json').open() as stream:
            for line in stream:
                if line.strip().startswith('"capacities":'):
                    capture = True
                    lines.append(line.split(':', 1)[1])
                elif capture:
                    lines.append(line)
                    if line.strip().startswith(']'):
                        break
        CAPACITIES[city] = np.asarray(json.loads(''.join(lines).strip().rstrip(',')))
    return CAPACITIES[city]


def route_warnings(path, seeds):
    result = {}
    pattern = re.compile(r"Vehicle '([^']+)'.*lane '([^']+)'.*time=([0-9]+(?:\.[0-9]+)?)")
    with path.open(errors='replace') as stream:
        for line in stream:
            if 'because there is no connection to the next edge' not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            identity, lane, time = match.groups()
            seed_match = re.match(r'ep(\d+)_', identity)
            if not seed_match or int(seed_match.group(1)) not in seeds:
                continue
            road = lane.rsplit('_', 1)[0]
            result.setdefault(int(seed_match.group(1)), {}).setdefault(identity, []).append((road, float(time)))
    return result


def physical_position(vehicle):
    junction = bool(vehicle.get('in_junction', False))
    key = 'junction_remaining_distance' if junction else 'remaining_distance'
    if vehicle.get(key) is None:
        return None
    return (int(vehicle['current_edge']), junction,
            vehicle.get('junction_lane') if junction else None, float(vehicle[key]))


def analyze(job):
    variant, city, method, path, warnings = job
    with zipfile.ZipFile(path) as archive:
        summary = read_json(archive, 'summary_json.npy')
        states = read_json(archive, 'states_json.npy')
        decisions = read_json(archive, 'decisions_json.npy')
        with archive.open('flow.npy') as stream:
            final_flow = np.lib.format.read_array(stream, allow_pickle=False)[-1]
    meta, metrics = summary['metadata'], summary['metrics']
    assert meta['session_protocol'] == 'event' and metrics['microstep_seconds'] == 1.
    seen, revisits, last_decision, events_by_window = {}, set(), {}, {}
    timestamps = np.asarray([s['time'] for s in states])
    repeated_events = 0
    for event in decisions:
        identity = str(event['vehicle_id'])
        source = event['current_edge']
        if source in seen.setdefault(identity, set()):
            revisits.add(identity)
            repeated_events += 1
        seen[identity].add(source)
        last_decision[identity] = max(last_decision.get(identity, -1.), event['decision_time'])
        window = event.get('window_index')
        if window is None:
            window = int(np.searchsorted(timestamps, event['decision_time'], side='right') - 1)
        events_by_window.setdefault(window, set()).add(identity)
    end = float(states[-1]['time'])
    final = {str(v['vehicle_id']): v for v in states[-1]['vehicles']}
    failed = {identity for identity, v in final.items() if v['departed'] and not v['arrived']}
    stats = Counter(N=len(final), arrived=sum(v['arrived'] for v in final.values()),
                    never_departed=sum(not v['departed'] for v in final.values()), departed_failed=len(failed),
                    decision_events=len(decisions), repeated_events=repeated_events,
                    failed_revisit=len(failed & revisits), failed_motionless_window_seconds=0)
    cap = capacities(city)
    queues = (final_flow[:, 2] * cap >= 5.) & (final_flow[:, 3] >= .9 * final_flow[:, 2]) & (final_flow[:, 1] < .05)
    stalled = {300: set(), 600: set()}
    tail = [(float(s['time']), {str(v['vehicle_id']): v for v in s['vehicles']}) for s in states if s['time'] >= end - 600.]
    for window in (300, 600):
        times = [(t, values) for t, values in tail if t >= end - window]
        for identity in failed:
            samples = [values[identity] for _, values in times]
            if len(samples) < 2 or not all(v['departed'] and not v['arrived'] for v in samples):
                continue
            positions = [physical_position(v) for v in samples]
            if (all(p is not None for p in positions)
                    and len({p[:3] for p in positions}) == 1
                    and max(p[3] for p in positions) - min(p[3] for p in positions) <= 1.
                    and last_decision.get(identity, -1.) < end - window):
                stalled[window].add(identity)
    stats.update(failed_stall300=len(stalled[300]), failed_stall600=len(stalled[600]),
                 failed_both=len(stalled[300] & revisits), failed_stall_only=len(stalled[300] - revisits),
                 failed_revisit_only=len((failed & revisits) - stalled[300]),
                 failed_neither=len(failed - stalled[300] - revisits))
    stats['failed_in_junction'] = sum(bool(final[i].get('in_junction', False)) for i in failed)
    stats['stall300_in_junction'] = sum(bool(final[i].get('in_junction', False)) for i in stalled[300])
    for identity in failed:
        v = final[identity]
        edge = int(v['current_edge'])
        stats['failed_low_speed'] += int(not v.get('in_junction', False) and final_flow[edge, 1] < .05)
        stats['failed_queue'] += int(not v.get('in_junction', False) and queues[edge])
        stats['failed_connection_warning'] += int(identity in warnings)
        if identity in stalled[300]:
            stats['stall300_connection_warning'] += int(identity in warnings)
            stats['stall300_preceding_connection_warning'] += int(any(t <= end - 300 for _, t in warnings.get(identity, [])))
    stats['arrived_connection_warning'] = sum(identity in warnings for identity, v in final.items() if v['arrived'])
    active_failed_seconds = 0.
    for identity in failed:
        v = final[identity]
        departure = v.get('actual_departure_time')
        if departure is None:
            departure = v.get('observed_departure_time', v['departure_time'])
        active_failed_seconds += max(0., end - float(departure))
    for window, (previous, current) in enumerate(zip(states, states[1:])):
        before = {str(v['vehicle_id']): v for v in previous['vehicles'] if str(v['vehicle_id']) in failed}
        for v in current['vehicles']:
            identity = str(v['vehicle_id'])
            if identity not in failed:
                continue
            old = before[identity]
            if not old['departed'] or not v['departed']:
                continue
            a, b = physical_position(old), physical_position(v)
            if a is not None and b is not None and a[:3] == b[:3] and abs(a[3] - b[3]) <= 1.:
                # Avoid calling a vehicle stationary when it crossed a fork and returned.
                if identity not in events_by_window.get(window, set()):
                    stats['failed_motionless_window_seconds'] += current['time'] - previous['time']
    cohort = 'random_control' if method == 'random' else ('early_ac' if meta['episode'] <= 110 else 'late_ac')
    return dict(variant=variant, city=city, method=method, cohort=cohort, episode=meta['episode'], seed=metrics['seed'],
        demand_hash=meta.get('demand_sha256', meta.get('demand_realization_sha256')),
        deadline=summary['deadline_seconds'], counts=dict(stats), failed_active_seconds=active_failed_seconds,
        completion=metrics['completion_rate'], reward=metrics['global_reward_sum'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    variants = dict(old_no_prior=ROOT/'outputs/city_baseline_dataset_5000ep_20261001_182056',
                    old_prior=ROOT/'outputs/city_baseline_dataset_sp_prior_5000ep_20261002_082844',
                    no_revisit=ROOT/'outputs/city_baseline_dataset_no_revisit_5000ep_20261004_201004')
    jobs = []
    for variant, root in variants.items():
        for city in CITIES:
            for method, _, episodes in cohorts(city):
                warnings = route_warnings(root/'logs'/f'{city}_{method}.log', {42 + e for e in episodes})
                for episode in episodes:
                    path = root/city/method/'episodes/train'/f'episode_{episode:06d}.npz'
                    jobs.append((variant, city, method, path, warnings.get(42 + episode, {})))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(analyze, jobs, chunksize=1), 1):
            results.append(result)
            if i % 15 == 0:
                print(json.dumps(dict(processed=i, total=len(jobs))), flush=True)
    by_case = {(r['variant'], r['city'], r['method'], r['episode']): r for r in results}
    for city in CITIES:
        for method, _, episodes in cohorts(city):
            for episode in episodes:
                pair = [by_case[(v, city, method, episode)] for v in variants]
                assert len({r['demand_hash'] for r in pair}) == 1 and pair[0]['demand_hash']
                assert len({r['counts']['N'] for r in pair}) == 1
                assert len({r['deadline'] for r in pair}) == 1
    groups = []
    for variant in variants:
        for city in CITIES:
            for method, cohort, _ in cohorts(city):
                rows = [r for r in results if (r['variant'], r['city'], r['method'], r['cohort']) == (variant, city, method, cohort)]
                counts = Counter()
                for r in rows:
                    counts.update(r['counts'])
                failure = counts['departed_failed']
                groups.append(dict(variant=variant, city=city, method=method, cohort=cohort, episodes=len(rows),
                    counts=dict(counts), mean_completion=100*np.mean([r['completion'] for r in rows]),
                    stall300_pct=100*counts['failed_stall300']/failure if failure else None,
                    stall600_pct=100*counts['failed_stall600']/failure if failure else None,
                    failed_revisit_pct=100*counts['failed_revisit']/failure if failure else None,
                    failed_warning_pct=100*counts['failed_connection_warning']/failure if failure else None,
                    failed_queue_pct=100*counts['failed_queue']/failure if failure else None,
                    failed_motionless_time_pct=100*counts['failed_motionless_window_seconds']/sum(r['failed_active_seconds'] for r in rows) if failure else None))
    report = dict(captured_at=datetime.now(timezone.utc).isoformat(), groups=groups, episodes=results,
        caveats=['Revisit is observed fork-source proxy in old archives, not full physical route history.',
                 'Stall proxy: constant road/junction and within1m position over last300/600 seconds, no decisions.',
                 'Snapshot movement sampled30s; inactivity-time metric is a proxy, not complete1s telemetry.',
                 'Variants are not randomized independent training replicates; comparisons do not identify a causal fraction.'])
    (args.output_dir/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
