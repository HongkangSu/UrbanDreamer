#!/usr/bin/env python3
"""Bounded read-only audit of completed early/late MESO episode archives."""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology, NextHopCandidates


def describe(values):
    x = np.asarray(values, dtype=float)
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                p90=float(np.quantile(x, .9)) if len(x) else None,
                sum=float(x.sum()))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(path, episode, cohort, candidates):
    before = path.stat()
    with np.load(path, allow_pickle=False) as archive:
        summary = json.loads(archive['summary_json'].tobytes())
        states = json.loads(archive['states_json'].tobytes())
        events = json.loads(archive['decisions_json'].tobytes())
        exposure = float(archive['active_vehicle_seconds'].sum())
    assert summary['source'] == 'real_sumo'
    assert summary['metadata']['config']['simulation']['traffic_model'] == 'meso'
    assert len(events) == summary['metrics']['online_events']
    network = candidates.network
    initial = {v['vehicle_id']: v for v in states[0]['vehicles']}
    final = {v['vehicle_id']: v for v in states[-1]['vehicles']}
    stats = defaultdict(Counter)
    seen = defaultdict(set)
    conditional = defaultdict(Counter)
    turn_counts = Counter()
    event_times = defaultdict(list)
    for event in events:
        vid = event['vehicle_id']
        current, target, destination = (int(event[k]) for k in ('current_edge', 'next_edge', 'destination_edge'))
        event_times[vid].append(float(event['decision_time']))
        turn_counts[(current, destination, target)] += 1
        choices = candidates.candidates(current, destination)
        assert target in choices
        costs = candidates.costs(destination)
        # Each alternative includes its own traversal cost, not just downstream cost.
        scores = {edge: float(network.free_flow_travel_seconds[edge] + costs[edge]) for edge in choices}
        excess = max(0., scores[target] - min(scores.values()))
        nonprogress = int(costs[target] >= costs[current] - 1e-8)
        repeated = int(current in seen[vid])
        seen[vid].add(current)
        values = dict(events=1, repeated_source_events=repeated,
                      nonprogress_events=nonprogress, nonshortest_events=int(excess > 1e-5),
                      excess_ff_seconds=excess, source_edge_length_sum_m=float(network.lengths_m[current]))
        stats[vid].update(values)
        conditional[(current, destination)].update(values)
    assert all(all(a <= b for a, b in zip(times, times[1:])) for times in event_times.values())
    snapshot_profiles = []
    for state in states:
        snapshot_profiles.append(dict(episode=episode, cohort=cohort, time=state['time'],
                                      active=state['active_vehicle_count'], arrived=state['arrived_vehicle_count']))
    for previous, current in zip(states, states[1:]):
        previous_by_id = {v['vehicle_id']: v for v in previous['vehicles']}
        for v in current['vehicles']:
            vid = v['vehicle_id']; old = previous_by_id[vid]
            if not old['departed'] or old['arrived']:
                continue
            dt = min(float(current['time']), float(v.get('arrival_time', current['time']))) - float(previous['time'])
            if dt <= 0:
                continue
            source, target = old['current_edge'], v['current_edge']
            same = source == target
            times = event_times[vid]
            no_events = bisect_right(times, float(current['time'])) == bisect_left(times, float(previous['time']))
            delta = float(old['remaining_distance']) - float(v['remaining_distance'])
            minimum = delta if same else delta + float(candidates.distances(target)[source])
            assert np.isfinite(minimum)
            stats[vid].update(dict(boundary_exposure_seconds=dt,
                                  boundary_distance_lower_bound_m=max(0., minimum),
                                  boundary_intervals=1,
                                  same_edge_intervals=int(same),
                                  same_edge_no_position_progress_intervals=int(same and abs(delta) < .1),
                                  same_edge_no_position_progress_seconds=dt if same and abs(delta) < .1 else 0.,
                                  stationary_no_event_seconds=dt if same and abs(delta) < .1 and no_events else 0.,
                                  stationary_no_event_intervals=int(same and abs(delta) < .1 and no_events),
                                  same_edge_negative_position_progress_intervals=int(same and delta < -.1)))
    rows = []
    end = float(states[-1]['time'])
    for vid, v in final.items():
        origin = initial[vid]['current_edge']; destination = v['destination_edge']
        actual = v['actual_departure_time']
        travel = float(v['arrival_time']) - float(actual) if v['arrived'] else None
        observed = (float(v.get('arrival_time', end)) - float(actual)) if actual is not None else 0.
        ff = float(network.free_flow_travel_seconds[origin] + candidates.costs(destination)[origin])
        row = dict(episode=episode, cohort=cohort, vehicle_id=vid, origin=origin, destination=destination,
                   scheduled_departure=initial[vid]['departure_time'], actual_departure=actual,
                   departure_bin=int(initial[vid]['departure_time'] // 300), departed=bool(v['departed']),
                   arrived=bool(v['arrived']), travel_seconds=travel, observed_travel_seconds=observed,
                   shortest_od_ff_seconds=ff,
                   shortest_od_distance_m=float(network.lengths_m[origin] + candidates.distances(destination)[origin]),
                   insertion_delay_seconds=float(actual)-float(initial[vid]['departure_time']) if actual is not None else None)
        for key in ('events', 'repeated_source_events', 'nonprogress_events', 'nonshortest_events',
                    'excess_ff_seconds', 'source_edge_length_sum_m', 'boundary_exposure_seconds',
                    'boundary_distance_lower_bound_m', 'boundary_intervals', 'same_edge_intervals',
                    'same_edge_no_position_progress_intervals', 'same_edge_no_position_progress_seconds',
                    'same_edge_negative_position_progress_intervals', 'stationary_no_event_seconds', 'stationary_no_event_intervals'):
            row[key] = stats[vid][key]
        rows.append(row)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert len(rows) == int(summary['metrics']['online_vehicle_count'])
    assert sum(not r['arrived'] for r in rows) == int(summary['metrics']['online_nonarrival_count'])
    assert abs(sum(r['observed_travel_seconds'] for r in rows) - exposure) < 1e-6
    result = dict(episode=episode, archive_index=episode-1, cohort=cohort, path=str(path),
                  archive_size=before.st_size, archive_mtime_ns=before.st_mtime_ns,
                  completed=summary['completed'], truncated=summary['episode_truncated'],
                  boundary_snapshots=len(states), exact_active_vehicle_seconds=exposure,
                  metrics=summary['metrics'], **aggregate(rows))
    conditions = [dict(episode=episode, cohort=cohort, current_edge=k[0], destination=k[1], **v) for k, v in conditional.items()]
    turns = [dict(episode=episode, cohort=cohort, current_edge=k[0], destination=k[1], next_edge=k[2], count=n) for k, n in turn_counts.items()]
    return result, rows, snapshot_profiles, conditions, turns


def aggregate(rows):
    n = len(rows); events = sum(r['events'] for r in rows)
    sums = {k: sum(r[k] for r in rows) for k in ('events', 'repeated_source_events', 'nonprogress_events',
            'nonshortest_events', 'excess_ff_seconds', 'source_edge_length_sum_m', 'boundary_exposure_seconds',
            'boundary_distance_lower_bound_m', 'boundary_intervals', 'same_edge_intervals',
            'same_edge_no_position_progress_intervals', 'same_edge_no_position_progress_seconds',
            'same_edge_negative_position_progress_intervals', 'observed_travel_seconds', 'stationary_no_event_seconds', 'stationary_no_event_intervals')}
    return dict(population=n, departed=sum(r['departed'] for r in rows), arrived=sum(r['arrived'] for r in rows),
                completion_rate=sum(r['arrived'] for r in rows)/n,
                event_vehicle_count=sum(r['events'] > 0 for r in rows),
                repeated_source_vehicle_count=sum(r['repeated_source_events'] > 0 for r in rows),
                events_per_vehicle=events/n, repeated_source_event_fraction=sums['repeated_source_events']/events,
                nonprogress_fraction=sums['nonprogress_events']/events,
                nonshortest_fraction=sums['nonshortest_events']/events,
                excess_ff_seconds_per_event=sums['excess_ff_seconds']/events,
                excess_ff_seconds_per_vehicle=sums['excess_ff_seconds']/n,
                boundary_distance_lower_bound_m_per_vehicle=sums['boundary_distance_lower_bound_m']/n,
                boundary_lower_bound_m_per_exposure_second=sums['boundary_distance_lower_bound_m']/sums['boundary_exposure_seconds'],
                same_edge_no_position_progress_exposure_fraction=sums['same_edge_no_position_progress_seconds']/sums['boundary_exposure_seconds'],
                stationary_no_event_exposure_fraction=sums['stationary_no_event_seconds']/sums['boundary_exposure_seconds'],
                completed_travel_seconds=describe([r['travel_seconds'] for r in rows if r['arrived']]),
                observed_travel_seconds_per_vehicle=sums['observed_travel_seconds']/n,
                shortest_od_ff_seconds=describe([r['shortest_od_ff_seconds'] for r in rows]),
                shortest_od_distance_m=describe([r['shortest_od_distance_m'] for r in rows]),
                insertion_delay_seconds=describe([r['insertion_delay_seconds'] for r in rows if r['departed']]),
                demand_300second_bins=np.histogram([r['scheduled_departure'] for r in rows], bins=np.arange(0,1801,300))[0].tolist(),
                totals=sums)


def standardized(rows, key_fields, metrics, min_count=5):
    groups = {c: defaultdict(list) for c in ('early','late')}
    for r in rows:
        groups[r['cohort']][tuple(r[k] for k in key_fields)].append(r)
    keys = [k for k in groups['early'] if len(groups['early'][k]) >= min_count and len(groups['late'].get(k, [])) >= min_count]
    weights = np.array([len(groups['early'][k])+len(groups['late'][k]) for k in keys], dtype=float)
    result = dict(strata=len(keys), minimum_vehicles_per_cohort_stratum=min_count,
                  coverage={c:sum(len(groups[c][k]) for k in keys) for c in groups}, metrics={})
    for metric in metrics:
        result['metrics'][metric] = {c: float(np.average([np.mean([r[metric] for r in groups[c][k]]) for k in keys], weights=weights)) for c in groups}
    return result



def conditional_standardized(rows, minimum_events=20):
    """Compare common decision locations, without claiming identical traffic states."""
    groups = {c: defaultdict(Counter) for c in ('early', 'late')}
    for row in rows:
        groups[row['cohort']][(row['current_edge'], row['destination'])].update(
            {key: float(row[key]) for key in ('events', 'nonprogress_events', 'nonshortest_events', 'excess_ff_seconds')})
    keys = [key for key in groups['early'] if groups['early'][key]['events'] >= minimum_events
            and groups['late'][key]['events'] >= minimum_events]
    weights = [groups['early'][key]['events'] + groups['late'][key]['events'] for key in keys]
    return dict(strata=len(keys), minimum_events_per_cohort=minimum_events,
                event_coverage={c: sum(groups[c][key]['events'] for key in keys) for c in groups},
                metrics={metric: {c: float(np.average([groups[c][key][metric]/groups[c][key]['events'] for key in keys], weights=weights))
                                  for c in groups} for metric in ('nonprogress_events', 'nonshortest_events', 'excess_ff_seconds')},
                limitation='Common current-edge/destination strata only; congestion and vehicle history are not matched. Observed actions, not same-state policy checkpoint inference.')



def turn_summary(turns, conditions):
    """Show frequent locations and changed turn choices; no state-matching claim."""
    grouped = {c: defaultdict(Counter) for c in ('early', 'late')}
    quality = {c: defaultdict(Counter) for c in grouped}
    for row in turns:
        grouped[row['cohort']][(int(row['current_edge']), int(row['destination']))][int(row['next_edge'])] += int(row['count'])
    for row in conditions:
        quality[row['cohort']][(int(row['current_edge']), int(row['destination']))].update(
            {k: float(row[k]) for k in ('events','nonprogress_events','nonshortest_events','excess_ff_seconds')})
    common = set(grouped['early']) & set(grouped['late'])
    output = []
    for key in common:
        counts = {c: sum(grouped[c][key].values()) for c in grouped}
        if min(counts.values()) < 100:
            continue
        targets = sorted(set(grouped['early'][key]) | set(grouped['late'][key]))
        probs = {c: {str(t): grouped[c][key][t]/counts[c] for t in targets} for c in grouped}
        output.append(dict(current_edge=key[0], destination=key[1], events=counts,
                           total_events=sum(counts.values()), turn_probabilities=probs,
                           total_variation=.5*sum(abs(probs['late'][str(t)]-probs['early'][str(t)]) for t in targets),
                           quality={c: {metric: quality[c][key][metric]/counts[c] for metric in ('nonprogress_events','nonshortest_events','excess_ff_seconds')} for c in grouped}))
    return dict(common_strata_at_least100_events_per_cohort=len(output),
                top20_by_traffic=sorted(output, key=lambda r: r['total_events'], reverse=True)[:20],
                top20_by_turn_distribution_shift=sorted(output, key=lambda r: r['total_variation'], reverse=True)[:20],
                limitation='Location and destination conditioned empirical executed turns; congestion and full observations remain unmatched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/ingolstadt_online_meso_20261006')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/meso_decline_diagnosis_20261007/trajectories')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.run/'run_config.toml')
    topology = load_traffic_topology(validate_prepared_inputs(cfg))
    candidates = NextHopCandidates(topology.network, traffic_model='meso')
    selected = dict(early=[101,110,120,130,140,150], late=[351,370,390,410,430,450])
    episodes=[]; vehicles=[]; profiles=[]; conditions=[]; turns=[]
    for cohort, ids in selected.items():
        for episode in ids:
            result, rows, profile, cond, turn = audit(args.run/f'episodes/online/episode_{episode-1:06d}.npz', episode, cohort, candidates)
            episodes.append(result); vehicles.extend(rows); profiles.extend(profile); conditions.extend(cond); turns.extend(turn)
            print(json.dumps({k: result[k] for k in ('episode','cohort','population','events_per_vehicle','completion_rate','observed_travel_seconds_per_vehicle','nonprogress_fraction','excess_ff_seconds_per_event')}), flush=True)
    result = dict(sample_selection='Six deterministic spread-out episodes per cohort; numbers are one-based, archive indices one less. Not selected using outcome.',
                  edge_ids_by_index={str(i): e for i, e in enumerate(topology.network.edge_ids)},
                  cohorts={c: aggregate([r for r in vehicles if r['cohort']==c]) for c in selected}, episodes=episodes,
                  executed_turn_summary=turn_summary(turns, conditions),
                  standardized_decision_locations=conditional_standardized(conditions),
                  standardized_od_departure=standardized(vehicles, ['origin','destination','departure_bin'], ['observed_travel_seconds','events','excess_ff_seconds','boundary_distance_lower_bound_m','same_edge_no_position_progress_seconds', 'stationary_no_event_seconds']),
                  limits=['Retrospective observational sample, not causal policy comparison; demand and congestion can differ.',
                          'Repeated decision-source edges are a branch-observation proxy, not all physical revisits.',
                          'FF regret uses next-edge FF traversal plus downstream FF cost, MESO zero junction length. Nonprogress uses downstream cost(next)>=downstream cost(current).',
                          'Boundary displacement uses shortest network path between 30-second snapshot positions: lower bound, not actual route length or exact speed. Same-edge zero position change is a waiting proxy; unobserved loops could return to the same position.',
                          'Stricter stationary_no_event excludes any interval with a recorded vehicle decision in its closed [previous_time,current_time] bounds; this reduces hidden branch-loop false positives, but is still a proxy.',
                          'Boundary exposure excludes departure-to-first-boundary intervals; exact archived active exposure is reported separately.',
                          'Completed travel times omit censored vehicles; observed travel includes time until deadline for departed nonarrivals.',
                          'No historical early actor checkpoints available; no same-state old-versus-new actor inference or entropy-growth assumption.'])
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    write_csv(args.output/'vehicles.csv', vehicles); write_csv(args.output/'boundary_profiles.csv', profiles); write_csv(args.output/'conditional_decisions.csv', conditions)
    write_csv(args.output/'executed_turns.csv', turns)
    print(json.dumps(result['cohorts'], indent=2), flush=True)

if __name__ == '__main__':
    main()
