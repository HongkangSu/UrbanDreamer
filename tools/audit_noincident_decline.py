#!/usr/bin/env python3
"""Decompose real no-incident episode costs and audit a bounded trajectory sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_meso_trajectories import aggregate, audit, standardized, write_csv
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology


def decompose(rows):
    """Exact exposure identity, with means explicitly conditional on arrival."""
    n = len(rows)
    completed = [r for r in rows if r['arrived']]
    unfinished = [r for r in rows if not r['arrived']]
    complete_seconds = sum(r['observed_travel_seconds'] for r in completed)
    unfinished_seconds = sum(r['observed_travel_seconds'] for r in unfinished)
    return dict(
        population=n, completed=len(completed), unfinished=len(unfinished),
        completion_rate=len(completed)/n,
        completed_mean_minutes=complete_seconds/len(completed)/60 if completed else None,
        unfinished_mean_observed_minutes=unfinished_seconds/len(unfinished)/60 if unfinished else None,
        completed_cost_minutes_per_population=complete_seconds/n/60,
        unfinished_cost_minutes_per_population=unfinished_seconds/n/60,
        total_observed_minutes_per_population=(complete_seconds+unfinished_seconds)/n/60,
        completed_vehicle_seconds=complete_seconds,
        unfinished_vehicle_seconds=unfinished_seconds,
    )


def stage_metrics(records):
    values = [r['metrics'] for r in records]
    n = sum(r['online_vehicle_count'] for r in values)
    nonarrivals = sum(r['online_nonarrival_count'] for r in values)
    return dict(
        episodes=len(values), population=int(n), nonarrivals=int(nonarrivals),
        completion_rate=1-nonarrivals/n,
        mean_episode_reward=float(np.mean([r['online_global_reward_sum'] for r in values])),
        weighted_observed_minutes=sum(r['online_vehicle_minutes'] for r in values)/n,
        mean_entropy=float(np.mean([r['policy_entropy_mean'] for r in values])),
        mean_events_per_vehicle=sum(r['online_events'] for r in values)/n,
    )


def cost_change(first, second):
    """Symmetric descriptive decomposition; not a causal mediation estimate."""
    p0, p1 = first['completion_rate'], second['completion_rate']
    c0, c1 = first['completed_mean_minutes'], second['completed_mean_minutes']
    u0, u1 = first['unfinished_mean_observed_minutes'], second['unfinished_mean_observed_minutes']
    completion = (p1-p0)*((c0+c1)/2-(u0+u1)/2)
    within_means = (p0+p1)/2*(c1-c0)+(1-(p0+p1)/2)*(u1-u0)
    total = second['total_observed_minutes_per_population']-first['total_observed_minutes_per_population']
    assert abs(total-completion-within_means) < 1e-9
    return dict(total_change_minutes=total,
                completion_composition_change_minutes=completion,
                conditional_mean_change_minutes=within_means,
                completion_composition_fraction=completion/total,
                limit='Algebraic decomposition of unpaired samples; groups change membership, so within-means changes are not individual treatment effects.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/ingolstadt_online_meso_noincident_noprior_20261007')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/noincident_decline_diagnosis_20261008/trajectory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    progress = json.loads((args.run/'pilot_progress.json').read_text())
    cfg = load_config(args.run/'run_config.toml')
    topology = load_traffic_topology(validate_prepared_inputs(cfg))
    candidates = NextHopCandidates(topology.network, traffic_model='meso')
    last = len(progress['online'])
    # Fixed selection independent of reward. At most twelve raw archives.
    selected = {'1-20': [10, 20], '51-100': [51, 75, 100],
                '101-150': [101, 125, 150], '401-450': [401, 425, 450],
                'last20': [last-3]}
    selected = {k: [ep for ep in v if ep <= last] for k, v in selected.items()}
    episodes = []
    vehicles = []
    for cohort, ids in selected.items():
        for episode in ids:
            path = args.run/f'episodes/online/episode_{episode-1:06d}.npz'
            result, rows, _, _, _ = audit(path, episode, cohort, candidates)
            assert result['metrics']['online_sp_prior_weight'] == 0
            result['exposure_decomposition'] = decompose(rows)
            episodes.append(result)
            vehicles.extend(rows)
            print(json.dumps(dict(episode=episode, cohort=cohort,
                                  decomposition=result['exposure_decomposition'])), flush=True)
    ranges = {'1-20': (1, 20), '51-100': (51, 100), '101-150': (101, 150),
              '401-450': (401, 450), 'last20': (max(1, last-19), last)}
    cohorts = {}
    for cohort in selected:
        rows = [r for r in vehicles if r['cohort'] == cohort]
        if rows:
            cohorts[cohort] = dict(sample_episodes=selected[cohort],
                                  decomposition=decompose(rows), routing=aggregate(rows))
    output = dict(run=str(args.run), completed_episode_count_at_snapshot=last,
                  selected=selected, episodes=episodes, sampled_cohorts=cohorts,
                  full_stage_metrics={k: stage_metrics(progress['online'][lo-1:hi])
                                      for k, (lo, hi) in ranges.items() if hi <= last},
                  limits=[
                      'Retrospective unpaired demands and changing congestion; no causal attribution.',
                      'At most twelve raw archives; stage statistics use all saved episodes in the stated range.',
                      'Exact cost identity: completion fraction times conditional completed mean plus unfinished fraction times conditional censored mean.',
                      'Conditional completed travel time is selected by arrival and cannot identify within-vehicle improvements.',
                      'Repeated branch sources and stationary boundary positions are proxies; physical distance is a lower bound.',
                  ])
    if all(k in cohorts for k in ('1-20', '51-100', '401-450')):
        output['early_descriptive_cost_change'] = cost_change(
            cohorts['1-20']['decomposition'], cohorts['51-100']['decomposition'])
        output['late_descriptive_cost_change'] = cost_change(
            cohorts['51-100']['decomposition'], cohorts['401-450']['decomposition'])
        matched = [dict(r, cohort='early' if r['cohort'] == '51-100' else 'late')
                   for r in vehicles if r['cohort'] in ('51-100', '401-450')]
        output['matched_od_departure'] = standardized(
            matched, ['origin', 'destination', 'departure_bin'],
            ['observed_travel_seconds', 'events', 'excess_ff_seconds',
             'boundary_distance_lower_bound_m', 'stationary_no_event_seconds'], min_count=3)
    (args.output/'results.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    write_csv(args.output/'vehicles.csv', vehicles)
    print(json.dumps(output['sampled_cohorts'], indent=2), flush=True)


if __name__ == '__main__':
    main()
