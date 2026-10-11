"""Independently audit a completed frozen-AC counterfactual experiment.

Reads the primary experiment artifacts and writes only verification.json. No
SUMO, PyTorch, simulator source, or experiment runner is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('travel_seconds', 'shortest_ff_seconds', 'actual_ff_seconds',
          'route_excess_seconds', 'residual_delay_seconds')
IDENTITY_FIELDS = ('time', 'sumo_sha256', 'python_sha256', 'decision_prefix_sha256')


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, context):
    if not condition:
        raise ValueError(context)


def ranked_subset(vehicle_ids, fraction):
    """Match the documented cohort randomization without importing the runner."""
    def ranking(vehicle_id):
        payload = json.dumps([90210, str(vehicle_id), 0], separators=(',', ':'))
        integer = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], 'big') >> 12
        return ((integer + .5) / 2**52, vehicle_id)
    ordered = sorted(set(vehicle_ids), key=ranking)
    return set(ordered[:round(fraction * len(ordered))])


class Auditor:
    def __init__(self, output, protocol):
        self.output = output
        self.protocol = protocol
        self.rng_captures = 0
        self.route_rows = 0
        self.branch_count = 0
        self.discovery_count = 0
        self.max_errors = {}
        self.seed_details = []

    def close(self, actual, expected, tolerance, name):
        require(math.isfinite(float(actual)) and math.isfinite(float(expected)),
                f'{name}: nonfinite value')
        error = abs(float(actual) - float(expected))
        self.max_errors[name] = max(self.max_errors.get(name, 0.), error)
        require(error <= tolerance, f'{name}: error {error} exceeds {tolerance}')

    def state(self, metrics, expected_time):
        label = metrics['label']
        state = metrics['state']
        require(state is not None, f'{label}: missing intervention state')
        path = ROOT / state['state_file']
        require(path.resolve().is_relative_to(self.output), f'{label}: state outside experiment')
        with gzip.open(path, 'rb') as stream:
            tree = ET.fromstring(stream.read())
        require(tree.find('rngState') is not None, f'{label}: missing SUMO RNG state')
        require(hashlib.sha256(ET.tostring(tree)).hexdigest() == state['sumo_sha256'],
                f'{label}: SUMO state hash differs')
        audit = read_json(path.with_suffix('.audit.json'))
        encoded = json.dumps(audit, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        require(hashlib.sha256(encoded).hexdigest() == state['python_sha256'],
                f'{label}: Python state hash differs')
        require(audit['decision_prefix_sha256'] == state['decision_prefix_sha256'],
                f'{label}: decision prefix hash differs')
        self.close(state['time'], expected_time, 1e-9, 'state_time')
        self.close(float(tree.attrib['time']), expected_time, 1e-9, 'xml_state_time')
        self.close(audit['time_ms'] / 1000., expected_time, 1e-9, 'python_state_time')
        self.rng_captures += 1
        return {key: state[key] for key in IDENTITY_FIELDS}, audit

    def routes(self, directory, metrics, expected_ids=None, audit=None):
        label = metrics['label']
        count = metrics['vehicle_count']
        require(count > 0, f'{label}: empty population')
        require(metrics['unfinished'] == 0 and metrics['never_departed'] == 0
                and metrics['trip_time_complete'] and metrics['completion_rate'] == 1.,
                f'{label}: incomplete trips invalidate complete-trip comparison')
        require(metrics['simulation_end_seconds'] <= self.protocol['deadline_seconds'],
                f'{label}: simulation beyond declared horizon')
        rows = []
        with gzip.open(directory / metrics['routes'], 'rt') as stream:
            for line in stream:
                rows.append(json.loads(line))
        ids = {row['vehicle_id'] for row in rows}
        require(len(rows) == len(ids) == count, f'{label}: route count or unique IDs differs')
        if expected_ids is not None:
            require(ids == expected_ids, f'{label}: realized vehicle IDs differ')
        if audit is not None:
            require(ids == set(audit['vehicles']), f'{label}: routes differ from queued/active demand')
            require(set(audit['travel_times']).issubset(ids), f'{label}: prefix arrivals missing')
        costs = {}
        for row in rows:
            travel, shortest, actual, detour, delay = (row[field] for field in FIELDS)
            require(all(math.isfinite(float(value)) for value in (travel, shortest, actual, detour, delay)),
                    f'{label}: nonfinite decomposition')
            require(travel >= 0 and shortest >= 0 and actual >= 0 and detour >= -1e-8,
                    f'{label}: invalid time or route excess')
            self.close(shortest + detour + delay, travel, 1e-7, 'trip_decomposition')
            self.close(actual - shortest, detour, 1e-7, 'route_excess')
            self.close(travel - actual, delay, 1e-7, 'residual_delay')
            self.close(row['arrival_time'] - row['actual_departure_time'], travel, 1e-8,
                       'departure_arrival_duration')
            require(row['external_path'][0] == row['origin']
                    and row['external_path'][-1] == row['destination'],
                    f'{label}: physical route endpoints differ')
            require(len(row['external_path']) == row['external_edge_count'],
                    f'{label}: physical edge count differs')
            if audit is not None and row['vehicle_id'] in audit['travel_times']:
                self.close(travel, audit['travel_times'][row['vehicle_id']], 1e-8,
                           'prefix_completed_duration')
            costs[row['vehicle_id']] = (row['origin'], row['destination'], shortest)
        means = {field: math.fsum(row[field] for row in rows) / count for field in FIELDS}
        for field in FIELDS:
            self.close(means[field], metrics['decomposition'][field], 1e-8, f'mean_{field}')
        self.close(means['travel_seconds'], metrics['mean_travel_seconds'], 1e-8, 'mean_trip_time')
        self.close(metrics['global_reward_sum'], -means['travel_seconds'] / 60., 1e-8,
                   'reward_scaling')
        self.close(metrics['mean_exposure_seconds'], means['travel_seconds'], 1e-8,
                   'exposure_trip_equivalence')
        self.route_rows += len(rows)
        return ids, costs

    def seed(self, seed, repeats, joint_repeats):
        directory = self.output / f'seed_{seed}'
        plan = read_json(directory / 'discovery_plan.json')
        discovery = read_json(directory / 'discovery.json')
        require(plan['demand_seed'] == seed and plan['bottleneck'] == self.protocol['bottleneck'],
                f'{seed}: discovery plan differs from protocol')
        require(plan['discovery'] == discovery, f'{seed}: discovery copy differs')
        require(discovery['demand_sha256'] == plan['demand_sha256'], f'{seed}: pilot demand differs')
        ids, costs = self.routes(directory, discovery)
        self.discovery_count += 1
        cohort = plan['target_ids']
        require(len(set(cohort)) == len(cohort) and set(cohort).issubset(ids),
                f'{seed}: invalid pilot eligible cohort')
        target = plan['target']
        require(target['vehicle_id'] in ids and target['a1'] != target['a2']
                and target['a1'] in target['legal'] and target['a2'] in target['legal'],
                f'{seed}: illegal target actions')
        require(self.protocol['joint_start'] <= target['time'] <= 1200.,
                f'{seed}: target outside discovery selection interval')
        labels = [f'single_{repeat:02d}_{arm}' for repeat in range(repeats) for arm in ('a1', 'a2')]
        labels += [f'joint_{repeat:02d}_{round(100 * fraction):02d}'
                   for repeat in range(joint_repeats) for fraction in self.protocol['fractions']]
        actual_labels = {path.stem for pattern in ('single_*_a?.json', 'joint_??_??.json')
                         for path in directory.glob(pattern)}
        require(actual_labels == set(labels), f'{seed}: expected branch files differ')
        branches = {}
        states = {}
        for label in labels:
            metrics = read_json(directory / f'{label}.json')
            require(metrics['label'] == label and metrics['seed'] == seed
                    and metrics['demand_seed'] == seed, f'{seed}/{label}: identity differs')
            require(metrics['demand_sha256'] == plan['demand_sha256']
                    and metrics['vehicle_count'] == discovery['vehicle_count'],
                    f'{seed}/{label}: demand or population differs')
            repeat = int(label.split('_')[1])
            require(metrics['future_seed'] == self.protocol['future_policy_seeds'][repeat],
                    f'{seed}/{label}: future policy seed differs')
            group = label.split('_')[0]
            identity, audit = self.state(metrics, target['time'] if group == 'single'
                                         else self.protocol['joint_start'])
            if group in states:
                require(states[group] == identity, f'{seed}/{label}: intervention prefix differs')
            states[group] = identity
            _, branch_costs = self.routes(directory, metrics, ids, audit)
            require(branch_costs == costs, f'{seed}/{label}: OD or shortest free-flow costs differ')
            interventions = metrics['interventions']
            require(len(interventions) == metrics['intervention_count'],
                    f'{seed}/{label}: intervention count differs')
            if group == 'single':
                require(len(interventions) == 1, f'{seed}/{label}: requires exactly one override')
                vehicle = audit['vehicles'][target['vehicle_id']]
                require(target['vehicle_id'] in audit['pending']
                        and vehicle['current_edge'] == target['current_edge']
                        and vehicle['destination_edge'] == target['destination_edge'],
                        f'{seed}/{label}: target not pending at its selected routing state')
                event = interventions[0]
                action = target[label.rsplit('_', 1)[1]]
                require(event['vehicle_id'] == target['vehicle_id'] and event['time'] == target['time']
                        and event['before'] == target['a1'] and event['after'] == action,
                        f'{seed}/{label}: target override differs')
            else:
                fraction = int(label.rsplit('_', 1)[1]) / 100.
                require(metrics['fraction'] == fraction, f'{seed}/{label}: dose differs')
                selected = ranked_subset(cohort, fraction)
                require(metrics['assigned_target_count'] == len(selected)
                        and len(interventions) <= len(selected), f'{seed}/{label}: allocation differs')
                require(len(interventions) <= metrics['eligible_vehicle_count'] <= len(ids),
                        f'{seed}/{label}: live eligible population differs')
                intervention_ids = [event['vehicle_id'] for event in interventions]
                require(len(set(intervention_ids)) == len(intervention_ids)
                        and set(intervention_ids).issubset(selected), f'{seed}/{label}: cohort or uniqueness differs')
                for event in interventions:
                    require(event['before'] == plan['bottleneck'] and event['after'] != plan['bottleneck']
                            and self.protocol['joint_start'] <= event['time'] < self.protocol['joint_end'],
                            f'{seed}/{label}: intervention action or timing differs')
                if 'intervention_uptake_fraction' in metrics:
                    self.close(metrics['intervention_uptake_fraction'],
                               len(interventions) / len(selected) if selected else 0., 1e-12,
                               'intervention_uptake')
            branches[label] = metrics
            self.branch_count += 1
        single = read_json(directory / 'single_pairs.json')
        joint = read_json(directory / 'joint_pairs.json')
        require(len(single) == repeats and len(joint) == joint_repeats * 3,
                f'{seed}: paired row counts differ')
        seen_single, seen_joint = set(), set()
        for row in single:
            repeat = row['repeat']
            require(repeat in range(repeats) and repeat not in seen_single
                    and row['demand_seed'] == seed
                    and row['future_seed'] == self.protocol['future_policy_seeds'][repeat],
                    f'{seed}: single pair identity differs')
            seen_single.add(repeat)
            a, b = (branches[f'single_{repeat:02d}_{arm}'] for arm in ('a1', 'a2'))
            require(row['a1'] == a and row['a2'] == b, f'{seed}: single pair copy differs')
            self.close(row['difference'], a['mean_travel_seconds'] - b['mean_travel_seconds'],
                       1e-8, 'single_paired_difference')
            self.close(row['reward_difference'], b['global_reward_sum'] - a['global_reward_sum'],
                       1e-8, 'single_paired_reward_difference')
        for row in joint:
            repeat, fraction = row['repeat'], row['fraction']
            key = (repeat, fraction)
            require(repeat in range(joint_repeats) and fraction in self.protocol['fractions'][1:]
                    and key not in seen_joint and row['demand_seed'] == seed,
                    f'{seed}: joint pair identity differs')
            seen_joint.add(key)
            a = branches[f'joint_{repeat:02d}_00']
            b = branches[f'joint_{repeat:02d}_{round(100 * fraction):02d}']
            require(row['baseline'] == a and row['treatment'] == b, f'{seed}: joint pair copy differs')
            self.close(row['difference'], a['mean_travel_seconds'] - b['mean_travel_seconds'],
                       1e-8, 'joint_paired_difference')
            self.close(row['reward_difference'], b['global_reward_sum'] - a['global_reward_sum'],
                       1e-8, 'joint_paired_reward_difference')
        subsets = [ranked_subset(cohort, fraction) for fraction in self.protocol['fractions']]
        require(all(a.issubset(b) for a, b in zip(subsets, subsets[1:])), f'{seed}: doses not nested')
        for label in ('single_00_a1', 'joint_00_00'):
            for field in ('event_count', 'exposure_seconds', 'global_reward_sum', 'mean_travel_seconds'):
                require(branches[label][field] == discovery[field], f'{seed}/{label}: pilot replay differs')
        self.seed_details.append(dict(demand_seed=seed, population=len(ids),
            pilot_eligible_vehicle_count=len(cohort), branch_runs=len(labels),
            single_pairs=len(single), joint_pairs=len(joint), single_state=states['single'],
            joint_state=states['joint'], complete_trips=True, prefix_states_equal=True))


def verify(output):
    require((output / 'completed.json').is_file(), 'completed.json is required before verification')
    protocol = read_json(output / 'protocol.json')
    completed = read_json(output / 'completed.json')
    seeds = protocol['demand_seeds']
    require(seeds and len(set(seeds)) == len(seeds), 'demand seeds must be nonempty and unique')
    repeats = len(protocol['future_policy_seeds'])
    require(repeats > 0 and len(set(protocol['future_policy_seeds'])) == repeats,
            'future policy seeds must be nonempty and unique')
    require(protocol['fractions'] == [0., .05, .10, .20], 'unexpected intervention doses')
    require(protocol['method'] == 'online_ac' and protocol['checkpoint_episode'] == 5000
            and protocol['training_updates'] == completed['training_updates'] == 0
            and completed['actor_unchanged'], 'frozen AC identity differs')
    require(sha256(protocol['actor']) == protocol['actor_sha256'], 'actor checkpoint changed')
    require(sha256(Path(protocol['source']) / 'urban_dreamer/traffic/event_session.py')
            == protocol['source_session_sha256'], 'frozen SUMO session source changed')
    joint_repeats = min(repeats, 3)
    results = completed['results']
    require(len(results) == len(seeds) and {row['demand_seed'] for row in results} == set(seeds),
            'completion manifest demand count differs')
    require(all(row['prefix_states_verified'] and row['single_pairs'] == repeats
                and row['joint_pairs'] == 3 * joint_repeats for row in results),
            'completion manifest pair counts or state verification differ')
    auditor = Auditor(output, protocol)
    for seed in seeds:
        auditor.seed(seed, repeats, joint_repeats)
    return dict(status='pass', passed=True, verified_at=datetime.now(timezone.utc).isoformat(),
        independent_verifier='tools/verify_ac_counterfactual.py',
        demand_count=len(seeds), run_counts=dict(discovery=auditor.discovery_count,
            branches=auditor.branch_count, total=auditor.discovery_count + auditor.branch_count),
        pair_counts=dict(single=len(seeds) * repeats, joint=len(seeds) * joint_repeats * 3),
        horizon_seconds=protocol['deadline_seconds'], all_full_trip_completion=True,
        sumo_rng_state_captures=auditor.rng_captures, route_rows_audited=auditor.route_rows,
        actor_hash_unchanged=True, frozen_source_session_hash_unchanged=True,
        all_prefix_hashes_recomputed=True, max_absolute_errors=auditor.max_errors,
        details=auditor.seed_details,
        interpretation_notes=[
            'Single difference is T(a1)-T(a2); positive values favor a2.',
            'Joint difference is T(baseline)-T(intervention); positive values favor intervention.',
            'Reward differences use intervention minus comparator, with reward=-mean_trip_seconds/60.',
            'Joint allocation uses nested fractions of the fixed pilot-eligible cohort; live uptake may vary.',
            'RNG state presence and exact saved prefix identity are verified; continuation randomness is policy sampling.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/ac_counterfactual_20261010')
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        result = verify(output)
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as error:
        print(f'Counterfactual verification failed: {error}', file=sys.stderr)
        return 1
    destination = output / 'verification.json'
    temporary = destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    print(json.dumps({key: result[key] for key in ('passed', 'run_counts', 'pair_counts',
        'sumo_rng_state_captures', 'all_full_trip_completion')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
