#!/usr/bin/env python3
"""Read-only frozen-archive route-signal audit; does not load/train a policy."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
sys.path.insert(0, str(BATCH / 'source'))

from urban_dreamer.learning.global_reward import causal_context
from urban_dreamer.learning.observations import prepare_actor_window
from urban_dreamer.traffic.route_candidates import RouteCandidates, build_route_observations
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def statistics(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {'count': 0}
    assert np.isfinite(array).all()
    return dict(count=int(array.size), mean=float(array.mean()),
                p10=float(np.quantile(array, .1)), median=float(np.median(array)),
                p90=float(np.quantile(array, .9)), p95=float(np.quantile(array, .95)),
                maximum=float(array.max()))


def histogram(values):
    counts = np.bincount(values, minlength=5)
    return {str(i): int(counts[i]) for i in range(1, 5)}


def analyze(city, episode):
    started = time.monotonic()
    directory = BATCH / city / 'online_ac'
    path = directory / 'episodes/train' / f'episode_{episode:06d}.npz'
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        summary = json.loads(archive['summary_json'].tobytes())
        states = json.loads(archive['states_json'].tobytes())
        decisions = json.loads(archive['policy_decisions_json'].tobytes())
        flow = archive['flow']
    metadata = summary['metadata']
    assert metadata['reward_objective'] == 'observed_mean'
    assert metadata['routing_mode'] == 'current_to_dest'
    config = load_config(BATCH / 'source/configs' / f'{city}.toml')
    topology = load_traffic_topology(validate_prepared_inputs(config))
    candidates = RouteCandidates(topology.network, traffic_model=config.simulation.traffic_model)
    network = topology.network
    deadline = float(summary['deadline_seconds'])
    by_window = {}
    for decision in decisions:
        by_window.setdefault(int(decision['window_index']), []).append(decision)
    feature_std, feature_range = [], []
    unique_routes, unique_features, unique_first_hops = [], [], []
    all_delays, shared_first_delays, earliest_delays = [], [], []
    snapshot_delays, earliest_snapshot_delays = [], []
    common_edge_counts = []
    same_first_pairs = 0
    pair_count = 0
    prefix_cache = {}
    state_matches = recorded_routes_match = 0
    context_errors = []
    for window, rows in sorted(by_window.items()):
        state = states[window]
        lookup = {vehicle['vehicle_id']: vehicle for vehicle in state['vehicles']}
        vehicles = [lookup[row['vehicle_id']] for row in rows]
        for row, vehicle in zip(rows, vehicles, strict=True):
            assert row['current_edge'] == vehicle['current_edge']
            assert row['destination_edge'] == vehicle['destination_edge']
            assert row['decision_time'] == state['time']
            assert vehicle['departed'] and not vehicle['arrived']
            state_matches += 1
        context = causal_context(float(state['time']), deadline,
            int(state['active_vehicle_count']), int(state['started_vehicle_count']),
            float(state['cumulative_vehicle_seconds']))
        observed = prepare_actor_window(flow[window][:, [2, 1]], network.edge_count)
        observations = build_route_observations(policy_edge_features=observed,
            vehicle_states=vehicles, candidates=candidates,
            episode_progress=context[0], active_vehicle_fraction=context[1])
        assert observations.mask.all()
        feature_std.extend(observations.features.std(axis=1, ddof=0))
        feature_range.extend(np.ptp(observations.features, axis=1))
        context_errors.append(float(np.abs(observations.features[:, :, 9] - context[1]).max()))
        limits = np.asarray(network.speed_limits_mps, dtype=np.float64)
        speed = observed.edge_features[:, 1].astype(np.float64) * limits
        speed = np.where((speed == 0) & (observed.edge_features[:, 0] == 0), limits, speed)
        speed = np.maximum(speed, config.method.sp_prior_dynamic_speed_floor_mps)
        speed = np.minimum(speed, config.simulation.vehicle_max_speed_mps)
        snapshot_edge_seconds = network.lengths_m / speed
        for row, vehicle, paths, features in zip(rows, vehicles,
                observations.routes, observations.features, strict=True):
            assert tuple(row['route']) in paths
            recorded_routes_match += 1
            distinct = tuple(dict.fromkeys(paths))
            unique_routes.append(len(distinct))
            unique_features.append(len(np.unique(features, axis=0)))
            unique_first_hops.append(len({path[1] if len(path) > 1 else path[0] for path in distinct}))
            current = int(vehicle['current_edge'])
            remaining = max(float(vehicle['remaining_distance']), 0.)
            fraction = min(remaining / max(float(network.lengths_m[current]), .1), 1.)
            current_seconds = fraction * float(network.free_flow_travel_seconds[current])
            current_snapshot_seconds = fraction * float(snapshot_edge_seconds[current])
            decision_delays, decision_snapshot_delays = [], []
            for first, second in itertools.combinations(distinct, 2):
                key = (first, second)
                if key not in prefix_cache:
                    common = 0
                    for left, right in zip(first, second):
                        if left != right:
                            break
                        common += 1
                    assert 1 <= common < min(len(first), len(second))
                    shared = first[1:common]
                    junction = sum(candidates.junction(source, target)[0]
                        / candidates.junction(source, target)[1]
                        for source, target in zip(first[:common], first[1:common]))
                    tail_seconds = float(network.free_flow_travel_seconds[list(shared)].sum()) + junction
                    prefix_cache[key] = (common, shared, junction, tail_seconds)
                common, shared, junction, tail_seconds = prefix_cache[key]
                delay = current_seconds + tail_seconds
                snapshot_delay = current_snapshot_seconds + float(snapshot_edge_seconds[list(shared)].sum()) + junction
                all_delays.append(delay)
                snapshot_delays.append(snapshot_delay)
                common_edge_counts.append(common)
                decision_delays.append(delay)
                decision_snapshot_delays.append(snapshot_delay)
                pair_count += 1
                if common > 1:
                    same_first_pairs += 1
                    shared_first_delays.append(delay)
            if decision_delays:
                earliest_delays.append(min(decision_delays))
                earliest_snapshot_delays.append(min(decision_snapshot_delays))
        if window % 30 == 0:
            print(f'{city}: processed window {window}/{len(states)-1}', flush=True)
    std = np.asarray(feature_std)
    ranges = np.asarray(feature_range)
    delays = np.asarray(all_delays)
    earliest = np.asarray(earliest_delays)
    snapshot = np.asarray(snapshot_delays)
    earliest_snapshot = np.asarray(earliest_snapshot_delays)
    training = next(json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()
                    if json.loads(line).get('phase') == 'train' and json.loads(line).get('episode') == episode)
    result = dict(city=city, training_episode=episode, archive=str(path.relative_to(ROOT)),
        archive_sha256=file_hash, archive_seed=metadata['seed'], archive_population=len(states[0]['vehicles']),
        policy_decisions=len(decisions), audited_decisions=len(unique_routes), windows_with_decisions=len(by_window),
        validation=dict(exact_state_endpoint_timestamp_matches=state_matches,
                        recorded_routes_in_frozen_candidate_set=recorded_routes_match,
                        maximum_active_started_context_feature_error=max(context_errors)),
        unique_route_count_histogram=histogram(unique_routes),
        unique_feature_vector_count_histogram=histogram(unique_features),
        unique_first_hop_count_histogram=histogram(unique_first_hops),
        padded_route_decision_fraction=float(np.mean(np.asarray(unique_routes) < 4)),
        single_distinct_route_decision_fraction=float(np.mean(np.asarray(unique_routes) == 1)),
        single_first_hop_decision_fraction=float(np.mean(np.asarray(unique_first_hops) == 1)),
        features=dict(within_decision_std_mean=std.mean(axis=0).tolist(),
                      within_decision_std_p90=np.quantile(std, .9, axis=0).tolist(),
                      within_decision_range_mean=ranges.mean(axis=0).tolist(),
                      nonzero_contrast_fraction=(ranges > 0).mean(axis=0).tolist()),
        distinct_route_pairs=pair_count, distinct_route_pairs_shared_first_hop=same_first_pairs,
        pair_shared_first_hop_fraction=same_first_pairs / max(pair_count, 1),
        common_prefix_edge_count=statistics(common_edge_counts),
        free_flow_first_divergence_seconds=statistics(delays),
        free_flow_first_divergence_seconds_shared_first_hop=statistics(shared_first_delays),
        free_flow_earliest_divergence_per_decision_seconds=statistics(earliest),
        snapshot_first_divergence_seconds=statistics(snapshot),
        snapshot_earliest_divergence_per_decision_seconds=statistics(earliest_snapshot),
        pair_divergence_after_30s_fraction=float(np.mean(delays > 30)),
        pair_divergence_after_60s_fraction=float(np.mean(delays > 60)),
        pair_divergence_after_300s_fraction=float(np.mean(delays > 300)),
        all_candidates_same_prefix_beyond_30s_decision_fraction=float(np.mean(earliest > 30)),
        snapshot_pair_divergence_after_30s_fraction=float(np.mean(snapshot > 30)),
        snapshot_all_candidates_same_prefix_beyond_30s_decision_fraction=float(np.mean(earliest_snapshot > 30)),
        lambda095_weight_at_free_flow_divergence=statistics(.95 ** (delays / 30)),
        lambda095_weight_at_snapshot_divergence=statistics(.95 ** (snapshot / 30)),
        training_metrics={key: training.get(key) for key in ['completion_rate', 'global_reward_sum',
            'event_count', 'policy_entropy_mean', 'max_action_probability_mean']},
        audit_seconds=time.monotonic() - started)
    assert result['audited_decisions'] == len(decisions)
    print(json.dumps({key: result[key] for key in ['city', 'audited_decisions',
        'single_first_hop_decision_fraction', 'pair_divergence_after_30s_fraction',
        'all_candidates_same_prefix_beyond_30s_decision_fraction', 'audit_seconds']}), flush=True)
    return result


def main():
    # Fixed after the initial completed-metric/archive boundary inspection.
    snapshot = [('coulommiers', 153), ('ingolstadt_custom', 140), ('montereau_fault_yonne', 140)]
    report = dict(schema_version=1, snapshot_utc='2026-10-14',
        method='all archived policy decisions; exact frozen builder and recorded causal context',
        feature_names=['route_mean_density', 'route_mean_speed_ratio', 'global_mean_density',
            'global_mean_speed_ratio', 'log_route_free_flow_seconds', 'log_route_distance',
            'log_current_remaining_distance', 'next_edge_is_destination', 'episode_progress',
            'active_started_fraction'],
        timing_definition='Time to end of last common-prefix edge: current remaining-edge fraction plus shared subsequent edges and shared junctions. Diverging junction is excluded.',
        timing_limits=['Free-flow and current frozen-speed timings are estimates, not actual simulated causal delays.',
            'Suffix routes may be superseded at a later 30-second decision; lane selection or congestion effects can precede edge divergence.',
            'Lambda weights are .95**(seconds/30) approximations; this is not measured actor-gradient attenuation.',
            'Coul is the latest complete episode at the snapshot boundary; other cities are episode140, not all episode matched.',
            'Each city is represented by one demand realization; these contrasts are descriptive, not a causal intervention.'],
        findings={
            'measured': [
                'Coul has less duplicate route padding than the two comparison archives.',
                'Every distinct route has a distinct ten-feature vector in all three archives.',
                'Coul within-decision feature contrast is comparable to or larger than Ingolstadt for the four varying features.',
                'Coul has more shared first hops and later snapshot-estimated route-prefix divergence.',
                'All three recorded training distributions are nearly uniform at these episodes; Montereau nevertheless has complete stochastic arrivals.'],
            'interpretation': [
                'Padding and feature collisions do not explain Coul-specific poor learning in these archives.',
                'Shared prefixes and congestion can contribute to replanning and delayed credit; first-divergence lambda estimates alone show modest mean attenuation.',
                'This audit does not establish the causal root of weak policy gradients, useful reward delay, or deterministic policy failure.']},
        rows=[])
    for city, episode in snapshot:
        report['rows'].append(analyze(city, episode))
        gc.collect()
    path = BATCH / 'coul_learning_route_signal_20261014.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {path}', flush=True)


if __name__ == '__main__':
    main()
