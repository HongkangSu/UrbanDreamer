"""Free 30/60/120 s development rollouts with causal frozen route commitments.

The factual real episode keeps dynamic routing and future demand. A shadow uses
only vehicles active at its seed, accepted route suffixes at that seed, and fixed
seed-boundary road speeds. Factual route deviations are reported, not hidden.
This is executor evidence, not a closed-loop counterfactual policy evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.causal_routing_policies import RealTimeShortestRouter
from tools.component_experiment_data import atomic_json, sha256
from tools.diagnose_component_executor import RouteMismatch
from tools.routing_cost_observer import PeriodicCostObserver
from urban_dreamer.learning import imagination as physical_engine
from urban_dreamer.learning.city_imagination import ChronologicalJunctionArbiter, advance_city_imagined_window
from urban_dreamer.traffic.demand import PoissonDemand, demand_identity
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.movement_controls import CityControlTable
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity, sumo_model_args
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
from urban_dreamer.workflow.config import load_config


class FrozenRouteController:
    """Follow only the accepted seed route; audit every physical entry by cursor."""

    def __init__(self, routes):
        self.routes = {key: tuple(route) for key, route in routes.items()}
        self.cursor = {key: 0 for key in routes}
        self.entries = []
        self.decisions = []

    def _current(self, state):
        key = state['vehicle_id']
        if key not in self.routes:
            raise RouteMismatch(f'unknown cohort vehicle {key}')
        cursor, route = self.cursor[key], self.routes[key]
        if cursor >= len(route) or route[cursor] != int(state['current_edge']):
            raise RouteMismatch(f'{key}: physical road disagrees with accepted seed-route cursor')
        return key, cursor, route

    def decide(self, pending, flow, now):
        result = []
        for state in pending:
            key, cursor, route = self._current(state)
            if cursor + 1 >= len(route):
                raise RouteMismatch(f'{key}: decision requested beyond recorded route')
            item = NextHopDecision(key, int(state['current_edge']), int(route[cursor+1]), now)
            result.append(item)
            self.decisions.append(asdict(item))
        return result

    def entry(self, state, target):
        key, cursor, route = self._current(state)
        if cursor + 1 >= len(route) or int(target) != route[cursor+1]:
            raise RouteMismatch(f'{key}: unrecorded physical entry {target} at cursor {cursor+1}')
        self.cursor[key] += 1
        self.entries.append(dict(vehicle_id=key, route_index=cursor+1, target_edge=int(target)))


def run_free_rollout(before, *, routes, implementation, candidates, controls, speeds, flow,
                     start_time, population, horizons, max_speed, seed):
    horizons = tuple(horizons)
    if not horizons or tuple(sorted(set(horizons))) != horizons or horizons[0] <= 0:
        raise ValueError('horizons must be sorted distinct positive seconds')
    states = deepcopy(before)
    if set(routes) != {s['vehicle_id'] for s in states}:
        raise RouteMismatch('recorded routes must cover exactly the seed cohort')
    for state in states:
        route = routes[state['vehicle_id']]
        if not route or route[0] != state['current_edge'] or route[-1] != state['destination_edge']:
            raise RouteMismatch('accepted route endpoints disagree with seed')
    controller = FrozenRouteController(routes)
    arbiter = (ChronologicalJunctionArbiter(controls, seed=seed, uncertain_order=False)
               if implementation == 'city' else None)
    original = physical_engine._record_physical_entry
    results, elapsed, cumulative, error = [], 0, 0., None

    def entry(state, edge, choices):
        controller.entry(state, edge)
        original(state, edge, choices)

    try:
        physical_engine._record_physical_entry = entry
        for horizon in horizons:
            if error is None and states:
                try:
                    engine = advance_city_imagined_window if implementation == 'city' else physical_engine.advance_imagined_window
                    extra = dict(junction_arbiter=arbiter) if arbiter else {}
                    result = engine(states, candidates=candidates, edge_speeds_mps=np.asarray(speeds),
                        frozen_flow=flow, start_time=start_time+elapsed, total_vehicle_count=population,
                        decide=controller.decide, interval_seconds=float(horizon-elapsed),
                        observation_step_seconds=1., session_protocol='event', traffic_model='meso',
                        vehicle_max_speed_mps=max_speed, stop_when_empty=False, **extra)
                    cumulative += result.active_vehicle_seconds
                except Exception as exc:
                    error = dict(type=type(exc).__name__, message=str(exc), first_failed_horizon=horizon)
            elapsed = horizon
            results.append(dict(horizon=horizon, status='error' if error else 'ok' if states else 'empty',
                error=error, states=deepcopy(states), active_seconds=cumulative if error is None else None,
                physical_entries=len(controller.entries), decisions=len(controller.decisions),
                arbiter_events=dict(Counter(item['event'] for item in arbiter.audit)) if arbiter else {}))
    finally:
        physical_engine._record_physical_entry = original
    return results


def cohort_truth(before, endpoint, *, start_time, horizon):
    if any(s['vehicle_id'] not in endpoint for s in before):
        raise ValueError('factual endpoint must cover every seed vehicle')
    rows, seconds = {}, 0.
    for state in before:
        key = state['vehicle_id']
        true = endpoint[key]
        duration = horizon
        if true['arrived']:
            duration = float(true['arrival_time']) - start_time
            if not 0 < duration <= horizon:
                raise ValueError('observed arrival falls outside the active-cohort horizon')
        seconds += duration
        rows[key] = true
    return dict(states=rows, active_seconds=seconds, arrivals=sum(s['arrived'] for s in rows.values()))


def compact(state):
    return {key: state[key] for key in ('vehicle_id', 'departed', 'arrived', 'current_edge',
            'remaining_distance', 'arrival_time') if key in state}


def collect_real(config, prepared, topology, *, seconds, sample_every, max_horizon, output):
    network = topology.network
    observer = PeriodicCostObserver(30, network.free_flow_travel_seconds)
    router = RealTimeShortestRouter(network)
    seeds, endpoints, entries = [], {}, defaultdict(list)
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        session.reset()
        api = session._connection
        version = list(api.getVersion())

        def policy(pending, flow, now):
            return [NextHopDecision(s['vehicle_id'], s['current_edge'], router.next_edge(s['current_edge'],
                s['destination_edge'], session.candidates.candidates(s['current_edge'], s['destination_edge'])), now) for s in pending]

        flow = session._flow().copy()
        for timestamp in range(seconds+1):
            if timestamp % sample_every == 0:
                endpoints[timestamp] = {key: compact(state) for key, state in session._vehicles.items() if state['departed']}
            if timestamp == seconds:
                break
            session._apply_incident()
            if observer.observe(timestamp, lambda: [api.edge.getTraveltime(edge) for edge in network.edge_ids]):
                router.update_costs(observer.values, observer.timestamp)
            sampled = timestamp % sample_every == 0 and timestamp + max_horizon <= seconds
            if sampled:
                flow = session._flow().copy()
            decisions = session._apply_decisions(policy, flow)
            tracking = {key: (session._actual_routes[key], int(api.vehicle.getRouteIndex(key))) for key in session._active}
            if sampled:
                states = deepcopy([state for state in session._vehicles.values() if state['departed'] and not state['arrived']])
                routes = {s['vehicle_id']: tuple(network.edge_to_index[edge] for edge in
                    tracking[s['vehicle_id']][0][tracking[s['vehicle_id']][1]:]) for s in states}
                seeds.append(dict(timestamp=timestamp, states=states, routes=routes, flow=flow.tolist(),
                    entry_cursors={s['vehicle_id']: len(entries[s['vehicle_id']]) for s in states},
                    applied_seed_controls=[asdict(d) for d in decisions], cost_observation_time=observer.timestamp))
            api.simulationStep()
            session._time_ms += 1000
            session._observe(np.zeros(network.edge_count))
            for key, (route, old_index) in tracking.items():
                state = session._vehicles[key]
                new_index = len(route)-1 if state['arrived'] else int(api.vehicle.getRouteIndex(key))
                if new_index < old_index:
                    raise ValueError('factual accepted route index moved backwards within physical step')
                for edge in route[old_index+1:new_index+1]:
                    entries[key].append((timestamp+1, int(network.edge_to_index[edge])))
            if timestamp % 600 == 0:
                print(json.dumps(dict(stage='real_collection', time=timestamp, active=len(session._active))), flush=True)
        completion = dict(observed_until=seconds, arrived=len(session.travel_times), active=len(session._active),
            never_departed=sum(not s['departed'] for s in session._vehicles.values()), population=len(topology.vehicles))
    with gzip.open(output/'real_cohorts.jsonl.gz', 'wt') as stream:
        for row in seeds:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    with gzip.open(output/'real_endpoints.jsonl.gz', 'wt') as stream:
        for timestamp, states in endpoints.items():
            stream.write(json.dumps(dict(timestamp=timestamp, states=states), allow_nan=False)+'\n')
    with gzip.open(output/'real_entries.json.gz', 'wt') as stream:
        json.dump(entries, stream)
    return seeds, endpoints, entries, dict(sumo_version=version, completion=completion,
        cost_observation_times=observer.sample_times, routing_stats=router.stats)


def comparison(seed, prediction, endpoint, actual_entries, edge_count, implementation):
    start, horizon = seed['timestamp'], prediction['horizon']
    truth = cohort_truth(seed['states'], endpoint, start_time=start, horizon=horizon)
    predicted = {s['vehicle_id']: s for s in prediction['states']}
    rows = []
    for old in seed['states']:
        key = old['vehicle_id']
        true = truth['states'][key]
        cursor = seed['entry_cursors'][key]
        traveled = [edge for observed, edge in actual_entries.get(key, [])[cursor:] if observed <= start+horizon]
        route = seed['routes'][key]
        compatible = len(traveled) < len(route) and tuple(traveled) == tuple(route[1:len(traveled)+1])
        row = dict(timestamp=start, horizon=horizon, implementation=implementation, vehicle_id=key,
            status=prediction['status'], error=prediction['error'], factual_route_prefix_compatible=compatible,
            true_road=int(true['current_edge']), true_remaining=float(true['remaining_distance']),
            true_arrived=bool(true['arrived']),
            true_active_seconds=(float(true['arrival_time'])-start if true['arrived'] else float(horizon)))
        if prediction['status'] == 'ok':
            pred = predicted[key]
            same = pred['current_edge'] == true['current_edge']
            row.update(predicted_road=int(pred['current_edge']), predicted_remaining=float(pred['remaining_distance']),
                predicted_arrived=bool(pred['arrived']), road_match=bool(same),
                arrival_match=bool(pred['arrived'] == true['arrived']),
                remaining_error_m=(float(pred['remaining_distance']-true['remaining_distance'])
                                   if same and not pred['arrived'] and not true['arrived'] else None),
                predicted_active_seconds=(float(pred['arrival_time'])-start if pred['arrived'] else float(horizon)))
        rows.append(row)
    window = dict(timestamp=start, horizon=horizon, implementation=implementation, status=prediction['status'],
        error=prediction['error'], cohort=len(rows), true_active_seconds=truth['active_seconds'],
        true_completed=truth['arrivals'], predicted_active_seconds=prediction['active_seconds'],
        physical_entries=prediction['physical_entries'], decisions=prediction['decisions'],
        arbiter_events=prediction['arbiter_events'], factual_route_prefix_deviations=sum(not r['factual_route_prefix_compatible'] for r in rows))
    if prediction['status'] in {'ok', 'empty'}:
        true_counts = np.zeros(edge_count+1, dtype=int)
        predicted_counts = np.zeros_like(true_counts)
        for key, true in truth['states'].items():
            pred = predicted[key]
            true_counts[edge_count if true['arrived'] else true['current_edge']] += 1
            predicted_counts[edge_count if pred['arrived'] else pred['current_edge']] += 1
        error = prediction['active_seconds']-truth['active_seconds']
        window.update(active_seconds_error=error, predicted_completed=int(predicted_counts[-1]),
            count_l1=int(np.abs(predicted_counts-true_counts).sum()),
            count_total_variation=(float(np.abs(predicted_counts-true_counts).sum())/(2*len(rows)) if rows else 0.),
            true_counts={str(i): int(v) for i, v in enumerate(true_counts) if v},
            predicted_counts={str(i): int(v) for i, v in enumerate(predicted_counts) if v})
        if abs(sum(r['predicted_active_seconds'] for r in rows)-prediction['active_seconds']) > 1e-6:
            raise AssertionError('executor cumulative cost differs from observed arrival accounting')
    return window, rows


def summarize(windows, rows, horizons):
    summaries = {}
    for horizon in horizons:
        selected_rows = [row for row in rows if row['horizon'] == horizon]
        paired = defaultdict(dict)
        for row in selected_rows:
            paired[(row['timestamp'], row['vehicle_id'])][row['implementation']] = row
        common = [pair for pair in paired.values() if set(pair) == {'legacy', 'city'} and all(r['status'] == 'ok' for r in pair.values())]
        common_position = [pair for pair in common if all(r['remaining_error_m'] is not None for r in pair.values())]
        common_route = [pair for pair in common if all(r['factual_route_prefix_compatible'] for r in pair.values())]
        arms = {}
        for implementation in ('legacy', 'city'):
            attempts = [w for w in windows if w['horizon'] == horizon and w['implementation'] == implementation]
            valid = [w for w in attempts if w['status'] in {'ok', 'empty'}]
            pairs = [p[implementation] for p in common]
            errors = [abs(p['remaining_error_m']) for p in pairs if p['remaining_error_m'] is not None]
            route_rows = [p[implementation] for p in common_route]
            arms[implementation] = dict(attempted_windows=len(attempts), successful_nonempty_windows=sum(w['status']=='ok' for w in attempts),
                empty_windows=sum(w['status']=='empty' for w in attempts), failed_windows=sum(w['status']=='error' for w in attempts),
                error_types=dict(Counter(w['error']['type'] for w in attempts if w['error'])),
                attempted_vehicle_rows=sum(w['cohort'] for w in attempts), common_rows=len(pairs),
                wrong_road=sum(not r['road_match'] for r in pairs), arrival_mismatches=sum(not r['arrival_match'] for r in pairs),
                predicted_completed=sum(r['predicted_arrived'] for r in pairs), true_completed=sum(r['true_arrived'] for r in pairs),
                own_position_valid_rows=len(errors), own_position_mae_m=float(np.mean(errors)) if errors else None,
                common_position_mae_m=(float(np.mean([abs(p[implementation]['remaining_error_m']) for p in common_position])) if common_position else None),
                common_active_seconds_error=sum(r['predicted_active_seconds']-r['true_active_seconds'] for r in pairs),
                common_active_seconds_mae_per_vehicle=float(np.mean([abs(r['predicted_active_seconds']-r['true_active_seconds']) for r in pairs])) if pairs else None,
                common_true_active_seconds=sum(r['true_active_seconds'] for r in pairs),
                window_active_seconds_mae=float(np.mean([abs(w['active_seconds_error']) for w in valid])) if valid else None,
                mean_count_total_variation=float(np.mean([w['count_total_variation'] for w in valid if w['cohort']])) if any(w['cohort'] for w in valid) else None,
                route_compatible_wrong_road=sum(not r['road_match'] for r in route_rows),
                route_compatible_cost_error=sum(r['predicted_active_seconds']-r['true_active_seconds'] for r in route_rows),
                route_compatible_cost_mae_per_vehicle=float(np.mean([abs(r['predicted_active_seconds']-r['true_active_seconds']) for r in route_rows])) if route_rows else None)
        summaries[str(horizon)] = dict(common_rows=len(common), common_position_rows=len(common_position),
            factual_route_compatible_common_rows=len(common_route), arms=arms)
    return summaries


def run(args):
    horizons = tuple(args.horizons)
    if not 610000 <= args.seed < 620000 or horizons != tuple(sorted(set(horizons))) or min(horizons) <= 0:
        raise ValueError('development seed and increasing positive horizons required')
    if args.seconds < max(horizons) or args.seconds % args.sample_every or any(h % args.sample_every for h in horizons):
        raise ValueError('complete endpoints require cutoff and horizons divisible by sample interval')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original = load_config(args.config)
    config = replace(original, run=replace(original.run, seed=args.seed, device='cpu', output_dir=output),
        simulation=replace(original.simulation, sumo_extra_args=tuple(original.simulation.sumo_extra_args)+('--error-log', str(output/'sumo_warnings.log'))))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    law = PoissonDemand.from_vehicles(topology.vehicles, duration_seconds=config.simulation.demand_duration_seconds)
    vehicles = law.sample(seed=args.base_seed, episode_index=args.seed)
    topology = replace(topology, vehicles=vehicles)
    network = topology.network
    controls = CityControlTable.from_network(prepared.network_xml, network.edge_ids, no_external_overrides=True)
    candidates = NextHopCandidates(network, traffic_model='meso')
    files = [Path(__file__), ROOT/'tools/diagnose_component_executor.py', ROOT/'tools/causal_routing_policies.py',
        ROOT/'tools/routing_cost_observer.py', ROOT/'urban_dreamer/learning/imagination.py',
        ROOT/'urban_dreamer/learning/city_imagination.py', ROOT/'urban_dreamer/traffic/movement_controls.py',
        ROOT/'urban_dreamer/traffic/event_session.py', ROOT/'urban_dreamer/traffic/demand.py', ROOT/'urban_dreamer/traffic/topology.py']
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    for source in files:
        target = output/'source'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    protocol = dict(schema_version=1, seed=args.seed, split='development', base_demand_seed=args.base_seed,
        config=str(args.config.resolve()), config_sha256=sha256(args.config), source_hashes=hashes,
        network_sha256=sha256(prepared.network_xml), registry_sha256=sha256(prepared.edge_registry),
        demand_sha256=demand_identity(vehicles), population=len(vehicles), horizons=horizons,
        simulation_seconds=args.seconds, sample_every_seconds=args.sample_every,
        sampled_ticks=list(range(0,args.seconds-max(horizons)+1,args.sample_every)),
        physics=physics_identity(config.simulation), sumo_model_args=sumo_model_args(config.simulation),
        edge_ids=list(network.edge_ids), count_arrived_bin=network.edge_count,
        real_policy='dynamic shortest, causal30s edge-cost refresh,1s physical observations',
        rollout_control='accepted full route suffix at seed after real control; frozen causal route cursor, every entry audited, no fallback',
        primary_speed='fixed boundary road mean for all120seconds; no future speed refresh',
        recursion='same rollout state and city arbiter preserved across horizon checkpoints; no truth reset after seed',
        fifo='spatial order approximation; tied seed positions rejected; no ID-derived exact order',
        initial_calendars='empty at seed; source/receiving/conflict reservations persist thereafter',
        background='real future births and real dynamic rerouting continue; absent from imagined cohort/control',
        interpretation='factual cohort comparison under shared background, not a closed-loop counterfactual or policy ranking',
        factual_route_check='physical entries inferred from actual accepted route indices each second; deviation from frozen suffix reported',
        threads=2, device='cpu', optimizer_updates=0)
    atomic_json(output/'protocol.json', protocol)
    atomic_json(output/'demand.json', dict(demand_sha256=protocol['demand_sha256'], vehicles=[asdict(v) for v in vehicles]))
    started = time.monotonic()
    seeds, endpoints, entries, real = collect_real(config, prepared, topology, seconds=args.seconds,
        sample_every=args.sample_every, max_horizon=max(horizons), output=output)
    protocol['sumo_version'] = real['sumo_version']
    atomic_json(output/'protocol.json', protocol)
    windows, rows = [], []
    with (output/'windows.jsonl').open('w') as window_file, gzip.open(output/'rows.jsonl.gz', 'wt') as row_file:
        for index, initial in enumerate(seeds):
            flow = np.asarray(initial['flow'], dtype=np.float32)
            speeds = np.maximum(0., flow[:,1]*network.speed_limits_mps)
            for implementation in ('legacy', 'city'):
                predictions = run_free_rollout(initial['states'], routes=initial['routes'], implementation=implementation,
                    candidates=candidates, controls=controls, speeds=speeds, flow=flow, start_time=initial['timestamp'],
                    population=len(vehicles), horizons=horizons, max_speed=config.simulation.vehicle_max_speed_mps,
                    seed=args.seed+initial['timestamp'])
                for prediction in predictions:
                    endpoint_time = initial['timestamp']+prediction['horizon']
                    if endpoint_time not in endpoints:
                        raise ValueError(f'missing complete factual endpoint {endpoint_time}')
                    window, new_rows = comparison(initial, prediction, endpoints[endpoint_time], entries, network.edge_count, implementation)
                    windows.append(window)
                    rows.extend(new_rows)
                    window_file.write(json.dumps(window, allow_nan=False)+'\n')
                    for row in new_rows:
                        row_file.write(json.dumps(row, allow_nan=False)+'\n')
            window_file.flush()
            row_file.flush()
            if index % 10 == 0:
                print(json.dumps(dict(stage='free_rollouts', seed_time=initial['timestamp'], seeds_done=index+1,
                    seed_count=len(seeds), wall_seconds=time.monotonic()-started)), flush=True)
    after_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    report = dict(status='complete' if hashes == after_hashes else 'source_changed_during_run',
        source_hashes_stable=hashes == after_hashes, source_hashes_after=after_hashes,
        horizons=summarize(windows, rows, horizons), real_episode=real, wall_seconds=time.monotonic()-started,
        scope='one development demand seed; overlapping active cohorts; factual background retained, not independent windows or counterfactual policy proof')
    atomic_json(output/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'outputs/component_experiments_20261011/snapshot/run_config.toml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=610004)
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--seconds', type=int, default=2400)
    p.add_argument('--sample-every', type=int, default=30)
    p.add_argument('--horizons', type=int, nargs='+', default=[30,60,120])
    return p


if __name__ == '__main__':
    run(parser().parse_args())
