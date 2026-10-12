"""Development-only real Ingolstadt reset-shadow audit; no world model or RL.

Each sampled one-second shadow starts from the actual active cohort. The real
episode continues to receive demand; shadows do not. Boundary road-mean speeds
are causal; retrospective next-step means are a separate diagnostic oracle.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import gzip
import json
import math
import os
from pathlib import Path
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
from tools.diagnose_component_executor import ExactStepRoute, RouteMismatch
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


class AuditArbiter(ChronologicalJunctionArbiter):
    """Observe diagnostic wait conditions without changing any grant decisions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_observations = set()

    def grants(self, now, speeds, *, control_time=None):
        control_tick = now if control_time is None else control_time
        heads = []
        for queue in self.queues.values():
            if queue:
                state = queue[0]
                movement = self.movement(state)
                edge = int(state['current_edge'])
                distance, speed = float(state['remaining_distance']), float(speeds[edge])
                eta = now if distance <= 1e-9 else now + distance / speed if speed > 0 else math.inf
                kind = 'protected' if movement is None else self.controls.classification(movement, control_tick)
                heads.append((state, movement, kind, eta))
        result = super().grants(now, speeds, control_time=control_time)
        granted = {str(state['vehicle_id']) for state, _, _ in result}
        for state, movement, kind, eta in heads:
            key = str(state['vehicle_id'])
            if eta > now + 1e-9 or key in granted or movement is None:
                continue
            reasons = []
            if kind == 'blocked':
                reasons.append('red_wait')
            if self.target_ready.get(movement.target_edge, -math.inf) > now + 1e-9:
                reasons.append('receiving_headway_wait')
            if kind in {'conditional', 'yield'} and any(
                    other is not state and other_m is not None and other_kind != 'blocked'
                    and other_eta <= now + self.clearance + 1e-9 and self._conflicts(movement, other_m)
                    and (other_kind == 'protected' or other_m.request_index in movement.yields_to)
                    for other, other_m, other_kind, other_eta in heads):
                reasons.append('priority_wait')
            if any(other is not state and other_m is not None
                   and other_m.target_edge == movement.target_edge and other_eta <= now + 1e-9
                   for other, other_m, _, other_eta in heads):
                reasons.append('merge_competition_wait')
            for reason in reasons:
                self.wait_observations.add((key, float(now), reason))
        return result


def run_shadow(before, *, implementation, candidates, controls, speeds, flow,
               timestamp, population, decisions, routes, max_speed, seed):
    states = deepcopy(before)
    controller = ExactStepRoute(decisions, routes, timestamp)
    arbiter = AuditArbiter(controls, seed=seed, uncertain_order=False) if implementation == 'city' else None
    original = physical_engine._record_physical_entry
    entries = []

    def checked_entry(state, edge, choices):
        source = int(state['current_edge'])
        controller.entry(state, edge)
        entries.append(dict(vehicle_id=state['vehicle_id'], source_edge=source, target_edge=int(edge),
                            signal_state=controls.state((source, int(edge)), timestamp)))
        original(state, edge, choices)

    result, error = None, None
    try:
        physical_engine._record_physical_entry = checked_entry
        engine = advance_city_imagined_window if implementation == 'city' else physical_engine.advance_imagined_window
        extra = dict(junction_arbiter=arbiter) if arbiter is not None else {}
        result = engine(states, candidates=candidates, edge_speeds_mps=np.asarray(speeds),
            frozen_flow=flow, start_time=timestamp, total_vehicle_count=population,
            decide=controller, interval_seconds=1., observation_step_seconds=1.,
            session_protocol='event', traffic_model='meso', vehicle_max_speed_mps=max_speed,
            stop_when_empty=False, **extra)
        if controller.used != set(controller.decisions):
            raise RouteMismatch('not all real controls applied in shadow')
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        physical_engine._record_physical_entry = original
    counters = Counter()
    counters['physical_entries'] = len(entries)
    counters['red_crossings'] = sum(row['signal_state'] == 'r' for row in entries)
    counters['conditional_crossings'] = sum(row['signal_state'] in {'g', 'y'} for row in entries)
    counters['yield_crossings'] = sum(row['signal_state'] in {'m', '=', 'o'} for row in entries)
    targets = defaultdict(set)
    for row in entries:
        targets[row['target_edge']].add(row['source_edge'])
    counters['multi_upstream_entry_targets'] = sum(len(sources) > 1 for sources in targets.values())
    audit = [] if arbiter is None else arbiter.audit
    if arbiter is not None:
        counters.update(reason for _, _, reason in arbiter.wait_observations)
        counters.update(row['event'] for row in audit if row['event'] != 'queue_initialization')
    return dict(states=states, result=result, error=error, entries=entries, counters=dict(counters), audit=audit)


def comparison_rows(before, actual, shadow, *, timestamp, speed_arm, implementation):
    predicted = {state['vehicle_id']: state for state in shadow['states']}
    rows = []
    for old in before:
        key = old['vehicle_id']
        truth = actual[key]
        row = dict(timestamp=timestamp, speed_arm=speed_arm, implementation=implementation,
            vehicle_id=key, status='error' if shadow['error'] else 'ok', error=shadow['error'],
            road_before=int(old['current_edge']), remaining_before=float(old['remaining_distance']),
            true_road=int(truth['current_edge']), true_remaining=float(truth['remaining_distance']),
            true_arrived=bool(truth['arrived']))
        if shadow['error'] is None:
            pred = predicted[key]
            same = pred['current_edge'] == truth['current_edge']
            valid = same and not pred['arrived'] and not truth['arrived']
            row.update(predicted_road=int(pred['current_edge']), predicted_remaining=float(pred['remaining_distance']),
                predicted_arrived=bool(pred['arrived']), road_match=bool(same),
                arrival_match=bool(pred['arrived'] == truth['arrived']),
                remaining_error_m=float(pred['remaining_distance']-truth['remaining_distance']) if valid else None)
        rows.append(row)
    return rows


def metrics(rows):
    errors = [abs(row['remaining_error_m']) for row in rows if row.get('remaining_error_m') is not None]
    return dict(vehicle_rows=len(rows), wrong_road=sum(not row['road_match'] for row in rows),
        arrival_mismatches=sum(not row['arrival_match'] for row in rows), position_valid_rows=len(errors),
        position_mae_m=float(np.mean(errors)) if errors else None,
        position_p95_abs_m=float(np.percentile(errors, 95)) if errors else None,
        position_max_abs_m=max(errors) if errors else None)


def summarize(windows, rows):
    arms, common = {}, {}
    for speed in ('boundary_causal', 'retrospective_oracle'):
        by_vehicle = defaultdict(dict)
        for row in rows:
            if row['speed_arm'] == speed:
                by_vehicle[(row['timestamp'], row['vehicle_id'])][row['implementation']] = row
        paired = [pair for pair in by_vehicle.values() if set(pair) == {'legacy', 'city'}
                  and all(row['status'] == 'ok' for row in pair.values())]
        valid_position_pairs = [pair for pair in paired if all(row['remaining_error_m'] is not None for row in pair.values())]
        common[speed] = dict(common_success_rows=len(paired), common_position_rows=len(valid_position_pairs),
            implementations={impl: dict(metrics([p[impl] for p in paired]),
                common_position_mae_m=(float(np.mean([abs(p[impl]['remaining_error_m']) for p in valid_position_pairs]))
                                       if valid_position_pairs else None)) for impl in ('legacy', 'city')})
        for implementation in ('legacy', 'city'):
            selected = [w for w in windows if w['speed_arm'] == speed and w['implementation'] == implementation]
            ok = [w for w in selected if w['status'] == 'ok']
            successful_rows = [r for r in rows if r['speed_arm'] == speed and r['implementation'] == implementation and r['status'] == 'ok']
            counters = Counter()
            for window in selected:
                counters.update(window['counters'])
            arms[f'{speed}/{implementation}'] = dict(metrics(successful_rows), attempted_windows=len(selected),
                successful_windows=len(ok), empty_windows=sum(w['status'] == 'empty' for w in selected),
                failed_windows=sum(w['status'] == 'error' for w in selected),
                attempted_vehicle_rows=sum(w['cohort'] for w in selected),
                error_types=dict(Counter(w['error']['type'] for w in selected if w['error'])),
                active_seconds_max_abs=max((abs(w['active_seconds_error']) for w in ok), default=None),
                counters=dict(counters))
    return dict(arms=arms, common_rows=common,
        cost_limit='one-second reset exposure is imposed by the observation convention; not free-rollout cost evidence')


def run(args):
    if not 610000 <= args.seed < 620000:
        raise ValueError('development seed must be in 610000..619999')
    if args.seconds <= 0 or args.sample_every <= 0:
        raise ValueError('positive duration and sampling interval required')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    original_config = load_config(args.config)
    config = replace(original_config, run=replace(original_config.run, device='cpu', seed=args.seed,
        output_dir=output), simulation=replace(original_config.simulation,
        sumo_extra_args=tuple(original_config.simulation.sumo_extra_args) + ('--error-log', str(output/'sumo_warnings.log'))))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    law = PoissonDemand.from_vehicles(topology.vehicles, duration_seconds=config.simulation.demand_duration_seconds)
    vehicles = law.sample(seed=args.base_seed, episode_index=args.seed)
    topology = replace(topology, vehicles=vehicles)
    network = topology.network
    controls = CityControlTable.from_network(prepared.network_xml, network.edge_ids, no_external_overrides=True)
    if any(m.support == 'unsupported' for m in controls.movements.values()):
        raise ValueError('shadow audit requires all city controls classified without rail assumptions')
    candidates = NextHopCandidates(network, traffic_model='meso')
    observer = PeriodicCostObserver(30, network.free_flow_travel_seconds)
    router = RealTimeShortestRouter(network)
    source_paths = [Path(__file__), ROOT/'tools/diagnose_component_executor.py',
        ROOT/'tools/causal_routing_policies.py', ROOT/'tools/routing_cost_observer.py',
        ROOT/'urban_dreamer/traffic/event_session.py', ROOT/'urban_dreamer/traffic/movement_controls.py',
        ROOT/'urban_dreamer/learning/imagination.py', ROOT/'urban_dreamer/learning/city_imagination.py']
    protocol = dict(schema_version=1, scope='development Ingolstadt real one-second reset shadow; no WM or optimizer',
        config_path=str(args.config.resolve()), config_sha256=sha256(args.config), seed=args.seed, base_demand_seed=args.base_seed,
        demand_sha256=demand_identity(vehicles), population=len(vehicles), seconds=args.seconds,
        sample_every_seconds=args.sample_every, sampled_ticks=list(range(0, args.seconds, args.sample_every)),
        physics=physics_identity(config.simulation), sumo_model_args=sumo_model_args(config.simulation),
        network_sha256=sha256(prepared.network_xml), registry_sha256=sha256(prepared.edge_registry),
        input_hashes=prepared.input_hashes, edge_ids=list(network.edge_ids),
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        real_policy='legal dynamic-shortest next hop with causal edge travel-time refresh at0,30,60,...; accepted suffix uses session routing',
        shadow_control='exact recorded real decisions plus every physical entry audited against accepted route suffix',
        primary_speed='boundary_causal: previous realized step road mean available at seed time',
        secondary_speed='retrospective_oracle: realized next-step road mean, diagnostic future information',
        empty_speed='network extractor empty-road convention preserved; negative speeds clamped0',
        fifo='unequal spatial-position approximation; tied seeds are errors; no ID-derived exact order',
        reservations='fresh empty reservation/source/receiving calendars at every reset seed',
        surrogate_headways_seconds=dict(discharge=1., receiving=1., conflict_clearance=1.),
        background='real full demand continues; shadow active-at-seed cohort only, no future births',
        counter_semantics='entry counters include partial execution before failures; wait counters are unique vehicle/time/condition observations, not seconds',
        cost_limit='one-second active-seconds match follows observation accounting; no multistep reliability claim',
        threads=2, device='cpu', optimizer_updates=0)
    atomic_json(output/'demand.json', dict(demand_sha256=protocol['demand_sha256'], vehicles=[asdict(v) for v in vehicles]))
    atomic_json(output/'protocol.json', protocol)
    windows, rows = [], []
    started = time.monotonic()
    with gzip.open(output/'real_samples.jsonl.gz', 'wt') as raw, gzip.open(output/'rows.jsonl.gz', 'wt') as row_file, \
            (output/'windows.jsonl').open('w') as window_file, SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        session.reset()
        api = session._connection
        protocol['sumo_version'] = list(api.getVersion())
        atomic_json(output/'protocol.json', protocol)

        def policy(pending, frozen_flow, now):
            return [NextHopDecision(v['vehicle_id'], v['current_edge'], router.next_edge(v['current_edge'],
                v['destination_edge'], session.candidates.candidates(v['current_edge'], v['destination_edge'])), now) for v in pending]

        flow_before = session._flow().copy()
        for timestamp in range(args.seconds):
            session._apply_incident()
            if observer.observe(timestamp, lambda: [api.edge.getTraveltime(edge) for edge in network.edge_ids]):
                router.update_costs(observer.values, observer.timestamp)
            sampled = timestamp % args.sample_every == 0
            if sampled:
                before = deepcopy([s for s in session._vehicles.values() if s['departed'] and not s['arrived']])
                flow_before = session._flow().copy()
            decisions = session._apply_decisions(policy, flow_before)
            routes = {}
            if sampled:
                for state in before:
                    key = state['vehicle_id']
                    suffix = api.vehicle.getRoute(key)[api.vehicle.getRouteIndex(key):]
                    routes[key] = tuple(network.edge_to_index[edge] for edge in suffix)
                    if not routes[key] or routes[key][0] != state['current_edge']:
                        raise ValueError('actual route cursor disagrees with seed')
            api.simulationStep()
            session._time_ms += 1000
            session._observe(np.zeros(network.edge_count))
            if sampled:
                after = {s['vehicle_id']: deepcopy(session._vehicles[s['vehicle_id']]) for s in before}
                flow_after = session._flow().copy()
                raw.write(json.dumps(dict(timestamp=timestamp, before=before, after=list(after.values()),
                    decisions=[asdict(d) for d in decisions], routes=routes,
                    flow_before=flow_before.tolist(), flow_after=flow_after.tolist(),
                    new_departures=list(api.simulation.getDepartedIDList()),
                    active_after=len(session._active), cost_observation_time=observer.timestamp), allow_nan=False)+'\n')
                for speed_arm, flow in [('boundary_causal', flow_before), ('retrospective_oracle', flow_after)]:
                    speeds = np.maximum(0., flow[:, 1]*network.speed_limits_mps)
                    for implementation in ('legacy', 'city'):
                        if before:
                            shadow = run_shadow(before, implementation=implementation, candidates=candidates,
                                controls=controls, speeds=speeds, flow=flow_before, timestamp=timestamp,
                                population=len(vehicles), decisions=decisions, routes=routes,
                                max_speed=config.simulation.vehicle_max_speed_mps, seed=args.seed+timestamp)
                        else:
                            shadow = dict(states=[], result=None, error=None, entries=[], counters={}, audit=[])
                        status = 'empty' if not before else 'error' if shadow['error'] else 'ok'
                        window = dict(timestamp=timestamp, speed_arm=speed_arm, implementation=implementation,
                            status=status, error=shadow['error'], cohort=len(before),
                            true_active_seconds=len(before), predicted_active_seconds=(shadow['result'].active_vehicle_seconds
                                if status == 'ok' else None), active_seconds_error=(shadow['result'].active_vehicle_seconds-len(before)
                                if status == 'ok' else None), entries=shadow['entries'], counters=shadow['counters'],
                            arbiter_audit=shadow['audit'])
                        windows.append(window)
                        window_file.write(json.dumps(window, allow_nan=False)+'\n')
                        new_rows = comparison_rows(before, after, shadow, timestamp=timestamp,
                            speed_arm=speed_arm, implementation=implementation)
                        rows.extend(new_rows)
                        for row in new_rows:
                            row_file.write(json.dumps(row, allow_nan=False)+'\n')
                window_file.flush()
                raw.flush()
                row_file.flush()
            if timestamp % 300 == 0:
                print(json.dumps(dict(time=timestamp, active=len(session._active), sampled_windows=len(windows),
                    wall_seconds=time.monotonic()-started)), flush=True)
        actual_summary = dict(observed_until=session._time_ms/1000., arrived=len(session.travel_times),
            active=len(session._active), never_departed=sum(not s['departed'] for s in session._vehicles.values()))
    source_hashes_after = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    source_stable = source_hashes_after == protocol['source_hashes']
    report = dict(status='complete' if source_stable else 'source_changed_during_run',
        source_hashes_stable=source_stable, source_hashes_after=source_hashes_after,
        **summarize(windows, rows), real_episode=actual_summary,
        cost_observation_times=observer.sample_times, routing_stats=router.stats, wall_seconds=time.monotonic()-started)
    atomic_json(output/'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, default=ROOT/'outputs/component_experiments_20261011/snapshot/run_config.toml')
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--seed', type=int, default=610004)
    result.add_argument('--base-seed', type=int, default=42)
    result.add_argument('--seconds', type=int, default=2400)
    result.add_argument('--sample-every', type=int, default=30)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
