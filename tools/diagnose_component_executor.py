"""Small real SUMO meso oracle-speed shadow audit, with strict route matching."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, sha256
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.learning.meso_constraints import MesoReleaseConstraint, StaticSignalSchedule, UnsupportedConstraint
import urban_dreamer.learning.imagination as physical_engine
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.simulation_settings import physics_identity, sumo_model_args
from urban_dreamer.traffic.topology import NextHopCandidates, VehicleDemand, load_traffic_topology


class RouteMismatch(ValueError):
    pass


class ExactStepRoute:
    """Use actual controls and audit every shadow physical entry; no fallback."""
    def __init__(self, decisions, routes, timestamp):
        self.decisions = {item.vehicle_id: item for item in decisions}
        self.routes = routes
        self.indices = {key: 0 for key in routes}
        self.timestamp = timestamp
        self.used = set()
        self.entries = []

    def __call__(self, pending, flow, now):
        result = []
        for state in pending:
            key = state['vehicle_id']
            recorded = self.decisions.get(key)
            if (recorded is None or key in self.used or now != self.timestamp
                    or recorded.current_edge != state['current_edge']):
                raise RouteMismatch(f'unmatched one-second callback for {key} at {now}')
            self.used.add(key)
            result.append(recorded)
        return result

    def entry(self, state, edge):
        key = state['vehicle_id']
        position = self.indices[key] + 1
        if position >= len(self.routes[key]) or int(edge) != self.routes[key][position]:
            raise RouteMismatch(f'{key}: shadow physical entry {edge} differs from commanded route')
        self.indices[key] = position
        self.entries.append(dict(vehicle_id=key, next_edge=int(edge)))


def shadow_step(initial_states, *, candidates, speeds, flow, timestamp, population,
                decisions, routes, max_speed, passage_constraint=None):
    states = deepcopy(initial_states)
    controller = ExactStepRoute(decisions, routes, timestamp)
    original = physical_engine._record_physical_entry

    def checked_entry(state, edge, choices):
        controller.entry(state, edge)
        original(state, edge, choices)

    try:
        physical_engine._record_physical_entry = checked_entry
        result = advance_imagined_window(states, candidates=candidates,
            edge_speeds_mps=np.asarray(speeds), frozen_flow=flow, start_time=timestamp,
            total_vehicle_count=population, decide=controller, interval_seconds=1.,
            observation_step_seconds=1., session_protocol='event', traffic_model='meso',
            vehicle_max_speed_mps=max_speed, stop_when_empty=False, passage_constraint=passage_constraint)
        if controller.used != set(controller.decisions):
            raise RouteMismatch('not all real controls were applied in the shadow step')
    finally:
        physical_engine._record_physical_entry = original
    return states, result, controller.entries


def make_fixture(output):
    """Reuse the real single-queue red/green-turn test construction in an isolated XML."""
    base = ROOT / 'networks/toy_cross_bottleneck'
    tree = ET.parse(base/'toy_cross_bottleneck.net.xml')
    root = tree.getroot()
    hold = next(edge for edge in root.findall('edge') if edge.get('id') == 'A_hold')
    lane = deepcopy(hold.find('lane'))
    lane.set('id', 'A_hold_1'); lane.set('index', '1'); hold.append(lane)
    junction = next(node for node in root.findall('junction') if node.get('id') == 'A2')
    junction.set('type', 'traffic_light'); junction.set('incLanes', 'A_hold_0 A_hold_1')
    for request in junction.findall('request'):
        junction.remove(request)
    for index in range(6):
        ET.SubElement(junction, 'request', index=str(index), response='000000', foes='000000', cont='0')
    phases = []
    for index, connection in enumerate([c for c in root.findall('connection') if c.get('from') == 'A_hold']):
        connection.set('tl', 'A2'); connection.set('linkIndex', str(index))
        phases.append('r' if connection.get('to') == 'A_to_R' else 'G')
        duplicate = deepcopy(connection); duplicate.set('fromLane', '1'); root.append(duplicate)
    logic = ET.Element('tlLogic', id='A2', type='static', programID='0', offset='0')
    ET.SubElement(logic, 'phase', duration='3600', state=''.join(phases)); root.insert(1, logic)
    net = output/'combined_queue.net.xml'; tree.write(net)
    prepared = SimpleNamespace(network_xml=net, network_agents_csv=base/'agents.csv',
        edge_registry=base/'edge_registry.json', od_file=base/'od_toy_cross_bottleneck.txt')
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=(VehicleDemand('front-red', 0, 0, 0.),
        VehicleDemand('behind-green', 0, 0, 3.)))
    simulation = SimpleNamespace(vehicle_max_speed_mps=40.,
        sumo_extra_args=('--error-log', str(output/'sumo_warnings.log')), incident_edge='',
        incident_start_seconds=0., incident_end_seconds=0., incident_speed_mps=.1,
        episode_wall_timeout_seconds=120., traffic_model='meso', backend='libsumo',
        meso_junction_control=True, meso_lane_queue=False, meso_multi_queue=False,
        meso_edge_length_m=98., meso_interpolate_positions=True)
    config = SimpleNamespace(run=SimpleNamespace(seed=123), simulation=simulation)
    return config, prepared, topology, ''.join(phases)


def run_audit(output, *, seconds=70, release_at=40):
    output = Path(output)
    if output.exists():
        raise ValueError('new output directory required; preserve earlier audit')
    if not 1 < release_at < seconds:
        raise ValueError('release must fall strictly inside simulation interval')
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    config, prepared, topology, red_phase = make_fixture(output)
    rows, windows, real_records = [], [], []
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        session.reset()
        api = session._connection
        version = api.getVersion()
        network = session.network
        candidates = NextHopCandidates(network, traffic_model='meso')
        hold = network.edge_to_index['A_hold']

        def real_policy(pending, flow, now):
            controls = []
            for state in pending:
                current, destination = state['current_edge'], state['destination_edge']
                if current == hold:
                    target = network.edge_to_index['A_to_R' if state['vehicle_id'] == 'front-red' else 'A_pre_X']
                else:
                    choices = session.candidates.candidates(current, destination)
                    costs = session.candidates.costs(destination)
                    target = min(choices, key=lambda edge: (network.free_flow_travel_seconds[edge] + costs[edge], edge))
                controls.append(NextHopDecision(state['vehicle_id'], current, int(target), now))
            return controls

        for timestamp in range(seconds):
            if timestamp == release_at:
                api.trafficlight.setRedYellowGreenState('A2', 'G'*len(red_phase))
            before = deepcopy([state for state in session._vehicles.values() if state['departed'] and not state['arrived']])
            flow_before = session._flow().copy()
            real_controls = session._apply_decisions(real_policy, flow_before)
            routes = {}
            for state in before:
                route = api.vehicle.getRoute(state['vehicle_id'])
                cursor = api.vehicle.getRouteIndex(state['vehicle_id'])
                routes[state['vehicle_id']] = tuple(network.edge_to_index[edge] for edge in route[cursor:])
                if routes[state['vehicle_id']][0] != state['current_edge']:
                    raise ValueError('real route cursor disagrees with snapshot')
            api.simulationStep(); session._time_ms += 1000
            session._observe(np.zeros(network.edge_count))
            flow_after = session._flow().copy()
            actual = deepcopy(session._vehicles)
            real_records.append(dict(timestamp=timestamp+1,
                tls=api.trafficlight.getRedYellowGreenState('A2'), states=list(actual.values()),
                controls=[dict(vehicle_id=item.vehicle_id, current_edge=item.current_edge,
                    next_edge=item.next_edge, decision_time=item.decision_time) for item in real_controls],
                commanded_route_suffixes=routes,
                edge_mean_speed_mps=(flow_after[:, 1]*network.speed_limits_mps).tolist(),
                vehicle_reported_speeds={key: float(api.vehicle.getSpeed(key)) for key in session._active}))
            if not before:
                continue
            for arm, flow in [('boundary_last_step', flow_before), ('realized_step_mean', flow_after)]:
                speeds = np.maximum(0., flow[:, 1]*network.speed_limits_mps)
                try:
                    imagined, result, entries = shadow_step(before, candidates=candidates, speeds=speeds,
                        flow=flow, timestamp=timestamp, population=2, decisions=real_controls, routes=routes,
                        max_speed=config.simulation.vehicle_max_speed_mps)
                except RouteMismatch as error:
                    windows.append(dict(timestamp=timestamp, arm=arm, route_match=False, error=str(error),
                                        cohort=len(before)))
                    continue
                windows.append(dict(timestamp=timestamp, arm=arm, route_match=True, cohort=len(before),
                    true_active_seconds=len(before), predicted_active_seconds=result.active_vehicle_seconds,
                    active_seconds_error=result.active_vehicle_seconds-len(before), entries=entries))
                for old, pred in zip(before, imagined):
                    truth = actual[old['vehicle_id']]
                    same_edge = pred['current_edge'] == truth['current_edge']
                    both_active = not pred['arrived'] and not truth['arrived']
                    same_original = truth['current_edge'] == old['current_edge'] and not truth['arrived']
                    rows.append(dict(timestamp=timestamp, arm=arm, vehicle_id=old['vehicle_id'],
                        road_before=network.edge_ids[old['current_edge']],
                        remaining_before=old['remaining_distance'],
                        true_road=network.edge_ids[truth['current_edge']], predicted_road=network.edge_ids[pred['current_edge']],
                        true_remaining=truth['remaining_distance'], predicted_remaining=pred['remaining_distance'],
                        true_arrived=truth['arrived'], predicted_arrived=pred['arrived'],
                        road_match=same_edge, arrival_match=pred['arrived'] == truth['arrived'],
                        remaining_error_m=(pred['remaining_distance']-truth['remaining_distance'] if same_edge and both_active else None),
                        true_progress_same_road=(old['remaining_distance']-truth['remaining_distance'] if same_original else None),
                        oracle_speed_on_start_road=float(speeds[old['current_edge']]),
                        tls=api.trafficlight.getRedYellowGreenState('A2')))
    for name, records in [('rows.jsonl', rows), ('windows.jsonl', windows), ('real_seconds.jsonl', real_records)]:
        (output/name).write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
    summary = {}
    for arm in ('boundary_last_step', 'realized_step_mean'):
        selected = [row for row in rows if row['arm'] == arm]
        attempts = [row for row in windows if row['arm'] == arm]
        errors = [abs(row['remaining_error_m']) for row in selected if row['remaining_error_m'] is not None]
        summary[arm] = dict(attempted_steps=len(attempts), route_matched_steps=sum(row['route_match'] for row in attempts),
            route_match_rate=sum(row['route_match'] for row in attempts)/len(attempts) if attempts else None,
            vehicle_steps=len(selected), road_mismatches=sum(not row['road_match'] for row in selected),
            road_match_rate=sum(row['road_match'] for row in selected)/len(selected) if selected else None,
            arrival_mismatches=sum(not row['arrival_match'] for row in selected),
            position_valid_vehicle_steps=len(errors),
            position_mae_m=float(np.mean(errors)) if errors else None,
            position_max_abs_m=max(errors) if errors else None,
            active_seconds_max_abs=max((abs(row['active_seconds_error']) for row in attempts if row['route_match']), default=None))
    paired = defaultdict(dict)
    for row in rows:
        if row['arm'] == 'realized_step_mean' and row['road_before'] == 'A_hold':
            paired[row['timestamp']][row['vehicle_id']] = row
    witnesses = [dict(timestamp=t, front=pair['front-red'], behind=pair['behind-green']) for t, pair in paired.items()
        if set(pair) == {'front-red', 'behind-green'} and pair['front-red']['true_progress_same_road'] is not None
        and abs(pair['front-red']['true_progress_same_road']) < 1e-8
        and pair['behind-green']['true_progress_same_road'] is not None
        and pair['behind-green']['true_progress_same_road'] > 1e-6 and t < release_at]
    protocol = dict(schema_version=1, scope='two-vehicle controlled toy meso shadow test, no WM',
        fixture_source='tests/test_meso_event_session.py::test_real_meso_multilane_single_queue_blocks_green_turn_behind_red_turn',
        sumo_version=list(version), physics=physics_identity(config.simulation), sumo_model_args=sumo_model_args(config.simulation),
        seed=123, network_sha256=sha256(prepared.network_xml), registry_sha256=sha256(prepared.edge_registry),
        simulation_parameters=vars(config.simulation), edge_ids=list(network.edge_ids),
        sumo_extra_args=list(config.simulation.sumo_extra_args),
        code_sha256=sha256(__file__), seconds=seconds, release_at=release_at,
        control='real fixed red/green turn policy; shadow consumes exact real controls and audits every physical entry',
        speed='boundary previous-step vs retrospective realized next-step road mean; negative empty-road API values floored at zero',
        reset='each shadow step starts from true active vehicle snapshot; no future births simulated',
        cost_limit='one-second exposure equals active-at-start population by shared observation convention; this cannot establish multistep cost accuracy',
        conclusion_limit='oracle is a road mean, not individual trajectory velocity; toy evidence does not estimate city-wide error')
    protocol['sumo_warnings'] = ((output/'sumo_warnings.log').read_text()
        if (output/'sumo_warnings.log').exists() else '')
    atomic_json(output/'protocol.json', protocol)
    report = dict(status='complete', arms=summary, stopped_front_moving_behind_witnesses=witnesses,
                  actual_arrivals={key: state.get('arrival_time') for key, state in actual.items()})
    atomic_json(output/'summary.json', report)
    return report


def replay_constraints(source, output):
    """Pair baseline/opt-in on the identical archived 69 real one-second states."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('new output directory required')
    torch.set_num_threads(2)
    source_protocol = json.loads((source/'protocol.json').read_text())
    if sha256(source/'combined_queue.net.xml') != source_protocol['network_sha256']:
        raise ValueError('audit network changed')
    actual = {row['timestamp']: row for row in map(json.loads, (source/'real_seconds.jsonl').read_text().splitlines())}
    base = ROOT/'networks/toy_cross_bottleneck'
    prepared = SimpleNamespace(network_xml=source/'combined_queue.net.xml', network_agents_csv=base/'agents.csv',
        edge_registry=base/'edge_registry.json', od_file=base/'od_toy_cross_bottleneck.txt')
    topology = load_traffic_topology(prepared)
    network = topology.network
    if list(network.edge_ids) != source_protocol['edge_ids']:
        raise ValueError('road registry differs')
    signal_base = StaticSignalSchedule.from_network(prepared.network_xml, network.edge_ids,
                                                    no_external_overrides=True)
    release_at = source_protocol['release_at']

    class DeclaredReleaseSchedule:
        def permits(self, current, target, timestamp):
            # This intervention time was fixed before the original real run.
            return timestamp >= release_at or signal_base.permits(current, target, timestamp)

    rows, windows = [], []
    for t in sorted(actual):
        if t+1 not in actual:
            continue
        before = [state for state in actual[t]['states'] if state['departed'] and not state['arrived']]
        after = {state['vehicle_id']: state for state in actual[t+1]['states']}
        if not before:
            continue
        controls = [NextHopDecision(**row) for row in actual[t+1]['controls']]
        routes = actual[t+1]['commanded_route_suffixes']
        for speed_arm, speed_row in [('boundary_last_step', actual[t]), ('realized_step_mean', actual[t+1])]:
            speeds = np.maximum(0., speed_row['edge_mean_speed_mps'])
            for implementation in ('baseline', 'constrained'):
                constraint = MesoReleaseConstraint(DeclaredReleaseSchedule()) if implementation == 'constrained' else None
                try:
                    states, result, entries = shadow_step(before,
                        candidates=NextHopCandidates(network, traffic_model='meso'), speeds=speeds,
                        flow=np.zeros((network.edge_count, 8)), timestamp=t, population=2,
                        decisions=controls, routes=routes,
                        max_speed=source_protocol['simulation_parameters']['vehicle_max_speed_mps'],
                        passage_constraint=constraint)
                except (RouteMismatch, UnsupportedConstraint) as error:
                    windows.append(dict(timestamp=t, speed_arm=speed_arm, implementation=implementation,
                        route_match=False, error=str(error)))
                    continue
                windows.append(dict(timestamp=t, speed_arm=speed_arm, implementation=implementation,
                    route_match=True, entries=entries, active_seconds_error=result.active_vehicle_seconds-len(before),
                    blocked_signal=0 if constraint is None else constraint.blocked_signal,
                    blocked_fifo=0 if constraint is None else constraint.blocked_fifo))
                for initial, predicted in zip(before, states):
                    truth = after[initial['vehicle_id']]
                    same = predicted['current_edge'] == truth['current_edge']
                    compare_position = same and not predicted['arrived'] and not truth['arrived']
                    rows.append(dict(timestamp=t, speed_arm=speed_arm, implementation=implementation,
                        vehicle_id=initial['vehicle_id'], road_match=same,
                        true_road=network.edge_ids[truth['current_edge']], predicted_road=network.edge_ids[predicted['current_edge']],
                        arrival_match=predicted['arrived'] == truth['arrived'],
                        true_remaining=truth['remaining_distance'], predicted_remaining=predicted['remaining_distance'],
                        position_error_m=predicted['remaining_distance']-truth['remaining_distance'] if compare_position else None,
                        red_hold_violation=(t < release_at and network.edge_ids[truth['current_edge']] == 'A_hold'
                            and network.edge_ids[predicted['current_edge']] != 'A_hold')))
    groups = defaultdict(list)
    for row in rows:
        groups[(row['speed_arm'], row['implementation'])].append(row)
    summary = []
    paired_positions = defaultdict(dict)
    for row in rows:
        paired_positions[(row['speed_arm'], row['timestamp'], row['vehicle_id'])][row['implementation']] = row['position_error_m']
    for (arm, implementation), selected in groups.items():
        trials = [row for row in windows if row['speed_arm'] == arm and row['implementation'] == implementation]
        errors = [abs(row['position_error_m']) for row in selected if row['position_error_m'] is not None]
        common = [abs(pair[implementation]) for key, pair in paired_positions.items()
                  if key[0] == arm and pair.get('baseline') is not None and pair.get('constrained') is not None]
        summary.append(dict(speed_arm=arm, implementation=implementation, steps=len(trials),
            route_matched_steps=sum(row['route_match'] for row in trials), vehicle_steps=len(selected),
            road_mismatches=sum(not row['road_match'] for row in selected),
            red_hold_violations=sum(row['red_hold_violation'] for row in selected),
            arrival_mismatches=sum(not row['arrival_match'] for row in selected),
            position_valid_steps=len(errors), position_mae_m=float(np.mean(errors)) if errors else None,
            position_max_abs_m=max(errors) if errors else None,
            paired_position_steps=len(common), paired_position_mae_m=float(np.mean(common)) if common else None,
            active_seconds_max_abs=max((abs(row['active_seconds_error']) for row in trials if row['route_match']), default=None)))
    output.mkdir(parents=True)
    for name, records in [('rows.jsonl', rows), ('windows.jsonl', windows)]:
        (output/name).write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in records))
    protocol = dict(schema_version=1, source=str(source.resolve()), source_protocol_sha256=sha256(source/'protocol.json'),
        source_real_seconds_sha256=sha256(source/'real_seconds.jsonl'), source_physics=source_protocol['physics'],
        source_sumo_version=source_protocol['sumo_version'], network_sha256=source_protocol['network_sha256'],
        code_hashes={path: sha256(ROOT/path) for path in ('tools/diagnose_component_executor.py',
            'urban_dreamer/learning/imagination.py', 'urban_dreamer/learning/meso_constraints.py')},
        mechanism='opt-in source stop-line signal permission plus edge FIFO; no headway/downstream/merge arbitration',
        signals='static toy net plus predeclared all-green intervention at source release_at; no future vehicle truth',
        pairing='same real starting states, accepted routes, actual controls and road means; every second resets to truth',
        position_pairing='paired_position_* uses the common subset where BOTH implementations and truth share a road; road mismatches separately cover all trials',
        limitations='single-step controlled toy; cost/arrival matches do not establish city-scale or multistep accuracy')
    atomic_json(output/'protocol.json', protocol)
    report = dict(status='complete', evaluations=summary)
    atomic_json(output/'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=70)
    parser.add_argument('--release-at', type=int, default=40)
    parser.add_argument('--replay-source', type=Path)
    args = parser.parse_args()
    result = (replay_constraints(args.replay_source, args.output_dir) if args.replay_source else
              run_audit(args.output_dir, seconds=args.seconds, release_at=args.release_at))
    print(json.dumps(result, indent=2))
