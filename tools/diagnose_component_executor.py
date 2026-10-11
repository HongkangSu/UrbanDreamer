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
                decisions, routes, max_speed):
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
            vehicle_max_speed_mps=max_speed, stop_when_empty=False)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=70)
    parser.add_argument('--release-at', type=int, default=40)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.output_dir, seconds=args.seconds, release_at=args.release_at), indent=2))
