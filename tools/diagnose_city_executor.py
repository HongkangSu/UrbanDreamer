"""Replay immutable real toy seconds through three route-audited meso executors.

This is a paired one-step diagnostic, not multistep or city-scale validation.
The retrospective speed arm intentionally uses the realized next-step road mean;
no executor receives future vehicle states, future departures or observed TLS.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.component_experiment_data import atomic_json, sha256
from tools.diagnose_component_executor import ExactStepRoute, RouteMismatch, shadow_step
from urban_dreamer.learning import imagination as physical_engine
from urban_dreamer.learning.city_imagination import ChronologicalJunctionArbiter, advance_city_imagined_window
from urban_dreamer.learning.meso_constraints import MesoReleaseConstraint, StaticSignalSchedule
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.movement_controls import CityControlTable
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology

IMPLEMENTATIONS = ('legacy', 'simple_gate', 'chronological_city')
SPEED_ARMS = ('boundary_last_step', 'realized_step_mean')


def resolve_declared_override(source_network, destination, *, release_at, horizon):
    """Materialize the fixture's previously declared A2 intervention, not truth.

    The resulting static XML is valid over this replay horizon only. The long
    final green phase exceeds that horizon, so its eventual cyclic wrap is not
    represented as an indefinitely valid schedule.
    """
    if not 0 < release_at < horizon or int(release_at) != release_at:
        raise ValueError('declared release must be integral and inside replay horizon')
    tree = ET.parse(source_network)
    programs = [p for p in tree.getroot().findall('tlLogic') if p.get('id') == 'A2']
    if len(programs) != 1:
        raise ValueError('fixture requires exactly one declared A2 program')
    program = programs[0]
    phases = program.findall('phase')
    if (program.get('type') != 'static' or float(program.get('offset', '0')) != 0.
            or len(phases) != 1 or float(phases[0].get('duration')) <= horizon):
        raise ValueError('fixture baseline must be one constant static phase over the replay horizon')
    phase = phases[0]
    initial = phase.attrib['state']
    phase.set('duration', str(int(release_at)))
    ET.SubElement(program, 'phase', duration=str(int(horizon) + 1), state='G' * len(initial))
    tree.write(destination)
    return dict(tls_id='A2', effective_at=float(release_at), old_state=initial,
                new_state='G' * len(initial), valid_through=float(horizon),
                authority='immutable source protocol.release_at; intervention fixed before real run',
                resolved_control_network_sha256=sha256(destination))


def city_shadow_step(initial_states, *, candidates, speeds, timestamp, population,
                     decisions, routes, max_speed, controls, seed):
    states = deepcopy(initial_states)
    controller = ExactStepRoute(decisions, routes, timestamp)
    arbiter = ChronologicalJunctionArbiter(controls, seed=seed)
    original = physical_engine._record_physical_entry

    def checked_entry(state, edge, choices):
        controller.entry(state, edge)
        original(state, edge, choices)

    try:
        physical_engine._record_physical_entry = checked_entry
        result = advance_city_imagined_window(states, candidates=candidates,
            edge_speeds_mps=np.asarray(speeds), frozen_flow=np.zeros((candidates.network.edge_count, 8)),
            start_time=timestamp, total_vehicle_count=population, decide=controller,
            interval_seconds=1., vehicle_max_speed_mps=max_speed, junction_arbiter=arbiter)
        if controller.used != set(controller.decisions):
            raise RouteMismatch('not all real controls were applied in the city shadow step')
    finally:
        physical_engine._record_physical_entry = original
    return states, result, controller.entries, arbiter.audit


def summarize(rows, windows):
    paired = defaultdict(dict)
    for row in rows:
        paired[row['speed_arm'], row['timestamp'], row['vehicle_id']][row['implementation']] = row
    summary = []
    for arm in SPEED_ARMS:
        for implementation in IMPLEMENTATIONS:
            selected = [r for r in rows if r['speed_arm'] == arm and r['implementation'] == implementation]
            trials = [r for r in windows if r['speed_arm'] == arm and r['implementation'] == implementation]
            successes = [r for r in selected if r['status'] == 'ok']
            errors = [abs(r['position_error_m']) for r in successes if r['position_error_m'] is not None]
            common = [abs(group[implementation]['position_error_m']) for key, group in paired.items()
                      if key[0] == arm and all(name in group and group[name]['status'] == 'ok'
                          and group[name]['position_error_m'] is not None for name in IMPLEMENTATIONS)]
            summary.append(dict(speed_arm=arm, implementation=implementation, attempted_steps=len(trials),
                route_matched_steps=sum(r['status'] == 'ok' for r in trials),
                unsupported_steps=sum(r['status'] == 'unsupported' for r in trials),
                error_steps=sum(r['status'] == 'error' for r in trials),
                attempted_vehicle_steps=len(selected), evaluated_vehicle_steps=len(successes),
                road_mismatches=sum(not r['road_match'] for r in successes),
                arrival_mismatches=sum(not r['arrival_match'] for r in successes),
                red_hold_violations=sum(r['red_hold_violation'] for r in successes),
                position_valid_steps=len(errors), position_mae_m=float(np.mean(errors)) if errors else None,
                position_max_abs_m=max(errors) if errors else None,
                common_position_steps=len(common), common_position_mae_m=float(np.mean(common)) if common else None,
                active_seconds_max_abs=max((abs(r['active_seconds_error']) for r in trials if r['status'] == 'ok'), default=None)))
    return summary


def replay_audit(source, output, *, seed=123):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('new output directory required; preserve previous audits')
    names = ('protocol.json', 'real_seconds.jsonl', 'combined_queue.net.xml')
    source_hashes = {name: sha256(source / name) for name in names}
    protocol = json.loads((source / 'protocol.json').read_text())
    if source_hashes['combined_queue.net.xml'] != protocol['network_sha256']:
        raise ValueError('immutable source network hash mismatch')
    records = [json.loads(line) for line in (source / 'real_seconds.jsonl').read_text().splitlines()]
    actual = {row['timestamp']: row for row in records}
    if len(actual) != len(records) or sorted(actual) != list(range(1, protocol['seconds'] + 1)):
        raise ValueError('source requires every unique real second from 1 through declared horizon')
    base = ROOT / 'networks/toy_cross_bottleneck'
    if sha256(base / 'edge_registry.json') != protocol['registry_sha256']:
        raise ValueError('immutable edge registry hash mismatch')
    prepared = SimpleNamespace(network_xml=source / 'combined_queue.net.xml', network_agents_csv=base / 'agents.csv',
        edge_registry=base / 'edge_registry.json', od_file=base / 'od_toy_cross_bottleneck.txt')
    topology = load_traffic_topology(prepared)
    network = topology.network
    if list(network.edge_ids) != protocol['edge_ids']:
        raise ValueError('road registry order mismatch')
    output.mkdir(parents=True)
    resolved = output / 'declared_control_program.net.xml'
    override = resolve_declared_override(prepared.network_xml, resolved,
        release_at=protocol['release_at'], horizon=protocol['seconds'])
    # This resolved program already contains the one declared intervention.
    # No unrepresented external override remains for either static parser.
    controls = CityControlTable.from_network(resolved, network.edge_ids, no_external_overrides=True)
    signals = StaticSignalSchedule.from_network(resolved, network.edge_ids, no_external_overrides=True)
    candidates = NextHopCandidates(network, traffic_model='meso')
    population = len(actual[min(actual)]['states'])
    windows, rows = [], []
    for timestamp in sorted(actual)[:-1]:
        before = [s for s in actual[timestamp]['states'] if s['departed'] and not s['arrived']]
        truth = {s['vehicle_id']: s for s in actual[timestamp + 1]['states']}
        # These commands were issued at the start of this second and stored in
        # its completed record. They are accepted controls, not future actions.
        decisions = [NextHopDecision(**r) for r in actual[timestamp + 1]['controls']]
        routes = actual[timestamp + 1]['commanded_route_suffixes']
        for state in before:
            if not routes.get(state['vehicle_id']) or routes[state['vehicle_id']][0] != state['current_edge']:
                raise ValueError('accepted route cursor disagrees with initial real snapshot')
        for arm, speed_record in ((SPEED_ARMS[0], actual[timestamp]), (SPEED_ARMS[1], actual[timestamp + 1])):
            speeds = np.maximum(0., speed_record['edge_mean_speed_mps'])
            for implementation in IMPLEMENTATIONS:
                trial = dict(timestamp=timestamp, speed_arm=arm, implementation=implementation,
                    cohort=len(before), route_match=False, status='ok')
                predictions = {}
                try:
                    if implementation == 'chronological_city':
                        states, result, entries, audit = city_shadow_step(before, candidates=candidates,
                            speeds=speeds, timestamp=timestamp, population=population,
                            decisions=decisions, routes=routes,
                            max_speed=protocol['simulation_parameters']['vehicle_max_speed_mps'], controls=controls, seed=seed)
                    else:
                        gate = MesoReleaseConstraint(signals) if implementation == 'simple_gate' else None
                        states, result, entries = shadow_step(before, candidates=candidates, speeds=speeds,
                            flow=np.zeros((network.edge_count, 8)), timestamp=timestamp, population=population,
                            decisions=decisions, routes=routes,
                            max_speed=protocol['simulation_parameters']['vehicle_max_speed_mps'], passage_constraint=gate)
                        audit = []
                    predictions = {s['vehicle_id']: s for s in states}
                    trial.update(route_match=True, entries=entries, audit=audit,
                        active_seconds_error=result.active_vehicle_seconds - len(before),
                        predicted_arrivals=result.arrivals, end_time=result.end_time)
                except Exception as error:
                    # Keep every attempted row, including unexpected execution
                    # errors. A failed arm never disappears from denominators.
                    trial.update(status='unsupported' if type(error).__name__ in {'UnsupportedControl', 'UnsupportedConstraint'} else 'error',
                                 error_type=type(error).__name__, error=str(error))
                windows.append(trial)
                for initial in before:
                    name = initial['vehicle_id']
                    actual_state, predicted = truth[name], predictions.get(name)
                    row = dict(timestamp=timestamp, speed_arm=arm, implementation=implementation,
                        vehicle_id=name, status=trial['status'], error=trial.get('error'),
                        road_before=network.edge_ids[initial['current_edge']], remaining_before=initial['remaining_distance'],
                        true_road=network.edge_ids[actual_state['current_edge']], true_remaining=actual_state['remaining_distance'],
                        true_arrived=actual_state['arrived'], predicted_road=None, predicted_remaining=None,
                        predicted_arrived=None, road_match=None, arrival_match=None, position_error_m=None, red_hold_violation=None)
                    if predicted is not None:
                        same = predicted['current_edge'] == actual_state['current_edge']
                        valid_position = same and not predicted['arrived'] and not actual_state['arrived']
                        row.update(predicted_road=network.edge_ids[predicted['current_edge']],
                            predicted_remaining=predicted['remaining_distance'], predicted_arrived=predicted['arrived'],
                            road_match=same, arrival_match=predicted['arrived'] == actual_state['arrived'],
                            position_error_m=predicted['remaining_distance'] - actual_state['remaining_distance'] if valid_position else None,
                            red_hold_violation=timestamp < protocol['release_at'] and row['true_road'] == 'A_hold'
                                and network.edge_ids[predicted['current_edge']] != 'A_hold')
                    rows.append(row)
    for name, records in (('rows.jsonl', rows), ('windows.jsonl', windows)):
        (output / name).write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    if source_hashes != {name: sha256(source / name) for name in names}:
        raise RuntimeError('immutable source changed during paired replay')
    files = ('tools/diagnose_city_executor.py', 'tools/diagnose_component_executor.py',
        'urban_dreamer/learning/imagination.py', 'urban_dreamer/learning/city_imagination.py',
        'urban_dreamer/learning/meso_constraints.py', 'urban_dreamer/traffic/movement_controls.py')
    audit_protocol = dict(schema_version=1, source=str(source.resolve()), source_hashes=source_hashes,
        source_physics=protocol['physics'], source_sumo_version=protocol['sumo_version'],
        code_hashes={name: sha256(ROOT / name) for name in files}, seed=seed,
        declared_signal_override=override, implementations=IMPLEMENTATIONS, speed_arms=SPEED_ARMS,
        pairing='same real active states, accepted start-of-step controls/routes and identical road-speed vectors for all arms',
        reset=f'each of {len(actual) - 1} adjacent archived one-second pairs independently resets to real active snapshot; no future births',
        queue_initialization='spatial approximation; equal positions fail explicitly; no invented authoritative order',
        chronological_assumptions=dict(discharge_headway_seconds=1., receiving_headway_seconds=1., conflict_clearance_seconds=1.,
            initial_calendar='unknown historical headways and clearances; unreserved at each independent real seed'),
        speed_information='boundary_last_step causal; realized_step_mean retrospective oracle road mean, not individual speeds',
        position_pairing='common_position uses identical vehicle/second subset valid in ALL THREE implementations; all failures retained',
        limitations='toy one-step audit; exposure equality follows observation accounting and does not validate multistep travel costs or SUMO capacity physics')
    atomic_json(output / 'protocol.json', audit_protocol)
    atomic_json(output / 'control_preflight.json', controls.preflight())
    report = dict(status='complete' if all(r['status'] == 'ok' for r in windows) else 'complete_with_failures',
        source_pairs=len(actual) - 1, attempted_windows=len(windows), attempted_vehicle_rows=len(rows),
        evaluations=summarize(rows, windows), errors=[r for r in windows if r['status'] != 'ok'])
    atomic_json(output / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    print(json.dumps(replay_audit(args.replay_source, args.output_dir, seed=args.seed), indent=2))
