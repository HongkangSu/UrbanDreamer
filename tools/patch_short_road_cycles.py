"""Prepare OD-preserving turn restrictions that break short passenger-road cycles.

Preparation writes a new directory only. Publishing network inputs is a separate
operation so running experiments can retain their original topology snapshots.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.patch_candidate_uturns import (CITIES, check_od, connection_keys,
                                         external_edges, reciprocal_turn_pairs)
from tools.prepare_maule_nangis import registry, sha
from urban_dreamer.traffic.topology import _lane_allows_passenger, _load_od_edges


def passenger_graph(root):
    """Only actual connections between passenger-permitted lanes form arcs."""
    edges = external_edges(root)
    lanes = {(name, lane.get('index')): lane for name, edge in edges.items()
             for lane in edge.findall('lane')}
    graph = nx.DiGraph()
    lengths = {}
    for name, edge in sorted(edges.items()):
        permitted = [lane for lane in edge.findall('lane') if _lane_allows_passenger(lane)]
        if permitted:
            graph.add_node(name)
            # An edge is short only if every passenger lane is short.
            lengths[name] = max(float(lane.get('length')) for lane in permitted)
    directions = defaultdict(set)
    for connection in root.findall('connection'):
        a, b = connection.get('from'), connection.get('to')
        source = lanes.get((a, connection.get('fromLane')))
        target = lanes.get((b, connection.get('toLane')))
        if (source is not None and target is not None
                and _lane_allows_passenger(source) and _lane_allows_passenger(target)):
            graph.add_edge(a, b)
            directions[a, b].add(connection.get('dir', 's'))
    return graph, lengths, directions


def active_od(directory, city):
    origins, destinations = _load_od_edges(directory / f'od_{city}.txt')
    with (directory / 'agents.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    pairs = {(origins[int(row['origin'])], destinations[int(row['destination'])]) for row in rows}
    return sorted(pairs), len(rows)


def unreachable_od(graph, od_pairs):
    """Group reverse searches by destination instead of per demand row."""
    grouped = defaultdict(set)
    for origin, destination in od_pairs:
        grouped[destination].add(origin)
    unreachable = []
    for destination, origins in sorted(grouped.items()):
        ancestors = nx.ancestors(graph, destination) | {destination} if destination in graph else set()
        unreachable.extend((origin, destination) for origin in sorted(origins - ancestors))
    return unreachable


def short_cycles(graph, lengths, threshold, max_edges=None):
    subgraph = graph.subgraph(sorted(name for name, length in lengths.items() if length <= threshold))
    # Canonical rotation makes reports independent of networkx enumeration order.
    cycles = []
    for cycle in nx.simple_cycles(subgraph, length_bound=max_edges):
        start = cycle.index(min(cycle))
        cycles.append(tuple(cycle[start:] + cycle[:start]))
    return sorted(cycles)


def cyclic_components(graph, lengths, threshold):
    subgraph = graph.subgraph(sorted(name for name, length in lengths.items() if length <= threshold))
    return sorted(sorted(component) for component in nx.strongly_connected_components(subgraph)
                  if len(component) > 1 or any(subgraph.has_edge(name, name) for name in component))


def cycle_pairs(cycle):
    return set(zip(cycle, cycle[1:] + cycle[:1]))


def plan_removals(root, od_pairs, short_threshold=15., local_threshold=30., max_cycle_edges=5):
    graph, lengths, directions = passenger_graph(root)
    unreachable = unreachable_od(graph, od_pairs)
    if unreachable:
        raise ValueError(f'input contains unreachable active OD pairs: {unreachable[:5]}')
    blocked, removed, stages = set(), [], []
    for threshold, bound in ((short_threshold, None), (local_threshold, max_cycle_edges)):
        components_before = cyclic_components(graph, lengths, threshold)
        cycles = short_cycles(graph, lengths, threshold, bound)
        before_count, before_removed = len(cycles), len(removed)
        while cycles:
            counts = Counter(pair for cycle in cycles for pair in cycle_pairs(cycle))
            candidates = sorted(counts, key=lambda pair: (
                -counts[pair], 0 if 't' in directions[pair] else 1,
                1 if 's' in directions[pair] else 0, pair))
            accepted = False
            for a, b in candidates:
                if (a, b) in blocked:
                    continue
                graph.remove_edge(a, b)
                redundant = nx.has_path(graph, a, b)
                # An alternate a->b path preserves all graph reachability. If
                # absent, preserving the actual active demands is sufficient.
                if not redundant and unreachable_od(graph, od_pairs):
                    graph.add_edge(a, b)
                    blocked.add((a, b))
                    continue
                removed.append(dict(source=a, target=b, threshold_m=threshold,
                    cycles_hit=counts[a, b], directions=sorted(directions[a, b]),
                    globally_redundant=redundant))
                cycles = [cycle for cycle in cycles if (a, b) not in cycle_pairs(cycle)]
                accepted = True
                break
            if not accepted:
                break
        stages.append(dict(threshold_m=threshold, max_cycle_edges=bound,
            cycles_before=before_count, cycles_remaining=len(cycles),
            remaining_cycles=[list(cycle) for cycle in cycles],
            removed_pairs=len(removed) - before_removed,
            cyclic_components_before=components_before,
            cyclic_components_after=cyclic_components(graph, lengths, threshold)))
    assert not unreachable_od(graph, od_pairs)
    return dict(removed_pairs=removed, rejected_pairs=[list(pair) for pair in sorted(blocked)],
                stages=stages, active_od_pairs=len(set(od_pairs)), all_active_od_reachable=True)


def apply_extra_removals(root, od_pairs, base_plan, extra_removals):
    """Add explicitly evidenced restrictions to a completed base plan.

    All checks use the cumulative graph. A rejected request raises without
    mutating the caller's plan; demand rows are never dropped to make it pass.
    """
    result = deepcopy(base_plan)
    graph, lengths, directions = passenger_graph(root)
    resolved = {(row['source'], row['target']) for row in result['removed_pairs']}
    for source, target in resolved:
        graph.remove_edge(source, target)
    if unreachable_od(graph, od_pairs):
        raise ValueError('base plan contains unreachable active OD pairs')
    result['base_removed_pairs_count'] = len(result['removed_pairs'])
    result['extra_removals'] = []
    if not isinstance(extra_removals, list):
        raise ValueError('extra removals must be a list of connection requests')
    for request in extra_removals:
        if not isinstance(request, dict) or any(not request.get(key) for key in
                                               ('source', 'target', 'reason', 'evidence')):
            raise ValueError('each extra removal requires source, target, reason and evidence')
        source, target = request['source'], request['target']
        if not all(isinstance(name, str) for name in (source, target)):
            raise ValueError('extra removal source and target must be road ID strings')
        pair = (source, target)
        record = dict(source=source, target=target, reason=request['reason'],
                      evidence=deepcopy(request['evidence']), scope='observed_dependency_cycle')
        if pair in resolved:
            result['extra_removals'].append(dict(record, status='already_resolved'))
            continue
        if not graph.has_edge(source, target):
            raise ValueError(f'extra removal is not an actual passenger connection: {source} -> {target}')
        graph.remove_edge(source, target)
        redundant = nx.has_path(graph, source, target)
        disconnected = [] if redundant else unreachable_od(graph, od_pairs)
        if disconnected:
            raise ValueError(f'extra removal disconnects active OD pairs: {source} -> {target}; '
                             f'{disconnected[:5]}')
        removed = dict(record, status='removed', threshold_m=None, cycles_hit=None,
                       directions=sorted(directions[pair]),
                       globally_redundant=redundant)
        result['removed_pairs'].append(removed)
        result['extra_removals'].append(deepcopy(removed))
        resolved.add(pair)
    for stage in result['stages']:
        cycles = short_cycles(graph, lengths, stage['threshold_m'], stage['max_cycle_edges'])
        stage['cycles_remaining'] = len(cycles)
        stage['remaining_cycles'] = [list(cycle) for cycle in cycles]
        stage['cyclic_components_after'] = cyclic_components(graph, lengths, stage['threshold_m'])
    return result


def signal_signature(root):
    """Compare retained connections' indications even when link indices change."""
    programs = {}
    for logic in root.findall('tlLogic'):
        key = (logic.get('id'), logic.get('programID'))
        attributes = {name: value for name, value in logic.attrib.items() if name != 'id'}
        phases = [{name: value for name, value in phase.attrib.items() if name != 'state'}
                  for phase in logic.findall('phase')]
        parameters = [dict(parameter.attrib) for parameter in logic.findall('param')]
        programs[key] = (attributes, phases, parameters)
    indications = {}
    for connection in root.findall('connection'):
        if connection.get('tl') is None or connection.get('linkIndex') is None:
            continue
        connection_key = tuple(connection.get(key) for key in ('from', 'to', 'fromLane', 'toLane'))
        for logic in root.findall('tlLogic'):
            if logic.get('id') != connection.get('tl'):
                continue
            index = int(connection.get('linkIndex'))
            indications[connection_key, logic.get('id'), logic.get('programID')] = tuple(
                phase.get('state')[index] for phase in logic.findall('phase'))
    return programs, indications


def _phase_duration(phase):
    duration = Decimal(phase['duration'])
    if not duration.is_finite() or duration <= 0:
        raise ValueError('signal phase duration must be finite and positive')
    return duration


def _signal_timeline(phases, indications):
    """Coalesce consecutive equal indications without rotating the cycle."""
    timeline = []
    for phase, indication in zip(phases, indications):
        duration = _phase_duration(phase)
        if timeline and timeline[-1][1] == indication:
            timeline[-1] = (timeline[-1][0] + duration, indication)
        else:
            timeline.append((duration, indication))
    return tuple(timeline)


def validate_signal_equivalence(old, new, retained_connections):
    """Permit static phase compaction only when every retained timeline agrees.

    Exact program attributes preserve the offset/phase origin. Programs with
    additional phase attributes or non-static control retain strict comparison.
    """
    old_programs, old_indications = signal_signature(old)
    new_programs, new_indications = signal_signature(new)
    if old_programs.keys() != new_programs.keys():
        raise ValueError('netconvert changed signal program IDs')
    semantic, compacted = set(), []
    for key, (attributes, phases, parameters) in old_programs.items():
        new_attributes, new_phases, new_parameters = new_programs[key]
        if attributes != new_attributes or parameters != new_parameters:
            raise ValueError('netconvert changed signal program attributes, offset or parameters')
        if sum(map(_phase_duration, phases)) != sum(map(_phase_duration, new_phases)):
            raise ValueError('netconvert changed signal cycle duration')
        if (attributes.get('type') == 'static'
                and all(set(phase) == {'duration'} for phase in phases + new_phases)):
            semantic.add(key)
            if phases != new_phases:
                compacted.append(list(key))
        elif phases != new_phases:
            raise ValueError('netconvert changed signal phases with non-static control or extra attributes')
    retained_old = {key: value for key, value in old_indications.items() if key[0] in retained_connections}
    retained_new = {key: value for key, value in new_indications.items() if key[0] in retained_connections}
    if retained_old.keys() != retained_new.keys():
        raise ValueError('netconvert changed signal assignment for a retained connection')
    for key, indications in retained_old.items():
        program_key = key[1:]
        new_states = retained_new[key]
        if program_key in semantic:
            old_timeline = _signal_timeline(old_programs[program_key][1], indications)
            new_timeline = _signal_timeline(new_programs[program_key][1], new_states)
            equal = old_timeline == new_timeline
        else:
            equal = indications == new_states
        if not equal:
            raise ValueError(f'netconvert changed signal timing or indications for retained connection {key}')
    return dict(retained_signal_connections_checked=len(retained_new),
                equivalent_static_phase_compactions=compacted)


def build_patch(original, destination, work, netconvert, removed_pairs):
    old = ET.parse(original).getroot()
    forbidden = {(row['source'], row['target']) for row in removed_pairs}
    deletes = ET.Element('connections')
    for a, b in sorted(forbidden):
        ET.SubElement(deletes, 'delete', {'from': a, 'to': b})
    delete_path = work / 'remove_short_cycle_connections.con.xml'
    ET.ElementTree(deletes).write(delete_path, encoding='utf-8', xml_declaration=True)
    converted = work / 'converted.net.xml'
    command = [str(netconvert), '--sumo-net-file', str(original),
               '--connection-files', str(delete_path), '--output-file', str(converted),
               '--offset.disable-normalization', 'true', '--no-warnings', 'true']
    with (work / 'netconvert.log').open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    new = ET.parse(converted).getroot()
    before, after = external_edges(old), external_edges(new)
    if before.keys() != after.keys():
        raise ValueError('netconvert changed external road IDs')
    expected = {key for key in connection_keys(old) if key[:2] not in forbidden}
    if connection_keys(new) != expected:
        raise ValueError('netconvert changed connections beyond requested deletions')
    for name, edge in after.items():
        edge.attrib.clear()
        edge.attrib.update(before[name].attrib)
        old_lanes = {lane.get('id'): lane for lane in before[name].findall('lane')}
        new_lanes = {lane.get('id'): lane for lane in edge.findall('lane')}
        if old_lanes.keys() != new_lanes.keys():
            raise ValueError(f'netconvert changed external lanes on {name}')
        for key, lane in new_lanes.items():
            lane.attrib.clear()
            lane.attrib.update(old_lanes[key].attrib)
    signal_audit = validate_signal_equivalence(old, new, expected)
    # External attributes (including every bidi binding and lane length) were
    # copied verbatim; netconvert exclusively rebuilds internal geometry/logic.
    if reciprocal_turn_pairs(new) != reciprocal_turn_pairs(old):
        raise ValueError('reciprocal turnaround set changed unexpectedly')
    ET.ElementTree(new).write(destination, encoding='utf-8', xml_declaration=True)
    return dict(netconvert_command=command, removed_lane_connections=len(connection_keys(old) - expected),
                **signal_audit,
                external_attributes_unchanged=True, reciprocal_turn_pairs_remaining=len(reciprocal_turn_pairs(new)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'networks',
                        help='Source city directories; may point to preserved original_inputs')
    parser.add_argument('--cities', nargs='+', choices=CITIES, default=list(CITIES))
    parser.add_argument('--short-threshold', type=float, default=15.)
    parser.add_argument('--local-threshold', type=float, default=30.)
    parser.add_argument('--max-cycle-edges', type=int, default=5)
    parser.add_argument('--extra-removals', type=Path,
                        help='Evidence-bearing JSON requests: city-to-list mapping, or list for one city')
    parser.add_argument('--netconvert', type=Path, default=Path(sys.executable).parent / 'netconvert')
    args = parser.parse_args()
    if not (0 < args.short_threshold <= args.local_threshold) or args.max_cycle_edges < 2:
        parser.error('thresholds must be positive and ordered; max-cycle-edges must be at least two')
    extra_requests = {}
    if args.extra_removals:
        extra_requests = json.loads(args.extra_removals.read_text())
        if isinstance(extra_requests, list):
            if len(args.cities) != 1:
                parser.error('a bare extra-removals list requires exactly one --cities value')
            extra_requests = {args.cities[0]: extra_requests}
        if not isinstance(extra_requests, dict) or set(extra_requests) - set(CITIES):
            parser.error('extra-removals must map supported city names to request lists')
    output = args.output_dir.resolve()
    input_root = args.input_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(applied=False, rule='Remove connection pairs only; preserve all active OD and physical roads.',
                    input_root=str(input_root),
                    short_threshold_m=args.short_threshold, local_threshold_m=args.local_threshold,
                    max_cycle_edges=args.max_cycle_edges, networks={})
    if args.extra_removals:
        manifest['extra_removals_file'] = str(args.extra_removals.resolve())
        manifest['extra_removals_sha256'] = sha(args.extra_removals)
    for city in args.cities:
        active = input_root / city
        backup = output / 'original_inputs' / city
        prepared = output / 'prepared' / city
        backup.mkdir(parents=True)
        prepared.mkdir(parents=True)
        names = (f'{city}.net.xml', 'edge_registry.json', 'agents.csv', f'od_{city}.txt')
        for name in names:
            shutil.copy2(active / name, backup / name)
        for name in ('agents.csv', f'od_{city}.txt'):
            shutil.copy2(backup / name, prepared / name)
        od_pairs, count = active_od(backup, city)
        root = ET.parse(backup / f'{city}.net.xml').getroot()
        result = plan_removals(root, od_pairs,
                              args.short_threshold, args.local_threshold, args.max_cycle_edges)
        if args.extra_removals:
            result = apply_extra_removals(root, od_pairs, result, extra_requests.get(city, []))
        result.update(build_patch(backup / f'{city}.net.xml', prepared / f'{city}.net.xml',
                                  prepared, args.netconvert, result['removed_pairs']))
        (prepared / 'edge_registry.json').write_text(json.dumps(
            registry(prepared / f'{city}.net.xml', city), separators=(',', ':')) + '\n')
        final_audit = check_od(prepared, city)
        if final_audit['unreachable_od_pairs']:
            raise ValueError(f'prepared SUMO topology made active OD unreachable: {city}')
        result.update(final_audit)
        result['expected_vehicles'] = count
        result['input_hashes_before'] = {name: sha(backup / name) for name in names}
        result['input_hashes_after'] = {name: sha(prepared / name) for name in names}
        for name in ('agents.csv', f'od_{city}.txt'):
            if result['input_hashes_before'][name] != result['input_hashes_after'][name]:
                raise ValueError(f'demand file changed: {name}')
        final_graph, lengths, _ = passenger_graph(ET.parse(prepared / f'{city}.net.xml').getroot())
        for stage in result['stages']:
            actual = short_cycles(final_graph, lengths, stage['threshold_m'], stage['max_cycle_edges'])
            if actual != [tuple(cycle) for cycle in stage['remaining_cycles']]:
                raise ValueError('prepared cycles differ from the planned residual cycles')
        manifest['networks'][city] = result
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(dict(city=city, removed_pairs=len(result['removed_pairs']),
              active_od_pairs=len(od_pairs), expected_vehicles=count,
              residual_cycles=[stage['cycles_remaining'] for stage in result['stages']])), flush=True)


if __name__ == '__main__':
    main()
