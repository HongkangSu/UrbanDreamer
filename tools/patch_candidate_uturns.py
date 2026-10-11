"""Randomize short candidate roads and remove reciprocal passenger turnarounds."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_maule_nangis import registry, sha
from urban_dreamer.traffic.topology import (
    _lane_allows_passenger, _load_edge_network, _load_connections, _load_od_edges,
    NextHopCandidates,
)

CITIES = ('coulommiers', 'montereau_fault_yonne')


def external_edges(root):
    return {edge.get('id'): edge for edge in root.findall('edge')
            if not edge.get('id', '').startswith(':') and edge.get('function') != 'internal'}


def reciprocal_turn_pairs(root):
    edges = external_edges(root)
    lanes = {(name, lane.get('index')): lane for name, edge in edges.items()
             for lane in edge.findall('lane')}
    turns = set()
    for connection in root.findall('connection'):
        a, b = connection.get('from'), connection.get('to')
        source = lanes.get((a, connection.get('fromLane')))
        target = lanes.get((b, connection.get('toLane')))
        if (a != b and connection.get('dir') == 't' and source is not None and target is not None
                and _lane_allows_passenger(source) and _lane_allows_passenger(target)):
            turns.add((a, b))
    return {tuple(sorted((a, b))) for a, b in turns if (b, a) in turns}


def randomize_short_lanes(root, city, seed=42):
    changes = []
    edges = external_edges(root)
    groups = {name:name for name in edges}
    for name, edge in edges.items():
        partner = edge.get('bidi')
        if partner in edges:
            groups[name] = groups[partner] = min(name, partner)
    for name, edge in sorted(edges.items()):
        digest = hashlib.sha256(f'{seed}:{city}:{groups[name]}'.encode()).digest()
        fraction = int.from_bytes(digest[:8], 'big') / 2**64
        length = min(14.999, max(10.001, round(10. + 5. * fraction, 3)))
        for lane in edge.findall('lane'):
            old = float(lane.get('length'))
            if not math.isfinite(old) or old <= 0:
                raise ValueError(f'invalid lane length {lane.get("id")}={old}')
            if old <= 10.:
                lane.set('length', f'{length:.3f}')
                changes.append(dict(edge=name, lane=lane.get('id'), old_length_m=old, new_length_m=length))
    return changes


def harmonize_bidi_lengths(root):
    """SUMO bidi-bound lanes represent one track and must have equal lengths."""
    edges = external_edges(root)
    pairs = {tuple(sorted((name, edge.get('bidi')))) for name, edge in edges.items()
             if edge.get('bidi') in edges}
    changes = []
    for a, b in sorted(pairs):
        left = {lane.get('index'):lane for lane in edges[a].findall('lane')}
        right = {lane.get('index'):lane for lane in edges[b].findall('lane')}
        if left.keys() != right.keys():
            raise ValueError(f'cannot align bidi lane indices: {a}, {b}')
        for index, reference in left.items():
            lane = right[index]
            if float(lane.get('length')) != float(reference.get('length')):
                changes.append(dict(edge=b, lane=lane.get('id'), bidi_edge=a,
                    old_length_m=float(lane.get('length')), new_length_m=float(reference.get('length'))))
                lane.set('length', reference.get('length'))
    return changes


def connection_keys(root):
    edges = external_edges(root)
    return {(c.get('from'), c.get('to'), c.get('fromLane'), c.get('toLane'))
            for c in root.findall('connection') if c.get('from') in edges and c.get('to') in edges}


def build_patch(original, destination, city, work, netconvert, seed=42):
    original_hash = sha(original)
    old = ET.parse(original).getroot()
    pairs = reciprocal_turn_pairs(old)
    forbidden = {(a, b) for a, b in pairs} | {(b, a) for a, b in pairs}
    patch = ET.Element('connections')
    for a, b in sorted(forbidden):
        ET.SubElement(patch, 'delete', {'from': a, 'to': b})
    delete_file = work/'remove_turnarounds.con.xml'
    ET.ElementTree(patch).write(delete_file, encoding='utf-8', xml_declaration=True)
    converted = work/'converted.net.xml'
    command = [str(netconvert), '--sumo-net-file', str(original), '--connection-files', str(delete_file),
               '--output-file', str(converted), '--offset.disable-normalization', 'true',
               '--no-warnings', 'true']
    with (work/'netconvert.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    new = ET.parse(converted).getroot()
    before, after = external_edges(old), external_edges(new)
    if before.keys() != after.keys():
        raise ValueError('netconvert changed external road IDs')
    expected = {key for key in connection_keys(old) if key[:2] not in forbidden}
    if connection_keys(new) != expected:
        raise ValueError(f'unexpected connection edits: added={len(connection_keys(new)-expected)}, missing={len(expected-connection_keys(new))}')
    # netconvert owns internal links, junction requests and signal link indices.
    # Preserve every external road/lane attribute before the requested length edit.
    for name, edge in after.items():
        edge.attrib.clear(); edge.attrib.update(before[name].attrib)
        old_lanes = {lane.get('id'): lane for lane in before[name].findall('lane')}
        new_lanes = {lane.get('id'): lane for lane in edge.findall('lane')}
        if old_lanes.keys() != new_lanes.keys():
            raise ValueError(f'netconvert changed lanes of {name}')
        for key, lane in new_lanes.items():
            lane.attrib.clear(); lane.attrib.update(old_lanes[key].attrib)
    changes = randomize_short_lanes(new, city, seed)
    bidi_changes = harmonize_bidi_lengths(new)
    assert not reciprocal_turn_pairs(new)
    ET.ElementTree(new).write(destination, encoding='utf-8', xml_declaration=True)
    return dict(seed=seed, random_rule='10+5*SHA256(seed:city:edge-group) fraction; bidi partners share canonical edge ID; 3 decimals, clipped10.001..14.999m',
                original_sha256=original_hash,
                patched_sha256=sha(destination),
                removed_road_pairs=sorted(pairs), removed_directional_turns=len(forbidden),
                removed_lane_connections=len(connection_keys(old)-connection_keys(new)),
                short_lane_changes=changes, patched_edges=len({row['edge'] for row in changes}),
                bidi_length_changes=bidi_changes,
                patched_lanes=len(changes), remaining_reciprocal_turn_pairs=0, netconvert_command=command)


def check_od(directory, city):
    graph = _load_connections(_load_edge_network(directory/'edge_registry.json'), directory/f'{city}.net.xml')
    origins, destinations = _load_od_edges(directory/f'od_{city}.txt')
    with (directory/'agents.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter((int(row['origin']), int(row['destination'])) for row in rows)
    candidates = NextHopCandidates(graph, traffic_model='meso')
    unreachable = []
    for (origin, destination), count in sorted(counts.items()):
        a = graph.edge_to_index[origins[origin]]; b = graph.edge_to_index[destinations[destination]]
        if not math.isfinite(candidates.distances(b)[a]):
            unreachable.append(dict(origin_index=origin, destination_index=destination,
                origin_edge=origins[origin], destination_edge=destinations[destination], expected_vehicles=count))
    return dict(total_od_pairs=len(counts), expected_vehicles=len(rows), unreachable_od_pairs=unreachable,
                affected_expected_vehicles=sum(r['expected_vehicles'] for r in unreachable))


def drop_unreachable_od(directory, city):
    """Delete every demand row of an unreachable OD; do not reweight survivors."""
    audit = check_od(directory, city)
    forbidden = {(r['origin_index'], r['destination_index']) for r in audit['unreachable_od_pairs']}
    path = directory/'agents.csv'
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    removed = [r for r in rows if (int(r['origin']), int(r['destination'])) in forbidden]
    kept = [r for r in rows if (int(r['origin']), int(r['destination'])) not in forbidden]
    if not kept:
        raise ValueError('OD deletion would leave an empty demand dataset')
    if removed:
        for target, values in ((directory/'removed_unreachable_agents.csv', removed), (path, kept)):
            temporary = target.with_suffix('.tmp')
            with temporary.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(values)
            temporary.replace(target)
    result = dict(before_expected_vehicles=len(rows), remaining_expected_vehicles=len(kept),
                  removed_expected_vehicles=len(removed), removed_od_pairs=audit['unreachable_od_pairs'],
                  remaining_od_pairs=len({(r['origin'], r['destination']) for r in kept}))
    (directory/'removed_unreachable_od.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def pin_historical_links(backup_root):
    """Keep old source snapshots that linked to the active candidate inputs intact."""
    pinned = []
    for pattern in ('*/source/networks', '*/*/source/networks', '*/*/*/source/networks'):
        for link in (ROOT/'outputs').glob(pattern):
            if not link.is_symlink() or link.resolve() != ROOT/'networks':
                continue
            link.unlink(); link.mkdir()
            for child in (ROOT/'networks').iterdir():
                target = backup_root/child.name if child.name in CITIES else child.resolve()
                (link/child.name).symlink_to(target, target_is_directory=child.is_dir())
            pinned.append(str(link))
    for link in (ROOT/'outputs').glob('*/load_probes/*/inputs/*'):
        if not link.is_symlink():
            continue
        target = link.resolve()
        for city in CITIES:
            if target.parent == ROOT/'networks'/city and (backup_root/city/target.name).exists():
                link.unlink(); link.symlink_to(backup_root/city/target.name)
                pinned.append(str(link))
    return pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--drop-unreachable-od', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    netconvert = Path(sys.executable).parent/'netconvert'
    report = dict(applied=False, seed=args.seed, networks={})
    for city in CITIES:
        active = ROOT/'networks'/city; backup = out/'original_inputs'/city; work = out/'prepared'/city
        backup.mkdir(parents=True); work.mkdir(parents=True)
        for name in (f'{city}.net.xml', 'edge_registry.json', 'agents.csv', f'od_{city}.txt'):
            shutil.copy2(active/name, backup/name)
        for name in ('agents.csv', f'od_{city}.txt'):
            shutil.copy2(active/name, work/name)
        result = build_patch(backup/f'{city}.net.xml', work/f'{city}.net.xml', city, work, netconvert, args.seed)
        (work/'edge_registry.json').write_text(json.dumps(registry(work/f'{city}.net.xml', city), separators=(',', ':'))+'\n')
        result.update(check_od(work, city))
        if args.drop_unreachable_od:
            result['od_deletion'] = drop_unreachable_od(work, city)
            result['post_deletion_audit'] = check_od(work, city)
        result['input_hashes_before'] = {p.name:sha(p) for p in backup.iterdir()}
        result['input_hashes_after'] = {name:sha(work/name) for name in result['input_hashes_before']}
        report['networks'][city] = result
        (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(city=city, patched_edges=result['patched_edges'], removed_pairs=len(result['removed_road_pairs']),
            unreachable_pairs=len(result['unreachable_od_pairs']), affected_vehicles=result['affected_expected_vehicles'])), flush=True)
    if args.apply:
        report['pinned_historical_links'] = pin_historical_links(out/'original_inputs')
        for city in CITIES:
            for name in (f'{city}.net.xml', 'edge_registry.json', 'agents.csv'):
                target = ROOT/'networks'/city/name
                temporary = target.with_name('.'+name+'.patch.tmp')
                shutil.copy2(out/'prepared'/city/name, temporary)
                temporary.replace(target)
        report['applied'] = True
        (out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
