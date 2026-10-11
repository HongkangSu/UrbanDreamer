"""Prepare the two unchanged city XMLs with proportional high-demand inputs."""
from __future__ import annotations
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import _lane_allows_passenger, load_traffic_topology
from urban_dreamer.workflow.config import load_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def registry(xml, name):
    root = ET.parse(xml).getroot()
    edges = sorted((e for e in root.findall('edge') if not e.get('id', '').startswith(':') and e.get('function') != 'internal'), key=lambda e: e.get('id'))
    ids = [e.get('id') for e in edges]
    indexes = {name: i for i, name in enumerate(ids)}
    lengths = np.array([float(e.find('lane').get('length')) for e in edges], dtype=np.float32)
    speeds = np.array([float(e.find('lane').get('speed')) for e in edges], dtype=np.float32)
    counts = np.array([len(e.findall('lane')) for e in edges], dtype=np.float32)
    types = sorted({e.get('type', '') for e in edges})
    kinds = np.array([types.index(e.get('type', '')) for e in edges], dtype=np.float32)
    features = np.stack([lengths / max(float(lengths.max()), 1), speeds / max(float(speeds.max()), 1), counts / max(float(counts.max()), 1), kinds / max(len(types)-1, 1)], axis=1)
    nodes = sorted({e.get(k) for e in edges for k in ('from', 'to')})
    node_indexes = {name: i for i, name in enumerate(nodes)}
    allowed = [any(_lane_allows_passenger(lane) for lane in e.findall('lane')) for e in edges]
    adjacency = np.zeros((len(edges), len(edges)), dtype=np.float32)
    lanes = {(e.get('id'), lane.get('index')): lane for e in edges for lane in e.findall('lane')}
    for c in root.findall('connection'):
        a, b = c.get('from'), c.get('to')
        if a in indexes and b in indexes and all(_lane_allows_passenger(lanes[(edge, c.get(field))]) for edge, field in ((a, 'fromLane'), (b, 'toLane'))):
            adjacency[indexes[a], indexes[b]] = 1
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1)
    return dict(network=name, n_edges=len(edges), edge_ids=ids, static_features=features.tolist(), speed_limits=speeds.tolist(), capacities=(counts * 10).astype(int).tolist(), adjacency=adjacency.tolist(), lengths=lengths.tolist(), from_node_idx=[node_indexes[e.get('from')] for e in edges], to_node_idx=[node_indexes[e.get('to')] for e in edges], passenger_allowed=allowed)


def main():
    manifest = {'demand_duration_seconds': 1800, 'departure_seed': 42, 'allocation': 'largest remainder proportional to original OD row counts; ties sorted by integer OD indices', 'runtime_demand': 'CSV counts specify per-OD Poisson expected counts, not fixed realized episode totals', 'registry': 'matching existing schema; float32 normalized length, speed, lane count, sorted type index; capacity = lane count * 10; passenger connection adjacency row-normalized', 'networks': {}}
    for name, count in [('maule', 1800), ('nangis', 2500)]:
        source = ROOT.parent / 'networks' / name
        target = ROOT / 'networks' / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in [f'{name}.net.xml', f'od_{name}.txt']:
            shutil.copyfile(source / filename, target / filename)
        with (source / 'agents.csv').open(newline='') as f:
            original = list(csv.DictReader(f))
        weights = Counter((int(r['origin']), int(r['destination'])) for r in original)
        allocations = {od: count * n // len(original) for od, n in weights.items()}
        ranked = sorted(weights, key=lambda od: (-(count * weights[od] % len(original)), od))
        for od in ranked[:count - sum(allocations.values())]:
            allocations[od] += 1
        assert min(allocations.values()) > 0 and sum(allocations.values()) == count
        rng = random.Random(42)
        rows = [(rng.randrange(1800), o, d) for (o, d), n in sorted(allocations.items()) for _ in range(n)]
        rows.sort()
        with (target / 'agents.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'origin', 'destination', 'start_time'])
            w.writerows((i, o, d, t) for i, (t, o, d) in enumerate(rows))
        reg = registry(target / f'{name}.net.xml', name)
        (target / 'edge_registry.json').write_text(json.dumps(reg, separators=(',', ':')) + '\n')
        od_definition = ast.literal_eval((target / f'od_{name}.txt').read_text())
        incident_placeholder = od_definition['origins'][0]
        for suffix in ['', '_offline']:
            config = (ROOT / 'configs' / f'ingolstadt_custom{suffix}.toml').read_text().replace('ingolstadt_custom', name).replace('Ingolstadt', name.capitalize()).replace('201201950#0', incident_placeholder)
            # Offline Ingolstadt also disables the incident by moving its window beyond the horizon.
            import re
            config = re.sub(r'^incident_edge = .*$', f'incident_edge = "{incident_placeholder}"', config, flags=re.M)
            config = re.sub(r'^incident_start_seconds = .*$', 'incident_start_seconds = 1000000.0', config, flags=re.M)
            config = re.sub(r'^incident_end_seconds = .*$', 'incident_end_seconds = 1001000.0', config, flags=re.M)
            if '# Disabled: keep the incident window beyond the episode horizon.' not in config:
                config = config.replace('incident_start_seconds', '# Disabled: keep the incident window beyond the episode horizon.\nincident_start_seconds', 1)
            (ROOT / 'configs' / f'{name}{suffix}.toml').write_text(config)
            cfg = load_config(ROOT / 'configs' / f'{name}{suffix}.toml')
            prepared = validate_prepared_inputs(cfg)
            topology = load_traffic_topology(prepared)
            assert prepared.vehicle_count == count
            assert cfg.simulation.incident_start_seconds > cfg.simulation.evaluation_horizon_seconds
            assert cfg.simulation.incident_edge in topology.network.edge_to_index
        assert sha(source / f'{name}.net.xml') == sha(target / f'{name}.net.xml')
        assert sha(source / f'od_{name}.txt') == sha(target / f'od_{name}.txt')
        manifest['networks'][name] = dict(source_vehicle_count=len(original), expected_vehicle_count=count, unique_ods=len(weights), all_od_reachable=True, external_edges=topology.network.edge_count, passenger_edges=int(topology.network.passenger_allowed.sum()), source_hashes={f: sha(source / f) for f in [f'{name}.net.xml', f'od_{name}.txt', 'agents.csv']}, prepared_hashes=prepared.input_hashes, incidents_enabled=False, od_allocations=[dict(origin=o, destination=d, source_count=weights[o,d], prepared_count=n) for (o,d),n in sorted(allocations.items())])
        print(name, count, topology.network.edge_count, 'validated', flush=True)
    destination = ROOT / 'outputs/network_selection_20261007/preparation_manifest.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
