"""Historical proportional-demand importer; never overwrite installed city defaults."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import shutil
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


from tools.prepare_maule_nangis import registry, sha
from tools.patch_candidate_uturns import build_patch, drop_unreachable_od


DEMAND_DURATION_SECONDS = 1800


def candidate_config(template, name, incident_placeholder, demand_duration_seconds=DEMAND_DURATION_SECONDS):
    config = template.replace('ingolstadt_custom', name).replace('Ingolstadt', name.capitalize())
    completion_buffer = tomllib.loads(template)['simulation']['training_completion_buffer_seconds']
    overrides = dict(demand_duration_seconds=f'{float(demand_duration_seconds)}',
        world_model_intervals_per_episode=str(demand_duration_seconds // 30),
        evaluation_horizon_seconds=str(demand_duration_seconds + completion_buffer),
        incident_edge=f'"{incident_placeholder}"', incident_start_seconds='1000000.0',
        incident_end_seconds='1001000.0', sp_prior_weight='0.0')
    for key, value in overrides.items():
        config = re.sub(rf'^{key} = .*$', f'{key} = {value}', config, flags=re.M)
    if '# Disabled: keep the incident window beyond the episode horizon.' not in config:
        config = config.replace('incident_start_seconds', '# Disabled: keep the incident window beyond the episode horizon.\nincident_start_seconds', 1)
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demand-duration-seconds', type=int, default=DEMAND_DURATION_SECONDS)
    parser.add_argument('--manifest', type=Path,
        help='Output manifest; defaults to a new timestamped preparation directory')
    args = parser.parse_args(argv)
    # This importer predates the accepted v5 / occupancy1500 mainline inputs.
    # Check both cities before writing anything, so one cannot be partly replaced.
    for city in ('coulommiers', 'montereau_fault_yonne'):
        installed = ROOT / 'networks' / city
        if installed.exists() and any(installed.iterdir()):
            parser.error(f'{installed} already contains installed defaults; historical import would '
                         'replace the accepted road network and demand. Use an isolated workspace.')
    if args.demand_duration_seconds <= 0 or args.demand_duration_seconds % 30:
        parser.error('demand duration must be a positive multiple of 30 seconds')
    duration = args.demand_duration_seconds
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    destination = args.manifest or ROOT / 'outputs' / f'ingo_reference_reselection_{stamp}' / 'preparation_manifest.json'
    if destination.exists():
        raise ValueError('refusing to overwrite an existing preparation manifest')
    manifest = {'demand_duration_seconds': duration, 'departure_seed': 42, 'allocation': 'largest remainder proportional to original OD row counts; ties sorted by integer OD indices', 'runtime_demand': 'CSV counts specify per-OD Poisson expected counts, not fixed realized episode totals', 'registry': 'matching existing schema; float32 normalized length, speed, lane count, sorted type index; capacity = lane count * 10; passenger connection adjacency row-normalized', 'networks': {}}
    for name, count in [('coulommiers', 2000), ('montereau_fault_yonne', 3000)]:
        source = ROOT.parent / 'networks' / name
        target = ROOT / 'networks' / name
        target.mkdir(parents=True, exist_ok=True)
        for filename in [f'{name}.net.xml', f'od_{name}.txt']:
            shutil.copyfile(source / filename, target / filename)
        patch_work = destination.parent / 'network_patch' / name
        patch_work.mkdir(parents=True, exist_ok=True)
        network_patch = build_patch(source / f'{name}.net.xml', target / f'{name}.net.xml',
                                    name, patch_work, Path(sys.executable).parent / 'netconvert')
        with (source / 'agents.csv').open(newline='') as f:
            original = list(csv.DictReader(f))
        weights = Counter((int(r['origin']), int(r['destination'])) for r in original)
        allocations = {od: count * n // len(original) for od, n in weights.items()}
        ranked = sorted(weights, key=lambda od: (-(count * weights[od] % len(original)), od))
        for od in ranked[:count - sum(allocations.values())]:
            allocations[od] += 1
        assert min(allocations.values()) > 0 and sum(allocations.values()) == count
        rng = random.Random(42)
        rows = [(rng.randrange(duration), o, d) for (o, d), n in sorted(allocations.items()) for _ in range(n)]
        rows.sort()
        with (target / 'agents.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['id', 'origin', 'destination', 'start_time'])
            w.writerows((i, o, d, t) for i, (t, o, d) in enumerate(rows))
        reg = registry(target / f'{name}.net.xml', name)
        (target / 'edge_registry.json').write_text(json.dumps(reg, separators=(',', ':')) + '\n')
        demand_filter = drop_unreachable_od(target, name)
        removed_ods = {(row['origin_index'], row['destination_index'])
                       for row in demand_filter['removed_od_pairs']}
        allocations = {od: n for od, n in allocations.items() if od not in removed_ods}
        remaining_count = demand_filter['remaining_expected_vehicles']
        assert sum(allocations.values()) == remaining_count
        od_definition = ast.literal_eval((target / f'od_{name}.txt').read_text())
        incident_placeholder = od_definition['origins'][0]
        for suffix in ['', '_offline']:
            config = candidate_config((ROOT / 'configs' / f'ingolstadt_custom{suffix}.toml').read_text(),
                                      name, incident_placeholder, duration)
            (ROOT / 'configs' / f'{name}{suffix}.toml').write_text(config)
            cfg = load_config(ROOT / 'configs' / f'{name}{suffix}.toml')
            prepared = validate_prepared_inputs(cfg)
            topology = load_traffic_topology(prepared)
            assert prepared.vehicle_count == remaining_count
            assert cfg.simulation.incident_start_seconds > cfg.simulation.evaluation_horizon_seconds
            assert cfg.simulation.incident_edge in topology.network.edge_to_index
        assert sha(source / f'{name}.net.xml') == network_patch['original_sha256']
        assert sha(target / f'{name}.net.xml') == network_patch['patched_sha256']
        assert sha(source / f'od_{name}.txt') == sha(target / f'od_{name}.txt')
        manifest['networks'][name] = dict(source_vehicle_count=len(original), initial_target_vehicle_count=count, expected_vehicle_count=remaining_count, unique_ods=len(allocations), all_od_reachable=True, external_edges=topology.network.edge_count, passenger_edges=int(topology.network.passenger_allowed.sum()), source_hashes={f: sha(source / f) for f in [f'{name}.net.xml', f'od_{name}.txt', 'agents.csv']}, prepared_hashes=prepared.input_hashes, incidents_enabled=False, od_allocations=[dict(origin=o, destination=d, source_count=weights[o,d], prepared_count=n) for (o,d),n in sorted(allocations.items())])
        manifest['networks'][name]['network_patch'] = network_patch
        manifest['networks'][name]['unreachable_od_removal'] = demand_filter
        print(name, remaining_count, topology.network.edge_count, 'validated', flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
