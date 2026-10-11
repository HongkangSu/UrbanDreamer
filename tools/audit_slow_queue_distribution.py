"""Read-only road-frame coverage census for frozen pretraining and codec datasets."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COUNT_MIN = 5 - 1e-5


def queue_masks(flow, capacities, speed_limits):
    """A codec frame uses current counts and the two-boundary mean speed label."""
    count = flow[:-1, :, 2] * capacities
    future_count = flow[1:, :, 2] * capacities
    speed = flow[:-1, :, 1] * speed_limits
    future_speed = flow[1:, :, 1] * speed_limits
    slow = (count >= COUNT_MIN) & ((speed + future_speed) / 2 <= 2)
    persistent = ((count >= COUNT_MIN) & (future_count >= COUNT_MIN)
                  & (speed <= 2) & (future_speed <= 2))
    per_window = slow.sum(axis=1)
    localized = (per_window >= 1) & (per_window <= 5)
    widespread = per_window >= 20
    return count, slow, persistent, localized, widespread


def summarize_flow(flow, capacities, speed_limits, critical_edges=()):
    count, slow, persistent, localized, widespread = queue_masks(flow, capacities, speed_limits)
    row = dict(episodes=1, windows=len(count), total_road_cells=count.size,
        occupied_cells=int((count > 0).sum()), slow_cells=int(slow.sum()),
        persistent_cells=int(persistent.sum()), slow_windows=int(slow.any(axis=1).sum()),
        localized_slow_windows=int(localized.sum()), widespread_slow_windows=int(widespread.sum()),
        localized_slow_cells=int(slow[localized].sum()), widespread_slow_cells=int(slow[widespread].sum()))
    edges = {}
    for edge in critical_edges:
        c, s, p = count[:, edge], slow[:, edge], persistent[:, edge]
        edges[str(edge)] = dict(occupied_frames=int((c > 0).sum()), queue_frames=int((c >= COUNT_MIN).sum()),
            slow_frames=int(s.sum()), persistent_frames=int(p.sum()),
            localized_slow_frames=int((s & localized).sum()), widespread_slow_frames=int((s & widespread).sum()),
            slow_count_5_9=int((s & (c < 10 - 1e-5)).sum()),
            slow_count_10_19=int((s & (c >= 10 - 1e-5) & (c < 20 - 1e-5)).sum()),
            slow_count_20_49=int((s & (c >= 20 - 1e-5) & (c < 50 - 1e-5)).sum()),
            slow_count_ge50=int((s & (c >= 50 - 1e-5)).sum()))
    return dict(totals=row, critical_edges=edges)


def add_stats(destination, source):
    for key, value in source.items():
        if isinstance(value, dict):
            add_stats(destination.setdefault(key, {}), value)
        else:
            destination[key] = destination.get(key, 0) + value


def finish_stats(value):
    for row in [value['totals'], *[r['totals'] for r in value['methods'].values()]]:
        row['slow_fraction_all_road_cells'] = row['slow_cells'] / max(row['total_road_cells'], 1)
        row['slow_fraction_occupied_cells'] = row['slow_cells'] / max(row['occupied_cells'], 1)
        row['persistent_fraction_occupied_cells'] = row['persistent_cells'] / max(row['occupied_cells'], 1)
    return value


def scan(entries, capacities, speed_limits, edges, workers):
    result = dict(totals={}, critical_edges={}, methods={})
    def one(entry):
        with np.load(entry['cache_path'], allow_pickle=False) as archive:
            stats = summarize_flow(archive['flow'], capacities, speed_limits, edges)
        return entry['method'], stats
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for method, stats in pool.map(one, entries):
            add_stats(result, stats)
            add_stats(result['methods'].setdefault(method, {}), stats)
    return finish_stats(result)


def critical_cases(results, by_path, capacities, limits, edge_ids):
    cases = []
    for case in results['recursive_cases']:
        if case['method'] not in {'random', 'online_ac'}:
            continue
        entry = by_path[case['path']]
        with np.load(entry['cache_path'], allow_pickle=False) as archive:
            flow = archive['flow']
            start, endpoint = case['actual_start'], case['actual_start'] + 8
            # Reproduce fixed C0 cohort count at 240 s without loading raw states_json.
            offsets = archive['boundary_offsets']
            left, right = offsets[start:start + 2]
            cohort = archive['active_vehicle'][left:right]
            left, right = offsets[endpoint:endpoint + 2]
            membership = np.isin(archive['active_vehicle'][left:right], cohort)
            external = membership & ~archive['active_in_junction'][left:right]
            road = archive['active_current_edge'][left:right][external]
            counts = np.bincount(road, minlength=len(capacities))
            speed = .5 * (flow[endpoint, :, 1] + flow[endpoint + 1, :, 1]) * limits
            slow = (counts >= COUNT_MIN) & (speed <= 2)
            cases.append(dict(method=case['method'], source_path=case['path'], split=entry['split'],
                start_frame=start, endpoint_frame=endpoint,
                roads=[dict(index=int(i), edge_id=edge_ids[i], cohort_count=int(counts[i]),
                            full_count=float(flow[endpoint, i, 2] * capacities[i]),
                            speed_label_mps=float(speed[i])) for i in np.flatnonzero(slow)]))
    return cases


def sampler_coverage(checkpoint, entries):
    state = checkpoint['sampler']
    total = sum(e['frames'] for e in entries)
    current = sum(entries[i]['frames'] for i in state['order'][:state['cursor']]) + state['frame_cursor']
    return dict(updates=checkpoint['update'], batch_size=checkpoint['batch_size'],
        complete_epochs=state['epoch'], total_frames_per_epoch=total,
        current_epoch_frames_seen=current, total_raw_frames_seen=state['epoch'] * total + current,
        epoch_equivalent=state['epoch'] + current / total,
        cohort_augmentation_fraction=checkpoint.get('cohort_augmentation_fraction'),
        note='Raw frame visits; hybrid cohort augmentation replaces part of each batch, so not all visits used the raw full-state input.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/ingolstadt_noincident_pretrain_20261008')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.config import load_config
    import torch
    started = time.monotonic()
    config = load_config(args.parent/'base_config.toml')
    network = load_traffic_topology(validate_prepared_inputs(config)).network
    capacities = np.maximum(network.capacities, 1.)
    limits = network.speed_limits_mps
    cache = json.loads((args.parent/'cache/cache_manifest.json').read_text())
    codec = json.loads((args.parent/'codec/codec_manifest.json').read_text())
    results = json.loads((args.parent/'world_accuracy_20261009/results.json').read_text())
    by_path = {e['path']: e for e in cache['sources']}
    cases = critical_cases(results, by_path, capacities, limits, codec['edge_ids'])
    edges = sorted({road['index'] for case in cases for road in case['roads']})
    output = args.parent/'slow_queue_diagnosis_20261009'
    output.mkdir(exist_ok=True)
    checkpoint = torch.load(args.parent/'codec/codec_latest.pt', map_location='cpu', weights_only=False)
    result = dict(cache_manifest_identity=cache['identity'], codec_manifest_identity=codec['identity'],
        critical_cases=cases, sampler=sampler_coverage(checkpoint, codec['entries']['train']),
        definitions=dict(slow='current count >=5 and two-boundary mean speed <=2 m/s',
            persistent='both counts >=5 and both boundary speeds <=2 m/s',
            localized='1..5 slow roads in full-state window', widespread='>=20 slow roads in full-state window'),
        limitations=['Road frames are correlated observations, not independent episodes.',
                    'Full-state coverage does not prove coverage of fixed-C0 hybrid inputs.',
                    'Codec cohort augmentation uses 16-step derived pools, unlike the 8-step diagnostic.'])
    result['codec_train'] = scan([by_path[e['path']] for e in codec['entries']['train']], capacities, limits, edges, args.workers)
    (output/'distribution_codec.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='codec', elapsed=time.monotonic()-started, totals=result['codec_train']['totals'], sampler=result['sampler'])), flush=True)
    result['world_model_train'] = scan([e for e in cache['entries'] if e['split']=='train'], capacities, limits, edges, args.workers)
    result['elapsed_seconds'] = time.monotonic()-started
    (output/'distribution_full.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(stage='complete', elapsed=result['elapsed_seconds'], totals=result['world_model_train']['totals'])), flush=True)

if __name__ == '__main__':
    main()
