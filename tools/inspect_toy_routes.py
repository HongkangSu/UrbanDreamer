#!/usr/bin/env python3
"""Trace executed Toy next hops and compare their length/cost to static lower bounds."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.train_toy_actor_critic import CHECKPOINT_FORMAT, RealActorCritic
from urban_dreamer.traffic.environment import SumoSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest.pt')
    parser.add_argument('--seed', type=int, default=50042)
    args = parser.parse_args()
    output = args.input_dir.resolve()
    checkpoint = torch.load(output / args.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('format') != CHECKPOINT_FORMAT:
        raise ValueError(f'unsupported checkpoint format; expected {CHECKPOINT_FORMAT}')
    options = argparse.Namespace(**checkpoint['args'])
    options.device = 'cpu'
    runner = RealActorCritic(options)
    runner.actor.load_state_dict(checkpoint['actor'])
    session_class = runner.session_factory.func if runner.session_factory is not None else SumoSession
    paths = defaultdict(list)
    travel_times = {}
    original = session_class._observe
    original_close = session_class.close
    def capture_close(session):
        travel_times.update(session.travel_times)
        return original_close(session)
    def observe(session, departure_counts):
        original(session, departure_counts)
        for vehicle_id, vehicle in session._vehicles.items():
            if vehicle['departed']:
                path = paths[vehicle_id]
                edge = vehicle['current_edge']
                if not path or path[-1] != edge:
                    path.append(edge)
    result = {'checkpoint_episode': checkpoint['episode'], 'seed': args.seed, 'policies': {}}
    session_class._observe = observe
    session_class.close = capture_close
    try:
        for policy in ('deterministic', 'stochastic'):
            paths.clear()
            travel_times.clear()
            _, metrics = runner.run_episode(policy=policy, seed=args.seed)
            if metrics['unfinished']:
                raise ValueError('route travel-time analysis requires complete trips')
            records = []
            for vehicle in runner.episode_topology.vehicles:
                origin = runner.network.edge_to_index[runner.topology.origins[vehicle.origin_index]]
                destination = runner.network.edge_to_index[runner.topology.destinations[vehicle.destination_index]]
                path = paths[vehicle.vehicle_id]
                if not path or path[0] != origin or path[-1] != destination:
                    raise AssertionError('incomplete observed route trace')
                lengths = [runner.candidates.junction(a,b)[0] for a,b in zip(path,path[1:])]
                times = [runner.candidates.junction(a,b)[0]/runner.candidates.junction(a,b)[1] for a,b in zip(path,path[1:])]
                distance = float(sum(runner.network.lengths_m[e] for e in path) + sum(lengths))
                static_time = float(sum(runner.network.free_flow_travel_seconds[e] for e in path) + sum(times))
                min_distance = float(runner.network.lengths_m[origin] + runner.candidates.distances(destination)[origin])
                min_time = float(runner.network.free_flow_travel_seconds[origin] + runner.candidates.costs(destination)[origin])
                edge_names = [runner.network.edge_ids[e] for e in path]
                category = ('X_to_R' if 'X_to_R' in edge_names else 'Y_to_R' if 'Y_to_R' in edge_names
                            else 'direct_R' if any(e in edge_names for e in ('A_to_R','B_to_R'))
                            else 'X_to_M' if 'X_to_M' in edge_names else 'Y_to_M')
                observed_trip = travel_times[vehicle.vehicle_id]
                records.append(dict(vehicle_id=vehicle.vehicle_id, category=category, edges=edge_names,
                    repeated_edge=len(path)!=len(set(path)), destination_reached=path[-1]==destination,
                    distance_m=distance, shortest_distance_m=min_distance, extra_distance_m=distance-min_distance,
                    static_time_s=static_time, shortest_static_time_s=min_time, extra_static_time_s=static_time-min_time,
                    actual_travel_seconds=observed_trip))
            groups = {}
            for category in sorted(set(r['category'] for r in records)):
                selected = [r for r in records if r['category']==category]
                groups[category] = dict(count=len(selected), mean_travel_seconds=float(np.mean([r['actual_travel_seconds'] for r in selected])),
                    mean_distance_m=float(np.mean([r['distance_m'] for r in selected])),
                    mean_extra_distance_m=float(np.mean([r['extra_distance_m'] for r in selected])),
                    mean_extra_static_time_s=float(np.mean([r['extra_static_time_s'] for r in selected])))
            result['policies'][policy] = dict(metrics=metrics, groups=groups,
                repeated_edge_count=sum(r['repeated_edge'] for r in records),
                longer_than_shortest_distance_count=sum(r['extra_distance_m']>.1 for r in records),
                slower_than_shortest_static_count=sum(r['extra_static_time_s']>.1 for r in records), vehicles=records)
            print(json.dumps(dict(policy=policy, checkpoint_episode=checkpoint['episode'], groups=groups,
                repeated_edges=result['policies'][policy]['repeated_edge_count'])), flush=True)
    finally:
        session_class._observe = original
        session_class.close = original_close
    path = output / f'route_diagnostic_episode_{checkpoint["episode"]}.json'
    path.write_text(json.dumps(result, indent=2))
    print(str(path), flush=True)


if __name__ == '__main__':
    main()
