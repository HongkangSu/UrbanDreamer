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
from tools.train_toy_actor_critic import RealActorCritic
from urban_dreamer.traffic.environment import SumoSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='latest.pt')
    parser.add_argument('--seed', type=int, default=50042)
    args = parser.parse_args()
    output = args.input_dir.resolve()
    checkpoint = torch.load(output / args.checkpoint, map_location='cpu', weights_only=False)
    options = argparse.Namespace(**checkpoint['args'])
    options.device = 'cpu'
    runner = RealActorCritic(options)
    runner.actor.load_state_dict(checkpoint['actor'])
    if runner.critic is not None:
        runner.critic.load_state_dict(checkpoint['critic'])
    traces = []
    original = SumoSession._apply_decisions
    def observe(session, callback, flow):
        decisions = original(session, callback, flow)
        for decision in decisions:
            vehicle = session._vehicles[decision.vehicle_id]
            traces.append(dict(vehicle_id=decision.vehicle_id, current_edge=decision.current_edge,
                next_edge=decision.next_edge, time=decision.decision_time,
                actual_departure_time=vehicle['actual_departure_time']))
        return decisions
    result = {'checkpoint_episode': checkpoint['episode'], 'seed': args.seed, 'policies': {}}
    SumoSession._apply_decisions = observe
    try:
        for policy in ('deterministic', 'stochastic'):
            traces.clear()
            rows, metrics = runner.run_episode(policy=policy, seed=args.seed, collect=True)
            if metrics['unfinished']:
                raise ValueError('route travel-time analysis requires complete trips')
            grouped = defaultdict(list)
            for event in traces:
                grouped[event['vehicle_id']].append(event)
            event_reward = defaultdict(float)
            for row in rows:
                event_reward[runner.topology.vehicles[row['vehicle_id']].vehicle_id] += row['reward']
            records = []
            for vehicle in runner.topology.vehicles:
                decisions = grouped[vehicle.vehicle_id]
                origin = runner.network.edge_to_index[runner.topology.origins[vehicle.origin_index]]
                destination = runner.network.edge_to_index[runner.topology.destinations[vehicle.destination_index]]
                path = [origin]
                for decision in decisions:
                    if decision['current_edge'] != path[-1]:
                        raise AssertionError('discontinuous executed route trace')
                    path.append(decision['next_edge'])
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
                # All Toy trips complete in this diagnostic; account for departure-observation lag.
                observed_trip = -30. * event_reward[vehicle.vehicle_id] + decisions[0]['time'] - decisions[0]['actual_departure_time']
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
        SumoSession._apply_decisions = original
    path = output / f'route_diagnostic_episode_{checkpoint["episode"]}.json'
    path.write_text(json.dumps(result, indent=2))
    print(str(path), flush=True)


if __name__ == '__main__':
    main()
