"""Instrument isolated AC collection without touching a training process or output."""
import argparse
from collections import Counter, defaultdict
import functools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'outputs/city_baselines_current_to_dest_1000_20261013_180607'
sys.path.insert(0, str(BATCH / 'source'))
import numpy as np
import torch
import libsumo
from tools.train_toy_actor_critic import RealActorCritic
from tools.queued_city_baselines import write_json, process_identity
from urban_dreamer.traffic import route_candidates as route_module
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.topology import NextHopCandidates
from urban_dreamer.learning import observations
from urban_dreamer.models.policy import NextHopActor


class Timings:
    def __init__(self):
        self.stack = []
        self.reset()

    def reset(self):
        self.seconds = defaultdict(float)
        self.exclusive = defaultdict(float)
        self.calls = Counter()
        self.cache = Counter()
        self.path_lengths = Counter()
        self.samples = []

    def wrap(self, owner, attr, name, hook=None):
        original = getattr(owner, attr)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if hook:
                hook(args, kwargs)
            start = time.perf_counter()
            frame = [name, 0.]
            self.stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.stack.pop()
                self.calls[name] += 1
                self.seconds[name] += elapsed
                self.exclusive[name] += elapsed - frame[1]
                if self.stack:
                    self.stack[-1][1] += elapsed
        setattr(owner, attr, wrapped)

    def snapshot(self):
        return dict(inclusive_seconds=dict(self.seconds), exclusive_seconds=dict(self.exclusive),
            calls=dict(self.calls), cache=dict(self.cache), path_length_histogram=dict(self.path_lengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', required=True)
    opts = parser.parse_args()
    output = BATCH / 'collection_profiles_20261014' / opts.city
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(BATCH / opts.city / 'online_ac/latest.pt', map_location='cpu', weights_only=False)
    args = argparse.Namespace(**checkpoint['args'])
    args.config = Path(args.config)
    args.output_dir = output
    args.device = 'cpu'
    args.collection_device = 'cpu'
    args.episodes = 2000
    args.entropy_schedule_episodes = 1000
    runner = RealActorCritic(args)
    runner.actor.load_state_dict(checkpoint['actor'])
    seed = args.seed + checkpoint['episode']
    checkpoint_episode = checkpoint['episode']
    del checkpoint
    timers = Timings()

    def cache_hook(pos, kw):
        candidate, current, destination = pos[:3]
        hit = (current, destination) in candidate._route_candidate_cache
        timers.cache['hits' if hit else 'misses'] += 1
        if hit:
            paths = candidate._route_candidate_cache[(current, destination)]
            for path in paths:
                timers.path_lengths[len(path)] += 1

    def observation_hook(pos, kw):
        if timers.calls['route_observations'] % 8 == 0:
            window = kw['policy_edge_features']
            vehicles = [{key: row[key] for key in ('current_edge', 'destination_edge', 'remaining_distance')}
                        for row in kw['vehicle_states']]
            timers.samples.append(dict(policy_edge_features=observations.prepare_actor_window(
                np.array(window.edge_features, copy=True), runner.network.edge_count),
                vehicle_states=vehicles, episode_progress=kw['episode_progress'],
                active_vehicle_fraction=kw['active_vehicle_fraction']))

    timers.wrap(route_module.RouteCandidates, 'generate_paths', 'generate_paths', cache_hook)
    timers.wrap(route_module.RouteCandidates, '_shortest_path', 'penalty_dijkstra')
    timers.wrap(route_module, 'build_route_observations', 'route_observations', observation_hook)
    timers.wrap(NextHopCandidates, 'distances', 'destination_distances')
    timers.wrap(NextHopCandidates, 'full_route', 'next_hop_full_route')
    for attr, label in (('_observe', 'vehicle_observation'), ('_route_commitment', 'route_commitment'),
                        ('_apply_decisions', 'apply_decisions'), ('_flow', 'flow'),
                        ('_state', 'state_snapshot'), ('_conditioning_turns', 'conditioning_turns')):
        timers.wrap(SumoEventSession, attr, label)
    timers.wrap(NextHopActor, 'sample_action', 'actor_inference')
    timers.wrap(libsumo, 'simulationStep', 'sumo_step')
    for attr in ('setRoute', 'getRoute', 'getRouteIndex', 'getRoadID', 'getLanePosition'):
        timers.wrap(libsumo.vehicle, attr, 'libsumo_' + attr)
    write_json(output / 'runtime.json', dict(pid=os.getpid(), identity=process_identity(os.getpid()),
        city=opts.city, checkpoint_episode=checkpoint_episode, seed=seed,
        note='Isolated frozen actor, same seed in cold/warm episodes; no gradients, no archive writes'))
    report = dict(city=opts.city, checkpoint_episode=checkpoint_episode, seed=seed,
        network_edges=runner.network.edge_count, episodes=[], note='Inclusive timings overlap; exclusive timings sum without nesting duplication. Timers and sampling add small overhead. Collection includes real SUMO and training-row construction, excluding learning and archive writes.')
    for phase in ('cold_cache', 'warm_cache'):
        timers.reset()
        start = time.perf_counter()
        rows, metrics = runner.run_episode(policy='stochastic', seed=seed, collect=True)
        elapsed = time.perf_counter() - start
        samples = timers.samples
        snapshot = timers.snapshot()
        lengths = timers.path_lengths
        mean_length = sum(int(k)*v for k,v in lengths.items()) / max(sum(lengths.values()), 1)
        record = dict(phase=phase, elapsed_seconds=elapsed, mean_cached_route_edges=mean_length,
            cache_entries=len(runner.candidates._route_candidate_cache), metrics=metrics, **snapshot)
        report['episodes'].append(record)
        write_json(output / 'profile.json', report)
        print(json.dumps(dict(city=opts.city, phase=phase, elapsed_seconds=elapsed,
            cache=snapshot['cache'], exclusive_seconds=snapshot['exclusive_seconds'])), flush=True)
        del rows
    # Same physical vehicle/traffic inputs isolate feature construction from policy trajectories.
    next_hop = NextHopCandidates(runner.network, traffic_model='meso')
    route_candidate = runner.candidates
    route_candidate._route_candidate_cache_limit = max(16384, sum(len(s['vehicle_states']) for s in samples))
    for sample in samples:
        for v in sample['vehicle_states']:
            route_candidate.generate_paths(v['current_edge'], v['destination_edge'])
        observations.build_actor_observations(candidates=next_hop, **sample)
    benchmark = {}
    for mode, candidate in (('current_to_dest', route_candidate), ('next_hop', next_hop)):
        timers.reset()
        start = time.perf_counter()
        for sample in samples:
            observations.build_actor_observations(candidates=candidate, **sample)
        benchmark[mode] = dict(seconds=time.perf_counter()-start,
            vehicle_rows=sum(len(s['vehicle_states']) for s in samples), **timers.snapshot())
    report['matched_observation_benchmark'] = benchmark
    write_json(output / 'profile.json', report)
    write_json(output / 'completed.json', dict(status='completed', city=opts.city, pid=os.getpid()))
    print(json.dumps(dict(city=opts.city, status='completed', matched_observation_benchmark=benchmark)), flush=True)


if __name__ == '__main__':
    main()
