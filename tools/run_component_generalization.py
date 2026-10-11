"""Paired SUMO demand-shift evaluation, isolated from training processes.

Development seeds are 610000..619999; final held-out seeds are 710000..719999.
Checkpoint adapter factories receive (checkpoint=Path, runner=EpisodeRunner) and
must return an object with select(..., explore=False). Selection never receives
the sampled demand table. Adapters must load frozen weights before evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from urban_dreamer.traffic.demand import PoissonDemand, demand_identity
from tools.routing_demand_profiles import PROFILES, ReTimedDemand, profile_metadata


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze_json(path: Path, value):
    """Resume only exactly matching inputs, never silently overwrite a manifest."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen experiment input differs: {path}')
    else:
        write_json(path, value)


def scaled_demand(base: PoissonDemand, scale: float, profile: str):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('rate scale must be finite and positive')
    scaled = PoissonDemand(tuple((o, d, rate * scale) for o, d, rate in base.od_rates),
                           base.duration_seconds)
    return ReTimedDemand(scaled, profile)


@dataclass(frozen=True)
class FrozenDemand:
    """A pre-sampled demand used identically by every policy in this condition."""
    base: ReTimedDemand
    vehicles: tuple
    base_seed: int
    episode_seed: int

    @property
    def duration_seconds(self):
        return self.base.duration_seconds

    @property
    def expected_vehicle_count(self):
        return self.base.expected_vehicle_count

    @property
    def od_rates(self):
        return self.base.od_rates

    def sample(self, *, seed, episode_index):
        if (seed, episode_index) != (self.base_seed, self.episode_seed):
            raise ValueError('Frozen demand seed mismatch')
        return self.vehicles


def validate_seed_split(seeds, split):
    lower = 610000 if split == 'development' else 710000
    if len(seeds) != len(set(seeds)) or any(not lower <= seed < lower + 10000 for seed in seeds):
        raise ValueError(f'{split} requires unique seeds in [{lower}, {lower + 10000})')


def normalize_metrics(metrics):
    result = dict(metrics)
    started = int(metrics['started_vehicle_count'])
    result['truncated_cost_seconds_per_started_vehicle'] = (
        metrics['cumulative_vehicle_seconds'] / started if started else None)
    result['truncated_cost_minutes_per_started_vehicle'] = (
        result['truncated_cost_seconds_per_started_vehicle'] / 60 if started else None)
    result['departed_unfinished'] = metrics['unfinished'] - metrics['never_departed']
    result['completed_vehicle_count'] = metrics['vehicle_count'] - metrics['unfinished']
    result['att_population'] = 'completed trips only; unfinished and never-departed excluded'
    result['cutoff_protocol'] = 'fixed configured deadline; no drain extension'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO / 'configs/ingolstadt_custom.toml')
    parser.add_argument('--output', type=Path, default=REPO / 'outputs/component_experiments_20261011/generalization')
    parser.add_argument('--methods', nargs='+', choices=('freeflow_shortest', 'dynamic_shortest', 'checkpoint'),
                        default=['freeflow_shortest', 'dynamic_shortest'])
    parser.add_argument('--scales', nargs='+', type=float, default=[.7, 1., 1.3])
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=['original', 'peaks', 'burst', 'od_stagger'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[610001])
    parser.add_argument('--split', choices=('development', 'test'), default='development')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--base-seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-adapter', help='module:factory; loaded once with checkpoint and runner')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args(argv)
    validate_seed_split(args.seeds, args.split)
    if args.threads not in (1, 2):
        parser.error('--threads must be 1 or 2 for this evaluation lane')
    if 'checkpoint' in args.methods and (args.checkpoint is None or not args.checkpoint_adapter):
        parser.error('checkpoint evaluation requires --checkpoint and --checkpoint-adapter')
    import torch
    import libsumo
    from urban_dreamer.learning.baselines.common import EpisodeRunner
    from urban_dreamer.traffic.event_session import SumoEventSession
    from tools.causal_routing_policies import RealTimeShortestRouter
    from tools.routing_cost_observer import PeriodicCostObserver

    handle = {}

    class ObservedSession(SumoEventSession):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.routing_cost_observer = PeriodicCostObserver(30, self.network.free_flow_travel_seconds)

        def _read_costs(self):
            return [self._connection.edge.getTraveltime(edge) for edge in self.network.edge_ids]

        def reset(self):
            state = super().reset()
            self.routing_cost_observer.observe(0, self._read_costs)
            return state

        def _apply_decisions(self, callback, frozen_flow):
            self.routing_cost_observer.observe(self._time_ms / 1000., self._read_costs)
            return super()._apply_decisions(callback, frozen_flow)

    def factory(*a, **kw):
        handle['session'] = ObservedSession(*a, **kw, step_seconds=1., backend='libsumo')
        return handle['session']

    runargs = argparse.Namespace(config=args.config.resolve(), device='cpu', threads=args.threads,
        seed=args.base_seed, vehicle_limit=0, session='event', backend='libsumo', step_seconds=1.,
        forbid_revisit=False, sp_prior_weight=0., global_reward_only=True, reward_objective='observed_mean')
    runner = EpisodeRunner(runargs, session_factory=factory)
    base_demand = runner.demand
    source_files = [Path(__file__), REPO / 'tools/routing_demand_profiles.py',
                    REPO / 'tools/routing_cost_observer.py', REPO / 'tools/causal_routing_policies.py',
                    REPO / 'urban_dreamer/learning/baselines/common.py',
                    REPO / 'urban_dreamer/traffic/event_session.py']
    protocol = dict(schema_version=1, split=args.split, city=runner.network.name if hasattr(runner.network, 'name') else args.config.stem,
        config=str(args.config.resolve()), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        source_hashes={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        input_hashes=runner.prepared.input_hashes, simulation=runner.simulation_settings,
        sumo_version=list(libsumo.getVersion()), base_seed=args.base_seed, deadline_seconds=runner.deadline,
        physical_step_seconds=1, road_cost_observation_seconds=30, optimizer_updates=0,
        future_demand_access=False, action_space='legal branch-only next-hop mask',
        heuristic_difference='identical edge-only shortest-path planner; freeflow static vs observed 30s costs',
        actor_information_difference='heuristics use global road costs and graph planning; actor feature access must be audited separately',
        cutoff_protocol='fixed deadline; completed-trip ATT is completion-selected; never-departed excluded from C/B',
        seed_contract='development 610000..619999; final test 710000..719999; exclude both from training',
        checkpoint=(dict(path=str(args.checkpoint.resolve()), sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                         adapter=args.checkpoint_adapter) if args.checkpoint else None))
    output = args.output.resolve() / args.split
    freeze_json(output / 'protocol.json', protocol)
    checkpoint_policy = None
    if 'checkpoint' in args.methods:
        module, symbol = args.checkpoint_adapter.split(':')
        checkpoint_policy = getattr(importlib.import_module(module), symbol)(checkpoint=args.checkpoint, runner=runner)
    for scale in args.scales:
        for profile in args.profiles:
            shifted = scaled_demand(base_demand, scale, profile)
            for seed in args.seeds:
                condition = output / f'rate_{scale:g}' / profile / f'seed_{seed}'
                vehicles = shifted.sample(seed=args.base_seed, episode_index=seed)
                identity = demand_identity(vehicles)
                manifest = dict(rate_scale=scale, profile=profile_metadata(profile, shifted.duration_seconds),
                    seed=seed, base_seed=args.base_seed, demand_sha256=identity,
                    expected_vehicle_count=shifted.expected_vehicle_count, vehicles=[asdict(v) for v in vehicles])
                freeze_json(condition / 'demand.json', manifest)
                runner.demand = FrozenDemand(shifted, vehicles, args.base_seed, seed)
                for method in args.methods:
                    result_path = condition / f'{method}.json'
                    if result_path.exists():
                        saved = json.loads(result_path.read_text())
                        if saved['metrics']['demand_sha256'] != identity:
                            raise ValueError(f'Existing evaluation has different demand: {result_path}')
                        print(json.dumps(dict(status='already_complete', result=str(result_path))), flush=True)
                        continue
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    router = RealTimeShortestRouter(runner.network)
                    progress = [-1]

                    def select(vehicles, actor_obs, frozen_flow, now):
                        bucket = int(now // 300)
                        if bucket != progress[0]:
                            progress[0] = bucket
                            print(json.dumps(dict(method=method, scale=scale, profile=profile, seed=seed, time=now)), flush=True)
                        if method == 'checkpoint':
                            with torch.inference_mode():
                                return checkpoint_policy.select(vehicles, actor_obs, frozen_flow, now, explore=False)
                        if method == 'dynamic_shortest':
                            observer = handle['session'].routing_cost_observer
                            if router.timestamp != observer.timestamp:
                                router.update_costs(observer.values, observer.timestamp)
                        choices = []
                        for row, vehicle in enumerate(vehicles):
                            cols = np.flatnonzero(actor_obs.mask[row])
                            legal = actor_obs.edge_indices[row, cols].tolist()
                            next_edge = router.next_edge(int(vehicle['current_edge']), int(vehicle['destination_edge']), legal)
                            choices.append(int(cols[legal.index(next_edge)]))
                        return choices, [dict(entropy=0.) for _ in choices]

                    print(json.dumps(dict(status='starting', method=method, scale=scale, profile=profile, seed=seed,
                                          vehicle_count=len(vehicles), demand_sha256=identity)), flush=True)
                    started = time.monotonic()
                    def archive_metadata(metadata):
                        metadata['demand']['timing_profile'] = manifest['profile']
                        metadata['demand']['rate_scale'] = scale
                        metadata['demand']['law'] = manifest['profile']['demand_law']
                        metadata['experiment_only'] = True
                        return metadata
                    _, _, metrics = runner.run_episode(select, seed=seed, label=method,
                        archive_path=condition / f'{method}.npz' if args.archive else None,
                        archive_metadata=dict(phase='evaluation', split=args.split), metadata_transform=archive_metadata)
                    if metrics['demand_sha256'] != identity or metrics['vehicle_count'] != len(vehicles):
                        raise AssertionError('SUMO demand differs from frozen manifest')
                    result = dict(method=method, scale=scale, profile=profile, seed=seed,
                        metrics=normalize_metrics(metrics), wall_seconds=time.monotonic() - started,
                        policy_stats=router.stats if method != 'checkpoint' else None,
                        cost_observation_times=handle['session'].routing_cost_observer.sample_times)
                    write_json(result_path, result)
                    print(json.dumps(dict(status='complete', result=str(result_path), metrics=result['metrics'])), flush=True)


if __name__ == '__main__':
    main()
