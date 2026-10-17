"""Evaluate the author GA with complete-route decisions on the common30s clock."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.causal_routing_policies import AuthorGeneticRouter
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.observations import CandidateObservations
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.topology import TopologyError


class GeneticRouteCandidates(RouteCandidates):
    """Validate GA's full-network route search against the physical legal graph."""

    def decision_route(self, decision, destination, visited_edges=()):
        route = tuple(decision.route)
        if (len(route) < 2 or route[0] != decision.current_edge
                or route[-1] != destination or route[1] != decision.next_edge):
            raise TopologyError('GA route endpoints/first turn do not match its decision')
        if len(set(route)) != len(route):
            raise TopologyError('GA submitted complete route must be loop-free')
        if any(not 0 <= edge < self.network.edge_count
               or not self.network.passenger_allowed[edge] for edge in route):
            raise TopologyError('GA route contains an unknown or forbidden edge')
        if any(target not in self.network.adjacency_list[source]
               for source, target in zip(route, route[1:])):
            raise TopologyError('GA route contains an invalid edge connection')
        return route


class GeneticRouteSession(SumoEventSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = GeneticRouteCandidates(self.network, traffic_model=self.traffic_model)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run_evaluation(config_path, output, author_source, *, seeds=(10042, 10043, 10044),
                   population_size=100, generations=50, cache_ttl=300,
                   vehicle_limit=0, deadline=None):
    """The environment controls30s timing; observations carry GA's chosen route."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('GA evaluation requires a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(config=Path(config_path), output_dir=output, method='ga_routing',
        device='cpu', threads=1, seed=42, vehicle_limit=vehicle_limit, session='event',
        backend='libsumo', step_seconds=1., forbid_revisit=False, sp_prior_weight=0.,
        global_reward_only=True, reward_objective='observed_mean')
    handle = {}
    def factory(*positional, **keywords):
        handle['session'] = GeneticRouteSession(*positional, **keywords,
            step_seconds=1., backend='libsumo')
        return handle['session']
    runner = EpisodeRunner(args, session_factory=factory)
    if runner.config.simulation.routing_mode != 'current_to_dest':
        raise ValueError('GA30 requires a current_to_dest configuration')
    runner.candidates = GeneticRouteCandidates(runner.network,
        traffic_model=runner.config.simulation.traffic_model)
    if deadline is not None:
        runner.deadline = float(deadline)
    policy = AuthorGeneticRouter(runner.network, Path(author_source),
        population_size=population_size, generations=generations, cache_ttl=cache_ttl)
    protocol = dict(method='ga_routing', checkpoint_episode=None,
        config=str(Path(config_path).resolve()), input_hashes=runner.prepared.input_hashes,
        simulation=runner.simulation_settings, routing_mode='current_to_dest',
        policy_decision_interval_seconds=30., decision_protocol='30s_complete_route',
        route_search_space='all legal destination-reaching paths',
        fallback='Original first-legal next hop plus shortest suffix on failure; submission loop erasure may remove a closed initial detour.',
        route_submission='Closed loops erased before installation; raw GA chromosomes/cache and random operators unchanged.',
        population_size=population_size, generations=generations, crossover_rate=.8,
        mutation_rate=.1, tournament_size=5, cache_ttl_seconds=cache_ttl,
        author_source=str(Path(author_source).resolve()),
        author_sha256=hashlib.sha256(Path(author_source).read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((ROOT/'tools/causal_routing_policies.py').read_bytes()).hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        demand_seed=42, seeds=list(seeds), deadline_seconds=runner.deadline,
        training_episodes=0, future_demand_access=False)
    atomic_json(output/'protocol.json', protocol)
    records = []
    for seed in seeds:
        policy.reset()
        random.seed(seed)
        np.random.seed(seed)
        last_timestamp = None
        query_count = invalid_count = 0
        policy_times = []
        def observations(*, vehicle_states, **unused):
            nonlocal last_timestamp, query_count, invalid_count
            session = handle['session']
            now = session._time_ms / 1000.
            if now % 30 != 0:
                raise RuntimeError(f'GA decision outside the30s policy clock: {now}')
            if last_timestamp != now:
                values = np.array([session._connection.edge.getTraveltime(edge)
                    for edge in runner.network.edge_ids], dtype=np.float64)
                invalid = ~np.isfinite(values) | (values <= 0)
                invalid_count += int(invalid.sum())
                values[invalid] = runner.network.free_flow_travel_seconds[invalid]
                policy.update_costs(values, now)
                last_timestamp = now
                query_count += 1
                policy_times.append(now)
            paths = tuple((policy.complete_route(int(vehicle['current_edge']),
                int(vehicle['destination_edge']), runner.candidates),) for vehicle in vehicle_states)
            n = len(paths)
            return CandidateObservations(np.zeros((n, 1, 10), dtype=np.float32),
                np.ones((n, 1), dtype=bool),
                np.asarray([[path[0][1]] for path in paths], dtype=np.int64).reshape(n, 1),
                routes=paths)
        def select(vehicles, observations, flow, now):
            return [0] * len(vehicles), [dict(entropy=0.) for _ in vehicles]
        tick = time.monotonic()
        archive = output/f'evaluation_{seed}.npz'
        _, _, metrics = runner.run_episode(select, seed=seed, label='deterministic',
            observation_builder=observations, archive_path=archive,
            archive_metadata=dict(phase='evaluation', episode=seed,
                method='ga_routing', decision_protocol='30s_complete_route'))
        records.append(dict(metrics, seed=seed, archive=str(archive.resolve()),
            policy_stats=dict(policy.stats), policy_decision_times=policy_times,
            cost_query_count=query_count, invalid_cost_fallback_count=invalid_count,
            evaluator_wall_seconds=time.monotonic()-tick))
        atomic_json(output/'result.json', dict(status='completed' if len(records)==len(seeds)
            else 'running', protocol=protocol, evaluations=records))
        print(json.dumps(dict(city=runner.config.inputs.network_name, seed=seed,
            ATT=metrics['all_demand_censored_att_seconds'], completion=metrics['completion_rate'],
            cost=metrics['branch_freeflow_excess_seconds_per_started_vehicle'],
            policy_stats=policy.stats)), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--author-source', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[10042, 10043, 10044])
    args = parser.parse_args()
    run_evaluation(args.config, args.output, args.author_source, seeds=args.seeds)


if __name__ == '__main__':
    main()
