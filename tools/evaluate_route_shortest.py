"""Paired static/dynamic shortest route controls in the current SUMO session."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.episode_archive import EpisodeArchive
from urban_dreamer.learning.observations import build_actor_observations, selected_decisions
from urban_dreamer.learning.sumo_sampling import SumoEpisodeSampler
from urban_dreamer.traffic.demand import PoissonDemand, demand_identity
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def select_shortest_routes(pending, observations, frozen, now, *, candidates, dynamic):
    if dynamic:
        costs = observations.prior_costs_seconds
        if costs is None:
            raise ValueError('dynamic shortest routes require observed full-route costs')
    else:
        costs = np.asarray([[candidates.network.free_flow_travel_seconds[list(route)].sum()
                             for route in routes] for routes in observations.routes])
    choices = np.argmin(np.where(observations.mask, costs, np.inf), axis=1).tolist()
    return selected_decisions(pending, observations, choices, now)


def run_episode(config, prepared, topology, demand, candidates, *, policy, episode, demand_seed, output,
                selector=None, on_demand=None, record_windows=False, archive_metadata=None):
    vehicles = demand.sample(seed=demand_seed, episode_index=episode)
    episode_topology = replace(topology, vehicles=vehicles)
    if on_demand is not None:
        on_demand(episode_topology, seed=config.run.seed + episode)
    population = len(vehicles)
    deadline = 30. * math.ceil(config.simulation.evaluation_horizon_seconds / 30.)
    dynamic = policy == 'dynamic_shortest'
    metadata = dict(city=config.inputs.network_name, method=policy, policy=policy,
        seed=config.run.seed, episode=episode, phase='evaluation', session_protocol='event',
        routing_mode='current_to_dest', total_vehicle_count=population,
        input_hashes=prepared.input_hashes, edge_ids=list(topology.network.edge_ids),
        config=config.resolved_snapshot(), config_fingerprint=config.semantic_fingerprint,
        physics_identity=physics_identity(config.simulation), backend=config.simulation.backend,
        step_seconds=config.simulation.observation_step_seconds,
        demand_seed=demand_seed, demand_index=episode, demand_sha256=demand_identity(vehicles),
        realized_demand=[asdict(vehicle) for vehicle in vehicles],
        demand=dict(law='independent_poisson_per_od', duration_seconds=demand.duration_seconds,
                    expected_vehicle_count=demand.expected_vehicle_count),
        dynamic_speed_floor_mps=config.method.sp_prior_dynamic_speed_floor_mps,
        candidate_cost=('observed_route_travel_seconds' if dynamic else
                        'learned_dta_marginal_cost' if selector is not None else 'free_flow_route_travel_seconds'))
    metadata.update(archive_metadata or {})
    archive = EpisodeArchive(metadata)
    started = time.monotonic()
    session = SumoEventSession(config, prepared, episode_topology, episode_index=episode,
        backend=config.simulation.backend, step_seconds=config.simulation.observation_step_seconds)
    policy_count = turns = 0
    curve = []
    windows = []
    with SumoEpisodeSampler(session=session, candidates=candidates,
            deadline_seconds=deadline, total_vehicle_count=population,
            context_mode='observation', sort_pending=True, pad_empty=False, advance_empty=True,
            wall_timeout_seconds=config.simulation.episode_wall_timeout_seconds,
            observation_builder=partial(build_actor_observations,
                prior_cost_mode='dynamic' if dynamic else 'free_flow',
                prior_speed_floor_mps=config.method.sp_prior_dynamic_speed_floor_mps,
                prior_speed_cap_mps=config.simulation.vehicle_max_speed_mps)) as sampler:
        archive.start(sampler.state)
        decide = partial(select_shortest_routes, candidates=candidates, dynamic=dynamic)
        def callback(pending, observations, frozen, now, index, context):
            if selector is not None:
                choices = selector(pending, observations, frozen, now)
                return selected_decisions(pending, observations, choices, now)
            return decide(pending, observations, frozen, now)
        for sample in sampler.windows(callback):
            result = sample.result
            archive.append(result)
            policy_count += len(result.decisions)
            turns += len(result.conditioning_decisions)
            state = result.next_state
            if record_windows:
                windows.append(dict(next_flow=np.asarray(state['flow'], dtype=np.float32).copy()))
            curve.append(dict(time=result.end_time, started=state['started_vehicle_count'],
                arrived=state['arrived_vehicle_count'], active=state['active_vehicle_count'],
                cumulative_vehicle_seconds=state['cumulative_vehicle_seconds'],
                observed_mean_seconds=sampler.accounting.mean_seconds))
            if sample.index % 10 == 0:
                atomic_json(output / 'runtime_status.json', dict(phase='running', policy=policy,
                    episode=episode, simulation_time=result.end_time, population=population,
                    arrived=state['arrived_vehicle_count'], active=state['active_vehicle_count'],
                    wall_seconds=time.monotonic() - started))
                print(json.dumps(dict(policy=policy, episode=episode, simulation_time=result.end_time,
                    arrived=state['arrived_vehicle_count'], population=population,
                    wall_seconds=round(time.monotonic() - started, 2))), flush=True)
        state = sampler.state
        durations = np.asarray(list(session.travel_times.values()), dtype=np.float64)
        metrics = dict(policy=policy, episode=episode, demand_seed=demand_seed,
            simulator_seed=config.run.seed + episode, demand_sha256=demand_identity(vehicles),
            vehicle_count=population, started_vehicle_count=sampler.accounting.started_count,
            arrived_vehicle_count=len(durations), unfinished=population - len(durations),
            never_departed=sum(not vehicle['departed'] for vehicle in state['vehicles']),
            completion_rate=len(durations) / population if population else 1.,
            observed_mean_seconds=sampler.accounting.mean_seconds,
            observed_mean_minutes=sampler.accounting.mean_seconds / 60.,
            mean_arrived_travel_seconds=float(durations.mean()) if durations.size else None,
            p90_arrived_travel_seconds=float(np.percentile(durations, 90)) if durations.size else None,
            p95_arrived_travel_seconds=float(np.percentile(durations, 95)) if durations.size else None,
            cumulative_vehicle_seconds=sampler.accounting.cumulative_vehicle_seconds,
            policy_decisions=policy_count, conditioning_turns=turns,
            simulation_end_seconds=float(state['time']), deadline_seconds=deadline,
            physical_completed=bool(session.done), wall_seconds=time.monotonic() - started)
        archive.save(output / 'episodes' / f'episode_{episode:03d}.npz', metrics=metrics,
            travel_times=session.travel_times, completed=bool(session.done), deadline_seconds=deadline)
        atomic_json(output / 'curves' / f'episode_{episode:03d}.json', curve)
        return (metrics, windows) if record_windows else metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', choices=('shortest', 'dynamic_shortest'), required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--demand-seed', type=int)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = load_config(args.config)
    config = replace(original, run=replace(original.run, output_dir=output, device='cpu'),
        simulation=replace(original.simulation, routing_mode='current_to_dest'))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        name: getattr(config, name) for name in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    candidates = RouteCandidates(topology.network, traffic_model=config.simulation.traffic_model)
    demand = PoissonDemand.from_vehicles(topology.vehicles,
        duration_seconds=config.simulation.demand_duration_seconds)
    demand_seed = config.method.evaluation_demand_seed if args.demand_seed is None else args.demand_seed
    source_files = [Path(__file__), ROOT / 'urban_dreamer/traffic/route_candidates.py',
        ROOT / 'urban_dreamer/traffic/event_session.py', ROOT / 'urban_dreamer/learning/observations.py',
        ROOT / 'urban_dreamer/learning/sumo_sampling.py', ROOT / 'urban_dreamer/learning/dynamic_prior.py']
    protocol = dict(policy=args.policy, episodes=args.episodes, demand_seed=demand_seed,
        simulation_seed=config.run.seed, input_hashes=prepared.input_hashes,
        config=config.resolved_snapshot(), config_fingerprint=config.semantic_fingerprint,
        source_config_sha256=sha256_file(args.config),
        source_hashes={str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        action_space='historical_four_penalty_dijkstra_current_to_dest_routes',
        policy_interval_seconds=30., dynamic_speed_floor_mps=config.method.sp_prior_dynamic_speed_floor_mps)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('refusing to resume a different route evaluation protocol')
    atomic_json(protocol_path, protocol)
    export_run_config(config, output / 'run_config.toml')
    rows_path = output / 'rows.json'
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    if [row['episode'] for row in rows] != list(range(len(rows))) or len(rows) > args.episodes:
        raise ValueError('invalid completed evaluation cursor')
    for episode in range(len(rows), args.episodes):
        row = run_episode(config, prepared, topology, demand, candidates,
            policy=args.policy, episode=episode, demand_seed=demand_seed, output=output)
        rows.append(row)
        atomic_json(rows_path, rows)
        print(json.dumps(dict(phase='episode_completed', **row)), flush=True)
    atomic_json(output / 'results.json', dict(protocol=protocol, rows=rows))
    atomic_json(output / 'runtime_status.json', dict(phase='completed', episodes=len(rows), policy=args.policy))


if __name__ == '__main__':
    main()
