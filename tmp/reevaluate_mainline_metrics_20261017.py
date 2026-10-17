"""Frozen-source CPU actor-only deterministic re-evaluation with shared metrics."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'checkpoint', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--metric-helper', type=Path, default=Path(__file__).resolve().parents[1] /
                        'urban_dreamer/learning/routing_evaluation_metrics.py')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    import torch
    from urban_dreamer.learning.training import MainlineTrainer
    from urban_dreamer.learning.episode_archive import EpisodeArchive
    from urban_dreamer.learning.sumo_sampling import resolve_session_factory
    from urban_dreamer.models.policy import NextHopActor
    from urban_dreamer.traffic.demand import PoissonDemand
    from urban_dreamer.traffic.topology import load_traffic_topology, NextHopCandidates
    from urban_dreamer.traffic.route_candidates import RouteCandidates
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash
    spec = importlib.util.spec_from_file_location('reevaluation_metrics', args.metric_helper)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch.set_num_threads(1)
    config = load_config(args.config)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    if payload['config_fingerprint'] != config.semantic_fingerprint:
        raise ValueError('checkpoint/config fingerprint mismatch')
    if payload['input_hash'] != combined_input_hash(prepared):
        raise ValueError('checkpoint/input hash mismatch')
    state = payload['training_state']
    actor = NextHopActor().eval().requires_grad_(False)
    actor.load_state_dict(state['models']['actor'], strict=True)
    saved_metrics = dict(state['metrics'])
    episode = int(saved_metrics.get('online_episodes_completed', payload['next_unit']))
    del state, payload
    gc.collect()
    trainer = MainlineTrainer.__new__(MainlineTrainer)
    trainer.config = config
    trainer.prepared_inputs = prepared
    trainer.topology = topology
    trainer.network = topology.network
    trainer.edge_count = topology.network.edge_count
    trainer.device = torch.device('cpu')
    trainer.actor = actor
    trainer.evaluation_temperature = None
    trainer.rng = np.random.default_rng(42)
    trainer.demand = PoissonDemand.from_vehicles(topology.vehicles,
        duration_seconds=config.simulation.demand_duration_seconds)
    candidate_cls = RouteCandidates if config.simulation.routing_mode == 'current_to_dest' else NextHopCandidates
    trainer.candidates = candidate_cls(topology.network, traffic_model=config.simulation.traffic_model)
    trainer.metrics = saved_metrics
    trainer._next_environment_episode = 0
    base_session = resolve_session_factory('event', 1., config.simulation.backend)
    def paired_session(*, config, prepared_inputs, topology, episode_index):
        return base_session(config=replace(config, run=replace(config.run, seed=episode_index)),
            prepared_inputs=prepared_inputs, topology=topology, episode_index=0)
    trainer._session_factory = paired_session
    original_save = EpisodeArchive.save
    def save_with_metrics(archive, path, *, metrics, travel_times, completed, deadline_seconds):
        final = archive._states[-1]
        destinations = {str(v['vehicle_id']): int(v['destination_edge']) for v in final['vehicles']}
        cost = helper.DiversionCost(trainer.candidates)
        cost.add([SimpleNamespace(**event) for event in archive._events], destinations)
        extra = cost.metrics(int(metrics['evaluation_started_vehicle_count']))
        final_rows = []
        for row in final['vehicles']:
            row = dict(row)
            if row['arrived'] and row.get('arrival_time') is None:
                departure = row.get('actual_departure_time')
                if departure is None:
                    departure = row['departure_time']
                row['arrival_time'] = float(departure) + float(travel_times[str(row['vehicle_id'])])
            final_rows.append(row)
        extra.update(helper.all_demand_censored_metrics(
            [SimpleNamespace(**v) for v in archive.metadata['realized_demand']], final_rows,
            min(float(final['time']), float(deadline_seconds))))
        metrics.update({'evaluation_' + key: value for key, value in extra.items()})
        archive.metadata.update(diversion_cost_definition=helper.DIVERSION_COST_DEFINITION,
            all_demand_censored_definition=helper.ALL_DEMAND_CENSORED_DEFINITION)
        return original_save(archive, path, metrics=metrics, travel_times=travel_times,
                             completed=completed, deadline_seconds=deadline_seconds)
    EpisodeArchive.save = save_with_metrics
    identity = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_episode=episode, source=str(args.source.resolve()), config=str(args.config.resolve()),
        config_fingerprint=config.semantic_fingerprint, config_sha256=sha256(args.config),
        input_hashes=prepared.input_hashes, policy='zero_prior_argmax', device='cpu',
        seeds=[10042, 10043, 10044], demand_seed=42, sumo_seeds=[10042, 10043, 10044],
        session_episode_index=0, effective_horizon_seconds=trainer._episode_end_time('evaluation'),
        traffic_model=config.simulation.traffic_model, backend=config.simulation.backend,
        routing_mode=config.simulation.routing_mode, shared_route_cache_directory=os.environ.get('URB_ROUTE_CACHE_DIR'),
        metric_helper_sha256=sha256(args.metric_helper),
        source_training_sha256=sha256(args.source / 'urban_dreamer/learning/training.py'),
        diversion_cost_definition=helper.DIVERSION_COST_DEFINITION,
        all_demand_censored_definition=helper.ALL_DEMAND_CENSORED_DEFINITION)
    (args.output / 'protocol.json').write_text(json.dumps(identity, indent=2) + '\n')
    print(json.dumps(identity), flush=True)
    rows = []
    for seed in identity['seeds']:
        directory = args.output / f'seed_{seed}'
        trainer.config = replace(config, run=replace(config.run, output_dir=directory, device='cpu'),
            method=replace(config.method, evaluation_demand_seed=42, sp_prior_evaluation_weight=0.))
        start = time.monotonic()
        collected = trainer._run_episode(mode='evaluation', episode_index=seed)
        metrics = dict(collected.metrics)
        normalized = {key.removeprefix('evaluation_'): value for key, value in metrics.items()}
        normalized['observed_mean_seconds'] = metrics['evaluation_cumulative_vehicle_seconds'] / max(metrics['evaluation_started_vehicle_count'], 1)
        demand = json.loads((directory / f'evaluation_demands/episode_{seed:04d}.json').read_text())
        row = dict(seed=seed, checkpoint_episode=episode, demand_hash=demand['demand_hash'],
            **normalized, evaluator_wall_seconds=time.monotonic() - start,
            archive=str(directory / f'episodes/evaluation/episode_{seed:06d}.npz'))
        rows.append(row)
        result = dict(status='completed' if len(rows) == 3 else 'running', method='urban_dreamer',
                      city=config.inputs.network_name, protocol=identity, evaluations=rows)
        temporary = args.output / 'result.json.tmp'
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output / 'result.json')
        print(json.dumps(row), flush=True)
        del collected
        gc.collect()


if __name__ == '__main__':
    main()
