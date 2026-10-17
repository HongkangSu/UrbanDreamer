"""Fit a K=4-constrained DTA assignment, then evaluate fixed paired demands."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tools.evaluate_route_shortest import run_episode
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.route_candidates import RouteCandidates
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def save_state(method, path):
    temporary = path.with_suffix('.tmp')
    torch.save(method.state_dict(), temporary)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error('--iterations must be positive')
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
    runner = SimpleNamespace(network=topology.network, topology=topology, candidates=candidates)
    method = DtaMethod(SimpleNamespace(seed=config.run.seed), runner)
    references = json.loads((args.reference_dir / 'rows.json').read_text())
    reference_protocol = json.loads((args.reference_dir / 'protocol.json').read_text())
    assert prepared.input_hashes == reference_protocol['input_hashes']
    assert config.resolved_snapshot()['simulation'] == reference_protocol['config']['simulation']
    source_files = [Path(__file__), ROOT / 'tools/evaluate_route_shortest.py',
        ROOT / 'urban_dreamer/learning/baselines/dta.py', ROOT / 'urban_dreamer/traffic/route_candidates.py',
        ROOT / 'urban_dreamer/traffic/event_session.py', ROOT / 'urban_dreamer/learning/observations.py',
        ROOT / 'urban_dreamer/learning/sumo_sampling.py']
    protocol = dict(method='dta_current_to_dest', iterations=args.iterations,
        reference_dir=str(args.reference_dir.resolve()), reference_protocol_sha256=sha256_file(args.reference_dir/'protocol.json'),
        source_config_sha256=sha256_file(args.config), input_hashes=prepared.input_hashes,
        config=config.resolved_snapshot(), config_fingerprint=config.semantic_fingerprint,
        training_demand_seed=config.run.seed, training_episodes=list(range(1,args.iterations+1)),
        evaluation_demand_seed=reference_protocol['demand_seed'], evaluation_episodes=[r['episode'] for r in references],
        source_hashes={str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        algorithm='Existing BPR marginal-cost/MSA DTA-SO constrained to the restored static K=4 candidate pool; full-suffix match or marginal-cost fallback.',
        selection='fixed final iteration; evaluation demands never update assignment', policy_interval_seconds=30.)
    path = output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('refusing to resume a different DTA protocol')
    atomic_json(path, protocol)
    export_run_config(config, output / 'run_config.toml')
    training_path = output / 'training.json'
    training = json.loads(training_path.read_text()) if training_path.exists() else []
    if (output / 'latest.pt').exists():
        method.load_state_dict(torch.load(output / 'latest.pt', map_location='cpu', weights_only=True))
        if method.iteration != len(training):
            raise ValueError('DTA assignment/training cursor differs')
    elif training:
        raise ValueError('DTA training history lacks its assignment checkpoint')
    for iteration in range(method.iteration+1, args.iterations+1):
        tick = time.monotonic()
        metrics, windows = run_episode(config, prepared, topology, demand, candidates,
            policy='dta_train', episode=iteration, demand_seed=config.run.seed, output=output/'training',
            selector=partial(method.select, explore=False), on_demand=method.begin_episode, record_windows=True,
            archive_metadata=dict(phase='train', assignment_iteration=method.iteration))
        update = method.on_episode([], windows, episode=iteration)
        del windows
        record = dict(iteration=iteration, metrics=metrics, update=update, seconds=time.monotonic()-tick,
            assigned_suffix_followed=method.route_assignment_follow_count,
            marginal_fallback=method.route_assignment_fallback_count)
        training.append(record)
        save_state(method, output / 'latest.pt')
        atomic_json(training_path, training)
        if iteration in (1,5,10,20,30,40,50) or iteration == args.iterations:
            save_state(method, output / f'assignment_{iteration:03d}.pt')
        print(json.dumps(dict(phase='dta_update', **record)), flush=True)
    rows_path = output / 'rows.json'
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    assert [r['episode'] for r in rows] == [r['episode'] for r in references[:len(rows)]]
    for ref in references[len(rows):]:
        row = run_episode(config, prepared, topology, demand, candidates,
            policy='dta', episode=ref['episode'], demand_seed=reference_protocol['demand_seed'], output=output,
            selector=partial(method.select, explore=False), on_demand=method.begin_episode,
            archive_metadata=dict(phase='evaluation', assignment_iteration=method.iteration))
        for key in ('episode','demand_sha256','vehicle_count','simulator_seed'):
            assert row[key] == ref[key], key
        row.update(assignment_iteration=method.iteration,
            assigned_suffix_followed=method.route_assignment_follow_count,
            marginal_fallback=method.route_assignment_fallback_count)
        assert method.iteration == args.iterations
        rows.append(row)
        atomic_json(rows_path, rows)
        print(json.dumps(dict(phase='dta_evaluation_completed', **row)), flush=True)
    atomic_json(output / 'results.json', dict(protocol=protocol, rows=rows,
        final_training_update=training[-1]['update']))
    atomic_json(output / 'runtime_status.json', dict(phase='completed', policy='dta', iterations=method.iteration,
        episodes=len(rows)))


if __name__ == '__main__':
    main()
