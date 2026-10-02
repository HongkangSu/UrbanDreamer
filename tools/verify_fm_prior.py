"""Real SUMO small-budget FM lifecycle and paired-demand ablation verification."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.training import MainlineTrainer
import urban_dreamer.learning.training as training
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import MainlinePipeline, combined_input_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    # Budget reductions are confined to this verification harness.
    training.ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE = 1
    training.IMAGINED_PASSES_PER_EPISODE = 1
    training.IMAGINED_ROLLOUTS_PER_PASS = 1
    base = load_config(ROOT / 'configs/toy_cross_bottleneck.toml')
    variants = {}
    for name, noise in [('full', .1), ('no_noise', 0.)]:
        variant_directory = args.output_dir.resolve() / name
        variant_directory.mkdir(parents=True, exist_ok=True)
        directory = variant_directory / 'run'
        config = replace(base, run=replace(base.run, output_dir=directory,
            device=args.device, warmup_episodes=1, world_model_initialization_epochs=2,
            online_training_episodes=1, evaluation_episodes=2),
            simulation=replace(base.simulation, demand_duration_seconds=30.,
                world_model_intervals_per_episode=10, warmup_horizon_seconds=300,
                warmup_completion_drain_seconds=0, training_completion_buffer_seconds=0,
                evaluation_horizon_seconds=300),
            resources=replace(base.resources, world_model_initialization_batch_size=1,
                world_model_initialization_sequence_length=3, online_world_model_batch_size=1),
            method=replace(base.method, prior_noise_scale=noise, supervision_increment_updates=1))
        # Loading the actual TOML derives the fingerprint for the modified budget.
        snapshot = config.resolved_snapshot()
        toml = '\n\n'.join('[' + section + ']\n' + '\n'.join(
            key + ' = ' + json.dumps(value) for key, value in values.items())
            for section, values in snapshot.items())
        config_path = variant_directory / 'verification.toml'
        config_path.write_text(toml + '\n')
        config = load_config(config_path)
        prepared = validate_prepared_inputs(config)
        topology = load_traffic_topology(prepared)
        topology = replace(topology, vehicles=topology.vehicles[:4])
        def factory():
            return MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
        before = time.monotonic()
        result = MainlinePipeline(config=config,
            input_hash=combined_input_hash(prepared) + '-first4-poisson-smoke-v1',
            trainer_factory=factory).run()
        hashes = [json.loads(path.read_text())['demand_hash']
                  for path in sorted((directory / 'evaluation_demands').glob('episode_*.json'))]
        variants[name] = {'summary': result, 'demand_hashes': hashes,
                          'wall_seconds': time.monotonic() - before}
        print(json.dumps({'variant': name, **variants[name]}, sort_keys=True), flush=True)
    assert variants['full']['demand_hashes'] == variants['no_noise']['demand_hashes']
    assert len(variants['full']['demand_hashes']) == 2
    assert all(row['summary']['status'] == 'completed' for row in variants.values())
    report = {'variants': variants, 'same_evaluation_demands': True,
              'wall_seconds': time.monotonic() - started,
              'scope': 'four-reference-vehicle Poisson Toy; one warmup, two initialization epochs, one online episode, two fixed evaluation demands per variant',
              'limitation': 'Smoke budgets verify execution and demand pairing; they do not establish learned policy quality or an ablation effect.'}
    report_path = args.output_dir.resolve() / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'Report: {report_path}', flush=True)


if __name__ == '__main__':
    main()
