"""Evaluate an immutable online checkpoint against the pilot's paired demands."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_online_pilot import (PilotTrainer, atomic_json, compare_paired,
    evaluation_row, isolated_evaluation, sha256_file)
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.learning.training import evaluation_protocol
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import load_checkpoint, latest_checkpoint_path
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def validate_baseline_policy(before, policy):
    """Legacy baseline rows represent argmax; never pair them with sampled actions."""
    for row in before:
        actual = row.get('metrics', {}).get('evaluation_policy', 'zero_prior_argmax')
        if actual != policy['archive_policy']:
            raise ValueError('Baseline evaluation policy differs from requested policy; '
                             'use --evaluation-policy argmax for legacy paired baselines')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--after-episodes', type=int, required=True)
    parser.add_argument('--evaluation-policy', choices=['sampled', 'argmax'], default='argmax')
    parser.add_argument('--evaluation-temperature', type=float, default=.2)
    args = parser.parse_args()
    if not math.isfinite(args.evaluation_temperature) or args.evaluation_temperature <= 0:
        parser.error('evaluation temperature must be finite and positive')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('interim evaluation requires a fresh output directory')
    torch.set_num_threads(2)
    free, _ = torch.cuda.mem_get_info(1)
    if free < 1024 ** 3:
        raise RuntimeError('GPU1 has less than 1 GiB free; refusing to interfere with training')
    torch.cuda.set_per_process_memory_fraction(.025, 1)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = latest_checkpoint_path(output)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    # The producer atomically replaces latest.pt; this open copies one complete inode.
    with latest_checkpoint_path(args.pilot_dir).open('rb') as source, snapshot.open('wb') as target:
        shutil.copyfileobj(source, target)
    config = load_config(args.pilot_dir / 'run_config.toml')
    config = replace(config, run=replace(config.run, output_dir=output, device='cuda:1'))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    checkpoint = load_checkpoint(output_dir=output,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(prepared))
    progress = checkpoint.training_state['metrics']['online_pilot_progress']
    if len(progress['online']) != args.after_episodes or checkpoint.next_unit != args.after_episodes:
        raise ValueError('snapshot does not match the requested completed online episode count')
    before = progress['before']
    if len(before) != 3:
        raise ValueError('three complete paired baseline demands are required')
    policy = evaluation_protocol(None if args.evaluation_policy == 'argmax' else args.evaluation_temperature,
        prior_weight=config.method.sp_prior_evaluation_weight,
        demand_seed=config.method.evaluation_demand_seed)
    validate_baseline_policy(before, policy)
    identity = dict(pid=os.getpid(), checkpoint_sha256=sha256_file(snapshot),
        checkpoint=str(snapshot), online_episodes_completed=args.after_episodes,
        device='cuda:1', actor_dtype='float32', gpu_memory_fraction=.025,
        evaluation_only=True, evaluation=policy, evaluation_policy=policy['archive_policy'],
        configured_evaluation_horizon_seconds=config.simulation.evaluation_horizon_seconds)
    atomic_json(output / 'protocol.json', identity)
    trainer = PilotTrainer(config=config, prepared_inputs=prepared, topology=topology,
        pilot_log=output / 'evaluation_metrics.jsonl')
    trainer.world_model.flow_projector.__class__ = SparseGraphCodec
    trainer.world_model.flow_projector.register_buffer('sparse_adjacency',
        trainer._tensor(topology.network.adjacency).to_sparse().coalesce(), persistent=False)
    trainer.load_state_dict(checkpoint.training_state)
    trainer.evaluation_temperature = policy['temperature']
    trainer.validate_codec_ready()
    after = []
    try:
        for baseline in before:
            index = baseline['episode_index']
            directory = output / 'evaluation'
            with isolated_evaluation(trainer, directory):
                collected = trainer._run_episode(mode='evaluation', episode_index=index)
            manifest = json.loads((directory / 'evaluation_demands' / f'episode_{index:04d}.json').read_text())
            after.append(evaluation_row(collected, index, manifest['demand_hash']))
            assessment = compare_paired(before[:len(after)], after)
            assessment['note'] = (f'Interim paired evaluation after {args.after_episodes} online episodes. '
                'Censored exposure is not completed travel time; five-episode pilot continues independently.')
            result = dict(status='completed' if len(after) == 3 else 'evaluating',
                protocol=identity, before=before, after=after, assessment=assessment,
                utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
            atomic_json(output / 'result.json', result)
            print(json.dumps(result, allow_nan=False), flush=True)
        if any(trainer.pilot_optimizer_steps.values()):
            raise RuntimeError('evaluation unexpectedly performed optimizer steps')
        if sha256_file(snapshot) != identity['checkpoint_sha256']:
            raise RuntimeError('immutable evaluation checkpoint changed')
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
