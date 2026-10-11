"""Collect isolated real SUMO evaluation episodes from frozen actor weights.

This diagnostic never resumes training, restores optimizer state, or updates
model parameters. All collection artifacts go to a fresh output directory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--initialize-only', action='store_true')
    args = parser.parse_args()
    if min(args.episodes) < 0 or len(set(args.episodes)) != len(args.episodes):
        parser.error('episode indices must be distinct and nonnegative')
    source = args.source_dir.resolve(strict=True)
    sys.path.insert(0, str(source))
    import torch
    import urban_dreamer.learning.training as training
    from urban_dreamer.models.policy import set_shortest_path_prior
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash

    for name, module in list(sys.modules.items()):
        if name.startswith('urban_dreamer') and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(source):
                raise RuntimeError(f'{name} did not load from requested frozen source')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('collection output must be fresh or empty')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    config = load_config(args.config)
    if output == config.run.output_dir.resolve() or output == args.checkpoint.parent.resolve():
        raise ValueError('diagnostic output cannot be the active run/checkpoint directory')
    prepared = validate_prepared_inputs(config)
    checkpoint_hash = sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload['config_fingerprint'] != config.semantic_fingerprint:
        raise ValueError('checkpoint configuration fingerprint mismatch')
    if payload['input_hash'] != combined_input_hash(prepared):
        raise ValueError('checkpoint input identity mismatch')
    next_unit = int(payload['next_unit'])
    diagnostic_config = replace(config, run=replace(config.run, device=args.device, output_dir=output))
    # The dummy sources prevent corpus loading. _run_episode reads neither;
    # constructor optimizers remain empty and are never stepped or restored.
    trainer = training.MainlineTrainer(config=diagnostic_config, prepared_inputs=prepared,
        topology=load_traffic_topology(prepared), _offline_data=(), _codec_validation_data=None)
    for name in ('world_model', 'actor', 'latent_critic', 'event_critic'):
        model = getattr(trainer, name)
        model.load_state_dict(payload['training_state']['models'][name], strict=True)
        model.eval().requires_grad_(False)
    trainer.metrics['online_episodes_completed'] = float(next_unit)
    set_shortest_path_prior(trainer.actor, 0., config.method.sp_prior_temperature)
    # Older frozen sources force zero evaluation prior inside _run_episode;
    # newer sources expose that field, so normalize it explicitly if present.
    if hasattr(diagnostic_config.method, 'sp_prior_evaluation_weight'):
        trainer.config = replace(diagnostic_config, method=replace(
            diagnostic_config.method, sp_prior_evaluation_weight=0.))
    del payload
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'collection.json'
    manifest = dict(status='initialized', checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_hash, checkpoint_episode=next_unit,
        source_dir=str(source), config=str(args.config.resolve()),
        config_sha256=sha256(args.config), policy='zero_prior_argmax',
        evaluation_demand_seed=config.method.evaluation_demand_seed,
        device=args.device, optimizer_steps=0, episodes=[], requested_episodes=args.episodes,
        output_dir=str(output), initialize_only=args.initialize_only)
    atomic_json(manifest_path, manifest)
    print(json.dumps(dict(status='initialized', checkpoint_episode=next_unit,
                         output_dir=str(output))), flush=True)
    if args.initialize_only:
        return
    began = time.monotonic()
    manifest['status'] = 'collecting'
    atomic_json(manifest_path, manifest)
    try:
        for episode in args.episodes:
            collected = trainer._run_episode(mode='evaluation', episode_index=episode)
            archive = output / 'episodes' / 'evaluation' / f'episode_{episode:06d}.npz'
            if collected.metrics['evaluation_sp_prior_weight'] != 0.:
                raise RuntimeError('collection unexpectedly enabled shortest-path prior')
            manifest['episodes'].append(dict(episode=episode, archive_path=str(archive),
                archive_sha256=sha256(archive), metrics=collected.metrics))
            manifest['elapsed_seconds'] = time.monotonic() - began
            atomic_json(manifest_path, manifest)
            print(json.dumps(manifest['episodes'][-1]), flush=True)
            del collected
        manifest['status'] = 'completed'
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - began
        atomic_json(manifest_path, manifest)


if __name__ == '__main__':
    main()
