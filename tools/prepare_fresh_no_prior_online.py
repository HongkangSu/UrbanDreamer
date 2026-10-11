"""Restart online learning from the same pretrained seed, with routing prior off."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.migrate_bounded_critics import migrate_state, sha256
from tools.migrate_mainline_prior import _assert_equal
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.workflow.checkpoint import (
    CheckpointPhase, _validate_loaded_payload, save_checkpoint,
)
from urban_dreamer.workflow.config import load_config, _semantic_fingerprint
from urban_dreamer.workflow.pipeline import combined_input_hash


def target_config(active, output):
    if active.method.critic_value_mode != 'bounded_cohort_v1':
        raise ValueError('fresh restart must retain the bounded critic contract')
    config = replace(active,
        run=replace(active.run, output_dir=Path(output).resolve()),
        method=replace(active.method, sp_prior_weight=0., sp_prior_evaluation_weight=0.,
                       sp_prior_cost_mode='free_flow'))
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def prepare(active_output, output):
    active_output, output = Path(active_output).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('fresh online restart requires an absent output directory')
    active = load_config(active_output / 'run_config.toml')
    source_config = load_config(active_output / 'source_config.toml')
    source_path = active_output / 'source_checkpoint.pt'
    raw = torch.load(source_path, map_location='cpu', weights_only=True)
    source = _validate_loaded_payload(payload=raw, path=source_path,
        expected_config_fingerprint=source_config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(validate_prepared_inputs(source_config)))
    if source.phase is not CheckpointPhase.ONLINE_TRAINING or source.next_unit != 0:
        raise ValueError('restart requires completed pretraining with zero online episodes')
    config = target_config(active, output)
    if combined_input_hash(validate_prepared_inputs(config)) != source.input_hash:
        raise ValueError('restart inputs differ from the pretrained seed')
    state = migrate_state(source.training_state, config.method,
                          seed=config.run.seed, fresh_online=True)
    _assert_equal(source.training_state['models']['world_model'], state['models']['world_model'])
    for key in ('offline', 'codec_validation'):
        _assert_equal(source.training_state['replay'][key], state['replay'][key])
    if (state['metrics']['online_episodes_completed'] or state['next_environment_episode']
            or state['replay']['online']['episodes']
            or any(opt['state'] for opt in state['optimizers'].values())):
        raise ValueError('fresh online state was not reset')
    output.mkdir(parents=True)
    shutil.copy2(source_path, output / 'source_checkpoint.pt')
    shutil.copy2(active_output / 'source_config.toml', output / 'source_config.toml')
    export_run_config(config, output / 'run_config.toml')
    checkpoint = save_checkpoint(output_dir=output,
        config_fingerprint=config.semantic_fingerprint, input_hash=source.input_hash,
        phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    shutil.copy2(checkpoint, output / 'initial_checkpoint.pt')
    proof = dict(schema_version=1, kind='fresh_bounded_online_without_routing_prior',
        source_output=str(active_output), source_checkpoint=str(source_path),
        source_checkpoint_sha256=sha256(output / 'source_checkpoint.pt'),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), next_unit=0,
        fresh_online=True, dynamic_prior=False, evaluation=dict(policy='argmax', temperature=None, prior_weight=0.),
        preserved=['pretrained_world_model', 'offline_data', 'codec_validation', 'pretraining_counters'],
        reset=['actor', 'bounded_V', 'bounded_Q', 'target_V', 'all_optimizer_states',
               'online_replay', 'online_episode_clock', 'primary_RNG'],
        unchanged=dict(seed=config.run.seed, episodes=config.run.online_training_episodes,
            rollouts=config.resources.imagination_rollouts_per_pass,
            passes=config.resources.imagination_passes_per_episode,
            precision=config.resources.world_model_imagination_precision,
            target_tau=config.method.critic_target_tau,
            terminal_aux_ratio=config.method.critic_terminal_aux_ratio),
        no_prior=dict(training_weight=0., evaluation_weight=0., cost_mode='free_flow'),
        actor_unfrozen=False, training_launched=False)
    (output / 'critic_migration.json').write_text(json.dumps(proof, indent=2)+'\n')
    (output / 'fresh_start_verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--active-output', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(prepare(args.active_output, args.output_dir), indent=2), flush=True)


if __name__ == '__main__':
    main()
