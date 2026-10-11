"""Prepare an isolated bounded-critic checkpoint; never launch or stop training."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_complete_pretrain import export_run_config
from urban_dreamer.models.policy import CentralizedLatentValueCritic, NextHopActor
from urban_dreamer.models.event_critic import EventActionCritic
from urban_dreamer.workflow.config import load_config, _semantic_fingerprint
from urban_dreamer.workflow.checkpoint import _validate_loaded_payload, save_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def migrate_state(state, method, *, seed, fresh_online=False):
    if method.critic_value_mode != 'bounded_cohort_v1' or method.disagreement_weight != 0.:
        raise ValueError('migration requires bounded physical-only critics')
    if state['reward_contract'] != 'causal_cumulative_mean_global_v1':
        raise ValueError('first migrate the physical reward contract explicitly')
    if state['metrics'].get('critic_value_mode', 'legacy') != 'legacy' or 'target_latent_critic' in state['models']:
        raise ValueError('source already has a nonlegacy critic contract')
    if fresh_online and (state['metrics'].get('online_episodes_completed', 0) != 0
                         or state['next_environment_episode'] != 0 or state['replay']['online']['episodes']):
        raise ValueError('fresh online training must start from a zero-online pretrained seed')
    target = copy.deepcopy(state)
    # Fresh raw heads have a new meaning; never sigmoid-transform old values.
    # CPU-only initialization must not advance the caller's CPU or CUDA streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        v = CentralizedLatentValueCritic(bounded=True)
        q = EventActionCritic(bounded=True)
        actor = NextHopActor() if fresh_online else None
    target['models'].update(latent_critic=v.state_dict(), event_critic=q.state_dict(),
                            target_latent_critic=copy.deepcopy(v.state_dict()))
    for name in ('latent_critic', 'event_critic'):
        target['optimizers'][name]['state'] = {}
    auxiliary_seed = seed + 91873
    target['rng'].update(
        critic_aux_numpy=copy.deepcopy(np.random.default_rng(auxiliary_seed).bit_generator.state),
        critic_aux_torch_cpu=torch.Generator(device='cpu').manual_seed(auxiliary_seed).get_state(),
        critic_aux_torch_cuda=[torch.Generator(device=f'cuda:{i}').manual_seed(auxiliary_seed+i).get_state()
                               for i in range(torch.cuda.device_count())])
    target['metrics'].update(critic_value_mode=method.critic_value_mode,
        critic_target_tau=method.critic_target_tau, critic_terminal_aux_ratio=method.critic_terminal_aux_ratio,
        critic_reset_at_online_episode=state['metrics'].get('online_episodes_completed', 0),
        critic_reset_seed=seed)
    if fresh_online:
        target['models']['actor'] = actor.state_dict()
        for optimizer in target['optimizers'].values():
            optimizer['state'] = {}
        online = target['replay']['online']
        online.update(episodes=[], sample_order=[], sample_cursor=0, next_storage_id=0)
        target['metrics'].pop('online_pilot_progress', None)
        target['metrics'].update(online_episodes_completed=0., evaluation_episodes_completed=0.,
            evaluation_t_all_sum=0., evaluation_t_all_mean=0., evaluation_vehicle_minutes_sum=0.,
            evaluation_vehicle_count_sum=0., evaluation_t_all_vehicle_weighted=0.,
            actor_loss_last=0., latent_critic_loss_last=0., event_critic_loss_last=0.,
            fresh_online_start=True, actor_initialization_seed=seed)
        target['next_environment_episode'] = 0
        target['rng'].update(python_random=random.Random(seed).getstate(),
            numpy_global=list(np.random.RandomState(seed).get_state()),
            numpy_generator=copy.deepcopy(np.random.default_rng(seed).bit_generator.state),
            torch_cpu=torch.Generator(device='cpu').manual_seed(seed).get_state(),
            torch_cuda=[torch.Generator(device=f'cuda:{i}').manual_seed(seed).get_state()
                        for i in range(torch.cuda.device_count())])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--target-tau', type=float, default=.01)
    parser.add_argument('--terminal-aux-ratio', type=float, default=.25)
    parser.add_argument('--dynamic-prior', action='store_true',
                        help='enable current-speed shortest-path guidance for remaining training and evaluation')
    parser.add_argument('--fresh-online', action='store_true',
                        help='retain pretrained world weights, reset actor, optimizers, online state and RNG')
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('migration requires a fresh directory')
    source_config = load_config(args.config)
    if source_config.method.critic_value_mode != 'legacy':
        raise ValueError('source config must use legacy critics')
    method = replace(source_config.method, critic_value_mode='bounded_cohort_v1',
                     critic_target_tau=args.target_tau, critic_terminal_aux_ratio=args.terminal_aux_ratio)
    if args.dynamic_prior:
        method = replace(method, sp_prior_cost_mode='dynamic', sp_prior_weight=1.,
            sp_prior_evaluation_weight=1., sp_prior_temperature=30.,
            sp_prior_hold_episodes=source_config.run.online_training_episodes,
            sp_prior_off_episode=source_config.run.online_training_episodes+1)
    target_config = replace(source_config, method=method,
                            run=replace(source_config.run, output_dir=output))
    target_config = replace(target_config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(target_config, key) for key in ('inputs','run','simulation','resources','method','data','codec')}))
    output.mkdir(parents=True)
    export_run_config(target_config, output / 'run_config.toml')
    # Validate the exported target including new settings before checkpoint work.
    target_config = load_config(output / 'run_config.toml')
    shutil.copyfile(args.checkpoint, output / 'source_checkpoint.pt')
    shutil.copyfile(args.config, output / 'source_config.toml')
    raw = torch.load(output / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
    source = _validate_loaded_payload(payload=raw, path=output / 'source_checkpoint.pt',
        expected_config_fingerprint=source_config.semantic_fingerprint, expected_input_hash=raw['input_hash'])
    if args.fresh_online and source.next_unit != 0:
        raise ValueError('fresh online source must be the completed-pretraining zero-episode seed')
    state = migrate_state(source.training_state, target_config.method, seed=args.seed, fresh_online=args.fresh_online)
    path = save_checkpoint(output_dir=output, config_fingerprint=target_config.semantic_fingerprint,
        input_hash=source.input_hash, phase=source.phase, next_unit=source.next_unit, training_state=state)
    protocol = dict(kind='fresh_bounded_critic_initialization', schema_version=1,
        source_checkpoint=str(args.checkpoint.resolve()), source_checkpoint_sha256=sha256(output/'source_checkpoint.pt'),
        checkpoint=str(path), checkpoint_sha256=sha256(path), next_unit=source.next_unit,
        seed=args.seed, critic_value_mode=method.critic_value_mode, target_tau=method.critic_target_tau,
        terminal_aux_ratio=method.critic_terminal_aux_ratio, actor_unfrozen=False, training_launched=False,
        dynamic_prior=args.dynamic_prior,
        fresh_online=args.fresh_online,
        preserved=(['world_model','offline_replay','codec_validation','pretraining_counters'] if args.fresh_online
                   else ['world_model','actor','noncritic_optimizers','replay','original_rng','episode_and_entropy_clocks']),
        reset=(['actor','latent_critic','event_critic','all_optimizer_states','online_replay','online_clocks','primary_rng']
               if args.fresh_online else ['latent_critic','event_critic','both_critic_optimizer_states']),
        added=['target_latent_critic','private_auxiliary_rng'],
        release_gate='Isolated candidate only; learned-quality protocol must pass before actor training.')
    (output / 'critic_migration.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
