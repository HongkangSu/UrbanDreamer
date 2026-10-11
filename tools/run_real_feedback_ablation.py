"""Bounded real-only credit diagnostic from a frozen mainline actor and Adam.

This experiment does not replace the world-model training pipeline. A freshly
initialized full-state critic is explicit; the baseline's real return, timeout,
and actor-update definitions are reused without modifications.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_toy_actor_critic as baseline
from tools.run_entropy_ablation import tree_hash
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.demand import demand_identity


def record(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')
    print(json.dumps(value, sort_keys=True), flush=True)


def evaluate(runner, output):
    """Use exactly the mainline sampler and stochastic evaluation seed protocol."""
    adapter = MainlineTrainer.__new__(MainlineTrainer)
    adapter.device = torch.device('cpu')
    adapter.config = replace(runner.config, run=replace(runner.config.run,
        output_dir=output, device='cpu', seed=314159))
    adapter.prepared_inputs = runner.prepared
    adapter.topology, adapter.network = runner.topology, runner.network
    adapter.candidates, adapter.demand, adapter.actor = runner.candidates, runner.demand, runner.actor
    adapter._session_factory = None
    rows = []
    for index in range(3):
        adapter._next_environment_episode = index
        random.seed(271828 + index)
        np.random.seed(271828 + index)
        torch.manual_seed(271828 + index)
        started = time.monotonic()
        episode = adapter._run_episode(mode='online', episode_index=index)
        rows.append(dict(index=index, demand_hash=demand_identity(runner.demand.sample(
            seed=314159, episode_index=index)), population=episode.total_vehicle_count,
            metrics=episode.metrics, wall_seconds=time.monotonic() - started))
        atomic_json(output / 'results.json', rows)
        print(json.dumps(dict(phase='evaluation', directory=str(output), **rows[-1])), flush=True)
    return rows


def aggregate(rows):
    population = sum(row['population'] for row in rows)
    unfinished = sum(row['metrics']['online_nonarrival_count'] for row in rows)
    return dict(population=population, unfinished=unfinished,
        completion_rate=1 - unfinished / population,
        weighted_global_reward=sum(row['metrics']['online_global_reward_sum'] * row['population']
                                   for row in rows) / population)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/convergence_diagnosis_20261005/frozen')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/convergence_diagnosis_20261005/real_feedback')
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--reference-before', type=Path,
        default=ROOT / 'outputs/convergence_diagnosis_20261005/entropy/before/results.json')
    args = parser.parse_args()
    if not 1 <= args.episodes <= 12:
        parser.error('bounded diagnostic requires 1..12 episodes')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'protocol.json').exists():
        raise RuntimeError('output already contains a run; use a fresh output directory')
    checkpoint_path = args.source / 'checkpoints/latest.pt'
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = source['training_state']
    start = int(source['next_unit'])
    options = SimpleNamespace(config=args.source / 'run_config.toml', output_dir=args.output,
        seed=42, device='cpu', threads=args.threads, actor_lr=3e-4, global_critic_lr=3e-4,
        vehicle_limit=0, session='event', step_seconds=1., backend='libsumo', forbid_revisit=False,
        sp_prior_weight=0., sp_prior_temperature=30., sp_prior_hold_episodes=200, sp_prior_off_episode=1000)
    runner = baseline.RealActorCritic(options)
    runner.actor.load_state_dict(state['models']['actor'], strict=True)
    runner.actor_optimizer.load_state_dict(state['optimizers']['actor'])
    actor_lrs = [group['lr'] for group in runner.actor_optimizer.param_groups]
    assert tree_hash(runner.actor.state_dict()) == tree_hash(state['models']['actor'])
    assert tree_hash(runner.actor_optimizer.state_dict()) == tree_hash(state['optimizers']['actor'])
    # Keep the baseline learning code, with sampling semantics matched to mainline.
    sampler_class = baseline.SumoEpisodeSampler
    def mainline_sampler(**kwargs):
        kwargs.update(context_mode='observation', sort_pending=True,
                      advance_empty=True, max_windows=int(runner.deadline / 30))
        return sampler_class(**kwargs)
    baseline.SumoEpisodeSampler = mainline_sampler
    protocol = dict(source=str(checkpoint_path), source_sha256=sha256_file(checkpoint_path),
        source_online_episodes=start, episodes=args.episodes,
        training_demand_seed=42, training_demand_indices=list(range(start, start + args.episodes)),
        evaluation_demand_seed=314159, evaluation_indices=[0, 1, 2],
        evaluation_policy_seeds=[271828, 271829, 271830], device='cpu', threads=args.threads,
        actor_optimizer='strict copied Adam including moments and step', actor_lrs=actor_lrs,
        fresh_critic='GlobalTrafficCritic full real network state', critic_init_seed=42 + 73861,
        critic_lr=3e-4, shortest_path_prior_weight=0., actor_updates_per_episode=1,
        world_model_updates=0, imagined_updates=0, entropy_coef=.002,
        baseline_update='unmodified RealActorCritic.update: macro lambda advantage plus full real individual timeout',
        sampling_adjustment='observation context, sorted pending, advance empty; matches mainline',
        initial_actor_hash=tree_hash(runner.actor.state_dict()),
        initial_actor_adam_hash=tree_hash(runner.actor_optimizer.state_dict()),
        initial_new_critic_hash=tree_hash(runner.global_critic.state_dict()))
    atomic_json(args.output / 'protocol.json', protocol)
    del state, source
    before = evaluate(runner, args.output / 'before')
    if args.reference_before.exists():
        reference = json.loads(args.reference_before.read_text())
        for actual, expected in zip(before, reference, strict=True):
            assert (actual['index'], actual['demand_hash'], actual['population']) == (
                expected['index'], expected['demand_hash'], expected['population'])
            for key in ('online_nonarrival_count', 'online_global_reward_sum', 'online_events'):
                if actual['metrics'][key] != expected['metrics'][key]:
                    raise RuntimeError(f'before evaluation differs from reference: {key}')
        atomic_json(args.output / 'before_matched.json', dict(exact=True, reference=str(args.reference_before)))
    training = []
    for index in range(start, start + args.episodes):
        begin = time.monotonic()
        rows, metrics = runner.run_episode(policy='stochastic', seed=index, collect=True)
        stats = runner.update(rows)
        value = dict(episode_index=index, episodes_completed=index - start + 1,
            collection=metrics, learning=stats, actor_hash=tree_hash(runner.actor.state_dict()),
            timeout_events=sum(row['timeout_return'] < 0 for row in rows),
            seconds=time.monotonic() - begin)
        runner.save(args.output / f'checkpoint_{index + 1:06d}.pt', index + 1, None)
        training.append(value)
        record(args.output / 'training.jsonl', value)
        atomic_json(args.output / 'progress.json', dict(episodes_completed=len(training), last=value))
    after = evaluate(runner, args.output / 'after')
    atomic_json(args.output / 'result.json', dict(protocol=protocol, before=aggregate(before),
        after=aggregate(after), training=training,
        note='12 real on-policy actor updates and a new real-state critic; not a convergence proof or production replacement.'))


if __name__ == '__main__':
    main()
