#!/usr/bin/env python3
"""Baseline trainer for the toy next-hop benchmark (GNAC and future dreamer/mbpo/dta)."""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.baselines import METHODS, build_method
from urban_dreamer.learning.baselines import common
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.models.policy import shortest_path_prior_weight, set_shortest_path_prior


def resolve_session_factory(session, step_seconds, backend='traci'):
    """Return the episode session override; None keeps EpisodeRunner's SumoSession default."""
    if session == 'event':
        return functools.partial(SumoEventSession, step_seconds=step_seconds, backend=backend)
    if session == 'microstep':
        return None
    raise ValueError(session)


def log_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    short = {key: value for key, value in record.items() if key not in {'branch_choices', 'choices'}}
    print(json.dumps(short, allow_nan=False), flush=True)


PRIOR_DEFAULTS = dict(sp_prior_weight=0., sp_prior_temperature=30.,
                      sp_prior_hold_episodes=200, sp_prior_off_episode=1000)


def prior_profile(args):
    return {key: getattr(args, key, default) for key, default in PRIOR_DEFAULTS.items()}


def validate_prior_resume(saved_args, args):
    for key, default in PRIOR_DEFAULTS.items():
        if saved_args.get(key, default) != getattr(args, key, default):
            raise ValueError(f'resume changed {key}')


def apply_episode_prior(method, args, episode):
    weight = shortest_path_prior_weight(episode, initial_weight=args.sp_prior_weight,
        hold_episodes=args.sp_prior_hold_episodes, off_episode=args.sp_prior_off_episode)
    if hasattr(method, 'actor'):
        set_shortest_path_prior(method.actor, weight, args.sp_prior_temperature)
    elif weight:
        raise ValueError('shortest-path prior requires a learned actor')
    return weight


def selection_score(records, *, prior_enabled):
    label = 'deterministic_no_prior' if prior_enabled else 'deterministic'
    return float(np.mean([record['selection_score'] for record in records if record['policy'] == label]))


def evaluate(runner, method, output, *, episode, seeds, phase, policies=('deterministic',)):
    allowed = {'deterministic', 'deterministic_no_prior', 'random', 'shortest'}
    if any(policy not in allowed for policy in policies):
        raise ValueError('actor evaluation requires argmax; only random/shortest reference baselines are allowed')
    actor = getattr(method, 'actor', None)
    original_weight = float(getattr(actor, 'prior_weight', 0.))
    temperature = float(getattr(actor, 'prior_temperature_seconds', 30.))
    records = []
    try:
        for policy in policies:
            weight = 0. if policy != 'deterministic' else original_weight
            if actor is not None:
                set_shortest_path_prior(actor, weight, temperature)
            for seed in seeds:
                learned = policy in {'deterministic', 'deterministic_no_prior'}
                select = functools.partial(method.select, explore=False) if learned else policy
                _, _, metrics = runner.run_episode(select, seed=seed, label=policy,
                    on_demand=getattr(method, 'begin_episode', None) if learned else None)
                record = dict(phase=phase, episode=episode, **metrics,
                              sp_prior_weight=weight, sp_prior_temperature=temperature)
                records.append(record)
                log_record(output / 'metrics.jsonl', record)
    finally:
        if actor is not None:
            set_shortest_path_prior(actor, original_weight, temperature)
    return records


def save(runner, method, args, path, episode, best):
    common.save_checkpoint(path, method_name=args.method, episode=episode, best=best,
        method_state=method.state_dict(), args=args, prepared=runner.prepared, config=runner.config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=sorted(METHODS))
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/toy_cross_bottleneck.toml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=150)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--eval-seeds', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--vehicle-limit', type=int, default=0)
    parser.add_argument('--session', choices=('microstep', 'event'), default='microstep')
    parser.add_argument('--step-seconds', type=float, default=1.)
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default='traci')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sp-prior-weight', type=float, default=0.)
    parser.add_argument('--sp-prior-temperature', type=float, default=30.)
    parser.add_argument('--sp-prior-hold-episodes', type=int, default=200)
    parser.add_argument('--sp-prior-off-episode', type=int, default=1000)
    parser.add_argument('--save-episodes', action='store_true',
                        help='Archive every real training episode under episodes/train for world-model training')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save latest checkpoint every N training episodes and at the final episode')
    parser.add_argument('--skip-initial-evaluation', action='store_true',
                        help='Start collecting training episodes immediately; keep periodic and final evaluation')
    parser.add_argument('--skip-reference-baselines', action='store_true',
                        help='Evaluate only the learned or assigned policy; omit random and shortest reference runs')
    args = parser.parse_args(argv)
    if args.episodes < 0 or min(args.eval_every, args.eval_seeds, args.threads, args.checkpoint_every) < 1:
        parser.error('invalid nonpositive experiment budget')
    shortest_path_prior_weight(0, initial_weight=args.sp_prior_weight,
        hold_episodes=args.sp_prior_hold_episodes, off_episode=args.sp_prior_off_episode)
    if not np.isfinite(args.sp_prior_temperature) or args.sp_prior_temperature <= 0:
        parser.error('sp-prior-temperature must be finite and positive')
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite a nonempty experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    runner = common.EpisodeRunner(args, session_factory=resolve_session_factory(args.session, args.step_seconds, args.backend))
    method = build_method(args.method, args, runner)
    evaluation_seeds = [args.seed + 10000 + i for i in range(args.eval_seeds)]
    evaluation_policies = ('deterministic',) if args.skip_reference_baselines else ('random', 'shortest', 'deterministic')
    prior_enabled = args.sp_prior_weight > 0
    learned_policies = ('deterministic', 'deterministic_no_prior') if prior_enabled else ('deterministic',)
    if prior_enabled:
        evaluation_policies += ('deterministic_no_prior',)
    apply_episode_prior(method, args, 0)
    start, best = 1, float('inf')
    if args.resume:
        state = common.load_checkpoint(output / 'latest.pt', method_name=args.method,
            prepared=runner.prepared, config=runner.config, device=runner.device)
        validate_prior_resume(state['args'], args)
        for key in ('seed', 'vehicle_limit', 'actor_lr', 'critic_lr'):
            if state['args'][key] != getattr(args, key):
                raise ValueError(f'resume changed {key}')
        if state['args'].get('session', 'microstep') != args.session:
            raise ValueError('resume changed session')
        if float(state['args'].get('step_seconds', 1.)) != args.step_seconds:
            raise ValueError('resume changed step_seconds')
        if state['args'].get('backend', 'traci') != args.backend:
            raise ValueError('resume changed backend')
        if state['args'].get('skip_reference_baselines', False) != args.skip_reference_baselines:
            raise ValueError('resume changed evaluation reference suite')
        if state['args'].get('save_episodes', False) != args.save_episodes:
            raise ValueError('resume changed episode archival setting')
        if state['args'].get('skip_initial_evaluation', False) != args.skip_initial_evaluation:
            raise ValueError('resume changed initial evaluation setting')
        method.load_state_dict(state['method_state'])
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        if state.get('python_rng') is not None:
            random.setstate(state['python_rng'])
        if state.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        start, best = state['episode'] + 1, state['best']
        apply_episode_prior(method, args, state['episode'])
    else:
        protocol = dict(world_model=args.method in {'dreamer', 'mbpo'},
            macro_latent_critic=args.method == 'dreamer', method=args.method,
            evaluation_policies=list(evaluation_policies),
            session=args.session, step_seconds=args.step_seconds, backend=args.backend,
            episode_archive=dict(enabled=args.save_episodes, directory='episodes/train',
                phases=['train'], source='real_sumo', format='npz',
                includes='all macro windows, actual demand, vehicle snapshots, raw decisions, completion status'),
            checkpoint_every=args.checkpoint_every,
            shortest_path_prior=dict(**prior_profile(args), schedule='hold_then_cosine_to_zero',
                scope='fixed per real episode including all imagination and policy updates',
                cost='junction + candidate free-flow travel + remaining free-flow cost',
                best_selection='deterministic_no_prior' if prior_enabled else 'deterministic',
                mbpo_synthetic_cache='clear on prior weight change; same-weight historical policy reuse remains heuristic'),
            skip_initial_evaluation=args.skip_initial_evaluation,
            reward='shared -active_vehicle_seconds/(N*60); individual -5000 only on each unfinished vehicle trajectory',
            gamma=1., lambda_return=.95, entropy_coefficient=.002,
            demand=dict(law='independent_poisson_per_od', duration_seconds=runner.demand.duration_seconds,
                expected_vehicle_count=runner.demand.expected_vehicle_count,
                od_rates=[dict(origin_index=o, destination_index=d, rate_vehicles_per_second=rate)
                          for o, d, rate in runner.demand.od_rates]),
            reward_mode='global_with_individual_timeout', training_policy='stochastic', evaluation_policy='argmax',
            advantage_normalization='global_scale_only_std_floor_1_then_individual_timeout_clip_2_5',
            global_reward='-active_vehicle_seconds/(N*60)',
            individual_timeout_penalty='-5000 only on unfinished vehicle trajectories at deadline',
            global_critic_observation='flattened physical dynamic flow + progress/active fraction',
            deadline_seconds=runner.deadline, config=runner.config.resolved_snapshot(),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            input_hashes=runner.prepared.input_hashes, evaluation_seeds=evaluation_seeds,
            note=(f'Reviewer baseline {args.method!r}: fixed global-reward setting; '
                  'DTA uses deterministic assignment, learned actors sample during training and use argmax for evaluation.'),
            entropy_note='branch_entropy is entropy of the policy distribution, including in argmax evaluation.',
            reward_note='global_reward_sum is -exposure/(N*60); selection_score includes individual timeout costs for ranking only.')
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        if not args.skip_initial_evaluation:
            baseline = evaluate(runner, method, output, episode=0, seeds=evaluation_seeds, phase='baseline',
                     policies=evaluation_policies)
            best = -selection_score(baseline, prior_enabled=prior_enabled)
        save(runner, method, args, output / 'best.pt', 0, best)
        save(runner, method, args, output / 'initial.pt', 0, best)
        save(runner, method, args, output / 'latest.pt', 0, best)
    completed_episodes = start - 1
    for episode in range(start, args.episodes + 1):
        prior_weight = apply_episode_prior(method, args, episode)
        rows, windows, metrics = runner.run_episode(functools.partial(method.select, explore=True),
            seed=args.seed + episode, collect=getattr(method, 'needs_rows', True), label='stochastic',
            record_windows=getattr(method, 'needs_window_records', False),
            row_flow=getattr(method, 'needs_row_flow', True), on_demand=getattr(method, 'begin_episode', None),
            archive_path=output / 'episodes' / 'train' / f'episode_{episode:06d}.npz' if args.save_episodes else None,
            archive_metadata=dict(method=args.method, phase='train', episode=episode,
                sp_prior_weight=prior_weight, shortest_path_prior=prior_profile(args)))
        for row in rows:
            row['sp_prior_weight'] = prior_weight
        update = method.on_episode(rows, windows, episode=episode)
        completed_episodes = episode
        log_record(output / 'metrics.jsonl', dict(phase='train', episode=episode, **metrics, **update,
            sp_prior_weight=prior_weight, sp_prior_temperature=args.sp_prior_temperature))
        if getattr(method, 'converged', False):
            records = evaluate(runner, method, output, episode=episode, seeds=evaluation_seeds, phase='evaluation',
                               policies=learned_policies)
            score = selection_score(records, prior_enabled=prior_enabled)
            if -score < best:
                best = -score
                save(runner, method, args, output / 'best.pt', episode, best)
            save(runner, method, args, output / 'latest.pt', episode, best)
            break
        if episode % args.eval_every == 0 or episode == args.episodes:
            records = evaluate(runner, method, output, episode=episode, seeds=evaluation_seeds, phase='evaluation',
                               policies=learned_policies)
            score = selection_score(records, prior_enabled=prior_enabled)
            if -score < best:
                best = -score
                save(runner, method, args, output / 'best.pt', episode, best)
        if episode % args.checkpoint_every == 0 or episode == args.episodes:
            save(runner, method, args, output / 'latest.pt', episode, best)
    if args.episodes:
        holdout = [args.seed + 20000 + i for i in range(args.eval_seeds)]
        evaluate(runner, method, output, episode=completed_episodes, seeds=holdout, phase='holdout_final',
                 policies=evaluation_policies)
        state = common.load_checkpoint(output / 'best.pt', method_name=args.method,
            prepared=runner.prepared, config=runner.config, device=runner.device)
        method.load_state_dict(state['method_state'])
        apply_episode_prior(method, args, state['episode'])
        evaluate(runner, method, output, episode=state['episode'], seeds=holdout, phase='holdout_best',
                 policies=learned_policies)
    print(json.dumps({'status': 'completed', 'output': str(output), 'episodes': completed_episodes,
                      'episode_budget': args.episodes}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
