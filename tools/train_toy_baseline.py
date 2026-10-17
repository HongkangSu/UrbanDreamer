#!/usr/bin/env python3
"""Baseline trainer for the toy next-hop benchmark (GNAC and future dreamer/mbpo/dta)."""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.baselines import METHODS, build_method
from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.collection_policy import CollectionPolicy
from urban_dreamer.learning.entropy_schedule import entropy_schedule_profile, entropy_schedule_episodes, SCHEDULE_FIELDS
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.models.policy import NextHopActor, shortest_path_prior_weight, set_shortest_path_prior
from urban_dreamer.workflow.config import load_config


def resolve_session_factory(session, step_seconds, backend='traci', *, forbid_revisit=False):
    """Return the episode session override; None keeps EpisodeRunner's SumoSession default."""
    if forbid_revisit and session != 'event':
        raise ValueError('forbid_revisit requires event session')
    if session == 'event':
        options = {'forbid_revisit': True} if forbid_revisit else {}
        return functools.partial(SumoEventSession, step_seconds=step_seconds, backend=backend, **options)
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
    if saved_args.get('reward_objective', 'legacy_exposure') != common.reward_objective(args):
        raise ValueError('resume changed reward_objective; migrate checkpoint explicitly')
    if saved_args.get('global_reward_only', False) != getattr(args, 'global_reward_only', False):
        raise ValueError('resume changed global_reward_only')
    if saved_args.get('forbid_revisit', False) != getattr(args, 'forbid_revisit', False):
        raise ValueError('resume changed forbid_revisit')
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


def apply_episode_entropy(method, method_config, *, completed_episodes, total_episodes):
    """Freeze one coefficient for all actor updates in the next real episode.

    Historical baselines used fixed .002 even when the mainline config named
    population scaling. Only explicit annealed/off modes change that behavior.
    Independent Q-learning and assignment have no entropy regularization loss.
    """
    profile = entropy_schedule_profile(method_config,
        completed_episodes=completed_episodes, total_episodes=max(total_episodes, 1))
    applicable = hasattr(method, 'entropy_coef')
    if applicable:
        method.entropy_coef = profile['coefficient']
    method.entropy_profile = dict(profile, applicable=applicable)
    return dict(entropy_mode=profile['mode'] if applicable else 'not_applicable',
        entropy_configured_mode=profile['mode'],
        entropy_coefficient=profile['coefficient'] if applicable else 0.,
        entropy_schedule_progress=profile['progress'],
        entropy_completed_episodes=profile['completed_episodes'],
        entropy_total_episodes=profile['total_episodes'])


def selection_score(records, *, prior_enabled, temperature=.2, deterministic=True):
    label = common.evaluation_policy_label(temperature, deterministic=deterministic)
    if prior_enabled:
        label += '_no_prior'
    selected = [record['selection_score'] for record in records if record['policy'] == label]
    if not selected:  # Explicit legacy argmax suites remain available.
        legacy = 'deterministic_no_prior' if prior_enabled else 'deterministic'
        selected = [record['selection_score'] for record in records if record['policy'] == legacy]
    return float(np.mean(selected))


def evaluate(runner, method, output, *, episode, seeds, phase, policies=None,
             temperature=.2, deterministic=True):
    sampled_label = common.evaluation_policy_label(temperature, deterministic=False)
    label = common.evaluation_policy_label(temperature, deterministic=deterministic)
    policies = (label,) if policies is None else policies
    allowed = {'deterministic', 'deterministic_no_prior', sampled_label,
               sampled_label + '_no_prior', 'random', 'shortest'}
    if any(policy not in allowed for policy in policies):
        raise ValueError('actor evaluation requires a temperature sampling or explicit argmax policy')
    actor = getattr(method, 'actor', None)
    original_weight = float(getattr(actor, 'prior_weight', 0.))
    prior_temperature = float(getattr(actor, 'prior_temperature_seconds', 30.))
    records = []
    collector = getattr(method, 'collection_policy', method)
    try:
        for policy in policies:
            weight = 0. if policy.endswith('_no_prior') or policy in {'random', 'shortest'} else original_weight
            if actor is not None:
                set_shortest_path_prior(actor, weight, prior_temperature)
            for seed in seeds:
                learned = policy not in {'random', 'shortest'}
                sampled = policy.startswith('sampled_temperature_') and actor is not None
                kwargs = dict(explore=False)
                if sampled:
                    kwargs.update(temperature=temperature, sample_actions=True)
                select = functools.partial(collector.select, **kwargs) if learned else policy
                with common.evaluation_rng(seed, method):
                    _, _, metrics = runner.run_episode(select, seed=seed, label=policy,
                        on_demand=getattr(collector, 'begin_episode', None) if learned else None)
                entropy = getattr(method, 'entropy_profile', {})
                record = dict(phase=phase, episode=episode, **metrics,
                              entropy_coefficient=entropy.get('coefficient', .002)
                                  if entropy.get('applicable', False) else 0.,
                              entropy_mode=entropy.get('mode', 'not_applicable')
                                  if entropy.get('applicable', False) else 'not_applicable',
                              sp_prior_weight=weight, sp_prior_temperature=prior_temperature,
                              evaluation_policy=('temperature_sampling' if sampled else
                                  'argmax' if learned and actor is not None else 'reference'),
                              evaluation_temperature=temperature if sampled else None,
                              evaluation_action_seed=int(seed), evaluation_demand_seed=int(seed),
                              evaluation_rng_isolated=True)
                records.append(record)
                log_record(output / 'metrics.jsonl', record)
    finally:
        if actor is not None:
            set_shortest_path_prior(actor, original_weight, prior_temperature)
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
    parser.add_argument('--entropy-schedule-episodes', type=int, default=None,
                        help='Preserve the original annealing horizon when extending training')
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--eval-seeds', type=int, default=3)
    parser.add_argument('--evaluation-temperature', type=float, default=.2)
    evaluation_mode = parser.add_mutually_exclusive_group()
    evaluation_mode.add_argument('--evaluation-argmax', dest='evaluation_argmax', action='store_true', default=True,
                        help='Deterministic argmax evaluation (default)')
    evaluation_mode.add_argument('--evaluation-sampled', dest='evaluation_argmax', action='store_false',
                        help='Explicitly sample evaluation actions with --evaluation-temperature')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--collection-device', default=None,
                        help='Inference device; defaults to CPU for feed-forward actors, learner device for graph/recurrent actors')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--ppo-minibatch-size', type=int, default=512)
    parser.add_argument('--score-gauge', default='legacy',
                        choices=['legacy', 'pre_head_legal_reference_v1'],
                        help='MAPPO/IPPO score stabilization; MBPO/Dreamer always use the stable gauge')
    parser.add_argument('--ppo-target-kl', type=float, default=.01,
                        help='Maximum accepted mean PPO KL; unsafe epochs roll back (0 disables this limit)')
    parser.add_argument('--ppo-max-policy-kl', type=float, default=None,
                        help='Maximum accepted per-decision KL (default: 10 times target KL; 0 disables)')
    parser.add_argument('--ppo-max-backtracks', type=int, default=3,
                        help='Maximum half-learning-rate retries after a rejected PPO epoch')
    parser.add_argument('--ppo-advantage-std-floor', type=float, default=1e-6)
    parser.add_argument('--ppo-critic-epochs', type=int, default=None,
                        help='Defaults to 8 for MAPPO and 4 for IPPO')
    parser.add_argument('--ppo-critic-min-epochs', type=int, default=None)
    parser.add_argument('--ppo-critic-patience', type=int, default=2)
    parser.add_argument('--iql-sample-to-insert-ratio', type=float, default=1.,
                        help='Replay optimization draws budgeted per newly collected vehicle transition')
    parser.add_argument('--iql-batch-size', type=int, default=512)
    parser.add_argument('--iql-replay-capacity', type=int, default=200_000)
    parser.add_argument('--iql-target-tau', type=float, default=.01,
                        help='Target synchronization amount per iql-batch-size sampled transitions')
    parser.add_argument('--iql-profile', choices=('legacy_v1', 'double_dueling_v2'),
                        default='double_dueling_v2',
                        help='Fresh IQL uses Double DQN, masked dueling values and epsilon-greedy exploration; old checkpoints require legacy_v1')
    parser.add_argument('--iql-epsilon-initial', type=float, default=1.)
    parser.add_argument('--iql-epsilon-final', type=float, default=.05)
    parser.add_argument('--iql-epsilon-decay-episodes', type=int, default=1000)
    parser.add_argument('--mamba-model-warmup-updates', type=int, default=100)
    parser.add_argument('--mamba-min-model-frames', type=int, default=100)
    parser.add_argument('--mamba-burn-in-windows', type=int, default=5)
    parser.add_argument('--mamba-target-kl', type=float, default=.01,
                        help='MAMBA PPO KL stopping threshold (0 disables)')
    parser.add_argument('--mbpo-model-train-rounds', type=int, default=2)
    parser.add_argument('--mbpo-rollout-seeds', type=int, default=4)
    parser.add_argument('--mbpo-rollout-length', type=int, default=1)
    parser.add_argument('--dreamer-world-model-updates', type=int, default=4)
    parser.add_argument('--dreamer-imagination-passes', type=int, default=1)
    parser.add_argument('--dreamer-imagination-rollouts', type=int, default=2)
    parser.add_argument('--dreamer-imagined-horizon', type=int, default=3)
    parser.add_argument('--dreamer-imagination-device', default='auto',
                        help='Dreamer imagined actor device: auto (CPU), learner, cpu, or an explicit torch device')
    parser.add_argument('--dreamer-allow-legacy-imagination-migration', action='store_true',
                        help='Explicitly allow an old unmarked Dreamer checkpoint to move imagined actor sampling off the learner device')
    parser.add_argument('--vehicle-limit', type=int, default=0)
    parser.add_argument('--session', choices=('microstep', 'event'), default=None,
                        help='Defaults to simulation.session_protocol from the config')
    parser.add_argument('--step-seconds', type=float, default=None)
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sp-prior-weight', type=float, default=0.,
                        help='Initial shortest-path prior weight (default: 0; pass 1 to enable)')
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
    parser.add_argument('--forbid-revisit', action='store_true')
    parser.add_argument('--reward-objective', choices=('observed_mean', 'legacy_exposure'), default='observed_mean',
                        help='Causal cumulative observed-mean reward; legacy_exposure only for explicit old checkpoint use')
    parser.add_argument('--global-reward-only', action=argparse.BooleanOptionalAction, default=True,
                        help='Use only the shared global reward; --no-global-reward-only explicitly selects legacy timeout shaping')
    args = parser.parse_args(argv)
    for name in ('mbpo_model_train_rounds', 'mbpo_rollout_seeds', 'mbpo_rollout_length',
                 'dreamer_world_model_updates', 'dreamer_imagination_passes',
                 'dreamer_imagination_rollouts', 'dreamer_imagined_horizon'):
        if getattr(args, name) < 1:
            parser.error(name.replace('_', '-') + ' must be positive')
    if not np.isfinite(args.mamba_target_kl) or args.mamba_target_kl < 0:
        parser.error('mamba-target-kl must be finite and nonnegative')
    for name in ('ppo_target_kl', 'ppo_max_policy_kl'):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value < 0):
            parser.error(name.replace('_', '-') + ' must be finite and nonnegative')
    if args.ppo_max_backtracks < 0:
        parser.error('ppo-max-backtracks must be nonnegative')
    if args.method in {'mbpo', 'dreamer'}:
        args.score_gauge = 'pre_head_legal_reference_v1'
    if args.score_gauge != 'legacy' and args.method not in {'mappo', 'ippo', 'mbpo', 'dreamer'}:
        parser.error('nonlegacy score-gauge requires mappo or ippo')
    try:
        actor_evaluation_label = common.evaluation_policy_label(args.evaluation_temperature, deterministic=args.evaluation_argmax)
    except ValueError as error:
        parser.error(str(error))
    if args.reward_objective == 'observed_mean' and not args.global_reward_only:
        parser.error('observed_mean requires --global-reward-only; use explicit legacy_exposure for old timeout shaping')
    try:
        simulation_settings = common.resolve_collection_settings(load_config(args.config), args)
    except ValueError as error:
        parser.error(str(error))
    if args.forbid_revisit and (args.session != 'event' or args.sp_prior_weight != 0.):
        parser.error('forbid-revisit requires event session and zero shortest-path prior')
    if args.episodes < 0 or min(args.eval_every, args.eval_seeds, args.threads, args.checkpoint_every) < 1:
        parser.error('invalid nonpositive experiment budget')
    try:
        entropy_horizon = entropy_schedule_episodes(args)
    except ValueError as error:
        parser.error(str(error))
    shortest_path_prior_weight(0, initial_weight=args.sp_prior_weight,
        hold_episodes=args.sp_prior_hold_episodes, off_episode=args.sp_prior_off_episode)
    if not np.isfinite(args.sp_prior_temperature) or args.sp_prior_temperature <= 0:
        parser.error('sp-prior-temperature must be finite and positive')
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite a nonempty experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    runner = common.EpisodeRunner(args, session_factory=resolve_session_factory(args.session, args.step_seconds, args.backend,
        **({'forbid_revisit': True} if args.forbid_revisit else {})))
    method = build_method(args.method, args, runner)
    if not hasattr(method, 'actor'):
        actor_evaluation_label = 'deterministic'  # DTA assignment is not a learned stochastic actor.
    collection_device = args.collection_device
    if collection_device is None and isinstance(getattr(method, 'actor', None), NextHopActor):
        collection_device = 'cpu'
    collector = CollectionPolicy(method, collection_device)
    method.collection_policy = collector
    evaluation_seeds = [args.seed + 10000 + i for i in range(args.eval_seeds)]
    evaluation_policies = (actor_evaluation_label,) if args.skip_reference_baselines else ('random', 'shortest', actor_evaluation_label)
    prior_enabled = args.sp_prior_weight > 0
    learned_policies = (actor_evaluation_label, actor_evaluation_label + '_no_prior') if prior_enabled else (actor_evaluation_label,)
    if prior_enabled:
        evaluation_policies += (actor_evaluation_label + '_no_prior',)
    apply_episode_prior(method, args, 0)
    entropy_metrics = apply_episode_entropy(method, runner.config.method,
        completed_episodes=0, total_episodes=entropy_horizon)
    start, best = 1, float('inf')
    if args.resume:
        state = common.load_checkpoint(output / 'latest.pt', method_name=args.method,
            prepared=runner.prepared, config=runner.config, device=runner.device)
        validate_prior_resume(state['args'], args)
        if (runner.config.method.actor_entropy_mode == 'annealed'
                and entropy_schedule_episodes(state['args']) != entropy_horizon):
            raise ValueError('resume changed entropy schedule episode budget')
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
        saved_label = common.evaluation_policy_label(state['args'].get('evaluation_temperature', .2),
            deterministic=state['args'].get('evaluation_argmax', 'evaluation_temperature' not in state['args']))
        if saved_label != actor_evaluation_label:
            historical_best = output / f'best_{saved_label}.pt'
            if (output / 'best.pt').exists() and not historical_best.exists():
                import shutil
                shutil.copy2(output / 'best.pt', historical_best)
            best = float('inf')
        apply_episode_prior(method, args, state['episode'])
        entropy_metrics = apply_episode_entropy(method, runner.config.method,
            completed_episodes=state['episode'], total_episodes=entropy_horizon)
        del state
    else:
        protocol = dict(world_model=args.method in {'dreamer', 'mbpo', 'mamba'},
            macro_latent_critic=args.method in {'dreamer', 'mamba'}, method=args.method,
            evaluation_policies=list(evaluation_policies),
            session=args.session, step_seconds=args.step_seconds, backend=args.backend,
            forbid_revisit=args.forbid_revisit,
            episode_archive=dict(enabled=args.save_episodes, directory='episodes/train',
                phases=['train'], source='real_sumo', format='npz',
                includes='all macro windows, actual demand, vehicle snapshots, raw decisions, completion status'),
            checkpoint_every=args.checkpoint_every,
            shortest_path_prior=dict(**prior_profile(args), schedule='hold_then_cosine_to_zero',
                scope='fixed per real episode including all imagination and policy updates',
                cost=('candidate free-flow travel + remaining free-flow cost; zero internal junction distance'
                    if simulation_settings['traffic_model'] == 'meso' else
                    'junction + candidate free-flow travel + remaining free-flow cost'),
                best_selection=actor_evaluation_label + '_no_prior' if prior_enabled else actor_evaluation_label,
                mbpo_synthetic_cache='fresh current-policy synthetic rows; one actor update per episode'),
            skip_initial_evaluation=args.skip_initial_evaluation,
            reward=(('shared -delta(C/max(B,1))/60' if args.reward_objective == 'observed_mean' else 'shared -active_vehicle_seconds/(N*60)') if args.global_reward_only else
                    'shared -active_vehicle_seconds/(N*60); individual -5000 only on each unfinished vehicle trajectory'),
            gamma=1., lambda_return=.95, entropy_coefficient=entropy_metrics['entropy_coefficient'],
            entropy_regularization=dict(mode=entropy_metrics['entropy_mode'],
                **{key: getattr(runner.config.method, key) for key in SCHEDULE_FIELDS},
                scope='fixed per real episode, including all real/imagined actor updates',
                population_scaling=False, advantage_scaling=False,
                progress='completed episodes divided by full episode budget, including restored progress'),
            demand=dict(law='independent_poisson_per_od', duration_seconds=runner.demand.duration_seconds,
                expected_vehicle_count=runner.demand.expected_vehicle_count,
                od_rates=[dict(origin_index=o, destination_index=d, rate_vehicles_per_second=rate)
                          for o, d, rate in runner.demand.od_rates]),
            reward_mode='global_only' if args.global_reward_only else 'global_with_individual_timeout',
            training_policy='stochastic', evaluation_policy='argmax' if args.evaluation_argmax else 'temperature_sampling',
            evaluation_temperature=None if args.evaluation_argmax else args.evaluation_temperature, evaluation_rng_isolated=True,
            advantage_normalization=('global_scale_only_std_floor_1_clip_2_5' if args.global_reward_only else
                                     'global_scale_only_std_floor_1_then_individual_timeout_clip_2_5'),
            reward_objective=args.reward_objective,
            global_reward=('-delta(cumulative_vehicle_seconds/max(started_vehicle_count,1))/60'
                           if args.reward_objective == 'observed_mean' else '-active_vehicle_seconds/(N*60)'),
            individual_timeout_penalty=('disabled' if args.global_reward_only else
                                        '-5000 only on unfinished vehicle trajectories at deadline'),
            global_critic_observation='flattened physical dynamic flow + time/active over started/log cumulative minutes/log starts',
            deadline_seconds=runner.deadline, config=runner.config.resolved_snapshot(),
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            input_hashes=runner.prepared.input_hashes, evaluation_seeds=evaluation_seeds,
            note=(f'Reviewer baseline {args.method!r}: fixed {"global_only" if args.global_reward_only else "global_with_individual_timeout"} setting; '
                  'DTA uses deterministic assignment, learned actors sample during training and use the configured evaluation policy.'),
            entropy_note='branch_entropy is entropy of the distribution used to select actions, including evaluation temperature.',
            timing_note='Host wall timings do not force CUDA synchronization. collection_seconds includes archive saving/session close; archive_seconds is the nested archive.save duration. Timings are reported in metrics.jsonl (archive summary precedes completion of save). learning_seconds includes replay preparation and all update phases. Dreamer imagination_seconds includes rollout scoring; critic_update_seconds covers critic fitting. episode_total_seconds includes collection/archive/learning, excludes evaluation/checkpoints.',
            reward_note=('global_reward_sum and selection_score follow the selected shared reward objective' if args.global_reward_only
                         else 'global_reward_sum is -exposure/(N*60); selection_score includes individual timeout costs for ranking only.'))
        protocol.update(simulation_settings)
        learning_protocol = getattr(method, 'learning_protocol', None)
        if learning_protocol is not None:
            protocol['learning_protocol'] = learning_protocol()
        if args.method in {'ippo', 'mappo'}:
            protocol['advantage_normalization'] = 'centered_std_with_near_constant_zero'
        elif args.method in {'iql', 'mamba'}:
            protocol.pop('advantage_normalization', None)
        if args.method == 'mamba':
            protocol['world_model'] = True
            protocol['global_critic_observation'] = 'global RSSM latent conditioned on per-road flow and fixed starting cohort'
        elif args.method == 'dreamer':
            protocol['global_critic_observation'] = 'macro RSSM latent'
        elif args.method == 'ippo':
            protocol['global_critic_observation'] = 'local candidate-set event critic'
        elif args.method == 'iql':
            protocol['global_critic_observation'] = 'none; candidate-set Q with same-vehicle team-tail targets'
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        if not args.skip_initial_evaluation:
            baseline = evaluate(runner, method, output, episode=0, seeds=evaluation_seeds, phase='baseline',
                     policies=evaluation_policies, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
            best = -selection_score(baseline, prior_enabled=prior_enabled, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
        save(runner, method, args, output / 'best.pt', 0, best)
        save(runner, method, args, output / 'initial.pt', 0, best)
        save(runner, method, args, output / 'latest.pt', 0, best)
    (output / 'evaluation_runtime.json').write_text(json.dumps(dict(
        evaluation_policy=actor_evaluation_label, evaluation_temperature=None if args.evaluation_argmax else args.evaluation_temperature,
        evaluation_rng_isolated=True, evaluation_seeds=evaluation_seeds, restored_episode=start - 1,
        checkpoint_selection_policy=actor_evaluation_label + '_no_prior' if prior_enabled else actor_evaluation_label), indent=2))
    (output / 'collection_runtime.json').write_text(json.dumps(dict(
        learner_device=str(runner.device), collection_device=str(collector.device),
        restored_episode=start - 1, synchronization='before_every_training_and_evaluation_episode',
        note='Cross-device sampling preserves the policy distribution, not identical random draws.'), indent=2))
    (output / 'entropy_runtime.json').write_text(json.dumps(dict(
        **entropy_metrics,
        **{key: getattr(runner.config.method, key) for key in SCHEDULE_FIELDS},
        population_scaling=False, advantage_scaling=False,
        restored_episode=start - 1,
        note='Coefficient is derived from config and full run progress; weights, replay, optimizers and RNG are restored.'), indent=2))
    completed_episodes = start - 1
    for episode in range(start, args.episodes + 1):
        episode_started = time.monotonic()
        prior_weight = apply_episode_prior(method, args, episode)
        entropy_metrics = apply_episode_entropy(method, runner.config.method,
            completed_episodes=episode - 1, total_episodes=entropy_horizon)
        rows, windows, metrics = runner.run_episode(functools.partial(collector.select, explore=True),
            seed=args.seed + episode, collect=getattr(method, 'needs_rows', True), label='stochastic',
            record_windows=getattr(method, 'needs_window_records', False),
            row_flow=getattr(method, 'needs_row_flow', True), on_demand=collector.begin_episode,
            archive_path=output / 'episodes' / 'train' / f'episode_{episode:06d}.npz' if args.save_episodes else None,
            archive_metadata=dict(method=args.method, phase='train', episode=episode,
                **entropy_metrics,
                sp_prior_weight=prior_weight, shortest_path_prior=prior_profile(args),
                collection_device=str(collector.device)))
        for row in rows:
            row['sp_prior_weight'] = prior_weight
        learning_started = time.monotonic()
        update = method.on_episode(rows, windows, episode=episode)
        update.setdefault("learning_seconds", time.monotonic() - learning_started)
        metrics["episode_total_seconds"] = time.monotonic() - episode_started
        completed_episodes = episode
        log_record(output / 'metrics.jsonl', dict(phase='train', episode=episode, **metrics, **update,
            **entropy_metrics,
            sp_prior_weight=prior_weight, sp_prior_temperature=args.sp_prior_temperature))
        if getattr(method, 'converged', False):
            records = evaluate(runner, method, output, episode=episode, seeds=evaluation_seeds, phase='evaluation',
                               policies=learned_policies, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
            score = selection_score(records, prior_enabled=prior_enabled, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
            if -score < best:
                best = -score
                save(runner, method, args, output / 'best.pt', episode, best)
            save(runner, method, args, output / 'latest.pt', episode, best)
            break
        if episode % args.eval_every == 0 or episode == args.episodes:
            records = evaluate(runner, method, output, episode=episode, seeds=evaluation_seeds, phase='evaluation',
                               policies=learned_policies, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
            score = selection_score(records, prior_enabled=prior_enabled, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
            if -score < best:
                best = -score
                save(runner, method, args, output / 'best.pt', episode, best)
        if episode % args.checkpoint_every == 0 or episode == args.episodes:
            save(runner, method, args, output / 'latest.pt', episode, best)
    if args.episodes:
        holdout = [args.seed + 20000 + i for i in range(args.eval_seeds)]
        evaluate(runner, method, output, episode=completed_episodes, seeds=holdout, phase='holdout_final',
                 policies=evaluation_policies, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
        state = common.load_checkpoint(output / 'best.pt', method_name=args.method,
            prepared=runner.prepared, config=runner.config, device=runner.device)
        method.load_state_dict(state['method_state'])
        apply_episode_prior(method, args, state['episode'])
        apply_episode_entropy(method, runner.config.method,
            completed_episodes=state['episode'], total_episodes=entropy_horizon)
        evaluate(runner, method, output, episode=state['episode'], seeds=holdout, phase='holdout_best',
                 policies=learned_policies, temperature=args.evaluation_temperature, deterministic=args.evaluation_argmax)
    print(json.dumps({'status': 'completed', 'output': str(output), 'episodes': completed_episodes,
                      'episode_budget': args.episodes}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
