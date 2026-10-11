#!/usr/bin/env python3
"""On-policy event Actor–Critic diagnostic using real SUMO, with no world model."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.learning.sumo_sampling import resolve_session_factory
from urban_dreamer.learning.baselines.common import (EpisodeRunner, resolve_collection_settings,
    accounting_context, reward_objective, reward_context_dim, evaluation_rng, evaluation_policy_label)
from urban_dreamer.learning.baselines.collection_policy import ActorMirror
from urban_dreamer.learning.entropy_schedule import entropy_schedule_profile
from urban_dreamer.models.policy import (NextHopActor,
    individual_actor_advantage, lambda_return, set_shortest_path_prior,
    shortest_path_prior_weight)
from urban_dreamer.traffic.environment import SumoSession
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.workflow.config import load_config

CHECKPOINT_FORMAT = 'toy_global_individual_timeout_poisson_ac_v1'
ADVANTAGE_NORMALIZATION = 'global_scale_only_std_floor_1_then_individual_timeout_clip_2_5'

PRIOR_DEFAULTS = dict(sp_prior_weight=0., sp_prior_temperature=30.,
                      sp_prior_hold_episodes=200, sp_prior_off_episode=1000)


def prior_profile(args):
    return {key: getattr(args, key, default) for key, default in PRIOR_DEFAULTS.items()}


def set_episode_prior(runner, episode):
    profile = prior_profile(runner.args)
    weight = shortest_path_prior_weight(episode, initial_weight=profile['sp_prior_weight'],
        hold_episodes=profile['sp_prior_hold_episodes'], off_episode=profile['sp_prior_off_episode'])
    set_shortest_path_prior(runner.actor, weight, profile['sp_prior_temperature'])
    return weight


def actor_prior_metrics(actor):
    return dict(sp_prior_weight=float(getattr(actor, 'prior_weight', 0.)),
                sp_prior_temperature=float(getattr(actor, 'prior_temperature_seconds', 30.)))


def set_episode_entropy(runner, episode):
    profile = entropy_schedule_profile(runner.config.method,
        completed_episodes=max(0, int(episode) - 1),
        total_episodes=max(1, int(getattr(runner.args, 'episodes', 5000))))
    # Historical real AC used a fixed coefficient, even with population-mode
    # mainline configs. Preserve that behavior for old experiment snapshots.
    coefficient = profile['coefficient'] if profile['mode'] in {'annealed', 'off'} else .002
    runner.entropy_coefficient = coefficient
    runner.entropy_profile = {**profile, 'coefficient': coefficient}
    return runner.entropy_profile


class GlobalTrafficCritic(torch.nn.Module):
    """Value of the real full-network traffic state, without a dynamics model."""
    def __init__(self, edge_count, context_dim=4):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(edge_count * 4 + context_dim, 256), torch.nn.Tanh(),
            torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 1))
        for layer in self.network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.))
                torch.nn.init.zeros_(layer.bias)
        torch.nn.init.orthogonal_(self.network[-1].weight, gain=1.)

    def forward(self, obs):
        return self.network(obs)


class RealActorCritic:
    def __init__(self, args):
        self.args = args
        args.reward_objective = reward_objective(args)
        args.global_reward_only = bool(getattr(args, 'global_reward_only', True))
        if args.reward_objective == 'observed_mean' and not args.global_reward_only:
            raise ValueError('observed_mean requires global_reward_only=True')
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device(args.device)
        self.config = load_config(args.config)
        if reward_objective(args) == 'legacy_exposure':
            from urban_dreamer.workflow.config import legacy_semantic_fingerprint
            self.config = replace(self.config, semantic_fingerprint=legacy_semantic_fingerprint(self.config))
        self.config = replace(self.config, run=replace(self.config.run, device=args.device))
        self.simulation_settings = resolve_collection_settings(self.config, args)
        if getattr(args, 'forbid_revisit', False):
            if args.session != 'event' or getattr(args, 'sp_prior_weight', 0.) != 0.:
                raise ValueError('forbid_revisit requires event session and zero shortest-path prior')
        self.prepared = validate_prepared_inputs(self.config)
        self.topology = load_traffic_topology(self.prepared)
        if args.vehicle_limit:
            self.topology = replace(self.topology, vehicles=self.topology.vehicles[:args.vehicle_limit])
        self.demand = PoissonDemand.from_vehicles(self.topology.vehicles,
            duration_seconds=self.config.simulation.demand_duration_seconds)
        self.network = self.topology.network
        self.n = len(self.topology.vehicles)
        self.candidates = NextHopCandidates(self.network,
            traffic_model=self.config.simulation.traffic_model, **(
            {'forbid_revisit': True} if getattr(args, 'forbid_revisit', False) else {}))
        self.session_factory = resolve_session_factory(getattr(args, 'session', 'microstep'),
                                                       float(getattr(args, 'step_seconds', 1.)),
                                                       getattr(args, 'backend', 'traci'),
                                                       **({'forbid_revisit': True} if getattr(args, 'forbid_revisit', False) else {}))
        simulation = self.config.simulation
        self.deadline = 30 * math.ceil((simulation.world_model_intervals_per_episode * 30
                                       + simulation.training_completion_buffer_seconds) / 30)
        self.advantage_normalization = ADVANTAGE_NORMALIZATION
        if getattr(args, 'global_reward_only', True):
            self.advantage_normalization = 'global_scale_only_std_floor_1_clip_2_5'
        self.actor = NextHopActor().to(self.device)
        set_episode_prior(self, 0)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        set_episode_entropy(self, 1)
        self.collected_windows = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 73861)
            self.global_critic = GlobalTrafficCritic(self.network.edge_count, reward_context_dim(args)).to(self.device)
        self.global_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=args.global_critic_lr)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(self.topology.vehicles)}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def global_observation(self, state):
        return np.concatenate((np.asarray(state['flow'][:, :4], dtype=np.float32).reshape(-1),
            (np.asarray([state['time'] / self.deadline, state['active_vehicle_count'] / max(self.n, 1)], dtype=np.float32)
             if reward_objective(self.args) == 'legacy_exposure' else accounting_context(state, self.deadline))))

    def global_targets(self, windows):
        if not windows:
            raise ValueError('global critic needs collected macro windows')
        values = self.global_critic(self.tensor(np.stack([w['obs'] for w in windows]))).squeeze(-1)
        with torch.no_grad():
            next_values = self.global_critic(self.tensor(np.stack([w['next_obs'] for w in windows]))).squeeze(-1)
            targets = lambda_return(self.tensor([w['reward'] for w in windows]), next_values,
                terminal=torch.tensor([w['terminal'] for w in windows], device=self.device, dtype=torch.bool))
        return targets - values.detach(), targets, values

    def run_episode(self, *, policy, seed, collect=False, archive_path=None, archive_metadata=None,
                    temperature=1.):
        if collect and policy != 'stochastic':
            raise ValueError('training collection requires stochastic categorical sampling')
        if not hasattr(self, '_collection_mirror'):
            self._collection_mirror = ActorMirror(self.device, getattr(self.args, 'collection_device', None))
        self.collection_actor = self._collection_mirror.sync(self.actor)
        prior_metrics = actor_prior_metrics(self.actor)
        if policy not in {'stochastic', 'deterministic'}:
            prior_metrics['sp_prior_weight'] = 0.

        def select(vehicles, actor_obs, frozen, now):
            actor = getattr(self, 'collection_actor', self.actor)
            device = self._collection_mirror.device
            with torch.no_grad():
                selection = actor.sample_action(torch.as_tensor(actor_obs.features,
                    dtype=torch.float32, device=device),
                    torch.as_tensor(actor_obs.mask, device=device), deterministic=policy == 'deterministic',
                    temperature=temperature)
            indices, entropy = selection.indices_and_entropy()
            return indices, [dict(entropy=value) for value in entropy]

        def archive_fields(shared):
            # Retain legacy archive provenance exactly so checkpoint replay can
            # replace an already archived episode under the same metadata.
            provenance = dict(episode=int(seed), phase='train' if collect else 'collection')
            provenance.update(archive_metadata or {})
            if getattr(self.args, 'collection_device', None) is not None:
                provenance['collection_device'] = str(self._collection_mirror.device)
            provenance.update(prior_metrics)
            if not self.n:
                provenance.update(padded_empty_episode=True, physical_simulation_end_seconds=0.)
            return dict(provenance,
                city=self.config.inputs.network_name, method='online_ac', policy=policy,
                seed=int(seed), demand_seed=self.args.seed, demand_episode_index=int(seed),
                vehicle_count=self.n, total_vehicle_count=self.n,
                demand_realization=shared['realized_demand'],
                demand_realization_sha256=shared['demand_sha256'],
                demand_law='independent_od_poisson', demand_duration_seconds=self.demand.duration_seconds,
                od_rates=[dict(origin_index=o, destination_index=d, rate_veh_per_second=rate)
                          for o, d, rate in self.demand.od_rates],
                input_hashes=self.prepared.input_hashes, edge_ids=list(self.network.edge_ids),
                config_fingerprint=self.config.semantic_fingerprint,
                forbid_revisit=bool(getattr(self.args, 'forbid_revisit', False)),
                reward_objective=reward_objective(self.args),
                reward_contract=shared.get('reward_contract'),
                reward_mode=('global_only' if getattr(self.args, 'global_reward_only', True)
                             else 'global_with_individual_timeout'),
                requested_step_seconds=float(getattr(self.args, 'step_seconds', 1.)),
                macro_step_seconds=30., **self.simulation_settings)

        def metrics_fields(metrics):
            metrics.update(prior_metrics,
                demand_law='independent_od_poisson',
                demand_rate_veh_per_second=sum(rate for _, _, rate in self.demand.od_rates),
                demand_expected_vehicle_count=self.demand.expected_vehicle_count,
                demand_realization_sha256=metrics.pop('demand_sha256'))
            return metrics

        rows, windows, metrics = EpisodeRunner.run_episode(self,
            select if policy in {'stochastic', 'deterministic'} else policy,
            seed=seed, collect=collect, row_flow=False, label=policy,
            archive_path=archive_path, archive_metadata=archive_metadata,
            session_factory=self.session_factory or SumoSession,
            observation_builder=build_actor_observations, empty_terminal=True,
            metadata_transform=archive_fields, metrics_transform=metrics_fields)
        # Shared timeout attribution uses physical IDs. AC's learner/checkpoint
        # interface continues to expose integer population indices and its old
        # compact row/window schema.
        for row in rows:
            row['vehicle_id'] = self.ids[row['vehicle_id']]
            for key in ('current_edge', 'edge_indices', 'entropy'):
                row.pop(key, None)
        for window in windows:
            window.pop('start_time', None)
            window.pop('end_time', None)
        if collect:
            self.collected_windows = windows
        return rows, metrics

    def update(self, rows, windows=None):
        learning_started = time.monotonic()
        windows = self.collected_windows if windows is None else windows
        critic_started = time.monotonic()
        global_advantage, global_targets, global_values = self.global_targets(windows)
        critic_seconds = time.monotonic() - critic_started
        actor_started = time.monotonic()
        actor_loss_value, actor_gradient_value = 0., 0.
        if rows:
            event_windows = torch.tensor([r['window'] for r in rows], device=self.device)
            advantage = individual_actor_advantage(global_advantage[event_windows],
                (torch.zeros(len(rows), device=self.device) if getattr(self.args, 'global_reward_only', True)
                 else self.tensor([r.get('timeout_return', 0.) for r in rows])))
            features = self.tensor(np.stack([r['features'] for r in rows]))
            mask = torch.as_tensor(np.stack([r['mask'] for r in rows]), device=self.device)
            actions = torch.tensor([r['action'] for r in rows], device=self.device)
            distribution = Categorical(logits=self.actor.forward_logits(features, mask))
            entropy = distribution.entropy()
            actor_loss = -(distribution.log_prob(actions) * advantage + self.entropy_coefficient * entropy).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            if not torch.isfinite(actor_gradient):
                raise RuntimeError('non-finite actor gradient')
            self.actor_optimizer.step()
            actor_loss_value, actor_gradient_value = float(actor_loss.detach()), float(actor_gradient)
        actor_seconds = time.monotonic() - actor_started
        critic_started = time.monotonic()
        global_loss = F.mse_loss(global_values, global_targets)
        self.global_optimizer.zero_grad(set_to_none=True)
        global_loss.backward()
        global_gradient = torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 10.)
        if not torch.isfinite(global_gradient):
            raise RuntimeError('non-finite global critic gradient')
        self.global_optimizer.step()
        with torch.no_grad():
            fitted = self.global_critic(self.tensor(np.stack([w['obs'] for w in windows]))).squeeze(-1)
            global_ev = 1 - torch.var(global_targets - fitted, unbiased=False) / torch.var(global_targets, unbiased=False).clamp_min(1e-8)
        critic_seconds += time.monotonic() - critic_started
        distribution_metrics = {}
        if rows:
            legal_count = mask.sum(-1)
            multi = legal_count >= 2
            if bool(multi.any()):
                distribution_metrics = dict(policy_entropy_mean=float(entropy.detach()[multi].mean()),
                    normalized_policy_entropy_mean=float((entropy.detach()[multi] / legal_count[multi].float().log()).mean()),
                    max_action_probability_mean=float(distribution.probs.detach().amax(-1)[multi].mean()))
        return dict(actor_loss=actor_loss_value, actor_gradient_norm=actor_gradient_value,
            actor_entropy_mode=self.entropy_profile['mode'], entropy_coefficient=self.entropy_coefficient,
            entropy_schedule_progress=self.entropy_profile['progress'], **distribution_metrics,
            global_advantage_std=float(global_advantage.std(unbiased=False)),
            global_critic_loss=float(global_loss.detach()), global_critic_gradient_norm=float(global_gradient),
            global_explained_variance=float(global_ev), actor_update_seconds=actor_seconds,
            critic_update_seconds=critic_seconds, learning_seconds=time.monotonic() - learning_started)

    def save(self, path, episode, best):
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        torch.save(dict(format=CHECKPOINT_FORMAT, episode=episode, best=best,
            advantage_normalization=self.advantage_normalization,
            actor=self.actor.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            global_critic=self.global_critic.state_dict(), global_optimizer=self.global_optimizer.state_dict(),
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items()},
            input_hashes=self.prepared.input_hashes, config_fingerprint=self.config.semantic_fingerprint,
            physics=physics_identity(self.config.simulation)), temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get('format') != CHECKPOINT_FORMAT:
            raise ValueError(f'unsupported checkpoint format; expected {CHECKPOINT_FORMAT}')
        if state['input_hashes'] != self.prepared.input_hashes or state['config_fingerprint'] != self.config.semantic_fingerprint:
            raise ValueError('experiment checkpoint mismatch')
        if state.get('physics', {'traffic_model': 'micro'}) != physics_identity(self.config.simulation):
            raise ValueError('experiment checkpoint physics mismatch')
        if self.config.method.actor_entropy_mode == 'annealed' and state['args'].get('episodes') != getattr(self.args, 'episodes', None):
            raise ValueError('resume changed entropy schedule episode budget')
        for key in ('seed', 'vehicle_limit', 'actor_lr', 'global_critic_lr', 'session', 'step_seconds', 'backend'):
            if state['args'][key] != getattr(self.args, key):
                raise ValueError(f'resume changed {key}')
        if state['args'].get('forbid_revisit', False) != getattr(self.args, 'forbid_revisit', False):
            raise ValueError('resume changed forbid_revisit')
        if state['args'].get('reward_objective', 'legacy_exposure') != reward_objective(self.args):
            raise ValueError('resume changed reward_objective; migrate checkpoint explicitly')
        if state['args'].get('global_reward_only', False) != getattr(self.args, 'global_reward_only', True):
            raise ValueError('resume changed global_reward_only')
        saved_prior = {key: state['args'].get(key, default) for key, default in PRIOR_DEFAULTS.items()}
        if saved_prior != prior_profile(self.args):
            raise ValueError('resume changed shortest-path prior protocol')
        if state['advantage_normalization'] != self.advantage_normalization:
            raise ValueError('resume changed advantage normalization')
        if state['args'].get('skip_reference_baselines', False) != getattr(self.args, 'skip_reference_baselines', False):
            raise ValueError('resume changed evaluation reference suite')
        if state['args'].get('save_episodes', False) != getattr(self.args, 'save_episodes', False):
            raise ValueError('resume changed episode archival protocol')
        if state['args'].get('skip_initial_evaluation', False) != getattr(self.args, 'skip_initial_evaluation', False):
            raise ValueError('resume changed initial evaluation protocol')
        self.actor.load_state_dict(state['actor'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.global_critic.load_state_dict(state['global_critic'])
        self.global_optimizer.load_state_dict(state['global_optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
        set_episode_prior(self, state['episode'])
        set_episode_entropy(self, state['episode'] + 1)
        return state['episode'] + 1, state['best']


def log_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    short = {key: value for key, value in record.items() if key not in {'branch_choices', 'choices'}}
    print(json.dumps(short, allow_nan=False), flush=True)


def evaluate(runner, output, *, episode, seeds, phase, policies=None, temperature=None,
             deterministic=None):
    args = getattr(runner, 'args', None)
    temperature = getattr(args, 'evaluation_temperature', .2) if temperature is None else temperature
    deterministic = getattr(args, 'evaluation_argmax', True) if deterministic is None else deterministic
    sampled_label = evaluation_policy_label(temperature, deterministic=False)
    actor_label = evaluation_policy_label(temperature, deterministic=deterministic)
    policies = (actor_label,) if policies is None else policies
    if any(policy not in {'deterministic', sampled_label, 'random', 'shortest'} for policy in policies):
        raise ValueError('actor evaluation requires temperature sampling or explicit argmax')
    records = []
    enabled = getattr(args, 'sp_prior_weight', 0.) > 0
    saved = actor_prior_metrics(runner.actor) if enabled else None
    try:
        if enabled:
            set_episode_prior(runner, episode)
        for policy in policies:
            learned = policy in {'deterministic', sampled_label}
            sampled = policy == sampled_label
            labels = (policy, policy + '_no_prior') if enabled and learned else (policy,)
            for label in labels:
                if enabled:
                    if label.endswith('_no_prior'):
                        set_shortest_path_prior(runner.actor, 0., prior_profile(args)['sp_prior_temperature'])
                    else:
                        set_episode_prior(runner, episode)
                for seed in seeds:
                    kwargs = dict(policy='stochastic' if sampled else policy, seed=seed)
                    if sampled:
                        kwargs['temperature'] = temperature
                    with evaluation_rng(seed):
                        _, metrics = runner.run_episode(**kwargs)
                    record = dict(phase=phase, episode=episode, **metrics)
                    record.update(policy=label, evaluation_policy='temperature_sampling' if sampled else policy,
                        evaluation_temperature=temperature if sampled else None,
                        evaluation_action_seed=int(seed), evaluation_demand_seed=int(seed), evaluation_rng_isolated=True)
                    if learned and hasattr(runner, 'actor'):
                        record.update(actor_prior_metrics(runner.actor))
                    records.append(record)
                    log_record(output / 'metrics.jsonl', record)
        return records
    finally:
        if saved is not None:
            set_shortest_path_prior(runner.actor, saved['sp_prior_weight'], saved['sp_prior_temperature'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/toy_cross_bottleneck.toml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=150)
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
        help='policy inference device; defaults to the learning device')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--vehicle-limit', type=int, default=0)
    parser.add_argument('--global-critic-lr', type=float, default=3e-4)
    parser.add_argument('--sp-prior-weight', type=float, default=0.,
                        help='Initial shortest-path prior weight (default: 0; pass 1 to enable)')
    parser.add_argument('--sp-prior-temperature', type=float, default=30., help='Free-flow cost temperature in seconds')
    parser.add_argument('--sp-prior-hold-episodes', type=int, default=200)
    parser.add_argument('--sp-prior-off-episode', type=int, default=1000)
    parser.add_argument('--session', choices=('microstep', 'event'), default=None,
                        help='Defaults to simulation.session_protocol from the config')
    parser.add_argument('--step-seconds', type=float, default=None)
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-episodes', action='store_true',
                        help='Archive every real training episode, including physical states and action events')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save latest checkpoint every N training episodes, plus evaluation and final episodes')
    parser.add_argument('--skip-reference-baselines', action='store_true',
                        help='Evaluate only the learned actor; omit random and shortest reference runs')
    parser.add_argument('--skip-initial-evaluation', action='store_true',
                        help='Start training immediately; retain periodic and final evaluations')
    parser.add_argument('--forbid-revisit', action='store_true')
    parser.add_argument('--reward-objective', choices=('observed_mean', 'legacy_exposure'), default='observed_mean',
                        help='Causal cumulative observed-mean reward; legacy_exposure only for explicit old checkpoint use')
    parser.add_argument('--global-reward-only', action=argparse.BooleanOptionalAction, default=True,
                        help='Use only the shared global reward; --no-global-reward-only explicitly selects legacy timeout shaping')
    args = parser.parse_args(argv)
    try:
        actor_evaluation_label = evaluation_policy_label(args.evaluation_temperature, deterministic=args.evaluation_argmax)
    except ValueError as error:
        parser.error(str(error))
    if args.reward_objective == 'observed_mean' and not args.global_reward_only:
        parser.error('observed_mean requires --global-reward-only; use explicit legacy_exposure for old timeout shaping')
    try:
        simulation_settings = resolve_collection_settings(load_config(args.config), args)
    except ValueError as error:
        parser.error(str(error))
    if args.forbid_revisit and (args.session != 'event' or args.sp_prior_weight != 0.):
        parser.error('forbid-revisit requires event session and zero shortest-path prior')
    if args.episodes < 0 or min(args.eval_every, args.eval_seeds, args.threads, args.checkpoint_every) < 1:
        parser.error('invalid nonpositive experiment budget')
    if (not math.isfinite(args.sp_prior_weight) or args.sp_prior_weight < 0
            or not math.isfinite(args.sp_prior_temperature) or args.sp_prior_temperature <= 0
            or args.sp_prior_hold_episodes < 0 or args.sp_prior_off_episode <= args.sp_prior_hold_episodes):
        parser.error('invalid shortest-path prior schedule or temperature')
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite a nonempty experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    runner = RealActorCritic(args)
    evaluation_seeds = [args.seed + 10000 + i for i in range(args.eval_seeds)]
    evaluation_policies = (actor_evaluation_label,) if args.skip_reference_baselines else ('random', 'shortest', actor_evaluation_label)
    ranking_policy = actor_evaluation_label + '_no_prior' if args.sp_prior_weight > 0 else actor_evaluation_label
    start, best = 1, float('inf')
    if args.resume:
        start, best = runner.load(output / 'latest.pt')
        state = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        saved_args = state['args']
        saved_label = evaluation_policy_label(saved_args.get('evaluation_temperature', .2),
            deterministic=saved_args.get('evaluation_argmax', 'evaluation_temperature' not in saved_args))
        if saved_label != actor_evaluation_label:
            historical_best = output / f'best_{saved_label}.pt'
            if (output / 'best.pt').exists() and not historical_best.exists():
                import shutil
                shutil.copy2(output / 'best.pt', historical_best)
            best = float('inf')
        del state
    else:
        protocol = dict(world_model=False, macro_latent_critic=False,
            shortest_path_prior=dict(**prior_profile(args), schedule='constant through hold_episodes; cosine decay to zero at off_episode',
                application='actor logits minus weight * candidate free-flow excess seconds / temperature; unchanged reward',
                clock='completed training episode number; fixed during collection and update; evaluation does not advance clock',
                checkpoint_selection_policy=ranking_policy,
                evaluation='paired identical demand seeds with scheduled prior and zero prior when enabled'),
            reward_mode='global_only' if args.global_reward_only else 'global_with_individual_timeout',
            evaluation_policies=list(evaluation_policies) + ([actor_evaluation_label + '_no_prior'] if args.sp_prior_weight > 0 else []),
            skip_initial_evaluation=args.skip_initial_evaluation,
            session=args.session, step_seconds=args.step_seconds, backend=args.backend,
            forbid_revisit=args.forbid_revisit,
            reward=(('shared -delta(C/max(B,1))/60' if args.reward_objective == 'observed_mean' else 'shared -active_vehicle_seconds/(N*60)') if args.global_reward_only else
                    'shared -active_vehicle_seconds/(N*60); individual -5000 for each unfinished vehicle at deadline'),
            gamma=1., lambda_return=.95, entropy_coefficient=runner.entropy_coefficient,
            entropy_schedule=dict(**runner.entropy_profile,
                settings={k:v for k,v in runner.config.resolved_snapshot()['method'].items() if k.startswith('actor_entropy_')},
                clock='completed real training episodes; fixed within each episode; evaluations do not advance'),
            demand=dict(law='independent_od_poisson', seed=args.seed,
                episode_index='SUMO episode seed', duration_seconds=runner.demand.duration_seconds,
                expected_vehicle_count=runner.demand.expected_vehicle_count,
                od_rates=[dict(origin_index=o, destination_index=d, rate_veh_per_second=rate)
                          for o, d, rate in runner.demand.od_rates],
                normalization='cumulative actual trip starts B(t); max(B,1) for empty episodes'),
            advantage_normalization=runner.advantage_normalization,
            reward_objective=args.reward_objective,
            global_reward=('-delta(cumulative_vehicle_seconds/max(started_vehicle_count,1))/60'
                           if args.reward_objective == 'observed_mean' else '-active_vehicle_seconds/(N*60)'),
            timeout_penalty=('disabled' if args.global_reward_only else
                '-5000 once on each unfinished vehicle last action at the real deadline; '
                'gamma=1 terminal return applies only to that vehicle prior actions; excluded from global critic'),
            global_critic_observation='flattened physical dynamic flow + time/active over started/log cumulative minutes/log starts',
            deadline_seconds=runner.deadline, config=runner.config.resolved_snapshot(),
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            training_policy='stochastic', evaluation_policy='argmax' if args.evaluation_argmax else 'temperature_sampling',
            evaluation_temperature=None if args.evaluation_argmax else args.evaluation_temperature,
            evaluation_rng_isolated=True, checkpoint_format=CHECKPOINT_FORMAT,
            input_hashes=runner.prepared.input_hashes, evaluation_seeds=evaluation_seeds,
            note=('Global Actor-Critic; no vehicle critic, world model, or imagination.' if args.global_reward_only else
                  'Global Actor-Critic with individual timeout penalties; no vehicle critic, world model, or imagination.'),
            entropy_note='branch_entropy is entropy of the distribution used to select actions, including evaluation temperature.',
            reward_note=('global_reward_sum and selection_score follow the selected shared reward objective' if args.global_reward_only else
                'global_reward_sum excludes timeouts; vehicle_timeout_penalty_sum is diagnostic only; '
                'selection_score combines both only for checkpoint ranking.'))
        protocol.update(simulation_settings)
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        if not args.skip_initial_evaluation:
            baseline = evaluate(runner, output, episode=0, seeds=evaluation_seeds, phase='baseline',
                     policies=evaluation_policies)
            best = -float(np.mean([r['selection_score'] for r in baseline if r['policy'] == ranking_policy]))
        runner.save(output / 'best.pt', 0, best)
        runner.save(output / 'initial.pt', 0, best)
        runner.save(output / 'latest.pt', 0, best)
    (output / 'evaluation_runtime.json').write_text(json.dumps(dict(
        evaluation_policy=actor_evaluation_label, evaluation_temperature=None if args.evaluation_argmax else args.evaluation_temperature,
        evaluation_rng_isolated=True, evaluation_seeds=evaluation_seeds, restored_episode=start - 1,
        checkpoint_selection_policy=ranking_policy), indent=2))
    for episode in range(start, args.episodes + 1):
        episode_started = time.monotonic()
        set_episode_prior(runner, episode)
        set_episode_entropy(runner, episode)
        archive_kwargs = {}
        if args.save_episodes:
            archive_kwargs = dict(archive_path=output / 'episodes' / 'train' / f'episode_{episode:06d}.npz',
                                  archive_metadata=dict(phase='train', episode=episode))
        rows, metrics = runner.run_episode(policy='stochastic', seed=args.seed + episode, collect=True,
                                           **archive_kwargs)
        update = runner.update(rows, runner.collected_windows)
        log_record(output / 'metrics.jsonl', dict(phase='train', episode=episode, **metrics, **update,
                   episode_total_seconds=time.monotonic() - episode_started))
        if episode % args.eval_every == 0 or episode == args.episodes:
            records = evaluate(runner, output, episode=episode, seeds=evaluation_seeds, phase='evaluation')
            score = float(np.mean([record['selection_score'] for record in records if record['policy'] == ranking_policy]))
            if -score < best:
                best = -score
                runner.save(output / 'best.pt', episode, best)
        if episode % args.checkpoint_every == 0 or episode % args.eval_every == 0 or episode == args.episodes:
            runner.save(output / 'latest.pt', episode, best)
    if args.episodes:
        holdout = [args.seed + 20000 + i for i in range(args.eval_seeds)]
        evaluate(runner, output, episode=args.episodes, seeds=holdout, phase='holdout_final',
                 policies=evaluation_policies)
        state = torch.load(output / 'best.pt', map_location=runner.device, weights_only=False)
        runner.actor.load_state_dict(state['actor'])
        evaluate(runner, output, episode=state['episode'], seeds=holdout, phase='holdout_best')
    print(json.dumps({'status': 'completed', 'output': str(output), 'episodes': args.episodes}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
