#!/usr/bin/env python3
"""On-policy event Actor–Critic diagnostic using real SUMO, with no world model."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import functools
import hashlib
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
from urban_dreamer.models.policy import (NextHopActor, attach_timeout_penalties,
    individual_actor_advantage, lambda_return, set_shortest_path_prior,
    shortest_path_prior_weight)
from urban_dreamer.traffic.environment import NextHopDecision, SumoSession, WindowResult
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
from urban_dreamer.workflow.config import NONARRIVAL_PENALTY, load_config

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


def resolve_session_factory(session, step_seconds, backend='traci'):
    """Return the episode session override; None keeps the module-level SumoSession lookup."""
    if session == 'event':
        return functools.partial(SumoEventSession, step_seconds=step_seconds, backend=backend)
    if session == 'microstep':
        return None
    raise ValueError(session)


class GlobalTrafficCritic(torch.nn.Module):
    """Value of the real full-network traffic state, without a dynamics model."""
    def __init__(self, edge_count):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(edge_count * 4 + 2, 256), torch.nn.Tanh(),
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
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device(args.device)
        self.config = load_config(args.config)
        self.config = replace(self.config, run=replace(self.config.run, device=args.device))
        self.prepared = validate_prepared_inputs(self.config)
        self.topology = load_traffic_topology(self.prepared)
        if args.vehicle_limit:
            self.topology = replace(self.topology, vehicles=self.topology.vehicles[:args.vehicle_limit])
        self.demand = PoissonDemand.from_vehicles(self.topology.vehicles,
            duration_seconds=self.config.simulation.demand_duration_seconds)
        self.network = self.topology.network
        self.n = len(self.topology.vehicles)
        self.candidates = NextHopCandidates(self.network)
        self.session_factory = resolve_session_factory(getattr(args, 'session', 'microstep'),
                                                       float(getattr(args, 'step_seconds', 1.)),
                                                       getattr(args, 'backend', 'traci'))
        simulation = self.config.simulation
        self.deadline = 30 * math.ceil((simulation.world_model_intervals_per_episode * 30
                                       + simulation.training_completion_buffer_seconds) / 30)
        self.advantage_normalization = ADVANTAGE_NORMALIZATION
        self.actor = NextHopActor().to(self.device)
        set_episode_prior(self, 0)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.collected_windows = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 73861)
            self.global_critic = GlobalTrafficCritic(self.network.edge_count).to(self.device)
        self.global_optimizer = torch.optim.Adam(self.global_critic.parameters(), lr=args.global_critic_lr)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(self.topology.vehicles)}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def global_observation(self, state):
        return np.concatenate((np.asarray(state['flow'][:, :4], dtype=np.float32).reshape(-1),
            np.asarray([state['time'] / self.deadline, state['active_vehicle_count'] / max(self.n, 1)], dtype=np.float32)))

    def global_targets(self, windows):
        if not windows:
            raise ValueError('global critic needs collected macro windows')
        values = self.global_critic(self.tensor(np.stack([w['obs'] for w in windows]))).squeeze(-1)
        with torch.no_grad():
            next_values = self.global_critic(self.tensor(np.stack([w['next_obs'] for w in windows]))).squeeze(-1)
            targets = lambda_return(self.tensor([w['reward'] for w in windows]), next_values,
                terminal=torch.tensor([w['terminal'] for w in windows], device=self.device, dtype=torch.bool))
        return targets - values.detach(), targets, values

    def run_episode(self, *, policy, seed, collect=False, archive_path=None, archive_metadata=None):
        if collect and policy != 'stochastic':
            raise ValueError('training collection requires stochastic categorical sampling')
        started = time.monotonic()
        prior_metrics = actor_prior_metrics(self.actor)
        if policy not in {'stochastic', 'deterministic'}:
            prior_metrics['sp_prior_weight'] = 0.
        config = replace(self.config, run=replace(self.config.run, seed=int(seed)))
        vehicles = self.demand.sample(seed=self.args.seed, episode_index=int(seed))
        episode_topology = replace(self.topology, vehicles=vehicles)
        self.episode_topology = episode_topology
        self.n = len(vehicles)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(vehicles)}
        denominator = max(self.n, 1)
        realization = [dict(vehicle_id=v.vehicle_id, origin_index=v.origin_index,
            destination_index=v.destination_index, departure_second=v.departure_second) for v in vehicles]
        demand_hash = hashlib.sha256(json.dumps(realization, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        archive = None
        if archive_path is not None:
            from urban_dreamer.learning.episode_archive import EpisodeArchive
            provenance = dict(episode=int(seed), phase='train' if collect else 'collection')
            provenance.update(archive_metadata or {})
            provenance.update(prior_metrics)
            if not self.n:
                provenance.update(padded_empty_episode=True, physical_simulation_end_seconds=0.)
            archive = EpisodeArchive(dict(provenance,
                city=self.config.inputs.network_name, method='online_ac', policy=policy,
                seed=int(seed), demand_seed=self.args.seed, demand_episode_index=int(seed),
                vehicle_count=self.n, total_vehicle_count=self.n, demand_realization=realization,
                demand_realization_sha256=demand_hash,
                demand_law='independent_od_poisson', demand_duration_seconds=self.demand.duration_seconds,
                od_rates=[dict(origin_index=o, destination_index=d, rate_veh_per_second=rate)
                          for o, d, rate in self.demand.od_rates],
                input_hashes=self.prepared.input_hashes, edge_ids=list(self.network.edge_ids),
                config_fingerprint=self.config.semantic_fingerprint,
                session_protocol=getattr(self.args, 'session', 'microstep'),
                backend=getattr(self.args, 'backend', 'traci'),
                requested_step_seconds=float(getattr(self.args, 'step_seconds', 1.)),
                macro_step_seconds=30.))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        rows = []
        choices, branch_choices = Counter(), Counter()
        branch_entropies = []
        exposure, event_count = 0., 0
        windows = []
        with (self.session_factory or SumoSession)(config, self.prepared, episode_topology) as session:
            state = session.reset()
            if archive is not None:
                archive.start(state)
            if not self.n:
                observation = self.global_observation(state)
                windows.append(dict(obs=observation, next_obs=observation.copy(), reward=0., terminal=True))
                if archive is not None:
                    # Preserve the same empty 30-second terminal transition as mainline replay.
                    # SUMO itself need not advance an already completed, zero-population episode.
                    next_state = dict(state, time=float(state['time']) + 30., done=True)
                    zero_counts = np.zeros(self.network.edge_count, dtype=np.int64)
                    archive.append(WindowResult((), zero_counts, zero_counts.copy(), 0.,
                        next_state, float(state['time']), float(next_state['time'])))
            while state['time'] < self.deadline and not session.done:
                progress = state['time'] / self.deadline
                active_fraction = state['active_vehicle_count'] / denominator
                window_index = len(windows)
                global_obs = self.global_observation(state)

                def decide(vehicles, frozen, now):
                    nonlocal event_count
                    kwargs = dict(policy_edge_features=np.asarray(frozen[:, [2, 1]], dtype=np.float32),
                        vehicle_states=vehicles, candidates=self.candidates,
                        episode_progress=progress, active_vehicle_fraction=active_fraction)
                    actor_obs = build_actor_observations(**kwargs)
                    if policy in {'stochastic', 'deterministic'}:
                        with torch.no_grad():
                            selection = self.actor.sample_action(self.tensor(actor_obs.features),
                                torch.as_tensor(actor_obs.mask, device=self.device), deterministic=policy == 'deterministic')
                        indices = selection.candidate_index.cpu().tolist()
                        entropy = selection.entropy.cpu().tolist()
                    elif policy == 'random':
                        indices = [int(rng.choice(np.flatnonzero(mask))) for mask in actor_obs.mask]
                        entropy = [math.log(int(mask.sum())) for mask in actor_obs.mask]
                    elif policy == 'shortest':
                        indices = []
                        for vehicle, mask, edge_indices in zip(vehicles, actor_obs.mask, actor_obs.edge_indices):
                            source, destination = vehicle['current_edge'], vehicle['destination_edge']
                            costs = self.candidates.costs(destination)
                            def cost(index):
                                edge = int(edge_indices[index])
                                length, speed = self.candidates.junction(source, edge)
                                return length / speed + self.network.free_flow_travel_seconds[edge] + costs[edge]
                            indices.append(min(np.flatnonzero(mask), key=cost))
                        entropy = [0.] * len(vehicles)
                    else:
                        raise ValueError(policy)
                    decisions = []
                    for i, vehicle in enumerate(vehicles):
                        vehicle_id = vehicle['vehicle_id']
                        edge = int(actor_obs.edge_indices[i, indices[i]])
                        pair = f"{self.network.edge_ids[vehicle['current_edge']]}->{self.network.edge_ids[edge]}"
                        choices[pair] += 1
                        if actor_obs.mask[i].sum() > 1:
                            branch_choices[pair] += 1
                            branch_entropies.append(entropy[i])
                        if collect:
                            row = dict(vehicle_id=self.ids[vehicle_id], time=float(now),
                                features=actor_obs.features[i].copy(), mask=actor_obs.mask[i].copy(), action=indices[i], window=window_index)
                            rows.append(row)
                        decisions.append(NextHopDecision(vehicle_id, int(vehicle['current_edge']), edge, float(now)))
                    event_count += len(decisions)
                    return decisions

                result = session.advance_window(decide)
                if archive is not None:
                    archive.append(result)
                exposure += result.active_vehicle_seconds
                state = result.next_state
                windows.append(dict(obs=global_obs, next_obs=self.global_observation(state),
                    reward=-result.active_vehicle_seconds / (denominator * 60.),
                    terminal=bool(session.done or state['time'] >= self.deadline)))
            durations = np.asarray(list(session.travel_times.values()), dtype=np.float64)
            unfinished = self.n - len(durations)
            failed_ids = ({self.ids[v.vehicle_id] for v in vehicles if v.vehicle_id not in session.travel_times}
                if state['time'] >= self.deadline else set())
            attach_timeout_penalties(rows, failed_ids)
            global_reward_sum = sum(w['reward'] for w in windows)
            vehicle_timeout_penalty_sum = NONARRIVAL_PENALTY * len(failed_ids)
            metrics = dict(policy=policy, seed=int(seed), vehicle_count=self.n,
                **prior_metrics,
                demand_law='independent_od_poisson', demand_duration_seconds=self.demand.duration_seconds,
                demand_rate_veh_per_second=sum(rate for _, _, rate in self.demand.od_rates),
                demand_expected_vehicle_count=self.demand.expected_vehicle_count,
                demand_realization_sha256=demand_hash,
                completion_rate=len(durations) / self.n if self.n else 1., unfinished=unfinished,
                never_departed=sum(not vehicle["departed"] for vehicle in state["vehicles"]),
                mean_travel_seconds=float(durations.mean()) if durations.size else None,
                p90_travel_seconds=float(np.percentile(durations, 90)) if durations.size else None,
                exposure_seconds=exposure, mean_exposure_seconds=exposure / denominator,
                simulation_end_seconds=float(state['time']), microstep_seconds=session.microstep_seconds,
                microsteps=int(session.metrics['microsteps']), event_count=event_count,
                branch_event_count=len(branch_entropies), branch_entropy=float(np.mean(branch_entropies)) if branch_entropies else 0.,
                branch_choices=dict(branch_choices), choices=dict(choices), wall_seconds=time.monotonic() - started,
                window_count=len(windows), global_reward_sum=global_reward_sum,
                vehicle_timeout_penalty_sum=vehicle_timeout_penalty_sum,
                selection_score=global_reward_sum + vehicle_timeout_penalty_sum)
            if archive is not None:
                archive.save(archive_path, metrics=metrics, travel_times=session.travel_times,
                    completed=bool(session.done), deadline_seconds=self.deadline)
        if collect:
            self.collected_windows = windows
        return rows, metrics

    def update(self, rows, windows=None):
        windows = self.collected_windows if windows is None else windows
        global_advantage, global_targets, global_values = self.global_targets(windows)
        actor_loss_value, actor_gradient_value = 0., 0.
        if rows:
            event_windows = torch.tensor([r['window'] for r in rows], device=self.device)
            advantage = individual_actor_advantage(global_advantage[event_windows],
                self.tensor([r.get('timeout_return', 0.) for r in rows]))
            features = self.tensor(np.stack([r['features'] for r in rows]))
            mask = torch.as_tensor(np.stack([r['mask'] for r in rows]), device=self.device)
            actions = torch.tensor([r['action'] for r in rows], device=self.device)
            distribution = Categorical(logits=self.actor.forward_logits(features, mask))
            actor_loss = -(distribution.log_prob(actions) * advantage + .002 * distribution.entropy()).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            if not torch.isfinite(actor_gradient):
                raise RuntimeError('non-finite actor gradient')
            self.actor_optimizer.step()
            actor_loss_value, actor_gradient_value = float(actor_loss.detach()), float(actor_gradient)
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
        return dict(actor_loss=actor_loss_value, actor_gradient_norm=actor_gradient_value,
            global_advantage_std=float(global_advantage.std(unbiased=False)),
            global_critic_loss=float(global_loss.detach()), global_critic_gradient_norm=float(global_gradient),
            global_explained_variance=float(global_ev))

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
            input_hashes=self.prepared.input_hashes, config_fingerprint=self.config.semantic_fingerprint), temporary)
        temporary.replace(path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get('format') != CHECKPOINT_FORMAT:
            raise ValueError(f'unsupported checkpoint format; expected {CHECKPOINT_FORMAT}')
        if state['input_hashes'] != self.prepared.input_hashes or state['config_fingerprint'] != self.config.semantic_fingerprint:
            raise ValueError('experiment checkpoint mismatch')
        for key in ('seed', 'vehicle_limit', 'actor_lr', 'global_critic_lr', 'session', 'step_seconds', 'backend'):
            if state['args'][key] != getattr(self.args, key):
                raise ValueError(f'resume changed {key}')
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
        return state['episode'] + 1, state['best']


def log_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    short = {key: value for key, value in record.items() if key not in {'branch_choices', 'choices'}}
    print(json.dumps(short, allow_nan=False), flush=True)


def evaluate(runner, output, *, episode, seeds, phase, policies=('deterministic',)):
    if any(policy not in {'deterministic', 'random', 'shortest'} for policy in policies):
        raise ValueError('actor evaluation requires argmax; only random/shortest reference baselines are allowed')
    records = []
    enabled = getattr(getattr(runner, 'args', None), 'sp_prior_weight', 0.) > 0
    saved = actor_prior_metrics(runner.actor) if enabled else None
    try:
        if enabled:
            set_episode_prior(runner, episode)
        for policy in policies:
            labels = (policy, 'deterministic_no_prior') if enabled and policy == 'deterministic' else (policy,)
            for label in labels:
                if enabled:
                    if label == 'deterministic_no_prior':
                        set_shortest_path_prior(runner.actor, 0., prior_profile(runner.args)['sp_prior_temperature'])
                    else:
                        set_episode_prior(runner, episode)
                for seed in seeds:
                    _, metrics = runner.run_episode(policy=policy, seed=seed)
                    record = dict(phase=phase, episode=episode, **metrics)
                    if enabled:
                        record.update(policy=label)
                        if policy == 'deterministic':
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--vehicle-limit', type=int, default=0)
    parser.add_argument('--global-critic-lr', type=float, default=3e-4)
    parser.add_argument('--sp-prior-weight', type=float, default=0.)
    parser.add_argument('--sp-prior-temperature', type=float, default=30., help='Free-flow cost temperature in seconds')
    parser.add_argument('--sp-prior-hold-episodes', type=int, default=200)
    parser.add_argument('--sp-prior-off-episode', type=int, default=1000)
    parser.add_argument('--session', choices=('microstep', 'event'), default='microstep')
    parser.add_argument('--step-seconds', type=float, default=1.)
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default='traci')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save-episodes', action='store_true',
                        help='Archive every real training episode, including physical states and action events')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save latest checkpoint every N training episodes, plus evaluation and final episodes')
    parser.add_argument('--skip-reference-baselines', action='store_true',
                        help='Evaluate only the argmax actor; omit random and shortest reference runs')
    parser.add_argument('--skip-initial-evaluation', action='store_true',
                        help='Start training immediately; retain periodic and final evaluations')
    args = parser.parse_args(argv)
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
    evaluation_policies = ('deterministic',) if args.skip_reference_baselines else ('random', 'shortest', 'deterministic')
    ranking_policy = 'deterministic_no_prior' if args.sp_prior_weight > 0 else 'deterministic'
    start, best = 1, float('inf')
    if args.resume:
        start, best = runner.load(output / 'latest.pt')
    else:
        protocol = dict(world_model=False, macro_latent_critic=False,
            shortest_path_prior=dict(**prior_profile(args), schedule='constant through hold_episodes; cosine decay to zero at off_episode',
                application='actor logits minus weight * candidate free-flow excess seconds / temperature; unchanged reward',
                clock='completed training episode number; fixed during collection and update; evaluation does not advance clock',
                checkpoint_selection_policy=ranking_policy,
                evaluation='paired identical demand seeds with scheduled prior and zero prior when enabled'),
            reward_mode='global_with_individual_timeout',
            evaluation_policies=list(evaluation_policies) + (['deterministic_no_prior'] if args.sp_prior_weight > 0 else []),
            skip_initial_evaluation=args.skip_initial_evaluation,
            session=args.session, step_seconds=args.step_seconds, backend=args.backend,
            reward='shared -active_vehicle_seconds/(N*60); individual -5000 for each unfinished vehicle at deadline',
            gamma=1., lambda_return=.95, entropy_coefficient=.002,
            demand=dict(law='independent_od_poisson', seed=args.seed,
                episode_index='SUMO episode seed', duration_seconds=runner.demand.duration_seconds,
                expected_vehicle_count=runner.demand.expected_vehicle_count,
                od_rates=[dict(origin_index=o, destination_index=d, rate_veh_per_second=rate)
                          for o, d, rate in runner.demand.od_rates],
                normalization='realized episode population N; denominator max(N,1) for empty episodes'),
            advantage_normalization=runner.advantage_normalization,
            global_reward='-active_vehicle_seconds/(N*60)',
            timeout_penalty='-5000 once on each unfinished vehicle last action at the real deadline; '
                'gamma=1 terminal return applies only to that vehicle prior actions; excluded from global critic',
            global_critic_observation='flattened physical dynamic flow + progress/active fraction',
            deadline_seconds=runner.deadline, config=runner.config.resolved_snapshot(),
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            training_policy='stochastic', evaluation_policy='argmax', checkpoint_format=CHECKPOINT_FORMAT,
            input_hashes=runner.prepared.input_hashes, evaluation_seeds=evaluation_seeds,
            note='Global Actor-Critic with individual timeout penalties; no vehicle critic, world model, or imagination.',
            entropy_note='branch_entropy is entropy of the policy distribution, including in argmax evaluation.',
            reward_note='global_reward_sum excludes timeouts; vehicle_timeout_penalty_sum is diagnostic only; '
                'selection_score combines both only for checkpoint ranking.')
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        if not args.skip_initial_evaluation:
            baseline = evaluate(runner, output, episode=0, seeds=evaluation_seeds, phase='baseline',
                     policies=evaluation_policies)
            best = -float(np.mean([r['selection_score'] for r in baseline if r['policy'] == ranking_policy]))
        runner.save(output / 'best.pt', 0, best)
        runner.save(output / 'initial.pt', 0, best)
        runner.save(output / 'latest.pt', 0, best)
    for episode in range(start, args.episodes + 1):
        set_episode_prior(runner, episode)
        archive_kwargs = {}
        if args.save_episodes:
            archive_kwargs = dict(archive_path=output / 'episodes' / 'train' / f'episode_{episode:06d}.npz',
                                  archive_metadata=dict(phase='train', episode=episode))
        rows, metrics = runner.run_episode(policy='stochastic', seed=args.seed + episode, collect=True,
                                           **archive_kwargs)
        update = runner.update(rows, runner.collected_windows)
        log_record(output / 'metrics.jsonl', dict(phase='train', episode=episode, **metrics, **update))
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
