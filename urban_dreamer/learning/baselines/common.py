"""Shared episode collection and global-reward updates for routing baselines."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from ..global_reward import (REWARD_CONTRACT, causal_context, observed_mean_seconds, started_count_from_state)
from ..observations import build_actor_observations
from ..sumo_sampling import SumoEpisodeSampler, resolve_session_factory
from ...models.policy import lambda_return, attach_timeout_penalties, individual_actor_advantage
from ...traffic.environment import NextHopDecision, SumoSession, WindowResult
from ...traffic.demand import PoissonDemand
from ...traffic.routing import validate_prepared_inputs
from ...traffic.topology import NextHopCandidates, load_traffic_topology
from ...traffic.simulation_settings import physics_identity
from ...workflow.config import NONARRIVAL_PENALTY, load_config

CHECKPOINT_FORMAT = 'toy_global_individual_timeout_poisson_baseline_v1'



def evaluation_policy_label(temperature=.2, *, deterministic=True):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('evaluation temperature must be finite and positive')
    return 'deterministic' if deterministic else f'sampled_temperature_{temperature:g}'


@contextmanager
def evaluation_rng(seed, method=None):
    """Fixed evaluation draws leave the training random streams untouched."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = getattr(method, 'rng', None)
    generator_state = copy.deepcopy(generator.bit_generator.state) if isinstance(generator, np.random.Generator) else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if generator_state is not None:
            generator.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if generator_state is not None:
            generator.bit_generator.state = generator_state


def reward_objective(args):
    return getattr(args, 'reward_objective', 'observed_mean')


def reward_context_dim(args):
    return 2 if reward_objective(args) == 'legacy_exposure' else 4


def accounting_context(state, deadline):
    """Only revealed trip starts and actually accumulated physical time."""
    started = state.get('started_vehicle_count')
    if started is None:
        started = started_count_from_state(state, deadline=deadline)
    return causal_context(timestamp=float(state['time']), deadline=deadline,
        active_count=int(state.get('active_vehicle_count', 0)), started_count=int(started),
        cumulative_seconds=float(state.get('cumulative_vehicle_seconds', 0.)))


def window_accounting(window):
    """Recover a saved seed's cumulative accounting from its causal context."""
    context = np.asarray(window['context'])
    if context.size != 4:
        raise ValueError('paper reward imagination requires four causal context features')
    return int(round(np.expm1(float(context[3])))), float(np.expm1(float(context[2])) * 60.)


def resolve_collection_settings(config, args):
    """Resolve omitted CLI settings from config and enforce the mesoscopic session contract."""
    simulation = config.simulation
    requested = {
        'session': getattr(args, 'session', None),
        'step_seconds': getattr(args, 'step_seconds', None),
        'backend': getattr(args, 'backend', None),
    }
    configured = dict(session=simulation.session_protocol,
        step_seconds=float(simulation.observation_step_seconds), backend=simulation.backend)
    if simulation.traffic_model == 'meso':
        for key, value in requested.items():
            if value is not None and value != configured[key]:
                raise ValueError(f'meso traffic requires {key}={configured[key]!r}; requested {value!r}')
    resolved = {key: configured[key] if value is None else value for key, value in requested.items()}
    if resolved['session'] not in {'event', 'microstep'}:
        raise ValueError('session must be event or microstep')
    if resolved['backend'] not in {'traci', 'libsumo'}:
        raise ValueError('backend must be traci or libsumo')
    if not math.isfinite(resolved['step_seconds']) or resolved['step_seconds'] <= 0:
        raise ValueError('step_seconds must be finite and positive')
    for key, value in resolved.items():
        setattr(args, key, value)
    return dict(physics_identity(simulation), session_protocol=resolved['session'],
        observation_step_seconds=float(resolved['step_seconds']), backend=resolved['backend'])


class EpisodeRunner:
    """SUMO episode lifecycle, event accounting and metrics; no networks."""

    def __init__(self, args, session_factory=None):
        self.args = args
        args.reward_objective = reward_objective(args)
        args.global_reward_only = bool(getattr(args, 'global_reward_only', True))
        if args.reward_objective == 'observed_mean' and not args.global_reward_only:
            raise ValueError('observed_mean requires global_reward_only=True')
        # Resolved lazily in run_episode so tests can monkeypatch common.SumoSession.
        self.session_factory = session_factory
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device(args.device)
        self.config = load_config(args.config)
        if reward_objective(args) == 'legacy_exposure':
            from ...workflow.config import legacy_semantic_fingerprint
            self.config = replace(self.config, semantic_fingerprint=legacy_semantic_fingerprint(self.config))
        self.config = replace(self.config, run=replace(self.config.run, device=args.device))
        self.simulation_settings = resolve_collection_settings(self.config, args)
        if getattr(args, 'forbid_revisit', False):
            if args.session != 'event' or getattr(args, 'sp_prior_weight', 0.) != 0.:
                raise ValueError('forbid_revisit requires event session and zero shortest-path prior')
        if session_factory is None:
            self.session_factory = resolve_session_factory(args.session, args.step_seconds, args.backend,
                forbid_revisit=bool(getattr(args, 'forbid_revisit', False)))
        self.prepared = validate_prepared_inputs(self.config)
        self.topology = load_traffic_topology(self.prepared)
        if getattr(args, 'vehicle_limit', 0):
            self.topology = replace(self.topology, vehicles=self.topology.vehicles[:args.vehicle_limit])
        self.network = self.topology.network
        self.demand = PoissonDemand.from_vehicles(self.topology.vehicles,
            duration_seconds=self.config.simulation.demand_duration_seconds)
        self.episode_topology = self.topology
        self.n = len(self.topology.vehicles)
        self.candidates = NextHopCandidates(self.network,
            traffic_model=self.config.simulation.traffic_model, **(
            {'forbid_revisit': True} if getattr(args, 'forbid_revisit', False) else {}))
        simulation = self.config.simulation
        self.deadline = 30 * math.ceil((simulation.world_model_intervals_per_episode * 30
                                       + simulation.training_completion_buffer_seconds) / 30)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(self.topology.vehicles)}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def global_observation(self, state):
        return np.concatenate((np.asarray(state['flow'][:, :4], dtype=np.float32).reshape(-1),
            (np.asarray([state['time'] / self.deadline, state['active_vehicle_count'] / max(self.n, 1)], dtype=np.float32)
             if reward_objective(self.args) == 'legacy_exposure' else accounting_context(state, self.deadline))))

    def make_random_select(self, rng):
        """Uniform-at-random evaluation policy over the legal candidates."""
        def select(vehicles, actor_obs, frozen_flow, now):
            indices = [int(rng.choice(np.flatnonzero(mask))) for mask in actor_obs.mask]
            aux = [dict(entropy=math.log(int(mask.sum()))) for mask in actor_obs.mask]
            return indices, aux
        return select

    def make_shortest_select(self):
        """Free-flow policy using the configured model's junction, edge and remaining costs."""
        def select(vehicles, actor_obs, frozen_flow, now):
            indices = []
            for vehicle, mask, edge_indices in zip(vehicles, actor_obs.mask, actor_obs.edge_indices):
                source, destination = vehicle['current_edge'], vehicle['destination_edge']
                costs = self.candidates.costs(destination)
                def cost(index):
                    edge = int(edge_indices[index])
                    length, speed = self.candidates.junction(source, edge)
                    return length / speed + self.network.free_flow_travel_seconds[edge] + costs[edge]
                indices.append(min(np.flatnonzero(mask), key=cost))
            return indices, [dict(entropy=0.)] * len(vehicles)
        return select

    def run_episode(self, select, *, seed, collect=False, record_windows=False, row_flow=True, label=None,
                    on_demand=None, archive_path=None, archive_metadata=None,
                    session_factory=None, observation_builder=None, empty_terminal=False,
                    metadata_transform=None, metrics_transform=None):
        """Run one episode; ``select(vehicles, actor_obs, frozen_flow, now)`` picks candidates.

        ``select`` returns either a list of candidate indices (one per pending
        vehicle) or a ``(indices, aux)`` tuple whose per-vehicle ``aux`` dicts
        are merged into the collected rows; ``aux[i]['entropy']`` feeds the
        branch-entropy statistics. With ``record_windows=True`` each window
        also carries the full macro transition (flow/next_flow/action/departure/
        context/vehicles/time/active_vehicle_seconds/total_vehicle_count) for
        model-based methods. ``on_demand(topology, seed=seed)`` runs after the
        independent Poisson draw and before session reset, so assignment methods
        can prepare routes for the realized vehicles without fixed ID assumptions.
        ``archive_path`` saves every real macro transition independently of
        model-specific learning records, including zero-population episodes.
        Returns ``(rows, windows, metrics)``.

        Compatibility adapters may inject session/observation factories and
        transform archive metadata and metrics before saving. ``empty_terminal``
        preserves AC's synthetic terminal window without advancing empty SUMO.
        """
        started = time.monotonic()
        vehicles = self.demand.sample(seed=self.args.seed, episode_index=int(seed))
        self.episode_topology = replace(self.topology, vehicles=vehicles)
        self.n = len(vehicles)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(vehicles)}
        denominator = max(self.n, 1)
        demand_payload = [{'vehicle_id': v.vehicle_id, 'origin_index': v.origin_index,
                           'destination_index': v.destination_index, 'departure_second': v.departure_second}
                          for v in vehicles]
        demand_sha256 = hashlib.sha256(json.dumps(demand_payload, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        if on_demand is not None:
            on_demand(self.episode_topology, seed=seed)
        config = replace(self.config, run=replace(self.config.run, seed=int(seed)))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if isinstance(select, str):
            label = label or select
            if select == 'random':
                select = EpisodeRunner.make_random_select(self, rng)
            elif select == 'shortest':
                select = EpisodeRunner.make_shortest_select(self)
            else:
                raise ValueError(select)
        label = label or getattr(select, 'label', 'method')
        rows = []
        choices, branch_choices = Counter(), Counter()
        branch_entropies = []
        exposure, event_count = 0., 0
        previous_mean = 0.
        windows = []
        session = (session_factory or self.session_factory or SumoSession)(config, self.prepared, self.episode_topology)
        with SumoEpisodeSampler(session=session, candidates=self.candidates,
                deadline_seconds=self.deadline, total_vehicle_count=self.n,
                context_mode='window', sort_pending=False, pad_empty=False,
                advance_empty=not empty_terminal,
                observation_builder=observation_builder or build_actor_observations,
                reward_objective=reward_objective(self.args)) as sampler:
            session = sampler.session
            state = sampler.state
            archive = None
            if archive_path is not None:
                from ..episode_archive import EpisodeArchive
                metadata = dict(archive_metadata or {})
                metadata.update(city=self.config.inputs.network_name,
                    method=getattr(self.args, 'method', label), policy=label,
                    seed=int(seed), demand_seed=int(self.args.seed),
                    total_vehicle_count=self.n, realized_demand=demand_payload,
                    demand_sha256=demand_sha256,
                    demand=dict(law='independent_poisson_per_od',
                        duration_seconds=self.demand.duration_seconds,
                        expected_vehicle_count=self.demand.expected_vehicle_count,
                        od_rates=[dict(origin_index=o, destination_index=d,
                                       rate_vehicles_per_second=rate)
                                  for o, d, rate in self.demand.od_rates]),
                    edge_ids=list(self.network.edge_ids),
                    config=config.resolved_snapshot(),
                    config_fingerprint=self.config.semantic_fingerprint,
                    input_hashes=self.prepared.input_hashes,
                    session=getattr(self.args, 'session', 'microstep'),
                    session_protocol=getattr(self.args, 'session', 'microstep'),
                    forbid_revisit=bool(getattr(self.args, 'forbid_revisit', False)),
                    reward_objective=reward_objective(self.args),
                    reward_contract=(REWARD_CONTRACT if reward_objective(self.args) == 'observed_mean' else 'legacy_exposure_v1'),
                    reward_mode=('global_only' if getattr(self.args, 'global_reward_only', True)
                                 else 'global_with_individual_timeout'),
                    backend=getattr(self.args, 'backend', 'traci'),
                    step_seconds=float(session.microstep_seconds))
                metadata.update(self.simulation_settings)
                if metadata_transform is not None:
                    metadata = metadata_transform(metadata)
                archive = EpisodeArchive(metadata)
                archive.start(state)
            if empty_terminal and not self.n:
                observation = self.global_observation(state)
                windows.append(dict(obs=observation, next_obs=observation.copy(), reward=0., terminal=True))
                if archive is not None:
                    next_state = dict(state, time=float(state['time']) + 30., done=True)
                    zero_counts = np.zeros(self.network.edge_count, dtype=np.int64)
                    archive.append(WindowResult((), zero_counts, zero_counts.copy(), 0.,
                        next_state, float(state['time']), float(next_state['time'])))
            global_obs = row_window_flow = None
            window_flow = window_vehicles = window_context = window_time = None

            def before_window(index, start_state, context):
                nonlocal global_obs, row_window_flow
                nonlocal window_flow, window_vehicles, window_context, window_time
                global_obs = self.global_observation(start_state)
                row_window_flow = None
                window_time = float(start_state['time'])
                if record_windows:
                    window_flow = np.asarray(start_state['flow'], dtype=np.float32).copy()
                    window_vehicles = copy.deepcopy(tuple(vehicle for vehicle in start_state['vehicles']
                        if vehicle['departed'] and not vehicle['arrived']))
                    window_context = np.asarray(context, dtype=np.float32)

            def decide(vehicles, actor_obs, frozen, now, window_index, context):
                nonlocal event_count, row_window_flow
                # Rows share a read-only macro snapshot, copied before session advancement.
                if collect and row_flow and vehicles and row_window_flow is None:
                    row_window_flow = np.asarray(frozen, dtype=np.float32).copy()
                selected = select(vehicles, actor_obs, frozen, now)
                if isinstance(selected, tuple):
                    indices, aux = selected
                    aux = [dict(entry) for entry in aux]
                else:
                    indices, aux = list(selected), [{} for _ in vehicles]
                indices = [int(index) for index in indices]
                if len(indices) != len(vehicles) or len(aux) != len(vehicles):
                    raise ValueError('select must return one candidate index per pending vehicle')
                decisions = []
                for i, vehicle in enumerate(vehicles):
                    vehicle_id = vehicle['vehicle_id']
                    if not 0 <= indices[i] < actor_obs.mask.shape[1] or not actor_obs.mask[i, indices[i]]:
                        raise ValueError(f'select returned illegal candidate index {indices[i]} for {vehicle_id}')
                    edge = int(actor_obs.edge_indices[i, indices[i]])
                    pair = f"{self.network.edge_ids[vehicle['current_edge']]}->{self.network.edge_ids[edge]}"
                    choices[pair] += 1
                    if actor_obs.mask[i].sum() > 1:
                        branch_choices[pair] += 1
                        branch_entropies.append(float(aux[i].get('entropy', 0.)))
                    if collect:
                        row = dict(vehicle_id=vehicle_id, time=float(now),
                            features=actor_obs.features[i].copy(), mask=actor_obs.mask[i].copy(), action=indices[i],
                            window=window_index, current_edge=int(vehicle['current_edge']),
                            edge_indices=actor_obs.edge_indices[i].copy())
                        if row_flow:
                            row['flow'] = row_window_flow
                        for key, value in aux[i].items():
                            if key in row:
                                raise ValueError(f'aux field {key!r} collides with a reserved row field')
                            row[key] = value
                        rows.append(row)
                    decisions.append(NextHopDecision(vehicle_id, int(vehicle['current_edge']), edge, float(now)))
                event_count += len(decisions)
                return decisions

            for sample in sampler.windows(decide, before_window=before_window):
                result = sample.result
                if archive is not None:
                    archive.append(result)
                exposure += result.active_vehicle_seconds
                state = result.next_state
                state.setdefault('cumulative_vehicle_seconds', exposure)
                state.setdefault('started_vehicle_count', started_count_from_state(state, deadline=self.deadline))
                current_mean = observed_mean_seconds(exposure, state['started_vehicle_count'])
                reward = (-result.active_vehicle_seconds / (denominator * 60.)
                          if reward_objective(self.args) == 'legacy_exposure'
                          else -(current_mean - previous_mean) / 60.)
                previous_mean = current_mean
                window = dict(obs=global_obs, next_obs=self.global_observation(state),
                    reward=reward,
                    start_time=window_time, end_time=float(state['time']),
                    terminal=bool(session.done or state['time'] >= self.deadline))
                if record_windows:
                    window.update(flow=window_flow, next_flow=np.asarray(state['flow'], dtype=np.float32).copy(),
                        action=(result.action_counts / denominator).astype(np.float32),
                        departure=(result.departure_counts / denominator).astype(np.float32),
                        context=window_context, vehicles=window_vehicles, time=window_time,
                        active_vehicle_seconds=float(result.active_vehicle_seconds), total_vehicle_count=self.n,
                        started_vehicle_count=int(sample.start_state.get('started_vehicle_count', 0)),
                        cumulative_vehicle_seconds=float(sample.start_state.get('cumulative_vehicle_seconds', 0.)),
                        next_started_vehicle_count=int(state['started_vehicle_count']),
                        next_cumulative_vehicle_seconds=float(state['cumulative_vehicle_seconds']))
                windows.append(window)
            durations = np.asarray(list(session.travel_times.values()), dtype=np.float64)
            unfinished = self.n - len(durations)
            failed_ids = ({vehicle.vehicle_id for vehicle in self.episode_topology.vehicles}
                          - set(session.travel_times)) if state['time'] >= self.deadline else set()
            global_only = bool(getattr(self.args, 'global_reward_only', True))
            if not global_only:
                attach_timeout_penalties(rows, failed_ids)
            timeout_penalty = 0. if global_only else NONARRIVAL_PENALTY * len(failed_ids)
            global_reward = sum(window['reward'] for window in windows)
            metrics = dict(policy=label, seed=int(seed), vehicle_count=self.n,
                demand_sha256=demand_sha256, demand_duration_seconds=self.demand.duration_seconds,
                completion_rate=len(durations) / self.n if self.n else 1., unfinished=unfinished,
                never_departed=sum(not vehicle["departed"] for vehicle in state["vehicles"]),
                mean_travel_seconds=float(durations.mean()) if durations.size else None,
                p90_travel_seconds=float(np.percentile(durations, 90)) if durations.size else None,
                exposure_seconds=exposure, mean_exposure_seconds=(exposure / denominator
                    if reward_objective(self.args) == 'legacy_exposure' else previous_mean),
                observed_mean_seconds=previous_mean, cumulative_vehicle_seconds=exposure,
                reward_contract=(REWARD_CONTRACT if reward_objective(self.args) == 'observed_mean' else 'legacy_exposure_v1'),
                started_vehicle_count=int(state.get('started_vehicle_count', 0)),
                reward_objective=reward_objective(self.args),
                simulation_end_seconds=float(state['time']), microstep_seconds=session.microstep_seconds,
                microsteps=int(session.metrics['microsteps']), event_count=event_count,
                branch_event_count=len(branch_entropies), branch_entropy=float(np.mean(branch_entropies)) if branch_entropies else 0.,
                branch_choices=dict(branch_choices), choices=dict(choices), wall_seconds=time.monotonic() - started,
                window_count=len(windows), global_reward_sum=global_reward,
                vehicle_timeout_penalty_sum=timeout_penalty, selection_score=global_reward + timeout_penalty)
            metrics.update(collection_seconds=metrics['wall_seconds'], archive_seconds=0.)
            if metrics_transform is not None:
                metrics = metrics_transform(metrics)
            if archive is not None:
                archive_started = time.monotonic()
                archive.save(archive_path, metrics=metrics, travel_times=dict(session.travel_times),
                    completed=bool(session.done), deadline_seconds=self.deadline)
                metrics['archive_seconds'] = time.monotonic() - archive_started
        metrics['collection_seconds'] = time.monotonic() - started
        return rows, windows, metrics


class GlobalTrafficCritic(torch.nn.Module):
    """Value of the real full-network traffic state for the shared window reward."""
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


def global_reward_integral(windows, *, deadline):
    """Return cumulative team reward and episode end for asynchronous events.

    Exact window integrals are allocated uniformly inside each window because
    archives do not contain a per-second reward trace. Prefer explicit physical
    bounds; normalized observations are only a legacy fallback.
    """
    if not windows:
        raise ValueError('global reward integration needs collected windows')
    starts, ends, rewards = [], [], []
    for index, window in enumerate(windows):
        start = window.get('start_time')
        if start is None:
            start = (float(window['obs'][-4 if len(window['obs']) % 4 == 0 else -2]) * deadline if 'obs' in window
                     else float(window.get('time', index * 30.)))
        end = window.get('end_time')
        if end is None:
            end = (float(window['next_obs'][-4 if len(window['next_obs']) % 4 == 0 else -2]) * deadline if 'next_obs' in window
                   else min(float(start) + 30., deadline))
        starts.append(float(start))
        ends.append(float(end))
        rewards.append(float(window['reward']))
    starts, ends, rewards = (np.asarray(values, dtype=np.float64)
                            for values in (starts, ends, rewards))
    if (not all(np.isfinite(values).all() for values in (starts, ends, rewards))
            or (ends <= starts).any()
            or not np.allclose(starts[1:], ends[:-1], atol=1e-3, rtol=0)):
        raise ValueError('global reward windows must be finite, positive and contiguous')
    cumulative = np.concatenate(([0.], np.cumsum(rewards)))

    def integrated(timestamp):
        timestamp = np.asarray(timestamp, dtype=np.float64)
        if (not np.isfinite(timestamp).all() or (timestamp < starts[0] - 1e-3).any()
                or (timestamp > ends[-1] + 1e-3).any()):
            raise ValueError('decision time outside collected reward windows')
        timestamp = np.clip(timestamp, starts[0], ends[-1])
        index = np.minimum(np.searchsorted(starts, timestamp, side='right') - 1, len(starts) - 1)
        fraction = (timestamp - starts[index]) / (ends[index] - starts[index])
        result = cumulative[index] + rewards[index] * fraction
        return float(result) if result.ndim == 0 else result

    return integrated, float(ends[-1])


def global_critic_update(*, global_critic, global_optimizer=None, windows, grad_clip=10.0, update=True):
    """Global-critic evaluation (and optional fit) on shared-reward windows.

    Mirrors the reference global branch: advantages are per-window lambda-return
    targets minus the pre-update value estimate. With ``update=False`` no
    optimizer step is taken (used to score imagined windows). Returns
    ``(advantages, metrics)`` with advantages detached, one per window.
    """
    if not windows:
        raise ValueError('global critic needs collected macro windows')
    device = next(global_critic.parameters()).device
    obs = torch.as_tensor(np.stack([w['obs'] for w in windows]), dtype=torch.float32, device=device)
    next_obs = torch.as_tensor(np.stack([w['next_obs'] for w in windows]), dtype=torch.float32, device=device)
    rewards = torch.as_tensor([w['reward'] for w in windows], dtype=torch.float32, device=device)
    terminal = torch.tensor([w['terminal'] for w in windows], dtype=torch.bool, device=device)
    if update:
        values = global_critic(obs).squeeze(-1)
    else:
        with torch.no_grad():
            values = global_critic(obs).squeeze(-1)
    with torch.no_grad():
        next_values = global_critic(next_obs).squeeze(-1)
        targets = lambda_return(rewards, next_values, terminal=terminal)
    advantages = (targets - values.detach()).detach()
    metrics = dict(global_advantage_std=float(advantages.std(unbiased=False)))
    if update:
        if global_optimizer is None:
            raise ValueError('global critic update needs an optimizer')
        loss = F.mse_loss(values, targets)
        global_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(global_critic.parameters(), grad_clip)
        if not torch.isfinite(gradient):
            raise RuntimeError('non-finite global critic gradient')
        global_optimizer.step()
        with torch.no_grad():
            fitted = global_critic(obs).squeeze(-1)
            explained = 1 - torch.var(targets - fitted, unbiased=False) / torch.var(targets, unbiased=False).clamp_min(1e-8)
        metrics.update(global_critic_loss=float(loss.detach()), global_critic_gradient_norm=float(gradient),
                       global_explained_variance=float(explained))
    return advantages, metrics


def policy_gradient_step(*, compute_logits, actor_parameters, actor_optimizer, rows, advantages,
                         entropy_coef=0.002, grad_clip=10.0, include_individual_timeout=True):
    """Actor-only update with externally supplied per-row advantages.

    Scale shared advantages without centering, then add each decision vehicle's
    own timeout return before clipping. Timeout costs never affect shared scaling.
    """
    if not rows:
        return dict(actor_loss=0., actor_gradient_norm=0., advantage_std=0.)
    actor_parameters = list(actor_parameters)
    device = actor_parameters[0].device
    if isinstance(advantages, torch.Tensor):
        # GNAC already supplies device-resident advantages; avoid one host read per event.
        advantage = advantages.detach().to(device=device, dtype=torch.float32)
    else:
        advantage = torch.as_tensor(np.asarray([float(a) for a in advantages], dtype=np.float32), device=device)
    if advantage.shape != (len(rows),) or not bool(torch.isfinite(advantage).all()):
        raise ValueError('advantages must be finite with one value per row')
    timeout_returns = (torch.tensor([row.get('timeout_return', 0.) for row in rows],
                                   dtype=advantage.dtype, device=device)
                       if include_individual_timeout else torch.zeros_like(advantage))
    advantage = individual_actor_advantage(advantage, timeout_returns)
    actions = torch.tensor([r['action'] for r in rows], device=device)
    distribution = Categorical(logits=compute_logits(rows))
    actor_loss = -(distribution.log_prob(actions) * advantage + entropy_coef * distribution.entropy()).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_gradient = torch.nn.utils.clip_grad_norm_(actor_parameters, grad_clip)
    if not torch.isfinite(actor_gradient):
        raise RuntimeError('non-finite actor gradient')
    actor_optimizer.step()
    return dict(actor_loss=float(actor_loss.detach()), actor_gradient_norm=float(actor_gradient),
                advantage_std=float(advantage.std(unbiased=False)))


def save_checkpoint(path, *, method_name, episode, best, method_state, args, prepared, config):
    payload = dict(format=CHECKPOINT_FORMAT, method=method_name, episode=episode, best=best,
        method_state=method_state,
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        input_hashes=prepared.input_hashes, config_fingerprint=config.semantic_fingerprint,
        physics=physics_identity(getattr(config, 'simulation', {})))
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path, *, method_name, prepared, config, device='cpu'):
    # Replays can be many GiB and belong on the host. Module/optimizer loading
    # moves only their own tensors to the existing learner parameter devices.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('format') != CHECKPOINT_FORMAT or state.get('method') != method_name
            or state.get('input_hashes') != prepared.input_hashes
            or state.get('config_fingerprint') != config.semantic_fingerprint
            or state.get('physics', {'traffic_model': 'micro'}) != physics_identity(getattr(config, 'simulation', {}))):
        raise ValueError('experiment checkpoint mismatch')
    return state
