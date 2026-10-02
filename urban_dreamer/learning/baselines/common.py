"""Shared episode collection and global-reward updates for routing baselines."""
from __future__ import annotations

from collections import Counter
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

from ..observations import build_actor_observations
from ...models.policy import lambda_return, attach_timeout_penalties, individual_actor_advantage
from ...traffic.environment import NextHopDecision, SumoSession
from ...traffic.demand import PoissonDemand
from ...traffic.routing import validate_prepared_inputs
from ...traffic.topology import NextHopCandidates, load_traffic_topology
from ...workflow.config import NONARRIVAL_PENALTY, load_config

CHECKPOINT_FORMAT = 'toy_global_individual_timeout_poisson_baseline_v1'


class EpisodeRunner:
    """SUMO episode lifecycle, event accounting and metrics; no networks."""

    def __init__(self, args, session_factory=None):
        self.args = args
        # Resolved lazily in run_episode so tests can monkeypatch common.SumoSession.
        self.session_factory = session_factory
        torch.set_num_threads(args.threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        self.device = torch.device(args.device)
        self.config = load_config(args.config)
        self.config = replace(self.config, run=replace(self.config.run, device=args.device))
        self.prepared = validate_prepared_inputs(self.config)
        self.topology = load_traffic_topology(self.prepared)
        if getattr(args, 'vehicle_limit', 0):
            self.topology = replace(self.topology, vehicles=self.topology.vehicles[:args.vehicle_limit])
        self.network = self.topology.network
        self.demand = PoissonDemand.from_vehicles(self.topology.vehicles,
            duration_seconds=self.config.simulation.demand_duration_seconds)
        self.episode_topology = self.topology
        self.n = len(self.topology.vehicles)
        self.candidates = NextHopCandidates(self.network)
        simulation = self.config.simulation
        self.deadline = 30 * math.ceil((simulation.world_model_intervals_per_episode * 30
                                       + simulation.training_completion_buffer_seconds) / 30)
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(self.topology.vehicles)}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def global_observation(self, state):
        return np.concatenate((np.asarray(state['flow'][:, :4], dtype=np.float32).reshape(-1),
            np.asarray([state['time'] / self.deadline, state['active_vehicle_count'] / max(self.n, 1)], dtype=np.float32)))

    def make_random_select(self, rng):
        """Uniform-at-random evaluation policy over the legal candidates."""
        def select(vehicles, actor_obs, frozen_flow, now):
            indices = [int(rng.choice(np.flatnonzero(mask))) for mask in actor_obs.mask]
            aux = [dict(entropy=math.log(int(mask.sum()))) for mask in actor_obs.mask]
            return indices, aux
        return select

    def make_shortest_select(self):
        """Free-flow shortest-path evaluation policy (junction + edge + remaining cost)."""
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
                    on_demand=None, archive_path=None, archive_metadata=None):
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
                select = self.make_random_select(rng)
            elif select == 'shortest':
                select = self.make_shortest_select()
            else:
                raise ValueError(select)
        label = label or getattr(select, 'label', 'method')
        rows = []
        choices, branch_choices = Counter(), Counter()
        branch_entropies = []
        exposure, event_count = 0., 0
        windows = []
        with (self.session_factory or SumoSession)(config, self.prepared, self.episode_topology) as session:
            state = session.reset()
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
                    backend=getattr(self.args, 'backend', 'traci'),
                    step_seconds=float(session.microstep_seconds))
                archive = EpisodeArchive(metadata)
                archive.start(state)
            while state['time'] < self.deadline and (not session.done or not windows):
                progress = state['time'] / self.deadline
                active_fraction = state['active_vehicle_count'] / denominator
                window_index = len(windows)
                global_obs = self.global_observation(state)
                row_window_flow = None
                if record_windows:
                    window_flow = np.asarray(state['flow'], dtype=np.float32).copy()
                    window_vehicles = copy.deepcopy(tuple(vehicle for vehicle in state['vehicles']
                        if vehicle['departed'] and not vehicle['arrived']))
                    window_context = np.asarray([progress, active_fraction], dtype=np.float32)
                    window_time = float(state['time'])

                def decide(vehicles, frozen, now):
                    nonlocal event_count, row_window_flow
                    # Sessions freeze traffic for a macro window. All event rows
                    # share one snapshot; consumers must treat it as read-only.
                    if collect and row_flow and vehicles and row_window_flow is None:
                        row_window_flow = np.asarray(frozen, dtype=np.float32).copy()
                    kwargs = dict(policy_edge_features=np.asarray(frozen[:, [2, 1]], dtype=np.float32),
                        vehicle_states=vehicles, candidates=self.candidates,
                        episode_progress=progress, active_vehicle_fraction=active_fraction)
                    actor_obs = build_actor_observations(**kwargs)
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

                result = session.advance_window(decide)
                if archive is not None:
                    archive.append(result)
                exposure += result.active_vehicle_seconds
                state = result.next_state
                window = dict(obs=global_obs, next_obs=self.global_observation(state),
                    reward=-result.active_vehicle_seconds / (denominator * 60.),
                    terminal=bool(session.done or state['time'] >= self.deadline))
                if record_windows:
                    window.update(flow=window_flow, next_flow=np.asarray(state['flow'], dtype=np.float32).copy(),
                        action=(result.action_counts / denominator).astype(np.float32),
                        departure=(result.departure_counts / denominator).astype(np.float32),
                        context=window_context, vehicles=window_vehicles, time=window_time,
                        active_vehicle_seconds=float(result.active_vehicle_seconds), total_vehicle_count=self.n)
                windows.append(window)
            durations = np.asarray(list(session.travel_times.values()), dtype=np.float64)
            unfinished = self.n - len(durations)
            failed_ids = ({vehicle.vehicle_id for vehicle in self.episode_topology.vehicles}
                          - set(session.travel_times)) if state['time'] >= self.deadline else set()
            attach_timeout_penalties(rows, failed_ids)
            timeout_penalty = NONARRIVAL_PENALTY * len(failed_ids)
            global_reward = sum(window['reward'] for window in windows)
            metrics = dict(policy=label, seed=int(seed), vehicle_count=self.n,
                demand_sha256=demand_sha256, demand_duration_seconds=self.demand.duration_seconds,
                completion_rate=len(durations) / self.n if self.n else 1., unfinished=unfinished,
                never_departed=sum(not vehicle["departed"] for vehicle in state["vehicles"]),
                mean_travel_seconds=float(durations.mean()) if durations.size else None,
                p90_travel_seconds=float(np.percentile(durations, 90)) if durations.size else None,
                exposure_seconds=exposure, mean_exposure_seconds=exposure / denominator,
                simulation_end_seconds=float(state['time']), microstep_seconds=session.microstep_seconds,
                microsteps=int(session.metrics['microsteps']), event_count=event_count,
                branch_event_count=len(branch_entropies), branch_entropy=float(np.mean(branch_entropies)) if branch_entropies else 0.,
                branch_choices=dict(branch_choices), choices=dict(choices), wall_seconds=time.monotonic() - started,
                window_count=len(windows), global_reward_sum=global_reward,
                vehicle_timeout_penalty_sum=timeout_penalty, selection_score=global_reward + timeout_penalty)
            if archive is not None:
                archive.save(archive_path, metrics=metrics, travel_times=dict(session.travel_times),
                    completed=bool(session.done), deadline_seconds=self.deadline)
        return rows, windows, metrics


class GlobalTrafficCritic(torch.nn.Module):
    """Value of the real full-network traffic state for the shared window reward."""
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
                         entropy_coef=0.002, grad_clip=10.0):
    """Actor-only update with externally supplied per-row advantages.

    Scale shared advantages without centering, then add each decision vehicle's
    own timeout return before clipping. Timeout costs never affect shared scaling.
    """
    if not rows:
        return dict(actor_loss=0., actor_gradient_norm=0., advantage_std=0.)
    actor_parameters = list(actor_parameters)
    device = actor_parameters[0].device
    advantage = torch.as_tensor(np.asarray([float(a) for a in advantages], dtype=np.float32), device=device)
    if advantage.shape != (len(rows),) or not bool(torch.isfinite(advantage).all()):
        raise ValueError('advantages must be finite with one value per row')
    timeout_returns = torch.tensor([row.get('timeout_return', 0.) for row in rows],
                                   dtype=advantage.dtype, device=device)
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
        input_hashes=prepared.input_hashes, config_fingerprint=config.semantic_fingerprint)
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
        temporary_path = Path(stream.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path, *, method_name, prepared, config, device='cpu'):
    state = torch.load(path, map_location=device, weights_only=False)
    if (state.get('format') != CHECKPOINT_FORMAT or state.get('method') != method_name
            or state.get('input_hashes') != prepared.input_hashes
            or state.get('config_fingerprint') != config.semantic_fingerprint):
        raise ValueError('experiment checkpoint mismatch')
    return state
