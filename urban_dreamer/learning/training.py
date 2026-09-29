from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..workflow.config import MainlineConfig, NONARRIVAL_PENALTY
from ..traffic.environment import NextHopDecision, SumoSession, safe_microstep_seconds
from ..traffic.routing import PreparedInputs
from ..traffic.topology import NextHopCandidates, TrafficTopology
from ..models.policy import (CentralizedLatentValueCritic, CentralizedVehicleValueCritic,
                             NextHopActor, lambda_return, lambda_returns_by_vehicle_ids)
from ..models.world_model import (FLOW_LATENT_DIM, UrbanDreamerWorldModel, cfm_loss,
    cfm_predict_one_timestep, decoder_bpr_supervision_loss, decoder_reconstruction_loss,
    episode_t_all_supervision_loss, transition_decoded_supervision_loss)
from .observations import build_actor_observations, build_vehicle_observations
from .imagination import advance_imagined_window
from .replay import MacroTransition, ReplayBuffer

WORLD_MODEL_INTERVAL_SECONDS = 30
ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE = 4
IMAGINED_ROLLOUTS_PER_PASS = 8
IMAGINED_PASSES_PER_EPISODE = 16
IMAGINED_HORIZON = 8
ACTOR_ENTROPY_COEF = 0.002
GRAD_CLIP = 10.0


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows if key in row]))
            for key in {key for row in rows for key in row}}


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, (list, tuple)) else value


def _close_event(row: dict[str, Any], *, end_time: float,
                 next_obs: torch.Tensor, terminal: bool, timed_out: bool = False) -> None:
    elapsed = float(end_time) - row['time']
    if elapsed < -1e-8:
        raise ValueError('event reward ends before the action')
    if timed_out and not terminal:
        raise ValueError('episode timeout must terminate the value target')
    row.update(reward=-max(elapsed, 0.) / WORLD_MODEL_INTERVAL_SECONDS
               + (NONARRIVAL_PENALTY if timed_out else 0.),
               next_obs=next_obs, terminal=terminal, timed_out=timed_out)


@dataclass(frozen=True)
class CollectedEpisode:
    transitions: tuple[MacroTransition, ...]
    total_travel_minutes: float
    metrics: dict[str, float]


class MainlineTrainer:
    """Collect physical event windows, then train macro dynamics and event policies."""
    def __init__(self, *, config: MainlineConfig, prepared_inputs: PreparedInputs,
                 topology: TrafficTopology, _session_factory: Callable[..., Any] | None = None) -> None:
        self.config, self.prepared_inputs, self.topology = config, prepared_inputs, topology
        self.device = torch.device(config.run.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('configured CUDA device is unavailable')
        random.seed(config.run.seed)
        np.random.seed(config.run.seed)
        torch.manual_seed(config.run.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.run.seed)
        self.rng = np.random.default_rng(config.run.seed)
        self.network = topology.network
        self.edge_count = self.network.edge_count
        self.total_vehicle_count = len(topology.vehicles)
        self.candidates = NextHopCandidates(self.network)
        lengths = list(self.network.external_lane_lengths_m.values()) or self.network.lengths_m[self.network.passenger_allowed]
        self.microstep_seconds = safe_microstep_seconds(lengths, config.simulation.vehicle_max_speed_mps)
        self._session_factory = _session_factory or SumoSession
        self._next_environment_episode = 0
        self.world_model = UrbanDreamerWorldModel(
            adjacency=self._tensor(self.network.adjacency),
            static_features=self._tensor(self.network.static_features)).to(self.device)
        self.actor = NextHopActor().to(self.device)
        self.vehicle_critic = CentralizedVehicleValueCritic().to(self.device)
        self.latent_critic = CentralizedLatentValueCritic().to(self.device)
        self._world_parameters = (list(self.world_model.flow_projector.parameters())
                                  + list(self.world_model.dynamics.parameters()))
        self.optimizers = {
            'world_model_initial': torch.optim.AdamW(self._world_parameters, lr=1e-4, weight_decay=1e-2),
            'world_model_online': torch.optim.Adam(self._world_parameters, lr=1e-5),
            'active_fraction': torch.optim.Adam(self.world_model.active_vehicle_fraction_readout.parameters(), lr=1e-3),
            'actor': torch.optim.Adam(self.actor.parameters(), lr=2e-4),
            'vehicle_critic': torch.optim.Adam(self.vehicle_critic.parameters(), lr=3e-4),
            'latent_critic': torch.optim.Adam(self.latent_critic.parameters(), lr=3e-4),
        }
        self.warmup_replay = ReplayBuffer(sequence_length=config.resources.world_model_initialization_sequence_length, max_episodes=None, active_only=True)
        self.online_replay = ReplayBuffer(sequence_length=1, active_only=True)
        self.metrics = {key: 0. for key in (
            'warmup_episodes_completed', 'world_model_initialization_steps_completed',
            'online_episodes_completed', 'evaluation_episodes_completed',
            'evaluation_t_all_sum', 'evaluation_t_all_mean', 'world_model_loss_last',
            'active_fraction_loss_last', 'actor_loss_last', 'vehicle_critic_loss_last',
            'latent_critic_loss_last', 't_all_last')}

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _context(self, states: list[dict[str, Any]] | tuple, timestamp: float, *, mode: str = 'online') -> np.ndarray:
        active = sum(bool(s['departed']) and not bool(s['arrived']) for s in states)
        horizon = self._episode_end_time(mode)
        return np.asarray([min(timestamp / horizon, 1.), active / self.total_vehicle_count], dtype=np.float32)

    def _policy_features(self, flow: np.ndarray) -> np.ndarray:
        # Volume and speed are decoded fields in both physical and imagined windows.
        return np.asarray(flow[:, [2, 1]], dtype=np.float32)

    def _observations(self, states, flow, context, timestamp):
        kwargs = dict(policy_edge_features=self._policy_features(flow), vehicle_states=states,
                      candidates=self.candidates, episode_progress=float(context[0]),
                      active_vehicle_fraction=float(context[1]))
        return (build_actor_observations(**kwargs),
                self._tensor(build_vehicle_observations(**kwargs, current_time=timestamp)))

    def _vehicle_observations(self, states, flow, context, timestamp):
        return self._tensor(build_vehicle_observations(
            policy_edge_features=self._policy_features(flow), vehicle_states=states,
            candidates=self.candidates, episode_progress=float(context[0]),
            active_vehicle_fraction=float(context[1]), current_time=timestamp))

    def _select(self, states, flow, timestamp, *, context, mode):
        states = sorted(states, key=lambda s: str(s['vehicle_id']))
        if mode == 'warmup':
            decisions = []
            for state in states:
                current, dest = state['current_edge'], state['destination_edge']
                choices = self.candidates.candidates(current, dest)
                costs = np.asarray([self.network.free_flow_travel_seconds[e]
                                    + self.candidates.costs(dest)[e]
                                    + self.candidates.junction(current, e)[0] / self.candidates.junction(current, e)[1]
                                    for e in choices])
                weights = np.exp(-(costs - costs.min()) / 30.)
                chosen = int(self.rng.choice(choices, p=weights / weights.sum()))
                decisions.append(NextHopDecision(str(state['vehicle_id']), current, chosen, timestamp))
            return decisions
        observations, _ = self._observations(states, flow, context, timestamp)
        with torch.no_grad():
            selection = self.actor.sample_action(self._tensor(observations.features),
                torch.as_tensor(observations.mask, device=self.device), deterministic=mode == 'evaluation')
        indices = selection.candidate_index.cpu().tolist()
        return [NextHopDecision(str(state['vehicle_id']), int(state['current_edge']),
            int(observations.edge_indices[i, indices[i]]), timestamp) for i, state in enumerate(states)]

    def _episode_step_limit(self, mode):
        simulation = self.config.simulation
        if mode == 'warmup':
            seconds = simulation.warmup_horizon_seconds + simulation.warmup_completion_drain_seconds
        elif mode == 'evaluation':
            seconds = simulation.evaluation_horizon_seconds
        else:
            seconds = (simulation.world_model_intervals_per_episode * WORLD_MODEL_INTERVAL_SECONDS
                       + simulation.training_completion_buffer_seconds)
        return math.ceil(seconds / WORLD_MODEL_INTERVAL_SECONDS)

    def _episode_end_time(self, mode: str = 'online') -> float:
        # Real collection uses complete 30-second windows, including the drain buffer.
        return float(self._episode_step_limit(mode) * WORLD_MODEL_INTERVAL_SECONDS)

    def _run_episode(self, *, mode: str, episode_index: int) -> CollectedEpisode:
        if mode not in {'warmup', 'online', 'evaluation'}:
            raise ValueError(f'unsupported collection mode {mode!r}')
        session = self._session_factory(config=self.config, prepared_inputs=self.prepared_inputs,
            topology=self.topology, episode_index=self._next_environment_episode)
        started = time.monotonic()
        transitions = []
        events = 0
        exposure = 0.
        try:
            state = session.reset()
            for step in range(self._episode_step_limit(mode)):
                if time.monotonic() - started > self.config.simulation.episode_wall_timeout_seconds:
                    raise TimeoutError(f'{mode} episode exceeded episode_wall_timeout_seconds')
                if session.done:
                    break
                flow = np.asarray(state['flow'], dtype=np.float32)
                vehicles = copy.deepcopy(tuple(vehicle for vehicle in state['vehicles']
                    if vehicle['departed'] and not vehicle['arrived']))
                timestamp = float(state['time'])
                context = self._context(vehicles, timestamp, mode=mode)
                window = session.advance_window(lambda pending, frozen, now:
                    self._select(pending, frozen, now, context=context, mode=mode))
                state = window.next_state
                exposure += window.active_vehicle_seconds
                events += len(window.decisions)
                transitions.append(MacroTransition(flow=flow, action=window.action_counts / self.total_vehicle_count,
                    departure=window.departure_counts / self.total_vehicle_count, context=context, next_flow=state['flow'],
                    active_vehicle_fraction=float(np.clip(window.active_vehicle_seconds / (self.total_vehicle_count * 30.), 0., 1.)),
                    vehicles=vehicles, time=timestamp, terminal=session.done,
                    episode=episode_index, step_index=step))
                if session.done:
                    break
            complete = session.done
            if not transitions:
                raise RuntimeError('episode produced no macro transitions')
            durations = session.travel_times
            if complete and len(durations) != self.total_vehicle_count:
                raise RuntimeError('episode completed without all vehicle travel times')
            # Incomplete online trajectories report actual observed exposure, never fabricated arrivals.
            travel_minutes = (sum(durations.values()) if complete else exposure) / self.total_vehicle_count / 60.
            unfinished = self.total_vehicle_count - len(durations)
            penalty = NONARRIVAL_PENALTY * unfinished
            metrics = {f'{mode}_nonarrival_count': float(unfinished),
                f'{mode}_nonarrival_penalty_sum': penalty,
                f'{mode}_local_reward_sum': -exposure / WORLD_MODEL_INTERVAL_SECONDS + penalty,
                f'{mode}_time_limit_seconds': self._episode_end_time(mode),
                f'{mode}_completion_rate': len(durations) / self.total_vehicle_count,
                f'{mode}_steps': float(len(transitions)), f'{mode}_events': float(events),
                f'{mode}_wall_seconds': time.monotonic() - started,
                f'{mode}_microstep_seconds': float(session.microstep_seconds),
                f'{mode}_microsteps': float(session.metrics["microsteps"]),
                f'{mode}_t_all': travel_minutes}
            self._next_environment_episode += 1
            return CollectedEpisode(tuple(transitions), travel_minutes, metrics)
        finally:
            session.close()

    def collect_warmup_episode(self, episode_index):
        collected = self._run_episode(mode='warmup', episode_index=episode_index)
        self.warmup_replay.append_episode(collected.transitions, total_travel_minutes=collected.total_travel_minutes)
        self.metrics['warmup_episodes_completed'] += 1
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(collected.metrics)
        return {'complete': True, 'transitions': len(collected.transitions), **collected.metrics}

    def _world_loss(self, batch):
        flow = batch.flow.to(self.device)
        batch_size, frames = flow.shape[:2]
        z = self.world_model.encode(flow.reshape(-1, self.edge_count, 8)).reshape(
            batch_size, frames, self.edge_count, FLOW_LATENT_DIM)
        action, context = (x.to(self.device) for x in (batch.action, batch.context))
        loss, _ = cfm_loss(self.world_model.dynamics, z_sequence=z,
                           action=action, context=context)
        decoded = self.world_model.decode(z)
        # decode(z_t).speed describes [t, t+30], while encoder/dynamics retain
        # boundary observations. The last frame has no following speed label.
        decoder_targets = flow.clone()
        decoder_targets[:, :-1, :, 1] = (flow[:, :-1, :, 1] + flow[:, 1:, :, 1]) / 2.
        speed_target_mask = torch.ones(flow.shape[:-1], dtype=torch.bool, device=self.device)
        speed_target_mask[:, -1] = False
        reconstruction, _ = decoder_reconstruction_loss(decoded_dyn=decoded,
            flow_targets=decoder_targets, speed_target_mask=speed_target_mask)
        bpr, _ = decoder_bpr_supervision_loss(decoded_dyn=decoded, flow_targets=flow,
            lengths=self._tensor(self.network.lengths_m), speed_limits=self._tensor(self.network.speed_limits_mps),
            capacities=self._tensor(self.network.capacities))
        predicted = z[:, :-1] + self.world_model.dynamics(action, context=context,
            packed_latents=z[:, :-1], flow_time=torch.zeros(action.shape[:2], device=self.device)).latent_velocity
        transition, _ = transition_decoded_supervision_loss(z_pred_next=predicted,
            flow_projector=self.world_model.flow_projector, flow_target_next=decoder_targets[:, 1:],
            speed_target_mask=speed_target_mask[:, 1:])
        loss = loss + reconstruction + bpr + transition
        targets = batch.total_travel_minutes.to(self.device)
        valid = (targets > 0) & batch.episode_complete.to(self.device)
        if bool(valid.any()):
            total_loss, _ = episode_t_all_supervision_loss(flow_projector=self.world_model.flow_projector,
                z_tokens=z[valid], reward_targets=-targets[valid])
            loss = loss + total_loss
        with torch.no_grad():
            raw_fraction = self.world_model.decode(predicted.detach())[..., 2].mean(dim=-1)
        fraction = self.world_model.active_vehicle_fraction_readout(predicted.detach(),
            z_prev=z[:, :-1].detach(), raw_decoder_fraction=raw_fraction, transition_seconds=30.)
        readout_loss = F.mse_loss(fraction, batch.active_vehicle_fraction.to(self.device))
        return loss, readout_loss

    def _update_world(self, replay, *, initial, accumulation=1):
        if not replay.has_samples:
            return {'world_model_loss': 0., 'active_fraction_loss': 0., 'world_model_update_skipped': 1.}
        self.world_model.train()
        if initial:
            self.world_model.set_initial_chunk_profile()
        else:
            self.world_model.set_online_chunk_profile()
        optimizer = self.optimizers['world_model_initial' if initial else 'world_model_online']
        readout_optimizer = self.optimizers['active_fraction']
        optimizer.zero_grad(set_to_none=True)
        readout_optimizer.zero_grad(set_to_none=True)
        losses = []
        size = (self.config.resources.world_model_initialization_batch_size if initial
                else self.config.resources.online_world_model_batch_size)
        for _ in range(accumulation):
            batch = replay.sample_batch(batch_size=size, rng=self.rng)
            loss, readout_loss = self._world_loss(batch)
            (loss / accumulation).backward()
            (readout_loss / accumulation).backward()
            losses.append({'world_model_loss': float(loss.detach()), 'active_fraction_loss': float(readout_loss.detach())})
        torch.nn.utils.clip_grad_norm_(self._world_parameters, GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(self.world_model.active_vehicle_fraction_readout.parameters(), GRAD_CLIP)
        optimizer.step()
        readout_optimizer.step()
        self.world_model.active_vehicle_fraction_readout.calibration_updates.add_(1)
        metrics = _mean_metrics(losses)
        self.metrics.update({key + '_last': value for key, value in metrics.items()})
        return metrics

    def initialize_world_model_step(self, step_index):
        accumulation = self.config.resources.world_model_gradient_accumulation_steps
        remaining = self.config.run.world_model_initialization_steps - step_index * accumulation
        if remaining <= 0:
            raise IndexError('world-model initialization exceeds configured budget')
        accumulation = min(accumulation, remaining)
        metrics = self._update_world(self.warmup_replay, initial=True, accumulation=accumulation)
        self.metrics['world_model_initialization_steps_completed'] += accumulation
        return {'microbatches': accumulation, **metrics}

    def train_online_episode(self, episode_index):
        collected = self._run_episode(mode='online', episode_index=episode_index)
        self.online_replay.append_episode(collected.transitions, total_travel_minutes=collected.total_travel_minutes)
        metrics = _mean_metrics([self._update_world(self.online_replay, initial=False)
                                for _ in range(ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE)])
        metrics.update(_mean_metrics([self._train_imagination_pass() for _ in range(IMAGINED_PASSES_PER_EPISODE)]))
        metrics.update(collected.metrics)
        self.metrics['online_episodes_completed'] += 1
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(metrics)
        return {'complete': True, **metrics}

    def evaluate_episode(self, episode_index):
        collected = self._run_episode(mode='evaluation', episode_index=episode_index)
        self.metrics['evaluation_episodes_completed'] += 1
        self.metrics['evaluation_t_all_sum'] += collected.total_travel_minutes
        self.metrics['evaluation_t_all_mean'] = (self.metrics['evaluation_t_all_sum']
                                               / self.metrics['evaluation_episodes_completed'])
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(collected.metrics)
        return {'complete': True, **collected.metrics}

    def _imagine_from_seed(self, batch, batch_index, *, rollout_index, critic_rows=None):
        states = copy.deepcopy([state for state in batch.vehicles[batch_index]
                                if state['departed'] and not state['arrived']])
        timestamp = float(batch.time[batch_index])
        deadline = self._episode_end_time()
        if not states or timestamp >= deadline:
            raise ValueError('imagination seed needs active vehicles before the episode deadline')
        with torch.no_grad():
            latent = self.world_model.encode(batch.flow[batch_index, 0].unsqueeze(0).to(self.device))
        rows, pending = [], {}
        values, next_values, rewards, terminals = [], [], [], []
        identity = {str(s['vehicle_id']): rollout_index * self.total_vehicle_count + i for i, s in enumerate(states)}
        for window_index in range(min(IMAGINED_HORIZON, math.ceil((deadline - timestamp) / WORLD_MODEL_INTERVAL_SECONDS))):
            context = self._context(states, timestamp)
            with torch.no_grad():
                decoded = self.world_model.decode(latent)[0].cpu().numpy()
            value = self.latent_critic(latent.detach(), self._tensor(context).unsqueeze(0)).reshape(())

            # In-transit features need value supervision even without a new action.
            # These intervals are kept separate from actor event returns.
            anchors = {}
            if critic_rows is not None:
                moving = [state for state in states if state['departed']
                          and not state['arrived'] and not state['pending_decision']]
                anchor_obs = self._vehicle_observations(moving, decoded, context, timestamp)
                anchors = {str(state['vehicle_id']): {'obs': anchor_obs[i], 'time': timestamp}
                           for i, state in enumerate(moving)}

            def finish_anchor(vehicle_id, end_time, next_obs, terminal, timed_out=False):
                anchor = anchors.pop(vehicle_id, None)
                if anchor is not None:
                    _close_event(anchor, end_time=end_time, next_obs=next_obs, terminal=terminal, timed_out=timed_out)
                    critic_rows.append(anchor)

            def decide(events, frozen_flow, now):
                observations, critic_obs = self._observations(events, frozen_flow, context, now)
                selection = self.actor.sample_action(self._tensor(observations.features),
                    torch.as_tensor(observations.mask, device=self.device))
                decisions = []
                for i, state in enumerate(events):
                    vehicle_id = str(state['vehicle_id'])
                    finish_anchor(vehicle_id, now, critic_obs[i], False)
                    if vehicle_id in pending:
                        _close_event(pending.pop(vehicle_id), end_time=now, next_obs=critic_obs[i], terminal=False)
                    decision = NextHopDecision(vehicle_id, int(state['current_edge']),
                        int(observations.edge_indices[i, int(selection.candidate_index[i])]), float(now))
                    row = dict(obs=critic_obs[i], log_prob=selection.log_probability[i],
                        entropy=selection.entropy[i], time=float(now), vehicle_id=vehicle_id,
                        trajectory_vehicle_key=identity[vehicle_id], window=window_index)
                    rows.append(row)
                    pending[vehicle_id] = row
                    decisions.append(decision)
                return decisions

            speeds = np.clip(decoded[:, 1], 0.05, 1.5) * self.network.speed_limits_mps
            speeds = np.minimum(speeds, self.config.simulation.vehicle_max_speed_mps)
            result = advance_imagined_window(states, candidates=self.candidates,
                edge_speeds_mps=speeds, frozen_flow=decoded, start_time=timestamp,
                total_vehicle_count=self.total_vehicle_count, decide=decide,
                observation_step_seconds=self.microstep_seconds,
                vehicle_max_speed_mps=self.config.simulation.vehicle_max_speed_mps)
            for vehicle_id, arrived_at in result.arrivals.items():
                if vehicle_id in anchors:
                    finish_anchor(vehicle_id, arrived_at, torch.zeros_like(anchors[vehicle_id]['obs']), True)
                if vehicle_id in pending:
                    row = pending.pop(vehicle_id)
                    _close_event(row, end_time=arrived_at, next_obs=torch.zeros_like(row['obs']), terminal=True)
            with torch.no_grad():
                next_latent = cfm_predict_one_timestep(self.world_model.dynamics,
                    past_latents=latent.unsqueeze(1), action=self._tensor(result.action).view(1, 1, -1),
                    context=self._tensor(context).view(1, 1, 2)).squeeze(1)
                next_decoded = self.world_model.decode(next_latent)
                fraction = self.world_model.active_vehicle_fraction_readout(next_latent, z_prev=latent,
                    raw_decoder_fraction=next_decoded[..., 2].mean(dim=-1), transition_seconds=30.).reshape(())
                reward = -fraction * (WORLD_MODEL_INTERVAL_SECONDS / 60.)
            timestamp += WORLD_MODEL_INTERVAL_SECONDS
            timed_out = timestamp >= deadline
            next_context = self._context(states, timestamp)
            if anchors:
                end_states = [state for state in states if str(state['vehicle_id']) in anchors]
                end_obs = self._vehicle_observations(end_states, next_decoded[0].cpu().numpy(), next_context, timestamp)
                for i, state in enumerate(end_states):
                    finish_anchor(str(state['vehicle_id']), timestamp, end_obs[i], timed_out, timed_out=timed_out)
            terminal = timed_out or all(bool(state['arrived']) for state in states)
            with torch.no_grad():
                next_value = self.latent_critic(next_latent, self._tensor(next_context).unsqueeze(0)).reshape(())
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            terminals.append(terminal)
            latent = next_latent
            if terminal:
                break
        with torch.no_grad():
            final_flow = self.world_model.decode(latent)[0].cpu().numpy()
            final_obs = self._vehicle_observations(states, final_flow, self._context(states, timestamp), timestamp)
        for i, state in enumerate(states):
            row = pending.get(str(state['vehicle_id']))
            if row is not None:
                _close_event(row, end_time=timestamp, next_obs=final_obs[i],
                             terminal=timestamp >= deadline, timed_out=timestamp >= deadline)
        targets = lambda_return(torch.stack(rewards), torch.stack(next_values),
                                terminal=torch.tensor(terminals, device=self.device))
        values_tensor = torch.stack(values)
        advantage = (targets - values_tensor).detach()
        for row in rows:
            row['global_advantage'] = advantage[row['window']]
            row['global_reward'] = rewards[row['window']]
        return rows, F.mse_loss(values_tensor, targets.detach())

    def _train_imagination_pass(self):
        if not self.online_replay.has_active_seeds:
            return {'latent_critic_loss': 0., 'imagined_events': 0., 'actor_loss': 0.,
                    'vehicle_critic_loss': 0., 'critic_only_intervals': 0.}
        self.world_model.eval()
        self.world_model.set_online_chunk_profile()
        rows, macro_losses, critic_rows = [], [], []
        for rollout in range(IMAGINED_ROLLOUTS_PER_PASS):
            batch = self.online_replay.sample_active_seeds(batch_size=1, rng=self.rng)
            events, loss = self._imagine_from_seed(batch, 0, rollout_index=rollout, critic_rows=critic_rows)
            rows.extend(events)
            macro_losses.append(loss)
        macro_loss = torch.stack(macro_losses).mean()
        # Macro learning is independent of whether this window contained policy events.
        self.optimizers['latent_critic'].zero_grad(set_to_none=True)
        macro_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latent_critic.parameters(), GRAD_CLIP)
        self.optimizers['latent_critic'].step()
        metrics = {'latent_critic_loss': float(macro_loss.detach()), 'imagined_events': float(len(rows)),
                   'actor_loss': 0., 'vehicle_critic_loss': 0., 'critic_only_intervals': float(len(critic_rows))}
        vehicle_losses = []
        if rows:
            current_values = self.vehicle_critic(torch.stack([row['obs'] for row in rows])).squeeze(-1)
            with torch.no_grad():
                next_values = self.vehicle_critic(torch.stack([row['next_obs'] for row in rows])).squeeze(-1)
                returns = lambda_returns_by_vehicle_ids(self._tensor([row['reward'] for row in rows]), next_values,
                    torch.ones(len(rows), dtype=torch.bool, device=self.device),
                    torch.tensor([row['trajectory_vehicle_key'] for row in rows], device=self.device),
                    terminal=torch.tensor([row['terminal'] for row in rows], device=self.device))
            combined = returns - current_values + torch.stack([row['global_advantage'] for row in rows])
            combined = ((combined - combined.mean()) / combined.std(unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
            actor_loss = -(torch.stack([row['log_prob'] for row in rows]) * combined.detach()
                           + ACTOR_ENTROPY_COEF * torch.stack([row['entropy'] for row in rows])).mean()
            vehicle_losses.append(F.mse_loss(current_values, returns))
            self.optimizers['actor'].zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.optimizers['actor'].step()
            metrics['actor_loss'] = float(actor_loss.detach())
        if critic_rows:
            anchor_values = self.vehicle_critic(torch.stack([row['obs'] for row in critic_rows])).squeeze(-1)
            with torch.no_grad():
                bootstrap = self.vehicle_critic(torch.stack([row['next_obs'] for row in critic_rows])).squeeze(-1)
                terminal = torch.tensor([row['terminal'] for row in critic_rows], device=self.device)
                targets = self._tensor([row['reward'] for row in critic_rows]) + bootstrap.masked_fill(terminal, 0.)
            vehicle_losses.append(F.mse_loss(anchor_values, targets))
        if vehicle_losses:
            vehicle_loss = torch.stack(vehicle_losses).mean()
            self.optimizers['vehicle_critic'].zero_grad(set_to_none=True)
            vehicle_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vehicle_critic.parameters(), GRAD_CLIP)
            self.optimizers['vehicle_critic'].step()
            metrics['vehicle_critic_loss'] = float(vehicle_loss.detach())
        self.metrics.update({key + '_last': value for key, value in metrics.items() if key.endswith('_loss')})
        return metrics

    def state_dict(self):
        numpy_state = np.random.get_state()
        return {'models': {name: getattr(self, name).state_dict() for name in
                    ('world_model', 'actor', 'vehicle_critic', 'latent_critic')},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            'replay': {'warmup': self.warmup_replay.state_dict(), 'online': self.online_replay.state_dict()},
            'rng': {'python_random': random.getstate(), 'numpy_global': list(numpy_state),
                'numpy_generator': copy.deepcopy(self.rng.bit_generator.state), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            'metrics': dict(self.metrics), 'next_environment_episode': self._next_environment_episode}

    def load_state_dict(self, state):
        if set(state) != {'models', 'optimizers', 'replay', 'rng', 'metrics', 'next_environment_episode'}:
            raise ValueError('MainlineTrainer checkpoint state has an invalid schema')
        for name in ('world_model', 'actor', 'vehicle_critic', 'latent_critic'):
            getattr(self, name).load_state_dict(state['models'][name], strict=True)
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][name])
        self.warmup_replay.load_state_dict(state['replay']['warmup'])
        self.online_replay.load_state_dict(state['replay']['online'])
        self._next_environment_episode = int(state['next_environment_episode'])
        self.metrics = dict(state['metrics'])
        rng = state['rng']
        random.setstate(_tuple_tree(rng['python_random']))
        numpy_state = list(rng['numpy_global'])
        numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
        np.random.set_state(tuple(numpy_state))
        self.rng.bit_generator.state = copy.deepcopy(rng['numpy_generator'])
        torch.set_rng_state(torch.as_tensor(rng['torch_cpu'], dtype=torch.uint8, device='cpu'))
        if torch.cuda.is_available() and rng['torch_cuda']:
            torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8, device='cpu') for s in rng['torch_cuda']])

    def summary(self):
        return {**self.metrics, 'warmup_replay_episodes': len(self.warmup_replay.episodes),
            'online_replay_episodes': len(self.online_replay.episodes),
            'next_environment_episode': self._next_environment_episode}

    def close(self):
        # Sessions are scoped to _run_episode and released even when collection fails.
        pass
