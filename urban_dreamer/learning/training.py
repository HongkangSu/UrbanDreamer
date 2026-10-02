from __future__ import annotations

import copy
import json
import math
import random
import time
from itertools import islice
from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..workflow.config import MainlineConfig, NONARRIVAL_PENALTY
from ..traffic.environment import NextHopDecision, SumoSession, safe_microstep_seconds
from ..traffic.demand import PoissonDemand, demand_identity
from ..traffic.routing import PreparedInputs
from ..traffic.topology import NextHopCandidates, TrafficTopology
from ..models.policy import (CentralizedLatentValueCritic, NextHopActor, lambda_return,
    attach_timeout_penalties, individual_actor_advantage)
from ..models.world_model import (FLOW_LATENT_DIM, UrbanDreamerWorldModel, recursive_fm_loss,
    decoder_bpr_supervision_loss, decoder_reconstruction_loss,
    episode_t_all_supervision_loss)
from .observations import build_actor_observations
from .imagination import advance_imagined_window
from .replay import MacroTransition, ReplayBuffer

WORLD_MODEL_INTERVAL_SECONDS = 30
ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE = 4
IMAGINED_ROLLOUTS_PER_PASS = 8
IMAGINED_PASSES_PER_EPISODE = 16
IMAGINED_HORIZON = 8
ACTOR_ENTROPY_COEF = 0.002
GRAD_CLIP = 10.0
REWARD_CONTRACT = 'global_physical_individual_timeout_v1'


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows if key in row]))
            for key in {key for row in rows for key in row}}


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True)
class CollectedEpisode:
    transitions: tuple[MacroTransition, ...]
    total_travel_minutes: float
    metrics: dict[str, float]
    total_vehicle_count: int


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
        self.demand = PoissonDemand.from_vehicles(topology.vehicles,
            duration_seconds=config.simulation.demand_duration_seconds)
        # Reference input count; physical and replay trajectories carry realized N.
        self.total_vehicle_count = len(topology.vehicles)
        self.candidates = NextHopCandidates(self.network)
        lengths = list(self.network.external_lane_lengths_m.values()) or self.network.lengths_m[self.network.passenger_allowed]
        self.microstep_seconds = safe_microstep_seconds(lengths, config.simulation.vehicle_max_speed_mps)
        self._session_factory = _session_factory or SumoSession
        self._next_environment_episode = 0
        self.world_model = UrbanDreamerWorldModel(
            adjacency=self._tensor(self.network.adjacency),
            static_features=self._tensor(self.network.static_features),
            low_frequency_rank=config.method.low_frequency_rank,
            prior_noise_scale=config.method.prior_noise_scale).to(self.device)
        self.actor = NextHopActor().to(self.device)
        self.latent_critic = CentralizedLatentValueCritic().to(self.device)
        self._world_parameters = (list(self.world_model.flow_projector.parameters())
                                  + list(self.world_model.dynamics.parameters()))
        self.optimizers = {
            'world_model_initial': torch.optim.AdamW(self._world_parameters, lr=1e-4, weight_decay=1e-2),
            'world_model_online': torch.optim.Adam(self._world_parameters, lr=1e-5),
            'active_fraction': torch.optim.Adam(self.world_model.active_vehicle_fraction_readout.parameters(), lr=1e-3),
            'actor': torch.optim.Adam(self.actor.parameters(), lr=2e-4),
            'latent_critic': torch.optim.Adam(self.latent_critic.parameters(), lr=3e-4),
        }
        self.warmup_replay = ReplayBuffer(sequence_length=config.resources.world_model_initialization_sequence_length, max_episodes=None)
        self.online_replay = ReplayBuffer(sequence_length=config.method.supervision_horizon)
        self.metrics = {key: 0. for key in (
            'warmup_episodes_completed', 'world_model_initialization_epochs_completed',
            'world_model_initialization_batches_completed', 'world_model_initialization_samples_seen',
            'world_model_initialization_updates',
            'online_episodes_completed', 'evaluation_episodes_completed',
            'evaluation_t_all_sum', 'evaluation_t_all_mean', 'evaluation_vehicle_minutes_sum',
            'evaluation_vehicle_count_sum', 'evaluation_t_all_vehicle_weighted',
            'world_model_updates', 'world_model_loss_last',
            'active_fraction_loss_last', 'actor_loss_last',
            'latent_critic_loss_last', 't_all_last')}

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _context(self, states: list[dict[str, Any]] | tuple, timestamp: float, *,
                 mode: str = 'online', total_vehicle_count: int | None = None) -> np.ndarray:
        active = sum(bool(s['departed']) and not bool(s['arrived']) for s in states)
        horizon = self._episode_end_time(mode)
        population = self.total_vehicle_count if total_vehicle_count is None else total_vehicle_count
        return np.asarray([min(timestamp / horizon, 1.), active / max(population, 1)], dtype=np.float32)

    def _policy_features(self, flow: np.ndarray) -> np.ndarray:
        # Volume and speed are decoded fields in both physical and imagined windows.
        return np.asarray(flow[:, [2, 1]], dtype=np.float32)

    def _observations(self, states, flow, context, timestamp):
        kwargs = dict(policy_edge_features=self._policy_features(flow), vehicle_states=states,
                      candidates=self.candidates, episode_progress=float(context[0]),
                      active_vehicle_fraction=float(context[1]))
        return build_actor_observations(**kwargs)

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
        observations = self._observations(states, flow, context, timestamp)
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
        evaluation = mode == 'evaluation'
        demand_seed = self.config.method.evaluation_demand_seed if evaluation else self.config.run.seed
        demand_index = episode_index if evaluation else self._next_environment_episode
        episode_topology = replace(self.topology, vehicles=self.demand.sample(
            seed=demand_seed, episode_index=demand_index))
        if evaluation:
            self._record_evaluation_demand(episode_topology.vehicles, episode_index)
        total_vehicle_count = len(episode_topology.vehicles)
        denominator = max(total_vehicle_count, 1)
        session_config = replace(self.config, run=replace(self.config.run, seed=demand_seed)) if evaluation else self.config
        session = self._session_factory(config=session_config, prepared_inputs=self.prepared_inputs,
            topology=episode_topology, episode_index=demand_index)
        started = time.monotonic()
        transitions = []
        events = 0
        exposure = 0.
        try:
            state = session.reset()
            for step in range(self._episode_step_limit(mode)):
                if time.monotonic() - started > self.config.simulation.episode_wall_timeout_seconds:
                    raise TimeoutError(f'{mode} episode exceeded episode_wall_timeout_seconds')
                if session.done and transitions:
                    break
                flow = np.asarray(state['flow'], dtype=np.float32)
                vehicles = copy.deepcopy(tuple(vehicle for vehicle in state['vehicles']
                    if vehicle['departed'] and not vehicle['arrived']))
                timestamp = float(state['time'])
                context = self._context(vehicles, timestamp, mode=mode,
                                        total_vehicle_count=total_vehicle_count)
                window = session.advance_window(lambda pending, frozen, now:
                    self._select(pending, frozen, now, context=context, mode=mode))
                state = window.next_state
                exposure += window.active_vehicle_seconds
                events += len(window.decisions)
                transitions.append(MacroTransition(flow=flow, action=window.action_counts / denominator,
                    departure=window.departure_counts / denominator, context=context, next_flow=state['flow'],
                    active_vehicle_fraction=float(np.clip(window.active_vehicle_seconds / (denominator * 30.), 0., 1.)),
                    vehicles=vehicles, time=timestamp, terminal=session.done,
                    episode=episode_index, step_index=step))
                if session.done:
                    break
            complete = session.done
            if not transitions:
                raise RuntimeError('episode produced no macro transitions')
            durations = session.travel_times
            if complete and len(durations) != total_vehicle_count:
                raise RuntimeError('episode completed without all vehicle travel times')
            # Incomplete online trajectories report actual observed exposure, never fabricated arrivals.
            travel_minutes = (sum(durations.values()) if complete else exposure) / denominator / 60.
            unfinished = total_vehicle_count - len(durations)
            penalty = NONARRIVAL_PENALTY * unfinished
            metrics = {f'{mode}_nonarrival_count': float(unfinished),
                f'{mode}_nonarrival_penalty_sum': penalty,
                f'{mode}_vehicle_timeout_penalty_sum': penalty,
                f'{mode}_global_reward_sum': -exposure / (denominator * 60.),
                f'{mode}_selection_score': -exposure / (denominator * 60.) + penalty,
                f'{mode}_vehicle_count': float(total_vehicle_count),
                f'{mode}_vehicle_minutes': travel_minutes * total_vehicle_count,
                f'{mode}_demand_seed': float(demand_seed),
                f'{mode}_demand_index': float(demand_index),
                f'{mode}_demand_duration_seconds': float(self.config.simulation.demand_duration_seconds),
                f'{mode}_time_limit_seconds': self._episode_end_time(mode),
                f'{mode}_completion_rate': len(durations) / total_vehicle_count if total_vehicle_count else 1.,
                f'{mode}_steps': float(len(transitions)), f'{mode}_events': float(events),
                f'{mode}_wall_seconds': time.monotonic() - started,
                f'{mode}_microstep_seconds': float(session.microstep_seconds),
                f'{mode}_microsteps': float(session.metrics["microsteps"]),
                f'{mode}_t_all': travel_minutes}
            self._next_environment_episode += 1
            return CollectedEpisode(tuple(transitions), travel_minutes, metrics, total_vehicle_count)
        finally:
            session.close()

    def _record_evaluation_demand(self, vehicles, episode_index):
        from dataclasses import asdict
        directory = self.config.run.output_dir / 'evaluation_demands'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'episode_{episode_index:04d}.json'
        payload = {'schema_version': 1, 'seed': self.config.method.evaluation_demand_seed,
                   'episode_index': episode_index, 'demand_hash': demand_identity(vehicles),
                   'duration_seconds': self.config.simulation.demand_duration_seconds,
                   'od_rates': getattr(self.demand, 'od_rates', None),
                   'vehicles': [asdict(vehicle) for vehicle in vehicles]}
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise ValueError('evaluation demand manifest differs from the fixed evaluation set')
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

    def collect_warmup_episode(self, episode_index):
        collected = self._run_episode(mode='warmup', episode_index=episode_index)
        self.warmup_replay.append_episode(collected.transitions, total_travel_minutes=collected.total_travel_minutes,
                                         total_vehicle_count=collected.total_vehicle_count)
        self.metrics['warmup_episodes_completed'] += 1
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(collected.metrics)
        return {'complete': True, 'transitions': len(collected.transitions), **collected.metrics}

    def _world_loss(self, batch):
        flow = batch.flow.to(self.device)
        batch_size, frames = flow.shape[:2]
        z = self.world_model.encode(flow.reshape(-1, self.edge_count, 8)).reshape(
            batch_size, frames, self.edge_count, FLOW_LATENT_DIM)
        action = batch.action.to(self.device)
        method = self.config.method
        depth = min(frames - 1, method.supervision_horizon,
                    1 + int(self.metrics['world_model_updates']) // method.supervision_increment_updates)
        loss, fm_metrics, predicted = recursive_fm_loss(self.world_model.dynamics,
            z_sequence=z[:, :depth + 1], action=action[:, :depth],
            steps=method.fm_steps, solver=method.fm_solver,
            picard_iterations=method.picard_iterations, prior_weight=method.prior_loss_weight)
        self.metrics.update({key + '_last': float(value) for key, value in fm_metrics.items()})
        self.metrics['world_model_supervision_depth_last'] = float(depth)
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
        # Reconstruction trains the representation; stochastic endpoints are
        # supervised by FM rather than forcing every noise sample onto one future.
        loss = loss + reconstruction + bpr
        targets = batch.total_travel_minutes.to(self.device)
        valid = (targets > 0) & batch.episode_complete.to(self.device)
        if bool(valid.any()):
            total_loss, _ = episode_t_all_supervision_loss(flow_projector=self.world_model.flow_projector,
                z_tokens=z[valid], reward_targets=-targets[valid])
            loss = loss + total_loss
        with torch.no_grad():
            raw_fraction = self.world_model.decode(predicted.detach())[..., 2].mean(dim=-1)
        fraction = self.world_model.active_vehicle_fraction_readout(predicted.detach(),
            z_prev=torch.cat([z[:, :1], predicted[:, :-1]], dim=1).detach(),
            raw_decoder_fraction=raw_fraction, transition_seconds=30.)
        readout_loss = F.mse_loss(fraction, batch.active_vehicle_fraction[:, :depth].to(self.device))
        return loss, readout_loss

    def _update_world(self, replay, *, initial, batches=None):
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
        if batches is None:
            size = (self.config.resources.world_model_initialization_batch_size if initial
                    else self.config.resources.online_world_model_batch_size)
            batches = [replay.sample_batch(batch_size=size, rng=self.rng)]
        sample_count = sum(batch.flow.shape[0] for batch in batches)
        metrics = {'world_model_loss': 0., 'active_fraction_loss': 0.}
        for batch in batches:
            weight = batch.flow.shape[0] / sample_count
            loss, readout_loss = self._world_loss(batch)
            (loss * weight).backward()
            (readout_loss * weight).backward()
            metrics['world_model_loss'] += float(loss.detach()) * weight
            metrics['active_fraction_loss'] += float(readout_loss.detach()) * weight
        torch.nn.utils.clip_grad_norm_(self._world_parameters, GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(self.world_model.active_vehicle_fraction_readout.parameters(), GRAD_CLIP)
        optimizer.step()
        readout_optimizer.step()
        self.world_model.active_vehicle_fraction_readout.calibration_updates.add_(1)
        self.metrics['world_model_updates'] += 1
        self.metrics.update({key + '_last': value for key, value in metrics.items()})
        return metrics

    def initialize_world_model_epoch(self, epoch_index):
        if not 0 <= epoch_index < self.config.run.world_model_initialization_epochs:
            raise IndexError('world-model initialization exceeds configured epoch budget')
        if epoch_index != int(self.metrics['world_model_initialization_epochs_completed']):
            raise ValueError('world-model initialization must resume at the next epoch boundary')
        batches = self.warmup_replay.iter_epoch(
            self.config.resources.world_model_initialization_batch_size, self.rng)
        accumulation = self.config.resources.world_model_gradient_accumulation_steps
        totals = {'world_model_loss': 0., 'active_fraction_loss': 0.}
        samples = batch_count = updates = 0
        while group := list(islice(batches, accumulation)):
            metrics = self._update_world(self.warmup_replay, initial=True, batches=group)
            count = sum(batch.flow.shape[0] for batch in group)
            for key in totals:
                totals[key] += metrics[key] * count
            samples += count
            batch_count += len(group)
            updates += 1
        self.metrics['world_model_initialization_epochs_completed'] += 1
        self.metrics['world_model_initialization_batches_completed'] += batch_count
        self.metrics['world_model_initialization_samples_seen'] += samples
        self.metrics['world_model_initialization_updates'] += updates
        return {'epoch': epoch_index + 1, 'microbatches': batch_count,
                'samples': samples, 'optimizer_updates': updates,
                **{key: value / samples for key, value in totals.items()}}

    def train_online_episode(self, episode_index):
        collected = self._run_episode(mode='online', episode_index=episode_index)
        self.online_replay.append_episode(collected.transitions, total_travel_minutes=collected.total_travel_minutes,
                                         total_vehicle_count=collected.total_vehicle_count)
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
        self.metrics['evaluation_vehicle_minutes_sum'] += collected.total_travel_minutes * collected.total_vehicle_count
        self.metrics['evaluation_vehicle_count_sum'] += collected.total_vehicle_count
        self.metrics['evaluation_t_all_vehicle_weighted'] = (
            self.metrics['evaluation_vehicle_minutes_sum'] / max(self.metrics['evaluation_vehicle_count_sum'], 1))
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(collected.metrics)
        return {'complete': True, **collected.metrics}

    def _predict_with_disagreement(self, latent, action):
        method = self.config.method
        samples, variance, _ = self.world_model.latent_disagreement(
            latent, self._tensor(action).reshape(1, self.edge_count),
            n_samples=method.disagreement_samples,
            steps=method.fm_steps, solver=method.fm_solver,
            picard_iterations=method.picard_iterations)
        # Keep a sampled successor; averaging would change the transition law.
        return samples[0], variance.reshape(())

    def _imagine_from_seed(self, batch, batch_index, *, rollout_index):
        # Future random inflow is represented by the conditional world model;
        # explicit lifecycles and travel costs follow only this active seed cohort.
        states = copy.deepcopy([state for state in batch.vehicles[batch_index]
                                if state['departed'] and not state['arrived']])
        timestamp = float(batch.time[batch_index])
        deadline = self._episode_end_time()
        total_vehicle_count = int(batch.total_vehicle_count[batch_index])
        if total_vehicle_count <= 0:
            raise ValueError('active imagination seed requires a positive episode population')
        if not states or timestamp >= deadline:
            raise ValueError('imagination seed needs active vehicles before the episode deadline')
        with torch.no_grad():
            latent = self.world_model.encode(batch.flow[batch_index, 0].unsqueeze(0).to(self.device))
        rows, event_values = [], []
        values, next_values, rewards, terminals, physical_windows = [], [], [], [], []
        windows = min(IMAGINED_HORIZON, math.ceil((deadline - timestamp) / WORLD_MODEL_INTERVAL_SECONDS))
        for window_index in range(windows):
            context = self._context(states, timestamp, total_vehicle_count=total_vehicle_count)
            with torch.no_grad():
                decoded = self.world_model.decode(latent)[0].cpu().numpy()
            value = self.latent_critic(latent.detach(), self._tensor(context).unsqueeze(0)).reshape(())

            def decide(events, frozen_flow, now):
                # Compute one action-independent baseline before sampling the
                # simultaneous joint action, using the current physical context.
                event_context = self._context(states, now, total_vehicle_count=total_vehicle_count)
                baseline = self.latent_critic(latent.detach(), self._tensor(event_context).unsqueeze(0)).reshape(())
                event_index = len(event_values)
                event_values.append(baseline)
                observations = self._observations(events, frozen_flow, event_context, now)
                selection = self.actor.sample_action(self._tensor(observations.features),
                    torch.as_tensor(observations.mask, device=self.device), deterministic=False)
                decisions = []
                for i, state in enumerate(events):
                    decision = NextHopDecision(str(state['vehicle_id']), int(state['current_edge']),
                        int(observations.edge_indices[i, int(selection.candidate_index[i])]), float(now))
                    rows.append(dict(log_prob=selection.log_probability[i],
                        vehicle_id=str(state['vehicle_id']),
                        entropy=selection.entropy[i], window=window_index,
                        decision_time=float(now), event_index=event_index))
                    decisions.append(decision)
                return decisions

            speeds = np.clip(decoded[:, 1], 0.05, 1.5) * self.network.speed_limits_mps
            speeds = np.minimum(speeds, self.config.simulation.vehicle_max_speed_mps)
            duration = min(WORLD_MODEL_INTERVAL_SECONDS, deadline - timestamp)
            result = advance_imagined_window(states, candidates=self.candidates,
                edge_speeds_mps=speeds, frozen_flow=decoded, start_time=timestamp,
                interval_seconds=duration, total_vehicle_count=total_vehicle_count, decide=decide,
                observation_step_seconds=self.microstep_seconds,
                vehicle_max_speed_mps=self.config.simulation.vehicle_max_speed_mps,
                stop_when_empty=True)
            full_window = abs(result.end_time - (timestamp + WORLD_MODEL_INTERVAL_SECONDS)) <= 1e-7
            with torch.no_grad():
                # A terminal partial tail has physical cost but no fictitious
                # fixed-30-second model transition or endpoint risk charge.
                if full_window:
                    next_latent, disagreement = self._predict_with_disagreement(latent, result.action)
                    risk_cost = self.config.method.disagreement_weight * disagreement
                else:
                    next_latent, risk_cost = latent, latent.new_zeros(())
                reward = latent.new_tensor(-result.active_vehicle_seconds / (total_vehicle_count * 60.)) - risk_cost
            timestamp = result.end_time
            unfinished = sum(not bool(state['arrived']) for state in states)
            timed_out = timestamp >= deadline - 1e-9
            terminal = timed_out or unfinished == 0
            next_context = self._context(states, timestamp, total_vehicle_count=total_vehicle_count)
            with torch.no_grad():
                next_value = self.latent_critic(next_latent, self._tensor(next_context).unsqueeze(0)).reshape(())
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            terminals.append(terminal)
            physical_windows.append(result)
            latent = next_latent
            if terminal:
                break
        # Physical arrivals at the endpoint have already been processed. Only
        # unfinished vehicles at the real deadline receive an individual charge;
        # the shared critic retains physical costs and endpoint model risk only.
        failed_ids = {str(state['vehicle_id']) for state in states if not state['arrived']} if timed_out else set()
        attach_timeout_penalties(rows, failed_ids)
        # Macro lambda returns retain the existing gamma=1 convention. Event
        # targets use full Monte Carlo reward-to-go with the same undiscounted
        # physical-time objective, plus a value bootstrap only at truncation.
        targets = lambda_return(torch.stack(rewards), torch.stack(next_values),
                                terminal=torch.tensor(terminals, device=self.device))
        values_tensor = torch.stack(values)
        critic_loss = F.mse_loss(values_tensor, targets.detach())
        bootstrap = next_values[-1] if not terminals[-1] else latent.new_zeros(())
        event_targets = {}
        for row in rows:
            window = row['window']
            # Remove only physical time already elapsed before this action.
            # Endpoint risk depends on all actions in the macro window and is
            # therefore retained in full, even for a late event.
            elapsed = sum(segment.active_count * max(0., min(segment.end_time, row['decision_time'])
                - segment.start_time) for segment in physical_windows[window].cost_segments)
            target = torch.stack(rewards[window:]).sum() + elapsed / (total_vehicle_count * 60.) + bootstrap
            event_targets[row['event_index']] = target.detach()
            row['global_return'] = target.detach()
            row['global_advantage'] = (target - event_values[row['event_index']]).detach()
            row['global_reward'] = rewards[window]
        if event_values:
            critic_loss = critic_loss + F.mse_loss(torch.stack(event_values),
                torch.stack([event_targets[index] for index in range(len(event_values))]))
        return rows, critic_loss

    def _train_imagination_pass(self):
        if not self.online_replay.has_active_seeds:
            return {'latent_critic_loss': 0., 'imagined_events': 0., 'actor_loss': 0.}
        self.world_model.eval()
        self.world_model.set_online_chunk_profile()
        rows, macro_losses = [], []
        for rollout in range(IMAGINED_ROLLOUTS_PER_PASS):
            batch = self.online_replay.sample_active_seeds(batch_size=1, rng=self.rng)
            events, loss = self._imagine_from_seed(batch, 0, rollout_index=rollout)
            rows.extend(events)
            macro_losses.append(loss)
        macro_loss = torch.stack(macro_losses).mean()
        # Macro learning is independent of whether this window contained policy events.
        self.optimizers['latent_critic'].zero_grad(set_to_none=True)
        macro_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latent_critic.parameters(), GRAD_CLIP)
        self.optimizers['latent_critic'].step()
        metrics = {'latent_critic_loss': float(macro_loss.detach()),
                   'imagined_events': float(len(rows)), 'actor_loss': 0.}
        if rows:
            advantage = individual_actor_advantage(
                torch.stack([row['global_advantage'] for row in rows]),
                self._tensor([row['timeout_return'] for row in rows]))
            actor_loss = -(torch.stack([row['log_prob'] for row in rows]) * advantage.detach()
                           + ACTOR_ENTROPY_COEF * torch.stack([row['entropy'] for row in rows])).mean()
            self.optimizers['actor'].zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.optimizers['actor'].step()
            metrics['actor_loss'] = float(actor_loss.detach())
        self.metrics.update({key + '_last': value for key, value in metrics.items() if key.endswith('_loss')})
        return metrics

    def state_dict(self):
        numpy_state = np.random.get_state()
        return {'reward_contract': REWARD_CONTRACT,
            'models': {name: getattr(self, name).state_dict() for name in
                    ('world_model', 'actor', 'latent_critic')},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            'replay': {'warmup': self.warmup_replay.state_dict(), 'online': self.online_replay.state_dict()},
            'rng': {'python_random': random.getstate(), 'numpy_global': list(numpy_state),
                'numpy_generator': copy.deepcopy(self.rng.bit_generator.state), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            'metrics': dict(self.metrics), 'next_environment_episode': self._next_environment_episode}

    def load_state_dict(self, state):
        if set(state) != {'reward_contract', 'models', 'optimizers', 'replay', 'rng', 'metrics', 'next_environment_episode'}:
            raise ValueError('MainlineTrainer checkpoint state has an invalid schema')
        if state['reward_contract'] != REWARD_CONTRACT:
            raise ValueError('MainlineTrainer checkpoint reward contract requires individual timeout penalties')
        if set(state['models']) != {'world_model', 'actor', 'latent_critic'}:
            raise ValueError('MainlineTrainer model schema requires global reward only')
        if set(state['optimizers']) != set(self.optimizers):
            raise ValueError('MainlineTrainer optimizer schema requires global reward only')
        for name in ('world_model', 'actor', 'latent_critic'):
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
