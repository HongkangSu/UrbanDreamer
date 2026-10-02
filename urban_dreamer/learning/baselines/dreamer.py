"""DreamerV3-style reviewer baseline: compact RSSM world model with latent imagination.

The policy structure is identical to the model-free baselines (NextHopActor over
candidate features); only the learning signal changes. A recurrent state-space
model over flattened edge flow (E*8) is trained on the window replay with
reconstruction, window-reward and free-nats KL losses. The actor is then improved
on short imagined rollouts seeded from observed windows: the RSSM decodes edge
speeds for the identity-preserving physical imagination
(``advance_imagined_window``), while a single latent value critic over [h, z]
supplies the per-window global advantage (window-reward lambda-return minus the
pre-update value), broadcast to every imagined decision row of that window.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ...models.policy import attach_timeout_penalties, NextHopActor, lambda_return
from ...traffic.environment import NextHopDecision, safe_microstep_seconds
from ..imagination import advance_imagined_window
from ..observations import build_actor_observations
from ..replay import MacroTransition, ReplayBuffer
from . import METHODS
from .common import policy_gradient_step

FLOW_FEATURE_DIM = 8
WINDOW_SECONDS = 30.0

ENCODER_DIM = 128
ACTION_EMBED_DIM = 128
DETERMINISTIC_DIM = 256
STOCHASTIC_DIM = 32
HIDDEN_DIM = 256

SEQUENCE_LENGTH = 8
REPLAY_EPISODES = 64
WORLD_MODEL_UPDATES_PER_EPISODE = 8
WORLD_MODEL_BATCH_SIZE = 16
WORLD_MODEL_LR = 3e-4
KL_SCALE = 0.1
FREE_NATS = 1.0

IMAGINATION_PASSES = 4
IMAGINATION_ROLLOUTS = 4
IMAGINED_HORIZON = 8

GRAD_CLIP = 10.0


def _mlp(*dims: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class CompactRSSM(nn.Module):
    """Recurrent state-space model over flattened macro flow with window-reward head."""

    def __init__(self, edge_count: int) -> None:
        super().__init__()
        self.edge_count = int(edge_count)
        flat = self.edge_count * FLOW_FEATURE_DIM
        self.encoder = _mlp(flat, HIDDEN_DIM, HIDDEN_DIM, ENCODER_DIM)
        self.action_embed = nn.Linear(2 * self.edge_count + 2, ACTION_EMBED_DIM)
        self.z_embed = nn.Linear(STOCHASTIC_DIM, ACTION_EMBED_DIM)
        self.gru = nn.GRUCell(2 * ACTION_EMBED_DIM, DETERMINISTIC_DIM)
        self.prior = _mlp(DETERMINISTIC_DIM, HIDDEN_DIM, 2 * STOCHASTIC_DIM)
        self.posterior = _mlp(DETERMINISTIC_DIM + ENCODER_DIM, HIDDEN_DIM, 2 * STOCHASTIC_DIM)
        self.decoder = _mlp(DETERMINISTIC_DIM + STOCHASTIC_DIM, HIDDEN_DIM, HIDDEN_DIM, flat)
        self.reward_head = _mlp(DETERMINISTIC_DIM + STOCHASTIC_DIM, HIDDEN_DIM, 1)

    @staticmethod
    def _gaussian(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_raw = raw.chunk(2, dim=-1)
        return mean, F.softplus(std_raw) + 0.1

    def prior_params(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gaussian(self.prior(h))

    def posterior_params(self, h: torch.Tensor, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._gaussian(self.posterior(torch.cat((h, enc), dim=-1)))

    def dynamics_step(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor,
                      departure: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        embed = self.action_embed(torch.cat((action, departure, context), dim=-1))
        return self.gru(torch.cat((self.z_embed(z), embed), dim=-1), h)

    def decode(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((h, z), dim=-1))

    def predict_reward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat((h, z), dim=-1))


class LatentCritic(nn.Module):
    """Value head over the RSSM state [h, z] (256 tanh -> 256 tanh -> 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.net = _mlp(DETERMINISTIC_DIM + STOCHASTIC_DIM, HIDDEN_DIM, HIDDEN_DIM, 1)

    def forward(self, hz: torch.Tensor) -> torch.Tensor:
        return self.net(hz)


class DreamerMethod:
    """DreamerV3-style method wrapper: RSSM imagination training of the shared actor."""

    needs_window_records = True
    needs_row_flow = False

    def __init__(self, args, runner) -> None:
        if args is None or runner is None:
            raise NotImplementedError('method not implemented yet')
        self.args, self.runner = args, runner
        self.device = runner.device
        self.network = runner.network
        self.candidates = runner.candidates
        self.deadline = float(runner.deadline)
        self.edge_count = int(self.network.edge_count)
        self.vehicle_max_speed = float(runner.config.simulation.vehicle_max_speed_mps)
        lengths = tuple(self.network.external_lane_lengths_m.values()) \
            or self.network.lengths_m[self.network.passenger_allowed]
        self.microstep_seconds = safe_microstep_seconds(lengths, self.vehicle_max_speed)
        self.speed_limits = np.asarray(self.network.speed_limits_mps, dtype=np.float64)
        self.actor = NextHopActor().to(self.device)
        self.world_model = CompactRSSM(self.edge_count).to(self.device)
        self.latent_critic = LatentCritic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=WORLD_MODEL_LR)
        self.latent_optimizer = torch.optim.Adam(self.latent_critic.parameters(), lr=args.critic_lr)
        self.sequence_buffer = ReplayBuffer(sequence_length=SEQUENCE_LENGTH, max_episodes=REPLAY_EPISODES)
        self.seed_buffer = ReplayBuffer(sequence_length=1, max_episodes=REPLAY_EPISODES, active_only=True)
        self.rng = np.random.default_rng(args.seed + 613567)

    # --------------------------------------------------------------- policy
    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=True):
        with torch.no_grad():
            selection = self.actor.sample_action(
                torch.as_tensor(actor_obs.features, dtype=torch.float32, device=self.device),
                torch.as_tensor(actor_obs.mask, device=self.device), deterministic=not explore)
        indices = selection.candidate_index.cpu().tolist()
        aux = [dict(entropy=float(value)) for value in selection.entropy.cpu().tolist()]
        return indices, aux

    def _compute_logits(self, rows):
        weight = float(getattr(self.actor, 'prior_weight', 0.))
        if any(float(row.get('sp_prior_weight', weight)) != weight for row in rows):
            raise ValueError('Dreamer rows use a different shortest-path prior weight')
        features = torch.as_tensor(np.stack([row['features'] for row in rows]), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(np.stack([row['mask'] for row in rows]), device=self.device)
        return self.actor.forward_logits(features, mask)

    # ------------------------------------------------------------ episode hook
    def on_episode(self, rows, windows, *, episode):
        if not windows:
            return self._zero_metrics()
        population = int(windows[0].get('total_vehicle_count', self.runner.n))
        denominator = max(population, 1)
        transitions = [MacroTransition(flow=window['flow'], action=window['action'],
            departure=window['departure'], context=window['context'], next_flow=window['next_flow'],
            active_vehicle_fraction=float(np.clip(window['active_vehicle_seconds'] / (denominator * WINDOW_SECONDS), 0., 1.)),
            vehicles=window['vehicles'], time=float(window['time']), terminal=bool(window['terminal']),
            episode=int(episode), step_index=index) for index, window in enumerate(windows)]
        # Only the seed buffer needs vehicle snapshots; the sequence buffer stores flows.
        total_travel_minutes = sum(window['active_vehicle_seconds'] for window in windows) / denominator / 60.
        self.seed_buffer.append_episode(transitions, total_travel_minutes=total_travel_minutes, total_vehicle_count=population)
        stripped = [MacroTransition(flow=t.flow, action=t.action, departure=t.departure, context=t.context,
            next_flow=t.next_flow, active_vehicle_fraction=t.active_vehicle_fraction, time=t.time,
            terminal=t.terminal, episode=t.episode, step_index=t.step_index) for t in transitions]
        self.sequence_buffer.append_episode(stripped, total_travel_minutes=total_travel_minutes, total_vehicle_count=population)
        metrics = self._train_world_model()
        metrics.update(self._train_imagination())
        return metrics

    @staticmethod
    def _zero_metrics():
        return dict(world_model_loss=0., reconstruction_loss=0., reward_loss=0., kl=0.,
                    latent_critic_loss=0., actor_loss=0., imagined_events=0.)

    # ---------------------------------------------------------- world model
    def _world_model_loss(self, batch):
        flow = batch.flow.to(self.device)
        batch_size, frames = flow.shape[:2]
        steps = frames - 1
        flat = flow.reshape(batch_size, frames, -1)
        action = batch.action.to(self.device)
        departure = batch.departure.to(self.device)
        context = batch.context.to(self.device)
        reward_target = -0.5 * batch.active_vehicle_fraction.to(self.device)
        h = torch.zeros(batch_size, DETERMINISTIC_DIM, device=self.device)
        reconstruction = reward_loss = kl_term = 0.
        for t in range(steps):
            prior_mean, prior_std = self.world_model.prior_params(h)
            enc = self.world_model.encoder(flat[:, t])
            post_mean, post_std = self.world_model.posterior_params(h, enc)
            z = post_mean + post_std * torch.randn_like(post_mean)
            reconstruction = reconstruction + F.mse_loss(self.world_model.decode(h, z), flat[:, t])
            reward_loss = reward_loss + F.mse_loss(self.world_model.predict_reward(h, z).squeeze(-1),
                                                   reward_target[:, t])
            kl_element = (torch.log(prior_std / post_std)
                          + (post_std ** 2 + (post_mean - prior_mean) ** 2) / (2 * prior_std ** 2) - 0.5)
            kl_term = kl_term + torch.clamp(kl_element, min=FREE_NATS).mean()
            h = self.world_model.dynamics_step(h, z, action[:, t], departure[:, t], context[:, t])
        loss = (reconstruction + reward_loss) / steps + KL_SCALE * kl_term / steps
        parts = dict(world_model_loss=float(loss.detach()),
                     reconstruction_loss=float(reconstruction.detach()) / steps,
                     reward_loss=float(reward_loss.detach()) / steps,
                     kl=float(kl_term.detach()) / steps)
        return loss, parts

    def _train_world_model(self):
        if not self.sequence_buffer.has_samples:
            return dict(world_model_loss=0., reconstruction_loss=0., reward_loss=0., kl=0.)
        self.world_model.train()
        losses = []
        for _ in range(WORLD_MODEL_UPDATES_PER_EPISODE):
            batch = self.sequence_buffer.sample_batch(WORLD_MODEL_BATCH_SIZE, self.rng)
            loss, parts = self._world_model_loss(batch)
            self.world_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), GRAD_CLIP)
            if not torch.isfinite(gradient):
                raise RuntimeError('non-finite world model gradient')
            self.world_optimizer.step()
            losses.append(parts)
        return {key: float(np.mean([entry[key] for entry in losses])) for key in losses[0]}

    # ------------------------------------------------------------ imagination
    def _sample_prior(self, h):
        mean, std = self.world_model.prior_params(h)
        return mean + std * torch.randn_like(mean)

    def _initial_latent(self, flow):
        """Encode a seed flow state into the initial RSSM state (h, z)."""
        with torch.no_grad():
            flat = torch.as_tensor(np.asarray(flow), dtype=torch.float32, device=self.device).reshape(1, -1)
            enc = self.world_model.encoder(flat)
            h = self.world_model.gru(torch.cat((enc, torch.zeros(1, ACTION_EMBED_DIM, device=self.device)), dim=-1),
                                     torch.zeros(1, DETERMINISTIC_DIM, device=self.device))
            z = self._sample_prior(h)
        return h, z

    def _prior_dynamics_step(self, h, z, action, departure, context):
        with torch.no_grad():
            h_next = self.world_model.dynamics_step(h, z,
                torch.as_tensor(np.asarray(action), dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.as_tensor(np.asarray(departure), dtype=torch.float32, device=self.device).unsqueeze(0),
                torch.as_tensor(np.asarray(context), dtype=torch.float32, device=self.device).unsqueeze(0))
            z_next = self._sample_prior(h_next)
        return h_next, z_next

    def _decode_flow(self, h, z):
        with torch.no_grad():
            decoded = self.world_model.decode(h, z)
        return decoded.reshape(self.edge_count, FLOW_FEATURE_DIM).cpu().numpy()

    def _speeds(self, decoded_flow):
        speeds = np.clip(decoded_flow[:, 1], 0.05, 1.5) * self.speed_limits
        return np.minimum(speeds, self.vehicle_max_speed)

    @staticmethod
    def _active_count(states):
        return sum(1 for state in states if state.get('departed') and not state.get('arrived'))

    def _context(self, states, timestamp, population):
        return np.asarray([min(timestamp / self.deadline, 1.),
                           self._active_count(states) / max(population, 1)], dtype=np.float32)

    def _imagine_rollout(self, rollout):
        batch = self.seed_buffer.sample_active_seeds(1, self.rng)
        population = int(batch.total_vehicle_count[0])
        states = copy.deepcopy([state for state in batch.vehicles[0]
                                if state.get('departed') and not state.get('arrived')])
        timestamp = float(batch.time[0])
        horizon = min(IMAGINED_HORIZON, math.ceil((self.deadline - timestamp) / WINDOW_SECONDS))
        if not states or horizon <= 0:
            return None
        h, z = self._initial_latent(batch.flow[0, 0].numpy())
        rows = []
        latents, next_latents, rewards, terminals = [], [], [], []
        for window in range(horizon):
            context = self._context(states, timestamp, population)
            flow_decoded = self._decode_flow(h, z)
            latents.append(torch.cat((h, z), dim=-1).detach())
            with torch.no_grad():
                rewards.append(float(self.world_model.predict_reward(h, z).reshape(())))

            def decide(pending, frozen, now):
                actor_obs = build_actor_observations(
                    policy_edge_features=np.asarray(frozen[:, [2, 1]], dtype=np.float32),
                    vehicle_states=pending, candidates=self.candidates,
                    episode_progress=float(context[0]), active_vehicle_fraction=float(context[1]))
                with torch.no_grad():
                    selection = self.actor.sample_action(
                        torch.as_tensor(actor_obs.features, dtype=torch.float32, device=self.device),
                        torch.as_tensor(actor_obs.mask, device=self.device))
                indices = selection.candidate_index.cpu().tolist()
                decisions = []
                for i, state in enumerate(pending):
                    rows.append(dict(vehicle_id=str(state['vehicle_id']), rollout=rollout, window=window,
                        features=actor_obs.features[i].copy(), mask=actor_obs.mask[i].copy(),
                        action=int(indices[i]), current_edge=int(state['current_edge']),
                        sp_prior_weight=float(getattr(self.actor, 'prior_weight', 0.)),
                        edge_indices=actor_obs.edge_indices[i].copy(),
                        flow=np.asarray(frozen, dtype=np.float32).copy()))
                    decisions.append(NextHopDecision(str(state['vehicle_id']), int(state['current_edge']),
                        int(actor_obs.edge_indices[i, indices[i]]), float(now)))
                return decisions

            result = advance_imagined_window(states, candidates=self.candidates,
                edge_speeds_mps=self._speeds(flow_decoded), frozen_flow=flow_decoded, start_time=timestamp,
                total_vehicle_count=population, decide=decide, observation_step_seconds=self.microstep_seconds,
                vehicle_max_speed_mps=self.vehicle_max_speed)
            timestamp = timestamp + WINDOW_SECONDS
            terminal = timestamp >= self.deadline or self._active_count(states) == 0
            h, z = self._prior_dynamics_step(h, z, result.action, result.departure, context)
            next_latents.append(torch.cat((h, z), dim=-1).detach())
            terminals.append(bool(terminal))
            if terminal:
                break
        failed_ids = {str(state['vehicle_id']) for state in states if not state.get('arrived')} \
            if timestamp >= self.deadline else set()
        attach_timeout_penalties(rows, failed_ids)
        return rows, (latents, next_latents, rewards, terminals)

    def _score_rollout(self, trajectory):
        """Per-window lambda-return targets and advantages for one imagined rollout."""
        latents, next_latents, rewards, terminals = trajectory
        with torch.no_grad():
            latents_tensor = torch.cat(latents)
            values = self.latent_critic(latents_tensor).squeeze(-1)
            next_values = self.latent_critic(torch.cat(next_latents)).squeeze(-1)
            targets = lambda_return(torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
                next_values, terminal=torch.as_tensor(terminals, dtype=torch.bool, device=self.device))
        return latents_tensor, targets, targets - values

    def _imagination_pass(self):
        all_rows, trajectories = [], []
        for rollout in range(IMAGINATION_ROLLOUTS):
            result = self._imagine_rollout(rollout)
            if result is None:
                continue
            rows, trajectory = result
            latents, targets, advantages = self._score_rollout(trajectory)
            for row in rows:
                row['advantage'] = float(advantages[row['window']])
            trajectories.append((latents, targets))
            all_rows.extend(rows)
        metrics = dict(latent_critic_loss=0., actor_loss=0., imagined_events=float(len(all_rows)))
        if trajectories:
            losses = [F.mse_loss(self.latent_critic(latents).squeeze(-1), targets)
                      for latents, targets in trajectories]
            latent_loss = torch.stack(losses).mean()
            self.latent_optimizer.zero_grad(set_to_none=True)
            latent_loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(self.latent_critic.parameters(), GRAD_CLIP)
            if not torch.isfinite(gradient):
                raise RuntimeError('non-finite latent critic gradient')
            self.latent_optimizer.step()
            metrics['latent_critic_loss'] = float(latent_loss.detach())
        if all_rows:
            update = policy_gradient_step(compute_logits=self._compute_logits,
                actor_parameters=self.actor.parameters(), actor_optimizer=self.actor_optimizer,
                rows=all_rows, advantages=[row['advantage'] for row in all_rows])
            metrics['actor_loss'] = update['actor_loss']
        return metrics

    def _train_imagination(self):
        zero = dict(latent_critic_loss=0., actor_loss=0., imagined_events=0.)
        if not self.seed_buffer.has_active_seeds:
            return zero
        self.world_model.eval()
        metrics = [entry for entry in (self._imagination_pass() for _ in range(IMAGINATION_PASSES))
                   if entry is not None]
        if not metrics:
            return zero
        return {key: float(np.mean([entry[key] for entry in metrics])) for key in zero}

    # ------------------------------------------------------------- checkpoint
    def state_dict(self):
        return dict(actor=self.actor.state_dict(), world_model=self.world_model.state_dict(),
            latent_critic=self.latent_critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            world_optimizer=self.world_optimizer.state_dict(),
            latent_optimizer=self.latent_optimizer.state_dict(),
            sequence_replay=self.sequence_buffer.state_dict(), seed_replay=self.seed_buffer.state_dict(),
            rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.world_model.load_state_dict(state['world_model'])
        self.latent_critic.load_state_dict(state['latent_critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.world_optimizer.load_state_dict(state['world_optimizer'])
        self.latent_optimizer.load_state_dict(state['latent_optimizer'])
        self.sequence_buffer.load_state_dict(state['sequence_replay'])
        self.seed_buffer.load_state_dict(state['seed_replay'])
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])


METHODS['dreamer'] = DreamerMethod
