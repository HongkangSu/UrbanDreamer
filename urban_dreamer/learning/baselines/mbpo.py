"""MBPO-style reviewer baseline: short-horizon ensemble model rollouts, global rewards.

The policy side is the global-only variant of the reference actor-critic
(NextHopActor updated by ``common.policy_gradient_step`` with per-window team
advantages from ``common.GlobalTrafficCritic`` plus an individual terminal
timeout cost, kept out of the shared critic and model reward labels). A 5-member probabilistic ensemble predicts the one-window flow
delta N(delta_flow; mean, var) and the global window reward from (flow, action,
departure, context). Each episode branches k=2 imagined windows from M=16
observed window seeds through the identity-preserving physical imagination
(``advance_imagined_window``); the imagined window sequence is scored by the
global critic without fitting it (``global_critic_update(..., update=False)``),
and the resulting imagined decision rows are mixed 1:1 with real rows in every
policy update, the MBPO model-augmented batch scheme.
"""
from __future__ import annotations

from collections import deque
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ...models.policy import attach_timeout_penalties, NextHopActor
from ...traffic.environment import NextHopDecision, safe_microstep_seconds
from ..imagination import advance_imagined_window
from ..observations import build_actor_observations
from . import METHODS
from .common import GlobalTrafficCritic, global_critic_update, policy_gradient_step

FLOW_FEATURE_DIM = 8
WINDOW_SECONDS = 30.0
HIDDEN_DIM = 256

ENSEMBLE_SIZE = 5
MODEL_TRAIN_ROUNDS = 4
MODEL_BATCH_SIZE = 64
MODEL_BUFFER_EPISODES = 200
MODEL_LR = 3e-4
LOGVAR_MIN = -8.0
LOGVAR_MAX = 2.0

ROLLOUT_SEEDS = 16
ROLLOUT_LENGTH = 2
MODEL_ROWS_LIMIT = 40000
MODEL_ROWS_CHECKPOINT = 5000
POLICY_UPDATES_PER_EPISODE = 4

GRAD_CLIP = 10.0


class EnsembleMember(nn.Module):
    """Gaussian flow-delta model with a scalar global window-reward head."""

    def __init__(self, edge_count: int) -> None:
        super().__init__()
        self.edge_count = int(edge_count)
        flat = self.edge_count * FLOW_FEATURE_DIM
        self.trunk = nn.Sequential(
            nn.Linear(flat + 2 * self.edge_count + 2, HIDDEN_DIM), nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh())
        self.head = nn.Linear(HIDDEN_DIM, 2 * flat + 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.head(self.trunk(x))
        flat = self.edge_count * FLOW_FEATURE_DIM
        mean = raw[..., :flat]
        logvar = torch.clamp(raw[..., flat:2 * flat], LOGVAR_MIN, LOGVAR_MAX)
        return mean, logvar, raw[..., -1]

    def gaussian_nll(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        mean, logvar, _ = self.forward(x)
        return (0.5 * (logvar + (delta - mean) ** 2 / torch.exp(logvar))).mean()


class MbpoMethod:
    """MBPO method wrapper: ensemble model rollouts mixed into global-reward actor updates."""

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
        # The added global critic must not change actor initialization (reference parity).
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 73861)
            self.global_critic = GlobalTrafficCritic(self.edge_count).to(self.device)
        self.global_optimizer = torch.optim.Adam(self.global_critic.parameters(),
            lr=getattr(args, 'global_critic_lr', args.critic_lr))
        self.ensemble = nn.ModuleList(EnsembleMember(self.edge_count) for _ in range(ENSEMBLE_SIZE)).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.model_optimizer = torch.optim.Adam(self.ensemble.parameters(), lr=MODEL_LR)
        self.window_buffer: deque = deque(maxlen=MODEL_BUFFER_EPISODES)
        self.model_rows: deque = deque(maxlen=MODEL_ROWS_LIMIT)
        self.model_rows_prior_weight = 0.
        self.rng = np.random.default_rng(args.seed + 911519)

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
        features = torch.as_tensor(np.stack([row['features'] for row in rows]), dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(np.stack([row['mask'] for row in rows]), device=self.device)
        return self.actor.forward_logits(features, mask)

    # ------------------------------------------------------------ episode hook
    def on_episode(self, rows, windows, *, episode):
        self._sync_model_rows_prior()
        if not windows:
            return dict(model_nll=0., reward_loss=0., global_critic_loss=0., actor_loss=0.,
                        imagined_events=0., model_rows=float(len(self.model_rows)))
        self.window_buffer.append([self._model_sample(window) for window in windows])
        metrics = self._train_model()
        advantages, critic_metrics = global_critic_update(global_critic=self.global_critic,
            global_optimizer=self.global_optimizer, windows=windows, update=True)
        metrics.update(critic_metrics)
        metrics['imagined_events'] = float(self._generate_model_rows(windows))
        metrics.update(self._policy_updates(rows, advantages))
        metrics['model_rows'] = float(len(self.model_rows))
        return metrics

    def _model_sample(self, window):
        return dict(x=np.concatenate((np.asarray(window['flow'], dtype=np.float32).reshape(-1),
                        np.asarray(window['action'], dtype=np.float32),
                        np.asarray(window['departure'], dtype=np.float32),
                        np.asarray(window['context'], dtype=np.float32))).astype(np.float32),
            delta=(np.asarray(window['next_flow'], dtype=np.float32)
                   - np.asarray(window['flow'], dtype=np.float32)).reshape(-1).astype(np.float32),
            reward=np.float32(window['reward']))

    # ----------------------------------------------------------------- model
    def _train_model(self):
        samples = [sample for episode in self.window_buffer for sample in episode]
        if not samples:
            return dict(model_nll=0., reward_loss=0.)
        self.ensemble.train()
        nlls, reward_losses = [], []
        for _ in range(MODEL_TRAIN_ROUNDS):
            self.model_optimizer.zero_grad(set_to_none=True)
            round_nll = round_reward = 0.
            for member in self.ensemble:
                picks = [samples[int(index)]
                         for index in self.rng.integers(len(samples), size=min(MODEL_BATCH_SIZE, len(samples)))]
                x = torch.as_tensor(np.stack([pick['x'] for pick in picks]), dtype=torch.float32, device=self.device)
                delta = torch.as_tensor(np.stack([pick['delta'] for pick in picks]), dtype=torch.float32, device=self.device)
                reward = torch.as_tensor([pick['reward'] for pick in picks], dtype=torch.float32, device=self.device)
                mean, logvar, predicted = member(x)
                nll = (0.5 * (logvar + (delta - mean) ** 2 / torch.exp(logvar))).mean()
                reward_loss = F.mse_loss(predicted, reward)
                (nll + reward_loss).backward()
                round_nll += float(nll.detach())
                round_reward += float(reward_loss.detach())
            gradient = torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), GRAD_CLIP)
            if not torch.isfinite(gradient):
                raise RuntimeError('non-finite model gradient')
            self.model_optimizer.step()
            nlls.append(round_nll / ENSEMBLE_SIZE)
            reward_losses.append(round_reward / ENSEMBLE_SIZE)
        return dict(model_nll=float(np.mean(nlls)), reward_loss=float(np.mean(reward_losses)))

    def _predict_transition(self, flow, action, departure, context):
        """One ensemble member's sampled next flow and its predicted global window reward."""
        member = self.ensemble[int(self.rng.integers(len(self.ensemble)))]
        x = np.concatenate((flow.reshape(-1), action, departure, context)).astype(np.float32)
        with torch.no_grad():
            mean, logvar, reward = member(torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0))
            delta = (mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)).squeeze(0).cpu().numpy()
        next_flow = flow + delta.reshape(self.edge_count, FLOW_FEATURE_DIM)
        next_flow[:, 0] = np.clip(next_flow[:, 0], 0., 1.2)
        next_flow[:, 1] = np.clip(next_flow[:, 1], 0.05, 1.5)
        next_flow[:, 2] = np.clip(next_flow[:, 2], 0., 1.2)
        next_flow[:, 3] = np.clip(next_flow[:, 3], 0., 1.2)
        next_flow[:, 4:] = flow[:, 4:]
        return next_flow.astype(np.float32), float(reward.reshape(()))

    # --------------------------------------------------------------- rollouts
    def _speeds(self, flow):
        speeds = np.clip(flow[:, 1], 0.05, 1.5) * self.speed_limits
        return np.minimum(speeds, self.vehicle_max_speed)

    @staticmethod
    def _active_count(states):
        return sum(1 for state in states if state.get('departed') and not state.get('arrived'))

    def _context(self, states, timestamp, population):
        return np.asarray([min(timestamp / self.deadline, 1.),
                           self._active_count(states) / max(population, 1)], dtype=np.float32)

    @staticmethod
    def _global_obs(flow, context):
        return np.concatenate((np.asarray(flow[:, :4], dtype=np.float32).reshape(-1),
                               np.asarray(context, dtype=np.float32)))

    def _rollout_from_seed(self, window, rollout):
        population = int(window.get('total_vehicle_count', self.runner.n))
        states = copy.deepcopy([state for state in window['vehicles']
                                if state.get('departed') and not state.get('arrived')])
        if not states:
            return []
        flow = np.asarray(window['flow'], dtype=np.float32).copy()
        timestamp = float(window['time'])
        rows, imagined_windows = [], []
        for step in range(ROLLOUT_LENGTH):
            context = self._context(states, timestamp, population)
            obs = self._global_obs(flow, context)

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
                    rows.append(dict(vehicle_id=str(state['vehicle_id']), rollout=rollout, window=step,
                        features=actor_obs.features[i].copy(), mask=actor_obs.mask[i].copy(),
                        action=int(indices[i]), current_edge=int(state['current_edge']),
                        sp_prior_weight=float(getattr(self.actor, 'prior_weight', 0.)),
                        edge_indices=actor_obs.edge_indices[i].copy(),
                        flow=np.asarray(frozen, dtype=np.float32).copy()))
                    decisions.append(NextHopDecision(str(state['vehicle_id']), int(state['current_edge']),
                        int(actor_obs.edge_indices[i, indices[i]]), float(now)))
                return decisions

            result = advance_imagined_window(states, candidates=self.candidates,
                edge_speeds_mps=self._speeds(flow), frozen_flow=flow, start_time=timestamp,
                total_vehicle_count=population, decide=decide, observation_step_seconds=self.microstep_seconds,
                vehicle_max_speed_mps=self.vehicle_max_speed)
            next_timestamp = timestamp + WINDOW_SECONDS
            terminal = next_timestamp >= self.deadline or self._active_count(states) == 0
            flow, reward = self._predict_transition(flow, result.action, result.departure, context)
            imagined_windows.append(dict(obs=obs, reward=reward, terminal=bool(terminal),
                next_obs=self._global_obs(flow, self._context(states, next_timestamp, population))))
            timestamp = next_timestamp
            if terminal:
                break
        failed_ids = {str(state['vehicle_id']) for state in states if not state.get('arrived')} \
            if timestamp >= self.deadline else set()
        attach_timeout_penalties(rows, failed_ids)
        advantages, _ = global_critic_update(global_critic=self.global_critic,
                                             windows=imagined_windows, update=False)
        for row in rows:
            row['advantage'] = float(advantages[row['window']])
        return rows

    def _sync_model_rows_prior(self):
        weight = float(getattr(self.actor, 'prior_weight', 0.))
        if weight != self.model_rows_prior_weight:
            self.model_rows.clear()
            self.model_rows_prior_weight = weight

    def _generate_model_rows(self, windows):
        self._sync_model_rows_prior()
        seeds = [window for window in windows if window['vehicles'] and float(window['time']) < self.deadline]
        if not seeds:
            return 0
        count = 0
        for rollout, pick in enumerate(self.rng.integers(len(seeds), size=min(ROLLOUT_SEEDS, len(seeds)))):
            rows = self._rollout_from_seed(seeds[int(pick)], rollout)
            self.model_rows.extend(rows)
            count += len(rows)
        return count

    # ---------------------------------------------------------------- updates
    def _policy_updates(self, rows, advantages):
        weight = float(getattr(self.actor, 'prior_weight', 0.))
        if self.model_rows_prior_weight != weight or any(
                float(row.get('sp_prior_weight', 0.)) != weight for row in self.model_rows):
            raise ValueError('MBPO synthetic rows use a different shortest-path prior weight')
        if any(float(row.get('sp_prior_weight', weight)) != weight for row in rows):
            raise ValueError('MBPO real rows use a different shortest-path prior weight')
        if not rows and not self.model_rows:
            return dict(actor_loss=0.)
        real_advantages = [float(advantages[row['window']]) for row in rows]
        updates = []
        model_rows = list(self.model_rows)
        for _ in range(POLICY_UPDATES_PER_EPISODE):
            batch_rows = list(rows)
            batch_advantages = list(real_advantages)
            if model_rows:
                take = min(len(model_rows), max(len(rows), 1))
                picks = [model_rows[int(index)] for index in self.rng.integers(len(model_rows), size=take)]
                batch_rows.extend(picks)
                batch_advantages.extend(pick['advantage'] for pick in picks)
            if not batch_rows:
                break
            updates.append(policy_gradient_step(compute_logits=self._compute_logits,
                actor_parameters=self.actor.parameters(), actor_optimizer=self.actor_optimizer,
                rows=batch_rows, advantages=batch_advantages))
        return dict(actor_loss=float(np.mean([entry['actor_loss'] for entry in updates])) if updates else 0.)

    # ------------------------------------------------------------- checkpoint
    def state_dict(self):
        return dict(actor=self.actor.state_dict(), global_critic=self.global_critic.state_dict(),
            ensemble=self.ensemble.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            global_optimizer=self.global_optimizer.state_dict(),
            model_optimizer=self.model_optimizer.state_dict(),
            model_rows=list(self.model_rows)[-MODEL_ROWS_CHECKPOINT:],
            model_rows_prior_weight=self.model_rows_prior_weight,
            rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        self.actor.load_state_dict(state['actor'])
        self.global_critic.load_state_dict(state['global_critic'])
        self.ensemble.load_state_dict(state['ensemble'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.global_optimizer.load_state_dict(state['global_optimizer'])
        self.model_optimizer.load_state_dict(state['model_optimizer'])
        self.model_rows.clear()
        self.model_rows.extend(copy.deepcopy(state['model_rows']))
        self.model_rows_prior_weight = float(state.get('model_rows_prior_weight', 0.))
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])


METHODS['mbpo'] = MbpoMethod
