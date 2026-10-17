"""Independent routing Q-learning with shared whole-team rewards.

IQL here means independent Q-learning, not offline implicit Q-learning. One
parameter-shared Q scorer serves the dynamic vehicle agents. Transitions link
consecutive decisions of the same vehicle and use gamma=1. The last decision
receives the remaining TEAM return through the end of the collected episode,
including reward accrued after that vehicle arrives.

The collector supplies 30-second integrated rewards, not per-second reward
traces. Reward between asynchronous decisions is therefore allocated uniformly
within each recorded window. This preserves complete window reward accounting
but approximates its within-window timing. In optional individual timeout mode,
one terminal penalty is added per failed vehicle. Local route costs never enter
the Bellman targets. The annealed shortest-path prior
affects collection/evaluation logits only; raw Q targets remain unbiased by it.

Fresh CLI runs use ``double_dueling_v2``: masked dueling Q values, Double DQN
targets, and episode-annealed epsilon-greedy collection. Forced actions are
compressed into the reward interval between real choices without dropping the
team episode tail. ``legacy_v1`` retains the historical checkpoint behavior;
missing profile fields in archived argument dictionaries select that version.
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
from fractions import Fraction
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ...models.policy import (CANDIDATE_FEATURE_DIM, NextHopActor,
                              ActionSelection, apply_shortest_path_prior)
from . import METHODS
from .common import global_reward_integral


class RoutingQActor(NextHopActor):
    """Candidate Q scorer exposing separate Bellman and behavior outputs."""

    def __init__(self, *, dueling=False):
        super().__init__()
        self.dueling = bool(dueling)
        if self.dueling:
            self.policy_head = torch.nn.Linear(self.policy_head.in_features, 1, bias=False)
            torch.nn.init.orthogonal_(self.policy_head.weight, gain=.01)
            self.value_head = torch.nn.Linear(self.policy_head.in_features, 1)
            torch.nn.init.zeros_(self.value_head.weight)
            torch.nn.init.zeros_(self.value_head.bias)
            # A buffer follows the actor's episode-frozen CPU collection mirror.
            self.register_buffer('exploration_epsilon', torch.ones(()))

    def _components(self, features, mask):
        if features.ndim != 3 or features.shape[-1] != CANDIDATE_FEATURE_DIM:
            raise ValueError('expected candidate features (B, K, 10)')
        if mask.shape != features.shape[:2] or features.shape[1] == 0:
            raise ValueError('candidate mask must have shape (B, K) with K positive')
        legal = mask.to(device=features.device, dtype=torch.bool)
        if not bool(legal.any(dim=-1).all()):
            raise ValueError('every decision must have at least one legal candidate')
        if not bool(torch.isfinite(features).all()):
            raise ValueError('candidate features must be finite')
        hidden = self.candidate_scorer(features)
        if not self.dueling:
            return self.policy_head(hidden).squeeze(-1), None, legal
        pooled = hidden.masked_fill(~legal[..., None], 0.).sum(1) / legal.sum(-1, keepdim=True)
        # Center before projection so the common value cannot erase small
        # action differences in the floating-point behavior logits.
        advantage = self.policy_head(hidden - pooled[:, None]).squeeze(-1)
        return advantage, self.value_head(pooled), legal

    def forward_q(self, features, mask):
        advantage, value, legal = self._components(features, mask)
        q = advantage if value is None else advantage + value
        return q.masked_fill(~legal, -torch.inf)

    def forward_logits(self, features, mask):
        if self.dueling:
            advantage, _, legal = self._components(features, mask)
            logits = advantage.masked_fill(~legal, -torch.inf)
        else:
            logits = self.forward_q(features, mask)
        return apply_shortest_path_prior(logits, features, mask,
            weight=self.prior_weight, temperature_seconds=self.prior_temperature_seconds)


def vehicle_decision_transitions(rows, windows, *, deadline, include_individual_timeout=False,
                                 branch_only=False):
    """Build same-vehicle Bellman transitions from shared integrated rewards.

    Uses exact physical window bounds, with normalized global observations as
    a legacy fallback. The optional timeout cost enters only the last decision
    of each vehicle, so Bellman propagation counts it once per trajectory.
    """
    if not rows:
        return []
    if not windows:
        raise ValueError('IQL decision transitions need shared reward windows')
    integrated_reward, episode_end = global_reward_integral(windows, deadline=deadline)

    trajectories = defaultdict(list)
    for row in rows:
        trajectories[row['vehicle_id']].append(row)
    transitions = []
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda row: float(row['time']))
        timeout = sum(float(row.get('timeout_penalty', 0.)) for row in trajectory) if include_individual_timeout else 0.
        if branch_only:
            # Skip forced actions but integrate across them, including the
            # team's full tail after the vehicle's final real choice.
            for row in trajectory:
                mask = np.asarray(row['mask'], dtype=np.bool_)
                action = int(row['action'])
                if mask.ndim != 1 or not 0 <= action < len(mask) or not mask[action]:
                    raise ValueError('collected IQL action must be legal')
            trajectory = [row for row in trajectory if np.count_nonzero(row['mask']) > 1]
        for index, row in enumerate(trajectory):
            successor = trajectory[index + 1] if index + 1 < len(trajectory) else None
            end_time = successor['time'] if successor is not None else episode_end
            mask = np.asarray(row['mask'], dtype=np.bool_).copy()
            action = int(row['action'])
            if not 0 <= action < len(mask) or not mask[action]:
                raise ValueError('collected IQL action must be legal')
            reward = integrated_reward(end_time) - integrated_reward(row['time'])
            if include_individual_timeout and successor is None:
                penalty = timeout if branch_only else float(row.get('timeout_penalty', 0.))
                if not math.isfinite(penalty):
                    raise ValueError('IQL individual timeout penalty must be finite')
                reward += penalty
            transitions.append(dict(features=np.asarray(row['features'], dtype=np.float32).copy(),
                mask=mask, action=action,
                reward=reward,
                next_features=(np.asarray(successor['features'], dtype=np.float32).copy()
                               if successor is not None else None),
                next_mask=(np.asarray(successor['mask'], dtype=np.bool_).copy()
                           if successor is not None else None), terminal=successor is None))
    return transitions


class IndependentQMethod:
    """IQL with a sampled-transition budget proportional to new transitions.

    The ratio specifies optimization draws, not unique replay coverage. Target
    tau is applied per ``batch_size`` draws, including fractional final batches.
    Replay retains its usual bounded FIFO semantics even when an entire new
    episode exceeds capacity; the draw budget still counts all inserted rows.
    """

    needs_row_flow = False
    schedule_version = 'sample_to_insert_v1'
    target_tau_unit = 'batch_size_sampled_transitions'

    def __init__(self, args, runner):
        self.args, self.runner, self.device = args, runner, runner.device
        self.profile = getattr(args, 'iql_profile', 'legacy_v1')
        if self.profile not in {'legacy_v1', 'double_dueling_v2'}:
            raise ValueError('unsupported IQL implementation profile')
        self.stable = self.profile == 'double_dueling_v2'
        self.epsilon_initial = float(getattr(args, 'iql_epsilon_initial', 1.))
        self.epsilon_final = float(getattr(args, 'iql_epsilon_final', .05))
        self.epsilon_decay_episodes = int(getattr(args, 'iql_epsilon_decay_episodes', 1000))
        if (not 0 <= self.epsilon_final <= self.epsilon_initial <= 1
                or self.epsilon_decay_episodes < 1):
            raise ValueError('invalid IQL exploration schedule')
        if self.stable:
            self.schedule_version = 'branch_double_dueling_v2'
        self.actor = RoutingQActor(dueling=self.stable).to(self.device)
        if self.stable:
            self.actor.exploration_epsilon.fill_(self.epsilon_initial)
        self.target = copy.deepcopy(self.actor).to(self.device).requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.replay = deque(maxlen=int(getattr(args, 'iql_replay_capacity', 200_000)))
        self.rng = np.random.default_rng(args.seed + 8191)
        self.batch_size = int(getattr(args, 'iql_batch_size', 512))
        if getattr(args, 'iql_updates', None) is not None:
            raise ValueError('iql_updates is obsolete; use iql_sample_to_insert_ratio')
        self.sample_to_insert_ratio = float(getattr(args, 'iql_sample_to_insert_ratio', 1.))
        self.target_tau = float(getattr(args, 'iql_target_tau', .01))
        if self.replay.maxlen <= 0 or self.batch_size <= 0 \
                or not math.isfinite(self.sample_to_insert_ratio) or self.sample_to_insert_ratio <= 0 \
                or not 0 < self.target_tau <= 1:
            raise ValueError('invalid IQL replay, batch, update or target settings')
        self.updates = 0
        self.draw_credit = Fraction(0)
        self.total_inserted = self.total_draws = 0
        self.completed_episodes = 0

    def _schedule_settings(self):
        settings = dict(replay_capacity=self.replay.maxlen, batch_size=self.batch_size,
                    sample_to_insert_ratio=self.sample_to_insert_ratio,
                    target_tau=self.target_tau, target_tau_unit=self.target_tau_unit,
                    actor_lr=float(self.args.actor_lr))
        if self.stable:
            settings.update(epsilon_initial=self.epsilon_initial, epsilon_final=self.epsilon_final,
                            epsilon_decay_episodes=self.epsilon_decay_episodes)
        return settings

    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=True, temperature=1., sample_actions=None):
        with torch.no_grad():
            features = torch.as_tensor(actor_obs.features, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(actor_obs.mask, device=self.device)
            if getattr(self, 'stable', False) and explore and sample_actions is None:
                logits = self.actor.forward_logits(features, mask)
                legal = mask.to(torch.bool)
                epsilon = self.actor.exploration_epsilon
                probabilities = legal.to(logits.dtype) * (epsilon / legal.sum(-1, keepdim=True))
                probabilities.scatter_add_(1, logits.argmax(-1, keepdim=True),
                    torch.ones_like(logits[:, :1]) * (1. - epsilon))
                distribution = torch.distributions.Categorical(probs=probabilities)
                chosen = distribution.sample()
                selection = ActionSelection(chosen, distribution.log_prob(chosen), distribution.entropy())
            else:
                selection = self.actor.sample_action(features, mask,
                    deterministic=not (explore if sample_actions is None else sample_actions),
                    temperature=temperature)
        indices, entropies = selection.indices_and_entropy()
        return indices, [dict(entropy=entropy) for entropy in entropies]

    def learning_protocol(self):
        return dict(version=self.schedule_version, settings=self._schedule_settings(),
                    reward=('whole_team_global_tail' if self.reward_mode == 'global_only'
                            else 'whole_team_global_tail_with_individual_terminal_timeout'),
                    prior='behavior_only', profile=self.profile,
                    bellman='double_dqn' if self.stable else 'target_max',
                    exploration='annealed_epsilon_greedy' if self.stable else 'fixed_temperature_softmax',
                    replay_events='real_choices_only' if self.stable else 'all_events',
                    value_architecture='masked_dueling' if self.stable else 'independent_candidate_scores')

    @property
    def reward_mode(self):
        return ('global_only' if getattr(self.args, 'global_reward_only', True)
                else 'global_with_individual_timeout')

    def _batch_targets(self, batch):
        targets = torch.as_tensor([entry['reward'] for entry in batch], dtype=torch.float32, device=self.device)
        continuing = [index for index, entry in enumerate(batch) if not entry['terminal']]
        if continuing:
            features = torch.as_tensor(np.stack([batch[index]['next_features'] for index in continuing]),
                                       dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(np.stack([batch[index]['next_mask'] for index in continuing]), device=self.device)
            with torch.no_grad():
                next_q = self.target.forward_q(features, mask)
                if self.stable:
                    # Select with raw online Q (never the optional behavior prior).
                    choices = self.actor.forward_q(features, mask).argmax(-1, keepdim=True)
                    targets[continuing] += next_q.gather(1, choices).squeeze(1)
                else:
                    targets[continuing] += next_q.amax(dim=-1)
        return targets

    def on_episode(self, rows, windows, *, episode):
        started = time.monotonic()
        transitions = vehicle_decision_transitions(rows, windows, deadline=self.runner.deadline,
            include_individual_timeout=self.reward_mode != 'global_only', branch_only=self.stable)
        inserted = len(transitions)
        self.replay.extend(transitions)
        self.total_inserted += inserted
        self.draw_credit += inserted * Fraction(str(self.sample_to_insert_ratio))
        draws = int(self.draw_credit)
        self.draw_credit -= draws
        losses, gradients = [], []
        replay_rows = list(self.replay) if draws else []
        remaining = draws
        terminal_count = nonterminal_count = 0
        terminal_td = nonterminal_td = terminal_abs_td = nonterminal_abs_td = 0.
        while remaining:
            count = min(self.batch_size, len(self.replay), remaining)
            indices = self.rng.choice(len(self.replay), size=count, replace=False)
            batch = [replay_rows[int(index)] for index in indices]
            features = torch.as_tensor(np.stack([entry['features'] for entry in batch]),
                                       dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(np.stack([entry['mask'] for entry in batch]), device=self.device)
            actions = torch.as_tensor([entry['action'] for entry in batch], dtype=torch.long, device=self.device)
            q_values = self.actor.forward_q(features, mask).gather(1, actions.unsqueeze(1)).squeeze(1)
            targets = self._batch_targets(batch)
            loss = F.smooth_l1_loss(q_values, targets)
            residuals = (targets - q_values).detach()
            terminal = torch.tensor([entry['terminal'] for entry in batch], device=self.device)
            terminal_count += int(terminal.sum())
            nonterminal_count += int((~terminal).sum())
            terminal_td += float(residuals[terminal].sum())
            nonterminal_td += float(residuals[~terminal].sum())
            terminal_abs_td += float(residuals[terminal].abs().sum())
            nonterminal_abs_td += float(residuals[~terminal].abs().sum())
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            if not bool(torch.isfinite(gradient)):
                raise RuntimeError('non-finite IQL gradient')
            self.optimizer.step()
            # Target timescale is measured in sampled transitions, so a short
            # final batch must not count as a full target update.
            tau = (1. if self.target_tau == 1. else
                   -math.expm1(math.log1p(-self.target_tau) * count / self.batch_size))
            with torch.no_grad():
                for target, online in zip(self.target.parameters(), self.actor.parameters()):
                    target.lerp_(online, tau)
            self.updates += 1
            remaining -= count
            self.total_draws += count
            losses.append(float(loss.detach()) * count)
            gradients.append(float(gradient))
        elapsed = time.monotonic() - started
        used_epsilon = float(self.actor.exploration_epsilon) if self.stable else None
        self.completed_episodes = int(episode)
        if self.stable:
            fraction = min(1., self.completed_episodes / self.epsilon_decay_episodes)
            self.actor.exploration_epsilon.fill_(self.epsilon_initial + fraction * (self.epsilon_final - self.epsilon_initial))
        return dict(q_loss=sum(losses) / draws if draws else 0.,
            q_gradient_norm=float(np.mean(gradients)) if gradients else 0.,
            q_updates=self.updates, replay_transitions=len(self.replay),
            q_inserted_transitions=inserted, q_sampled_transitions=draws,
            q_sample_to_insert_ratio=draws / inserted if inserted else 0.,
            q_total_inserted_transitions=self.total_inserted, q_total_sampled_transitions=self.total_draws,
            q_draw_credit=float(self.draw_credit),
            q_new_transitions_evicted=max(0, inserted - self.replay.maxlen),
            q_terminal_samples=terminal_count, q_nonterminal_samples=nonterminal_count,
            q_terminal_td_mean=terminal_td / terminal_count if terminal_count else 0.,
            q_nonterminal_td_mean=nonterminal_td / nonterminal_count if nonterminal_count else 0.,
            q_terminal_td_abs_mean=terminal_abs_td / terminal_count if terminal_count else 0.,
            q_nonterminal_td_abs_mean=nonterminal_abs_td / nonterminal_count if nonterminal_count else 0.,
            q_target_tau=self.target_tau, q_target_tau_reference_samples=self.batch_size,
            q_implementation_profile=self.profile, q_exploration_epsilon=used_epsilon,
            q_forced_events_skipped=len(rows)-inserted if self.stable else 0,
            actor_update_seconds=elapsed, critic_update_seconds=0.,
            world_model_seconds=0., imagination_seconds=0., learning_seconds=elapsed)

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), target=self.target.state_dict(),
            optimizer=self.optimizer.state_dict(), replay=list(self.replay),
            rng=copy.deepcopy(self.rng.bit_generator.state), updates=self.updates,
            schedule_version=self.schedule_version, schedule_settings=self._schedule_settings(),
            reward_mode=self.reward_mode,
            draw_credit=(self.draw_credit.numerator, self.draw_credit.denominator),
            total_inserted=self.total_inserted, total_draws=self.total_draws,
            prior_weight=self.actor.prior_weight,
            implementation_profile=self.profile, completed_episodes=self.completed_episodes,
            prior_temperature_seconds=self.actor.prior_temperature_seconds)

    def load_state_dict(self, state):
        if state.get('implementation_profile', 'legacy_v1') != self.profile:
            raise ValueError('IQL implementation profile mismatch: start a fresh run')
        if state.get('reward_mode', 'global_only') != self.reward_mode:
            raise ValueError('IQL checkpoint reward mode mismatch: start a fresh run')
        if state.get('schedule_version') != self.schedule_version:
            raise ValueError('IQL checkpoint schedule mismatch: fixed-update checkpoints require a fresh run')
        if state.get('schedule_settings') != self._schedule_settings():
            raise ValueError('IQL checkpoint setting mismatch')
        credit = Fraction(*state['draw_credit'])
        if (not 0 <= credit < 1 or state['total_inserted'] < 0 or state['total_draws'] < 0
                or state['total_draws'] + credit !=
                state['total_inserted'] * Fraction(str(self.sample_to_insert_ratio))):
            raise ValueError('IQL checkpoint draw budget mismatch')
        self.actor.load_state_dict(state['actor'])
        self.target.load_state_dict(state['target'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.replay.clear()
        self.replay.extend(state['replay'])
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.updates = int(state['updates'])
        self.draw_credit = credit
        self.total_inserted = int(state['total_inserted'])
        self.total_draws = int(state['total_draws'])
        self.completed_episodes = int(state.get('completed_episodes', 0))
        self.actor.prior_weight = float(state['prior_weight'])
        self.actor.prior_temperature_seconds = float(state['prior_temperature_seconds'])


METHODS['iql'] = IndependentQMethod
