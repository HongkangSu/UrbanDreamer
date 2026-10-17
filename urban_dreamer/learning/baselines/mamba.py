"""Routing adaptation of AAMAS 2022 MAMBA's discrete RSSM and agent attention.

Model and critic rewards use the shared full-network window cost. An optional
actor-only penalty applies to individual nonarrivals at the real deadline. Communication is
limited to 32-agent chunks; global context/reward is never averaged over a
sampled chunk. Realized identities are not reused inside an episode. A separate
global RSSM models spatial traffic conditioned on a fixed physical start cohort.
Agent RSSMs follow real decision clocks; a shared physical stepper determines
legal events and arrivals in imagination. See mamba_core/README.md for provenance.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import replace
import math
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from ...models.policy import (apply_shortest_path_prior, individual_actor_advantage,
    set_shortest_path_prior)
from ...traffic.environment import safe_microstep_seconds
from ...workflow.config import NONARRIVAL_PENALTY
from ..global_reward import causal_context
from .common import reward_objective, reward_context_dim, window_accounting
from ..imagination import advance_imagined_window, clip_road_speed_fractions
from ..observations import build_actor_observations, prepare_actor_window, selected_decisions
from . import METHODS
from .mamba_core.config import MambaConfig
from .mamba_core.networks import MambaActor, MambaCritic, MambaWorldModel, RSSMState

SCHEMA = 'routing_mamba_physical_cohort_v2'
CHUNK = 32
REPLAY_EPISODES = 8
WORLD_UPDATES = 4
SEQUENCE_LENGTH = 8
IMAGINATION_HORIZON = 8
PPO_EPOCHS = 4
TARGET_POLICY_KL = .01


def clipped_policy_loss(log_prob, old_log_prob, advantage):
    # Clip the selected surrogate branch before exp: computing both branches
    # first can create inf * 0 or NaN gradients even when PPO clips the result.
    advantage = advantage.detach()
    log_ratio = log_prob - old_log_prob.detach()
    log_ratio = torch.where(advantage >= 0., log_ratio.clamp_max(math.log1p(.075)),
        log_ratio.clamp_min(math.log1p(-.075)))
    return -(log_ratio.exp() * advantage).mean()


def _kl(prior, posterior):
    """Balanced categorical KL, as in the vendored MAMBA learner."""
    p = posterior.logits.softmax(-1)
    q = prior.logits.softmax(-1)
    def divergence(left, right):
        return (left * (left.clamp_min(1e-8).log() - right.clamp_min(1e-8).log())).sum(-1).mean()
    return .2 * divergence(p, q.detach()) + .8 * divergence(p.detach(), q)


def _take(state, indices):
    return state.map(lambda tensor: tensor[:, indices])


def _replace_state(state, indices, update):
    for name in ('logits', 'stoch', 'deter'):
        getattr(state, name)[:, indices] = getattr(update, name)


class RoutingMambaActor(MambaActor):
    """Recurrent latent actor with the same candidate prior as other methods."""
    def __init__(self, cfg):
        super().__init__(cfg)
        set_shortest_path_prior(self, 0., 30.)

    def routing_distribution(self, latent, candidates, mask, *, weight=None):
        logits = self.logits(latent, mask)
        logits = apply_shortest_path_prior(logits, candidates, mask,
            weight=self.prior_weight if weight is None else float(weight),
            temperature_seconds=self.prior_temperature_seconds)
        return Categorical(logits=logits)


class MambaMethod:
    entropy_coef = .002
    needs_window_records = True
    needs_row_flow = False

    def __init__(self, args, runner):
        self.args, self.runner, self.device = args, runner, runner.device
        target_kl = float(getattr(args, 'mamba_target_kl', TARGET_POLICY_KL))
        if not np.isfinite(target_kl) or target_kl < 0.:
            raise ValueError('mamba_target_kl must be finite and nonnegative')
        self.route_mode = getattr(runner.candidates, 'routing_mode', 'next_hop') == 'current_to_dest'
        self.width = int(runner.candidates.max_candidates)
        self.local_dim = self.width * 11
        self.cfg = MambaConfig(obs_dim=self.local_dim, action_dim=self.width,
            device=str(self.device), seed=args.seed, transformer_dropout=0.,
            imagination_reward_scope='team', gamma=1., max_agents_per_chunk=CHUNK)
        self.world_model = MambaWorldModel(self.cfg).to(self.device)
        self.edge_count = int(runner.network.edge_count)
        self.global_dim = 10 * self.edge_count + reward_context_dim(args)
        self.global_model = MambaWorldModel(replace(self.cfg, obs_dim=self.global_dim,
            action_dim=2 * self.edge_count + 1)).to(self.device)
        self.actor = RoutingMambaActor(self.cfg).to(self.device)
        self.critic = MambaCritic(self.cfg).to(self.device)
        self.model_optimizer = torch.optim.Adam(list(self.world_model.parameters())
            + list(self.global_model.parameters()), lr=self.cfg.model_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.replay = deque(maxlen=REPLAY_EPISODES)
        self.rng = np.random.default_rng(args.seed + 932407)
        self.slots = {}
        self._events = []
        self.model_updates = 0
        self.local_model_updates = 0
        self.model_frames = 0
        self.routing_frames = 0
        self.warmup_updates = int(getattr(args, 'mamba_model_warmup_updates', 100))
        self.min_model_frames = int(getattr(args, 'mamba_min_model_frames', 100))
        self.burn_in_windows = int(getattr(args, 'mamba_burn_in_windows', 5))
        if min(self.warmup_updates, self.min_model_frames, self.burn_in_windows) < 0:
            raise ValueError('MAMBA warm-up and burn-in counts must be nonnegative')

    def _tensor(self, value, *, dtype=torch.float32):
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def begin_episode(self, topology, *, seed):
        """Reset recurrent memory for every independent realized demand draw."""
        self.slots = {vehicle.vehicle_id: i for i, vehicle in enumerate(topology.vehicles)}
        if len(self.slots) != len(topology.vehicles):
            raise ValueError('MAMBA requires unique realized vehicle identities')
        self.population = len(topology.vehicles)
        self._latent = self.world_model.initial_state(1, max(self.population, 1), device=self.device)
        self._previous_action = torch.zeros(1, max(self.population, 1), self.width, device=self.device)
        self._events = []

    def _observation(self, candidates, mask):
        candidates = np.asarray(candidates, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if candidates.shape != (*mask.shape, 10) or mask.shape[-1] != self.width:
            raise ValueError('MAMBA candidate observation width changed')
        return np.concatenate((candidates.reshape(len(mask), -1), mask.astype(np.float32)), axis=-1)

    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=True, temperature=1., sample_actions=None):
        if not vehicles:
            return [], []
        if any(v.get('arrived', False) or not v.get('departed', True) for v in vehicles):
            raise ValueError('arrived or undeployed vehicles cannot act in MAMBA')
        slots = np.asarray([self.slots[v['vehicle_id']] for v in vehicles], dtype=np.int64)
        observations = self._observation(actor_obs.features, actor_obs.mask)
        output, aux = [], []
        with torch.no_grad():
            for start in range(0, len(slots), CHUNK):
                chosen = slots[start:start + CHUNK]
                obs = self._tensor(observations[start:start + CHUNK])[None]
                present = torch.ones(1, len(chosen), dtype=torch.bool, device=self.device)
                _, post = self.world_model.observe(obs, self._previous_action[:, chosen],
                    _take(self._latent, chosen), present, present, sample=explore)
                features = self._tensor(actor_obs.features[start:start + CHUNK])
                mask = self._tensor(actor_obs.mask[start:start + CHUNK], dtype=torch.bool)
                latent = post.features[0]
                distribution = self.actor.routing_distribution(latent, features, mask)
                if isinstance(temperature, bool) or not np.isfinite(temperature) or temperature <= 0:
                    raise ValueError("action temperature must be finite and positive")
                if temperature != 1.:
                    distribution = Categorical(logits=distribution.logits / temperature)
                sample = explore if sample_actions is None else sample_actions
                action = distribution.sample() if sample else distribution.logits.argmax(-1)
                log_prob, entropy = distribution.log_prob(action), distribution.entropy()
                _replace_state(self._latent, chosen, post)
                self._previous_action[:, chosen] = F.one_hot(action, self.width).float()[None]
                # No stale posterior reservoir: imagination reconstructs states
                # from raw event observations under the updated model weights.
                packed = torch.cat((action.float(), entropy, log_prob)).cpu().numpy()
                count = len(chosen)
                output.extend(packed[:count].astype(np.int64).tolist())
                aux.extend(dict(entropy=float(e), old_log_prob=float(p),
                    sp_prior_weight=float(self.actor.prior_weight))
                    for e, p in zip(packed[count:2 * count], packed[2 * count:3 * count]))
        # All events remain in the replay, not merely the last action per window.
        self._events.append(dict(time=float(now), slots=slots.copy(), obs=observations,
            action=np.asarray(output, dtype=np.int64), mask=actor_obs.mask.copy()))
        return output, aux

    def _global_observation(self, window, population, cohort):
        """Spatial traffic and a fixed cohort with causal cumulative accounting."""
        flow = np.asarray(window['flow'], dtype=np.float32)
        occupancy = np.zeros((2, self.edge_count), dtype=np.float32)
        started_count = (window_accounting(window)[0] if reward_objective(self.args) != 'legacy_exposure'
                         else int(population))
        denominator = max(int(population) if reward_objective(self.args) == 'legacy_exposure' else started_count, 1)
        for vehicle in window.get('vehicles', ()):
            if (str(vehicle['vehicle_id']) not in cohort or
                    not vehicle.get('departed', True) or vehicle.get('arrived', False)):
                continue
            occupancy[0, int(vehicle['current_edge'])] += 1. / denominator
            occupancy[1, int(vehicle['destination_edge'])] += 1. / denominator
        return np.concatenate((flow.reshape(-1), occupancy.reshape(-1),
            (np.asarray([window['time'] / self.runner.deadline, np.log1p(population)], dtype=np.float32)
             if reward_objective(self.args) == 'legacy_exposure' else np.asarray(window['context'], dtype=np.float32))))

    @staticmethod
    def _cohort(window):
        return frozenset(str(v['vehicle_id']) for v in window['vehicles']
            if v.get('departed', True) and not v.get('arrived', False))

    def _cohort_action(self, rows, population, duration):
        action = np.zeros(2 * self.edge_count + 1, dtype=np.float32)
        for row in rows:
            action[int(row['current_edge'])] += 1. / max(population, 1)
            edge = int(row['edge_indices'][int(row['action'])])
            action[self.edge_count + edge] += 1. / max(population, 1)
        action[-1] = duration / 30.
        return action

    def _cohort_turn_action(self, events, population, duration, cohort):
        """Keep route policy slots distinct from the observed physical turns."""
        action = np.zeros(2 * self.edge_count + 1, dtype=np.float32)
        for event in events:
            if str(event.vehicle_id) in cohort:
                action[int(event.current_edge)] += 1. / max(population, 1)
                action[self.edge_count + int(event.next_edge)] += 1. / max(population, 1)
        action[-1] = duration / 30.
        return action

    def _pack_episode(self, rows, windows, episode):
        population = int(windows[0].get('total_vehicle_count', self.population))
        if population != self.population:
            raise ValueError('MAMBA realized population changed within the episode')
        reward = np.asarray([float(w['reward']) for w in windows], dtype=np.float32)
        # Source shared labels are checked, never replaced with local returns.
        expected = [-(float(w['next_cumulative_vehicle_seconds']) / max(int(w['next_started_vehicle_count']), 1)
                      - float(w['cumulative_vehicle_seconds']) / max(int(w['started_vehicle_count']), 1)) / 60.
                    if reward_objective(self.args) != 'legacy_exposure' and 'next_started_vehicle_count' in w
                    else float(w['reward']) for w in windows]
        if not np.allclose(reward, expected, atol=1e-6):
            raise ValueError('MAMBA window reward is not the exact shared global objective')
        times = np.asarray([w['time'] for w in windows], dtype=np.float64)
        decisions = [[] for _ in windows]
        for row in rows:
            if row['vehicle_id'] not in self.slots:
                raise ValueError('MAMBA event references an unknown realized vehicle')
            decisions[int(row['window'])].append(dict(vehicle_id=str(row['vehicle_id']),
                current_edge=int(row['current_edge']), action=int(row['action']),
                edge_indices=np.asarray(row['edge_indices']).copy()))
        if sum(len(event['slots']) for event in self._events) != len(rows):
            raise ValueError('MAMBA requires all actual collected events, including repeated decisions')
        return dict(episode=int(episode), population=population,
            windows=copy.deepcopy(windows), decisions=decisions,
            reward=reward, terminal=np.asarray([w['terminal'] for w in windows], dtype=bool),
            times=times, slots=copy.deepcopy(self.slots), events=copy.deepcopy(self._events))

    def _step(self, optimizer, loss, parameters):
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite MAMBA loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(parameters, 10.)
        if not torch.isfinite(norm):
            raise RuntimeError('non-finite MAMBA gradient')
        optimizer.step()

    def _global_sequence(self, data, start, *, train, cohort=None):
        stop = min(len(data['reward']), start + SEQUENCE_LENGTH)
        cohort = self._cohort(data['windows'][start]) if cohort is None else cohort
        begin = max(0, start - self.burn_in_windows)
        state = self.global_model.initial_state(1, 1, device=self.device)
        previous = torch.zeros(1, 1, 2 * self.edge_count + 1, device=self.device)
        presence = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        states, losses = [], []
        for t in range(begin, stop):
            window = data['windows'][t]
            duration = float(window.get('end_time', window['time'] + 30.)) - window['time']
            action = self._tensor(self._cohort_turn_action(window['action_events'],
                data['population'], duration, cohort) if self.route_mode else self._cohort_action(
                [r for r in data['decisions'][t] if r['vehicle_id'] in cohort],
                data['population'], duration))[None, None]
            # The selected start cohort remains identical in burn-in and loss
            # windows; background/new arrivals are never explicit actions.
            with torch.set_grad_enabled(train and t >= start):
                observation = self._tensor(self._global_observation(window, data['population'], cohort))[None, None]
                prior, posterior = self.global_model.observe(observation, previous, state, presence, presence)
                if t >= start:
                    consequence = self.global_model.transition(action, posterior, presence)
                    feature = torch.cat((posterior.stoch, consequence.deter), -1)
                    prediction = self.global_model.reward_model(feature).squeeze()
                    continuation = self.global_model.continuation_model(feature).squeeze()
                    reward_loss = F.mse_loss(prediction, self._tensor(data['reward'][t]))
                    continuation_loss = F.binary_cross_entropy_with_logits(continuation,
                        self._tensor(float(not data['terminal'][t])))
                    reconstruction = F.smooth_l1_loss(self.global_model.observation_decoder(posterior.features), observation)
                    losses.append(reward_loss + continuation_loss + reconstruction + .1 * _kl(prior, posterior))
                    states.append(posterior)
                state, previous = posterior, action
        return torch.stack(losses).mean(), states, start, stop

    def _agent_sequence_loss(self, data):
        events = data['events']
        if not events:
            return next(self.world_model.parameters()).sum() * 0.
        start = int(self.rng.integers(len(events)))
        picked = events[start:min(len(events), start + 32)]
        # Loss sampling is by event batches. Prefix events from physical macro
        # windows warm recurrent states without gradient; no invented 30s step.
        first_window = max(0, int(np.searchsorted(data['times'], picked[0]['time'], side='right') - 1))
        boundary = float(data['times'][max(0, first_window - self.burn_in_windows)])
        prefix = [event for event in events[:start] if event['time'] >= boundary]
        slots = self._sample_agent_slots(picked)
        state = self.world_model.initial_state(1, len(slots), device=self.device)
        previous = torch.zeros(1, len(slots), self.width, device=self.device)
        local_ids = {int(slot): i for i, slot in enumerate(slots)}
        losses = []
        for event in prefix + picked:
            obs = np.zeros((len(slots), self.local_dim), dtype=np.float32)
            valid = np.zeros(len(slots), dtype=bool)
            actions = np.zeros(len(slots), dtype=np.int64)
            masks = np.zeros((len(slots), self.width), dtype=bool)
            for i, slot in enumerate(event['slots']):
                j = local_ids.get(int(slot))
                if j is not None:
                    obs[j], actions[j], masks[j], valid[j] = event['obs'][i], event['action'][i], event['mask'][i], True
            if not valid.any():
                continue
            observed = self._tensor(valid, dtype=torch.bool)[None]
            observation = self._tensor(obs)[None]
            # Event clocks are individual: a vehicle not acting at this time
            # retains EXACTLY its previous recurrent state, as during select().
            warming = event['time'] < picked[0]['time']
            with torch.set_grad_enabled(not warming):
                prior, post = self._observe_agent_event(observation, previous, state, observed)
            if warming:
                onehot = F.one_hot(self._tensor(actions, dtype=torch.long), self.width).float()[None]
                previous = torch.where(observed[..., None], onehot, previous)
                state = post.detach()
                continue
            reconstruction = F.smooth_l1_loss(self.world_model.observation_decoder(post.features)[observed],
                observation[observed])
            availability = F.binary_cross_entropy_with_logits(self.world_model.availability_model(post.features)[observed],
                self._tensor(masks)[None][observed])
            action = self._tensor(actions, dtype=torch.long)
            onehot = F.one_hot(action, self.width).float()[None]
            consequence = self.world_model.transition(onehot, post, observed)
            information = F.cross_entropy(self.world_model.action_information_model(consequence.features)[observed], action[observed[0]])
            # Arrival is decided by identity-preserving physical motion, never
            # by a macro-boundary label attached to an event-clock transition.
            active_indices = torch.nonzero(observed[0], as_tuple=False).flatten()
            losses.append(reconstruction + availability + information
                + .1 * _kl(_take(prior, active_indices), _take(post, active_indices)))
            previous = torch.where(observed[..., None], onehot, previous)
            state = post
        return torch.stack(losses).mean()

    def _sample_agent_slots(self, events):
        slots = np.unique(np.concatenate([event['slots'] for event in events]))
        if len(slots) > CHUNK:
            slots = np.sort(self.rng.choice(slots, size=CHUNK, replace=False))
        return slots

    def _observe_agent_event(self, observation, previous, state, observed):
        prior, posterior = self.world_model.observe(observation, previous, state, observed, observed)
        keep = observed.unsqueeze(-1)
        posterior = RSSMState(
            torch.where(keep.unsqueeze(-1), posterior.logits, state.logits),
            torch.where(keep, posterior.stoch, state.stoch),
            torch.where(keep, posterior.deter, state.deter))
        return prior, posterior

    def _global_returns(self, data):
        # Shared undiscounted reward-to-go extends through the actual episode END,
        # including all windows after an individual routing agent has arrived.
        return np.cumsum(data['reward'][::-1], dtype=np.float64)[::-1].copy().astype(np.float32)

    def _train_real_critic(self, data, states, start, stop):
        features = torch.cat([state.features.detach() for state in states], dim=0)
        predictions = self.critic(features).reshape(-1)
        targets = self._tensor(self._global_returns(data)[start:stop])
        loss = F.smooth_l1_loss(predictions, targets)
        self._step(self.critic_optimizer, loss, list(self.critic.parameters()))
        return float(loss.detach())

    def _reconstruct_local(self, data, window_index, cohort):
        """Re-encode raw prefix events at the same macro boundary as global state."""
        slots = {vehicle_id: i for i, vehicle_id in enumerate(sorted(cohort))}
        state = self.world_model.initial_state(1, max(len(slots), 1), device=self.device)
        previous = torch.zeros(1, max(len(slots), 1), self.width, device=self.device)
        timestamp = float(data['times'][window_index])
        lower = float(data['times'][max(0, window_index - self.burn_in_windows)])
        old_to_new = {data['slots'][vehicle_id]: slot for vehicle_id, slot in slots.items()}
        with torch.no_grad():
            for event in data['events']:
                if not lower <= event['time'] < timestamp:
                    continue
                selected = [i for i, slot in enumerate(event['slots']) if int(slot) in old_to_new]
                for begin in range(0, len(selected), CHUNK):
                    indices = selected[begin:begin + CHUNK]
                    chosen = [old_to_new[int(event['slots'][i])] for i in indices]
                    valid = torch.ones(1, len(chosen), dtype=torch.bool, device=self.device)
                    _, post = self.world_model.observe(self._tensor(event['obs'][indices])[None],
                        previous[:, chosen], _take(state, chosen), valid, valid, sample=False)
                    _replace_state(state, chosen, post)
                    previous[:, chosen] = F.one_hot(self._tensor(event['action'][indices],
                        dtype=torch.long), self.width).float()[None]
        return slots, state, previous

    def _imagined_rollout(self, data, global_state, window_index, cohort):
        """Physical decision events coupled to a matching macro traffic RSSM."""
        window = data['windows'][window_index]
        vehicles = copy.deepcopy([v for v in window['vehicles'] if str(v['vehicle_id']) in cohort])
        slots, local_state, previous = self._reconstruct_local(data, window_index, cohort)
        timestamp = float(window['time'])
        remaining = max(float(self.runner.deadline) - timestamp, 0.)
        horizon = min(IMAGINATION_HORIZON, int(np.ceil(remaining / 30.)))
        population = data['population']
        started_count, cumulative_seconds = (window_accounting(window)
            if reward_objective(self.args) != 'legacy_exposure' else (population, 0.))
        simulation = self.runner.config.simulation
        traffic_model = simulation.traffic_model
        maximum_speed = float(simulation.vehicle_max_speed_mps)
        network = self.runner.network
        lengths = tuple(network.external_lane_lengths_m.values()) or network.lengths_m[network.passenger_allowed]
        observation_step = (1. if self.route_mode else float(self.args.step_seconds) if traffic_model == 'meso'
            else safe_microstep_seconds(lengths, maximum_speed))
        presence = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        records, rewards, features, continuation = [], [], [], []
        with torch.no_grad():
            for step in range(horizon):
                features.append(global_state.features.detach())
                decoded = self.global_model.observation_decoder(global_state.features)[0, 0]
                flow = decoded[:8 * self.edge_count].reshape(self.edge_count, 8).cpu().numpy()
                actor_window = prepare_actor_window(flow[:, [2, 1]], self.edge_count)
                duration = min(30., self.runner.deadline - timestamp)
                rows = []

                def decide(pending, frozen, now):
                    obs = build_actor_observations(policy_edge_features=actor_window,
                        vehicle_states=pending, candidates=self.runner.candidates,
                        episode_progress=min((now if reward_objective(self.args) != 'legacy_exposure'
                            else timestamp) / self.runner.deadline, 1.),
                        active_vehicle_fraction=sum(not v.get('arrived', False) for v in vehicles) / max(started_count, 1))
                    selected = []
                    raw_obs = self._observation(obs.features, obs.mask)
                    for begin in range(0, len(pending), CHUNK):
                        chunk = pending[begin:begin + CHUNK]
                        chosen = [slots[str(v['vehicle_id'])] for v in chunk]
                        valid = torch.ones(1, len(chunk), dtype=torch.bool, device=self.device)
                        _, post = self.world_model.observe(self._tensor(raw_obs[begin:begin + CHUNK])[None],
                            previous[:, chosen], _take(local_state, chosen), valid, valid)
                        candidates = self._tensor(obs.features[begin:begin + CHUNK])
                        mask = self._tensor(obs.mask[begin:begin + CHUNK], dtype=torch.bool)
                        distribution = self.actor.routing_distribution(post.features[0], candidates, mask)
                        action = distribution.sample()
                        records.append(dict(latent=post.features[0].detach(), candidates=candidates,
                            mask=mask, action=action, old_log_prob=distribution.log_prob(action), step=step,
                            time=float(now), vehicle_ids=tuple(str(v['vehicle_id']) for v in chunk)))
                        if obs.routes is not None:
                            records[-1]['candidate_routes'] = obs.routes[begin:begin + CHUNK]
                        _replace_state(local_state, chosen, post)
                        previous[:, chosen] = F.one_hot(action, self.width).float()[None]
                        for offset, (vehicle, index) in enumerate(zip(chunk, action.cpu().tolist())):
                            row_index = begin + offset
                            row = dict(vehicle_id=str(vehicle['vehicle_id']), current_edge=int(vehicle['current_edge']),
                                action=index, edge_indices=obs.edge_indices[row_index].copy())
                            rows.append(row)
                            selected.append(index)
                    return selected_decisions(pending, obs, selected, now)

                physical_result = None
                if vehicles and any(not v.get('arrived', False) for v in vehicles):
                    speeds = clip_road_speed_fractions(flow[:, 1], traffic_model) * network.speed_limits_mps
                    physical_result = advance_imagined_window(vehicles, candidates=self.runner.candidates,
                        edge_speeds_mps=np.minimum(speeds, maximum_speed), frozen_flow=flow,
                        start_time=timestamp, total_vehicle_count=population, decide=decide,
                        interval_seconds=duration, observation_step_seconds=observation_step,
                        vehicle_max_speed_mps=maximum_speed, traffic_model=traffic_model,
                        session_protocol='event' if traffic_model == 'meso' or self.route_mode else 'legacy')
                    cumulative_seconds += float(getattr(physical_result, 'active_vehicle_seconds', 0.) if reward_objective(self.args) == 'legacy_exposure'
                                                else physical_result.active_vehicle_seconds)
                action = self._tensor(self._cohort_turn_action(
                    physical_result.action_events if physical_result is not None else (),
                    population, duration, cohort) if self.route_mode else
                    self._cohort_action(rows, population, duration))[None, None]
                next_global = self.global_model.transition(action, global_state, presence)
                consequence = torch.cat((global_state.stoch, next_global.deter), -1)
                # Duration is an explicit model input in both real and imagined
                # windows. Reward already predicts that whole interval: no double scaling.
                rewards.append(self.global_model.reward_model(consequence).reshape(()))
                timestamp += duration
                # The real simulation deadline is known. Learned survival must
                # not discount whole-network rewards before that boundary.
                continuation.append(torch.as_tensor(float(timestamp < self.runner.deadline), device=self.device))
                global_state = next_global
            running = (torch.zeros((), device=self.device) if timestamp >= self.runner.deadline else
                self.critic(global_state.features).reshape(()))
            returns = []
            for reward, discount in zip(reversed(rewards), reversed(continuation)):
                running = reward + discount * running
                returns.append(running)
            returns = torch.stack(list(reversed(returns))) if returns else torch.empty(0, device=self.device)
            if not getattr(self.args, 'global_reward_only', True):
                # A short imagination horizon is a bootstrap boundary, not a
                # nonarrival. Only the actual physical deadline can fail agents.
                failed = (frozenset(str(v['vehicle_id']) for v in vehicles
                    if not v.get('arrived', False)) if timestamp >= self.runner.deadline else frozenset())
                for record in records:
                    record['timeout_returns'] = self._tensor([
                        NONARRIVAL_PENALTY if vehicle_id in failed else 0.
                        for vehicle_id in record['vehicle_ids']])
        return records, features, returns

    def _train_imagination(self, data):
        starts = [i for i, window in enumerate(data['windows']) if self._cohort(window)]
        if not starts:
            return 0., 0., 0
        wi = int(self.rng.choice(starts))
        cohort = self._cohort(data['windows'][wi])
        with torch.no_grad():
            _, states, _, _ = self._global_sequence(data, wi, train=False, cohort=cohort)
            initial = states[0].detach()
        records, global_features, returns = self._imagined_rollout(data, initial, wi, cohort)
        if not global_features:
            return 0., 0., 0
        features = torch.cat(global_features, dim=0)
        values = self.critic(features).reshape(-1)
        advantages = (returns - values.detach()).detach()
        if getattr(self.args, 'global_reward_only', True):
            advantages = advantages / advantages.std(unbiased=False).clamp_min(1.)
        critic_loss = F.smooth_l1_loss(values, returns)
        self._step(self.critic_optimizer, critic_loss, list(self.critic.parameters()))
        if not records:
            return 0., float(critic_loss.detach()), 0
        actor_loss = self._ppo_update(records, advantages)
        return float(actor_loss.detach()), float(critic_loss.detach()), sum(len(r['action']) for r in records)

    def _ppo_update(self, records, advantages):
        """Frozen imagined behavior is reused for meaningful clipped PPO updates."""
        shared = torch.cat([advantages[record['step']].detach().expand(len(record['action']))
            for record in records])
        if getattr(self.args, 'global_reward_only', True):
            actor_advantages = shared.clamp(-2.5, 2.5)
        else:
            timeout_returns = torch.cat([record['timeout_returns'] for record in records])
            actor_advantages = individual_actor_advantage(shared, timeout_returns)
        offset = 0
        for record in records:
            count = len(record['action'])
            record['actor_advantage'] = actor_advantages[offset:offset + count].detach()
            offset += count
        choices = []
        for record in records:
            choice = (record['mask'].sum(-1) > 1 if record['mask'] is not None else
                torch.ones_like(record['action'], dtype=torch.bool))
            count = int(choice.sum())
            if count:
                choices.append((record, choice, count))
        has_signal = choices and (self.entropy_coef != 0. or any(
            bool(record['actor_advantage'][choice].ne(0.).any()) for record, choice, _ in choices))
        if not has_signal:
            # Zero gradients still move Adam parameters through stored momentum.
            self.actor_optimizer.zero_grad(set_to_none=True)
            return next(self.actor.parameters()).new_zeros(())
        choice_count = sum(count for _, _, count in choices)
        target_kl = float(getattr(self.args, 'mamba_target_kl', TARGET_POLICY_KL))
        frozen_prior = float(self.actor.prior_weight)
        for _ in range(PPO_EPOCHS):
            losses, log_ratios = [], []
            for record, choice, count in choices:
                distribution = self.actor.routing_distribution(record['latent'], record['candidates'],
                    record['mask'], weight=frozen_prior)
                log_prob = distribution.log_prob(record['action'])[choice]
                old_log_prob = record['old_log_prob'][choice]
                advantage = record['actor_advantage'][choice]
                losses.append((clipped_policy_loss(log_prob, old_log_prob, advantage)
                    - self.entropy_coef * distribution.entropy()[choice].mean()) * count)
                log_ratios.append(log_prob.detach() - old_log_prob.detach())
            # Chunks are a memory bound, not equal-weight training samples.
            actor_loss = torch.stack(losses).sum() / choice_count
            log_ratio = torch.cat(log_ratios)
            ratio = log_ratio.exp()
            self._last_ppo_ratios = ratio.cpu()
            self._last_ppo_kl = float((ratio - 1. - log_ratio).mean())
            if target_kl > 0. and self._last_ppo_kl > 1.5 * target_kl:
                break
            self._step(self.actor_optimizer, actor_loss, list(self.actor.parameters()))
        return actor_loss

    def on_episode(self, rows, windows, *, episode):
        started = time.monotonic()
        if not windows:
            return dict(world_model_loss=0., actor_loss=0., critic_loss=0., learning_seconds=0.)
        data = self._pack_episode(rows, windows, episode)
        self.replay.append(data)
        self.model_frames += len(windows)
        self.routing_frames += sum(bool(self._cohort(window)) for window in windows)
        model_started = time.monotonic()
        model_losses, critic_losses = [], []
        for _ in range(WORLD_UPDATES):
            sampled = self.replay[int(self.rng.integers(len(self.replay)))]
            start = int(self.rng.integers(len(sampled['reward'])))
            global_loss, states, start, stop = self._global_sequence(sampled, start, train=True)
            local_loss = self._agent_sequence_loss(sampled)
            loss = global_loss + local_loss
            self._step(self.model_optimizer, loss, list(self.world_model.parameters()) + list(self.global_model.parameters()))
            self.model_updates += 1
            self.local_model_updates += int(bool(sampled['events']))
            model_losses.append(float(loss.detach()))
            critic_losses.append(self._train_real_critic(sampled, states, start, stop))
        world_seconds = time.monotonic() - model_started
        imagined_started = time.monotonic()
        ready = self.local_model_updates >= self.warmup_updates and self.routing_frames >= self.min_model_frames
        actor_loss, imag_critic_loss, events = self._train_imagination(data) if ready else (0., 0., 0)
        return dict(world_model_loss=float(np.mean(model_losses)), actor_loss=actor_loss,
            critic_loss=float(np.mean(critic_losses)), imagined_critic_loss=imag_critic_loss,
            imagined_events=events, real_event_count=sum(len(e['slots']) for e in data['events']),
            model_updates=self.model_updates, model_frames=self.model_frames, model_warmup_complete=int(ready),
            local_model_updates=self.local_model_updates, routing_frames=self.routing_frames,
            world_model_seconds=world_seconds, imagination_seconds=time.monotonic() - imagined_started,
            learning_seconds=time.monotonic() - started)

    def learning_protocol(self):
        cfg = self.cfg.to_dict()
        cfg.pop('device')  # checkpoint transport between CPU/GPU is not a learning change
        protocol = dict(schema=SCHEMA, local_clock='physical_decision_event', global_clock='macro_window',
            global_observation=('per_road_flow_fixed_cohort_occupancy_time_population'
                if reward_objective(self.args) == 'legacy_exposure' else
                'per_road_flow_fixed_cohort_occupancy_causal_time_and_cumulative_accounting'),
            global_action='fixed_cohort_source_and_successor_counts_over_actual_N_and_duration',
            global_reward='whole_network_team_tail', cohort='all_active_at_seed_boundary',
            arrivals='physical', model_warmup_updates=self.warmup_updates,
            warmup_update_scope='model_updates_with_real_local_decision_events',
            min_model_frames=self.min_model_frames, min_frames_scope='nonempty_start_cohort',
            burn_in_windows=self.burn_in_windows, actor_lr=float(self.args.actor_lr),
            critic_lr=float(self.args.critic_lr), cfg=cfg,
            world_updates=WORLD_UPDATES, replay_episodes=REPLAY_EPISODES,
            sequence_length=SEQUENCE_LENGTH, local_event_sequence_length=32,
            imagination_horizon=IMAGINATION_HORIZON, ppo_epochs=PPO_EPOCHS,
            # Keep the historical base in the state schema; the outer config
            # and restored episode counter determine self.entropy_coef.
            ppo_clip=.075, ppo_target_kl=float(getattr(self.args, 'mamba_target_kl', TARGET_POLICY_KL)),
            ppo_kl_stop_multiplier=1.5, ppo_reduction='nonforced_individual_decisions',
            actor_zero_signal='skip_adam_without_choice_advantage_or_entropy',
            imagined_actor_context=('macro_window_time' if reward_objective(self.args) == 'legacy_exposure'
                else 'physical_decision_event_time'),
            imagination_continuation='known_physical_deadline',
            entropy_coef=.002, gradient_clip=10., advantage_clip=2.5,
            chunk_size=CHUNK, global_obs_dim=self.global_dim, global_action_dim=2 * self.edge_count + 1)
        if self.route_mode:
            protocol.update(routing_mode='current_to_dest', local_clock='route_decision_30_seconds',
                global_action='fixed_cohort_observed_branch_turn_counts_over_actual_N_and_duration')
        if not getattr(self.args, 'global_reward_only', True):
            protocol.update(reward_mode='global_with_individual_timeout',
                actor_timeout='physical_nonarrival_at_real_deadline_only',
                nonarrival_penalty=float(NONARRIVAL_PENALTY),
                actor_advantage_scale='shared_event_advantages_before_individual_timeout')
        return protocol

    def state_dict(self):
        result = dict(schema=SCHEMA, protocol=self.learning_protocol(), cfg=self.cfg.to_dict(),
            model_updates=self.model_updates, model_frames=self.model_frames, routing_frames=self.routing_frames,
            local_model_updates=self.local_model_updates,
            replay=copy.deepcopy(list(self.replay)),
            rng=copy.deepcopy(self.rng.bit_generator.state), torch_rng=torch.get_rng_state(),
            prior_weight=float(self.actor.prior_weight), prior_temperature=float(self.actor.prior_temperature_seconds))
        for name in ('world_model', 'global_model', 'actor', 'critic',
                     'model_optimizer', 'actor_optimizer', 'critic_optimizer'):
            result[name] = getattr(self, name).state_dict()
        if torch.cuda.is_available():
            result['cuda_rng'] = torch.cuda.get_rng_state_all()
        return result

    def load_state_dict(self, state):
        if (state.get('schema') != SCHEMA or state.get('protocol') != self.learning_protocol()
                or state['cfg']['obs_dim'] != self.local_dim):
            raise ValueError('incompatible routing MAMBA checkpoint: physical cohort v2 requires a fresh run')
        for name in ('world_model', 'global_model', 'actor', 'critic',
                     'model_optimizer', 'actor_optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        self.replay = deque(copy.deepcopy(state['replay']), maxlen=REPLAY_EPISODES)
        self.model_updates, self.model_frames = int(state['model_updates']), int(state['model_frames'])
        self.routing_frames = int(state['routing_frames'])
        self.local_model_updates = int(state['local_model_updates'])
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if 'cuda_rng' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state['cuda_rng']])
        set_shortest_path_prior(self.actor, state['prior_weight'], state['prior_temperature'])


METHODS['mamba'] = MambaMethod
