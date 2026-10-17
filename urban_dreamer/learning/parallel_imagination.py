"""Persistent CPU event windows with centralized, batched differentiable learning.

Workers never own CUDA objects. Each task has a private rollout RNG state, so
scheduling cannot change sampling; all policy/critic parameters stay fixed for
an entire collection pass. The parent alone creates backward graphs.
"""
from __future__ import annotations

import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .action_events import pack_action_events
from .imagination import advance_imagined_window, clip_road_speed_fractions
from .observations import build_actor_observations, prepare_actor_window, selected_decisions
from ..models.event_critic import (EventActionCritic, build_event_q_inputs,
                                  detached_counterfactual_baseline)
from ..models.policy import NextHopActor, lambda_return, set_shortest_path_prior
from ..models.world_model import EventActionEncoder
from .global_reward import causal_context
from .critic_budget import remaining_cost_budget

_WORKER = None


def _numpy_state(module):
    return {name: value.detach().cpu().numpy().copy() for name, value in module.state_dict().items()}


def _load_numpy(module, state):
    module.load_state_dict({name: torch.from_numpy(value) for name, value in state.items()}, strict=True)


def _initialize_worker(candidates, adjacency, static):
    global _WORKER
    torch.set_num_threads(1)
    actor = NextHopActor().eval().requires_grad_(False)
    critic = EventActionCritic().eval().requires_grad_(False)
    encoder = EventActionEncoder(adjacency=torch.from_numpy(adjacency),
                                 static_features=torch.from_numpy(static)).eval().requires_grad_(False)
    _WORKER = SimpleNamespace(candidates=candidates, actor=actor, critic=critic,
        world=SimpleNamespace(dynamics=SimpleNamespace(event_encoder=encoder)), version=None)


def _context(states, timestamp, deadline, population, cumulative_seconds):
    active = sum(bool(state['departed']) and not bool(state['arrived']) for state in states)
    return causal_context(timestamp, deadline, active, population, cumulative_seconds)


@torch.no_grad()
def _collect_window(task):
    worker = _WORKER
    if worker.version != task['version']:
        weights = task['weights']
        _load_numpy(worker.actor, weights['actor'])
        bounded = weights.get('bounded', False)
        if getattr(worker.critic, 'bounded', False) != bounded:
            worker.critic = EventActionCritic(bounded=bounded).eval().requires_grad_(False)
        _load_numpy(worker.critic, weights['critic'])
        _load_numpy(worker.world.dynamics.event_encoder, weights['encoder'])
        set_shortest_path_prior(worker.actor, *weights['prior'])
        worker.version = task['version']
    states = task['states']
    generator = torch.Generator(device='cpu')
    generator.set_state(torch.from_numpy(task['rng']))
    decoded, population, deadline = task['decoded'], task['population'], task['deadline']
    cumulative_seconds = task['cumulative_vehicle_seconds']
    network = worker.candidates.network
    window = prepare_actor_window(np.asarray(decoded[:, [2, 1]], dtype=np.float32), network.edge_count)
    groups, observations, masks, edges, contexts, probabilities, selections = [], [], [], [], [], [], []
    event_count = task['event_count']
    prior_cost_rows = []

    def decide(pending, frozen_flow, now):
        nonlocal event_count
        event_cumulative = cumulative_seconds + sum(
            max(0., min(float(state.get('arrival_time', now)), now) - task['start'])
            for state in states)
        context = _context(states, now, deadline, population, event_cumulative)
        prior_kwargs = (dict(prior_cost_mode='dynamic', prior_speed_floor_mps=task['prior_speed_floor_mps'],
                             prior_speed_cap_mps=task['max_speed'])
                        if task.get('prior_cost_mode') == 'dynamic' else {})
        observation = build_actor_observations(policy_edge_features=window,
            vehicle_states=pending, candidates=worker.candidates,
            episode_progress=float(context[0]), active_vehicle_fraction=float(context[1]), **prior_kwargs)
        features, mask = torch.from_numpy(observation.features), torch.from_numpy(observation.mask)
        prior_costs = getattr(observation, 'prior_costs_seconds', None)
        actor_kwargs = ({'prior_costs_seconds': torch.as_tensor(prior_costs, dtype=torch.float32)}
                        if prior_costs is not None else {})
        logits = worker.actor.forward_logits(features, mask, **actor_kwargs)
        if prior_costs is not None:
            prior_cost_rows.append(prior_costs)
        if not bool(torch.isfinite(logits[mask]).all()):
            raise ValueError('legal actor logits must be finite')
        probs = logits.softmax(-1)
        selected = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        choices = selected.tolist()
        decisions = selected_decisions(pending, observation, choices, now)
        groups.append(dict(vehicle_id=[str(state['vehicle_id']) for state in pending],
            event_index=list(range(event_count, event_count + len(pending))), decision_time=float(now),
            budget=remaining_cost_budget(now, deadline, sum(not state['arrived'] for state in states), population)))
        observations.append(features)
        masks.append(mask)
        edges.append(observation.edge_indices)
        contexts.append(np.repeat(context[None], len(pending), axis=0))
        probabilities.append(probs)
        selections.append(selected)
        event_count += len(pending)
        return decisions

    speeds = np.minimum(clip_road_speed_fractions(decoded[:, 1],
        task.get('traffic_model', 'micro')) * network.speed_limits_mps, task['max_speed'])
    result = advance_imagined_window(states, candidates=worker.candidates,
        edge_speeds_mps=speeds, frozen_flow=decoded, start_time=task['start'],
        interval_seconds=min(30., deadline - task['start']), total_vehicle_count=task['total_vehicle_count'],
        decide=decide, observation_step_seconds=1., session_protocol='event',
        traffic_model=task.get('traffic_model', 'micro'),
        event_index_start=task['event_count'], vehicle_max_speed_mps=task['max_speed'], stop_when_empty=True)
    output = dict(states=states, result=result, groups=groups, event_count=event_count,
        rng=generator.get_state().numpy().copy(),
        context=_context(states, result.end_time, deadline, population,
                         cumulative_seconds + result.active_vehicle_seconds))
    if result.action_events:
        output['event_count'] = max(event_count, result.action_events[-1].event_index + 1)
    if groups:
        features, mask, selected = torch.cat(observations), torch.cat(masks), torch.cat(selections)
        events = result.action_events if result.policy_events is None else result.policy_events
        sparse = pack_action_events([[events]], window_starts=[[task['start']]])
        q_features = build_event_q_inputs(worker.world, z=torch.from_numpy(task['latent'])[None],
            observations=features, candidate_edges=torch.from_numpy(np.concatenate(edges)), mask=mask,
            current_edges=torch.tensor([event.current_edge for event in events]),
            destination_edges=torch.tensor([event.destination_edge for event in events]),
            decision_times=torch.tensor([event.decision_time for event in events]),
            event_indices=torch.tensor([event.event_index for event in events]),
            context=torch.from_numpy(np.concatenate(contexts)), events=sparse, window_start=task['start'])
        budgets = torch.tensor([group['budget'] for group in groups for _ in group['vehicle_id']], dtype=torch.float32)
        q_kwargs = {'budget': budgets} if task['weights'].get('bounded', False) else {}
        baseline = detached_counterfactual_baseline(torch.cat(probabilities), worker.critic(q_features, mask, **q_kwargs), mask)
        if prior_cost_rows:
            output['prior_costs_seconds'] = np.concatenate(prior_cost_rows)
        output.update(observations=features.numpy(), mask=mask.numpy(), selected=selected.numpy(),
            baseline=baseline.numpy(), budgets=budgets.numpy(), executed_features=q_features[torch.arange(len(selected)), selected].numpy())
    return output


def close_parallel_imagination(trainer):
    """Release workers at trainer shutdown or after a failed collection pass."""
    executor = getattr(trainer, '_parallel_imagination_executor', None)
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)
        del trainer._parallel_imagination_executor


def _executor(trainer, workers):
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('parallel imagination workers must be a positive integer')
    if (getattr(trainer, '_parallel_imagination_executor', None) is not None
            and trainer._parallel_imagination_workers != workers):
        close_parallel_imagination(trainer)
    if getattr(trainer, '_parallel_imagination_executor', None) is None:
        encoder = trainer.world_model.dynamics.event_encoder
        adjacency = np.zeros((encoder.n_edges, encoder.n_edges), dtype=np.float32)
        turns = encoder.turns.detach().cpu().numpy()
        adjacency[turns[:, 0], turns[:, 1]] = 1.
        trainer._parallel_imagination_executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=mp.get_context('spawn'), initializer=_initialize_worker,
            initargs=(trainer.candidates, adjacency, encoder.static_features.detach().cpu().numpy()))
        trainer._parallel_imagination_workers = workers
    return trainer._parallel_imagination_executor


def collect_parallel_imagination(trainer, workers=4):
    """Return event groups and per-rollout V losses; never take optimizer steps.

    Parent-generated rollout seeds are part of the existing checkpointed CPU
    RNG stream. Independent generators make CPU tasks scheduling-independent.
    Explicit per-rollout WM noise also makes active-batch packing irrelevant.
    Worker parameter payloads are NumPy-only and cached by pass version; no
    tensors or autograd graphs cross a process boundary.
    """
    from .training import IMAGINED_HORIZON

    rollout_count = trainer.config.resources.imagination_rollouts_per_pass
    executor = _executor(trainer, workers)
    trainer._parallel_imagination_version = getattr(trainer, '_parallel_imagination_version', 0) + 1
    version = trainer._parallel_imagination_version
    weights = dict(actor=_numpy_state(trainer.actor), critic=_numpy_state(trainer.event_critic),
        encoder=_numpy_state(trainer.world_model.dynamics.event_encoder),
        prior=(trainer.actor.prior_weight, trainer.actor.prior_temperature_seconds),
        bounded=trainer._bounded_critic())
    batch = trainer.online_replay.sample_active_seeds(batch_size=rollout_count,
                                                     rng=trainer.rng, initial_only=True)
    seeds = torch.randint(0, 2**63 - 1, (rollout_count, 2), device='cpu').tolist()
    deadline = trainer._episode_end_time()
    tracks = []
    for i, (actor_seed, noise_seed) in enumerate(seeds):
        states = [state for state in batch.vehicles[i] if state['departed'] and not state['arrived']]
        if batch.started_count is None or batch.cumulative_vehicle_seconds is None:
            raise ValueError('imagination requires causal cumulative reward accounting')
        population, start = int(batch.started_count[i]), float(batch.time[i])
        cumulative_seconds = float(batch.cumulative_vehicle_seconds[i])
        if population <= 0 or not states or start >= deadline:
            raise ValueError('imagination seed needs positive population and active vehicles before deadline')
        tracks.append(dict(states=copy.deepcopy(states), population=population, start=start,
            total_vehicle_count=int(batch.total_vehicle_count[i]),
            cumulative_vehicle_seconds=cumulative_seconds,
            rng=torch.Generator().manual_seed(actor_seed).get_state().numpy().copy(),
            noise_rng=torch.Generator(device=trainer.device).manual_seed(noise_seed),
            event_count=0, windows=[], terminal=False))
    with torch.no_grad():
        latent = trainer.world_model.encode(batch.flow[:, 0].to(trainer.device))
    try:
        for window_index in range(IMAGINED_HORIZON):
            active = [i for i, track in enumerate(tracks) if not track['terminal']]
            if not active:
                break
            with torch.no_grad():
                decoded = trainer.world_model.decode(latent[active]).cpu().numpy()
                latent_host = latent[active].cpu().numpy()
            tasks = []
            for ordinal, i in enumerate(active):
                track = tracks[i]
                tasks.append(dict(version=version, weights=weights, states=track['states'],
                    rng=track['rng'], decoded=decoded[ordinal], latent=latent_host[ordinal],
                    population=track['population'], deadline=deadline, start=track['start'],
                    total_vehicle_count=track['total_vehicle_count'],
                    cumulative_vehicle_seconds=track['cumulative_vehicle_seconds'],
                    max_speed=trainer.config.simulation.vehicle_max_speed_mps,
                    traffic_model=getattr(trainer.config.simulation, 'traffic_model', 'micro'),
                    event_count=track['event_count'],
                    prior_cost_mode=getattr(trainer.config.method, 'sp_prior_cost_mode', 'free_flow'),
                    prior_speed_floor_mps=getattr(trainer.config.method, 'sp_prior_dynamic_speed_floor_mps', .5)))
            outputs = list(executor.map(_collect_window, tasks, chunksize=1))
            full = [ordinal for ordinal, output in enumerate(outputs)
                    if abs(output['result'].end_time - tracks[active[ordinal]]['start'] - 30.) <= 1e-7]
            next_latent = latent[active].clone()
            risk = next_latent.new_zeros(len(active))
            with torch.no_grad():
                if full:
                    method = trainer.config.method
                    rank = trainer.world_model.dynamics.graph_prior.rank
                    noise = torch.stack([torch.randn((method.disagreement_samples, rank, 64),
                        generator=tracks[active[ordinal]]['noise_rng'], device=trainer.device,
                        dtype=latent.dtype) for ordinal in full], dim=1)
                    sparse = pack_action_events([[outputs[ordinal]['result'].action_events] for ordinal in full],
                        window_starts=[[tracks[active[ordinal]]['start']] for ordinal in full])
                    actions = torch.as_tensor(np.stack([outputs[ordinal]['result'].action for ordinal in full]),
                                              device=trainer.device, dtype=latent.dtype)
                    samples, variance, _ = trainer.world_model.latent_disagreement(latent[active][full],
                        actions, action_events=sparse.to(trainer.device), noise=noise,
                        n_samples=method.disagreement_samples, steps=method.fm_steps,
                        solver=method.fm_solver, picard_iterations=method.picard_iterations)
                    next_latent[full] = samples[0]
                    risk[full] = method.disagreement_weight * variance.reshape(-1)
                contexts = torch.as_tensor(np.stack([output['context'] for output in outputs]),
                                           device=trainer.device, dtype=latent.dtype)
                budgets = [remaining_cost_budget(output['result'].end_time, deadline,
                    sum(not state['arrived'] for state in output['states']), tracks[i]['population'])
                    for i, output in zip(active, outputs)]
                next_values = trainer._value_prediction(next_latent, contexts, budgets, target=True).reshape(-1)
            for ordinal, i in enumerate(active):
                track, output = tracks[i], outputs[ordinal]
                result = output['result']
                # Worker outputs own a copied state, so this is the causal
                # physical boundary before the just-completed interval.
                context = _context(track['states'], track['start'], deadline, track['population'],
                                   track['cumulative_vehicle_seconds'])
                terminal = result.end_time >= deadline - 1e-9 or all(state['arrived'] for state in output['states'])
                track['windows'].append(dict(output=output, start=track['start'], pooled=latent[i].mean(0).detach(),
                    context=context, budget=remaining_cost_budget(track['start'], deadline,
                        sum(not state['arrived'] for state in track['states']), track['population']), next_value=next_values[ordinal].detach(), terminal=terminal,
                    cumulative_vehicle_seconds=track['cumulative_vehicle_seconds'],
                    global_reward=latent.new_tensor(-result.active_vehicle_seconds / (track['population'] * 60.)),
                    risk_cost=risk[ordinal],
                    reward=latent.new_tensor(-result.active_vehicle_seconds / (track['population'] * 60.)) - risk[ordinal]))
                track.update(states=output['states'], start=result.end_time, rng=output['rng'],
                             event_count=output['event_count'], terminal=terminal,
                             cumulative_vehicle_seconds=track['cumulative_vehicle_seconds'] + result.active_vehicle_seconds)
            latent[active] = next_latent
    except BaseException:
        close_parallel_imagination(trainer)
        raise
    windows = [window for track in tracks for window in track['windows']]
    trainer._last_parallel_work = dict(rollouts=len(tracks), windows=len(windows),
        seconds=sum(window['output']['result'].end_time - window['start'] for window in windows),
        active_vehicle_seconds=sum(window['output']['result'].active_vehicle_seconds for window in windows))
    return _finish_pass(trainer, tracks, deadline)


def _finish_pass(trainer, tracks, deadline):
    """Create the exact event objective after all frozen rollout sampling."""
    windows = [window for track in tracks for window in track['windows']]
    pooled = torch.stack([window['pooled'] for window in windows])[:, None]
    context = torch.as_tensor(np.stack([window['context'] for window in windows]),
                              device=trainer.device, dtype=pooled.dtype)
    values = trainer._value_prediction(pooled, context,
        [window['budget'] for window in windows] if trainer._bounded_critic() else None).reshape(-1)
    losses, groups, offset = [], [], 0
    pending = []
    for track in tracks:
        trajectory = track['windows']
        rewards = torch.stack([window['reward'] for window in trajectory])
        next_values = torch.stack([window['next_value'] for window in trajectory])
        terminals = torch.tensor([window['terminal'] for window in trajectory], device=trainer.device)
        targets = lambda_return(rewards, next_values, terminal=terminals)
        if trainer._bounded_critic():
            trainer._validate_critic_targets(targets, [window['budget'] for window in trajectory])
        losses.append(F.mse_loss(values[offset:offset + len(trajectory)], targets.detach()))
        offset += len(trajectory)
        bootstrap = next_values[-1] if not trajectory[-1]['terminal'] else next_values.new_zeros(())
        # The observed-time objective adds no penalty for unfinished trips.
        for index, window in enumerate(trajectory):
            output = window['output']
            if not output['groups']:
                continue
            cursor = 0
            for metadata in output['groups']:
                size = len(metadata['vehicle_id'])
                elapsed = sum(segment.active_count * max(0., min(segment.end_time, metadata['decision_time'])
                    - segment.start_time) for segment in output['result'].cost_segments)
                event_mean = (window['cumulative_vehicle_seconds'] + elapsed) / track['population']
                physical_suffix = -(track['cumulative_vehicle_seconds'] / track['population'] - event_mean) / 60.
                risk_suffix = torch.stack([item['risk_cost'] for item in trajectory[index:]]).sum()
                target = rewards.new_tensor(physical_suffix) - risk_suffix + bootstrap
                if trainer._bounded_critic():
                    trainer._validate_critic_targets(target.reshape(1), [metadata['budget']])
                baseline = torch.as_tensor(output['baseline'][cursor:cursor + size], device=trainer.device)
                group = dict(metadata, window=index, population=track['population'], global_return=target.detach().expand(size),
                    global_advantage=target.detach().expand(size) - baseline,
                    counterfactual_baseline=baseline, global_reward=window['global_reward'],
                    risk_cost=window['risk_cost'], timeout_return=target.new_zeros(size))
                groups.append(group)
                cursor += size
            pending.append(output)
    if pending:
        sizes = [len(group['vehicle_id']) for group in groups]
        # Bound MLP activation peaks independently of episode population.
        observations = torch.as_tensor(np.concatenate([row['observations'] for row in pending]), device=trainer.device)
        masks = torch.as_tensor(np.concatenate([row['mask'] for row in pending]), device=trainer.device)
        selected = torch.as_tensor(np.concatenate([row['selected'] for row in pending]), device=trainer.device)
        features = torch.as_tensor(np.concatenate([row['executed_features'] for row in pending]), device=trainer.device)
        budgets = (torch.as_tensor(np.concatenate([row['budgets'] for row in pending]), device=trainer.device,
                   dtype=torch.float32) if trainer._bounded_critic() else None)
        prior_costs = (torch.as_tensor(np.concatenate([row['prior_costs_seconds'] for row in pending]),
                      dtype=torch.float32, device=trainer.device) if 'prior_costs_seconds' in pending[0] else None)
        logs, entropies, executed = [], [], []
        for start in range(0, len(selected), 8192):
            stop = start + 8192
            distribution = torch.distributions.Categorical(logits=trainer.actor.forward_logits(
                observations[start:stop], masks[start:stop],
                **({'prior_costs_seconds': prior_costs[start:stop]} if prior_costs is not None else {})))
            logs.append(distribution.log_prob(selected[start:stop]))
            entropies.append(distribution.entropy())
            conditions = features[start:stop, None]
            executed.append(trainer.event_critic(conditions,
                torch.ones(conditions.shape[:2], dtype=torch.bool, device=trainer.device),
                **({'budget': budgets[start:stop]} if budgets is not None else {})).squeeze(-1))
        for group, log_prob, entropy, q in zip(groups, torch.cat(logs).split(sizes),
                                              torch.cat(entropies).split(sizes), torch.cat(executed).split(sizes)):
            group.update(log_prob=log_prob, entropy=entropy, executed_q=q)
    return groups, losses
