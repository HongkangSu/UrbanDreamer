#!/usr/bin/env python3
"""On-policy event Actor–Critic diagnostic using real SUMO, with no world model."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
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

from urban_dreamer.learning.observations import build_actor_observations, build_vehicle_observations
from urban_dreamer.models.policy import NextHopActor, CentralizedVehicleValueCritic, lambda_returns_by_vehicle_ids, lambda_return
from urban_dreamer.traffic.environment import NextHopDecision, SumoSession
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
from urban_dreamer.workflow.config import NONARRIVAL_PENALTY, load_config


def close_event(row, end_time, next_obs, *, terminal, timeout=False):
    elapsed = float(end_time) - row['time']
    if elapsed < 0:
        raise ValueError('negative event elapsed time')
    row.update(reward=-elapsed / 30. + (NONARRIVAL_PENALTY if timeout else 0.),
               next_obs=next_obs, terminal=terminal)


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
        if self.config.inputs.network_name != 'toy_cross_bottleneck':
            raise ValueError('this bounded diagnostic is for Toy only')
        if args.vehicle_limit:
            self.topology = replace(self.topology, vehicles=self.topology.vehicles[:args.vehicle_limit])
        self.network = self.topology.network
        self.n = len(self.topology.vehicles)
        self.candidates = NextHopCandidates(self.network)
        simulation = self.config.simulation
        self.deadline = 30 * math.ceil((simulation.world_model_intervals_per_episode * 30
                                       + simulation.training_completion_buffer_seconds) / 30)
        self.local_weight = float(getattr(args, 'local_reward_weight', 1.))
        self.advantage_normalization = 'center_and_scale' if self.local_weight else 'scale_only_std_floor_1'
        self.actor = NextHopActor().to(self.device)
        self.critic = CentralizedVehicleValueCritic().to(self.device) if self.local_weight else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr) if self.critic is not None else None
        self.global_weight = float(getattr(args, 'global_reward_weight', 0.))
        self.global_critic = None
        self.global_optimizer = None
        self.collected_windows = []
        if self.global_weight:
            # The added critic must not change actor/local-critic initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(args.seed + 73861)
                self.global_critic = GlobalTrafficCritic(self.network.edge_count).to(self.device)
            self.global_optimizer = torch.optim.Adam(self.global_critic.parameters(),
                lr=getattr(args, 'global_critic_lr', args.critic_lr))
        self.ids = {vehicle.vehicle_id: i for i, vehicle in enumerate(self.topology.vehicles)}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def global_observation(self, state):
        return np.concatenate((np.asarray(state['flow'][:, :4], dtype=np.float32).reshape(-1),
            np.asarray([state['time'] / self.deadline, state['active_vehicle_count'] / self.n], dtype=np.float32)))

    def global_targets(self, windows):
        if self.global_critic is None or not windows:
            raise ValueError('global critic needs collected macro windows')
        values = self.global_critic(self.tensor(np.stack([w['obs'] for w in windows]))).squeeze(-1)
        with torch.no_grad():
            next_values = self.global_critic(self.tensor(np.stack([w['next_obs'] for w in windows]))).squeeze(-1)
            targets = lambda_return(self.tensor([w['reward'] for w in windows]), next_values,
                terminal=torch.tensor([w['terminal'] for w in windows], device=self.device, dtype=torch.bool))
        return targets - values.detach(), targets, values

    def run_episode(self, *, policy, seed, collect=False):
        started = time.monotonic()
        config = replace(self.config, run=replace(self.config.run, seed=int(seed)))
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        rows, pending, seen_arrivals = [], {}, set()
        choices, branch_choices = Counter(), Counter()
        branch_entropies = []
        exposure, event_count = 0., 0
        windows = []
        with SumoSession(config, self.prepared, self.topology) as session:
            state = session.reset()
            while state['time'] < self.deadline and not session.done:
                progress = state['time'] / self.deadline
                active_fraction = state['active_vehicle_count'] / self.n
                window_index = len(windows)
                global_obs = self.global_observation(state)

                def decide(vehicles, frozen, now):
                    nonlocal event_count
                    kwargs = dict(policy_edge_features=np.asarray(frozen[:, [2, 1]], dtype=np.float32),
                        vehicle_states=vehicles, candidates=self.candidates,
                        episode_progress=progress, active_vehicle_fraction=active_fraction)
                    actor_obs = build_actor_observations(**kwargs)
                    critic_obs = build_vehicle_observations(**kwargs, current_time=now) if collect and self.local_weight else None
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
                            if vehicle_id in pending:
                                close_event(pending.pop(vehicle_id), now, critic_obs[i].copy() if critic_obs is not None else None, terminal=False)
                            row = dict(vehicle_id=self.ids[vehicle_id], time=float(now), obs=critic_obs[i].copy() if critic_obs is not None else None,
                                features=actor_obs.features[i].copy(), mask=actor_obs.mask[i].copy(), action=indices[i], window=window_index)
                            rows.append(row)
                            pending[vehicle_id] = row
                        decisions.append(NextHopDecision(vehicle_id, int(vehicle['current_edge']), edge, float(now)))
                    event_count += len(decisions)
                    return decisions

                result = session.advance_window(decide)
                exposure += result.active_vehicle_seconds
                state = result.next_state
                global_timeout_penalty = (NONARRIVAL_PENALTY * (self.n - len(session.travel_times))
                    if not self.local_weight and state['time'] >= self.deadline else 0.)
                windows.append(dict(obs=global_obs, next_obs=self.global_observation(state),
                    reward=-result.active_vehicle_seconds / (self.n * 60.) + global_timeout_penalty,
                    terminal=bool(session.done or state['time'] >= self.deadline)))
                if collect:
                    for vehicle in state['vehicles']:
                        vehicle_id = vehicle['vehicle_id']
                        if vehicle['arrived'] and vehicle_id not in seen_arrivals:
                            seen_arrivals.add(vehicle_id)
                            if vehicle_id in pending:
                                close_event(pending.pop(vehicle_id), vehicle['arrival_time'], np.zeros(14, dtype=np.float32), terminal=True)
            if collect:
                for row in pending.values():
                    close_event(row, self.deadline, np.zeros(14, dtype=np.float32), terminal=True, timeout=True)
            durations = np.asarray(list(session.travel_times.values()), dtype=np.float64)
            unfinished = self.n - len(durations)
            metrics = dict(policy=policy, seed=int(seed), vehicle_count=self.n,
                completion_rate=len(durations) / self.n, unfinished=unfinished,
                never_departed=sum(not vehicle["departed"] for vehicle in state["vehicles"]),
                mean_travel_seconds=float(durations.mean()) if durations.size else None,
                p90_travel_seconds=float(np.percentile(durations, 90)) if durations.size else None,
                exposure_seconds=exposure, mean_exposure_seconds=exposure / self.n,
                reward_per_vehicle=(-exposure / 30. + NONARRIVAL_PENALTY * unfinished) / self.n,
                simulation_end_seconds=float(state['time']), microstep_seconds=session.microstep_seconds,
                microsteps=int(session.metrics['microsteps']), event_count=event_count,
                branch_event_count=len(branch_entropies), branch_entropy=float(np.mean(branch_entropies)) if branch_entropies else 0.,
                branch_choices=dict(branch_choices), choices=dict(choices), wall_seconds=time.monotonic() - started,
                window_count=len(windows), global_reward_sum=sum(w['reward'] for w in windows))
        if collect and any('reward' not in row for row in rows):
            raise AssertionError('unclosed event trajectory')
        if collect:
            self.collected_windows = windows
        return rows, metrics

    def update(self, rows, windows=None):
        if not rows:
            raise ValueError('no event trajectories')
        advantage = torch.zeros(len(rows), device=self.device)
        local_advantage_std = 0.
        local_metrics = {}
        if self.local_weight:
            obs = self.tensor(np.stack([r['obs'] for r in rows]))
            next_obs = self.tensor(np.stack([r['next_obs'] for r in rows]))
            reward = self.tensor([r['reward'] for r in rows])
            terminal = torch.tensor([r['terminal'] for r in rows], dtype=torch.bool, device=self.device)
            ids = torch.tensor([r['vehicle_id'] for r in rows], device=self.device)
            with torch.no_grad():
                values = self.critic(obs).squeeze(-1)
                next_values = self.critic(next_obs).squeeze(-1)
                targets = lambda_returns_by_vehicle_ids(reward, next_values,
                    torch.ones(len(rows), dtype=torch.bool, device=self.device), ids, terminal=terminal)
                advantage = targets - values
            local_advantage_std = float(advantage.std(unbiased=False))
            advantage = self.local_weight * advantage
        global_metrics = {}
        if self.global_weight:
            global_advantage, global_targets, global_values = self.global_targets(windows)
            event_windows = torch.tensor([r['window'] for r in rows], device=self.device)
            advantage = advantage + self.global_weight * global_advantage[event_windows]
            global_metrics['global_advantage_std'] = float(global_advantage.std(unbiased=False))
        if self.local_weight:
            advantage = ((advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)).clamp(-2.5, 2.5)
        else:
            # Preserve a team-wide signal: centering a shared constant advantage
            # would erase it entirely. The value critic already supplies a baseline.
            advantage = (advantage / advantage.std(unbiased=False).clamp_min(1.)).clamp(-2.5, 2.5)
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
        if self.local_weight:
            for _ in range(self.args.critic_updates):
                predicted = self.critic(obs).squeeze(-1)
                critic_loss = F.mse_loss(predicted, targets)
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_gradient = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
                if not torch.isfinite(critic_gradient):
                    raise RuntimeError('non-finite critic gradient')
                self.critic_optimizer.step()
        if self.global_weight:
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
            global_metrics.update(global_critic_loss=float(global_loss.detach()),
                global_critic_gradient_norm=float(global_gradient), global_explained_variance=float(global_ev))
        if self.local_weight:
            with torch.no_grad():
                explained_variance = 1 - torch.var(targets - self.critic(obs).squeeze(-1), unbiased=False) / torch.var(targets, unbiased=False).clamp_min(1e-8)
            local_metrics.update(critic_loss=float(critic_loss.detach()), critic_gradient_norm=float(critic_gradient),
                explained_variance=float(explained_variance), target_mean=float(targets.mean()))
        return dict(actor_loss=float(actor_loss.detach()), actor_gradient_norm=float(actor_gradient),
            local_advantage_std=local_advantage_std, **local_metrics, **global_metrics)

    def save(self, path, episode, best):
        torch.save(dict(format='toy_real_actor_critic_v2', episode=episode, best=best,
            advantage_normalization=self.advantage_normalization,
            actor=self.actor.state_dict(), critic=self.critic.state_dict() if self.critic is not None else None,
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
            global_critic=self.global_critic.state_dict() if self.global_critic is not None else None,
            global_optimizer=self.global_optimizer.state_dict() if self.global_optimizer is not None else None,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items()},
            input_hashes=self.prepared.input_hashes, config_fingerprint=self.config.semantic_fingerprint), path)


def log_record(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')
    short = {key: value for key, value in record.items() if key not in {'branch_choices', 'choices'}}
    print(json.dumps(short, allow_nan=False), flush=True)


def evaluate(runner, output, *, episode, seeds, phase, policies=('deterministic',)):
    records = []
    for policy in policies:
        for seed in seeds:
            _, metrics = runner.run_episode(policy=policy, seed=seed)
            record = dict(phase=phase, episode=episode, **metrics)
            records.append(record)
            log_record(output / 'metrics.jsonl', record)
    return records


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
    parser.add_argument('--critic-lr', type=float, default=3e-4)
    parser.add_argument('--critic-updates', type=int, default=1)
    parser.add_argument('--vehicle-limit', type=int, default=0)
    parser.add_argument('--global-reward-weight', type=float, default=0.)
    parser.add_argument('--local-reward-weight', type=float, default=1.)
    parser.add_argument('--global-critic-lr', type=float, default=3e-4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.episodes < 0 or min(args.eval_every, args.eval_seeds, args.threads, args.critic_updates) < 1:
        parser.error('invalid nonpositive experiment budget')
    if any(not math.isfinite(v) or v < 0 for v in (args.local_reward_weight,args.global_reward_weight)) or args.local_reward_weight + args.global_reward_weight <= 0:
        parser.error('reward weights must be finite, nonnegative, and not both zero')
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite a nonempty experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    runner = RealActorCritic(args)
    evaluation_seeds = [args.seed + 10000 + i for i in range(args.eval_seeds)]
    start, best = 1, float('inf')
    if args.resume:
        state = torch.load(output / 'latest.pt', map_location=runner.device, weights_only=False)
        if state['format'] != 'toy_real_actor_critic_v2' or state['input_hashes'] != runner.prepared.input_hashes or state['config_fingerprint'] != runner.config.semantic_fingerprint:
            raise ValueError('experiment checkpoint mismatch')
        for key in ('seed', 'vehicle_limit', 'actor_lr', 'critic_lr', 'critic_updates', 'global_reward_weight', 'global_critic_lr'):
            if state['args'][key] != getattr(args, key):
                raise ValueError(f'resume changed {key}')
        if state['args'].get('local_reward_weight', 1.) != args.local_reward_weight:
            raise ValueError('resume changed local_reward_weight')
        if state.get('advantage_normalization', 'center_and_scale') != runner.advantage_normalization:
            raise ValueError('resume changed advantage normalization')
        runner.actor.load_state_dict(state['actor'])
        runner.actor_optimizer.load_state_dict(state['actor_optimizer'])
        if runner.critic is not None:
            runner.critic.load_state_dict(state['critic'])
            runner.critic_optimizer.load_state_dict(state['critic_optimizer'])
        if runner.global_critic is not None:
            runner.global_critic.load_state_dict(state['global_critic'])
            runner.global_optimizer.load_state_dict(state['global_optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng'])
        start, best = state['episode'] + 1, state['best']
    else:
        protocol = dict(world_model=False, macro_latent_critic=False,
            reward=('-active_vehicle_seconds/(N*60); -5000 per unfinished vehicle at deadline'
                    if not runner.local_weight else '-event_elapsed/30, -5000 at episode deadline'),
            gamma=1., lambda_return=.95, entropy_coefficient=.002, vehicle_count=runner.n,
            local_reward_weight=runner.local_weight, global_reward_weight=runner.global_weight,
            advantage_normalization=runner.advantage_normalization,
            global_reward='-active_vehicle_seconds/(N*60)',
            global_only_timeout_penalty='-5000 per unfinished vehicle at the real deadline',
            global_critic_observation='flattened physical dynamic flow + progress/active fraction',
            deadline_seconds=runner.deadline, config=runner.config.resolved_snapshot(),
            args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            input_hashes=runner.prepared.input_hashes, evaluation_seeds=evaluation_seeds,
            note=('Global-only Actor-Critic; no vehicle critic, world model, or imagination.' if not runner.local_weight else
                  'Actor, local critic and real-state global critic; measured global reward; no world model or imagination.'
                  if runner.global_weight else 'Actor and vehicle critic only; no learned macro reward or imagination.'),
            entropy_note='branch_entropy is entropy of the policy distribution, including in argmax evaluation.',
            reward_note='Travel metrics include initial observation lag; event rewards start at the first decision.')
        (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        baseline = evaluate(runner, output, episode=0, seeds=evaluation_seeds, phase='baseline',
                 policies=('random', 'shortest', 'deterministic'))
        best = -float(np.mean([r['reward_per_vehicle'] for r in baseline if r['policy'] == 'deterministic']))
        runner.save(output / 'best.pt', 0, best)
        runner.save(output / 'initial.pt', 0, best)
        runner.save(output / 'latest.pt', 0, best)
    for episode in range(start, args.episodes + 1):
        rows, metrics = runner.run_episode(policy='stochastic', seed=args.seed + episode, collect=True)
        update = runner.update(rows, runner.collected_windows)
        log_record(output / 'metrics.jsonl', dict(phase='train', episode=episode, **metrics, **update))
        if episode % args.eval_every == 0 or episode == args.episodes:
            records = evaluate(runner, output, episode=episode, seeds=evaluation_seeds, phase='evaluation')
            score = float(np.mean([record['reward_per_vehicle'] for record in records]))
            if -score < best:
                best = -score
                runner.save(output / 'best.pt', episode, best)
        runner.save(output / 'latest.pt', episode, best)
    if args.episodes:
        holdout = [args.seed + 20000 + i for i in range(args.eval_seeds)]
        evaluate(runner, output, episode=args.episodes, seeds=holdout, phase='holdout_final',
                 policies=('random', 'shortest', 'deterministic', 'stochastic'))
        state = torch.load(output / 'best.pt', map_location=runner.device, weights_only=False)
        runner.actor.load_state_dict(state['actor'])
        evaluate(runner, output, episode=state['episode'], seeds=holdout, phase='holdout_best')
    print(json.dumps({'status': 'completed', 'output': str(output), 'episodes': args.episodes}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
