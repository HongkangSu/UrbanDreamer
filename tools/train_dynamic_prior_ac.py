#!/usr/bin/env python3
"""Real AC residual over causal 30-second shortest-time routing (no world model)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.train_toy_actor_critic import RealActorCritic, set_episode_entropy
from tools.dynamic_routing_prior import DynamicPriorActor
from tools.routing_cost_observer import PeriodicCostObserver
from tools.routing_demand_profiles import ReTimedDemand, profile_metadata
from tools.causal_routing_policies import RealTimeShortestRouter
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.models.policy import NextHopActor, individual_actor_advantage
from urban_dreamer.traffic.event_session import SumoEventSession

FORMAT = 'dynamic_prior_ac_v1'
EVALUATION_DEFINITION = 'deterministic: masked raw actor NN argmax; no dynamic or static routing prior'


def evaluation_policy(args):
    return ('deterministic' if args.evaluation_argmax else
            f'sampled_temperature_{args.evaluation_temperature:g}')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def fingerprint(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def set_episode_prior(runner, episode):
    """Freeze the episode's sampling coefficient until collection and update finish."""
    protocol = runner.protocol
    weight = protocol['prior_weight']
    if protocol['prior_schedule'] == 'linear':
        weight *= max(0., 1. - max(0, episode - 1) / (protocol['prior_off_episode'] - 1))
    runner.actor.dynamic_prior_weight = float(weight)
    return runner.actor.dynamic_prior_weight


def summarize(rows):
    total = sum(r['vehicle_count'] for r in rows)
    completed = sum(r['vehicle_count'] - r['unfinished'] for r in rows)
    started = sum(r['started_vehicle_count'] for r in rows)
    return dict(completed=completed, vehicle_count=total,
                completion_rate=completed / max(total, 1),
                ATT_seconds=sum(r['cumulative_vehicle_seconds'] for r in rows) / max(started, 1),
                never_departed=sum(r['never_departed'] for r in rows))


def score(rows):
    summary = summarize(rows)
    return summary['completed'], -summary['ATT_seconds']


def beats_reference(rows, reference):
    value, baseline = summarize(rows), summarize(reference)
    return (value['completion_rate'] >= baseline['completion_rate']
            and value['ATT_seconds'] < baseline['ATT_seconds'])


class DynamicPriorAC(RealActorCritic):
    def __init__(self, args):
        super().__init__(args)
        self.actor = DynamicPriorActor(prior_weight=args.prior_weight,
            temperature_seconds=args.prior_temperature).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.demand = ReTimedDemand(self.demand, args.profile)
        self.router = RealTimeShortestRouter(self.network)
        self.session_handle = {}
        handle = self.session_handle
        period = args.cost_observation_seconds

        class ObservedSession(SumoEventSession):
            def __init__(session, *a, **kw):
                super().__init__(*a, **kw)
                session.routing_cost_observer = PeriodicCostObserver(period, session.network.free_flow_travel_seconds)

            def _read_routing_costs(session):
                return [session._connection.edge.getTraveltime(edge) for edge in session.network.edge_ids]

            def reset(session):
                state = super().reset()
                if handle['observe_costs']:
                    session.routing_cost_observer.observe(0, session._read_routing_costs)
                return state

            def _apply_decisions(session, callback, frozen_flow):
                if handle['observe_costs']:
                    session.routing_cost_observer.observe(session._time_ms / 1000., session._read_routing_costs)
                return super()._apply_decisions(callback, frozen_flow)

        def factory(*a, **kw):
            handle['session'] = ObservedSession(*a, **kw, step_seconds=1., backend=args.backend)
            return handle['session']
        self.session_factory = factory
        semantic_keys = ('episodes', 'seed', 'profile', 'prior_weight', 'prior_temperature',
                         'prior_schedule', 'prior_off_episode', 'cost_observation_seconds',
                         'actor_lr', 'global_critic_lr', 'backend', 'eval_every',
                         'evaluation_temperature')
        self.protocol = {key: getattr(args, key) for key in semantic_keys}
        reference_path = Path(args.baseline_reference)
        reference = json.loads(reference_path.read_text())
        if reference['protocol']['input_hashes'] != self.prepared.input_hashes:
            raise ValueError('Reference input hash mismatch')
        if (reference['protocol'].get('cost_observation_seconds') != args.cost_observation_seconds
                or reference['protocol'].get('demand_profile', {}).get('profile') != args.profile):
            raise ValueError('Reference observation or demand profile mismatch')
        self.protocol['baseline_reference_sha256'] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        self.protocol.update(format=FORMAT, input_hashes=self.prepared.input_hashes,
            config_fingerprint=self.config.semantic_fingerprint, simulation=self.simulation_settings,
            reward_objective='observed_mean', individual_timeout=False, entropy_budget=args.episodes,
            gamma=1., lambda_return=.95, gradient_clip=10., actor_features='original_10',
            prior='additive -episode_weight*(cost-min_legal_cost)/temperature; float64 centering',
            prior_schedule_definition='constant: initial_weight; linear: initial_weight*max(0,1-max(0,episode-1)/(prior_off_episode-1)); fixed within collection/update',
            evaluation_policy=evaluation_policy(args),
            evaluation_definition=('masked raw actor NN argmax; no routing prior' if args.evaluation_argmax
                                   else f'masked raw actor NN sampling T={args.evaluation_temperature:g}; no routing prior; fixed isolated action RNG'),
            evaluation_action_seed_rule='1000003 + demand seed',
            reference_evaluation_policy='deterministic_with_prior' if args.prior_weight > 0 else 'prior',
            demand=profile_metadata(args.profile, self.demand.duration_seconds),
            observation='road features frozen per30s; dynamic costs global ticks0,30,... only when routing_prior_applied; event positions live',
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [Path(__file__), ROOT/'tools/dynamic_routing_prior.py',
                          ROOT/'tools/routing_cost_observer.py', ROOT/'tools/routing_demand_profiles.py',
                          ROOT/'tools/causal_routing_policies.py', ROOT/'tools/train_toy_actor_critic.py',
                          ROOT/'urban_dreamer/learning/baselines/common.py',
                          ROOT/'urban_dreamer/learning/observations.py', ROOT/'urban_dreamer/models/policy.py',
                          ROOT/'urban_dreamer/traffic/event_session.py']})

    def run_episode(self, *, policy, seed, collect=False, archive_path=None, check_initial=False):
        sampled_evaluation = policy.startswith('sampled_temperature_')
        temperature = float(policy.removeprefix('sampled_temperature_')) if sampled_evaluation else 1.
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Evaluation temperature must be finite and positive')
        if not sampled_evaluation and policy not in {'stochastic', 'prior', 'deterministic', 'deterministic_with_prior'}:
            raise ValueError(f'Unknown policy: {policy}')
        if check_initial and policy != 'deterministic_with_prior':
            raise ValueError('Initial prior equivalence requires deterministic_with_prior')
        if collect and policy != 'stochastic':
            raise ValueError('Collection requires stochastic policy')
        routing_prior_applied = (policy == 'prior' or
            (policy not in {'deterministic'} and not sampled_evaluation and self.actor.dynamic_prior_weight > 0))
        if check_initial and not routing_prior_applied:
            raise ValueError('Initial prior equivalence requires a positive prior weight')
        # Actor-only evaluation and zero-weight sampling need no dynamic costs.
        self.session_handle['observe_costs'] = routing_prior_applied
        prior_weight = self.actor.dynamic_prior_weight
        self.router.reset()
        agreed = events = 0

        def select(vehicles, obs, frozen, now):
            nonlocal agreed, events
            observer = self.session_handle['session'].routing_cost_observer
            excess = np.zeros(obs.mask.shape, dtype=np.float32)
            expected = []
            if routing_prior_applied:
                if self.router.timestamp != observer.timestamp:
                    self.router.update_costs(observer.values, observer.timestamp)
                for i, vehicle in enumerate(vehicles):
                    indices = np.flatnonzero(obs.mask[i])
                    edges = obs.edge_indices[i, indices]
                    costs = self.router._distance(int(vehicle['destination_edge']))[edges]
                    if not np.isfinite(costs).all():
                        raise ValueError('Unreachable candidate in dynamic prior')
                    # Center before casting so absolute large costs do not erase small differences.
                    excess[i, indices] = (costs - costs.min()).astype(np.float32)
                    expected.append(int(min(indices, key=lambda j: (
                        float(self.router._distance(int(vehicle['destination_edge']))[obs.edge_indices[i, j]]),
                        int(obs.edge_indices[i, j])))))
            with torch.no_grad():
                features = self.tensor(obs.features)
                mask = torch.as_tensor(obs.mask, device=self.device)
                if policy == 'deterministic' or sampled_evaluation:
                    if self.actor.prior_weight != 0:
                        raise ValueError('Static prior must remain disabled for actor-only evaluation')
                    raw_logits = NextHopActor.forward_logits(self.actor, features, mask)
                else:
                    raw_logits = self.actor.forward_logits(features, mask, self.tensor(excess))
                distribution = Categorical(logits=raw_logits / temperature)
                selected = (torch.tensor(expected, device=self.device) if policy == 'prior' else
                    raw_logits.argmax(-1) if policy in {'deterministic', 'deterministic_with_prior'}
                    else distribution.sample())
                actions = selected.cpu().tolist()
                logs, entropy = distribution.log_prob(selected).cpu().tolist(), distribution.entropy().cpu().tolist()
            if check_initial and actions != expected:
                raise ValueError('Zero residual actor did not exactly reproduce prior decisions')
            agreed += sum(a == b for a, b in zip(actions, expected))
            events += len(actions)
            return actions, [dict(prior_excess_seconds=excess[i].copy(), old_log_prob=logs[i],
                prior_weight=prior_weight, entropy=entropy[i], cost_observation_time=observer.timestamp)
                for i in range(len(actions))]

        def metadata(shared):
            shared['demand']['law'] = 'independent_nonhomogeneous_poisson_per_od'
            shared.update(demand_profile=profile_metadata(self.args.profile, self.demand.duration_seconds),
                dynamic_prior_protocol=self.protocol, method='dynamic_prior_ac',
                action_policy=policy, evaluation_policy=None if collect else policy,
                evaluation_temperature=temperature if sampled_evaluation else None,
                prior_weight=prior_weight,
                routing_prior_applied=routing_prior_applied)
            return shared

        rows, windows, metrics = EpisodeRunner.run_episode(self, select, seed=seed,
            collect=collect, row_flow=False, label=policy, archive_path=archive_path,
            archive_metadata=dict(phase='train' if collect else 'evaluation', episode=int(seed)),
            session_factory=self.session_factory, observation_builder=build_actor_observations,
            empty_terminal=True, metadata_transform=metadata)
        if collect:
            self.collected_windows = windows
        metrics.update(action_policy=policy, evaluation_policy=None if collect else policy,
            evaluation_temperature=temperature if sampled_evaluation else None,
            prior_weight=prior_weight,
            routing_prior_applied=routing_prior_applied,
            prior_agreement=agreed / events if routing_prior_applied and events else None,
            cost_observation_times=self.session_handle['session'].routing_cost_observer.sample_times,
            demand_profile=self.args.profile)
        return rows, metrics

    def update(self, rows, windows=None):
        started = time.monotonic()
        if any(r['prior_weight'] != self.actor.dynamic_prior_weight for r in rows):
            raise ValueError('Frozen behavior prior weight mismatch')
        windows = self.collected_windows if windows is None else windows
        advantage, targets, values = self.global_targets(windows)
        stats = dict(actor_loss=0., actor_gradient_norm=0., policy_entropy_mean=0.,
                     behavior_logprob_max_error=0.)
        if rows:
            event_windows = torch.tensor([r['window'] for r in rows], device=self.device)
            actor_advantage = individual_actor_advantage(advantage[event_windows],
                torch.zeros(len(rows), device=self.device))
            actions = torch.tensor([r['action'] for r in rows], device=self.device)
            distribution = Categorical(logits=self.actor.forward_logits(
                self.tensor(np.stack([r['features'] for r in rows])),
                torch.as_tensor(np.stack([r['mask'] for r in rows]), device=self.device),
                self.tensor(np.stack([r['prior_excess_seconds'] for r in rows]))))
            logprob = distribution.log_prob(actions)
            error = (logprob.detach() - self.tensor([r['old_log_prob'] for r in rows])).abs().max()
            if not torch.isfinite(error) or error > .005:
                raise ValueError(f'Frozen behavior log-probability mismatch: {float(error)}')
            entropy = distribution.entropy()
            loss = -(logprob * actor_advantage + self.entropy_coefficient * entropy).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
            if not torch.isfinite(gradient):
                raise ValueError('Non-finite actor gradient')
            self.actor_optimizer.step()
            stats.update(actor_loss=float(loss.detach()), actor_gradient_norm=float(gradient),
                policy_entropy_mean=float(entropy.detach().mean()), behavior_logprob_max_error=float(error),
                actor_advantage_mean=float(actor_advantage.mean()), actor_advantage_std=float(actor_advantage.std(unbiased=False)))
        loss = F.mse_loss(values, targets)
        self.global_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.global_critic.parameters(), 10.)
        if not torch.isfinite(gradient):
            raise ValueError('Non-finite critic gradient')
        self.global_optimizer.step()
        return dict(stats, global_critic_loss=float(loss.detach()), global_critic_gradient_norm=float(gradient),
                    entropy_coefficient=self.entropy_coefficient, learning_seconds=time.monotonic() - started)

    def save_checkpoint(self, path, episode, progress):
        path = Path(path)
        temporary = path.with_name(path.name + '.tmp')
        torch.save(dict(format=FORMAT, protocol=self.protocol, fingerprint=fingerprint(self.protocol),
            episode=episode, progress=progress, actor=self.actor.state_dict(), critic=self.global_critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.global_optimizer.state_dict(),
            rng=rng_state()), temporary)
        temporary.replace(path)

    def load_checkpoint(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get('format') != FORMAT or state.get('fingerprint') != fingerprint(self.protocol):
            raise ValueError('Checkpoint format or semantic protocol mismatch')
        if fingerprint(state['protocol']) != state['fingerprint']:
            raise ValueError('Checkpoint protocol integrity mismatch')
        self.actor.load_state_dict(state['actor'])
        self.global_critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.global_optimizer.load_state_dict(state['critic_optimizer'])
        set_episode_prior(self, state['episode'])
        restore_rng(state['rng'])
        return state['episode'], state['progress']


def evaluate(runner, output, seeds, *, policy=None, check_initial=False):
    policy = policy or runner.protocol.get('evaluation_policy', 'deterministic')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with isolated_rng():
        for seed in seeds:
            action_seed = 1000003 + int(seed)
            random.seed(action_seed)
            np.random.seed(action_seed % (2 ** 32))
            torch.manual_seed(action_seed)
            _, metrics = runner.run_episode(policy=policy, seed=seed, check_initial=check_initial,
                                           archive_path=output / f'episode_{seed}.npz')
            results.append(metrics)
            write_json(output/'results.json', dict(evaluation_policy=policy, rows=results, summary=summarize(results)))
    return results


def check_reference(runner, rows, path):
    reference = json.loads(Path(path).read_text())
    if reference['protocol']['input_hashes'] != runner.prepared.input_hashes:
        raise ValueError('Reference input hash mismatch')
    lookup = {row['seed']: row['metrics'] for row in reference['rows']}
    for row in rows:
        expected = lookup[row['seed']]
        for key in ('vehicle_count', 'demand_sha256', 'unfinished', 'never_departed'):
            if row[key] != expected[key]:
                raise ValueError(f'Initial prior reference mismatch {row["seed"]}: {key}')
        if abs(row['observed_mean_seconds'] - expected['observed_mean_seconds']) > 1e-6:
            raise ValueError(f'Initial prior ATT reference mismatch {row["seed"]}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-reference', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--profile', default='od_stagger', choices=['od_stagger'])
    parser.add_argument('--cost-observation-seconds', type=int, default=30, choices=[30])
    parser.add_argument('--prior-weight', type=float, default=1.)
    parser.add_argument('--prior-schedule', choices=['linear', 'constant'], default='linear')
    parser.add_argument('--prior-off-episode', type=int, default=250)
    parser.add_argument('--prior-temperature', type=float, default=30.)
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument('--evaluation-argmax', dest='evaluation_argmax', action='store_true')
    evaluation.add_argument('--evaluation-sampled', dest='evaluation_argmax', action='store_false')
    parser.set_defaults(evaluation_argmax=True)
    parser.add_argument('--evaluation-temperature', type=float, default=.2)
    parser.add_argument('--actor-lr', type=float, default=2e-4)
    parser.add_argument('--global-critic-lr', type=float, default=3e-4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int, default=None, help='Execution slice; total entropy budget unchanged')
    parser.set_defaults(vehicle_limit=0, session='event', backend='libsumo', step_seconds=1.,
                        sp_prior_weight=0., global_reward_only=True, reward_objective='observed_mean')
    args = parser.parse_args(argv)
    if not math.isfinite(args.evaluation_temperature) or args.evaluation_temperature <= 0:
        parser.error('Evaluation temperature must be finite and positive')
    if args.episodes <= 0 or args.eval_every <= 0 or (args.stop_after is not None and args.stop_after <= 0):
        parser.error('Episode budgets and evaluation intervals must be positive')
    if args.prior_off_episode <= 1:
        parser.error('Prior off episode must be greater than one')
    return args


def main(argv=None):
    args = parse_args(argv)
    actor_evaluation_policy = evaluation_policy(args)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output/'latest.pt').exists():
        raise ValueError('Existing checkpoint requires --resume')
    runner = DynamicPriorAC(args)
    validation_seeds = [10042, 10043, 10044]
    if args.resume:
        completed, progress = runner.load_checkpoint(output/'latest.pt')
        write_json(output/'protocol.json', runner.protocol)
    else:
        write_json(output/'protocol.json', runner.protocol)
        write_json(output/'status.json', dict(phase='initial_validation', completed_episodes=0, total_episodes=args.episodes))
        reference_policy = 'deterministic_with_prior' if args.prior_weight > 0 else 'prior'
        initial = evaluate(runner, output/'evaluation/initial', validation_seeds,
                           policy=reference_policy, check_initial=args.prior_weight > 0)
        check_reference(runner, initial, args.baseline_reference)
        initial_actor_only = evaluate(runner, output/'evaluation/initial_actor_only', validation_seeds,
                                      policy=actor_evaluation_policy)
        completed = 0
        progress = dict(initial=initial, initial_actor_only=initial_actor_only,
            evaluation_policy=actor_evaluation_policy, reference_evaluation_policy=reference_policy,
            best_score=list(score(initial_actor_only)), best_episode=0, validation=[])
        runner.save_checkpoint(output/'initial.pt', 0, progress)
        runner.save_checkpoint(output/'best.pt', 0, progress)
        runner.save_checkpoint(output/'latest.pt', 0, progress)
    end = min(args.episodes, completed + args.stop_after) if args.stop_after is not None else args.episodes
    for episode in range(completed + 1, end + 1):
        write_json(output/'status.json', dict(phase='collection', completed_episodes=episode-1,
                   current_episode=episode, total_episodes=args.episodes))
        set_episode_entropy(runner, episode)
        set_episode_prior(runner, episode)
        archive = output/f'episodes/train/episode_{episode:06d}.npz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        rows, metrics = runner.run_episode(policy='stochastic', seed=args.seed+episode, collect=True, archive_path=archive)
        write_json(output/'status.json', dict(phase='update', completed_episodes=episode-1, current_episode=episode,
                                             total_episodes=args.episodes))
        metrics.update(runner.update(rows), episode=episode)
        if episode % args.eval_every == 0 or episode == args.episodes:
            write_json(output/'status.json', dict(phase='validation', current_episode=episode,
                       completed_episodes=episode-1, total_episodes=args.episodes))
            validation = evaluate(runner, output/f'evaluation/actor_only/episode_{episode:06d}', validation_seeds,
                                  policy=actor_evaluation_policy)
            progress['validation'].append(dict(episode=episode, evaluation_policy=actor_evaluation_policy, rows=validation,
                summary=summarize(validation), beats_reference=beats_reference(validation, progress['initial'])))
            if score(validation) > tuple(progress['best_score']):
                progress.update(best_score=list(score(validation)), best_episode=episode)
                runner.save_checkpoint(output/'best.pt', episode, progress)
        runner.save_checkpoint(output/'latest.pt', episode, progress)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + '\n')
        print(json.dumps(dict(episode=episode, ATT_seconds=metrics['observed_mean_seconds'],
            completion=metrics['completion_rate'], actor_loss=metrics['actor_loss'],
            critic_loss=metrics['global_critic_loss'], learning_seconds=metrics['learning_seconds'])), flush=True)
        completed = episode
    if completed == args.episodes:
        write_json(output/'status.json', dict(phase='holdout', completed_episodes=completed, total_episodes=args.episodes))
        seeds = list(range(20042, 20047))
        reference = evaluate(runner, output/'holdout/prior', seeds, policy='prior')
        final = evaluate(runner, output/'holdout/final', seeds, policy=actor_evaluation_policy)
        runner.save_checkpoint(output/'final.pt', completed, progress)
        best = final
        if progress['best_episode'] != completed:
            runner.load_checkpoint(output/'best.pt')
            best = evaluate(runner, output/'holdout/best', seeds, policy=actor_evaluation_policy)
        write_json(output/'holdout/results.json', dict(evaluation_policy=actor_evaluation_policy,
            reference_evaluation_policy='prior', prior=reference, final=final, best=best,
            best_episode=progress['best_episode'], final_beats_reference=beats_reference(final, reference),
            best_beats_reference=beats_reference(best, reference)))
    write_json(output/'status.json', dict(phase='completed' if completed == args.episodes else 'paused_slice',
        completed_episodes=completed, total_episodes=args.episodes, best_episode=progress['best_episode']))


if __name__ == '__main__':
    main()
