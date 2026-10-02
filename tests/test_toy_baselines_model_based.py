from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines import build_method, common
from urban_dreamer.learning.baselines.dreamer import (DETERMINISTIC_DIM, STOCHASTIC_DIM,
                                                      DreamerMethod)
from urban_dreamer.learning.baselines.mbpo import ENSEMBLE_SIZE, MbpoMethod
from urban_dreamer.learning.replay import MacroTransition

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/train_toy_baseline.py'


def module():
    spec = importlib.util.spec_from_file_location('toy_baseline_tool', TOOL)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_args(tmp_path, method='dreamer', **overrides):
    values = dict(method=method, config=ROOT / 'configs/toy_cross_bottleneck.toml',
        output_dir=tmp_path, episodes=1, eval_every=1, eval_seeds=1, seed=42, device='cpu',
        threads=1, actor_lr=2e-4, critic_lr=3e-4, critic_updates=1, vehicle_limit=8, resume=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def sumo_available():
    return shutil.which('sumo') is not None or (Path(sys.executable).parent / 'sumo').exists()


def active_snapshot(runner):
    network = runner.network
    demand = runner.topology.vehicles[0]
    current = network.edge_to_index[runner.topology.origins[demand.origin_index]]
    destination = network.edge_to_index[runner.topology.destinations[demand.destination_index]]
    return dict(vehicle_id=demand.vehicle_id, current_edge=current, destination_edge=destination,
        departure_time=0., departed=True, arrived=False,
        remaining_distance=float(network.lengths_m[current]), committed_next_edge=None,
        pending_decision=True, in_junction=False, junction_remaining_distance=0., junction_speed=1.)


def synthetic_episode(runner, rng, *, steps=5, episode=0, vehicles=()):
    edge_count = runner.network.edge_count
    flows = [rng.random((edge_count, 8), dtype=np.float32) for _ in range(steps + 1)]
    transitions = []
    for step in range(steps):
        transitions.append(MacroTransition(flow=flows[step], next_flow=flows[step + 1],
            action=rng.random(edge_count, dtype=np.float32) * 0.01,
            departure=rng.random(edge_count, dtype=np.float32) * 0.01,
            context=np.asarray([step / steps, 0.5], dtype=np.float32),
            active_vehicle_fraction=0.5, vehicles=vehicles, time=30. * step,
            terminal=step == steps - 1, episode=episode, step_index=step))
    return transitions


def synthetic_window(runner, rng, *, time, terminal, vehicles=()):
    edge_count = runner.network.edge_count
    flow = rng.random((edge_count, 8), dtype=np.float32)
    next_flow = np.clip(flow + rng.normal(0., 0.01, (edge_count, 8)), 0., 1.5).astype(np.float32)
    context = np.asarray([time / runner.deadline, 0.5], dtype=np.float32)
    next_context = np.asarray([(time + 30.) / runner.deadline, 0.5], dtype=np.float32)
    return dict(flow=flow, next_flow=next_flow,
        action=np.zeros(edge_count, dtype=np.float32), departure=np.zeros(edge_count, dtype=np.float32),
        context=context,
        obs=np.concatenate((flow[:, :4].reshape(-1), context)).astype(np.float32),
        next_obs=np.concatenate((next_flow[:, :4].reshape(-1), next_context)).astype(np.float32),
        reward=-0.25, terminal=terminal, vehicles=vehicles, time=float(time),
        active_vehicle_seconds=0.5 * runner.n * 30., total_vehicle_count=runner.n)


@pytest.mark.parametrize('name', ['dreamer', 'mbpo'])
def test_imagination_uses_seed_population_instead_of_latest_episode(tmp_path, monkeypatch, name):
    from types import SimpleNamespace
    import urban_dreamer.learning.baselines.dreamer as dreamer
    import urban_dreamer.learning.baselines.mbpo as mbpo
    runner = common.EpisodeRunner(make_args(tmp_path, method=name))
    method = build_method(name, runner.args, runner)
    vehicle = active_snapshot(runner)
    flow = runner.network.extract_flow({})
    timestamp = runner.deadline - 30.
    captured = []

    def advance(states, **kwargs):
        captured.append(kwargs['total_vehicle_count'])
        return SimpleNamespace(action=np.zeros(runner.network.edge_count),
                               departure=np.zeros(runner.network.edge_count))

    runner.n = 100
    if name == 'dreamer':
        transition = MacroTransition(flow=flow, next_flow=flow, action=np.zeros(runner.network.edge_count),
            departure=np.zeros(runner.network.edge_count), context=np.array([timestamp / method.deadline, .5]),
            active_vehicle_fraction=.5, vehicles=(vehicle,), time=timestamp,
            terminal=False, episode=0, step_index=0)
        method.seed_buffer.append_episode([transition], total_vehicle_count=2, total_travel_minutes=1.)
        monkeypatch.setattr(dreamer, 'advance_imagined_window', advance)
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: flow.copy())
        contexts = []
        monkeypatch.setattr(method, '_prior_dynamics_step',
                            lambda h, z, action, departure, context: contexts.append(context) or (h, z))
        method._imagine_rollout(0)
        assert contexts[0][1] == pytest.approx(.5)
    else:
        window = synthetic_window(runner, np.random.default_rng(0), time=timestamp,
                                  terminal=False, vehicles=(vehicle,))
        window['total_vehicle_count'] = 2
        monkeypatch.setattr(mbpo, 'advance_imagined_window', advance)
        contexts = []
        monkeypatch.setattr(method, '_predict_transition',
                            lambda flow, action, departure, context: contexts.append(context) or (flow, 0.))
        method._rollout_from_seed(window, 0)
        assert contexts[0][1] == pytest.approx(.5)
    assert captured == [2]


def assert_actor_rows(rows):
    """Imagined rows carry only what the actor update needs, plus rollout bookkeeping."""
    assert rows, 'expected at least one imagined event row'
    for row in rows:
        for key in ('rollout', 'window', 'features', 'mask', 'action', 'current_edge',
                    'edge_indices', 'flow'):
            assert key in row, key
        assert row['features'].shape[-1] == 10
        assert row['mask'].any()
        assert 0 <= row['action'] < row['mask'].shape[0]
        assert row['mask'][row['action']]
        assert row['edge_indices'][row['action']] >= 0


@pytest.mark.parametrize('name', ['dreamer', 'mbpo'])
@pytest.mark.parametrize('scenario', ['deadline', 'truncated', 'deadline_arrival', 'mixed_deadline'])
def test_imagined_individual_timeout_penalty_only_at_real_deadline(tmp_path, monkeypatch, name, scenario):
    from types import SimpleNamespace
    import urban_dreamer.learning.baselines.dreamer as dreamer
    import urban_dreamer.learning.baselines.mbpo as mbpo
    runner = common.EpisodeRunner(make_args(tmp_path, method=name))
    method = build_method(name, runner.args, runner)
    timestamp = 0. if scenario == 'truncated' else method.deadline - 30.
    vehicle = active_snapshot(runner)
    cohort = (vehicle,)
    if scenario == 'mixed_deadline':
        cohort = (vehicle, dict(vehicle, vehicle_id='arrived_at_deadline'))
    flow = runner.network.extract_flow({})
    captured = []

    def advance(states, **kwargs):
        kwargs['decide'](states, kwargs['frozen_flow'], kwargs['start_time'])
        if scenario == 'mixed_deadline':
            states[1]['arrived'] = True
        if scenario == 'deadline_arrival':
            for state in states:
                state['arrived'] = True
        return SimpleNamespace(action=np.zeros(runner.network.edge_count),
                               departure=np.zeros(runner.network.edge_count))

    if name == 'dreamer':
        transition = MacroTransition(flow=flow, next_flow=flow, action=np.zeros(runner.network.edge_count),
            departure=np.zeros(runner.network.edge_count), context=np.array([timestamp / method.deadline, 1. / runner.n]),
            active_vehicle_fraction=1. / runner.n, vehicles=cohort, time=timestamp,
            terminal=False, episode=0, step_index=0)
        method.seed_buffer.append_episode([transition], total_vehicle_count=runner.n, total_travel_minutes=1.)
        monkeypatch.setattr(dreamer, 'advance_imagined_window', advance)
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: flow.copy())
        monkeypatch.setattr(method, '_prior_dynamics_step', lambda h, z, *args: (h, z))
        monkeypatch.setattr(method.world_model, 'predict_reward', lambda h, z: torch.tensor([-.25]))
        rows, trajectory = method._imagine_rollout(0)
        rewards, terminal = trajectory[2:]
    else:
        monkeypatch.setattr(mbpo, 'advance_imagined_window', advance)
        monkeypatch.setattr(method, '_predict_transition', lambda *args: (flow.copy(), -.25))
        def critic_update(**kwargs):
            captured.extend(kwargs['windows'])
            return torch.zeros(len(kwargs['windows'])), {}
        monkeypatch.setattr(mbpo, 'global_critic_update', critic_update)
        window = synthetic_window(runner, np.random.default_rng(0), time=timestamp,
                                  terminal=False, vehicles=cohort)
        rows = method._rollout_from_seed(window, 0)
        rewards = [window['reward'] for window in captured]
        terminal = [window['terminal'] for window in captured]
    if scenario == 'mixed_deadline':
        assert rewards == [-.25] and terminal == [True]
        assert sum(row['timeout_penalty'] for row in rows) == -5000.
        assert [row['timeout_return'] for row in rows] == [-5000., 0.]
    elif scenario == 'deadline':
        assert rewards == [-.25] and terminal == [True]
        assert sum(row['timeout_penalty'] for row in rows) == -5000.
        assert all(row['timeout_return'] == -5000. for row in rows)
    elif scenario == 'deadline_arrival':
        assert rewards == [-.25] and terminal == [True]
        assert all(row['timeout_return'] == 0. for row in rows)
    else:
        assert all(reward == -.25 for reward in rewards)
        assert not terminal[-1]
        assert all(row['timeout_return'] == 0. for row in rows)


def test_mbpo_model_does_not_learn_the_separate_timeout_penalty(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path, method='mbpo'))
    method = MbpoMethod(runner.args, runner)
    window = synthetic_window(runner, np.random.default_rng(0), time=runner.deadline - 30., terminal=True)
    window.update(reward=-.25)
    assert method._model_sample(window)['reward'] == -.25


def test_rssm_sequence_loss_gradients_and_latent_step(tmp_path):
    import urban_dreamer.learning.baselines.dreamer as dreamer
    args = make_args(tmp_path)
    runner = common.EpisodeRunner(args)
    method = DreamerMethod(args, runner)
    rng = np.random.default_rng(0)
    for episode in range(2):
        method.sequence_buffer.append_episode(
            synthetic_episode(runner, rng, episode=episode), total_vehicle_count=runner.n, total_travel_minutes=1.)
    batch = method.sequence_buffer.sample_batch(2, method.rng)
    loss, parts = method._world_model_loss(batch)
    assert math.isfinite(float(loss.detach()))
    for key in ('world_model_loss', 'reconstruction_loss', 'reward_loss', 'kl'):
        assert math.isfinite(parts[key]), key
    method.world_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in method.world_model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    h, z = method._initial_latent(batch.flow[0, 0].numpy())
    assert h.shape == (1, DETERMINISTIC_DIM) and z.shape == (1, STOCHASTIC_DIM)
    edge_count = runner.network.edge_count
    h_next, z_next = method._prior_dynamics_step(h, z, np.zeros(edge_count, dtype=np.float32),
        np.zeros(edge_count, dtype=np.float32), np.zeros(2, dtype=np.float32))
    assert h_next.shape == h.shape and z_next.shape == z.shape
    assert torch.isfinite(h_next).all() and torch.isfinite(z_next).all()
    decoded = method._decode_flow(h_next, z_next)
    assert decoded.shape == (edge_count, 8) and np.isfinite(decoded).all()
    assert dreamer.IMAGINED_HORIZON >= 1


def test_ensemble_member_nll_shapes_and_logvar_clamp(tmp_path):
    args = make_args(tmp_path, method='mbpo')
    runner = common.EpisodeRunner(args)
    method = MbpoMethod(args, runner)
    assert len(method.ensemble) == ENSEMBLE_SIZE
    edge_count = runner.network.edge_count
    member = method.ensemble[0]
    in_dim = edge_count * 8 + 2 * edge_count + 2
    x = torch.randn(4, in_dim) * 100.
    mean, logvar, reward = member(x)
    logvar = logvar.detach()
    assert mean.shape == (4, edge_count * 8)
    assert logvar.shape == (4, edge_count * 8)
    assert reward.shape == (4,)
    assert float(logvar.min()) >= -8. and float(logvar.max()) <= 2.
    target = torch.randn(4, edge_count * 8)
    nll = member.gaussian_nll(x[:2], target[:2])
    assert torch.isfinite(nll)
    nll.backward()
    gradients = [p.grad for p in member.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_dreamer_rollout_survives_sub_tolerance_junction_residue(tmp_path, monkeypatch):
    """Regression: a mid-junction residue in (tol, speed*tol] stalled the imagined clock."""
    import urban_dreamer.learning.baselines.dreamer as dreamer
    monkeypatch.setattr(dreamer, 'IMAGINED_HORIZON', 1)
    args = make_args(tmp_path)
    runner = common.EpisodeRunner(args)
    method = DreamerMethod(args, runner)
    network = runner.network
    snapshot = None
    for demand in runner.topology.vehicles:
        current = network.edge_to_index[runner.topology.origins[demand.origin_index]]
        destination = network.edge_to_index[runner.topology.destinations[demand.destination_index]]
        choices = runner.candidates.candidates(current, destination)
        if choices and choices[0] != destination:
            # Field values captured from the resume crash dump (vehicle '96').
            snapshot = dict(vehicle_id=demand.vehicle_id, current_edge=current,
                destination_edge=destination, departure_time=0., departed=True, arrived=False,
                remaining_distance=0., committed_next_edge=int(choices[0]), pending_decision=False,
                in_junction=True, junction_remaining_distance=4.02480182515319e-09,
                junction_speed=8.75)
            break
    assert snapshot is not None
    transitions = synthetic_episode(runner, np.random.default_rng(0), steps=1, vehicles=(snapshot,))
    method.seed_buffer.append_episode(transitions, total_vehicle_count=runner.n, total_travel_minutes=1.)
    rows, _ = method._imagine_rollout(0)  # raised 'imagined vehicle clock failed to advance'
    assert_actor_rows(rows)


def test_dreamer_imagined_rows_and_advantage_broadcast(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.dreamer as dreamer
    monkeypatch.setattr(dreamer, 'IMAGINED_HORIZON', 1)
    monkeypatch.setattr(dreamer, 'IMAGINATION_ROLLOUTS', 2)
    args = make_args(tmp_path)
    runner = common.EpisodeRunner(args)
    method = DreamerMethod(args, runner)
    transitions = synthetic_episode(runner, np.random.default_rng(0), steps=1,
                                    vehicles=(active_snapshot(runner),))
    method.seed_buffer.append_episode(transitions, total_vehicle_count=runner.n, total_travel_minutes=1.)
    assert method.seed_buffer.has_active_seeds
    rows, trajectory = method._imagine_rollout(0)
    assert_actor_rows(rows)
    assert all(row['rollout'] == 0 and row['window'] == 0 for row in rows)
    latents, targets, advantages = method._score_rollout(trajectory)
    assert latents.shape == (1, DETERMINISTIC_DIM + STOCHASTIC_DIM)
    assert targets.shape == advantages.shape == (1,)
    assert torch.isfinite(targets).all() and torch.isfinite(advantages).all()
    calls = []
    original = dreamer.policy_gradient_step

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(dreamer, 'policy_gradient_step', spy)
    metrics = method._imagination_pass()
    assert math.isfinite(metrics['latent_critic_loss']) and math.isfinite(metrics['actor_loss'])
    assert len(calls) == 1
    batch_rows, batch_advantages = calls[0]['rows'], calls[0]['advantages']
    assert len(batch_rows) == len(batch_advantages)
    assert metrics['imagined_events'] == float(len(batch_rows))
    grouped = {}
    for row, advantage in zip(batch_rows, batch_advantages):
        grouped.setdefault((row['rollout'], row['window']), set()).add(float(advantage))
    # one shared team advantage per (rollout, window) for every decision in that window
    assert grouped and all(len(values) == 1 for values in grouped.values())


def test_mbpo_imagined_rows_semantics(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.mbpo as mbpo
    monkeypatch.setattr(mbpo, 'ROLLOUT_LENGTH', 1)
    args = make_args(tmp_path, method='mbpo')
    runner = common.EpisodeRunner(args)
    method = MbpoMethod(args, runner)
    window = dict(flow=runner.network.extract_flow({}), vehicles=(active_snapshot(runner),), time=0.)
    rows = method._rollout_from_seed(window, 3)
    assert_actor_rows(rows)
    assert all(row['rollout'] == 3 and row['window'] == 0 for row in rows)
    assert all(math.isfinite(row['advantage']) for row in rows)
    edge_count = runner.network.edge_count
    next_flow, reward = method._predict_transition(window['flow'],
        np.zeros(edge_count, dtype=np.float32), np.zeros(edge_count, dtype=np.float32),
        np.zeros(2, dtype=np.float32))
    assert next_flow.shape == window['flow'].shape and np.isfinite(next_flow).all()
    assert 0.05 <= float(next_flow[:, 1].min()) and float(next_flow[:, 1].max()) <= 1.5
    assert math.isfinite(reward)


def test_mbpo_global_critic_trains_only_on_real_windows(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.mbpo as mbpo
    args = make_args(tmp_path, method='mbpo')
    runner = common.EpisodeRunner(args)
    method = MbpoMethod(args, runner)
    rng = np.random.default_rng(0)
    snapshot = active_snapshot(runner)
    windows = [synthetic_window(runner, rng, time=30. * index, terminal=index == 2,
                                vehicles=(snapshot,) if index == 0 else ())
               for index in range(3)]
    calls = []
    original = mbpo.global_critic_update

    def spy(**kwargs):
        calls.append(dict(update=kwargs['update'], count=len(kwargs['windows'])))
        return original(**kwargs)

    monkeypatch.setattr(mbpo, 'global_critic_update', spy)
    before = [parameter.detach().clone() for parameter in method.global_critic.parameters()]
    metrics = method.on_episode([], windows, episode=1)
    assert calls[0] == dict(update=True, count=3)
    assert calls[1:] and all(call['update'] is False for call in calls[1:])
    parameters = list(method.global_critic.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, parameters))
    assert math.isfinite(metrics['global_critic_loss']) and math.isfinite(metrics['actor_loss'])
    assert metrics['imagined_events'] > 0


@pytest.mark.parametrize('method_name,method_cls', [('dreamer', DreamerMethod), ('mbpo', MbpoMethod)])
def test_model_based_checkpoint_roundtrip(tmp_path, method_name, method_cls):
    args = make_args(tmp_path, method=method_name)
    runner = common.EpisodeRunner(args)
    method = method_cls(args, runner)
    rng = np.random.default_rng(0)
    if method_name == 'dreamer':
        transitions = synthetic_episode(runner, rng, steps=2, vehicles=(active_snapshot(runner),))
        method.sequence_buffer.append_episode(transitions, total_vehicle_count=runner.n, total_travel_minutes=1.)
        method.seed_buffer.append_episode(transitions, total_vehicle_count=runner.n, total_travel_minutes=1.)
    else:
        method.model_rows.append(dict(rollout=0, window=0, advantage=-0.5,
            features=np.zeros((1, 10), dtype=np.float32), mask=np.array([True]), action=0,
            current_edge=0, edge_indices=np.zeros(1, dtype=np.int64),
            flow=np.zeros((runner.network.edge_count, 8), dtype=np.float32)))
    original = copy.deepcopy(method.state_dict())
    path = tmp_path / 'ckpt.pt'
    common.save_checkpoint(path, method_name=method_name, episode=2, best=3.5,
        method_state=method.state_dict(), args=args, prepared=runner.prepared, config=runner.config)
    with torch.no_grad():
        for parameter in method.actor.parameters():
            parameter.add_(1.)
    state = common.load_checkpoint(path, method_name=method_name,
        prepared=runner.prepared, config=runner.config, device=runner.device)
    assert state['episode'] == 2 and state['best'] == 3.5
    method.load_state_dict(state['method_state'])
    for key, value in original['actor'].items():
        torch.testing.assert_close(method.actor.state_dict()[key], value, rtol=0, atol=0)
    if method_name == 'dreamer':
        for key, value in original['world_model'].items():
            torch.testing.assert_close(method.world_model.state_dict()[key], value, rtol=0, atol=0)
        for key, value in original['latent_critic'].items():
            torch.testing.assert_close(method.latent_critic.state_dict()[key], value, rtol=0, atol=0)
        assert len(method.sequence_buffer.episodes) == 1
        assert len(method.seed_buffer.episodes) == 1
    else:
        for key, value in original['ensemble'].items():
            torch.testing.assert_close(method.ensemble.state_dict()[key], value, rtol=0, atol=0)
        for key, value in original['global_critic'].items():
            torch.testing.assert_close(method.global_critic.state_dict()[key], value, rtol=0, atol=0)
        assert len(method.model_rows) == 1
    assert method.rng.bit_generator.state['state']['state'] == original['rng']['state']['state']


@pytest.mark.parametrize('method_name', ['dreamer', 'mbpo'])
def test_model_based_cli_smoke(tmp_path, method_name):
    if not sumo_available():
        pytest.skip('SUMO unavailable')
    tool = module()
    output = tmp_path / 'run'
    status = tool.main(['--method', method_name, '--output-dir', str(output), '--episodes', '2',
                        '--eval-every', '1', '--eval-seeds', '1', '--vehicle-limit', '8'])
    assert status == 0
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    train = [record for record in records if record['phase'] == 'train']
    assert len(train) == 2
    for record in train:
        assert math.isfinite(record['global_reward_sum'])
        assert math.isfinite(record['actor_loss'])
        if method_name == 'dreamer':
            for key in ('world_model_loss', 'kl', 'reward_loss', 'latent_critic_loss',
                        'imagined_events'):
                assert math.isfinite(record[key]), key
        else:
            for key in ('model_nll', 'reward_loss', 'global_critic_loss', 'model_rows',
                        'imagined_events'):
                assert math.isfinite(record[key]), key
    assert {record['policy'] for record in records if record['phase'] == 'baseline'} \
        == {'random', 'shortest', 'deterministic'}
    assert any(record['phase'] == 'holdout_best' for record in records)
    for name in ('best.pt', 'initial.pt', 'latest.pt'):
        assert (output / name).is_file()
