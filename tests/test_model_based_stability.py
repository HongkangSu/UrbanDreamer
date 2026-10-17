"""Regression checks for model-based learning signals and policy freshness."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_route_model_based_baselines import collect_toy_episode, make_method
from urban_dreamer.learning.replay import MacroTransition


def one_step_batch(method):
    edge_count = method.edge_count
    transition = MacroTransition(flow=np.zeros((edge_count, 8), dtype=np.float32),
        next_flow=np.ones((edge_count, 8), dtype=np.float32),
        action=np.ones(edge_count, dtype=np.float32), departure=np.zeros(edge_count, dtype=np.float32),
        context=np.asarray([0., 1., 1., 0.], dtype=np.float32),
        active_vehicle_fraction=1., global_reward=-.5, time=0., terminal=True,
        episode=0, step_index=0)
    method.sequence_buffer.append_episode([transition], total_travel_minutes=.5, total_vehicle_count=1)
    return method.sequence_buffer.sample_batch(1, method.rng)


def test_dreamer_seed_uses_the_observation_posterior():
    method = make_method('dreamer')
    flow = method.runner.network.extract_flow({})
    torch.manual_seed(15)
    h, z = method._initial_latent(flow)
    expected_h = torch.zeros_like(h)
    with torch.no_grad():
        encoded = method.world_model.encoder(torch.as_tensor(flow).float().reshape(1, -1))
        mean, std = method.world_model.posterior_params(expected_h, encoded)
    torch.manual_seed(15)
    expected_z = mean + std * torch.randn_like(mean)
    torch.testing.assert_close(h, expected_h)
    torch.testing.assert_close(z, expected_z)


def test_dreamer_terminal_next_flow_is_supervised():
    method = make_method('dreamer')
    batch = one_step_batch(method)
    torch.manual_seed(7)
    _, before = method._world_model_loss(batch)
    changed = copy.deepcopy(batch)
    changed.flow[:, -1].fill_(4.)
    torch.manual_seed(7)
    _, after = method._world_model_loss(changed)
    assert after['reconstruction_loss'] > before['reconstruction_loss'] + 1.


def test_dreamer_immediate_reward_training_conditions_on_current_action():
    method = make_method('dreamer')
    batch = one_step_batch(method)
    torch.manual_seed(7)
    _, before = method._world_model_loss(batch)
    changed = copy.deepcopy(batch)
    changed.action.zero_()
    torch.manual_seed(7)
    _, after = method._world_model_loss(changed)
    assert abs(after['reward_loss'] - before['reward_loss']) > 1e-7


def test_dreamer_free_nats_apply_to_the_joint_latent(monkeypatch):
    method = make_method('dreamer')
    batch = one_step_batch(method)
    def prior(h):
        mean = h.new_zeros((len(h), 32))
        return mean, torch.ones_like(mean)
    def posterior(h, encoded):
        mean = h.new_full((len(h), 32), .4)
        return mean, torch.ones_like(mean)
    monkeypatch.setattr(method.world_model, 'prior_params', prior)
    monkeypatch.setattr(method.world_model, 'posterior_params', posterior)
    _, metrics = method._world_model_loss(batch)
    assert abs(metrics['kl'] - 32 * .4 ** 2 / 2) < 1e-5


def test_dreamer_imagined_reward_uses_action_successor(monkeypatch):
    method = make_method('dreamer')
    method.args.dreamer_imagined_horizon = 1
    _, windows = collect_toy_episode(method)
    window = windows[0]
    method.seed_buffer.append_episode([MacroTransition(flow=window['flow'], next_flow=window['next_flow'],
        action=window['action'], departure=window['departure'], context=window['context'],
        vehicles=window['vehicles'], active_vehicle_fraction=1., global_reward=-.5,
        time=0., terminal=True, episode=0, step_index=0)], total_travel_minutes=.5, total_vehicle_count=1)
    monkeypatch.setattr(method, '_prior_dynamics_step', lambda h, z, *args: (torch.full_like(h, 2.), z))
    monkeypatch.setattr(method.world_model, 'predict_reward', lambda h, z, context=None: h[:, :1])
    _, trajectory = method._imagine_rollout(0)
    assert trajectory[2] == [2.]


def test_dreamer_real_actor_learns_when_imagination_has_no_seeds(monkeypatch):
    method = make_method('dreamer')
    rows, windows = collect_toy_episode(method)
    before = [p.detach().clone() for p in method.actor.parameters()]
    monkeypatch.setattr(method, '_train_world_model', lambda: {})
    monkeypatch.setattr(method, '_train_imagination', lambda: dict(
        latent_critic_loss=0., actor_loss=0., imagined_events=0.,
        imagination_seconds=0., actor_update_seconds=0., critic_update_seconds=0.))
    metrics = method.on_episode(rows, windows, episode=0)
    assert np.isfinite(metrics['actor_loss'])
    assert any(not torch.equal(old, new) for old, new in zip(before, method.actor.parameters()))


def test_mbpo_discards_stale_synthetic_actions_before_new_episode(monkeypatch):
    method = make_method('mbpo')
    _, windows = collect_toy_episode(method)
    method.model_rows.append(dict(stale=True))
    monkeypatch.setattr(method, '_train_model', lambda: {})
    observed = []
    monkeypatch.setattr(method, '_generate_model_rows', lambda windows: observed.append(list(method.model_rows)) or 0)
    method.on_episode([], windows, episode=1)
    assert observed == [[]]


def test_mbpo_does_not_repeat_uncorrected_policy_gradient(monkeypatch):
    import urban_dreamer.learning.baselines.mbpo as mbpo
    method = make_method('mbpo')
    rows, windows = collect_toy_episode(method)
    calls = []
    monkeypatch.setattr(mbpo, 'policy_gradient_step', lambda **kwargs: calls.append(kwargs) or dict(actor_loss=0.))
    method._policy_updates(rows, torch.ones(len(windows)))
    assert len(calls) == 1


def test_model_based_imagination_budgets_are_configurable():
    method = make_method('dreamer')
    method.args.dreamer_imagination_passes = 2
    method.args.dreamer_imagination_rollouts = 1
    method.args.dreamer_imagined_horizon = 1
    protocol = method.learning_protocol()
    assert protocol['imagination_passes'] == 2
    assert protocol['rollouts_per_pass'] == 1
    assert protocol['horizon_windows'] == 1


@pytest.mark.parametrize('name', ['mbpo', 'dreamer'])
def test_old_learning_protocol_cannot_silently_resume(name):
    method = make_method(name)
    state = method.state_dict()
    assert state['learning_protocol'] == method.learning_protocol()
    state.pop('learning_protocol')
    with pytest.raises(ValueError, match='learning protocol changed'):
        method.load_state_dict(state)


@pytest.mark.parametrize('name,knob', [('mbpo', 'mbpo_rollout_length'), ('dreamer', 'dreamer_imagination_passes')])
@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_direct_constructor_rejects_invalid_learning_budget(name, knob, value):
    method = make_method(name)
    setattr(method.args, knob, value)
    with pytest.raises(ValueError, match='positive integer'):
        type(method)(method.args, method.runner)


@pytest.mark.parametrize('name', ['dreamer', 'mbpo'])
@pytest.mark.parametrize('timestamp,terminal', [(0., False), (30., True)])
def test_seed_cohort_arrival_only_terminates_at_the_physical_deadline(monkeypatch, name, timestamp, terminal):
    implementation = __import__(f'urban_dreamer.learning.baselines.{name}', fromlist=['x'])
    method = make_method(name)
    _, windows = collect_toy_episode(method)
    window = dict(windows[0], time=timestamp)
    def arrive(states, **kwargs):
        for state in states:
            state['arrived'] = True
        return SimpleNamespace(action=np.zeros(method.edge_count), departure=np.zeros(method.edge_count),
            active_vehicle_seconds=1.)
    monkeypatch.setattr(implementation, 'advance_imagined_window', arrive)
    if name == 'dreamer':
        method.seed_buffer.append_episode([MacroTransition(flow=window['flow'], next_flow=window['next_flow'],
            action=window['action'], departure=window['departure'], context=window['context'],
            vehicles=window['vehicles'], active_vehicle_fraction=1., global_reward=-.5,
            time=timestamp, terminal=False, episode=0, step_index=0)], total_travel_minutes=.5, total_vehicle_count=1)
        _, trajectory = method._imagine_rollout(0)
        assert trajectory[3] == [terminal]
    else:
        scored_windows = []
        monkeypatch.setattr(implementation, 'global_critic_update',
            lambda **kwargs: (scored_windows.extend(kwargs['windows']) or torch.zeros(len(kwargs['windows'])), {}))
        method._rollout_from_seed(window, 0)
        assert [window['terminal'] for window in scored_windows] == [terminal]
