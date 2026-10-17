"""Accepted PPO updates honor whole-rollout trust limits and real-time traces."""
from copy import deepcopy
import math

import numpy as np
import pytest
import torch
from torch import nn

from test_routing_ppo_baselines import data, method
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod, _local_lambda_targets
from urban_dreamer.models.policy import lambda_return


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_ippo_trace_depends_on_elapsed_time_not_inserted_decisions():
    windows = [dict(start_time=0., end_time=30., reward=0.),
               dict(start_time=30., end_time=60., reward=-10.)]
    sparse = [dict(vehicle_id='v', time=t) for t in (0., 30.)]
    dense = [dict(vehicle_id='v', time=float(t)) for t in range(31)]
    a = _local_lambda_targets(sparse, torch.zeros(2), windows, deadline=60.)
    b = _local_lambda_targets(dense, torch.zeros(31), windows, deadline=60.)
    assert a[0] == pytest.approx(-9.5)
    torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=1e-6)


def test_ippo_same_timestamp_decision_does_not_decay_team_tail():
    windows = [dict(start_time=0., end_time=30., reward=0.),
               dict(start_time=30., end_time=60., reward=-10.)]
    rows = [dict(vehicle_id='v', time=t) for t in (0., 0., 30.)]
    actual = _local_lambda_targets(rows, torch.zeros(3), windows, deadline=60.)
    torch.testing.assert_close(actual, torch.tensor([-9.5, -9.5, -10.]))


def test_regular_macro_decisions_keep_macro_lambda_returns():
    windows = [dict(start_time=30. * i, end_time=30. * (i + 1), reward=-float(i + 1))
               for i in range(3)]
    rows = [dict(vehicle_id='v', time=float(i * 30)) for i in range(3)]
    values = torch.tensor([.2, -.4, .3])
    expected = lambda_return(torch.tensor([-1., -2., -3.]), torch.tensor([-.4, .3, 0.]),
                             terminal=torch.tensor([False, False, True]))
    torch.testing.assert_close(_local_lambda_targets(rows, values, windows, deadline=90.), expected)


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
@pytest.mark.parametrize('primed', [False, True])
def test_real_adam_overshoot_is_rolled_back_with_optimizer(cls, primed):
    model = method(cls)
    if primed:
        model.on_episode(*data(model), episode=1)
        model.target_kl, model.max_policy_kl = 1e-8, 1e-7
    model.actor_optimizer.param_groups[0]['lr'] = .1
    model.max_backtracks = 0
    rows, windows = data(model)
    actor = deepcopy(model.actor.state_dict())
    optimizer = deepcopy(model.actor_optimizer.state_dict())
    metrics = model.on_episode(rows, windows, episode=1)
    assert metrics['policy_kl'] <= model.target_kl + 1e-7
    assert metrics['actor_optimizer_steps'] == 0
    torch.testing.assert_close(model.actor.state_dict(), actor, rtol=0, atol=0)
    torch.testing.assert_close(model.actor_optimizer.state_dict(), optimizer, rtol=0, atol=0)


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_backtracking_can_accept_a_smaller_step_without_changing_base_lr(cls):
    model = method(cls)
    model.actor_optimizer.param_groups[0]['lr'] = .03
    metrics = model.on_episode(*data(model), episode=1)
    assert metrics['actor_rejected_updates'] > 0
    assert metrics['actor_optimizer_steps'] > 0
    assert 0 < metrics['actor_lr_scale'] < 1.
    assert metrics['policy_kl'] <= model.target_kl
    assert metrics['policy_kl_max'] <= model.max_policy_kl
    assert model.actor_optimizer.param_groups[0]['lr'] == .03
    steps = [float(state['step']) for state in model.actor_optimizer.state.values()]
    assert max(steps) == metrics['actor_optimizer_steps']


def test_optimizer_exception_restores_actor_and_moments(monkeypatch):
    model = method(MappoMethod)
    model.on_episode(*data(model), episode=1)
    rows, windows = data(model)
    before = deepcopy(model.actor.state_dict())
    optimizer = deepcopy(model.actor_optimizer.state_dict())
    original = model.actor_optimizer.step
    def failing_step(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected optimizer failure')
    monkeypatch.setattr(model.actor_optimizer, 'step', failing_step)
    with pytest.raises(RuntimeError, match='injected optimizer failure'):
        model.on_episode(rows, windows, episode=2)
    torch.testing.assert_close(model.actor.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(model.actor_optimizer.state_dict(), optimizer, rtol=0, atol=0)


class RareStateActor(nn.Module):
    prior_weight = 0.
    prior_temperature_seconds = 30.
    score_gauge = 'legacy'

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward_logits(self, features, mask):
        return (features[..., 0] * self.weight).masked_fill(~mask, -torch.inf)


def test_rare_state_kl_cannot_hide_under_small_average(monkeypatch):
    model = method(MappoMethod)
    model.actor = RareStateActor()
    model.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=10.)
    model.ppo_epochs, model.minibatch_size = 1, 2000
    model.max_policy_kl, model.max_backtracks = .1, 0
    rows = [dict(features=np.zeros((2, 10), np.float32), mask=np.ones(2, bool),
        action=0, old_log_prob=-math.log(2.)) for _ in range(2000)]
    rows[0]['features'][0, 0] = 1.
    advantages = torch.zeros(2000)
    advantages[0] = 1.
    monkeypatch.setattr(model, '_targets', lambda *a: (torch.zeros(1), advantages, (torch.zeros(1, 10),)))
    monkeypatch.setattr(model, '_fit_critic', lambda *a: {})
    metrics = model.on_episode(rows, [{}], episode=1)
    assert metrics['policy_kl_max'] <= .1
    assert model.actor.weight == 0.
    assert not model.actor_optimizer.state


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_maximum_entropy_zero_gradient_preserves_momentum_state(cls, monkeypatch):
    model = method(cls)
    model.on_episode(*data(model), episode=1)
    rows, windows = data(model)
    for row in rows:
        row['features'][:] = 0.
        row['old_log_prob'] = -math.log(2.)
    targets, _, inputs = model._targets(rows, windows)
    monkeypatch.setattr(model, '_targets', lambda *a: (targets, torch.zeros(len(rows)), inputs))
    before = deepcopy(model.actor.state_dict())
    optimizer = deepcopy(model.actor_optimizer.state_dict())
    metrics = model.on_episode(rows, windows, episode=2)
    assert metrics['actor_optimizer_steps'] == 0
    torch.testing.assert_close(model.actor.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(model.actor_optimizer.state_dict(), optimizer, rtol=0, atol=0)
