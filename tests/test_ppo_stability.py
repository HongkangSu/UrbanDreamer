"""PPO update guards apply to actual choices and every optimizer minibatch."""
from copy import deepcopy

import pytest
import torch

from test_routing_ppo_baselines import data, method
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_kl_guard_stops_remaining_minibatches_in_same_epoch(cls, monkeypatch):
    model = method(cls)
    model.minibatch_size = 1
    rows, windows = data(model)
    original = model.actor_optimizer.step

    def overshoot(*args, **kwargs):
        result = original(*args, **kwargs)
        with torch.no_grad():
            model.actor.policy_head.weight.mul_(10000.)
        return result

    monkeypatch.setattr(model.actor_optimizer, 'step', overshoot)
    metrics = model.on_episode(rows, windows, episode=1)
    assert metrics['actor_optimizer_steps'] == 0
    assert metrics['actor_attempted_steps'] == model.max_backtracks + 1
    assert metrics['actor_rejected_updates'] == model.max_backtracks + 1
    assert metrics['kl_early_stopped']
    assert metrics['policy_kl'] <= model.target_kl


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_forced_actions_do_not_move_actor_from_existing_adam_momentum(cls):
    model = method(cls)
    model.on_episode(*data(model), episode=1)
    assert model.actor_optimizer.state
    rows, windows = data(model)
    for row in rows:
        row['mask'][:] = False
        row['mask'][row['action']] = True
        row['old_log_prob'] = 0.
    before = deepcopy(model.actor.state_dict())
    optimizer_before = deepcopy(model.actor_optimizer.state_dict())
    # The entropy schedule stays enabled: a one-action categorical has no entropy.
    assert model.entropy_coef > 0
    metrics = model.on_episode(rows, windows, episode=2)
    assert metrics['actor_optimizer_steps'] == 0
    assert metrics['critic_optimizer_steps'] > 0
    assert metrics['actor_decision_count'] == 0
    assert metrics['forced_decision_count'] == len(rows)
    torch.testing.assert_close(model.actor.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(model.actor_optimizer.state_dict(), optimizer_before, rtol=0, atol=0)


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_forced_rows_do_not_change_policy_advantage_normalization(cls, monkeypatch):
    clean, padded = method(cls), method(cls)
    rows, windows = data(clean)
    padded_rows = deepcopy(rows)
    forced = deepcopy(rows[0])
    forced['mask'][:] = False
    forced['mask'][forced['action']] = True
    forced['old_log_prob'] = 0.
    padded_rows.extend(deepcopy(forced) for _ in range(20))
    for model, source in ((clean, rows), (padded, padded_rows)):
        targets, _, inputs = model._targets(source, windows)
        advantages = torch.cat((torch.tensor([-2., -1., 1., 2.]),
                                torch.full((len(source) - 4,), 1000.)))
        monkeypatch.setattr(model, '_targets', lambda *args, result=(targets, advantages, inputs): result)
        monkeypatch.setattr(model, '_fit_critic', lambda *args: {})
    torch.manual_seed(38)
    clean_metrics = clean.on_episode(rows, windows, episode=1)
    torch.manual_seed(38)
    padded_metrics = padded.on_episode(padded_rows, windows, episode=1)
    assert clean_metrics['actor_optimizer_steps'] == padded_metrics['actor_optimizer_steps']
    torch.testing.assert_close(clean.actor.state_dict(), padded.actor.state_dict(), rtol=0, atol=0)


def test_zero_kl_limits_keep_guards_disabled():
    model = method(MappoMethod)
    model.target_kl = 0.
    model.max_policy_kl = 0.
    model.minibatch_size = 1
    rows, windows = data(model)
    metrics = model.on_episode(rows, windows, episode=1)
    assert metrics['actor_optimizer_steps'] == len(rows) * model.ppo_epochs
    assert not metrics['kl_early_stopped']
