"""Regression coverage for physical returns and bounded MAMBA policy updates."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from test_routing_mamba_baseline import episode, fixed_flow, make_method
from urban_dreamer.learning.baselines.mamba import IMAGINATION_HORIZON, MambaMethod


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('deadline', [90., 300.])
def test_known_physical_deadline_ignores_erroneous_learned_continuation(monkeypatch, deadline):
    method = make_method()
    rows, windows = episode(method)
    method.runner.deadline = deadline
    data = method._pack_episode(rows, windows, 1)
    fixed_flow(method, monkeypatch)
    monkeypatch.setattr(method.global_model.reward_model, 'forward',
        lambda x: x.new_full((*x.shape[:-1], 1), -1.))
    monkeypatch.setattr(method.global_model.continuation_model, 'forward',
        lambda x: x.new_full((*x.shape[:-1], 1), -100.))
    monkeypatch.setattr(method.critic, 'forward',
        lambda x: x.new_full((*x.shape[:-1], 1), -10.))
    initial = method.global_model.initial_state(1, 1, device=method.device)
    _, _, returns = method._imagined_rollout(data, initial, 0, {'a'})
    # The whole-network reward continues after an individual arrival, through
    # the known real deadline. An untrained head must not erase that tail.
    if deadline == 90.:
        torch.testing.assert_close(returns, torch.tensor([-3., -2., -1.]))
    else:
        # A shortened imagination remains a bootstrap boundary. Its critic
        # value must survive even if the learned survival head predicts zero.
        torch.testing.assert_close(returns,
            -torch.arange(IMAGINATION_HORIZON, 0, -1).float() - 10.)


def test_checkpoint_rejects_previous_continuation_and_policy_reduction():
    method = make_method()
    state = method.state_dict()
    for key in ('ppo_target_kl', 'ppo_kl_stop_multiplier', 'ppo_reduction', 'imagination_continuation'):
        state['protocol'].pop(key)
    with pytest.raises(ValueError, match='fresh run'):
        method.load_state_dict(state)


class StubActor(nn.Module):
    prior_weight = 0.

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def routing_distribution(self, latent, candidates, mask, **kwargs):
        logits = self.logits.expand(len(latent), -1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf)
        return Categorical(logits=logits)


def policy_method():
    method = MambaMethod.__new__(MambaMethod)
    method.args = SimpleNamespace(global_reward_only=True)
    method.actor = StubActor()
    method.actor_optimizer = object()
    method.entropy_coef = 0.
    return method


def record(method, count, step, *, forced=False):
    latent = torch.zeros(count, 1)
    action = torch.zeros(count, dtype=torch.long)
    mask = torch.tensor([[True, not forced]]).expand(count, -1)
    return dict(step=step, latent=latent, candidates=None, mask=mask, action=action,
        old_log_prob=method.actor.routing_distribution(latent, None, mask).log_prob(action).detach())


def test_partial_agent_chunk_does_not_receive_full_chunk_weight():
    method = policy_method()
    losses = []
    method._step = lambda optimizer, loss, parameters: losses.append(float(loss.detach()))
    method._ppo_update([record(method, 32, 0), record(method, 1, 1)], torch.tensor([1., -1.]))
    assert losses
    assert losses == pytest.approx([-31. / 33.] * len(losses))


@pytest.mark.parametrize(('target_kl', 'expected_steps'), [(.01, 1), (.1, 2), (0., 4)])
def test_policy_update_respects_custom_kl_threshold_and_zero_disables(target_kl, expected_steps):
    method = policy_method()
    method.args.mamba_target_kl = target_kl
    steps = []

    def step(optimizer, loss, parameters):
        steps.append(float(loss.detach()))
        with torch.no_grad():
            method.actor.logits[0].add_(1.)

    method._step = step
    result = method._ppo_update([record(method, 32, 0)], torch.ones(1))
    assert torch.isfinite(result)
    assert len(steps) == expected_steps


@pytest.mark.parametrize('target_kl', [-.1, float('nan'), float('inf')])
def test_constructor_rejects_invalid_policy_kl(target_kl):
    method = make_method()
    method.args.mamba_target_kl = target_kl
    with pytest.raises(ValueError, match='mamba_target_kl'):
        MambaMethod(method.args, method.runner)


def test_policy_kl_threshold_is_checkpointed():
    method = make_method()
    method.args.mamba_target_kl = .123
    assert method.learning_protocol()['ppo_target_kl'] == .123


def test_forced_actions_do_not_dilute_choice_loss():
    method = policy_method()
    losses = []
    method._step = lambda optimizer, loss, parameters: losses.append(float(loss.detach()))
    method._ppo_update([record(method, 1, 0), record(method, 32, 1, forced=True)],
        torch.tensor([1., -1.]))
    assert losses == pytest.approx([-1.] * len(losses))


def test_forced_only_batch_preserves_primed_adam_parameters_and_state():
    method = policy_method()
    method.actor_optimizer = torch.optim.Adam(method.actor.parameters(), lr=.1)
    method._step = MambaMethod._step.__get__(method)
    initial = record(method, 1, 0)
    loss = -method.actor.routing_distribution(initial['latent'], None, None).log_prob(initial['action']).mean()
    method._step(method.actor_optimizer, loss, list(method.actor.parameters()))
    before = method.actor.logits.detach().clone()
    previous_step = method.actor_optimizer.state[method.actor.logits]['step'].clone()
    result = method._ppo_update([record(method, 32, 0, forced=True)], torch.ones(1))
    torch.testing.assert_close(method.actor.logits, before, rtol=0., atol=0.)
    torch.testing.assert_close(method.actor_optimizer.state[method.actor.logits]['step'],
        previous_step, rtol=0., atol=0.)
    assert float(result.detach()) == 0.
