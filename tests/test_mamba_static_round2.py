"""Static-audit regressions for MAMBA policy signals and event observations."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_mamba_stability import policy_method, record
from test_routing_mamba_baseline import episode, fixed_flow, make_method
from urban_dreamer.learning.baselines import mamba
from urban_dreamer.learning.global_reward import causal_context


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('forced_signal', [False, True])
def test_zero_choice_signal_preserves_primed_adam(forced_signal):
    method = policy_method()
    method.actor_optimizer = torch.optim.Adam(method.actor.parameters(), lr=.01)
    method._step = mamba.MambaMethod._step.__get__(method)
    initial = record(method, 1, 0)
    loss = -method.actor.routing_distribution(initial['latent'], None, None).log_prob(initial['action']).mean()
    method._step(method.actor_optimizer, loss, list(method.actor.parameters()))
    before = method.actor.logits.detach().clone()
    old_state = deepcopy(method.actor_optimizer.state_dict())
    records = [record(method, 2, 0)]
    if forced_signal:
        records.append(record(method, 4, 1, forced=True))
    result = method._ppo_update(records, torch.tensor([0., 1.]))
    torch.testing.assert_close(method.actor.logits, before, rtol=0., atol=0.)
    for key, value in old_state['state'][0].items():
        torch.testing.assert_close(method.actor_optimizer.state_dict()['state'][0][key], value, rtol=0., atol=0.)
    assert float(result) == 0.


def test_entropy_signal_still_updates_zero_advantage_policy():
    method = policy_method()
    method.entropy_coef = .1
    method.actor_optimizer = torch.optim.Adam(method.actor.parameters(), lr=.01)
    method._step = mamba.MambaMethod._step.__get__(method)
    with torch.no_grad():
        method.actor.logits.copy_(torch.tensor([1., -1.]))
    before = method.actor.logits.detach().clone()
    method._ppo_update([record(method, 2, 0)], torch.zeros(1))
    assert not torch.equal(method.actor.logits, before)


@pytest.mark.parametrize('advantage', [0., 1.])
def test_clipped_surrogate_has_finite_gradient_for_extreme_old_probability(advantage):
    log_prob = torch.tensor([-1.], requires_grad=True)
    loss = mamba.clipped_policy_loss(log_prob, torch.tensor([-1001.]), torch.tensor([advantage]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(log_prob.grad).all()
    torch.testing.assert_close(log_prob.grad, torch.zeros_like(log_prob))


def test_safe_surrogate_matches_ordinary_clipped_ppo_value_and_gradient():
    new = torch.tensor([-.1, -.7, -2., -.3, -1.2, -2.4], requires_grad=True)
    old = torch.tensor([-.2, -.8, -1.8, -.5, -1.1, -2.])
    advantage = torch.tensor([1., -1., 2., -2., 0., 1.])
    ratio = (new - old).exp()
    reference = -torch.minimum(ratio * advantage, ratio.clamp(.925, 1.075) * advantage).mean()
    expected_gradient = torch.autograd.grad(reference, new)[0]
    actual = mamba.clipped_policy_loss(new, old, advantage)
    gradient = torch.autograd.grad(actual, new)[0]
    torch.testing.assert_close(actual, reference)
    torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize(('objective', 'expected_progress'), [('observed_mean', 7. / 90.), ('legacy_exposure', 0.)])
def test_imagined_event_time_matches_real_observation_contract(monkeypatch, objective, expected_progress):
    method = make_method()
    rows, windows = episode(method)
    data = method._pack_episode(rows, windows, 1)
    method.args.reward_objective = objective
    data['windows'][0].update(started_vehicle_count=2, cumulative_vehicle_seconds=0.,
        context=causal_context(0., 90., 1, 2, 0.))
    fixed_flow(method, monkeypatch)
    seen = []
    original = mamba.build_actor_observations

    def observe(**kwargs):
        seen.append(kwargs['episode_progress'])
        return original(**kwargs)

    def motion(vehicles, *, decide, frozen_flow, start_time, **kwargs):
        decide(vehicles, frozen_flow, start_time + 7.)
        for vehicle in vehicles:
            vehicle['arrived'] = True
        return SimpleNamespace(active_vehicle_seconds=7.)

    monkeypatch.setattr(mamba, 'build_actor_observations', observe)
    monkeypatch.setattr(mamba, 'advance_imagined_window', motion)
    initial = method.global_model.initial_state(1, 1, device=method.device)
    method._imagined_rollout(data, initial, 0, {'a'})
    assert seen == pytest.approx([expected_progress])


def test_checkpoint_rejects_missing_event_and_zero_signal_contracts():
    method = make_method()
    state = method.state_dict()
    for key in ('actor_zero_signal', 'imagined_actor_context'):
        state['protocol'].pop(key, None)
    with pytest.raises(ValueError, match='fresh run'):
        method.load_state_dict(state)
