"""PPO uses frozen behavior ratios and complete global team episode returns."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines.ppo import (
    IppoMethod, LocalEventCritic, MappoMethod, clipped_policy_loss)
from urban_dreamer.models.policy import set_shortest_path_prior


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def method(cls):
    torch.manual_seed(73)
    args = SimpleNamespace(reward_objective='legacy_exposure', seed=42, actor_lr=3e-4, critic_lr=3e-4)
    runner = SimpleNamespace(device=torch.device('cpu'), network=SimpleNamespace(edge_count=2), deadline=90.)
    return cls(args, runner)


def data(model):
    rng = np.random.default_rng(27)
    features = rng.random((4, 3, 10), dtype=np.float32)
    features[:, :, 4] = np.log1p([30., 120., 1.]) / math.log1p(3600.)
    mask = np.array([[True, True, False]] * 4)
    obs = SimpleNamespace(features=features, mask=mask)
    indices, aux = model.select([{}] * 4, obs, None, 0.)
    rows = [dict(vehicle_id=['arrives_early', 'later', 'later', 'last'][i],
                 window=[0, 0, 1, 2][i], time=[0., 15., 30., 60.][i], features=features[i],
                 mask=mask[i], action=indices[i], **aux[i]) for i in range(4)]
    windows = [dict(obs=rng.random(10, dtype=np.float32), next_obs=rng.random(10, dtype=np.float32),
                    start_time=i * 30., end_time=(i + 1) * 30.,
                    reward=-float(i + 1), terminal=i == 2) for i in range(3)]
    return rows, windows


def test_clipping_uses_pessimistic_positive_and_negative_surrogates():
    log_prob = torch.tensor([math.log(2.), math.log(.5)], requires_grad=True)
    loss = clipped_policy_loss(log_prob, torch.zeros(2, requires_grad=True), torch.tensor([1., -1.]))
    torch.testing.assert_close(loss, torch.tensor(-.2))
    loss.backward()
    torch.testing.assert_close(log_prob.grad, torch.zeros(2))
    # Improving an already worse action is not clipped away.
    log_prob = torch.tensor([math.log(.5), math.log(2.)], requires_grad=True)
    clipped_policy_loss(log_prob, torch.zeros(2), torch.tensor([1., -1.])).backward()
    assert (log_prob.grad != 0).all()


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_selection_retains_legal_mask_prior_and_true_behavior_log_probability(cls):
    model = method(cls)
    for parameter in model.actor.parameters():
        parameter.data.zero_()
    set_shortest_path_prior(model.actor, 1., 30.)
    rows, _ = data(model)
    features, mask = model._row_tensors(rows)
    distribution = torch.distributions.Categorical(logits=model.actor.forward_logits(features, mask))
    torch.testing.assert_close(distribution.probs[:, 0], torch.full((4,), 1 / (1 + math.exp(-3.))))
    assert torch.all(distribution.probs[:, 2] == 0)
    for index, row in enumerate(rows):
        assert row['mask'][row['action']]
        assert row['old_log_prob'] == pytest.approx(float(distribution.log_prob(
            torch.tensor([r['action'] for r in rows]))[index].detach()))
        assert row['sp_prior_weight'] == 1.


def test_local_critic_ignores_padding_and_candidate_order():
    critic = LocalEventCritic()
    features = torch.randn(3, 4, 10)
    mask = torch.tensor([[True, True, False, False]] * 3)
    expected = critic(features, mask)
    padded = features.clone()
    padded[:, 2:] = 10000.
    torch.testing.assert_close(critic(padded, mask), expected)
    permutation = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(critic(features[:, permutation], mask[:, permutation]), expected)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_early_arrival_target_includes_rewards_until_team_episode_end(cls):
    model = method(cls)
    for parameter in model.critic.parameters():
        parameter.data.zero_()
    rows, windows = data(model)
    _, original_advantage, _ = model._targets(rows, windows)
    changed = deepcopy(windows)
    changed[-1]['reward'] -= 7.
    targets, altered_advantage, _ = model._targets(rows, changed)
    # The early vehicle only has an event in window zero, yet the later cost matters.
    assert altered_advantage[0] < original_advantage[0]
    if cls is IppoMethod:
        assert targets[0] == pytest.approx(-13.)
    else:
        assert targets[0] == pytest.approx(-1. + .95 * (-2. + .95 * -10.))


@pytest.mark.parametrize('later_team_window', [False, True])
def test_ippo_same_window_events_prorate_future_rewards_from_actual_decision_times(later_team_window):
    model = method(IppoMethod)
    for parameter in model.critic.parameters():
        parameter.data.zero_()
    source_rows, _ = data(model)
    rows = deepcopy(source_rows[:2])
    for row, timestamp in zip(rows, [15., 25.]):
        row.update(vehicle_id='one_vehicle', time=timestamp, window=0)
    windows = [dict(start_time=0., end_time=30., reward=-30., terminal=not later_team_window)]
    if later_team_window:
        windows.append(dict(start_time=30., end_time=60., reward=-60., terminal=True))
    targets, _, _ = model._targets(rows, windows)
    # At t=25 the vehicle receives the final five seconds plus all subsequent
    # team windows, never the 25 seconds of costs before its own decision.
    tail = -65. if later_team_window else -5.
    torch.testing.assert_close(targets, torch.tensor([-10. + .95 ** (10. / 30.) * tail, tail]))


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_actor_and_critic_update_and_ignore_poisoned_individual_timeout_cost(cls):
    clean = method(cls)
    poisoned = method(cls)
    rows, windows = data(clean)
    poison_rows = deepcopy(rows)
    for row in poison_rows:
        row.update(timeout_return=float('nan'), timeout_penalty=-1e12)
    actor_before = deepcopy(clean.actor.state_dict())
    critic_before = deepcopy(clean.critic.state_dict())
    torch.manual_seed(982)
    metrics = clean.on_episode(rows, windows, episode=1)
    torch.manual_seed(982)
    other = poisoned.on_episode(poison_rows, windows, episode=1)
    assert metrics['ppo_epochs'] == 4
    assert math.isfinite(metrics['actor_loss']) and math.isfinite(metrics['critic_loss'])
    assert any(not torch.equal(actor_before[key], value) for key, value in clean.actor.state_dict().items())
    assert any(not torch.equal(critic_before[key], value) for key, value in clean.critic.state_dict().items())
    assert metrics['actor_loss'] == other['actor_loss']
    for original, poison in zip(clean.actor.parameters(), poisoned.actor.parameters()):
        torch.testing.assert_close(original, poison, rtol=0, atol=0)
    for original, poison in zip(clean.critic.parameters(), poisoned.critic.parameters()):
        torch.testing.assert_close(original, poison, rtol=0, atol=0)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_checkpoint_restores_actor_critic_and_optimizer_updates(cls, tmp_path):
    original = method(cls)
    rows, windows = data(original)
    original.on_episode(rows, windows, episode=1)
    path = tmp_path / 'ppo.pt'
    torch.save(original.state_dict(), path)
    restored = method(cls)
    restored.load_state_dict(torch.load(path, weights_only=False))
    for component in ('actor', 'critic'):
        for a, b in zip(getattr(original, component).parameters(), getattr(restored, component).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    # A continuation step also agrees, exercising restored Adam moment state.
    rows, windows = data(original)
    torch.manual_seed(634)
    original.on_episode(rows, windows, episode=2)
    torch.manual_seed(634)
    restored.on_episode(rows, windows, episode=2)
    for component in ('actor', 'critic'):
        for a, b in zip(getattr(original, component).parameters(), getattr(restored, component).parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_mismatched_prior_and_missing_behavior_probability_rejected(cls):
    model = method(cls)
    rows, windows = data(model)
    set_shortest_path_prior(model.actor, .5, 30.)
    with pytest.raises(ValueError, match='same shortest-path prior'):
        model.on_episode(rows, windows, episode=1)
    set_shortest_path_prior(model.actor, 0., 30.)
    del rows[0]['old_log_prob']
    with pytest.raises(ValueError, match='old_log_prob'):
        model.on_episode(rows, windows, episode=1)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_zero_decision_episode_is_supported(cls):
    model = method(cls)
    _, windows = data(model)
    metrics = model.on_episode([], windows, episode=1)
    assert metrics['actor_loss'] == 0.
    assert math.isfinite(metrics['critic_loss'])


def test_advantage_normalization_removes_common_offset_and_near_constant_noise():
    from urban_dreamer.learning.baselines.ppo import normalize_advantages
    for advantage in (torch.full((32,), -100.), torch.tensor([-1., -1.0000001])):
        assert torch.equal(normalize_advantages(advantage, std_floor=1e-6),
                           torch.zeros_like(advantage))
    expected = normalize_advantages(torch.tensor([-2., 0., 2.]), std_floor=1e-6)
    torch.testing.assert_close(expected, normalize_advantages(
        torch.tensor([-102., -100., -98.]), std_floor=1e-6))
    assert expected.mean() == pytest.approx(0.)
    assert expected.std(unbiased=False) == pytest.approx(1.)


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
@pytest.mark.parametrize('existing_momentum', [False, True])
def test_constant_advantage_has_no_policy_update_when_entropy_disabled(cls, existing_momentum, monkeypatch):
    model = method(cls)
    if existing_momentum:
        model.on_episode(*data(model), episode=1)
    model.entropy_coef = 0.
    rows, windows = data(model)
    targets, _, inputs = model._targets(rows, windows)
    monkeypatch.setattr(model, '_targets', lambda *args: (
        targets, torch.full((len(rows),), -10.), inputs))
    before = deepcopy(model.actor.state_dict())
    optimizer_before = deepcopy(model.actor_optimizer.state_dict())
    metrics = model.on_episode(rows, windows, episode=1)
    for key, value in model.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert metrics['raw_advantage_mean'] == -10.
    assert metrics['raw_advantage_std'] == 0.
    assert metrics['advantage_std'] == 0.
    assert metrics['policy_kl'] == pytest.approx(0., abs=1e-7)
    assert metrics['actor_optimizer_steps'] == 0
    torch.testing.assert_close(model.actor_optimizer.state_dict(), optimizer_before, rtol=0, atol=0)


def test_full_distribution_kl_rejects_unsafe_epochs(monkeypatch):
    model = method(MappoMethod)
    rows, windows = data(model)
    step = model.actor_optimizer.step

    def large_step(*args, **kwargs):
        result = step(*args, **kwargs)
        with torch.no_grad():
            model.actor.policy_head.weight.mul_(10000.)
        return result

    monkeypatch.setattr(model.actor_optimizer, 'step', large_step)
    metrics = model.on_episode(rows, windows, episode=1)
    assert metrics['policy_kl'] <= model.target_kl
    assert metrics['actor_epochs_completed'] == 0
    assert metrics['actor_optimizer_steps'] == 0
    assert metrics['actor_attempted_steps'] == model.max_backtracks + 1
    assert metrics['kl_early_stopped']
    assert math.isfinite(metrics['policy_kl'])  # masked candidates include -inf logits


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_critic_restores_best_parameters_and_optimizer_when_fit_worsens(cls, monkeypatch):
    model = method(cls)
    rows, windows = data(model)
    before = deepcopy(model.critic.state_dict())
    optimizer_before = deepcopy(model.critic_optimizer.state_dict())
    step = model.critic_optimizer.step

    def overshoot(*args, **kwargs):
        result = step(*args, **kwargs)
        with torch.no_grad():
            for parameter in model.critic.parameters():
                parameter.add_(100.)
        return result

    monkeypatch.setattr(model.critic_optimizer, 'step', overshoot)
    metrics = model.on_episode(rows, windows, episode=1)
    assert metrics['critic_best_epoch'] == 0
    assert metrics['critic_mse_after'] == pytest.approx(metrics['critic_mse_before'])
    for key, value in model.critic.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert model.critic_optimizer.state_dict() == optimizer_before


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_checkpoint_rejects_legacy_or_changed_learning_protocol_before_loading(cls):
    model = method(cls)
    state = deepcopy(model.state_dict())
    assert state['learning_protocol']['version'] >= 2
    del state['learning_protocol']
    with pytest.raises(ValueError, match='learning protocol'):
        model.load_state_dict(state)
    state = deepcopy(model.state_dict())
    state['learning_protocol']['target_kl'] *= 2
    with pytest.raises(ValueError, match='learning protocol'):
        model.load_state_dict(state)


def test_stale_behavior_policy_is_rejected_before_critic_update():
    model = method(MappoMethod)
    rows, windows = data(model)
    rows[0]['old_log_prob'] -= .1
    critic_before = deepcopy(model.critic.state_dict())
    with pytest.raises(ValueError, match='behavior'):
        model.on_episode(rows, windows, episode=1)
    for key, value in model.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)


@pytest.mark.parametrize(('name', 'value'), [
    ('ppo_epochs', 0), ('ppo_minibatch_size', -1), ('ppo_target_kl', float('nan')),
    ('ppo_target_kl', -1.), ('ppo_advantage_std_floor', 0.), ('ppo_critic_epochs', 0),
    ('ppo_critic_patience', -1), ('ppo_critic_min_epochs', 99)])
def test_invalid_learning_settings_rejected(name, value):
    args = SimpleNamespace(reward_objective='legacy_exposure', seed=42, actor_lr=3e-4, critic_lr=3e-4, **{name: value})
    runner = SimpleNamespace(device=torch.device('cpu'), network=SimpleNamespace(edge_count=2), deadline=90.)
    with pytest.raises(ValueError, match='PPO'):
        MappoMethod(args, runner)


def test_critic_budget_defaults_distinguish_windows_from_local_events():
    assert method(MappoMethod).critic_epochs == 8
    assert method(IppoMethod).critic_epochs == 4
