"""PPO clipped objectives stay finite and zero minibatches preserve Adam state."""
import pytest
import torch

from test_routing_ppo_baselines import data, method
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod, clipped_policy_loss


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('advantage', [0., 1.])
def test_clipped_positive_or_zero_advantage_does_not_overflow_before_clipping(advantage):
    # These are valid finite log probabilities. Their ratio overflows float32,
    # although the pessimistic PPO objective is exactly bounded by 1 + clip.
    current = torch.tensor([-1.], requires_grad=True)
    behavior = torch.tensor([-1001.], requires_grad=True)
    advantage_tensor = torch.tensor([advantage], requires_grad=True)
    loss = clipped_policy_loss(current, behavior, advantage_tensor)
    torch.testing.assert_close(loss, torch.tensor(-1.2 * advantage))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))
    assert behavior.grad is None
    assert advantage_tensor.grad is None


def test_log_space_clipping_preserves_surrogate_values_and_gradients():
    current = torch.linspace(-12., -.1, 71, dtype=torch.float64, requires_grad=True)
    behavior = torch.full_like(current, -5.)
    advantages = torch.linspace(-2., 2., len(current), dtype=torch.float64)
    ratio = (current - behavior).exp()
    expected = -torch.minimum(ratio * advantages, ratio.clamp(.8, 1.2) * advantages).mean()
    expected_grad, = torch.autograd.grad(expected, current, retain_graph=True)
    actual = clipped_policy_loss(current, behavior, advantages)
    actual_grad, = torch.autograd.grad(actual, current)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize('cls', [IppoMethod, MappoMethod])
def test_zero_advantage_minibatches_skip_adam_even_with_nonzero_episode_advantages(cls, monkeypatch):
    model = method(cls)
    model.on_episode(*data(model), episode=1)
    assert model.actor_optimizer.state  # Real momentum from a previous episode.
    model.entropy_coef = 0.
    model.ppo_epochs = 1
    model.minibatch_size = 1
    model.target_kl = 0.
    rows, windows = data(model)
    targets, _, inputs = model._targets(rows, windows)
    monkeypatch.setattr(model, '_targets', lambda *args: (targets, torch.tensor([-1., 0., 0., 1.]), inputs))
    monkeypatch.setattr(model, '_fit_critic', lambda *args: {})
    monkeypatch.setattr(torch, 'randperm', lambda count, device: torch.arange(count, device=device))
    step_before = [float(state['step']) for state in model.actor_optimizer.state.values()]
    metrics = model.on_episode(rows, windows, episode=2)
    assert metrics['actor_optimizer_steps'] == 2
    step_after = [float(state['step']) for state in model.actor_optimizer.state.values()]
    assert step_after == [step + 2 for step in step_before]
