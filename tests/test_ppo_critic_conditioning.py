"""Wide traffic observations must retain a trainable centralized value signal."""
import torch
from torch import nn

from urban_dreamer.learning.baselines.common import GlobalTrafficCritic


def _wide_observations():
    progress = torch.linspace(0., 1., 32)
    flow = torch.zeros(32, 1024, 4)
    flow[:, :, 1] = 1.  # Thousands of almost constant free-flow speed features.
    flow[:, :, 0] = progress[:, None] * .01
    context = torch.stack((progress, 1. - progress, 8. * progress, 6. * progress), dim=1)
    return torch.cat((flow.flatten(1), context), dim=1), progress


def test_default_critic_keeps_legacy_architecture_and_state_keys():
    critic = GlobalTrafficCritic(3)
    assert [type(layer) for layer in critic.network] == [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh, nn.Linear]
    assert list(critic.state_dict()) == [f'network.{index}.{parameter}'
        for index in (0, 2, 4) for parameter in ('weight', 'bias')]


def test_free_flow_centering_removes_coherent_speed_offset_without_erasing_congestion():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        critic = GlobalTrafficCritic(1024, normalize_hidden=True, center_free_flow=True)
        observations, _ = _wide_observations()
        with torch.no_grad():
            before = critic(observations)
            critic.network[0].weight[:, 1:4096:4].add_(.02)
            torch.testing.assert_close(critic(observations), before, rtol=0., atol=0.)
            congested = observations.clone()
            congested[:, 1:4096:4] *= .5
            assert not torch.allclose(critic(congested), before)
        assert 'input_offset' in critic.state_dict()
        assert critic.input_offset.shape == (4100,)
        assert torch.equal(critic.input_offset[1:4096:4], torch.ones(1024))
        assert torch.equal(critic.input_offset[-4:], torch.zeros(4))
    finally:
        torch.set_num_threads(previous_threads)


def test_normalized_wide_critic_survives_coherent_projection_shift_and_fits():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(73861)
        legacy = GlobalTrafficCritic(1024)
        normalized = GlobalTrafficCritic(1024, normalize_hidden=True)
        # Compare identical projections: normalization is the only difference.
        normalized_linears = [layer for layer in normalized.network if isinstance(layer, nn.Linear)]
        for target, source in zip(normalized_linears, (layer for layer in legacy.network if isinstance(layer, nn.Linear))):
            target.load_state_dict(source.state_dict())
        observations, progress = _wide_observations()
        with torch.no_grad():
            for critic in (legacy, normalized):
                critic.network[0].weight[:, 1:4096:4].add_(.02)
            legacy_hidden = legacy.network[:2](observations)
            normalized_hidden = normalized.network[:3](observations)
            assert float((legacy_hidden.abs() > .99).float().mean()) > .99
            assert float((normalized_hidden.abs() > .99).float().mean()) < .1
            assert float(legacy(observations).std()) < 1e-5
            assert float(normalized(observations).std()) > 1e-3
        targets = -2. * progress + .25 * torch.sin(6. * progress)
        optimizer = torch.optim.Adam(normalized.parameters(), lr=3e-4)
        before = torch.nn.functional.mse_loss(normalized(observations).flatten(), targets).item()
        for _ in range(60):
            loss = torch.nn.functional.mse_loss(normalized(observations).flatten(), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in normalized.parameters())
            torch.nn.utils.clip_grad_norm_(normalized.parameters(), 10.)
            optimizer.step()
        after = torch.nn.functional.mse_loss(normalized(observations).flatten(), targets).item()
        assert after < before * .1
    finally:
        torch.set_num_threads(previous_threads)
