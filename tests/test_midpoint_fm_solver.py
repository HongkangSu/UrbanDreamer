"""Midpoint integration is independent of physical rollout and FM regression depth."""
from types import SimpleNamespace
import math

import pytest
import torch

from urban_dreamer.models.world_model import (
    ConditionEncoding, Dynamics, independent_transition_fm_loss, recursive_fm_loss,
)


class AnalyticDynamics:
    solve = Dynamics.solve
    transition = Dynamics.transition
    n_edges = 2

    def __init__(self, field='constant'):
        self.field = field
        self.gain = torch.tensor(.1, requires_grad=True)
        self.graph_prior = SimpleNamespace(rank=1, project=lambda value: value)
        self.calls, self.conditions, self.prior_calls = [], [], 0

    def encode_condition(self, action, **kwargs):
        condition = ConditionEncoding(action.unsqueeze(-1))
        self.conditions.append(condition)
        return condition

    def prior_start(self, z, condition, noise=None):
        self.prior_calls += 1
        perturbation = torch.zeros_like(z) if noise is None else noise.expand_as(z)
        return z + perturbation, torch.zeros_like(z), condition

    def __call__(self, action, *, packed_latents, flow_time, condition):
        self.calls.append((flow_time.detach().clone(), id(condition)))
        if self.field == 'constant':
            velocity = torch.full_like(packed_latents, 2.)
        elif self.field == 'linear':
            velocity = packed_latents
        elif self.field == 'time':
            velocity = flow_time[..., None, None].expand_as(packed_latents)
        else:
            velocity = self.gain * packed_latents + condition.edge_tokens
        return SimpleNamespace(latent_velocity=velocity)


@pytest.mark.parametrize('shape', [(1, 2, 64), (2, 3, 2, 64)])
def test_euler_100_intervals_have_100_field_calls_and_reuse_noise_condition(shape):
    dynamics = AnalyticDynamics()
    start = torch.ones(shape, dtype=torch.float64)
    action = torch.zeros(shape[:-1], dtype=torch.float64)
    noise = torch.full((*shape[:-2], 1, 64), .3, dtype=torch.float64)
    result = dynamics.transition(start, action, noise=noise, steps=100, solver='euler')
    torch.testing.assert_close(result, torch.full_like(start, 3.3))
    assert len(dynamics.calls) == 100
    assert dynamics.prior_calls == len(dynamics.conditions) == 1
    assert len({condition for _, condition in dynamics.calls}) == 1
    times = torch.stack([time for time, _ in dynamics.calls]).flatten(1)
    torch.testing.assert_close(times, (torch.arange(100, dtype=torch.float64) / 100)
                              .unsqueeze(1).expand_as(times))


@pytest.mark.parametrize('shape', [(1, 2, 64), (2, 3, 2, 64)])
def test_midpoint_100_intervals_reuse_condition_and_single_noise(shape):
    dynamics = AnalyticDynamics()
    start = torch.ones(shape, dtype=torch.float64)
    action = torch.zeros(shape[:-1], dtype=torch.float64)
    noise = torch.full((*shape[:-2], 1, 64), .3, dtype=torch.float64)
    result = dynamics.transition(start, action, noise=noise, steps=100, solver='midpoint')
    torch.testing.assert_close(result, torch.full_like(start, 3.3))
    assert len(dynamics.calls) == 200
    assert dynamics.prior_calls == len(dynamics.conditions) == 1
    assert len({condition for _, condition in dynamics.calls}) == 1
    assert dynamics.calls[0][0].eq(0.).all()
    assert dynamics.calls[1][0].eq(.005).all()
    assert dynamics.calls[-1][0].eq(.995).all()
    second = dynamics.transition(start, action, noise=noise, steps=100, solver='midpoint')
    torch.testing.assert_close(result, second, rtol=0, atol=0)


def test_midpoint_uses_both_midpoint_time_and_state_and_has_second_order_accuracy():
    start = torch.ones(1, 1, 2, 64, dtype=torch.float64)
    action = torch.zeros(1, 1, 2, dtype=torch.float64)
    errors = []
    for steps in (4, 8, 16):
        dynamics = AnalyticDynamics('linear')
        result = dynamics.transition(start, action, steps=steps, solver='midpoint')
        expected = (1 + 1 / steps + .5 / steps**2)**steps
        torch.testing.assert_close(result, torch.full_like(start, expected))
        errors.append(abs(float(result.mean()) - math.e))
    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.7
    dynamics = AnalyticDynamics('time')
    result = dynamics.transition(start, action, steps=4, solver='midpoint')
    torch.testing.assert_close(result, torch.full_like(start, 1.5))


def test_fm_observed_endpoint_training_does_not_integrate_midpoint(monkeypatch):
    dynamics = AnalyticDynamics()
    monkeypatch.setattr(dynamics, 'solve', lambda *a, **k: pytest.fail('FM regression must not solve ODE'))
    inputs = torch.ones(2, 3, 2, 64)
    targets = inputs + 2
    loss, _, prediction = independent_transition_fm_loss(dynamics,
        z_inputs=inputs, z_targets=targets, action=torch.zeros(2, 3, 2),
        flow_time=torch.full((2, 3), .35), solver='midpoint', steps=100,
        predict_endpoints=False, prior_weight=0.)
    assert loss.item() == 0
    assert len(dynamics.calls) == 1
    torch.testing.assert_close(prediction, targets)


def test_recursive_midpoint_retains_early_endpoint_gradient_and_no_future_condition():
    dynamics = AnalyticDynamics('learned')
    endpoints = []
    original = dynamics.solve

    def capture(*args, **kwargs):
        endpoint = original(*args, **kwargs)
        endpoint.retain_grad()
        endpoints.append(endpoint)
        return endpoint

    dynamics.solve = capture
    sequence = torch.ones(1, 4, 2, 64)
    action = torch.tensor([[[.1, .2], [.3, .4], [.5, .6]]])
    kwargs = dict(z_sequence=sequence, action=action, steps=2, solver='midpoint',
                  flow_time=torch.full((1, 3), .3), depth_weights=torch.tensor([0., 0., 1.]),
                  prior_weight=0.)
    loss, _, before = recursive_fm_loss(dynamics, **kwargs)
    loss.backward()
    assert endpoints[0].grad is not None and endpoints[0].grad.abs().sum() > 0
    assert dynamics.gain.grad is not None and dynamics.gain.grad.abs() > 0
    assert len(dynamics.conditions) == dynamics.prior_calls == 3
    assert len(dynamics.calls) == 3 * (1 + 2 * 2)
    changed = action.clone()
    changed[:, 1:] += 5
    other = AnalyticDynamics('learned')
    _, _, after = recursive_fm_loss(other, **{**kwargs, 'action': changed})
    torch.testing.assert_close(before[:, 0], after[:, 0], atol=0, rtol=0)
    assert not torch.allclose(before[:, 1:], after[:, 1:])


@pytest.mark.parametrize('solver,steps,field_calls', [('midpoint', 4, 8), ('euler', 100, 100)])
def test_production_solver_reproducible_with_fixed_noise_and_prior_draw_once(monkeypatch, solver, steps, field_calls):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    dynamics = Dynamics(adjacency=torch.tensor([[0., 1.], [1., 0.]]),
                        static_features=torch.ones(2, 4)).eval()
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.01)
    start = torch.randn(1, 2, 64)
    action = torch.ones(1, 2)
    noise = torch.randn(1, dynamics.graph_prior.rank, 64)
    counts = {'noise': 0, 'condition': 0, 'forward': 0}
    sample, encode, forward = dynamics.graph_prior.sample_noise, dynamics.encode_condition, dynamics.forward

    def count_noise(*a, **k):
        counts['noise'] += 1
        return sample(*a, **k)

    def count_condition(*a, **k):
        counts['condition'] += 1
        return encode(*a, **k)

    def count_forward(*a, **k):
        counts['forward'] += 1
        return forward(*a, **k)

    monkeypatch.setattr(dynamics.graph_prior, 'sample_noise', count_noise)
    monkeypatch.setattr(dynamics, 'encode_condition', count_condition)
    monkeypatch.setattr(dynamics, 'forward', count_forward)
    with torch.no_grad():
        before = dynamics.transition(start, action, noise=noise, steps=steps, solver=solver)
        assert counts == {'noise': 1, 'condition': 1, 'forward': field_calls}
        assert torch.isfinite(before).all()
        after = dynamics.transition(start, action, noise=noise, steps=steps, solver=solver)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
