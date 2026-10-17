"""Condition-only graph replays must refresh all solve-local state."""
from dataclasses import fields, replace
from types import SimpleNamespace

import torch
from torch import nn

from urban_dreamer.learning.fm_graph_solver import EulerGraphSolver, _CapturedSolve
from urban_dreamer.models.world_model import ConditionEncoding


class ConditionDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.transformer = SimpleNamespace(layers=[])

    def solve(self, start, action, condition, **kwargs):
        output = start * self.weight + action[..., None] + condition.edge_tokens
        for name in ('reference_tokens', 'current_state', 'background'):
            value = getattr(condition, name)
            if value is not None:
                output = output + value
        return output


def fake_capture(solver, monkeypatch):
    monkeypatch.setattr(solver, '_supported', lambda *args: True)
    def capture(start, action, condition, steps):
        copied = ConditionEncoding(**{
            field.name: None if (value := getattr(condition, field.name)) is None else value.clone()
            for field in fields(ConditionEncoding)})
        cached = _CapturedSolve(None, start.clone(), action.clone(), copied, start.clone())
        def replay():
            cached.output.copy_(solver.eager(cached.start, cached.action, cached.condition, steps=steps))
        cached.graph = SimpleNamespace(replay=replay)
        return cached
    monkeypatch.setattr(solver, '_capture', capture)


def test_replay_refreshes_current_and_background_without_recapture(monkeypatch):
    dynamics = ConditionDynamics().eval()
    solver = EulerGraphSolver(dynamics)
    fake_capture(solver, monkeypatch)
    start = torch.ones(3, 1, 2, 1)
    action = torch.zeros(3, 1, 2)
    condition = ConditionEncoding(start * 2, start * 3, start * 4, start * 5)
    with torch.no_grad():
        first = solver(start, action, condition)
        updated = replace(condition, current_state=start * 40, background=start * 50)
        second = solver(start, action, updated)
        torch.testing.assert_close(second, dynamics.solve(start, action, updated), rtol=0, atol=0)
        assert not torch.equal(first, second)
        cached = next(iter(solver.cache.values()))
        assert cached.condition.current_state.data_ptr() != updated.current_state.data_ptr()
        assert cached.condition.background.data_ptr() != updated.background.data_ptr()
    assert solver.stats['captures'] == 1
    assert solver.stats['replays'] == 2


def test_optional_condition_presence_changes_capture_key(monkeypatch):
    dynamics = ConditionDynamics().eval()
    solver = EulerGraphSolver(dynamics, max_cached_shapes=4)
    fake_capture(solver, monkeypatch)
    start = torch.ones(3, 1, 2, 1)
    action = torch.zeros(3, 1, 2)
    base = ConditionEncoding(start * 2, start * 3)
    with torch.no_grad():
        for condition in (base, replace(base, current_state=start),
                          replace(base, background=start),
                          replace(base, current_state=start, background=start)):
            actual = solver(start, action, condition)
            torch.testing.assert_close(actual, dynamics.solve(start, action, condition), rtol=0, atol=0)
    assert solver.stats['captures'] == 4


def test_cuda_condition_capture_and_replay():
    # Opt in so normal CI never claims shared GPU memory.
    import os
    import pytest
    if os.environ.get('URB_TEST_CONDITION_CUDA_GRAPH') != '1' or not torch.cuda.is_available():
        pytest.skip('opt-in CUDA graph validation')
    from urban_dreamer.models.world_model import Dynamics
    torch.cuda.set_per_process_memory_fraction(.03, 0)
    torch.manual_seed(87)
    model = Dynamics(adjacency=torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
        static_features=torch.ones(3, 5), low_frequency_rank=2, prior_filter='soft',
        prior_filter_strength=.1, prior_action_conditioned=False, prior_usage='condition_only').cuda().eval()
    with torch.no_grad():
        model.flow_head.weight.normal_(std=.02)
        model.condition_residual.output.weight.normal_(std=.02)
        model.graph_prior.output.weight.normal_(std=.02)
        z = torch.randn(3, 1, 3, 64, device='cuda')
        action = torch.rand(3, 1, 3, device='cuda')
        start, _, condition = model.prior_start(z, model.encode_condition(action))
        solver = EulerGraphSolver(model, memory_reserve_bytes=0)
        for current in (condition, replace(condition, current_state=z + .3,
                                            background=condition.background + .7)):
            actual = solver(start, action, current, steps=2)
            expected = model.solve(start, action, current, steps=2)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        assert solver.stats['captures'] == 1
        assert solver.stats['replays'] == 2
        assert solver.stats['fallbacks'] == 0
        assert solver.stats['failures'] == []
        assert not solver._supported(start, action, replace(condition, current_state=z.cpu()), 2, 'euler')
        assert not solver._supported(start, action, replace(condition, background=z.double()), 2, 'euler')
        assert not solver._supported(start, action, replace(condition, background=z[:, :, :2]), 2, 'euler')
