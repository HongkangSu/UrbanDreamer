"""Cache, fallback and checkpoint ownership checks; GPU numerical proof is a tool."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from urban_dreamer.learning.fm_graph_solver import EulerGraphSolver, _CapturedSolve
from urban_dreamer.models.world_model import ConditionEncoding


class TinyDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(.25))
        self.transformer = SimpleNamespace(layers=[])
        self.calls = 0

    def solve(self, start, action, condition, *, steps=2, solver='euler', **options):
        if steps < 1:
            raise ValueError('steps must be positive')
        self.calls += 1
        state = start
        for _ in range(steps):
            state = state + self.rate * state / steps
        return state + action[..., None] + condition.edge_tokens


def inputs(batch=3):
    start = torch.ones(batch, 1, 2, 1)
    return start, torch.zeros(batch, 1, 2), ConditionEncoding(torch.zeros_like(start))


def mock_capture(solver, monkeypatch):
    monkeypatch.setattr(solver, '_supported', lambda *args: True)
    def capture(start, action, condition, steps):
        cached = _CapturedSolve(None, start.clone(), action.clone(),
                                ConditionEncoding(condition.edge_tokens.clone()), start.clone())
        def replay():
            cached.output.copy_(solver.eager(cached.start, cached.action, cached.condition, steps=steps))
        cached.graph = SimpleNamespace(replay=replay)
        return cached
    monkeypatch.setattr(solver, '_capture', capture)


@pytest.mark.parametrize('batch', [3, 6, 12, 24])
def test_cpu_fallback_preserves_euler_and_gradients(batch):
    dynamics = TinyDynamics().eval()
    solver = EulerGraphSolver(dynamics)
    start, action, condition = inputs(batch)
    start.requires_grad_(True)
    output = solver(start, action, condition, steps=100)
    output.sum().backward()
    assert torch.isfinite(start.grad).all()
    assert dynamics.rate.grad != 0
    assert solver.stats['captures'] == 0
    assert solver.stats['fallbacks'] == 1


def test_cache_reads_live_weights_and_returns_owned_outputs(monkeypatch):
    dynamics = TinyDynamics().eval()
    keys = tuple(dynamics.state_dict())
    parameters = tuple(id(parameter) for parameter in dynamics.parameters())
    solver = EulerGraphSolver(dynamics)
    mock_capture(solver, monkeypatch)
    with torch.no_grad():
        args = inputs()
        original = solver(*args, steps=100)
        saved = original.clone()
        dynamics.rate.add_(.1)
        updated = solver(*args, steps=100)
        expected = dynamics.solve(*args, steps=100)
        assert torch.equal(updated, expected)
        assert not torch.equal(original, updated)
        assert torch.equal(original, saved)
    assert solver.stats['captures'] == 1
    assert tuple(dynamics.state_dict()) == keys
    assert tuple(id(parameter) for parameter in dynamics.parameters()) == parameters


def test_parameter_storage_replacement_invalidates_cache(monkeypatch):
    dynamics = TinyDynamics().eval()
    solver = EulerGraphSolver(dynamics)
    mock_capture(solver, monkeypatch)
    with torch.no_grad():
        solver(*inputs(), steps=100)
        dynamics.rate = nn.Parameter(torch.tensor(.5))
        output = solver(*inputs(), steps=100)
        assert torch.equal(output, dynamics.solve(*inputs(), steps=100))
    assert solver.stats['captures'] == 2
    assert len(solver.cache) == 1


def test_shape_cache_is_bounded(monkeypatch):
    solver = EulerGraphSolver(TinyDynamics().eval(), max_cached_shapes=2)
    mock_capture(solver, monkeypatch)
    with torch.no_grad():
        for batch in [3, 6, 12, 24]:
            solver(*inputs(batch), steps=100)
            assert len(solver.cache) <= 2
    assert solver.stats['captures'] == 4


def test_capture_failure_is_eager_and_is_not_repeated(monkeypatch):
    dynamics = TinyDynamics().eval()
    solver = EulerGraphSolver(dynamics)
    monkeypatch.setattr(solver, '_supported', lambda *args: True)
    attempted = []
    def fail(*args):
        attempted.append(True)
        raise RuntimeError('bounded capture unavailable')
    monkeypatch.setattr(solver, '_capture', fail)
    with torch.no_grad():
        for _ in range(2):
            actual = solver(*inputs(), steps=100)
            assert torch.equal(actual, dynamics.solve(*inputs(), steps=100))
    assert len(attempted) == 1
    assert len(solver.stats['failures']) == 1
    assert solver.stats['fallbacks'] == 2


def test_validation_uses_original_solver():
    solver = EulerGraphSolver(TinyDynamics().eval())
    with pytest.raises(ValueError, match='positive'):
        solver(*inputs(), steps=0)
    with pytest.raises(ValueError):
        EulerGraphSolver(TinyDynamics(), max_cached_shapes=0)


def test_deepcopy_discards_graph_workspaces_and_rebinds_dynamics():
    import copy
    dynamics = TinyDynamics().eval()
    solver = EulerGraphSolver(dynamics)
    dynamics.solve = solver
    cloned = copy.deepcopy(dynamics)
    assert cloned.solve.dynamics is cloned
    assert cloned.solve.cache == {}
    with torch.no_grad():
        cloned.rate.add_(.1)
        actual = cloned.solve(*inputs(), steps=100)
        expected = TinyDynamics.solve(cloned, *inputs(), steps=100)
        assert torch.equal(actual, expected)
        assert not torch.equal(actual, dynamics.solve(*inputs(), steps=100))
