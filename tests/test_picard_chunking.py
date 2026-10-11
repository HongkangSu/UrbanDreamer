"""Picard chunks preserve the simultaneous iteration and sample/window ordering."""
from types import SimpleNamespace
import pytest
import torch
from urban_dreamer.models.world_model import ConditionEncoding, Dynamics


class Field:
    solve = Dynamics.solve

    def __init__(self):
        self.shapes = []

    def __call__(self, action, *, packed_latents, flow_time, condition):
        self.shapes.append(packed_latents.shape[0])
        return SimpleNamespace(latent_velocity=.3 * packed_latents + flow_time[..., None, None]
            + action[..., None] + condition.edge_tokens + condition.reference_tokens)


def test_two_picard_sweeps_are_not_equivalent_to_euler_100():
    class LinearField(Field):
        def __call__(self, action, *, packed_latents, **kwargs):
            return SimpleNamespace(latent_velocity=packed_latents)
    field = LinearField()
    start = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    action = torch.zeros(1, 1, 1, dtype=torch.float64)
    condition = ConditionEncoding(start, start)
    euler = field.solve(start, action, condition, solver='euler', steps=100)
    approximate = field.solve(start, action, condition, solver='picard', steps=100,
                              picard_iterations=2)
    converged = field.solve(start, action, condition, solver='picard', steps=100,
                            picard_iterations=100)
    torch.testing.assert_close(approximate, torch.full_like(start, 2.495))
    torch.testing.assert_close(euler, torch.full_like(start, 1.01**100))
    torch.testing.assert_close(converged, euler, rtol=1e-12, atol=1e-12)
    assert float((approximate - euler).abs()) > .2


def solve_case(chunk):
    field = Field()
    start = torch.arange(2 * 3 * 2 * 4, dtype=torch.float64).reshape(2, 3, 2, 4).div(100).requires_grad_()
    action = torch.arange(12, dtype=torch.float64).reshape(2, 3, 2).div(20).requires_grad_()
    edge = (start.detach() + .2).requires_grad_()
    reference = (start.detach() + .4).requires_grad_()
    result = field.solve(start, action, ConditionEncoding(edge, reference),
                         steps=7, solver='picard', picard_iterations=4, picard_batch_chunk_size=chunk)
    result.square().sum().backward()
    return result, [x.grad for x in (start, action, edge, reference)], field.shapes


@pytest.mark.parametrize('chunk', [1, 3, 5, 14, 64])
def test_chunked_picard_matches_unbounded_outputs_and_gradients(chunk):
    reference, gradients, _ = solve_case(0)
    observed, actual_gradients, shapes = solve_case(chunk)
    torch.testing.assert_close(observed, reference, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(actual_gradients, gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert max(shapes) <= chunk
    assert len(shapes) == 4 * ((14 + chunk - 1) // chunk)


@pytest.mark.parametrize('chunk', [-1, True, 1.5])
def test_invalid_chunk_is_rejected(chunk):
    start = torch.ones(2, 1, 2, 4)
    with pytest.raises(ValueError, match='picard_batch_chunk_size'):
        Field().solve(start, torch.zeros(2, 1, 2), ConditionEncoding(start, start),
                      solver='picard', picard_batch_chunk_size=chunk)


def test_actual_dynamics_chunking_preserves_fixed_noise_samples_and_disagreement():
    torch.manual_seed(134)
    dynamics = Dynamics(adjacency=torch.tensor([[0., 1.], [1., 0.]]),
                        static_features=torch.ones(2, 4)).eval()
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.01)
    start = torch.randn(2, 2, 64)
    action = torch.randn(2, 2)
    noise = torch.randn(3, 2, dynamics.graph_prior.rank, 64)
    with torch.no_grad():
        before = dynamics.latent_disagreement(start, action, noise=noise, n_samples=3,
            steps=7, solver='picard', picard_iterations=4, picard_batch_chunk_size=0)
        after = dynamics.latent_disagreement(start, action, noise=noise, n_samples=3,
            steps=7, solver='picard', picard_iterations=4, picard_batch_chunk_size=5)
    for actual, expected in zip(after, before):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(torch.isfinite(value).all() for value in after)
