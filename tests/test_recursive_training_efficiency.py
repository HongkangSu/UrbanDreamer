"""Execution savings must preserve recursive supervision and full gradients."""
import copy

import pytest
import torch

from urban_dreamer.models.world_model import Dynamics, recursive_fm_loss


def dynamics():
    torch.manual_seed(117)
    model = Dynamics(adjacency=torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
        static_features=torch.zeros(3, 4), low_frequency_rank=2,
        prior_filter='soft', prior_filter_strength=.1,
        prior_action_conditioned=False, prior_usage='condition_only')
    with torch.no_grad():
        model.flow_head.weight.normal_(std=.01)
        model.condition_residual.output.weight.normal_(std=.01)
        model.graph_prior.output.weight.normal_(std=.01)
    return model


@pytest.mark.parametrize('horizon', [1, 3])
@pytest.mark.parametrize('solver', ['euler', 'midpoint', 'picard'])
def test_omitting_unused_endpoints_preserves_losses_gradients_and_rng(horizon, solver):
    reference = dynamics()
    candidate = copy.deepcopy(reference)
    sequence = torch.randn(1, horizon + 1, 3, 64)
    action = torch.rand(1, horizon, 3)
    results = []
    for model, predict in ((reference, True), (candidate, False)):
        z = sequence.clone().requires_grad_()
        a = action.clone().requires_grad_()
        calls = []
        solve = model.solve
        def counted(*args, **kwargs):
            value = solve(*args, **kwargs)
            value.retain_grad()
            calls.append(value)
            return value
        model.solve = counted
        torch.manual_seed(336)
        loss, metrics, predicted = recursive_fm_loss(model, z_sequence=z, action=a,
            steps=2, solver=solver, picard_iterations=2, predict_endpoints=predict)
        assert len(calls) == (horizon if predict else horizon - 1)
        assert (predicted is not None) == predict
        loss.backward()
        if horizon > 1:
            assert calls[0].grad is not None and calls[0].grad.abs().sum() > 0
        results.append((loss.detach(), metrics, z.grad, a.grad,
                        {name: None if p.grad is None else p.grad.clone()
                         for name, p in model.named_parameters()}, torch.get_rng_state()))
    for left, right in zip(results[0], results[1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_recursive_endpoint_output_remains_the_default():
    model = dynamics()
    _, _, predicted = recursive_fm_loss(model, z_sequence=torch.randn(1, 3, 3, 64),
        action=torch.zeros(1, 2, 3), steps=1)
    assert predicted.shape == (1, 2, 3, 64)


@pytest.mark.parametrize('solver', ['euler', 'midpoint'])
@pytest.mark.parametrize('interval', [1, 2, 4])
def test_solver_checkpoints_preserve_endpoints_and_all_gradients(solver, interval):
    reference = dynamics()
    candidate = copy.deepcopy(reference)
    sequence = torch.randn(1, 4, 3, 64)
    action = torch.rand(1, 3, 3)
    noise = torch.randn(1, 3, 2, 64)
    time = torch.rand(1, 3)
    results = []
    for model, checkpoint_interval in ((reference, 0), (candidate, interval)):
        z = sequence.clone().requires_grad_()
        a = action.clone().requires_grad_()
        loss, metrics, predicted = recursive_fm_loss(model, z_sequence=z, action=a,
            steps=5, solver=solver, noise=noise, flow_time=time,
            checkpoint_interval=checkpoint_interval)
        # Include endpoint loss only in this test, proving caller gradients from
        # all returned endpoints and later physical-transition losses survive.
        (loss + predicted.square().mean()).backward()
        results.append((loss.detach(), metrics, predicted.detach(), z.grad, a.grad,
                        {name: None if p.grad is None else p.grad.clone()
                         for name, p in model.named_parameters()}, torch.get_rng_state()))
    for left, right in zip(results[0][:-1], results[1][:-1]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(results[0][-1], results[1][-1], rtol=0, atol=0)


@pytest.mark.parametrize('invalid', [True, -1, 1.5])
def test_solver_checkpoint_interval_rejects_invalid_values(invalid):
    model = dynamics()
    z = torch.randn(1, 1, 3, 64)
    action = torch.zeros(1, 1, 3)
    condition = model.encode_condition(action)
    start, _, condition = model.prior_start(z, condition)
    with pytest.raises(ValueError, match='checkpoint_interval'):
        model.solve(start, action, condition, checkpoint_interval=invalid)
    with pytest.raises(ValueError, match='checkpoint_interval'):
        recursive_fm_loss(model, z_sequence=torch.randn(1, 2, 3, 64),
            action=torch.zeros(1, 1, 3), checkpoint_interval=invalid, predict_endpoints=False)


def test_checkpointed_recursive_solve_supports_compiled_blocks(monkeypatch):
    import urban_dreamer.models.world_model as module
    monkeypatch.setattr(module, 'TRANSFORMER_DEPTH', 1)
    reference = dynamics()
    candidate = copy.deepcopy(reference)
    for layer in candidate.transformer.layers:
        layer.forward = torch.compile(layer.forward, backend='aot_eager', fullgraph=True)
    sequence = torch.randn(1, 4, 3, 64)
    action = torch.rand(1, 3, 3)
    results = []
    for model, interval in ((reference, 0), (candidate, 2)):
        torch.manual_seed(35)
        loss, _, _ = recursive_fm_loss(model, z_sequence=sequence, action=action,
            steps=3, checkpoint_interval=interval, predict_endpoints=False)
        loss.backward()
        results.append((loss.detach(), {name: None if p.grad is None else p.grad
                                       for name, p in model.named_parameters()}))
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=2e-6)
