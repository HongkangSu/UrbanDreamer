import copy
from types import MethodType

import pytest
import torch
from torch import nn

from tools.train_direct_context_fm import DirectResidual
from tools.train_partial_context_fm import (
    ARMS, EXPECTED_PARAMETERS, allowed_name, cfm_loss, configure_trainable,
    gradient_proof, parameter_digest, partial_context, run_pilot, selected_state,
)
from urban_dreamer.models.world_model import ConditionEncoding, Dynamics


class TinyModel(nn.Module):
    """Actual six-block production dynamics, with a three-road graph only."""
    def __init__(self):
        super().__init__()
        self.dynamics = Dynamics(adjacency=torch.ones(3, 3) - torch.eye(3),
                                 static_features=torch.randn(3, 5), low_frequency_rank=2)
        # A trained checkpoint has a nonzero flow head; constructor defaults are
        # deliberately zero and would hide all initial reference/body gradients.
        with torch.no_grad():
            self.dynamics.flow_head.weight.normal_(std=.02)


@pytest.fixture(scope='module')
def template():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = TinyModel().eval()
    yield model
    torch.set_num_threads(old_threads)


def example():
    return dict(z=torch.ones(1, 1, 3, 64), target=torch.full((1, 1, 3, 64), 1.3),
                b=torch.full((1, 1, 3, 64), .2), condition=torch.zeros(1, 1, 3, 256),
                action=torch.zeros(1, 1, 3), noise=torch.full((1, 1, 2, 64), .4),
                time=torch.full((1, 1), .35))


def test_exact_trainable_modules_parameter_count_and_saved_scope(template):
    model = copy.deepcopy(template)
    residual = DirectResidual()
    params = configure_trainable(model, residual)
    assert sum(p.numel() for p in params) == EXPECTED_PARAMETERS == 1_414_528
    assert sum(p.numel() for p in model.dynamics.transformer.layers[-1].parameters()) == 1_315_840
    assert sum(p.numel() for p in model.dynamics.reference_proj.parameters()) == 16_384
    assert sum(p.numel() for p in model.dynamics.flow_head.parameters()) == 16_448
    assert all(p.requires_grad == allowed_name(name) for name, p in model.named_parameters())
    assert not any(p.requires_grad for layer in model.dynamics.transformer.layers[:-1] for p in layer.parameters())
    state = selected_state(model, residual)
    assert all(allowed_name(name) for name in state['model'])
    assert sum(v.numel() for v in state['model'].values()) + sum(v.numel() for v in state['residual'].values()) == EXPECTED_PARAMETERS


@pytest.mark.parametrize('arm', tuple(ARMS))
def test_source_reference_and_all_hints_are_independent_without_future_input(template, arm):
    model = copy.deepcopy(template)
    residual = DirectResidual()
    configure_trainable(model, residual)
    d, batch = model.dynamics, example()
    old_head, original_method = d.flow_head, d.prior_start.__func__
    condition = ConditionEncoding(batch['condition'])
    source_on, reference_on = ARMS[arm]
    with partial_context(d, residual, arm, dict(z=1., b=1., relative=1.), background=batch['b']) as c:
        source, increment, cached = d.prior_start(batch['z'], condition, batch['noise'])
        mean = batch['z'] + batch['b'] if source_on else batch['z']
        reference = batch['z'] + batch['b'] if reference_on else batch['z']
        torch.testing.assert_close(source, mean + d.graph_prior.sample_noise(batch['z'], batch['noise']), rtol=0, atol=0)
        torch.testing.assert_close(increment, batch['b'] if source_on else torch.zeros_like(batch['b']), rtol=0, atol=0)
        torch.testing.assert_close(cached.reference_tokens, d.reference_proj(reference), rtol=0, atol=0)
        torch.testing.assert_close(c['mean'], mean, rtol=0, atol=0)
        torch.testing.assert_close(c['hint'], batch['b'], rtol=0, atol=0)
        assert cached.reference_tokens.requires_grad
        generated = torch.full_like(batch['z'], 7.)
        d(batch['action'], packed_latents=generated, flow_time=batch['time'], condition=cached)
        assert c['xt'] is generated and c['time'] is batch['time']
        assert 'target' not in c
    assert d.flow_head is old_head and d.prior_start.__func__ is original_method
    assert 'prior_start' not in d.__dict__ and not d._forward_pre_hooks


def test_restores_existing_instance_method_and_head_on_exception(template):
    model = copy.deepcopy(template)
    d = model.dynamics
    original = MethodType(lambda self, *a, **kw: None, d)
    d.prior_start = original
    old_head = d.flow_head
    with pytest.raises(RuntimeError):
        with partial_context(d, DirectResidual(), 'C1', dict(z=1., b=1., relative=1.)):
            raise RuntimeError('deliberate failure')
    assert d.prior_start is original and d.flow_head is old_head and not d._forward_pre_hooks


@pytest.mark.parametrize('arm', tuple(ARMS))
def test_full_velocity_target_and_gradient_through_frozen_prefix(template, arm):
    model = copy.deepcopy(template)
    residual = DirectResidual()
    configure_trainable(model, residual)
    batch = example()
    batch['z'].requires_grad_(True)
    batch['target'].requires_grad_(True)
    batch['b'].requires_grad_(True)
    scales = dict(z=1., b=1., relative=1.)
    before = parameter_digest(model, residual)
    loss = cfm_loss(model, residual, arm, batch, scales)
    # Independent reconstruction of the ordinary CFM velocity loss, rather
    # than the old cached residual-to-original-head target.
    z, target = batch['z'].detach(), batch['target'].detach()
    condition = ConditionEncoding(batch['condition'])
    with partial_context(model.dynamics, residual, arm, scales, background=batch['b'].detach()):
        source, _, cached = model.dynamics.prior_start(z, condition, batch['noise'])
        t = batch['time'][:, :, None, None]
        velocity = model.dynamics(batch['action'], packed_latents=(1-t)*source+t*target,
                                  flow_time=batch['time'], condition=cached).latent_velocity
        expected = (velocity - (target-source).detach()).square().mean()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    proof = gradient_proof(model, residual)
    assert proof['frozen_parameter_gradients_absent']
    assert all(v > 0 for v in proof['nonzero_gradient_norms'].values())
    assert batch['z'].grad is None and batch['target'].grad is None and batch['b'].grad is None
    assert parameter_digest(model, residual) == before


def test_pilot_does_not_step_mutate_parameters_consume_rng_or_leave_gradients(template):
    models, residuals, params = {}, {}, {}
    for arm in ARMS:
        models[arm] = copy.deepcopy(template)
        torch.manual_seed(37)
        residuals[arm] = DirectResidual()
        params[arm] = configure_trainable(models[arm], residuals[arm])
    proof = run_pilot(models, residuals, params, example(), dict(z=1., b=1., relative=1.), warmups=0, repeats=1)
    assert proof['optimizer_steps'] == 0 and len(proof['records']) == 3
    assert proof['parameters_unchanged'] and proof['rng_unchanged'] and proof['gradients_cleared']
    assert all(p.grad is None for arm in ARMS for p in params[arm])
