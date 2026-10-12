"""Residual units, frozen-center isolation, reproducible sampling and strict restore."""
import copy
import io

import pytest
import torch
from torch import nn

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.experimental_dynamics import apply_variant
from urban_dreamer.models.residual_dynamics import ResidualDynamics, residual_fm_loss
from urban_dreamer.models.world_model import UrbanDreamerWorldModel


def identity(**updates):
    result = dict(mean_source_sha256='a' * 64, codec_sha256='b' * 64,
                  manifest_sha256='c' * 64, scale_artifact_sha256='d' * 64)
    result.update(updates)
    return result


def center(seed=19):
    torch.manual_seed(seed)
    model = UrbanDreamerWorldModel(adjacency=torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]),
        static_features=torch.ones(3, 4), low_frequency_rank=1, event_protocol=True)
    model.dynamics.transformer.layers = nn.ModuleList(model.dynamics.transformer.layers[:1])
    nn.init.normal_(model.dynamics.flow_head.weight, std=.01)
    apply_variant(model, 'regression', seed=42)
    return model.eval()


@pytest.fixture
def case():
    torch.set_num_threads(1)
    world = center()
    z, action = torch.randn(2, 1, 3, 64), torch.rand(2, 1, 3)
    events = pack_action_events([[(ActionEvent(str(batch), batch, 1., 0, 1, 2),)] for batch in range(2)])
    scale = torch.linspace(.2, 2., 64)
    dynamics = ResidualDynamics(world.dynamics, scale, identity=identity(), initialization_seed=42)
    return world, dynamics, z, action, events


def clone_adapter(dynamics, *, random_shell=True, identity_override=None):
    shell = center(seed=123).dynamics if random_shell else copy.deepcopy(dynamics.mean_dynamics)
    return ResidualDynamics(shell, dynamics.scale, identity=identity_override or dynamics.identity, initialization_seed=999)


def test_center_parity_frozen_modes_single_ownership_and_counts(case):
    world, dynamics, z, action, events = case
    expected = world.dynamics.transition(z, action, action_events=events)
    mean, condition = dynamics.mean_and_condition(z, action, action_events=events)
    torch.testing.assert_close(mean, expected, rtol=0, atol=0)
    assert not mean.requires_grad and not condition.edge_tokens.requires_grad
    assert dynamics.mean_dynamics is world.dynamics
    assert dynamics.event_encoder is world.dynamics.event_encoder
    assert not hasattr(dynamics, 'graph_prior')
    assert all(not p.requires_grad for p in dynamics.mean_dynamics.parameters())
    assert {id(p) for p in dynamics.parameters() if p.requires_grad} == {id(p) for p in dynamics.residual_velocity.parameters()}
    world.dynamics = dynamics
    world.train()
    assert dynamics.training and dynamics.residual_velocity.training
    assert not dynamics.mean_dynamics.training
    for setter in (dynamics.set_initial_chunk_profile, dynamics.set_online_chunk_profile):
        setter()
        assert not dynamics.mean_dynamics.transformer.activation_checkpointing
    dynamics.set_pretraining_execution_profile(space_chunk=1, time_chunk=2, mlp_chunk=8, activation_checkpointing=True)
    assert not dynamics.mean_dynamics.transformer.activation_checkpointing
    assert not any('flow_projector' in name for name, _ in dynamics.named_parameters())
    metadata = dynamics.metadata()
    assert metadata['trainable_parameters'] == sum(p.numel() for p in dynamics.residual_velocity.parameters())
    assert metadata['expected_source_energy'] == pytest.approx(3 * dynamics.scale.double().square().sum().item())
    torch.testing.assert_close(torch.tensor(metadata['source_variance_per_channel']), dynamics.scale.square())


@pytest.mark.parametrize('solver,steps,nfe', [('euler', 1, 1), ('euler', 7, 7), ('midpoint', 1, 2), ('midpoint', 5, 10)])
def test_constant_field_integrates_in_correct_original_units(case, monkeypatch, solver, steps, nfe):
    _, dynamics, z, action, events = case
    mean, _ = dynamics.mean_and_condition(z, action, action_events=events)
    noise = torch.linspace(-1., 1., z.numel()).reshape_as(z)
    calls = []
    def constant(residual, tau, context):
        calls.append(tau.clone())
        return torch.full_like(residual, .25)
    monkeypatch.setattr(dynamics.residual_velocity, 'forward', constant)
    actual = dynamics.transition(z, action, action_events=events, noise=noise, solver=solver, steps=steps)
    torch.testing.assert_close(actual, mean + dynamics.scale * (noise + .25), rtol=1e-6, atol=1e-6)
    assert len(calls) == nfe
    assert dynamics.last_execution['residual_nfe_per_sample'] == nfe
    assert dynamics.last_execution['center_forwards'] == 1


@pytest.mark.parametrize('solver', ['euler', 'midpoint'])
def test_solver_nodes_and_numerical_update_match_linear_field(case, monkeypatch, solver):
    _, dynamics, z, action, events = case
    mean, _ = dynamics.mean_and_condition(z, action, action_events=events)
    nodes = []
    def linear(residual, tau, context):
        nodes.append(float(tau[0, 0]))
        return .5 * residual
    monkeypatch.setattr(dynamics.residual_velocity, 'forward', linear)
    actual = dynamics.transition(z, action, noise=torch.ones_like(z), action_events=events, steps=2, solver=solver)
    factor = (1.25 if solver == 'euler' else 1. + .25 + .25 ** 2 / 2) ** 2
    torch.testing.assert_close(actual, mean + dynamics.scale * factor, rtol=1e-6, atol=1e-6)
    assert nodes == ([0., .5] if solver == 'euler' else [0., .25, .5, .75])


def test_known_normalized_target_interpolant_velocity_and_loss(case, monkeypatch):
    _, dynamics, z, action, events = case
    mean, _ = dynamics.mean_and_condition(z, action, action_events=events)
    residual_target = torch.full_like(z, 2.)
    target = mean + dynamics.scale * residual_target
    noise = torch.full_like(z, -.5)
    tau = torch.full((2, 1), .3)
    captured = []
    def constant(point, flow_time, context):
        captured.append((point.detach().clone(), flow_time.detach().clone()))
        return torch.full_like(point, .25)
    monkeypatch.setattr(dynamics.residual_velocity, 'forward', constant)
    loss, metrics, predicted = residual_fm_loss(dynamics, z_inputs=z, z_targets=target,
        action=action, action_events=events, noise=noise, flow_time=tau, predict_endpoints=False)
    torch.testing.assert_close(captured[0][0], torch.full_like(z, .25), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loss, torch.tensor((.25 - 2.5) ** 2))
    torch.testing.assert_close(metrics['normalized_target_rms'], torch.tensor(2.))
    torch.testing.assert_close(predicted, target, rtol=0, atol=0)
    assert not predicted.requires_grad


def test_zero_head_has_full_support_and_expected_original_unit_variance(case):
    _, dynamics, z, action, events = case
    mean, _ = dynamics.mean_and_condition(z, action, action_events=events)
    basis = dynamics.mean_dynamics.graph_prior.noise_basis
    direction = torch.tensor([1., -2., 3.])
    direction -= basis @ (basis.T @ direction)
    assert direction.abs().sum() > .1
    perturbation = direction.reshape(1, 1, 3, 1).expand_as(z)
    noises = torch.stack([-perturbation, torch.zeros_like(z), perturbation])
    before = torch.random.get_rng_state()
    samples, global_variance, edge_variance = dynamics.latent_disagreement(z, action,
        action_events=events, n_samples=3, noise=noises)
    assert torch.equal(before, torch.random.get_rng_state())
    expected = mean.unsqueeze(0) + dynamics.scale * noises
    torch.testing.assert_close(samples, expected, rtol=0, atol=0)
    torch.testing.assert_close(edge_variance, expected.var(0, unbiased=True).mean(-1), rtol=0, atol=0)
    torch.testing.assert_close(global_variance, edge_variance.mean(-1), rtol=0, atol=0)
    with pytest.raises(ValueError, match='full latent shape'):
        dynamics.transition(z, action, action_events=events, noise=torch.ones(2, 1, 1, 64))


@pytest.mark.parametrize('squeezed', [False, True])
def test_sample_order_and_chunking_cache_one_center_and_context(case, monkeypatch, squeezed):
    _, dynamics, z, action, events = case
    if squeezed:
        z, action = z[:, 0], action[:, 0]
    noises = torch.stack([torch.full_like(z, value) for value in (-1., 0., .5, 1.)])
    count = dict(center=0, context=0, condition=0)
    center_call = dynamics.mean_dynamics.backbone.register_forward_hook(
        lambda *args: count.__setitem__('center', count['center'] + 1))
    original_context = dynamics.residual_velocity.context
    original_condition = dynamics.mean_dynamics.encode_condition
    def context(*args):
        count['context'] += 1
        return original_context(*args)
    def condition(*args, **kwargs):
        count['condition'] += 1
        return original_condition(*args, **kwargs)
    monkeypatch.setattr(dynamics.residual_velocity, 'context', context)
    monkeypatch.setattr(dynamics.mean_dynamics, 'encode_condition', condition)
    expected, _, _ = dynamics.latent_disagreement(z, action, noise=noises, n_samples=4, action_events=events)
    assert count == dict(center=1, context=1, condition=1)
    count.update(center=0, context=0, condition=0)
    actual, _, _ = dynamics.latent_disagreement(z, action, noise=noises, n_samples=4,
        sample_chunk_size=1, action_events=events)
    assert count == dict(center=1, context=1, condition=1)
    assert actual.shape == (4, *z.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for sample in range(4):
        single = dynamics.transition(z, action, noise=noises[sample], action_events=events)
        torch.testing.assert_close(single, actual[sample], rtol=0, atol=0)
    center_call.remove()


def test_random_draw_order_matches_chunked_call_and_explicit_noise(case):
    _, dynamics, z, action, events = case
    rng = torch.random.get_rng_state()
    noise = torch.randn(3, *z.shape)
    expected = dynamics.latent_disagreement(z, action, noise=noise, n_samples=3, action_events=events)[0]
    torch.random.set_rng_state(rng)
    actual = dynamics.latent_disagreement(z, action, n_samples=3, action_events=events, sample_chunk_size=1)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_only_residual_head_gets_gradients_even_with_live_external_codec(case):
    world, dynamics, _, action, events = case
    nn.init.normal_(dynamics.residual_velocity.output.weight, std=.01)
    raw = torch.randn(2, 1, 3, 8, requires_grad=True)
    current = world.encode(raw)
    target = world.encode(raw + .3)
    current.retain_grad()
    before = {name: value.clone() for name, value in dynamics.mean_dynamics.named_parameters()}
    loss, _, _ = residual_fm_loss(dynamics, z_inputs=current, z_targets=target,
        action=action.requires_grad_(), action_events=events, predict_endpoints=False)
    loss.backward()
    assert raw.grad is None and current.grad is None and action.grad is None
    assert all(p.grad is None for p in world.flow_projector.parameters())
    assert all(p.grad is None for p in dynamics.mean_dynamics.parameters())
    for layer in (dynamics.residual_velocity.output, dynamics.residual_velocity.state_proj,
                  dynamics.residual_velocity.mean_proj, dynamics.residual_velocity.condition_proj,
                  dynamics.residual_velocity.time_proj, dynamics.residual_velocity.input_proj):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    optimizer = torch.optim.Adam([p for p in dynamics.parameters() if p.requires_grad], lr=1e-3)
    optimizer.step()
    assert all(torch.equal(value, before[name]) for name, value in dynamics.mean_dynamics.named_parameters())


def test_train_and_inference_detach_current_state_and_do_not_cache_across_calls(case):
    _, dynamics, z, action, events = case
    nn.init.normal_(dynamics.residual_velocity.output.weight, std=.01)
    z.requires_grad_()
    result = dynamics.transition(z, action, action_events=events, noise=torch.zeros_like(z))
    result.square().mean().backward()
    assert z.grad is None
    assert dynamics.residual_velocity.output.weight.grad.abs().sum() > 0
    dynamics.zero_grad(set_to_none=True)
    result2 = dynamics.transition(z + .2, action, action_events=events, noise=torch.zeros_like(z))
    result2.square().mean().backward()
    assert not torch.allclose(result, result2)


def test_frozen_center_stays_fp32_under_head_autocast(case, monkeypatch):
    _, dynamics, z, action, events = case
    seen = []
    hook = dynamics.mean_dynamics.backbone.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append((kwargs['packed_latents'].dtype,
            torch.is_autocast_enabled('cpu'))), with_kwargs=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _, prediction = residual_fm_loss(dynamics, z_inputs=z, z_targets=z + .5,
            action=action, action_events=events)
    assert loss.dtype == prediction.dtype == torch.float32
    assert seen == [(torch.float32, False)]
    hook.remove()


def test_initialization_does_not_shift_rng_and_is_independent_of_mean_identity(case):
    _, dynamics, _, _, _ = case
    first = center(seed=2).dynamics
    second = center(seed=3).dynamics
    before = torch.random.get_rng_state()
    one = ResidualDynamics(first, dynamics.scale, identity=identity(), initialization_seed=42)
    assert torch.equal(before, torch.random.get_rng_state())
    two = ResidualDynamics(second, dynamics.scale, identity=identity(mean_source_sha256='e' * 64), initialization_seed=42)
    assert torch.equal(before, torch.random.get_rng_state())
    for key, value in one.residual_velocity.state_dict().items():
        torch.testing.assert_close(value, two.residual_velocity.state_dict()[key], rtol=0, atol=0)


def test_checkpoint_is_self_contained_and_restores_random_shell(case):
    _, dynamics, z, action, events = case
    nn.init.normal_(dynamics.residual_velocity.output.weight, std=.01)
    noise = torch.randn_like(z)
    expected = dynamics.transition(z, action, noise=noise, action_events=events)
    stream = io.BytesIO()
    torch.save(dynamics.state_dict(), stream)
    stream.seek(0)
    restored = clone_adapter(dynamics)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert restored.get_extra_state() == dynamics.get_extra_state()
    actual = restored.transition(z, action, noise=noise, action_events=events)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any('action_encoder.static_features' in key for key in restored.state_dict())
    assert restored.mean_dynamics.get_extra_state()['variant'] == 'regression'


@pytest.mark.parametrize('key', ['mean_source_sha256', 'codec_sha256', 'manifest_sha256', 'scale_artifact_sha256'])
def test_checkpoint_rejects_source_codec_and_scale_identity_mismatch(case, key):
    _, dynamics, _, _, _ = case
    mismatched = dynamics.identity
    mismatched[key] = 'f' * 64
    restored = clone_adapter(dynamics, identity_override=mismatched)
    with pytest.raises(ValueError, match='identity'):
        restored.load_state_dict(dynamics.state_dict(), strict=True)


def test_scale_and_mean_tampering_are_detected(case):
    _, dynamics, z, action, events = case
    restored = clone_adapter(dynamics)
    state = copy.deepcopy(dynamics.state_dict())
    state['scale'][0] *= 2
    with pytest.raises(ValueError, match='scale buffer'):
        restored.load_state_dict(state)
    state = copy.deepcopy(dynamics.state_dict())
    parameter = next(key for key in state if key.startswith('mean_dynamics.') and key.endswith('flow_head.weight'))
    state[parameter][0, 0] += .1
    with pytest.raises(ValueError, match='mean content digest'):
        restored.load_state_dict(state)
    dynamics.scale[0] *= 2
    with pytest.raises(ValueError, match='scale is immutable'):
        dynamics.transition(z, action, action_events=events)


def test_optimizer_and_rng_restore_reproduce_exact_next_update(case):
    _, dynamics, z, action, events = case
    optimizer = torch.optim.Adam(dynamics.residual_velocity.parameters(), lr=1e-3)
    def step(model, opt):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = residual_fm_loss(model, z_inputs=z, z_targets=z + .1,
            action=action, action_events=events, predict_endpoints=False)
        loss.backward()
        opt.step()
        return loss.detach()
    step(dynamics, optimizer)
    saved = copy.deepcopy(dynamics.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_rng = torch.random.get_rng_state()
    expected = step(dynamics, optimizer)
    restored = clone_adapter(dynamics)
    restored.load_state_dict(saved)
    restored_optimizer = torch.optim.Adam(restored.residual_velocity.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(saved_optimizer)
    torch.random.set_rng_state(saved_rng)
    actual = step(restored, restored_optimizer)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for left, right in zip(restored.parameters(), dynamics.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize('kwargs', [dict(solver='picard'), dict(picard_iterations=2),
                                    dict(steps=0), dict(steps=True), dict(steps=1.5)])
def test_unsupported_solver_and_steps_are_explicit(case, kwargs):
    _, dynamics, z, action, events = case
    with pytest.raises(ValueError, match='solver'):
        dynamics.transition(z, action, action_events=events, **kwargs)


def test_shapes_finiteness_empty_events_and_single_step_training_contract(case):
    _, dynamics, z, action, _ = case
    empty = pack_action_events([[()] for _ in range(2)])
    result = dynamics.transition(z, action, action_events=empty, noise=torch.zeros_like(z))
    assert result.shape == z.shape
    with pytest.raises(ValueError, match='action shape'):
        dynamics.transition(z, action[:, :, :2], action_events=empty)
    with pytest.raises(ValueError, match='finite'):
        dynamics.transition(z * float('nan'), action, action_events=empty)
    with pytest.raises(ValueError, match='finite'):
        dynamics.transition(z, action, action_events=empty, noise=torch.full_like(z, float('inf')))
    with pytest.raises(ValueError, match='single-step'):
        residual_fm_loss(dynamics, z_inputs=z.expand(-1, 2, -1, -1),
            z_targets=z.expand(-1, 2, -1, -1), action=action.expand(-1, 2, -1), action_events=empty)
    with pytest.raises(ValueError, match='at least two'):
        dynamics.latent_disagreement(z, action, n_samples=1, action_events=empty)
    with pytest.raises(ValueError, match='flow_time'):
        residual_fm_loss(dynamics, z_inputs=z, z_targets=z, action=action,
            action_events=empty, flow_time=torch.full((2, 1), 2.))


@pytest.mark.parametrize('scale', [torch.zeros(64), torch.ones(63), torch.full((64,), float('nan'))])
def test_bad_scales_fail_before_center_freezing(case, scale):
    world, _, _, _, _ = case
    with pytest.raises(ValueError, match='64 finite positive'):
        ResidualDynamics(world.dynamics, scale, identity=identity(), initialization_seed=42)
