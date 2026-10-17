import copy
import io

import pytest
import torch
from torch import nn

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.experimental_dynamics import (
    VARIANTS, ExperimentalPrior, apply_variant, experimental_loss, predict_next, variant_metadata,
)
from urban_dreamer.models.world_model import (
    UrbanDreamerWorldModel, LowFrequencyGraphPrior, independent_transition_fm_loss,
)


@pytest.fixture
def case():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = UrbanDreamerWorldModel(adjacency=torch.ones(3, 3) - torch.eye(3),
        static_features=torch.ones(3, 4), low_frequency_rank=1, event_protocol=True)
    # Keep the actual attention/condition implementation, with one layer for CPU tests.
    model.dynamics.transformer.layers = nn.ModuleList(model.dynamics.transformer.layers[:1])
    model.dynamics.transformer.activation_checkpointing = False
    nn.init.normal_(model.dynamics.flow_head.weight, std=.01)
    nn.init.normal_(model.dynamics.graph_prior.output.weight, std=.01)
    model.eval()
    inputs, targets = torch.randn(2, 2, 3, 64), torch.randn(2, 2, 3, 64)
    action = torch.rand(2, 2, 3)
    events = pack_action_events([
        [(ActionEvent(f"vehicle{batch}-{step}", 2 * batch + step, 30. * step + 1., 0, 1, 2),)
         for step in range(2)] for batch in range(2)])
    return model, inputs, targets, action, events


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_train_backbone_and_event_encoder(case, variant):
    model, inputs, targets, action, events = case
    apply_variant(model, variant, seed=4)
    loss, metrics, predicted = experimental_loss(model, z_inputs=inputs,
        z_targets=targets, action=action, action_events=events)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert predicted.shape == inputs.shape and not predicted.requires_grad
    assert all(torch.isfinite(value).all() for value in metrics.values())
    loss.backward()
    for parameter in (model.dynamics.flow_head.weight,
                      model.dynamics.event_encoder.event_mlp[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    if variant == "gaussian":
        assert model.dynamics.flow_head.weight.grad[64:].abs().sum() > 0
    if variant in {"no_prior_fm", "regression", "gaussian", "isotropic_noise_no_prior_fm"}:
        assert list(model.dynamics.graph_prior.parameters()) == []
        assert metrics["prior_loss"] == 0


def test_graph_fm_is_exact_production_path(case):
    model, inputs, targets, action, events = case
    original = model.dynamics
    state_keys = tuple(model.state_dict())
    apply_variant(model, "graph_fm")
    assert model.dynamics is original
    assert tuple(model.state_dict()) == state_keys
    args = dict(z_inputs=inputs, z_targets=targets, action=action, action_events=events,
                flow_time=torch.full((2, 2), .3), noise=torch.randn(2, 2, 1, 64))
    actual = experimental_loss(model, **args)
    expected = independent_transition_fm_loss(original, **args)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["no_prior_fm", "random_prior_fm"])
def test_coarse_prior_change_preserves_noise_exactly(case, variant):
    model, z, _, action, events = case
    prior = copy.deepcopy(model.dynamics.graph_prior)
    noise = torch.randn(2, 2, 1, 64)
    apply_variant(model, variant, seed=99)
    changed = model.dynamics.graph_prior
    torch.testing.assert_close(changed.sample_noise(z, noise), prior.sample_noise(z, noise), rtol=0, atol=0)
    condition = model.dynamics.encode_condition(action, action_events=events)
    if variant == "no_prior_fm":
        assert torch.count_nonzero(changed(z, condition)) == 0
        assert torch.count_nonzero(prior(z, condition)) > 0
    else:
        assert not torch.allclose(changed.basis, prior.basis)
        assert not torch.allclose(changed(z, condition), prior(z, condition))


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_and_sampling_roundtrip(case, variant):
    model, z, _, action, events = case
    restored = copy.deepcopy(model)
    apply_variant(model, variant, seed=7)
    apply_variant(restored, variant, seed=123)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert variant_metadata(restored)["variant"] == variant
    if variant != "graph_fm":
        assert variant_metadata(restored)["projection_seed"] == 7
    shape = (3, *z.shape) if variant == "gaussian" else (3, 2, 2, model.dynamics.graph_prior.rank, 64)
    noise = torch.randn(shape)
    kwargs = dict(action_events=events, n_samples=3, noise=noise)
    samples, global_variance, edge_variance = model.latent_disagreement(z, action, **kwargs)
    expected, _, _ = restored.latent_disagreement(z, action, **kwargs)
    assert samples.shape == (3, *z.shape)
    assert global_variance.shape == (2, 2) and edge_variance.shape == (2, 2, 3)
    torch.testing.assert_close(samples, expected, rtol=0, atol=0)
    if variant in {"regression", "graph_regression"}:
        assert torch.count_nonzero(edge_variance) == 0
    else:
        assert edge_variance.sum() > 0
    prediction = predict_next(model, z[:, 0], action[:, 0], action_events=events.for_step(0))
    assert prediction.shape == z[:, 0].shape


@pytest.mark.parametrize("variant", ["regression", "graph_regression", "gaussian"])
def test_training_flattens_independent_windows(case, variant):
    model, inputs, targets, action, events = case
    apply_variant(model, variant)
    before, _, predictions = experimental_loss(model, z_inputs=inputs,
        z_targets=targets, action=action, action_events=events)
    losses = []
    for step in range(2):
        loss, _, prediction = experimental_loss(model, z_inputs=inputs[:, step:step+1],
            z_targets=targets[:, step:step+1], action=action[:, step:step+1],
            action_events=events.for_step(step))
        losses.append(loss)
        torch.testing.assert_close(predictions[:, step:step+1], prediction)
    torch.testing.assert_close(before, torch.stack(losses).mean())


def test_gaussian_rejects_low_rank_noise(case):
    model, z, _, action, events = case
    apply_variant(model, "gaussian")
    with pytest.raises(ValueError, match="full latent shape"):
        predict_next(model, z, action, action_events=events, noise=torch.randn(2, 2, 1, 64))


def test_reapplying_variant_is_rejected(case):
    model = case[0]
    apply_variant(model, "regression")
    with pytest.raises(ValueError, match="fresh"):
        apply_variant(model, "gaussian")


@pytest.mark.parametrize("projection_rank", [64, 128])
def test_large_projection_preserves_rank16_noise_and_all_trainable_parameters(projection_rank):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    # A cycle has a nontrivial spectrum and enough roads for either candidate.
    graph = torch.zeros(130, 130)
    graph[torch.arange(130), (torch.arange(130) + 1) % 130] = 1
    prior = LowFrequencyGraphPrior(graph, rank=16)
    parameters = {name: value for name, value in prior.named_parameters()}
    z = torch.randn(2, 1, 130, 64)
    noise = torch.randn(2, 1, 16, 64)
    original_noise = prior.sample_noise(z, noise)
    original_basis = prior.basis.clone()
    rng = torch.random.get_rng_state()
    changed = ExperimentalPrior(prior, random_projection=False, seed=0, projection_rank=projection_rank)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert changed.rank == changed.noise_rank == 16
    assert changed.projection_rank == projection_rank
    assert changed.basis.shape == (130, projection_rank)
    assert torch.equal(changed.noise_basis, original_basis)
    assert torch.equal(changed.sample_noise(z, noise), original_noise)
    for name, parameter in changed.prediction.named_parameters():
        assert parameter is parameters[name]
    assert sum(p.numel() for p in changed.parameters()) == sum(p.numel() for p in parameters.values())
    with pytest.raises(ValueError, match="coefficient shape"):
        changed.sample_noise(z, torch.randn(2, 1, projection_rank, 64))
    torch.testing.assert_close(changed.basis.T @ changed.basis, torch.eye(projection_rank), atol=1e-5, rtol=1e-5)
    # Projection is differentiable and does not silently use the source-noise rank.
    signal = changed.basis[:, -1].reshape(1, 1, 130, 1).expand(1, 1, 130, 64).clone().requires_grad_()
    projection = changed.project(signal)
    torch.testing.assert_close(projection, signal, atol=1e-5, rtol=1e-5)
    projection.square().sum().backward()
    assert torch.isfinite(signal.grad).all() and signal.grad.abs().sum() > 0
    restored = copy.deepcopy(changed)
    restored.noise_basis.zero_()
    restored.prediction.basis.zero_()
    stream = io.BytesIO()
    torch.save(changed.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert restored.rank == 16 and restored.projection_rank == projection_rank
    assert torch.equal(restored.sample_noise(z, noise), original_noise)
    assert torch.equal(restored.project(signal), changed.project(signal))


@pytest.mark.parametrize("variant", ["graph_prior_r64_fm", "graph_prior_r128_fm"])
def test_rank_candidate_metadata_and_parameter_budget(case, variant):
    model, inputs, targets, action, events = case
    original = variant_metadata(model)
    apply_variant(model, variant)
    metadata = variant_metadata(model)
    # Small-graph tests cap the requested rank at roads - 1, as production does.
    assert metadata['projection_rank'] == 2
    assert metadata['noise_rank'] == 1
    assert metadata['parameters'] == original['parameters']
    loss, _, _ = experimental_loss(model, z_inputs=inputs, z_targets=targets,
                                   action=action, action_events=events,
                                   noise=torch.randn(2, 2, 1, 64))
    loss.backward()
    gradient = model.dynamics.graph_prior.prediction.output.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("variant,reference", [
    ("isotropic_noise_graph_fm", "graph_fm"),
    ("isotropic_noise_no_prior_fm", "no_prior_fm"),
])
def test_isotropic_changes_only_source_covariance_not_coarse_prediction(case, variant, reference):
    model, z, _, action, events = case
    baseline = copy.deepcopy(model)
    apply_variant(baseline, reference)
    rng = torch.random.get_rng_state()
    apply_variant(model, variant)
    assert torch.equal(torch.random.get_rng_state(), rng)
    condition = baseline.dynamics.encode_condition(action, action_events=events)
    assert torch.equal(model.dynamics.graph_prior(z, condition), baseline.dynamics.graph_prior(z, condition))
    assert torch.equal(model.dynamics.graph_prior.project(z), baseline.dynamics.graph_prior.project(z))
    metadata, source_metadata = variant_metadata(model), variant_metadata(baseline)
    assert metadata['parameters'] == source_metadata['parameters']
    assert metadata['original_noise_rank'] == 1
    assert metadata['noise_rank'] == 3
    assert metadata['source_std'] == pytest.approx(.1 * (1 / 3) ** .5)
    assert metadata['expected_energy'] == pytest.approx(source_metadata['expected_energy'])
    assert metadata['coarse_projection_rank'] == (1 if reference == 'graph_fm' else 0)


def test_isotropic_covariance_is_diagonal_with_preserved_total_trace():
    # Probe every noise-coordinate basis vector; this computes covariance exactly,
    # rather than relying on a random Monte Carlo tolerance.
    torch.set_num_threads(1)
    prior = LowFrequencyGraphPrior(torch.ones(20, 20) - torch.eye(20), rank=16, noise_scale=.1)
    original = copy.deepcopy(prior)
    isotropic = ExperimentalPrior(prior, random_projection=False, seed=0,
                                 isotropic_noise=True, keep_prediction=True)
    coefficients = torch.eye(20).unsqueeze(-1).expand(20, 20, 64)
    transformed = isotropic.sample_noise(torch.zeros(20, 20, 64), coefficients)[..., 0]
    covariance = transformed.T @ transformed
    torch.testing.assert_close(covariance, torch.eye(20) * (.1 ** 2 * 16 / 20), rtol=1e-6, atol=1e-8)
    original_coefficients = torch.eye(16).unsqueeze(-1).expand(16, 16, 64)
    original_transformed = original.sample_noise(torch.zeros(16, 20, 64), original_coefficients)[..., 0]
    original_covariance = original_transformed.T @ original_transformed
    torch.testing.assert_close(covariance.trace(), original_covariance.trace(), rtol=1e-5, atol=1e-7)
    assert covariance.trace().item() * 64 == pytest.approx(.1 ** 2 * 16 * 64, rel=1e-6)


@pytest.mark.parametrize("variant", ["isotropic_noise_graph_fm", "isotropic_noise_no_prior_fm"])
def test_isotropic_transition_backpropagates_and_rejects_old_coefficient_shape(case, variant):
    model, z, _, action, events = case
    apply_variant(model, variant)
    with pytest.raises(ValueError, match='coefficient shape'):
        predict_next(model, z, action, action_events=events, noise=torch.randn(2, 2, 1, 64))
    result = predict_next(model, z.requires_grad_(), action, action_events=events,
                          noise=torch.randn(2, 2, 3, 64))
    result.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    gradient = model.dynamics.flow_head.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize('variant', ['regression', 'graph_regression', 'gaussian'])
def test_metadata_describes_actual_sampling_not_unused_prior_helper(case, variant):
    model = apply_variant(case[0], variant)
    metadata = variant_metadata(model)
    if variant == 'gaussian':
        assert metadata['expected_energy'] is None
        assert metadata['source_std'] is None
        assert metadata['source_std_coordinates'] == 'learned_diagonal_latent'
    else:
        assert metadata['noise_rank'] == 0
        assert metadata['source_std'] == 0.
        assert metadata['expected_energy'] == 0.


def test_isolated_prior_has_exact_forward_and_inference_parity(case):
    model, inputs, targets, action, events = case
    isolated = copy.deepcopy(model)
    before_keys = tuple(model.dynamics.state_dict())
    rng = torch.random.get_rng_state()
    apply_variant(isolated, 'isolated_prior_graph_fm', seed=42)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert tuple(isolated.dynamics.backbone.state_dict()) == before_keys
    assert variant_metadata(isolated)['parameters'] == variant_metadata(model)['parameters']
    noise = torch.randn(2, 2, 1, 64)
    kwargs = dict(z_inputs=inputs, z_targets=targets, action=action, action_events=events,
                  noise=noise, flow_time=torch.tensor([[.1, .3], [.6, .9]]))
    baseline = experimental_loss(model, **kwargs)
    actual = experimental_loss(isolated, **kwargs)
    for left, right in ((actual[0], baseline[0]), (actual[2], baseline[2])):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert actual[1].keys() == baseline[1].keys()
    for key in actual[1]:
        torch.testing.assert_close(actual[1][key], baseline[1][key], rtol=0, atol=0)
    torch.testing.assert_close(predict_next(isolated, inputs, action, noise=noise, action_events=events),
        predict_next(model, inputs, action, noise=noise, action_events=events), rtol=0, atol=0)
    assert variant_metadata(isolated)['gradient_policy']['condition_encoder'].startswith('shared_trainable')


def test_isolated_fm_only_blocks_prior_parameters_but_trains_condition_reference_and_field(case):
    model, inputs, targets, action, events = case
    baseline = copy.deepcopy(model)
    apply_variant(model, 'isolated_prior_graph_fm')
    kwargs = dict(z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        prior_weight=0., predict_endpoints=False, noise=torch.randn(2, 2, 1, 64),
        flow_time=torch.full((2, 2), .35))
    loss, _, _ = experimental_loss(model, **kwargs)
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in model.dynamics.graph_prior.parameters())
    for parameter in (model.dynamics.flow_head.weight, model.dynamics.reference_proj.weight,
                      model.dynamics.event_encoder.event_mlp[0].weight,
                      model.dynamics.action_encoder.edge_conditioning.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    # A nonzero control proves that the test would detect the original coupling.
    baseline_loss, _, _ = experimental_loss(baseline, **kwargs)
    baseline_loss.backward()
    assert baseline.dynamics.graph_prior.output.weight.grad.abs().sum() > 0


def test_isolated_combined_prior_gradients_equal_only_weighted_prior_supervision(case):
    model, inputs, targets, action, events = case
    apply_variant(model, 'isolated_prior_graph_fm')
    prior = model.dynamics.graph_prior
    parameters = tuple(prior.parameters())
    weight = .37
    condition = model.dynamics.encode_condition(action, action_events=events)
    increment = prior(inputs, condition)
    prior_mse = (increment.float() - prior.project(targets - inputs).detach().float()).square().mean()
    expected = torch.autograd.grad(weight * prior_mse, parameters)
    assert any(gradient.abs().sum() > 0 for gradient in expected)
    loss, metrics, _ = experimental_loss(model, z_inputs=inputs, z_targets=targets,
        action=action, action_events=events, prior_weight=weight, predict_endpoints=False,
        noise=torch.randn(2, 2, 1, 64), flow_time=torch.full((2, 2), .35))
    actual = torch.autograd.grad(loss, parameters)
    torch.testing.assert_close(metrics['prior_loss'], weight * prior_mse.detach())
    for observed, wanted in zip(actual, expected):
        torch.testing.assert_close(observed, wanted, rtol=1e-6, atol=1e-8)


def test_prior_start_isolation_preserves_direct_current_state_and_live_original_increment(case):
    model, inputs, _, action, events = case
    dynamics = model.dynamics
    current = inputs.clone().requires_grad_()
    condition = dynamics.encode_condition(action, action_events=events)
    noise = torch.randn(2, 2, 1, 64)
    start, increment, cached = dynamics.prior_start(current, condition, noise,
                                                   detach_prior_increment=True)
    assert cached.edge_tokens is condition.edge_tokens
    assert increment.requires_grad and cached.reference_tokens.requires_grad
    torch.testing.assert_close(torch.autograd.grad(start.sum(), current, retain_graph=True)[0],
                               torch.ones_like(current), rtol=0, atol=0)
    reference_gradient = torch.autograd.grad(cached.reference_tokens.sum(), current, retain_graph=True)[0]
    expected = dynamics.reference_proj.weight.sum(0).reshape(1, 1, 1, -1).expand_as(current)
    torch.testing.assert_close(reference_gradient, expected, rtol=1e-6, atol=1e-6)
    prior_gradient = torch.autograd.grad(increment.square().mean(), dynamics.graph_prior.output.weight)[0]
    assert prior_gradient.abs().sum() > 0
    ordinary = dynamics.prior_start(current, condition, noise)
    explicit_default = dynamics.prior_start(current, condition, noise, detach_prior_increment=False)
    torch.testing.assert_close(ordinary[0], explicit_default[0], rtol=0, atol=0)
    assert ordinary[1].requires_grad


def test_isolated_checkpoint_restores_gradient_policy_and_fm_isolation(case):
    model, inputs, targets, action, events = case
    restored = copy.deepcopy(model)
    apply_variant(model, 'isolated_prior_graph_fm', seed=42)
    apply_variant(restored, 'isolated_prior_graph_fm', seed=0)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert variant_metadata(restored) == variant_metadata(model)
    loss, _, _ = experimental_loss(restored, z_inputs=inputs, z_targets=targets,
        action=action, action_events=events, prior_weight=0., predict_endpoints=False)
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in restored.dynamics.graph_prior.parameters())
    assert restored.dynamics.reference_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('invalid', [1, 'true', None])
def test_prior_detachment_flag_requires_explicit_boolean(case, invalid):
    model, inputs, targets, action, events = case
    with pytest.raises(ValueError, match='detach_prior_increment must be a boolean'):
        independent_transition_fm_loss(model.dynamics, z_inputs=inputs, z_targets=targets,
            action=action, action_events=events, detach_prior_increment=invalid)


@pytest.mark.parametrize('variant', ['regression', 'graph_regression', 'gaussian'])
def test_point_predictors_adapt_new_condition_only_mainline_without_flow_head_conflict(variant):
    model = UrbanDreamerWorldModel(adjacency=torch.eye(3), static_features=torch.ones(3, 5),
        low_frequency_rank=2, prior_filter='soft', prior_filter_strength=.1,
        prior_action_conditioned=False, prior_usage='condition_only').eval()
    apply_variant(model, variant)
    dynamics = model.dynamics
    assert dynamics.prior_usage == 'source_reference'
    assert not hasattr(dynamics.backbone, 'condition_residual')
    assert not any('condition_rms_' in key for key in dynamics.state_dict())
    assert variant_metadata(model)['condition_adapter'] == 'non_fm_regressor_without_flow_condition_residual'
    z = torch.randn(1, 3, 64)
    action = torch.zeros(1, 3)
    output = model.transition(z, action)
    assert output.shape == z.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert dynamics.flow_head.weight.grad is not None
    samples, _, variance = model.latent_disagreement(z, action, n_samples=2)
    assert samples.shape == (2, 1, 3, 64) and torch.isfinite(variance).all()
