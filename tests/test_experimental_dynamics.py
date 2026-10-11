import copy
import io

import pytest
import torch
from torch import nn

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.experimental_dynamics import (
    VARIANTS, apply_variant, experimental_loss, predict_next, variant_metadata,
)
from urban_dreamer.models.world_model import UrbanDreamerWorldModel, independent_transition_fm_loss


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
    if variant in {"no_prior_fm", "regression", "gaussian"}:
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
    shape = (3, *z.shape) if variant == "gaussian" else (3, 2, 2, 1, 64)
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
