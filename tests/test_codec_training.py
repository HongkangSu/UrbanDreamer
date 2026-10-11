from types import SimpleNamespace

import pytest
import torch

from urban_dreamer.learning.codec_training import (
    codec_decoded_metrics, codec_objective, codec_train_step, make_codec_optimizer,
)
from urban_dreamer.models.world_model import FlowProjector, UrbanDreamerWorldModel


class ConstantCodec(torch.nn.Module):
    def __init__(self, decoded):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(decoded, dtype=torch.float32))

    def forward(self, flow):
        return flow

    def decode(self, latent):
        return self.value.expand(*latent.shape[:-1], 4)


def network(capacities=(10., 100.)):
    return SimpleNamespace(capacities=capacities, speed_limits_mps=(10., 20.))


def test_volume_labels_use_input_frame_not_future_cohort():
    codec = ConstantCodec([0., .8, .1, 0.])
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = .8
    flow[..., 2] = torch.tensor([.2, .03])
    future = flow.clone()
    loss, metrics = codec_objective(codec, flow, future, network())
    future[..., 2] = 500.
    changed, changed_metrics = codec_objective(codec, flow, future, network())
    assert torch.equal(loss, changed)
    assert metrics['codec_target_count_total'] == 5.
    assert torch.equal(metrics['codec_count_loss'], changed_metrics['codec_count_loss'])


def test_count_loss_vehicle_units_and_global_mass_match_explicit_formula():
    codec = ConstantCodec([0., 1., .1, 0.])
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = 1.
    flow[..., 2] = torch.tensor([.2, 0.])
    loss, metrics = codec_objective(codec, flow, flow, network())
    square = (torch.log1p(torch.tensor([1., 10.])) - torch.log1p(torch.tensor([2., 0.]))).square()
    expected_count = .8 * square.mean() + .2 * square[0]
    expected_mass = torch.asinh(torch.tensor((11. - 2.) / (2. + 1.))).square()
    assert torch.allclose(metrics['codec_count_loss'], expected_count)
    assert torch.allclose(metrics['codec_mass_loss'], expected_mass)
    assert torch.allclose(loss, expected_count + .1 * expected_mass)
    assert metrics['codec_count_mae_occupied'] == 1.
    assert metrics['codec_count_mae_empty'] == 10.
    assert metrics['codec_total_relative_error'] == 4.5


def test_empty_and_zero_population_loss_and_gradients_remain_finite():
    codec = ConstantCodec([0., 0., 1e-3, 0.])
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = 1.
    loss, metrics = codec_objective(codec, flow, flow, network())
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(codec.value.grad).all()
    assert codec.value.grad[1].abs() <= 1.
    assert metrics['codec_count_occupied_roads'] == 0
    assert metrics['codec_count_mae_occupied'] == 0
    assert metrics['codec_total_relative_error'] > 0


def test_exact_counts_have_zero_vehicle_loss_with_heterogeneous_capacity():
    flow = torch.zeros(2, 2, 8)
    flow[..., 1] = .5
    flow[..., 2] = torch.tensor([.2, .03])
    decoded = flow[..., :4].clone()
    metrics = codec_decoded_metrics(decoded, flow, flow, network())
    assert metrics['codec_target_count_total'] == 5.
    assert metrics['codec_count_mae_all'] == 0
    assert metrics['codec_total_relative_error'] == 0
    assert metrics['codec_speed_mae_mps'] == 0


def test_positive_speed_decoder_is_actual_interface_and_legacy_is_unchanged():
    graph = torch.eye(2)
    legacy = FlowProjector(adjacency=graph)
    positive = FlowProjector(adjacency=graph, positive_speed=True)
    latent = torch.zeros(1, 2, 64)
    assert torch.equal(legacy.decode(latent)[..., 1], torch.zeros(1, 2))
    assert torch.allclose(positive.decode(latent)[..., 1], torch.ones(1, 2))
    positive.decoder_speed.bias.data.fill_(-.1)
    speed = positive.decode(latent)[..., 1]
    assert (speed > 0).all()
    speed.sum().backward()
    assert torch.isfinite(positive.decoder_speed.bias.grad).all()
    model = UrbanDreamerWorldModel(adjacency=graph, static_features=torch.ones(2, 4), codec_speed_positive=True)
    assert model.flow_projector.positive_speed


def test_codec_optimizer_and_step_do_not_modify_dynamics():
    torch.set_num_threads(1)
    model = UrbanDreamerWorldModel(adjacency=torch.eye(2), static_features=torch.ones(2, 4),
                                   event_protocol=True, codec_speed_positive=True)
    before = {name: value.clone() for name, value in model.dynamics.state_dict().items()}
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = 1.
    flow[..., 2] = torch.tensor([.1, .01])
    optimizer = make_codec_optimizer(model)
    metrics = codec_train_step(model, flow, flow, network(), None, optimizer)
    assert torch.isfinite(metrics['codec_gradient_norm'])
    assert model.flow_projector.decoder_volume.weight.grad.abs().sum() > 0
    assert all(torch.equal(before[name], value) for name, value in model.dynamics.state_dict().items())
    assert all(parameter.grad is None for parameter in model.dynamics.parameters())


def test_future_speed_label_has_no_gradient_and_invalid_weights_are_rejected():
    codec = ConstantCodec([0., .8, .1, 0.])
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = .8
    future = torch.ones_like(flow, requires_grad=True)
    loss, _ = codec_objective(codec, flow, future, network())
    loss.backward()
    assert future.grad is None
    with pytest.raises(ValueError, match='foreground_fraction'):
        codec_objective(codec, flow, future, network(), SimpleNamespace(codec_foreground_fraction=.5 + 1))
