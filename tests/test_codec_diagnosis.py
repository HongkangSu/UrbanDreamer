"""Controls for the standalone fixed-data diagnosis, separate from production codec tests."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.diagnose_codec_training import (
    balanced_group_mean, codec_loss, reconstruction_metrics, select_frames,
    state_hash, train_arm, volume_objective,
)
from urban_dreamer.models.world_model import FlowProjector, decoder_bpr_supervision_loss, decoder_reconstruction_loss


def test_sampler_preserves_exact_observation_and_next_speed_endpoint():
    flow = np.zeros((5, 3, 8), dtype=np.float32)
    flow[..., 1] = 1.
    flow[1, 0, 2] = .5
    flow[2, 1, 2] = 1.
    flow[3, 0, 2] = .7
    flow[3, 0, 1] = .1
    flow[-1, :, 2] = 100.
    indexes, labels = select_frames(flow, np.array([1., 10., 100.]))
    assert indexes == [0, 2, 3]
    assert labels == ['empty', 'occupied', 'jam']
    assert all(index + 1 < len(flow) for index in indexes)
    with pytest.raises(ValueError, match='both empty and occupied'):
        select_frames(np.zeros((5, 3, 8)), np.ones(3))


def test_unit_change_and_foreground_weight_are_separate_controls():
    def terms(empty_roads):
        target = torch.zeros(1, empty_roads + 1, 8)
        prediction = torch.zeros(1, empty_roads + 1, 4)
        target[0, 0, 2] = .1
        prediction[0, 0, 2] = .2
        return volume_objective(prediction, target, torch.full((empty_roads + 1,), 10.))
    original, unit_only, balanced = terms(1)
    original_many, unit_many, balanced_many = terms(99)
    assert unit_only > original
    assert unit_many == pytest.approx(float(unit_only) / 50)
    assert original_many == pytest.approx(float(original) / 50)
    assert balanced_many == pytest.approx(float(balanced))
    assert balanced_group_mean(torch.tensor([1., 3.]), torch.tensor([True, True])) == 2.
    assert balanced_group_mean(torch.tensor([1., 3.]), torch.tensor([False, False])) == 2.


def fixture():
    torch.manual_seed(4)
    projector = FlowProjector(adjacency=torch.eye(4))
    with torch.no_grad():
        projector.decoder_speed.weight.zero_()
        projector.decoder_speed.bias.fill_(1.)
    flow = torch.zeros(3, 4, 8)
    flow[..., 1] = 1.
    flow[1:, :, 2] = torch.tensor([.2, .0, .1, .0])
    flow[..., 4:] = torch.tensor([1., .5, 0., 1.])
    following = flow.clone()
    following[1:, 0, 1] = .3
    physical = dict(capacities=torch.tensor([10., 20., 10., 20.]), lengths=torch.full((4,), 30.),
                    speed_limits=torch.full((4,), 10.))
    return projector, flow, following, physical


def test_original_codec_loss_exactly_matches_production_targets_and_bpr():
    projector, flow, following, physical = fixture()
    loss, _, decoded = codec_loss(projector, flow, following, arm='original', **physical)
    target = flow.clone()
    target[..., 1] = (flow[..., 1] + following[..., 1]) / 2
    recon, _ = decoder_reconstruction_loss(decoded_dyn=decoded, flow_targets=target)
    bpr, _ = decoder_bpr_supervision_loss(decoded_dyn=decoded, flow_targets=flow, **physical)
    assert torch.allclose(loss, recon + bpr)
    metrics = reconstruction_metrics(projector, flow, following, physical['capacities'], physical['speed_limits'])
    assert metrics['occupied_roads'] == 4
    assert metrics['empty_roads'] == 8
    assert metrics['occupied_speed_mae_mps'] == pytest.approx(1.75)


def test_codec_fit_changes_codec_without_mutating_start_or_unused_head():
    projector, flow, following, physical = fixture()
    initial = deepcopy(projector.state_dict())
    before_hash = state_hash(initial)
    result, fitted = train_arm(initial, torch.eye(4), flow, following,
                              arm='vehicle_units_all', steps=20, **physical)
    assert state_hash(initial) == before_hash
    assert result['initial_codec_hash'] == before_hash
    assert result['final_codec_hash'] != before_hash
    assert result['unused_total_travel_head_unchanged']
    assert result['checkpoints'][-1]['training_loss'] < result['checkpoints'][0]['training_loss']
    assert result['checkpoints'][0]['volume_head_gradient_norm'] > 0
    assert all(torch.isfinite(tensor).all() for tensor in fitted.values())
