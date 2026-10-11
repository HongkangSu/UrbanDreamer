from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.diagnose_codec_gradients import analyze_batch, gradient_cosine, gradient_statistics
from tests.test_world_model_evaluation import make_batch, make_trainer
from urban_dreamer.models.world_model import UrbanDreamerWorldModel, recursive_fm_loss


def test_statistics_separate_unused_zero_and_nonfinite_gradients():
    first, unused = torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(2))
    stats = gradient_statistics((torch.tensor([3., 4., 0., float('nan')]), None), (first, unused))
    assert stats['l2_norm'] == 5.
    assert stats['unused_elements'] == 2
    assert stats['zero_fraction'] == pytest.approx(1 / 6)
    assert stats['nonfinite_fraction'] == pytest.approx(1 / 6)
    assert gradient_cosine((torch.tensor([1., 2.]),), (torch.tensor([-1., -2.]),))['cosine'] == pytest.approx(-1.)
    assert not gradient_cosine((None,), (torch.tensor([1.]),))['defined']


def test_exact_channel_decomposition_labels_softplus_and_no_parameter_mutation():
    torch.manual_seed(42)
    model = UrbanDreamerWorldModel(adjacency=torch.tensor([[0., 1.], [1., 0.]]),
        static_features=torch.ones(2, 4), low_frequency_rank=1, event_protocol=True)
    trainer = make_trainer(model)
    trainer.network.lengths_m = np.array([30., 40.])
    trainer.config.method.prior_loss_weight = .1
    batch = make_batch(frames=2)
    batch.flow[..., 1] /= 8.
    batch.flow[..., 2] /= 5.
    model.train()
    model.flow_projector.decoder_shared.eval()
    before = deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    report = analyze_batch(model, batch, trainer.network, trainer.config.method, seed=37)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(37)
        z = model.encode(batch.flow.reshape(-1, 2, 8)).reshape(1, 2, 2, 64)
        reference_fm, reference_metrics, _ = recursive_fm_loss(model.dynamics,
            z_sequence=z, action=batch.action, action_events=batch.action_events,
            prior_weight=.1, steps=2, solver='picard', picard_iterations=3)
    assert report['components']['fm_total']['loss'] == pytest.approx(float(reference_fm.detach()), rel=1e-6)
    assert report['components']['fm_prior']['loss'] == pytest.approx(float(reference_metrics['prior_loss']), rel=1e-6)
    assert report['labels']['reconstruction_channel_sum_error'] < 1e-6
    assert report['labels']['reconstruction_volume_target_input_max_abs'] == 0.
    assert report['labels']['physical_counts_integer_mae'] < 1e-6
    assert report['labels']['t_all_completed_sequences'] == 0
    assert report['labels']['speed_forward_mean_target']
    assert report['labels']['last_speed_label_excluded']
    assert report['components']['recon_volume']['blocks']['decoder_volume']['l2_norm'] > 0.
    assert report['components']['recon_speed']['blocks']['decoder_volume']['l2_norm'] == 0.
    assert report['components']['fm_total']['blocks']['decoder_volume']['unused_elements'] > 0
    assert report['components']['recon_volume']['blocks']['dynamics']['unused_elements'] > 0
    assert 0. < report['hypothetical_clip10_factor'] <= 1.
    assert 0. <= report['raw_decoder_probe']['softplus_derivative']['min'] <= 1.
    assert all(component['global_gradient']['nonfinite_fraction'] == 0. for component in report['components'].values())
    assert model.training and not model.flow_projector.decoder_shared.training
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], atol=0., rtol=0.)
