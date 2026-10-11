import pytest
import torch

from tools.benchmark_euler_resolution import relative_l2, compare_samples


def test_relative_l2_is_not_diluted_by_mean_signed_errors():
    reference = torch.tensor([1., -1.])
    assert relative_l2(reference * 1.1, reference) == pytest.approx(.1)
    assert relative_l2(torch.zeros(2), torch.zeros(2)) == 0
    assert relative_l2(torch.ones(2), torch.zeros(2)) is None


def test_numerical_gate_checks_latent_variance_and_slow_maximum():
    reference = torch.tensor([0., 1., 2.])[:, None, None, None].expand(3, 1, 2, 4).clone()
    decoded = torch.zeros(3, 1, 2, 4)
    speed_target = torch.zeros(1, 2)
    count = torch.ones(1, 2) * 5
    scores = compare_samples(reference, reference, decoded, decoded, speed_target, count, torch.ones(2), torch.ones(2))
    assert scores['numerical_gate']
    changed = decoded.clone()
    changed[0, 0, 0, 1] = .2
    scores = compare_samples(reference, reference, changed, decoded, speed_target, count, torch.ones(2), torch.ones(2))
    assert not scores['numerical_gate']
    assert scores['sample0_slow_speed_max_abs_vs_reference_mps'] == pytest.approx(.2)


def test_latent_accuracy_alone_cannot_pass_if_variance_changes():
    reference = torch.tensor([0., 1., -1.])[:, None, None, None].expand(3, 1, 2, 4).clone()
    samples = reference.clone()
    samples[1:] *= 2
    decoded = torch.zeros(3, 1, 2, 4)
    scores = compare_samples(samples, reference, decoded, decoded, torch.zeros(1, 2), torch.ones(1, 2), torch.ones(2), torch.ones(2))
    assert scores['sample0_latent_relative_l2'] == 0
    assert scores['global_variance_ratio'] == 4
    assert not scores['numerical_gate']


def test_two_samples_report_mean_and_unbiased_variance():
    reference = torch.tensor([0., 2.])[:, None, None, None].expand(2, 1, 2, 4).clone()
    decoded = torch.zeros(2, 1, 2, 4)
    scores = compare_samples(reference, reference, decoded, decoded, torch.zeros(1, 2),
        torch.ones(1, 2) * 5, torch.ones(2), torch.ones(2))
    assert scores['sample_mean_latent_relative_l2'] == 0
    assert scores['global_variance_ratio'] == 1
    assert scores['numerical_gate']
