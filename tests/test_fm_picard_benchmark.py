"""Units and reference semantics for the private Picard inference diagnostic."""
import pytest
import torch
from tools.benchmark_fm_picard import physical_errors, tensor_comparison


def test_physical_errors_use_vehicle_counts_and_physical_speed():
    decoded=torch.zeros(1,2,8)
    decoded[...,2]=torch.tensor([[.5,.1]])
    decoded[...,1]=torch.tensor([[.3,.6]])
    result=physical_errors(decoded,torch.tensor([[6.,0.]]),torch.tensor([[1.,5.]]),
        torch.tensor([10.,10.]),torch.tensor([10.,10.]))
    assert result['count_mae']==pytest.approx(1)
    assert result['occupied_count_mae']==pytest.approx(1)
    assert result['speed_mae']==pytest.approx(1.5)
    assert result['slow_queue_speed_mae']==pytest.approx(2)
    assert result['slow_queue_road_count']==1


def test_empty_queue_reports_no_slow_error_not_zero_error():
    result=physical_errors(torch.zeros(1,2,8),torch.zeros(1,2),torch.zeros(1,2),
        torch.ones(2),torch.ones(2))
    assert result['occupied_speed_mae'] is None
    assert result['slow_queue_speed_mae'] is None
    assert result['slow_queue_road_count']==0


def test_tensor_reference_difference_and_nonfinite_flag():
    assert tensor_comparison(torch.tensor([2.,4.]),torch.tensor([1.,2.]))['relative_l2']==pytest.approx(1)
    assert tensor_comparison(torch.zeros(2),torch.zeros(2))['relative_l2']==0
    assert not tensor_comparison(torch.tensor([float('nan')]),torch.ones(1))['finite']
