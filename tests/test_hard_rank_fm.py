import numpy as np
import pytest
import torch

from tools.evaluate_hard_rank_fm import complete_rank,HardMeanPrior,endpoint_target
from urban_dreamer.models.world_model import LowFrequencyGraphPrior,ConditionEncoding


def test_boundary_rank_keeps_complete_eigenspaces():
    values=np.array([0.,0.,.1,.2-1e-10,.2+1e-10,.3])
    assert complete_rank(values,1)==2
    assert complete_rank(values,4)==5
    assert complete_rank(values,6)==6


@pytest.mark.parametrize('values,rank',[([.2,.1],1),([0.,1.],0),([0.,1.],3),([0.,np.nan],1)])
def test_invalid_rank_or_spectrum_rejected(values,rank):
    with pytest.raises(ValueError):complete_rank(values,rank)


def test_prediction_rank_does_not_change_original_source_noise():
    torch.manual_seed(13);graph=torch.ones(5,5)-torch.eye(5)
    native=LowFrequencyGraphPrior(graph,rank=2,prior_filter='hard',prior_action_conditioned=True).eval()
    basis=np.linalg.qr(np.random.default_rng(8).normal(size=(5,5)))[0]
    z=torch.randn(4,1,5,64);noise=torch.randn(4,1,2,64)
    for rank in (1,3,5):
        prior=HardMeanPrior(native,basis[:,:rank])
        assert prior.projection_rank==rank and prior.rank==2
        torch.testing.assert_close(prior.basis,native.basis,rtol=0,atol=0)
        torch.testing.assert_close(prior.sample_noise(z,noise),native.sample_noise(z,noise),rtol=0,atol=0)
        torch.testing.assert_close(prior.project(z),torch.tensor(basis[:,:rank],dtype=z.dtype)@(torch.tensor(basis[:,:rank],dtype=z.dtype).T@z),rtol=0,atol=0)


def test_state_prior_head_has_correct_projection_and_no_action_dependence():
    torch.manual_seed(3);native=LowFrequencyGraphPrior(torch.ones(5,5)-torch.eye(5),rank=2,prior_filter='hard').eval()
    basis=np.eye(5)[:,:3];coef=np.zeros((65,64));coef[-1]=2
    prior=HardMeanPrior(native,basis,coefficients=coef,state_only=True)
    z=torch.randn(1,1,5,64);a=ConditionEncoding(torch.randn(1,1,5,256));b=ConditionEncoding(a.edge_tokens*37)
    output=prior(z,a)
    torch.testing.assert_close(output,prior(z,b),rtol=0,atol=0)
    assert bool((output[...,:3,:]==2).all()) and not bool(output[...,3:,:].any())


def test_production_label_helper_uses_supplied_hybrid_counts_and_following_speed():
    hybrid=torch.zeros(1,3,8);following=torch.zeros_like(hybrid)
    hybrid[...,2]=torch.tensor([.2,.4,.6]);hybrid[...,1]=torch.tensor([.1,.2,.3])
    following[...,1]=torch.tensor([.3,.4,.5])
    target=endpoint_target(hybrid,following,torch.tensor([5.,10.,20.]),torch.tensor([10.,20.,30.]))
    torch.testing.assert_close(target[...,0],torch.tensor([[1.,4.,12.]]))
    torch.testing.assert_close(target[...,1],torch.tensor([[2.,6.,12.]]))
