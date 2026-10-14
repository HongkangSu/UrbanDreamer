import numpy as np
import pytest
import torch

from tools.train_nonlinear_fm_residual import NonlinearResidual,ResidualVelocityHead,residual_head,stratified_draws,endpoint_target,study_summary,score,selection_improves


def test_residual_size_and_exact_noop_initialization():
    torch.manual_seed(4);old=torch.nn.Linear(256,64);new=NonlinearResidual();x=torch.randn(7,256)
    assert sum(p.numel() for p in new.parameters())==41152
    torch.testing.assert_close(ResidualVelocityHead(old,new)(x),old(x),rtol=0,atol=0)


def test_backprop_only_updates_new_residual_and_context_restores_original():
    class D(torch.nn.Module):
        def __init__(self):super().__init__();self.flow_head=torch.nn.Linear(256,64).requires_grad_(False)
    d=D();original=d.flow_head;before={k:v.clone() for k,v in original.state_dict().items()};new=NonlinearResidual()
    with residual_head(d,new):(d.flow_head(torch.randn(8,256))-1).square().mean().backward()
    assert d.flow_head is original
    assert all(p.grad is None for p in original.parameters())
    assert any(p.grad is not None for p in new.parameters())
    for k,v in original.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_restores_original_head_even_on_exception():
    d=torch.nn.Module();d.flow_head=torch.nn.Linear(256,64);original=d.flow_head
    with pytest.raises(RuntimeError):
        with residual_head(d,NonlinearResidual()):raise RuntimeError('test')
    assert d.flow_head is original


def test_stratified_times_and_noises_are_paired_reproducible_and_stage_distinct():
    entry=dict(cache_path='frozen-entry');starts=[1,7,13]
    for q in range(4):
        t,n,r=stratified_draws(41,'train',entry,starts,q,16);t2,n2,r2=stratified_draws(41,'train',entry,starts,q,16)
        assert bool((t>=q/4).all()) and bool((t<(q+1)/4).all())
        torch.testing.assert_close(t,t2,rtol=0,atol=0);torch.testing.assert_close(n,n2,rtol=0,atol=0);assert r==r2
        tv,nv,_=stratified_draws(41,'validation',entry,starts,q,16)
        assert not torch.equal(n,nv) and not torch.equal(t,tv)


def test_label_helper_available_for_hybrid_count_and_following_speed():
    f=torch.zeros(1,2,8);f[...,2]=.5;next=f.clone();next[...,1]=1
    torch.testing.assert_close(endpoint_target(f,next,torch.tensor([2.,4.]),torch.tensor([10.,20.])),torch.tensor([[[1.,5.],[2.,10.]]]))


def test_epoch_zero_candidate_retained_if_training_worsens_validation():
    baseline=dict(ES=3.,epoch=0,patience=0)
    assert not selection_improves(baseline,4.)
    assert not selection_improves(baseline,3.)
    assert selection_improves(baseline,2.9)
    assert selection_improves(None,3.)


@pytest.mark.parametrize('arms',[('zero_mean','weak_soft'),('old_zero','old_soft','adapter_zero','adapter_soft')])
def test_actual_study_arms_summary_needs_no_native(arms):
    rows=[]
    for state,error in enumerate((3.,4.)):
        for arm in arms:
            samples=np.full((8,1,1,2),1+error)
            rows.append(score(samples,np.ones((1,1,2)),np.ones(2),10,np.array([20.]),
                dict(condition_index=state,arm=arm,demand_group='group',episode='episode')))
    result=study_summary(rows,91)
    assert {r['arm'] for r in result['arms']}==set(arms)
    assert all(r['metrics']['count_rmse_all']['estimate']==pytest.approx(np.sqrt(12.5)) for r in result['arms'])
