import numpy as np
import pytest
import torch

from tools.adapt_fm_output_head import anchored_ridge,capture_final_features,head_coefficients,install_head,endpoint_target


def test_anchored_ridge_exact_identity_when_original_head_is_already_correct():
    rng=np.random.default_rng(6);x=rng.normal(size=(100,7));w=rng.normal(size=(7,3));y=x@w
    fitted,proof=anchored_ridge(x.T@x,x.T@y,float(np.square(y).sum()),w)
    np.testing.assert_allclose(fitted,w,rtol=1e-13,atol=1e-13)
    assert proof['normal_equation_relative_residual']<1e-12


def test_anchored_ridge_matches_normal_equation_and_lowers_known_linear_loss():
    rng=np.random.default_rng(9);x=rng.normal(size=(150,9));truth=rng.normal(size=(9,4));anchor=np.zeros((9,4));y=x@truth
    g=x.T@x;c=x.T@y;w,proof=anchored_ridge(g,c,float(np.square(y).sum()),anchor,relative=.02)
    np.testing.assert_allclose((g+np.eye(9)*proof['ridge'])@w,c+proof['ridge']*anchor,rtol=1e-12,atol=1e-12)
    assert proof['adapted_train_sse']+proof['anchor_penalty']<proof['original_train_sse']


def test_temporary_hook_captures_actual_head_input_without_body_updates_and_is_removed():
    torch.manual_seed(2);body=torch.nn.Linear(5,8);head=torch.nn.Linear(8,3)
    before={k:v.clone() for k,v in body.state_dict().items()};x=torch.randn(4,5)
    features,result=capture_final_features(head,lambda:head(body(x)))
    torch.testing.assert_close(features,body(x))
    torch.testing.assert_close(result,head(features))
    assert not head._forward_pre_hooks
    for k,v in body.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_hook_is_removed_on_exception_and_original_head_roundtrips_exactly():
    head=torch.nn.Linear(4,3);original=head_coefficients(head)
    with pytest.raises(RuntimeError):capture_final_features(head,lambda:(_ for _ in ()).throw(RuntimeError('test')))
    assert not head._forward_pre_hooks
    install_head(head,original+1);install_head(head,original)
    np.testing.assert_array_equal(head_coefficients(head),original)


def test_production_endpoint_label_helper_is_available():
    flow=torch.zeros(1,2,8);flow[...,2]=.5;following=flow.clone();following[...,1]=1
    out=endpoint_target(flow,following,torch.tensor([2.,4.]),torch.tensor([10.,20.]))
    torch.testing.assert_close(out,torch.tensor([[[1.,5.],[2.,10.]]]))
