from types import SimpleNamespace
import copy
import pytest
import torch
from torch import nn
from tools.source_condition_controls import source_context,StandardGaussianPrior,source_cfm_loss
from tools.train_direct_context_fm import DirectResidual
from tools.train_partial_context_fm import (
    DirectVelocityHead, configure_trainable, gradient_proof, parameter_digest,
)
from urban_dreamer.models.world_model import ConditionEncoding
from urban_dreamer.models.world_model import Dynamics as ProductionDynamics


class Prior(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,z,condition):self.calls+=1;return torch.full_like(z,.3)
    def sample_noise(self,z,noise):return noise


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__();self.flow_head=nn.Linear(256,64);self.reference_proj=nn.Linear(64,256,bias=False);self.graph_prior=Prior()
    def prior_start(self,*args,**kwargs):raise AssertionError('original method should not execute inside control')
    def forward(self,action,*,packed_latents,flow_time,condition):return self.flow_head(torch.zeros(*packed_latents.shape[:-1],256))


@pytest.mark.parametrize('arm,mean,ref,hint,calls',[('A',1.,1.,0.,0),('B',1.,1.,.3,1),('G',0.,1.,0.,0),('C2',1.3,1.3,.3,1)])
def test_source_reference_and_background_presence(arm,mean,ref,hint,calls):
    d=Dynamics();head=DirectResidual();z=torch.ones(1,1,3,64);noise=torch.full_like(z,.2);old=d.flow_head;condition=ConditionEncoding(torch.zeros(1,1,3,256))
    with source_context(d,head,arm,dict(z=1.,b=1.,relative=1.)) as c:
        start,_,cached=d.prior_start(z,condition,noise)
        torch.testing.assert_close(start,torch.full_like(z,mean)+noise)
        torch.testing.assert_close(c['hint'],torch.full_like(z,hint));torch.testing.assert_close(c['mean'],torch.full_like(z,mean))
        torch.testing.assert_close(cached.reference_tokens,d.reference_proj(torch.full_like(z,ref)))
        assert 'target' not in c and d.graph_prior.calls==calls
    assert d.flow_head is old and 'prior_start' not in d.__dict__ and not d._forward_pre_hooks


def test_background_cache_is_ignored_when_ablation_omits_condition():
    d=Dynamics();z=torch.ones(1,1,3,64)
    with source_context(d,DirectResidual(),'A',dict(z=1.,b=1.,relative=1.),background=torch.full_like(z,999.)) as c:
        d.prior_start(z,ConditionEncoding(torch.zeros(1,1,3,256)),torch.zeros_like(z))
        assert not bool(c['hint'].any()) and d.graph_prior.calls==0


def test_standard_gaussian_is_dense_unit_noise_without_state_or_graph_projection():
    native=Prior();prior=StandardGaussianPrior(native,3);z=torch.full((2,1,3,64),10.);noise=torch.randn_like(z)
    assert prior.rank==3 and prior.noise_scale==1.
    torch.testing.assert_close(prior.sample_noise(z,noise),noise,rtol=0,atol=0)
    assert native.calls==0
    with pytest.raises(ValueError,match='one coefficient'):prior.sample_noise(z,torch.zeros(2,1,2,64))


class ProductionModel(nn.Module):
    """The real six-block model, using a small complete road graph for tests."""
    def __init__(self):
        super().__init__()
        self.dynamics=ProductionDynamics(adjacency=torch.ones(3,3)-torch.eye(3),
            static_features=torch.randn(3,5),low_frequency_rank=2)
        # Completed pretraining has a nonzero original velocity head. Keeping
        # the constructor's zero head would hide the reference/prefix gradient.
        with torch.no_grad():self.dynamics.flow_head.weight.normal_(std=.02)


@pytest.fixture(scope='module')
def production_template():
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(107)
    model=ProductionModel().eval()
    yield model
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize('arm',('A','G'))
def test_real_dynamics_full_cfm_target_and_gradients_without_background(production_template,monkeypatch,arm):
    model=copy.deepcopy(production_template)
    d=model.dynamics
    native=d.graph_prior
    if arm=='G':d.graph_prior=StandardGaussianPrior(native,d.n_edges)
    # Either direct or nested use of the coarse predictor is a hard failure.
    def forbidden_background(*args,**kwargs):
        raise AssertionError('A/G must never evaluate a background predictor')
    monkeypatch.setattr(native,'forward',forbidden_background)
    if arm=='G':monkeypatch.setattr(d.graph_prior,'forward',forbidden_background)
    residual=DirectResidual()
    # Once its zero output layer has taken a training step, the hidden layer
    # must also receive gradients. Exercise this nonzero learned-head regime.
    with torch.no_grad():residual.output.weight.normal_(std=.01)
    parameters=configure_trainable(model,residual)
    assert sum(p.numel() for p in parameters)==1_414_528
    batch=dict(z=torch.linspace(.2,1.1,2*3*64).reshape(2,1,3,64).requires_grad_(),
        target=torch.linspace(.7,1.5,2*3*64).reshape(2,1,3,64).requires_grad_(),
        b=torch.full((2,1,3,64),999.,requires_grad=True),
        condition=torch.linspace(-.3,.4,2*3*256).reshape(2,1,3,256).requires_grad_(),
        action=torch.zeros(2,1,3),time=torch.tensor([[.23],[.71]]))
    generator=torch.Generator().manual_seed(211)
    batch['noise']=torch.randn(2,1,d.graph_prior.rank,64,generator=generator)
    z,target=batch['z'].detach(),batch['target'].detach()
    if arm=='G':
        assert d.graph_prior.rank==3 and batch['noise'].shape==z.shape
        epsilon=batch['noise']
        source=epsilon  # Pure N(0,I): no Z addition, graph basis or .1 scaling.
        mean=torch.zeros_like(z)
        assert not torch.allclose(native.basis@(native.basis.T@epsilon),epsilon)
    else:
        assert d.graph_prior.rank==2
        epsilon=native.sample_noise(z,batch['noise'])
        source=z+epsilon
        mean=z
    scales=dict(z=1.7,b=2.3,relative=.8)
    before=parameter_digest(model,residual)
    original_head,original_method=d.flow_head,d.prior_start.__func__
    loss=source_cfm_loss(model,residual,arm,batch,scales)
    # Reconstruct the prediction and FULL velocity label independently of
    # source_context/source_cfm_loss, including generated-state direct inputs.
    t=batch['time'][:,:,None,None]
    xt=(1-t)*source+t*target
    context=dict(z=z,hint=torch.zeros_like(z),mean=mean,xt=xt,time=batch['time'])
    cached=ConditionEncoding(batch['condition'].detach(),d.reference_proj(z))
    d.flow_head=DirectVelocityHead(original_head,residual,context,scales)
    try:
        velocity=d(batch['action'],packed_latents=xt,flow_time=batch['time'],condition=cached).latent_velocity
    finally:d.flow_head=original_head
    expected=(velocity-(target-source).detach()).square().mean()
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    loss.backward()
    proof=gradient_proof(model,residual)
    assert proof['frozen_parameter_gradients_absent']
    assert all(norm>0 for norm in proof['nonzero_gradient_norms'].values())
    for parameter in residual.parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        assert float(torch.linalg.vector_norm(parameter.grad))>0
    last=d.transformer.layers[-1]
    for parameter in (last.space.attention.qkv.weight,last.mlp.input.weight,d.reference_proj.weight,original_head.weight):
        assert parameter.grad is not None and float(torch.linalg.vector_norm(parameter.grad))>0
    assert all(parameter.grad is None for layer in d.transformer.layers[:-1] for parameter in layer.parameters())
    assert all(batch[key].grad is None for key in ('z','target','b','condition'))
    assert d.flow_head is original_head and d.prior_start.__func__ is original_method
    assert 'prior_start' not in d.__dict__ and not d._forward_pre_hooks
    assert parameter_digest(model,residual)==before
