from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.train_direct_context_fm import DirectResidual,direct_inputs,direct_context,context_rows,selection_update,study_summary,score


def test_size_and_zero_initialization():
    head=DirectResidual()
    assert sum(p.numel() for p in head.parameters())==65856
    torch.testing.assert_close(head(torch.randn(7,449)),torch.zeros(7,64),rtol=0,atol=0)


def test_uncentered_shared_block_normalizers_preserve_zero_hint():
    h=torch.ones(2,256);z=torch.full((2,64),4.);hint=torch.zeros(2,64);r=torch.full((2,64),12.)
    x=direct_inputs(h,z,hint,r,torch.tensor([.1,.9]),dict(z=2.,b=7.,relative=3.))
    assert x.shape==(2,449)
    torch.testing.assert_close(x[:,256:320],torch.full((2,64),2.))
    assert not bool(x[:,320:384].any())
    torch.testing.assert_close(x[:,384:448],torch.full((2,64),4.))
    torch.testing.assert_close(x[:,-1],torch.tensor([.1,.9]))


class Dynamics(torch.nn.Module):
    def __init__(self,increment):
        super().__init__();self.flow_head=torch.nn.Linear(256,64).requires_grad_(False);self.increment=increment
    def prior_start(self,z,condition,noise=None,**kwargs):
        b=torch.full_like(z,self.increment)
        return z+b+noise,b,SimpleNamespace(reference=z+b)
    def forward(self,action,*,packed_latents,flow_time,condition=None):
        h=torch.zeros(*packed_latents.shape[:-1],256)
        return self.flow_head(h)


class CaptureResidual(torch.nn.Module):
    def forward(self,x):self.seen=x.clone();return torch.zeros(*x.shape[:-1],64)


@pytest.mark.parametrize('arm,increment,hint',[('zero_direct',0.,0.),('condition_only',0.,3.),('soft_direct',2.,2.)])
def test_generated_state_context_source_hint_and_exact_restoration(arm,increment,hint):
    d=Dynamics(increment);old=d.flow_head;z=torch.ones(2,1,3,64);noise=torch.full_like(z,.4);new=CaptureResidual()
    with direct_context(d,new,lambda z,c:torch.full_like(z,3.),arm,dict(z=1.,b=1.,relative=1.)) as context:
        source,b,c=d.prior_start(z,None,noise)
        torch.testing.assert_close(source,z+increment+noise)
        torch.testing.assert_close(c.reference,z+increment)
        generated=torch.full_like(z,7.)
        d(None,packed_latents=generated,flow_time=torch.full((2,1),.5),condition=c)
        torch.testing.assert_close(new.seen[...,256:320],z)
        torch.testing.assert_close(new.seen[...,320:384],torch.full_like(z,hint))
        torch.testing.assert_close(new.seen[...,384:448],generated-z-increment)
        torch.testing.assert_close(new.seen[...,-1],torch.full((2,1,3),.5))
    assert d.flow_head is old and 'prior_start' not in d.__dict__ and not d._forward_pre_hooks


def test_restores_existing_instance_method_and_head_on_exception():
    d=Dynamics(0);bound=d.prior_start;d.prior_start=bound;old=d.flow_head
    with pytest.raises(RuntimeError):
        with direct_context(d,DirectResidual(),lambda z,c:z,'zero_direct',dict(z=1.,b=1.,relative=1.)):raise RuntimeError('test')
    assert d.prior_start is bound and d.flow_head is old and not d._forward_pre_hooks


def test_only_residual_receives_gradient():
    d=Dynamics(0);old=d.flow_head;old_values={k:v.clone() for k,v in old.state_dict().items()};head=DirectResidual();z=torch.ones(1,1,2,64)
    with direct_context(d,head,lambda z,c:z,'zero_direct',dict(z=1.,b=1.,relative=1.)):
        d.prior_start(z,None,torch.zeros_like(z));d(None,packed_latents=z,flow_time=torch.zeros(1,1)).sum().backward()
    assert all(p.grad is None for p in old.parameters()) and any(p.grad is not None for p in head.parameters())
    for k,v in old.state_dict().items():torch.testing.assert_close(v,old_values[k],rtol=0,atol=0)


def test_parent_rows_map_same_z_hint_and_family_relative_without_reordering():
    z=np.arange(2*2*64,dtype=np.float32).reshape(2,2,64);b=z+1
    aux=dict(z=z,b=b,window=np.array([0,0,1,1,0,0,1,1]),time=np.arange(8,dtype=np.float32),roads=2,
        relative={'zero_mean':np.zeros((8,64),dtype=np.float32),'weak_soft':np.ones((8,64),dtype=np.float32)})
    index=np.array([1,6]);zz,bb,rr,tt=context_rows(aux,index,'condition_only','cpu')
    torch.testing.assert_close(zz,torch.tensor(z[[0,1],[1,0]]));torch.testing.assert_close(bb,torch.tensor(b[[0,1],[1,0]]))
    assert not bool(rr.any());torch.testing.assert_close(tt,torch.tensor([1.,6.]))
    _,zero,_,_=context_rows(aux,index,'zero_direct','cpu');assert not bool(zero.any())
    _,_,soft,_=context_rows(aux,index,'soft_direct','cpu');assert bool((soft==1).all())


def test_minor_best_update_does_not_reset_meaningful_patience():
    best=dict(ES=10.,epoch=0,reset_ES=10.,reset_epoch=0,patience=0)
    best,improved,reset=selection_update(best,9.99,20)
    assert improved and not reset and best['patience']==1 and best['reset_ES']==10.
    best,improved,reset=selection_update(best,9.975,25)
    assert improved and reset and best['patience']==0 and best['reset_epoch']==25
    best,improved,reset=selection_update(best,10.,30)
    assert not improved and not reset and best['patience']==1


def test_three_actual_arms_summary_needs_no_native():
    rows=[]
    for i in range(2):
        for arm in ('zero_direct','soft_direct','condition_only'):
            rows.append(score(np.ones((8,1,1,2)),np.ones((1,1,2)),np.ones(2),1,np.array([10.]),dict(arm=arm,condition_index=i,demand_group=str(i),episode=str(i))))
    summary=study_summary(rows,5)
    assert len(summary['arms'])==3 and all(a['metrics']['energy_score']['estimate']==0 for a in summary['arms'])
