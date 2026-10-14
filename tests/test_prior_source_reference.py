from types import MethodType

import pytest
import torch

from tools.evaluate_prior_source_reference import split_prior_start,endpoint_target
from urban_dreamer.models.world_model import LowFrequencyGraphPrior,ConditionEncoding


class TinyDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__();self.graph_prior=LowFrequencyGraphPrior(torch.ones(5,5)-torch.eye(5),rank=2,
            prior_filter='hard',prior_action_conditioned=True);self.reference_proj=torch.nn.Linear(64,256,bias=False)
        with torch.no_grad():self.graph_prior.output.bias.fill_(.3)

    def prior_start(self,z,condition,noise=None,*,detach_prior_increment=False):
        b=self.graph_prior(z,condition);coarse=z+b
        return coarse+self.graph_prior.sample_noise(z,noise),b,ConditionEncoding(condition.edge_tokens,self.reference_proj(coarse))


@pytest.mark.parametrize('source_on,reference_on',[(False,False),(True,False),(False,True),(True,True)])
def test_source_reference_toggles_preserve_noise_parameters_and_restore_method(source_on,reference_on):
    torch.manual_seed(7);d=TinyDynamics().eval();z=torch.randn(2,1,5,64);condition=ConditionEncoding(torch.randn(2,1,5,256));noise=torch.randn(2,1,2,64)
    before={k:v.clone() for k,v in d.state_dict().items()};b=d.graph_prior(z,condition);epsilon=d.graph_prior.sample_noise(z,noise)
    original=d.prior_start.__func__
    with split_prior_start(d,source_on=source_on,reference_on=reference_on):
        start,increment,cached=d.prior_start(z,condition,noise)
        torch.testing.assert_close(start,(z+b if source_on else z)+epsilon,rtol=0,atol=0)
        torch.testing.assert_close(increment,b,rtol=0,atol=0)
        torch.testing.assert_close(cached.reference_tokens,d.reference_proj(z+b if reference_on else z),rtol=0,atol=0)
        assert cached.edge_tokens is condition.edge_tokens
    assert 'prior_start' not in d.__dict__ and d.prior_start.__func__ is original
    for k,v in d.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_restores_instance_override_on_exception():
    d=TinyDynamics();old=MethodType(lambda self,*args:None,d);d.prior_start=old
    with pytest.raises(RuntimeError):
        with split_prior_start(d,source_on=False,reference_on=True):raise RuntimeError('test')
    assert d.__dict__['prior_start'] is old


def test_production_label_helper_available():
    f=torch.zeros(1,2,8);f[...,2]=.5;next=f.clone();next[...,1]=1
    torch.testing.assert_close(endpoint_target(f,next,torch.tensor([2.,4.]),torch.tensor([10.,20.])),torch.tensor([[[1.,5.],[2.,10.]]]))
