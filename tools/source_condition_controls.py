"""Explicit source and background-condition controls for the frozen-codec study."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch import nn
from tools.train_partial_context_fm import DirectVelocityHead,ConditionEncoding


class StandardGaussianPrior(nn.Module):
    """Expose E independent Gaussian coefficients to the existing bank sampler."""
    def __init__(self,native,roads):
        super().__init__();self.native=native;self.rank=int(roads);self.noise_scale=1.;self.prior_action_conditioned=False
    def forward(self,z,condition):return torch.zeros_like(z)
    def sample_noise(self,z,noise=None):
        if noise is None:return torch.randn_like(z)
        if noise.shape!=z.shape:raise ValueError('standard Gaussian requires one coefficient per road and latent dimension')
        return noise.to(z)


@contextmanager
def source_context(dynamics,head,arm,scales,*,background=None):
    """A/G omit b completely; Z and action remain available to every model."""
    if arm not in ('A','B','G','C2'):raise ValueError('unknown source control')
    context={};old_head=dynamics.flow_head;had_start='prior_start' in dynamics.__dict__;old_start=dynamics.__dict__.get('prior_start')
    def start(this,z,condition,noise=None,*,detach_prior_increment=False):
        if arm in ('B','C2'):
            with torch.no_grad():b=this.graph_prior(z,condition) if background is None else background
            b=b.detach()
        else:b=torch.zeros_like(z)
        mean=torch.zeros_like(z) if arm=='G' else z+b if arm=='C2' else z
        reference=z+b if arm=='C2' else z
        context.update(z=z,hint=b,mean=mean)
        cached=ConditionEncoding(condition.edge_tokens,this.reference_proj(reference))
        return mean+this.graph_prior.sample_noise(z,noise),mean-z,cached
    def capture(module,args,kwargs):context.update(xt=kwargs['packed_latents'],time=kwargs['flow_time'])
    dynamics.prior_start=MethodType(start,dynamics);hook=dynamics.register_forward_pre_hook(capture,with_kwargs=True)
    dynamics.flow_head=DirectVelocityHead(old_head,head,context,scales)
    try:yield context
    finally:
        hook.remove();dynamics.flow_head=old_head
        if had_start:dynamics.prior_start=old_start
        else:del dynamics.__dict__['prior_start']


def source_cfm_loss(model,head,arm,batch,scales):
    z,target=batch['z'].detach(),batch['target'].detach();condition=ConditionEncoding(batch['condition'].detach())
    with source_context(model.dynamics,head,arm,scales,background=batch.get('b')):
        source,_,cached=model.dynamics.prior_start(z,condition,batch['noise'])
        t=batch['time'][:,:,None,None];xt=(1-t)*source+t*target
        velocity=model.dynamics(batch['action'],packed_latents=xt,flow_time=batch['time'],condition=cached).latent_velocity
        return (velocity.float()-(target-source).detach().float()).square().mean()
