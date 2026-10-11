import torch

from tools.probe_slow_codec import masks, weighted_speed_objective, freeze_encoder_for_decoder_probe, frozen_state_hash
from urban_dreamer.models.world_model import FlowProjector


def test_slow_masks_use_physical_counts_and_endpoint_mean():
    flow=torch.zeros(1,3,8)
    following=flow.clone()
    flow[...,2]=torch.tensor([.5,.5,.4])
    flow[...,1]=torch.tensor([.1,.1,.1])
    following[...,1]=torch.tensor([.1,.5,.1])
    occupied,slow=masks(flow,following,torch.ones(3)*10,torch.ones(3)*10)
    assert occupied.all()
    assert slow.tolist()==[[True,False,False]]


def test_weighted_speed_objective_empty_masks_are_finite_and_differentiable():
    flow=torch.zeros(1,3,8)
    decoded=torch.ones(1,3,4,requires_grad=True)
    loss=weighted_speed_objective(decoded,flow,flow,torch.ones(3),torch.ones(3))
    loss.backward()
    assert loss==0
    assert torch.isfinite(decoded.grad).all()


def test_weighted_speed_objective_reaches_slow_speed_without_other_channel_gradient():
    flow=torch.zeros(1,2,8)
    flow[...,2]=torch.tensor([5.,0.])
    decoded=torch.ones(1,2,4,requires_grad=True)
    weighted_speed_objective(decoded,flow,flow,torch.ones(2),torch.ones(2)).backward()
    assert decoded.grad[0,0,1]>0
    assert decoded.grad[0,1].abs().sum()==0
    assert decoded.grad[...,0].abs().sum()==0


def test_decoder_probe_freezes_encoder_and_global_head_exactly():
    codec=FlowProjector(adjacency=torch.eye(2))
    freeze_encoder_for_decoder_probe(codec)
    before=frozen_state_hash(codec)
    assert all(not parameter.requires_grad for name,parameter in codec.named_parameters()
               if name.startswith(('input_proj','gnn_layers','t_all_head')))
    optimizer=torch.optim.AdamW([p for p in codec.parameters() if p.requires_grad],lr=1e-3)
    flow=torch.rand(1,2,8)
    codec.decode(codec(flow)).sum().backward()
    optimizer.step()
    assert frozen_state_hash(codec)==before
