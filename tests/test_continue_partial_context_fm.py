import copy

import pytest
import torch
from torch import nn

from tools.continue_partial_context_fm import plateau_step, restore_optimizer, assert_nested_equal
from tools.run_partial_context_fm import load_selected
from tools.train_partial_context_fm import configure_trainable, selected_state, DirectResidual
from urban_dreamer.models.world_model import Dynamics


def controller():
    return dict(best_ES=10.,best_epoch=4,anchor_ES=10.,anchor_epoch=4,patience=0,
                reductions=0,lrs=[.0001,.001],converged=False)


def test_two_lr_reductions_then_five_more_stalls_are_required():
    c=controller()
    for epoch in range(5,20):
        before=copy.deepcopy(c);c,improved,event=plateau_step(c,10.1,epoch)
        assert not improved
        if epoch in (9,14):assert event=='reduce_lr' and c['patience']==0
        if epoch<19:assert not c['converged']
        assert before!=c or c['patience']==before['patience']+1
    assert c['converged'] and event=='validation_plateau_at_min_lr'
    assert c['reductions']==2 and c['best_epoch']==4
    assert c['lrs']==pytest.approx([.000009,.00009])


def test_small_best_improvements_accumulate_without_premature_patience_reset():
    c=controller();c,improved,event=plateau_step(c,9.99,5)
    assert improved and c['best_epoch']==5 and c['patience']==1 and c['anchor_ES']==10.
    c,improved,event=plateau_step(c,9.975,6)
    assert improved and event=='meaningful_improvement' and c['patience']==0
    assert c['anchor_ES']==9.975 and c['anchor_epoch']==6


def test_controller_does_not_mutate_saved_history_and_rejects_nonfinite_score():
    c=controller();before=copy.deepcopy(c)
    plateau_step(c,9.,5);assert_nested_equal(c,before)
    with pytest.raises(ValueError,match='finite'):plateau_step(c,float('nan'),5)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dynamics=Dynamics(adjacency=torch.ones(3,3)-torch.eye(3),static_features=torch.randn(3,5),low_frequency_rank=2)


def test_adamw_restoration_matches_uninterrupted_next_step():
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(23);model,head=Model(),DirectResidual()
        fresh_model,fresh_head=copy.deepcopy(model),copy.deepcopy(head)
        params=configure_trainable(model,head)
        opt=torch.optim.AdamW([dict(params=[p for p in model.parameters() if p.requires_grad],lr=.0001),dict(params=list(head.parameters()),lr=.001)],weight_decay=.0001)
        for _ in range(4):
            for i,p in enumerate(params):p.grad=torch.full_like(p,.01*(i+1))
            opt.step()
        payload=dict(state=selected_state(model,head),optimizer=copy.deepcopy(opt.state_dict()))
        load_selected(fresh_model,fresh_head,payload['state'])
        resumed_params,resumed_opt=restore_optimizer(fresh_model,fresh_head,payload)
        assert_nested_equal(resumed_opt.state_dict(),payload['optimizer'])
        for i,(a,b) in enumerate(zip(params,resumed_params)):
            a.grad=torch.full_like(a,-.02*(i+1));b.grad=a.grad.clone()
        opt.step();resumed_opt.step()
        assert_nested_equal(selected_state(model,head),selected_state(fresh_model,fresh_head))
        assert_nested_equal(opt.state_dict(),resumed_opt.state_dict())
        assert all(float(s['step'])==5 and s['step'].device.type=='cpu' for s in resumed_opt.state.values())
    finally:torch.set_num_threads(old_threads)
