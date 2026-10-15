import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import tools.evaluate_source_condition_controls as evaluation


def selection():
    return dict(status='ready_for_shared_test',budget_limited=False,last_epoch=24,
        total_updates=24*1536,controller=dict(converged=True,reductions=2,patience=5,
                                            best_epoch=11,best_ES=1.25))


def ready_files(study):
    for arm in ('A','G','B'):
        folder=study/arm;folder.mkdir()
        (folder/'validation_selection.json').write_text(json.dumps(selection()))


@pytest.mark.parametrize('arm',('A','G','B'))
def test_unready_arm_blocks_before_any_split_test_access_or_output(tmp_path,monkeypatch,arm):
    ready_files(tmp_path)
    path=tmp_path/arm/'validation_selection.json'
    data=selection();data['status']='budget_limited_test_closed';data['controller']['converged']=False
    path.write_text(json.dumps(data))
    actual_read=evaluation.read;opened=[]
    def guarded_read(file):
        file=str(file);opened.append(file)
        assert file.endswith('validation_selection.json'), 'Blind split/design/test was accessed before readiness'
        return actual_read(file)
    monkeypatch.setattr(evaluation,'read',guarded_read)
    # No other argument fields are supplied: readiness must be the first gate.
    with pytest.raises(RuntimeError,match='plateau'):
        evaluation.run(SimpleNamespace(study=tmp_path))
    assert opened and not (tmp_path/'shared_test').exists()


def test_ready_requires_all_three_converged_final_lr_controllers(tmp_path):
    ready_files(tmp_path)
    assert set(evaluation.require_ready(tmp_path))=={'A','G','B'}
    path=tmp_path/'G/validation_selection.json'
    data=selection();data['controller']['reductions']=1;path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='incomplete'):evaluation.require_ready(tmp_path)
    data=selection();data['budget_limited']=True;path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='budget limited'):evaluation.require_ready(tmp_path)


def test_selected_checkpoint_must_match_validation_choice_without_reselection(tmp_path):
    path=tmp_path/'best_A.pt';data=dict(epoch=11,validation_ES=1.25,state={'model':{},'residual':{}})
    torch.save(data,path)
    assert evaluation.load_selected_checkpoint(path,selection(),'A')['epoch']==11
    bad=copy.deepcopy(data);bad['epoch']=12;torch.save(bad,path)
    with pytest.raises(ValueError,match='selected epoch'):evaluation.load_selected_checkpoint(path,selection(),'A')
    bad=copy.deepcopy(data);bad['validation_ES']=1.24;torch.save(bad,path)
    with pytest.raises(ValueError,match='score differs'):evaluation.load_selected_checkpoint(path,selection(),'A')


def test_fixed_c2_hash_and_epoch_cannot_be_changed_after_other_arms_fit(tmp_path):
    path=tmp_path/'C2.pt';torch.save(dict(epoch=10,validation_ES=.9,state={}),path)
    reference=dict(path=str(path),sha256=evaluation.sha256(path))
    assert evaluation.load_fixed_reference(reference)['epoch']==10
    with pytest.raises(ValueError,match='hash differs'):
        evaluation.load_fixed_reference(dict(reference,sha256='0'*64))
    torch.save(dict(epoch=12,validation_ES=.8,state={}),path)
    with pytest.raises(ValueError,match='epoch10'):
        evaluation.load_fixed_reference(dict(reference,sha256=evaluation.sha256(path)))


@pytest.mark.parametrize('field,value',[('z',0.),('b',float('nan')),('relative',-1.)])
def test_context_scales_reject_invalid_training_normalizers(field,value):
    data=dict(fit_split='trainonly',centering=False,scales=dict(z=1.,b=2.,relative=3.))
    data['scales'][field]=value
    with pytest.raises(ValueError,match='positive finite'):evaluation.context_scales(data)


def test_context_scales_preserve_per_arm_values_and_reject_validation_fit():
    data=dict(fit_split='trainonly',centering=False,scales=dict(z=1.,b=2.,relative=3.))
    assert evaluation.context_scales(data)==dict(z=1.,b=2.,relative=3.)
    data['fit_split']='validation'
    with pytest.raises(ValueError,match='only on training'):evaluation.context_scales(data)


def test_install_each_prior_from_native_and_restore_module_even_on_failure(monkeypatch):
    model=nn.Module();model.dynamics=nn.Module();native=nn.Linear(2,2)
    model.dynamics.graph_prior=native;seen=[]
    def install(model,arm,basis,head):
        assert model.dynamics.graph_prior is native
        seen.append(arm);model.dynamics.graph_prior=nn.Linear(2,2)
    monkeypatch.setattr(evaluation,'install_prior',install)
    priors=evaluation.build_priors(model,'unused-basis','unused-head')
    assert seen==list(evaluation.ARMS) and set(priors)==set(evaluation.ARMS)
    assert model.dynamics.graph_prior is native
    assert all(not p.training and not any(w.requires_grad for w in p.parameters()) for p in priors.values())
    def failure(model,arm,basis,head):
        model.dynamics.graph_prior=nn.Linear(2,2)
        raise ValueError('deliberate failure')
    monkeypatch.setattr(evaluation,'install_prior',failure)
    with pytest.raises(ValueError,match='deliberate'):evaluation.build_priors(model,'x','y')
    assert model.dynamics.graph_prior is native
