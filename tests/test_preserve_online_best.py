import copy
import hashlib
import json
import os

import pytest
import torch

from tools.preserve_online_best import evaluation_candidate, inspect_and_preserve, publish_open_checkpoint, stamp


def fixture(episode=20, score=-3):
    rows=[dict(episode_index=i,demand_hash=f'demand{i}',episode_terminal=True,complete=False,
               metrics=dict(evaluation_demand_seed=314159,evaluation_selection_score=score)) for i in range(3)]
    progress=dict(online=[{}]*episode,before=rows,after=[],phase='online_training',
                  milestones=[dict(online_episodes=episode,phase='completed',evaluations=rows)])
    payload=dict(config_fingerprint='config',input_hash='input',phase='online_training',next_unit=episode,
                 training_state=dict(models={'actor':torch.tensor([float(episode)])},
                 metrics=dict(online_episodes_completed=episode,online_pilot_progress=progress)))
    protocol=dict(evaluation=dict(policy='argmax'),paired_evaluation_indices=[0,1,2],
                  evaluation_demand_seed=314159,runtime_config_fingerprint='config')
    return payload,protocol


def test_historical_evaluation_is_not_current_weights():
    payload,protocol=fixture()
    payload['training_state']['metrics']['online_pilot_progress']['milestones'][0]['online_episodes']=19
    with pytest.raises(ValueError,match='no complete evaluation'):
        evaluation_candidate(payload,protocol)


@pytest.mark.parametrize('defect',['phase','rows','terminal','seed','hash','sampling','cursor','weights'])
def test_incomplete_or_mismatched_rejected(defect):
    payload,protocol=fixture();p=payload['training_state']['metrics']['online_pilot_progress']
    if defect=='phase':p['milestones'][0]['phase']='evaluation'
    if defect=='rows':p['milestones'][0]['evaluations']=p['before'][:2]
    if defect=='terminal':p['milestones'][0]['evaluations'][0]['episode_terminal']=False
    if defect=='seed':p['milestones'][0]['evaluations'][0]['metrics']['evaluation_demand_seed']=1
    if defect=='hash':
        p['milestones'][0]['evaluations']=copy.deepcopy(p['before']);p['milestones'][0]['evaluations'][0]['demand_hash']='wrong'
    if defect=='sampling':protocol['evaluation']['policy']='sampling'
    if defect=='cursor':payload['next_unit']=21
    if defect=='weights':payload['training_state']['metrics']['online_episodes_completed']=21
    with pytest.raises(ValueError):evaluation_candidate(payload,protocol)


def test_best_replacement_and_metadata_match(tmp_path):
    source=tmp_path/'latest.pt';output=tmp_path/'retain'
    payload,protocol=fixture();torch.save(payload,source)
    assert inspect_and_preserve(source,protocol,output)['status']=='saved'
    first=(output/'current').readlink()
    payload,protocol=fixture(40,-4);torch.save(payload,source)
    assert inspect_and_preserve(source,protocol,output)['status']=='not_improved'
    assert (output/'current').readlink()==first
    payload,protocol=fixture(60,-2);torch.save(payload,source)
    assert inspect_and_preserve(source,protocol,output)['status']=='saved'
    meta=json.loads((output/'best_available_argmax.json').read_text())
    weight=output/'best_available_argmax.pt'
    assert meta['episode']==60
    assert meta['checkpoint_sha256']==hashlib.sha256(weight.read_bytes()).hexdigest()
    assert torch.load(weight,weights_only=True)['training_state']['models']['actor'].item()==60
    assert (output/'generations').joinpath(first.name,'checkpoint.pt').exists()


def test_copy_uses_open_inode_if_source_atomically_replaced(tmp_path):
    source=tmp_path/'latest.pt';other=tmp_path/'new.pt';out=tmp_path/'retain'
    payload,protocol=fixture();torch.save(payload,source);original=source.read_bytes()
    new,_=fixture(40);torch.save(new,other)
    with source.open('rb') as handle:
        identity=stamp(os.fstat(handle.fileno()));loaded=torch.load(handle,weights_only=True)
        os.replace(other,source)
        publish_open_checkpoint(handle,identity,out,evaluation_candidate(loaded,protocol))
    assert (out/'best_available_argmax.pt').read_bytes()==original
    assert source.read_bytes()!=original


def test_completed_latest_uses_durable_after(tmp_path):
    payload,protocol=fixture(2000);p=payload['training_state']['metrics']['online_pilot_progress']
    p['phase']='completed';p['after']=p['before'];p['milestones']=[]
    payload['phase']='completed';payload['next_unit']=0
    assert evaluation_candidate(payload,protocol)['episode']==2000
