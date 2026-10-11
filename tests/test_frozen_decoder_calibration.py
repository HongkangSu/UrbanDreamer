from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.calibrate_frozen_decoder import (calibration_pairs,completed_epochs,sampled_raw_frames,gate,
    restore_calibration,save_calibration,verify_partition_separation,ValidationAccumulator)
from tools.probe_slow_codec import freeze_encoder_for_decoder_probe,frozen_state_hash
from tools.train_codec_ready import FrameSampler,SparseGraphCodec
from urban_dreamer.learning.codec_training import codec_objective,make_decoder_optimizer
from urban_dreamer.workflow.config import CodecConfig
from test_pretrain_cache import make_cache
from test_offline_data import setup


def trainable_codec():
    model=SparseGraphCodec(adjacency=torch.eye(2))
    freeze_encoder_for_decoder_probe(model)
    return model


def test_only_decoder_changes_and_encoder_has_no_grad():
    model=trainable_codec();before=frozen_state_hash(model)
    speed_before=model.decoder_speed.weight.detach().clone()
    optimizer=make_decoder_optimizer(model,lr=1e-3)
    flow=torch.rand(3,2,8)
    network=SimpleNamespace(capacities=[10,10],speed_limits_mps=[10,10])
    loss,_=codec_objective(model,flow,flow,network,codec=replace(CodecConfig(),occupied_speed_weight=1.,slow_speed_weight=1.),decoder_only=True)
    loss.backward();optimizer.step()
    assert frozen_state_hash(model)==before
    assert not torch.equal(speed_before,model.decoder_speed.weight)
    assert all(p.grad is None for p in model.input_proj.parameters())
    assert all(p.grad is None for p in model.t_all_head.parameters())


def test_resume_preserves_next_samples_optimizer_and_frozen_parameters(tmp_path):
    model=trainable_codec();optimizer=make_decoder_optimizer(model)
    sampler=FrameSampler([dict(frames=5),dict(frames=7)],3,12)
    sampler.next()
    contract=dict(frozen_parameters_sha256=frozen_state_hash(model),manifest='identity')
    path=tmp_path/'state.pt'
    save_calibration(path,model,optimizer,sampler,contract,1,[],[],'training')
    copy=trainable_codec();copy.load_state_dict(model.state_dict())
    restored=FrameSampler(sampler.entries,3,99)
    restored_optimizer=make_decoder_optimizer(copy)
    payload=torch.load(path,weights_only=False)
    assert restore_calibration(payload,contract,copy,restored_optimizer,restored)==(1,[],[])
    left=sampler.next();right=restored.next()
    assert left[0]==right[0] and np.array_equal(left[1],right[1])
    with pytest.raises(ValueError,match='contract differs'):
        restore_calibration(payload,dict(contract,manifest='changed'),copy,restored_optimizer,restored)
    assert 'entries' not in payload['sampler']


@pytest.mark.parametrize('field',['path','demand_group','trajectory_hash'])
def test_split_leakage_is_rejected(field):
    train=dict(path='a',demand_group='g1',trajectory_hash='t1')
    validation=dict(path='b',demand_group='g2',trajectory_hash='t2')
    validation[field]=train[field]
    with pytest.raises(ValueError,match=field):verify_partition_separation([train],[validation])


def test_raw_all_starts_and_hybrid_excludes_only_current_births(setup,tmp_path,monkeypatch):
    dataset,_=make_cache(setup,tmp_path,monkeypatch,indexes=(0,),windows=3)
    arrays=dataset._episode(0)
    raw,hybrid=calibration_pairs(arrays,[0,1,2],dataset.topology.network.capacities)
    assert len(raw[0])==3 and len(hybrid[0])==2
    assert torch.equal(raw[1],torch.from_numpy(arrays['flow'][1:]))
    assert torch.equal(hybrid[1],torch.from_numpy(arrays['flow'][2:]))
    assert hybrid[0][0,:,2].sum()==0  # initially-empty prior excludes first births
    assert hybrid[0][1,:,2].sum()==1  # earlier births are eligible in next interval


def test_partial_final_batch_marks_epoch_complete_without_extra_next():
    sampler=FrameSampler([dict(frames=5)],3,12)
    sampler.next();assert completed_epochs(sampler)==0
    assert sampled_raw_frames(sampler)==3
    assert len(sampler.next()[1])==2
    assert completed_epochs(sampler)==1
    assert sampled_raw_frames(sampler)==5


def test_gate_requires_slow_local_occupied_and_count_metrics():
    config=replace(CodecConfig(),slow_speed_mae_mps_max=.5)
    values=dict(codec_count_mae_occupied=.1,codec_count_mae_all=.01,codec_count_mae_empty=.01,
        codec_total_relative_error=.01,codec_speed_mae_mps=.4,codec_speed_mae_occupied_mps=.4,
        codec_speed_mae_slow_mps=.2,localized_slow_road_mae_max_mps=.3,
        local_context_slow_mae_mps=.2,local_context_slow_road_max_mps=.3)
    assert gate(values,config,1.)['passed']
    assert not gate(dict(values,codec_speed_mae_slow_mps=.8),config,1.)['passed']
    assert not gate(dict(values,localized_slow_road_mae_max_mps=3.),config,1.)['passed']
    assert not gate(dict(values,codec_count_mae_occupied=.8),config,1.)['passed']
    assert not gate(dict(values,codec_speed_mae_mps=2.),config,1.)['passed']
    assert not gate(dict(values,local_context_slow_mae_mps=3.),config,1.)['passed']
    assert not gate(dict(values,local_context_slow_road_max_mps=None),config,1.)['passed']


def test_validation_mask_metrics_weight_exact_cells_and_local_road_groups():
    acc=ValidationAccumulator(2)
    flow=torch.zeros(2,2,8);flow[:,0,2]=5.;flow[:,0,1]=.1
    decoded=flow[...,:4].clone();decoded[:,0,1]=.2
    network=SimpleNamespace(capacities=[1,1],speed_limits_mps=[10,10])
    from urban_dreamer.learning.codec_training import codec_decoded_metrics
    cfg=CodecConfig()
    metrics=codec_decoded_metrics(decoded,flow,flow,network,cfg)
    acc.add(metrics,decoded,flow,flow,torch.ones(2),torch.ones(2)*10,cfg)
    result=acc.result(2)
    assert result['codec_speed_mae_slow_mps']==pytest.approx(1.)
    assert result['localized_slow_road_mae_max_mps']==pytest.approx(1.)
    assert result['localized_slow_eligible_roads']==1


def test_ac_crossing_endpoint_still_slow_but_not_persistent_and_count_rounding_is_allowed():
    acc=ValidationAccumulator(1)
    flow=torch.zeros(1,1,8);flow[...,2]=5.-5e-6;flow[...,1]=.1306
    following=flow.clone();following[...,1]=.2093
    decoded=flow[...,:4].clone();decoded[...,1]=.5296
    network=SimpleNamespace(capacities=[1],speed_limits_mps=[10])
    from urban_dreamer.learning.codec_training import codec_decoded_metrics
    cfg=CodecConfig()
    metrics=codec_decoded_metrics(decoded,flow,following,network,cfg)
    acc.add(metrics,decoded,flow,following,torch.ones(1),torch.ones(1)*10,cfg)
    result=acc.result(1)
    assert result['codec_slow_roads']==1
    assert result['codec_persistent_slow_roads']==0
    assert result['codec_speed_mae_persistent_slow_mps'] is None
    assert result['codec_speed_mae_slow_mps']==pytest.approx(3.5965,abs=1e-4)
    assert result['localized_slow_road_mae_max_mps']==pytest.approx(3.5965,abs=1e-4)


def test_broad_gridlock_cannot_dilute_isolated_bad_slow_prediction():
    from urban_dreamer.learning.codec_training import codec_decoded_metrics
    cfg=CodecConfig();acc=ValidationAccumulator(6)
    network=SimpleNamespace(capacities=[1]*6,speed_limits_mps=[10]*6)
    capacity=torch.ones(6);limits=torch.ones(6)*10
    broad=torch.zeros(100,6,8);broad[...,2]=5.
    broad_prediction=broad[...,:4].clone()
    acc.add(codec_decoded_metrics(broad_prediction,broad,broad,network,cfg),
            broad_prediction,broad,broad,capacity,limits,cfg)
    isolated=torch.zeros(2,6,8);isolated[:,0,2]=5.
    bad=isolated[...,:4].clone();bad[:,0,1]=.4
    acc.add(codec_decoded_metrics(bad,isolated,isolated,network,cfg),bad,isolated,isolated,capacity,limits,cfg)
    metrics=acc.result(2)
    assert metrics['codec_speed_mae_slow_mps']<.02
    assert metrics['localized_slow_road_mae_max_mps']<.1
    assert metrics['local_context_frames']==2
    assert metrics['local_context_slow_cells']==2
    assert metrics['local_context_slow_mae_mps']==pytest.approx(4.)
    assert metrics['local_context_slow_road_max_mps']==pytest.approx(4.)
    scarce=acc.result(20)
    assert scarce['local_context_eligible_roads']==0
    assert scarce['local_context_slow_road_max_mps'] is None
