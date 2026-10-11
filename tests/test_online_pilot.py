from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import run_online_pilot as pilot
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint
from test_urban_dreamer_config import _write_config


@pytest.mark.parametrize('device', ['cuda:0', 'cuda:1'])
def test_online_cli_uses_selected_gpu_for_preflight_and_memory_limit(tmp_path, monkeypatch, device):
    calls = []
    monkeypatch.setattr(pilot.torch.cuda, 'set_device', lambda value: calls.append(('select', str(value))))
    monkeypatch.setattr(pilot.torch.cuda, 'mem_get_info',
        lambda value: (calls.append(('memory', str(value))) or (3 * 1024**3, 24 * 1024**3)))
    monkeypatch.setattr(pilot.torch.cuda, 'set_per_process_memory_fraction',
        lambda fraction, value: calls.append(('limit', fraction, str(value))))
    class StopBeforeLoading(RuntimeError):
        pass
    def stop(path):
        raise StopBeforeLoading
    monkeypatch.setattr(pilot, 'load_config', stop)
    with pytest.raises(StopBeforeLoading):
        pilot.main(['--source-dir', str(tmp_path / 'source'), '--output-dir', str(tmp_path / 'online'),
                    '--device', device, '--gpu-memory-fraction', '.25', '--online-episodes', '2000'])
    assert calls == [('select', device), ('memory', device), ('limit', .25, device)]


class FakeTrainer:
    def __init__(self, tmp_path):
        self.config = load_config(_write_config(tmp_path / 'config.toml'))
        self.config = replace(self.config, run=replace(self.config.run, output_dir=tmp_path / 'pilot'))
        self.world_model = torch.nn.Linear(1, 1)
        self.actor = torch.nn.Linear(1, 1)
        self.latent_critic = torch.nn.Linear(1, 1)
        self.event_critic = torch.nn.Linear(1, 1)
        self.rng = np.random.default_rng(12)
        self._next_environment_episode = 0
        self.metrics = {'online_episodes_completed': 0}
        self.pilot_optimizer_steps = dict.fromkeys(pilot.MODEL_NAMES, 0)
        self.logs = []
        self.train_calls = []
        self.eval_calls = []

    def log(self, record):
        self.logs.append(record)

    def _run_episode(self, *, mode, episode_index):
        assert mode == 'evaluation'
        self.eval_calls.append(episode_index)
        self._next_environment_episode += 1
        self.metrics['evaluation_mutation'] = 1
        random.random();np.random.random();torch.rand(1);self.rng.random()
        directory = self.config.run.output_dir / 'evaluation_demands'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'episode_{episode_index:04d}.json').write_text(json.dumps({'demand_hash': str(episode_index)}))
        metrics = dict(evaluation_nonarrival_count=0., evaluation_completion_rate=1.,
                       evaluation_selection_score=-2.)
        return SimpleNamespace(metrics=metrics,total_vehicle_count=2,
                               transitions=[SimpleNamespace(terminal=True)], total_travel_minutes=2.)

    def train_online_episode(self, index):
        self.train_calls.append(index)
        self._next_environment_episode += 1
        self.metrics['online_episodes_completed'] += 1
        for name in pilot.MODEL_NAMES:
            with torch.no_grad():
                for parameter in getattr(self, name).parameters():parameter.add_(.01)
            self.pilot_optimizer_steps[name] += 16
        return {'imagined_events': 128., 'event_critic_loss': .2, 'actor_loss': -.2}

    def state_dict(self):
        return dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
            models={name:getattr(self,name).state_dict() for name in ('world_model',*pilot.MODEL_NAMES)},
            optimizers={},replay={},rng={},metrics=dict(self.metrics),next_environment_episode=self._next_environment_episode)

    def load_state_dict(self,state):
        for name,values in state['models'].items():getattr(self,name).load_state_dict(values)
        self.metrics = copy.deepcopy(state['metrics'])
        self._next_environment_episode = state['next_environment_episode']


def test_evaluation_preserves_global_policy_rng_counters_parameters_and_modes(tmp_path):
    trainer = FakeTrainer(tmp_path)
    state = copy.deepcopy(trainer.rng.bit_generator.state)
    random.seed(13);np.random.seed(14);torch.manual_seed(15)
    states = (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    expected_parameters = pilot.parameter_snapshot(trainer)
    trainer.actor.train(False)
    with pilot.isolated_evaluation(trainer,tmp_path/'evaluation'):
        trainer._run_episode(mode='evaluation',episode_index=0)
        assert not trainer.world_model.training
    assert trainer.metrics == {'online_episodes_completed':0}
    assert trainer._next_environment_episode == 0
    assert trainer.rng.bit_generator.state == state
    assert random.getstate() == states[0]
    np.testing.assert_array_equal(np.random.get_state()[1],states[1][1])
    torch.testing.assert_close(torch.get_rng_state(),states[2])
    assert trainer.world_model.training and not trainer.actor.training
    assert all(change['l2_change']==0 for change in pilot.parameter_changes(trainer,expected_parameters).values())


def test_evaluation_restores_training_state_even_if_sumo_fails(tmp_path):
    trainer=FakeTrainer(tmp_path)
    with pytest.raises(RuntimeError):
        with pilot.isolated_evaluation(trainer,tmp_path/'evaluation'):
            trainer._next_environment_episode=99
            trainer.metrics['mutation']=10
            raise RuntimeError('sumo interrupted')
    assert trainer._next_environment_episode==0 and trainer.metrics=={'online_episodes_completed':0}


def test_pilot_before_and_after_eval_do_not_consume_five_real_actor_critic_episodes(tmp_path):
    trainer=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',
                             protocol={'source':{'sha256':'frozen'},'budgets':pilot.BUDGETS})
    result=runner.run()
    assert result['status']=='completed'
    assert trainer.train_calls==[0,1,2,3,4]
    assert trainer.eval_calls==[0,1,2,0,1,2]
    assert trainer._next_environment_episode==5
    assert trainer.metrics['online_episodes_completed']==5
    for episode in result['online']:
        assert episode['optimizer_steps']==dict.fromkeys(pilot.MODEL_NAMES,16)
        assert all(value['l2_change']>0 for value in episode['parameter_changes'].values())
    assert [row['demand_hash'] for row in result['before']]==[row['demand_hash'] for row in result['after']]
    restored=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(restored,output=trainer.config.run.output_dir,input_hash='network',protocol=runner.protocol)
    assert resumed.run()==result
    assert restored.train_calls==[] and restored.eval_calls==[]


def test_baseline_only_resumes_without_repeating_eval_or_reducing_online_budget(tmp_path):
    first=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(first,output=first.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}})
    baseline=runner.run(baseline_only=True)
    assert baseline['status']=='baseline_evaluated'
    assert first.train_calls==[] and first.eval_calls==[0,1,2]
    assert first._next_environment_episode==0
    second=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(second,output=first.config.run.output_dir,input_hash='network',protocol=runner.protocol)
    resumed.run()
    assert second.train_calls==[0,1,2,3,4] and second.eval_calls==[0,1,2]


def test_checkpoint_embedded_progress_recovers_episode_committed_before_json_write(tmp_path):
    trainer=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}})
    runner.run(baseline_only=True)
    previous=json.loads(runner.progress_path.read_text())
    trainer.train_online_episode(0)
    runner.progress['online'].append({'episode_index':0,'metrics':{'imagined_events':128}})
    runner.save()
    pilot.atomic_json(runner.progress_path,previous)  # Simulated power loss before secondary JSON commit.
    second=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(second,output=trainer.config.run.output_dir,input_hash='network',protocol=runner.protocol)
    resumed.resume()
    assert len(resumed.progress['online'])==1
    assert second._next_environment_episode==1


def test_censored_pairs_do_not_claim_completed_travel_time_improvement():
    row=dict(episode_index=0,demand_hash='same',total_vehicle_count=10,arrived_count=5,
             nonarrival_count=5,complete=False,observed_minutes_per_vehicle=4.,travel_minutes_per_vehicle=None,
             metrics={'evaluation_completion_rate':.5,'evaluation_selection_score':-25004.})
    after=copy.deepcopy(row);after['observed_minutes_per_vehicle']=3.
    result=pilot.compare_paired([row],[after])
    assert result['pairs'][0]['censored_pair']
    assert result['pairs'][0]['completed_travel_minutes_delta'] is None
    assert result['pairs'][0]['observed_minutes_delta']==-1
    after['demand_hash']='different'
    with pytest.raises(ValueError,match='identical realized demand'):pilot.compare_paired([row],[after])


def test_original_mainline_actor_critic_budgets_are_enforced(monkeypatch):
    pilot.assert_original_budgets()
    monkeypatch.setattr(pilot.training,'IMAGINED_PASSES_PER_EPISODE',1)
    with pytest.raises(ValueError,match='original full online actor-critic'):pilot.assert_original_budgets()


def test_source_must_be_strictly_completed_pretraining_with_no_online_state(tmp_path):
    trainer=FakeTrainer(tmp_path)
    state=trainer.state_dict();state['metrics']['world_model_initialization_epochs_completed']=trainer.config.run.world_model_initialization_epochs
    state['replay']={'online':{'episodes':[]}}
    source=SimpleNamespace(phase=CheckpointPhase.ONLINE_TRAINING,next_unit=0,training_state=state)
    pilot.validate_pretrained_source(source,trainer.config)
    source.next_unit=1
    with pytest.raises(ValueError,match='online-start'):pilot.validate_pretrained_source(source,trainer.config)
    source.next_unit=0;state['next_environment_episode']=1
    with pytest.raises(ValueError,match='prior online'):pilot.validate_pretrained_source(source,trainer.config)


def test_completed_run_monotonically_extends_without_resetting_actor_or_training_history(tmp_path):
    trainer=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}})
    old=runner.run()
    before_params=pilot.parameter_snapshot(trainer)
    extended=FakeTrainer(tmp_path)
    continuation=pilot.OnlinePilot(extended,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}},
                                  online_episodes=7,evaluation_every=2,extend_run=True)
    continuation.resume()
    assert continuation.progress['phase']=='online_training'
    assert len(continuation.progress['online'])==5 and continuation.progress['after']==[]
    assert extended._next_environment_episode==5
    assert all(delta['l2_change']==0 for delta in pilot.parameter_changes(extended,before_params).values())
    assert json.loads((trainer.config.run.output_dir/'milestones/after_000005/result.json').read_text())==old
    result=continuation.run()
    assert extended.train_calls==[5,6]
    assert result['effective_online_episode_budget']==7
    assert result['environment_episodes_consumed']==7
    assert len(result['online'])==7
    milestones={row['online_episodes']:row for row in continuation.progress['milestones']}
    assert milestones[5]['phase']=='completed'
    assert milestones[6]['phase']=='completed' and len(milestones[6]['evaluations'])==3
    assert extended.eval_calls==[0,1,2,0,1,2]  # milestone6, then final7; original baseline preserved.
    budget=json.loads((trainer.config.run.output_dir/'effective_run_budget.json').read_text())
    assert budget['authoritative_runner_online_episodes']==7
    assert budget['source_config_online_training_episodes']==4


@pytest.mark.parametrize('budget,every,extend,message',[(4,0,True,'cannot decrease'),(7,0,False,'--extend-run'),(5,20,False,'--extend-run')])
def test_extension_requires_explicit_monotonic_budget_and_preserves_other_protocol_fields(tmp_path,budget,every,extend,message):
    trainer=FakeTrainer(tmp_path)
    original=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}})
    original.run(baseline_only=True)
    other=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(other,output=trainer.config.run.output_dir,input_hash='network',protocol=original.protocol,
                              online_episodes=budget,evaluation_every=every,extend_run=extend)
    with pytest.raises(ValueError,match=message):resumed.resume()
    altered=pilot.OnlinePilot(other,output=trainer.config.run.output_dir,input_hash='network',
                              protocol={'source':{'sha256':'different'}},online_episodes=2000,extend_run=True)
    with pytest.raises(ValueError,match='source identity'):altered.resume()


def test_periodic_evaluation_restores_rng_counters_and_resumes_each_demand(tmp_path):
    trainer=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}},
                             online_episodes=3,evaluation_every=2)
    original=trainer._run_episode
    failures={'active':True}
    def fail_once(*,mode,episode_index):
        if 'milestones' in trainer.config.run.output_dir.parts and episode_index==1 and failures['active']:
            failures['active']=False
            raise RuntimeError('periodic eval interrupted')
        return original(mode=mode,episode_index=episode_index)
    trainer._run_episode=fail_once
    with pytest.raises(RuntimeError,match='periodic eval interrupted'):runner.run()
    assert trainer._next_environment_episode==2
    progress=json.loads(runner.progress_path.read_text())
    assert len(progress['online'])==2 and len(progress['milestones'][0]['evaluations'])==1
    restored=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(restored,output=trainer.config.run.output_dir,input_hash='network',protocol=runner.protocol,
                              online_episodes=3,evaluation_every=2)
    result=resumed.run()
    assert restored.train_calls==[2]
    assert restored.eval_calls==[1,2,0,1,2]
    assert restored._next_environment_episode==3
    assert result['effective_online_episode_budget']==3


def test_partial_after_evaluation_is_archived_when_extended_instead_of_reused_at_new_end(tmp_path):
    trainer=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}})
    original=trainer._run_episode
    def interrupt_after(*,mode,episode_index):
        if trainer.config.run.output_dir.name=='after_evaluation' and episode_index==1:
            raise RuntimeError('after5 interrupted')
        return original(mode=mode,episode_index=episode_index)
    trainer._run_episode=interrupt_after
    with pytest.raises(RuntimeError,match='after5 interrupted'):runner.run()
    assert len(runner.progress['online'])==5 and len(runner.progress['after'])==1
    resumed_trainer=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(resumed_trainer,output=trainer.config.run.output_dir,input_hash='network',protocol=runner.protocol,
                              online_episodes=6,extend_run=True)
    resumed.run()
    assert resumed_trainer.train_calls==[5]
    assert resumed_trainer.eval_calls==[0,1,2]
    assert resumed.progress['milestones'][0]['phase']=='incomplete'
    assert len(resumed.progress['milestones'][0]['evaluations'])==1
    assert len(resumed.progress['after'])==3


def test_periodic_checkpoint_newer_than_json_at_equal_online_count_is_authoritative(tmp_path):
    trainer=FakeTrainer(tmp_path)
    runner=pilot.OnlinePilot(trainer,output=trainer.config.run.output_dir,input_hash='network',protocol={'source':{'sha256':'frozen'}},
                             online_episodes=3,evaluation_every=2)
    original=trainer._run_episode
    def interrupt(*,mode,episode_index):
        if 'milestones' in trainer.config.run.output_dir.parts and episode_index==1:
            raise RuntimeError('periodic power loss')
        return original(mode=mode,episode_index=episode_index)
    trainer._run_episode=interrupt
    with pytest.raises(RuntimeError,match='periodic power loss'):runner.run()
    stale=copy.deepcopy(runner.progress)
    stale['milestones'][0]['evaluations']=[]
    pilot.atomic_json(runner.progress_path,stale)
    resumed_trainer=FakeTrainer(tmp_path)
    resumed=pilot.OnlinePilot(resumed_trainer,output=trainer.config.run.output_dir,input_hash='network',protocol=runner.protocol,
                              online_episodes=3,evaluation_every=2)
    resumed.run()
    assert resumed_trainer.eval_calls==[1,2,0,1,2]
    assert len(resumed.progress['milestones'][0]['evaluations'])==3
