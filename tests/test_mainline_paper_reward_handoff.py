from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from tools import continue_mainline_paper_reward as future
from tools import queue_pretrained_online as queue
from tests.test_queued_pretrained_online import source_fixture
from urban_dreamer.workflow.config import load_config


def test_critic_expansion_preserves_function_and_adam_moments(tmp_path):
    _, _, state = source_fixture(tmp_path)
    for name, key, width in (('latent_critic', 'trunk.0.weight', 66),
                             ('event_critic', 'scorer.0.weight', 413)):
        weight = torch.randn(4, width)
        state['models'][name][key] = weight
        state['optimizers'][name] = dict(param_groups=[dict(params=[11, 12])], state={
            11: dict(step=torch.tensor(7.), exp_avg=weight + 1, exp_avg_sq=weight.square()),
            12: dict(step=torch.tensor(7.), exp_avg=torch.ones(4), exp_avg_sq=torch.ones(4))})
    original = deepcopy(state)
    result, manifest = queue.migrate_pretrained_reward(state)
    queue._assert_equal(original, state)
    for name, key, width, insertion in (('latent_critic', 'trunk.0.weight', 66, 66),
                                       ('event_critic', 'scorer.0.weight', 413, 147)):
        x = torch.randn(5, width)
        expanded = torch.cat((x[:, :insertion], torch.randn(5, 2), x[:, insertion:]), dim=1)
        torch.testing.assert_close(x @ state['models'][name][key].T,
                                   expanded @ result['models'][name][key].T)
        for moment in ('exp_avg', 'exp_avg_sq'):
            old = state['optimizers'][name]['state'][11][moment]
            new = result['optimizers'][name]['state'][11][moment]
            assert torch.equal(old[:, :insertion], new[:, :insertion])
            assert torch.equal(old[:, insertion:], new[:, insertion + 2:])
            assert not torch.count_nonzero(new[:, insertion:insertion + 2])
        queue._assert_equal(state['optimizers'][name]['state'][12], result['optimizers'][name]['state'][12])
    assert result['replay']['online']['schema_version'] == 8
    assert manifest['training_state_preserved_exactly'] is False
    queue._assert_equal(state['rng'], result['rng'])
    queue._assert_equal(state['metrics'], result['metrics'])
    queue._assert_equal(state['optimizers']['world_model_initial'], result['optimizers']['world_model_initial'])


@pytest.mark.parametrize('activity', ['counter', 'replay', 'environment'])
def test_pretrain_migration_refuses_legacy_online_activity(tmp_path, activity):
    _, _, state = source_fixture(tmp_path)
    if activity == 'counter':
        state['metrics']['online_episodes_completed'] = 1
    elif activity == 'replay':
        state['replay']['online']['episodes'] = [{}]
    else:
        state['next_environment_episode'] = 1
    with pytest.raises(ValueError, match='zero prior online'):
        queue.migrate_pretrained_reward(state)


def test_event_critic_wrong_width_is_rejected(tmp_path):
    _, _, state = source_fixture(tmp_path)
    state['models']['event_critic']['scorer.0.weight'] = torch.zeros(3, 412)
    with pytest.raises(ValueError, match='unexpected shape'):
        queue.migrate_pretrained_reward(state)


def test_legacy_fingerprint_is_bound_to_contract_and_exact_payload(tmp_path):
    source, config, state = source_fixture(tmp_path)
    current = load_config(source / 'run_config.toml')
    raw = dict(config_fingerprint=config.semantic_fingerprint, training_state=state)
    bound = queue.checkpoint_source_config(current, raw)
    assert bound.semantic_fingerprint == config.semantic_fingerprint
    assert bound.semantic_fingerprint != current.semantic_fingerprint
    queue._assert_equal(current.resolved_snapshot(), bound.resolved_snapshot())
    raw['training_state'] = dict(state, reward_contract=queue.REWARD_CONTRACT)
    with pytest.raises(ValueError, match='contract and semantic fingerprint disagree'):
        queue.checkpoint_source_config(current, raw)
    raw['training_state'] = state
    raw['config_fingerprint'] = 'unrecognized'
    with pytest.raises(ValueError, match='contract and semantic fingerprint disagree'):
        queue.checkpoint_source_config(current, raw)


def test_paper_waiter_refuses_live_prior_owner_without_signals(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(dict(previous_waiter={'pid': 9}, run_dir=str(tmp_path))))
    monkeypatch.setattr(future, 'validate_future_plan', lambda plan: None)
    monkeypatch.setattr(future, 'process_record', lambda pid: dict(pid=pid, state='S'))
    monkeypatch.setattr(future.os, 'kill', lambda *_: pytest.fail('must never signal a process'))
    monkeypatch.setattr(future.subprocess, 'Popen', lambda *_a, **_kw: pytest.fail('duplicate launch'))
    with pytest.raises(ValueError, match='previous waiter must exit'):
        future.run_future(path)
    assert not (tmp_path / 'status.json').exists()


def test_future_guard_binds_pretrain_config_and_original_plan(tmp_path, monkeypatch):
    for name in ('training_plan.json', 'pretrain_config.toml', 'previous.json'):
        (tmp_path / name).write_text('{}')
    monkeypatch.setattr(future, 'verify_execution_source', lambda root: 'manifest')
    old = dict(run_dir=str(tmp_path), revision_root=str(tmp_path / 'old_source'),
               controller={'pid': 1}, pretrainer={'pid': 2},
               online_command=['python', str(tmp_path / 'old_source/tools/queue_pretrained_online.py'),
                               '--execution-source', str(tmp_path / 'old_source')])
    (tmp_path / 'previous.json').write_text(json.dumps(old))
    plan = dict(schema_version=1, kind='future_online_paper_reward_handoff',
        reward_contract=queue.REWARD_CONTRACT, online_disagreement_weight=0.,
        run_dir=str(tmp_path), revision_root=str(tmp_path / 'source'),
        execution_manifest_sha256='manifest', previous_plan_path=str(tmp_path / 'previous.json'),
        previous_plan_sha256=queue.sha256_file(tmp_path / 'previous.json'),
        original_training_plan_sha256=queue.sha256_file(tmp_path / 'training_plan.json'),
        original_pretrain_config_sha256=queue.sha256_file(tmp_path / 'pretrain_config.toml'),
        controller=old['controller'], pretrainer=old['pretrainer'],
        online_command=['python', str(tmp_path / 'source/tools/queue_pretrained_online.py'),
                        '--execution-source', str(tmp_path / 'source')])
    future.validate_future_plan(plan)
    (tmp_path / 'pretrain_config.toml').write_text('changed')
    with pytest.raises(ValueError, match='immutable plan or source changed'):
        future.validate_future_plan(plan)
