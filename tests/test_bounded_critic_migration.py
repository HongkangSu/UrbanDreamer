"""Critic migration is explicit and preserves every noncritic learning state."""
import copy
from dataclasses import replace

import pytest
import torch

from tools.migrate_bounded_critics import migrate_state
from tools.migrate_mainline_prior import _assert_equal
from test_urban_dreamer_training import make_trainer
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.workflow.checkpoint import _normalize_training_state


def test_migration_preserves_policy_world_replay_rng_and_clock(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.metrics['online_episodes_completed'] = 345.
    source = _normalize_training_state(trainer.state_dict())
    before = copy.deepcopy(source)
    method = replace(trainer.config.method, critic_value_mode='bounded_cohort_v1',
                     disagreement_weight=0.)
    rng = torch.get_rng_state().clone()
    target = migrate_state(source, method, seed=42)
    _assert_equal(source, before)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    for name in ('actor', 'world_model'):
        _assert_equal(source['models'][name], target['models'][name])
    for name in source['optimizers']:
        if name not in ('latent_critic', 'event_critic'):
            _assert_equal(source['optimizers'][name], target['optimizers'][name])
    _assert_equal(source['replay'], target['replay'])
    for key in source['rng']:
        _assert_equal(source['rng'][key], target['rng'][key])
    assert target['metrics']['online_episodes_completed'] == 345.
    assert target['next_environment_episode'] == source['next_environment_episode']
    assert not target['optimizers']['latent_critic']['state']
    _assert_equal(target['models']['latent_critic'], target['models']['target_latent_critic'])
    destination = MainlineTrainer(config=replace(trainer.config, method=method),
        prepared_inputs=trainer.prepared_inputs, topology=trainer.topology,
        _offline_data=trainer.offline_data, _codec_validation_data=trainer.codec_validation_data)
    destination.load_state_dict(target)
    restored = destination.state_dict()
    _assert_equal(restored['models']['target_latent_critic'], target['models']['target_latent_critic'])
    for key in ('critic_aux_numpy','critic_aux_torch_cpu','critic_aux_torch_cuda'):
        _assert_equal(restored['rng'][key], target['rng'][key])
    with pytest.raises(ValueError, match='nonlegacy'):
        migrate_state(target, method, seed=42)


def test_fresh_online_preserves_only_pretrained_learning_and_resets_online_state(tmp_path):
    trainer = make_trainer(tmp_path)
    source = _normalize_training_state(trainer.state_dict())
    method = replace(trainer.config.method, critic_value_mode='bounded_cohort_v1', disagreement_weight=0.)
    primary_before = torch.get_rng_state().clone()
    target = migrate_state(source, method, seed=42, fresh_online=True)
    torch.testing.assert_close(primary_before, torch.get_rng_state())
    _assert_equal(source['models']['world_model'], target['models']['world_model'])
    _assert_equal(source['replay']['offline'], target['replay']['offline'])
    _assert_equal(source['replay']['codec_validation'], target['replay']['codec_validation'])
    assert any(not torch.equal(source['models']['actor'][k], v) for k,v in target['models']['actor'].items())
    assert all(not opt['state'] for opt in target['optimizers'].values())
    assert target['replay']['online']['episodes'] == []
    assert target['replay']['online']['next_storage_id'] == 0
    assert target['metrics']['online_episodes_completed'] == target['next_environment_episode'] == 0
    assert target['metrics']['fresh_online_start'] is True
    target_again = migrate_state(source, method, seed=42, fresh_online=True)
    _assert_equal(target['models'], target_again['models'])
    source['metrics']['online_episodes_completed'] = 1
    with pytest.raises(ValueError, match='zero-online'):
        migrate_state(source, method, seed=42, fresh_online=True)
