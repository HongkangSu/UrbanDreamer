from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tools import migrate_pretrain_compile as migration
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def _fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config, run=replace(config.run, output_dir=source, world_model_initialization_epochs=5),
                     resources=replace(config.resources, world_model_pretrain_compile_blocks=False))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, source / 'run_config.toml')
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'world_model': {'weight': torch.arange(6.).reshape(2, 3)}},
        optimizers={'world_model_initial': {'state': {0: {
            'exp_avg': torch.ones(2, 3), 'exp_avg_sq': torch.ones(2, 3) * 2, 'step': torch.tensor(80)}}}},
        rng={'torch': torch.get_rng_state()},
        replay={'online': {'episodes': []}, 'codec_validation': {'identity': 'held-out'},
                'offline': dict(epochs_completed=1, epoch_order=[2, 0, 1], episode_cursor=1,
                                start_order=[5, 4, 0, 1, 3, 2], start_cursor=2)},
        metrics=dict(codec_ready=True, online_episodes_completed=0,
                     world_model_initialization_epochs_completed=1,
                     world_model_initialization_samples_seen=1002, world_model_epoch_samples=2,
                     world_model_initialization_updates=80), next_environment_episode=0)
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.PRETRAIN, next_unit=1, training_state=state)
    return source, config, state


def test_partial_epoch_preserves_warm_optimizer_rng_and_sampler(tmp_path):
    source, _, state = _fixture(tmp_path)
    old_hash = migration.sha256_file(source / 'checkpoints/latest.pt')
    target = tmp_path / 'target'
    report = migration.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    checkpoint = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                                 expected_input_hash='input')
    migration._assert_equal(checkpoint.training_state, state)
    assert checkpoint.next_unit == 1
    assert report['samples_seen'] == 1002
    assert report['epoch_samples'] == 2
    cursor = checkpoint.training_state['replay']['offline']
    assert cursor['start_order'][cursor['start_cursor']:] == [0, 1, 3, 2]
    assert config.resources.world_model_pretrain_compile_blocks
    assert migration.sha256_file(source / 'checkpoints/latest.pt') == old_hash
    assert migration.validate_compile_continuation(target, config, 'input') == report


@pytest.mark.parametrize('section,field,value', [
    ('resources', 'world_model_initialization_batch_size', 999),
    ('method', 'supervision_horizon', 7), ('simulation', 'incident_speed_mps', 2.),
    ('run', 'seed', 400)])
def test_only_execution_flag_and_location_may_change(tmp_path, section, field, value):
    _, config, _ = _fixture(tmp_path)
    target = replace(config, resources=replace(config.resources, world_model_pretrain_compile_blocks=True))
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only compile flag'):
        migration.validate_compile_transfer_config(config, target)


def test_wrong_input_identity_and_changed_frozen_source_rejected(tmp_path):
    source, _, _ = _fixture(tmp_path)
    target = tmp_path / 'target'
    migration.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    with pytest.raises(ValueError, match='target identity'):
        migration.validate_compile_continuation(target, config, 'different')
    with (target / 'source_checkpoint.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        migration.validate_compile_continuation(target, config, 'input')


def test_online_training_rejected(tmp_path):
    source, config, state = _fixture(tmp_path)
    state['optimizers']['actor'] = {'state': {0: {'step': torch.tensor(1)}}}
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.PRETRAIN, next_unit=1, training_state=state)
    with pytest.raises(ValueError, match='online optimizers'):
        migration.migrate(source, tmp_path / 'target')


def test_changed_source_configuration_rejected(tmp_path):
    source, config, _ = _fixture(tmp_path)
    changed = replace(config, run=replace(config.run, seed=config.run.seed + 1))
    changed = replace(changed, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(changed, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(changed, source / 'run_config.toml')
    with pytest.raises(ValueError, match='fingerprint'):
        migration.migrate(source, tmp_path / 'target')
