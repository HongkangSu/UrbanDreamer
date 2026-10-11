import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import migrate_pretrain_single_step as migration
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


class Dataset:
    def __init__(self, split, length):
        self.manifest = dict(contract=dict(split=split, sequence_length=length), entries=[{'path': split}])
        self.manifest['identity'] = _identity(self.manifest)

    def state_dict(self):
        return dict(schema_version=1, manifest_identity=self.manifest['identity'], epoch_order=[],
                    start_order=[], episode_cursor=0, start_cursor=0, epochs_completed=0)


def fingerprint(config):
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def fixture(tmp_path):
    output = tmp_path / 'source'
    output.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = fingerprint(replace(config,
        run=replace(config.run, output_dir=output, world_model_initialization_epochs=5),
        resources=replace(config.resources, world_model_initialization_sequence_length=3),
        method=replace(config.method, fm_training_mode='legacy_independent', fm_solver='euler', fm_steps=2),
        data=replace(config.data, pretrain_cache_manifest=tmp_path / 'cache.json')))
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    old_train, old_validation = Dataset('train', 3), Dataset('validation', 3)
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'world_model': {'weight': parameter.detach().clone()}},
        optimizers={'world_model_initial': optimizer.state_dict()},
        rng={'torch': torch.get_rng_state()}, next_environment_episode=0,
        replay=dict(online={'episodes': []}, offline=old_train.state_dict(),
                    codec_validation=old_validation.state_dict()),
        metrics=dict(codec_ready=1., codec_updates=4500, codec_validation_updates=4500,
                     codec_validation_manifest_identity=old_validation.manifest['identity'],
                     world_model_initialization_epochs_completed=1.,
                     world_model_initialization_samples_seen=1002.,
                     world_model_initialization_batches_completed=51.,
                     world_model_initialization_updates=80., world_model_updates=80.,
                     world_model_epoch_samples=2., world_model_epoch_updates=1.,
                     online_episodes_completed=0.))
    state['replay']['offline'].update(epoch_order=[0], start_order=[3, 2, 0, 1], start_cursor=2,
                                      epochs_completed=1)
    export_run_config(config, output / 'run_config.toml')
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.PRETRAIN, next_unit=1, training_state=state)
    trainer = SimpleNamespace(offline_data=Dataset('train', 1), codec_validation_data=Dataset('validation', 1),
                              optimizers={'world_model_initial': optimizer},
                              load_state_dict=lambda state: None)
    state = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                            expected_input_hash='input').training_state
    return output, config, state, trainer


def test_new_coverage_keeps_models_warm_optimizer_rng_and_lifetime_updates(tmp_path, monkeypatch):
    source, _, state, trainer = fixture(tmp_path)
    monkeypatch.setattr(migration, '_make_trainer', lambda config: (trainer, 'input'))
    old_hash = migration.sha256_file(source / 'checkpoints/latest.pt')
    target = tmp_path / 'target'
    report = migration.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    loaded = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash='input')
    result = loaded.training_state
    for key in ('models', 'optimizers', 'rng'):
        migration._assert_equal(result[key], state[key])
    assert config.method.fm_training_mode == 'single_step'
    assert config.method.fm_solver == 'euler' and config.method.fm_steps == 100
    assert config.resources.world_model_initialization_sequence_length == 1
    assert config.resources.world_model_pretrain_compile_blocks
    assert loaded.next_unit == 0
    assert result['metrics']['world_model_updates'] == 80
    assert result['metrics']['world_model_initialization_updates'] == 80
    assert result['metrics']['world_model_initialization_samples_seen'] == 0
    assert result['metrics']['world_model_epoch_samples'] == 0
    assert result['replay']['offline'] == trainer.offline_data.state_dict()
    assert report['source_sampler']['start_cursor'] == 2
    assert report['source_progress']['world_model_initialization_samples_seen'] == 1002
    assert not result['metrics']['codec_ready']
    assert 'codec_validation_manifest_identity' not in result['metrics']
    assert result['metrics']['single_step_migration_codec_revalidation_required']
    assert migration.sha256_file(source / 'checkpoints/latest.pt') == old_hash
    assert migration.validate_single_step_continuation(target, config, 'input') == report


def test_reject_other_data_identity(tmp_path):
    _, config, state, trainer = fixture(tmp_path)
    state['replay']['offline']['manifest_identity'] = 'wrong'
    with pytest.raises(ValueError, match='manifest changed'):
        migration.migrate_training_state(state, config, trainer.offline_data, trainer.codec_validation_data)


@pytest.mark.parametrize('damage', ['nan', 'shape', 'variance', 'missing'])
def test_reject_corrupt_optimizer(tmp_path, damage):
    _, _, state, trainer = fixture(tmp_path)
    state = copy.deepcopy(state)
    moments = state['optimizers']['world_model_initial']['state'][0]
    if damage == 'nan':
        moments['exp_avg'][0, 0] = float('nan')
    elif damage == 'shape':
        moments['exp_avg'] = torch.ones(4)
    elif damage == 'variance':
        moments['exp_avg_sq'][0, 0] = -1
    else:
        del moments['exp_avg']
    with pytest.raises(ValueError, match='optimizer'):
        migration.validate_optimizer_state(trainer, state)


def test_reject_unrelated_configuration_change(tmp_path):
    _, config, _, _ = fixture(tmp_path)
    target = replace(config, method=replace(config.method, fm_training_mode='single_step', fm_steps=100),
                     resources=replace(config.resources, world_model_initialization_sequence_length=1,
                                       world_model_initialization_batch_size=999))
    with pytest.raises(ValueError, match='only FM'):
        migration.validate_single_step_transfer_config(config, target)


def test_reject_online_source(tmp_path, monkeypatch):
    source, config, state, trainer = fixture(tmp_path)
    state['replay']['online']['episodes'] = [{}]
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.PRETRAIN, next_unit=1, training_state=state)
    monkeypatch.setattr(migration, '_make_trainer', lambda config: (trainer, 'input'))
    with pytest.raises(ValueError, match='no online'):
        migration.migrate(source, tmp_path / 'target')
