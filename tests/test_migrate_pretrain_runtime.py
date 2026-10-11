from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import migrate_pretrain_runtime as migration
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config, run=replace(config.run, output_dir=source, world_model_initialization_epochs=5),
        resources=replace(config.resources, world_model_initialization_batch_size=1,
            world_model_gradient_accumulation_steps=32, world_model_pretrain_compile_blocks=False),
        codec=replace(config.codec, decoder_pretrain_enabled=True, decoder_allow_unready=False))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, source / 'run_config.toml')
    state = dict(reward_contract='causal_cumulative_mean_global_v1', models={'world_model': {'weight': torch.arange(6.).reshape(2, 3)}},
        optimizers={'world_model_initial': {'state': {0: dict(exp_avg=torch.ones(2, 3),
            exp_avg_sq=torch.full((2, 3), 2.), step=torch.tensor(3.))}}},
        rng={'torch_cpu': torch.get_rng_state()}, next_environment_episode=0,
        replay={'online': {'episodes': []}, 'codec_validation': {'manifest_identity': 'held-out'},
            'offline': dict(schema_version=1, manifest_identity='train', epochs_completed=0,
                epoch_order=[0, 1], episode_cursor=0, start_order=list(range(130)), start_cursor=70)},
        metrics=dict(codec_ready=False, codec_updates=500, decoder_pretrain_updates=3,
            decoder_pretrain_authorized_manifest_identity='held-out',
            decoder_pretrain_authorized_codec_updates=500,
            world_model_initialization_epochs_completed=0, world_model_initialization_samples_seen=70,
            world_model_epoch_samples=70, world_model_initialization_updates=3,
            world_model_epoch_batches=70, online_episodes_completed=0))
    for filename, identity in [('offline_manifest.json', 'train'), ('codec_validation_manifest.json', 'held-out')]:
        (source / filename).write_text(json.dumps(dict(identity=identity)))
    save(source, config, state)
    return source, config, state


def save(source, config, state, next_unit=0):
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.PRETRAIN, next_unit=next_unit, training_state=state)


def test_partial_continuous_decoder_preserves_every_training_state_value(tmp_path):
    source, old_config, state = fixture(tmp_path)
    before = migration.sha256_file(source / 'checkpoints/latest.pt')
    output = tmp_path / 'world_model_ingo_profile'
    proof = migration.migrate(source, output)
    config = load_config(output / 'run_config.toml')
    checkpoint = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                                 expected_input_hash='input')
    migration._assert_equal(state, checkpoint.training_state)
    assert not checkpoint.training_state['metrics']['codec_ready']
    assert checkpoint.next_unit == 0
    assert config.run.device == old_config.run.device
    assert config.run.world_model_initialization_epochs == 1
    assert all(getattr(config.resources, key) == value for key, value in migration.PROFILE.items())
    assert proof['samples_seen'] == 70
    assert proof['epoch_samples'] == 70
    assert migration.sha256_file(source / 'checkpoints/latest.pt') == before
    assert migration.validate_runtime_continuation(output, config, 'input') == proof
    for name in ('offline_manifest.json', 'codec_validation_manifest.json'):
        assert (output / name).read_bytes() == (source / name).read_bytes()


def test_compact_sampler_rebatching_does_not_repeat_or_skip_remaining_starts():
    dataset = object.__new__(CompactPretrainDataset)
    dataset.manifest = {'entries': [dict(windows=130), dict(windows=5)]}
    dataset.sequence_length = 1
    dataset.config = SimpleNamespace(method=SimpleNamespace(supervision_horizon=1, fm_training_mode='single_step'))
    dataset._epoch_order, dataset._episode_cursor = [0, 1], 0
    dataset._start_order, dataset._start_cursor = list(range(130)), 70
    dataset.epochs_completed = 0
    dataset.verify_integrity = lambda **kwargs: None
    dataset.batch_from_starts = lambda episode, starts, **kwargs: [(episode, start) for start in starts]
    batches = list(dataset.iter_epoch(64, np.random.default_rng(42)))
    assert len(batches[0]) == 60
    assert batches[0] == [(0, start) for start in range(70, 130)]
    remaining = [pair for batch in batches for pair in batch]
    assert len(remaining) == len(set(remaining)) == 65
    assert sorted(remaining) == [(0, start) for start in range(70, 130)] + [(1, start) for start in range(5)]
    assert dataset.epochs_completed == 1


@pytest.mark.parametrize('section,field,value', [
    ('run', 'device', 'cuda:1'), ('run', 'seed', 999),
    ('method', 'fm_steps', 2), ('resources', 'world_model_initialization_sequence_length', 3),
    ('codec', 'decoder_allow_unready', True)])
def test_only_approved_runtime_fields_and_budget_change_allowed(tmp_path, section, field, value):
    _, config, _ = fixture(tmp_path)
    target = migration.target_config(config, tmp_path / 'target')
    if field == 'device' and value == config.run.device:
        value = 'cpu'
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only the approved'):
        migration.validate_runtime_transfer_config(config, target)


def test_epoch_budget_cannot_discard_completed_epoch(tmp_path):
    source, config, state = fixture(tmp_path)
    state['metrics']['world_model_initialization_epochs_completed'] = 1
    state['replay']['offline']['epochs_completed'] = 1
    save(source, config, state, next_unit=1)
    with pytest.raises(ValueError, match='epoch budget'):
        migration.migrate(source, tmp_path / 'target')


@pytest.mark.parametrize('mutation', ['missing_admission', 'changed_manifest', 'changed_encoder', 'online'])
def test_invalid_quality_admission_and_online_state_rejected(tmp_path, mutation):
    source, config, state = fixture(tmp_path)
    if mutation == 'missing_admission':
        del state['metrics']['decoder_pretrain_authorized_manifest_identity']
    elif mutation == 'changed_manifest':
        state['replay']['codec_validation']['manifest_identity'] = 'changed'
    elif mutation == 'changed_encoder':
        state['metrics']['codec_updates'] += 1
    else:
        state['optimizers']['actor'] = {'state': {0: dict(step=torch.tensor(1.))}}
    save(source, config, state)
    with pytest.raises(ValueError, match='held-out codec proof|online optimizers'):
        migration.migrate(source, tmp_path / 'target')


def test_frozen_proof_detects_tampering(tmp_path):
    source, _, _ = fixture(tmp_path)
    target = tmp_path / 'target'
    migration.migrate(source, target)
    config = load_config(target / 'run_config.toml')
    with (target / 'continuation_checkpoint.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        migration.validate_runtime_continuation(target, config, 'input')
