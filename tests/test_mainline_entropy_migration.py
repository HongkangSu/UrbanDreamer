import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from tools import migrate_mainline_entropy as migration
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def _fingerprint(config):
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def _fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = _fingerprint(replace(config,
        run=replace(config.run, output_dir=source, online_training_episodes=2000),
        method=replace(config.method, actor_entropy_mode='legacy', sp_prior_weight=0.)))
    export_run_config(config, source / 'run_config.toml')
    protocol = dict(source={'sha256': 'pretrained'}, runtime_config_fingerprint=config.semantic_fingerprint,
        calibrated_noincident_initialization={'ancestry': 'unchanged'}, online_episode_budget=2000,
        evaluation_every=20, gpu_memory_fraction=.25, threads=4)
    (source / 'pilot_protocol.json').write_text(json.dumps(protocol))
    progress = dict(online=[{'episode': i + 1} for i in range(507)], before=[{'score': 3.}], after=[],
                    milestones=[{'online_episodes': 500}], phase='online')
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'actor': {'weight': torch.arange(6.).reshape(2, 3)}, 'world_model': {'weight': torch.eye(3)}},
        optimizers={'actor': {'state': {0: {'exp_avg': torch.ones(2, 3), 'step': torch.tensor(80)}}}},
        rng={'torch': torch.get_rng_state()}, replay={'online': {'episodes': [{'events': [1, 2, 3]}]}},
        metrics={'online_pilot_progress': progress, 'online_episodes_completed': 507,
                 'actor_entropy_mode': 'legacy'}, next_environment_episode=507)
    _save(source, config, state, 507)
    (source / 'pilot_progress.json').write_text(json.dumps({'online': []}))
    for name in ('offline_manifest.json', 'codec_validation_manifest.json',
                 'calibrated_noincident_initialization.json', 'initialization_checkpoint.pt'):
        (source / name).write_text('{}')
    return source, config, state


def _save(output, config, state, cursor):
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=cursor, training_state=state)


def test_lossless_continuation_preserves_ancestry_and_checkpoint_owned_progress(tmp_path):
    source, source_config, state = _fixture(tmp_path)
    old_hash = migration.sha256_file(source / 'checkpoints/latest.pt')
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(restored.training_state, state)
    assert restored.next_unit == manifest['global_completed_episodes'] == 507
    assert config.method.actor_entropy_mode == 'population'
    assert config.method.sp_prior_weight == 0
    assert config.semantic_fingerprint != source_config.semantic_fingerprint
    assert migration.sha256_file(source / 'checkpoints/latest.pt') == old_hash
    assert len(json.loads((output / 'pilot_progress.json').read_text())['online']) == 507
    protocol = json.loads((output / 'pilot_protocol.json').read_text())
    assert protocol['calibrated_noincident_initialization'] == {'ancestry': 'unchanged'}
    assert protocol['gpu_memory_fraction'] == .25
    assert protocol['online_episode_budget'] == 2000
    assert protocol['actor_entropy'] == migration.entropy_protocol()
    assert set(manifest['copied_artifacts']) == {'offline_manifest.json', 'codec_validation_manifest.json',
        'calibrated_noincident_initialization.json', 'initialization_checkpoint.pt'}
    assert migration.validate_entropy_continuation(output, config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('section,field,value', [
    ('method', 'fm_steps', 99), ('method', 'sp_prior_weight', 1.),
    ('simulation', 'incident_speed_mps', 2.), ('run', 'seed', 400),
    ('run', 'online_training_episodes', 3000),
    ('resources', 'world_model_initialization_sequence_length', 99)])
def test_only_entropy_mode_and_runtime_location_can_change(tmp_path, section, field, value):
    _, config, _ = _fixture(tmp_path)
    target = replace(config, method=replace(config.method, actor_entropy_mode='population'))
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only actor entropy mode'):
        migration.validate_entropy_transfer_config(config, target)


@pytest.mark.parametrize('mode', ['population', 'off'])
def test_source_must_have_legacy_entropy(tmp_path, mode):
    source, config, _ = _fixture(tmp_path)
    config = _fingerprint(replace(config, method=replace(config.method, actor_entropy_mode=mode)))
    export_run_config(config, source / 'run_config.toml')
    with pytest.raises(ValueError, match='legacy to population'):
        migration.migrate(source, tmp_path / 'target')


@pytest.mark.parametrize('key', ['online_episodes_completed', 'next_environment_episode'])
def test_inconsistent_durable_counters_rejected(tmp_path, key):
    source, config, state = _fixture(tmp_path)
    if key == 'next_environment_episode':
        state[key] = 508
    else:
        state['metrics'][key] = 508
    _save(source, config, state, 507)
    with pytest.raises(ValueError, match='online counters'):
        migration.migrate(source, tmp_path / 'target')


@pytest.mark.parametrize('filename', ['source_checkpoint.pt', 'source_config.toml',
    'source_protocol.json', 'continuation_checkpoint.pt', 'calibrated_noincident_initialization.json'])
def test_tampered_frozen_artifact_rejected(tmp_path, filename):
    source, _, _ = _fixture(tmp_path)
    output = tmp_path / 'target'
    migration.migrate(source, output)
    with (output / filename).open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match=f'{filename} hash'):
        migration.validate_entropy_continuation(output, load_config(output / 'run_config.toml'), 'input', 'pretrained')


def test_advanced_checkpoint_can_resume_without_rewriting_immutable_initial_state(tmp_path):
    source, _, _ = _fixture(tmp_path)
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    state = copy.deepcopy(restored.training_state)
    state['models']['actor']['weight'] += 1
    state['metrics']['online_pilot_progress']['online'].append({'episode': 508})
    state['metrics']['online_episodes_completed'] = 508
    state['next_environment_episode'] = 508
    _save(output, config, state, 508)
    assert migration.validate_entropy_continuation(output, config, 'input', 'pretrained') == manifest
    with pytest.raises(ValueError, match='pretrained ancestry'):
        migration.validate_entropy_continuation(output, config, 'input', 'different')


def test_existing_output_refused(tmp_path):
    source, _, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match='new absent'):
        migration.migrate(source, source)
