import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from tools import migrate_mainline_prior as migration
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def _fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config, run=replace(config.run, output_dir=source, online_training_episodes=2000),
                     method=replace(config.method, sp_prior_weight=0.))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, source / 'run_config.toml')
    protocol = dict(source={'sha256': 'pretrained'}, runtime_config_fingerprint=config.semantic_fingerprint,
                    meso_initialization={'ancestry': 'unchanged'}, online_episode_budget=2000, evaluation_every=20)
    (source / 'pilot_protocol.json').write_text(json.dumps(protocol))
    progress = dict(online=[{'episode': i} for i in range(507)], before=[{'score': 3.}], after=[],
                    milestones=[{'online_episodes': 500}], phase='online')
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'actor': {'weight': torch.arange(6.).reshape(2, 3)}},
        optimizers={'actor': {'state': {0: {'exp_avg': torch.ones(2, 3), 'step': torch.tensor(80)}}}},
        rng={'torch': torch.get_rng_state()}, replay={'online': {'episodes': [{'events': [1, 2, 3]}]}},
        metrics={'online_pilot_progress': progress, 'online_episodes_completed': 507}, next_environment_episode=507)
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=507, training_state=state)
    # JSON is intentionally stale; the checkpoint owns durable progress.
    (source / 'pilot_progress.json').write_text(json.dumps({'online': []}))
    for name in ('offline_manifest.json', 'codec_validation_manifest.json'):
        (source / name).write_text('{}')
    return source, config, state


def test_continuation_preserves_all_state_and_global_schedule(tmp_path):
    source, source_config, state = _fixture(tmp_path)
    old_hash = migration.sha256_file(source / 'checkpoints/latest.pt')
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    target_config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=target_config.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(restored.training_state, state)
    assert restored.next_unit == manifest['global_completed_episodes'] == 507
    assert target_config.method.sp_prior_weight == 1.
    assert target_config.semantic_fingerprint != source_config.semantic_fingerprint
    assert migration.sha256_file(source / 'checkpoints/latest.pt') == old_hash
    assert len(json.loads((output / 'pilot_progress.json').read_text())['online']) == 507
    protocol = json.loads((output / 'pilot_protocol.json').read_text())
    assert protocol['meso_initialization'] == {'ancestry': 'unchanged'}
    assert protocol['shortest_path_prior']['off_episode'] == 1000
    assert protocol['shortest_path_prior']['schedule'] == 'cosine_global_one_based_episode'
    assert migration.validate_prior_continuation(output, target_config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('section,field,value', [('method', 'fm_steps', 99),
    ('simulation', 'incident_speed_mps', 2.), ('run', 'seed', 400),
    ('resources', 'world_model_initialization_sequence_length', 99)])
def test_only_prior_and_resource_location_can_change(tmp_path, section, field, value):
    _, config, _ = _fixture(tmp_path)
    target = replace(config, method=replace(config.method, sp_prior_weight=1.))
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only policy prior'):
        migration.validate_prior_transfer_config(config, target)


def test_tampered_frozen_source_is_rejected(tmp_path):
    source, _, _ = _fixture(tmp_path)
    output = tmp_path / 'target'
    migration.migrate(source, output)
    with (output / 'source_checkpoint.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='source_checkpoint.pt hash'):
        migration.validate_prior_continuation(output, load_config(output / 'run_config.toml'), 'input', 'pretrained')


def test_advanced_checkpoint_is_allowed_without_rewriting_initial_proof(tmp_path):
    source, _, _ = _fixture(tmp_path)
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    state = copy.deepcopy(restored.training_state)
    state['models']['actor']['weight'] += 1
    state['metrics']['online_pilot_progress']['online'].append({'episode': 507})
    state['next_environment_episode'] = 508
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=508, training_state=state)
    assert migration.validate_prior_continuation(output, config, 'input', 'pretrained') == manifest
    with pytest.raises(ValueError, match='pretrained ancestry'):
        migration.validate_prior_continuation(output, config, 'input', 'different')
