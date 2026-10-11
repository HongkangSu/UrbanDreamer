import copy
from dataclasses import replace
import json

import pytest
import torch

from tools import migrate_calibrated_decoder as migration
from test_imagination_budget_migration import fixture as budget_fixture
from urban_dreamer.models.world_model import FlowProjector
from urban_dreamer.workflow.checkpoint import CheckpointPhase, save_checkpoint, load_checkpoint
from urban_dreamer.workflow.config import load_config


def fixture(tmp_path):
    source, config, state, protocol = budget_fixture(tmp_path, passes=8)
    codec = FlowProjector(adjacency=torch.zeros(1, 1)).state_dict()
    state['models']['world_model'].update({'flow_projector.' + k: v for k, v in codec.items()})
    state['metrics']['codec_validation_manifest_identity'] = 'old-proof'
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=81, training_state=state)
    candidate = copy.deepcopy(codec)
    for key in candidate:
        if key.startswith('decoder_'):
            candidate[key] += .01
    calibration = tmp_path / 'calibration.pt'
    torch.save(dict(codec=candidate, updates=20682, status='candidate_unready'), calibration)
    return source, config, state, calibration, candidate


def test_installs_only_decoder_and_fresh_optimizers_preserving_online_state(tmp_path):
    source, _, state, calibration, candidate = fixture(tmp_path)
    output = tmp_path / 'target'
    manifest = migration.migrate(source, calibration, output)
    config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input')
    result = restored.training_state
    expected = copy.deepcopy(state)
    for key, value in candidate.items():
        if key.startswith('decoder_'):
            expected['models']['world_model']['flow_projector.' + key] = value
    expected['metrics']['codec_ready'] = 0.
    del expected['metrics']['codec_validation_manifest_identity']
    for name in ('decoder_pretrain', 'decoder_online'):
        assert result['optimizers'][name]['state'] == {}
        assert len(result['optimizers'][name]['param_groups'][0]['params']) == 8
        expected['optimizers'][name] = result['optimizers'][name]
        expected['metrics'][name + '_updates'] = 0.
    migration._assert_equal(expected, result)
    assert config.codec.decoder_online_enabled and config.codec.decoder_allow_unready
    assert restored.next_unit == 81
    assert manifest['calibration_updates'] == 20682
    assert manifest['calibration_status'] == 'candidate_unready'
    assert migration.validate_decoder_continuation(output, config, 'input', 'pretrained') == manifest


def test_encoder_mismatch_rejected_without_output(tmp_path):
    source, _, _, calibration, _ = fixture(tmp_path)
    raw = torch.load(calibration, weights_only=True)
    raw['codec']['input_proj.weight'] += 1
    torch.save(raw, calibration)
    with pytest.raises(ValueError, match='nondecoder'):
        migration.migrate(source, calibration, tmp_path / 'target')
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('filename', ['calibration_checkpoint.pt', 'source_checkpoint.pt', 'source_history/episodes/history.json'])
def test_frozen_artifacts_are_checked(tmp_path, filename):
    source, _, _, calibration, _ = fixture(tmp_path)
    output = tmp_path / 'target'
    migration.migrate(source, calibration, output)
    with (output / filename).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        migration.validate_decoder_continuation(output, load_config(output / 'run_config.toml'), 'input', 'pretrained')


def test_non_decoder_setting_change_rejected(tmp_path):
    _, config, _, _, _ = fixture(tmp_path)
    target = replace(config, codec=replace(config.codec, **migration.CODEC_CHANGES),
                     method=replace(config.method, fm_steps=50))
    with pytest.raises(ValueError, match='only authorized'):
        migration.validate_transfer_config(config, target)


def test_resume_can_advance_models_and_metrics(tmp_path):
    source, _, _, calibration, _ = fixture(tmp_path)
    output = tmp_path / 'target'
    manifest = migration.migrate(source, calibration, output)
    config = load_config(output / 'run_config.toml')
    checkpoint = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input')
    state = copy.deepcopy(checkpoint.training_state)
    state['models']['world_model']['flow_projector.decoder_speed.bias'] += 1
    state['metrics']['decoder_online_updates'] = 4
    state['metrics']['online_pilot_progress']['online'].append({'episode':81})
    state['metrics']['online_episodes_completed'] = state['next_environment_episode'] = 82
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=82, training_state=state)
    assert migration.validate_decoder_continuation(output, config, 'input', 'pretrained') == manifest


def test_initial_actor_tamper_rejected_even_if_rehashed(tmp_path):
    source, _, _, calibration, _ = fixture(tmp_path)
    output = tmp_path / 'target'
    manifest = migration.migrate(source, calibration, output)
    path = output / 'continuation_checkpoint.pt'
    raw = torch.load(path, weights_only=True)
    raw['training_state']['models']['actor']['w'] += 1
    torch.save(raw, path)
    manifest['frozen_artifacts']['continuation_checkpoint.pt'] = migration.sha256_file(path)
    migration.atomic_json(output / 'decoder_migration.json', manifest)
    with pytest.raises(ValueError, match="checkpoint tensor"):
        migration.validate_decoder_continuation(output, load_config(output / 'run_config.toml'), 'input', 'pretrained')
