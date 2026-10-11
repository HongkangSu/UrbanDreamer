from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import migrate_calibrated_meso_noincident as migration
from tools.migrate_pretrained_meso import model_hashes, transfer_models
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


ROOT = Path(__file__).resolve().parents[1]


def _configs(tmp_path):
    source = load_config(ROOT / 'configs/ingolstadt_custom.toml')
    source = replace(source, simulation=replace(source.simulation,
        incident_start_seconds=500., incident_end_seconds=1500.))
    target = replace(source, run=replace(source.run, output_dir=tmp_path, device='cpu'),
        simulation=replace(source.simulation,
            incident_start_seconds=1_000_000., incident_end_seconds=1_001_000.),
        data=replace(source.data, archive_roots=(tmp_path / 'new-corpus',)))
    target = replace(target, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(target, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    return source, target


def test_config_contract_accepts_only_disabled_incident_and_new_runtime_sources(tmp_path):
    source, target = _configs(tmp_path)
    migration.validate_noincident_config(source, target)
    alterations = (
        replace(target, method=replace(target.method, sp_prior_weight=1.)),
        replace(target, simulation=replace(target.simulation, meso_lane_queue=True)),
        replace(target, simulation=replace(target.simulation, incident_start_seconds=1000.)),
        replace(target, resources=replace(target.resources, online_world_model_batch_size=8)),
        replace(target, run=replace(target.run, seed=99)),
        replace(target, data=replace(target.data, archive_roots=())),
    )
    for changed in alterations:
        with pytest.raises(ValueError):
            migration.validate_noincident_config(source, changed)


def test_fresh_weight_transfer_has_exact_models_without_source_training_state():
    names = ('world_model', 'actor', 'latent_critic', 'event_critic')
    class Trainer:
        def __init__(self):
            for name in names:
                setattr(self, name, torch.nn.Linear(2, 2))
            self.optimizers = {name: torch.optim.Adam(getattr(self, name).parameters()) for name in names}
            self.metrics = {'online_episodes_completed': 0, 'world_model_initialization_epochs_completed': 0}
            self.next_environment_episode = 0
        def state_dict(self):
            return {'models': {name: getattr(self, name).state_dict() for name in names}}
    source, target = Trainer(), Trainer()
    frozen = source.state_dict()['models']
    transfer_models(target, SimpleNamespace(training_state={'models': frozen}))
    assert model_hashes(target.state_dict()['models']) == model_hashes(frozen)
    assert all(not optimizer.state for optimizer in target.optimizers.values())
    assert target.metrics['online_episodes_completed'] == 0
    assert target.next_environment_episode == 0


def test_manifest_binds_frozen_initializations_corpus_and_codec_proof(tmp_path, monkeypatch):
    source_config, target_config = _configs(tmp_path)
    source_dir = tmp_path / 'old'
    source_dir.mkdir()
    source_checkpoint = source_dir / 'initialization_checkpoint.pt'
    source_checkpoint.write_bytes(b'calibrated-source')
    source_meso_manifest = source_dir / 'meso_initialization.json'
    source_meso_manifest.write_text('{}')
    output = tmp_path / 'new'
    output.mkdir()
    target_checkpoint = output / 'initialization_checkpoint.pt'
    target_checkpoint.write_bytes(b'new-initialization')
    for name, identity in (('offline_manifest.json', 'train-id'),
                           ('codec_validation_manifest.json', 'validation-id')):
        (output / name).write_text(json.dumps({'identity': identity}))
    models = {'world_model': {'x': torch.ones(1)},
              'actor': {'x': torch.ones(1)},
              'latent_critic': {'x': torch.ones(1)},
              'event_critic': {'x': torch.ones(1)}}
    source = SimpleNamespace(path=source_checkpoint, training_state={'models': models})
    state = dict(models=models, optimizers={'actor': {'state': {}}},
        replay={'online': {'episodes': []}}, next_environment_episode=0,
        metrics=dict(online_episodes_completed=0, world_model_initialization_epochs_completed=0,
            codec_updates=0, codec_ready=1,
            codec_validation_updates=0, codec_validation_manifest_identity='validation-id'))
    monkeypatch.setattr(migration, '_source_initialization',
                        lambda directory, input_hash: (source, source_config, source_meso_manifest))
    monkeypatch.setattr(migration.torch, 'load', lambda *args, **kwargs: {'training_state': state})
    monkeypatch.setattr(migration, '_validate_loaded_payload', lambda **kwargs: SimpleNamespace(training_state=state))
    import libsumo
    manifest = dict(schema_version=1, kind='calibrated_meso_to_fresh_noincident_online',
        source_output=str(source_dir), target_output=str(output),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=target_config.semantic_fingerprint,
        input_hash='inputs', source_initialization_sha256=migration.sha256_file(source_checkpoint),
        source_meso_manifest_sha256=migration.sha256_file(source_meso_manifest),
        source_model_hashes=model_hashes(models), model_hashes=model_hashes(models),
        target_initialization_sha256=migration.sha256_file(target_checkpoint),
        training_manifest_sha256=migration.sha256_file(output / 'offline_manifest.json'),
        validation_manifest_sha256=migration.sha256_file(output / 'codec_validation_manifest.json'),
        training_manifest_identity='train-id', validation_manifest_identity='validation-id',
        codec_validation={'codec_ready': True}, target_engine_version=libsumo.__version__)
    migration.validate_calibrated_noincident_manifest(manifest, target_config, output, 'inputs')
    source_checkpoint.write_bytes(b'changed-source')
    with pytest.raises(ValueError, match='source_initialization_sha256'):
        migration.validate_calibrated_noincident_manifest(manifest, target_config, output, 'inputs')
    source_checkpoint.write_bytes(b'calibrated-source')
    target_checkpoint.write_bytes(b'changed-target')
    with pytest.raises(ValueError, match='target_initialization_sha256'):
        migration.validate_calibrated_noincident_manifest(manifest, target_config, output, 'inputs')
    target_checkpoint.write_bytes(b'new-initialization')
    state['metrics']['codec_ready'] = 0
    with pytest.raises(ValueError, match='codec proof'):
        migration.validate_calibrated_noincident_manifest(manifest, target_config, output, 'inputs')


def test_zero_round_weight_transfer_allows_corrected_entropy_objective(tmp_path):
    source, target = _configs(tmp_path)
    source = replace(source, method=replace(source.method, actor_entropy_mode="legacy"))
    target = replace(target, method=replace(target.method, actor_entropy_mode="population"))
    migration.validate_noincident_config(source, target)
