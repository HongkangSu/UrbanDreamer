from dataclasses import replace
import hashlib
import json

import pytest
import torch

from test_complete_pretrain_runner import fixture_artifacts, migration_fixture
from test_offline_data import setup, write_episode
from test_urban_dreamer_config import _write_config
from tools import run_complete_pretrain as runner
from tools.train_codec_ready import freeze_sources
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.traffic.simulation_settings import physics_identity


def with_model(setup, model, **settings):
    config, prepared, topology = setup
    return replace(config, simulation=replace(config.simulation,
        traffic_model=model, **settings)), prepared, topology


def source_split(train, validation=None):
    validation = train if validation is None else validation
    return {partition: [dict(path=str(write_episode(source, index)),
        demand_group=partition + '-group', method='random',
        root=str(source[0].data.archive_roots[0]), metadata_compatible=True)]
        for index, (partition, source) in enumerate((('train', train), ('validation', validation)))}


def frozen(split, target):
    config, prepared, topology = target
    return freeze_sources(split, prepared, config, topology.network)


def signed_artifacts(tmp_path, *, codec_physics=None, cache_physics=None):
    codec_dir, cache_path, result, checkpoint, cache = fixture_artifacts(tmp_path)
    manifest_path = codec_dir / 'codec_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if codec_physics is not None:
        manifest['physics_identity'] = codec_physics
    manifest.pop('identity')
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    result['manifest_identity'] = checkpoint['manifest_identity'] = manifest['identity']
    (codec_dir / 'result.json').write_text(json.dumps(result))
    torch.save(checkpoint, codec_dir / 'codec_latest.pt')
    if cache_physics is not None:
        cache['contract']['physics_identity'] = cache_physics
    cache['identity'] = _identity({key: value for key, value in cache.items() if key != 'identity'})
    cache_path.write_text(json.dumps(cache))
    return codec_dir, cache_path


def readiness(artifacts, **kwargs):
    return runner.prerequisite_state(*artifacts, expected_train=1, expected_validation=1, **kwargs)


def test_codec_freeze_preserves_legacy_micro_manifest_contract(setup):
    contract = frozen(source_split(setup), setup)
    assert 'physics_identity' not in contract
    assert len(contract['entries']['train']) == len(contract['entries']['validation']) == 1


def test_codec_freeze_retains_exact_meso_physics_for_both_partitions(setup):
    meso = with_model(setup, 'meso', meso_edge_length_m=120.)
    contract = frozen(source_split(meso), meso)
    assert contract['physics_identity'] == physics_identity(meso[0].simulation)
    assert contract['physics_identity']['meso_edge_length_m'] == 120.


@pytest.mark.parametrize('source,target', [('micro', 'meso'), ('meso', 'micro')])
def test_codec_freeze_rejects_other_traffic_model(setup, source, target):
    sources, wanted = with_model(setup, source), with_model(setup, target)
    with pytest.raises(ValueError, match='traffic physics'):
        frozen(source_split(sources), wanted)


@pytest.mark.parametrize('model', ['micro', 'meso'])
def test_codec_freeze_rejects_mixed_partition_models(setup, model):
    micro, meso = with_model(setup, 'micro'), with_model(setup, 'meso')
    with pytest.raises(ValueError, match='traffic physics'):
        frozen(source_split(micro, meso), with_model(setup, model))


def test_codec_freeze_rejects_different_meso_segment_length(setup):
    source = with_model(setup, 'meso', meso_edge_length_m=120.)
    with pytest.raises(ValueError, match='traffic physics'):
        frozen(source_split(source), with_model(setup, 'meso'))


def test_legacy_micro_codec_and_cache_remain_accepted_only_for_micro(tmp_path):
    artifacts = signed_artifacts(tmp_path)
    assert readiness(artifacts)['ready']
    assert readiness(artifacts, expected_simulation={})['ready']
    with pytest.raises(ValueError, match='traffic physics differ from the requested run'):
        readiness(artifacts, expected_simulation={'traffic_model': 'meso'})


def test_micro_codec_proof_is_rejected_for_meso_even_while_cache_is_pending(tmp_path):
    codec_dir, cache_path = signed_artifacts(tmp_path)
    cache_path.unlink()
    with pytest.raises(ValueError, match='accepted codec traffic physics differ'):
        readiness((codec_dir, cache_path), expected_simulation={'traffic_model': 'meso'})


@pytest.mark.parametrize('codec_model,cache_model', [('micro', 'meso'), ('meso', 'micro')])
def test_codec_acceptance_and_cache_cannot_mix_models(tmp_path, codec_model, cache_model):
    artifacts = signed_artifacts(tmp_path,
        codec_physics=physics_identity({'traffic_model': codec_model}),
        cache_physics=physics_identity({'traffic_model': cache_model}))
    with pytest.raises(ValueError, match='different traffic physics contracts'):
        readiness(artifacts)


def test_matching_meso_acceptance_is_bound_to_requested_meso_settings(tmp_path):
    physics = physics_identity({'traffic_model': 'meso', 'meso_edge_length_m': 120.})
    artifacts = signed_artifacts(tmp_path, codec_physics=physics, cache_physics=physics)
    assert readiness(artifacts, expected_simulation=physics)['ready']
    with pytest.raises(ValueError, match='traffic physics differ from the requested run'):
        readiness(artifacts, expected_simulation={'traffic_model': 'meso'})


def test_codec_and_cache_meso_parameters_must_match(tmp_path):
    artifacts = signed_artifacts(tmp_path,
        codec_physics=physics_identity({'traffic_model': 'meso', 'meso_edge_length_m': 120.}),
        cache_physics=physics_identity({'traffic_model': 'meso'}))
    with pytest.raises(ValueError, match='different traffic physics contracts'):
        readiness(artifacts)


@pytest.mark.parametrize('artifact', ['codec', 'cache'])
def test_incomplete_meso_readiness_identity_is_rejected(tmp_path, artifact):
    physics = physics_identity({'traffic_model': 'meso'})
    incomplete = {key: value for key, value in physics.items() if key != 'vehicle_position'}
    artifacts = signed_artifacts(tmp_path,
        codec_physics=incomplete if artifact == 'codec' else physics,
        cache_physics=incomplete if artifact == 'cache' else physics)
    with pytest.raises(ValueError, match='physics_identity is incomplete'):
        readiness(artifacts)


def test_meso_main_rejects_micro_ready_proof_before_training_outputs(tmp_path, monkeypatch):
    artifacts = signed_artifacts(tmp_path)
    config = _write_config(tmp_path / 'meso.toml')
    config.write_text(config.read_text().replace('[simulation]', '[simulation]\ntraffic_model="meso"'))
    original = runner.prerequisite_state
    monkeypatch.setattr(runner, 'prerequisite_state', lambda *args, **kwargs:
        original(*args, expected_train=1, expected_validation=1, **kwargs))
    output = tmp_path / 'must-not-be-created'
    with pytest.raises(ValueError, match='traffic physics differ from the requested run'):
        runner.main(['--config', str(config), '--codec-dir', str(artifacts[0]),
            '--cache-manifest', str(artifacts[1]), '--output-dir', str(output), '--device', 'cpu'])
    assert not output.exists()
    assert not output.with_suffix('.status.json').exists()


@pytest.mark.parametrize('target_model', ['micro', 'meso'])
def test_old_micro_migration_snapshot_cannot_be_promoted_to_meso(tmp_path, target_model):
    trainer, checkpoint, _, snapshot = migration_fixture(tmp_path)
    snapshot['simulation'] = {key: value for key, value in snapshot['simulation'].items()
        if key != 'traffic_model' and not key.startswith('meso_')}
    (checkpoint.parent.parent / 'resolved_config.json').write_text(json.dumps(snapshot))
    trainer.config = replace(trainer.config, simulation=replace(
        trainer.config.simulation, traffic_model=target_model))
    if target_model == 'micro':
        assert runner.migrate_correct_scope_checkpoint(trainer, checkpoint)['codec_revalidation_required']
        assert trainer.loaded['metrics']['codec_ready'] == 0.
    else:
        with pytest.raises(ValueError, match='simulation traffic physics contract'):
            runner.migrate_correct_scope_checkpoint(trainer, checkpoint)
        assert not hasattr(trainer, 'loaded')


def test_migration_requires_a_complete_physical_configuration_snapshot(tmp_path):
    trainer, checkpoint, _, snapshot = migration_fixture(tmp_path)
    snapshot['simulation'] = None
    (checkpoint.parent.parent / 'resolved_config.json').write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match='simulation contract differs'):
        runner.migrate_correct_scope_checkpoint(trainer, checkpoint)
    assert not hasattr(trainer, 'loaded')
