from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_offline_data import setup, write_episode, rewrite_npz
from test_urban_dreamer_config import _write_config
from urban_dreamer.learning.episode_archive import (
    EpisodeArchive, archive_physics_identity, read_episode_archive,
)
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, _identity, archive_content_hash
from urban_dreamer.learning.pretrain_cache import (
    CompactPretrainDataset, compact_arrays, read_hash_source,
)
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.workflow.config import ConfigError, load_config


def configured(tmp_path, setting=''):
    path = _write_config(tmp_path / 'config.toml', routing_mode='next_hop')
    path.write_text(path.read_text().replace('[simulation]', '[simulation]\n' + setting))
    return load_config(path)


def meso_setup(setup, **changes):
    config, prepared, topology = setup
    return replace(config, simulation=replace(config.simulation, traffic_model='meso', **changes)), prepared, topology


def dataset(setup, name):
    config, prepared, topology = setup
    return OfflineEpisodeDataset(config, prepared, topology, 2,
        manifest_path=config.run.output_dir / name)


def cache(setup, tmp_path):
    config, prepared, topology = setup
    topology.network.capacities = np.ones(3, dtype=np.float32)
    source_path = write_episode(setup)
    data, identity = read_hash_source(source_path)
    cache_path = tmp_path / 'compact.npz'
    np.savez_compressed(cache_path, **compact_arrays(data))
    entry = dict(path=str(source_path), **identity, split='train', episode=0,
        windows=3, events=6, population=1, completed=False,
        cache_path=str(cache_path), cache_content_hash=archive_content_hash(cache_path))
    physical = {key: config.resolved_snapshot()['simulation'][key] for key in (
        'vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds',
        'incident_end_seconds', 'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')}
    manifest = dict(schema_version=1, status='complete', contract=dict(
        city=config.inputs.network_name, input_hashes=prepared.input_hashes,
        physical=physical, physics_identity=physics_identity(config.simulation)),
        entries=[entry], sources=[entry], duplicate_files=0)
    path = tmp_path / 'cache_manifest.json'
    save_manifest(path, manifest)
    return path, entry


def save_manifest(path, manifest):
    manifest['identity'] = _identity({key: value for key, value in manifest.items() if key != 'identity'})
    path.write_text(json.dumps(manifest))


def test_legacy_micro_fingerprint_is_byte_compatible(tmp_path):
    config = configured(tmp_path)
    assert config.simulation.traffic_model == 'micro'
    assert config.simulation.meso_edge_length_m == 98.
    # Current execution defaults already moved to FP16 in d255ac3.
    assert config.semantic_fingerprint == '1eace275bc3805321e9c71b53a5e20beb0fd2cc99f5057168858590329ec94a5'


def test_meso_fingerprint_binds_model_length_and_state_semantics(tmp_path):
    micro = configured(tmp_path)
    meso = configured(tmp_path, 'traffic_model="meso"')
    alternate = configured(tmp_path, 'traffic_model="meso"\nmeso_edge_length_m=120.0')
    assert len({micro.semantic_fingerprint, meso.semantic_fingerprint, alternate.semantic_fingerprint}) == 3
    identity = physics_identity(meso.simulation)
    assert identity['junction_state'] == 'zero_internal_distance'
    assert identity['vehicle_position'] == 'interpolated_external_edge'
    assert identity['meso_junction_control'] is True
    assert identity['meso_lane_queue'] is identity['meso_multi_queue'] is False
    assert meso.resolved_snapshot()['simulation']['traffic_model'] == 'meso'


@pytest.mark.parametrize('setting', [
    'traffic_model="wrong"', 'traffic_model=true', 'traffic_model=[]',
    'meso_junction_control=1', 'meso_lane_queue="false"', 'meso_multi_queue=0',
    'meso_interpolate_positions="true"', 'meso_edge_length_m=true',
    'meso_edge_length_m=0', 'meso_edge_length_m=-1', 'meso_edge_length_m=nan',
    'meso_edge_length_m=inf',
])
def test_physics_config_is_strictly_typed(tmp_path, setting):
    with pytest.raises(ConfigError, match='simulation'):
        configured(tmp_path, setting)


@pytest.mark.parametrize('setting', [
    'meso_junction_control=false', 'meso_lane_queue=true', 'meso_multi_queue=true',
    'meso_interpolate_positions=false', 'session_protocol="microstep"',
])
def test_meso_requires_verified_event_profile(tmp_path, setting):
    with pytest.raises(ConfigError, match='simulation'):
        configured(tmp_path, 'traffic_model="meso"\n' + setting)


@pytest.mark.parametrize('argument', [
    '--mesosim', '--mesosim=false', '--meso-edgelength=50',
    '--meso-junction-control=false', '--meso-lane-queue', '--meso-multi-queue=true',
    '--meso-interpolate-pos=false', '--no-internal-links=true',
])
@pytest.mark.parametrize('model', ['micro', 'meso'])
def test_raw_extra_args_cannot_override_physics(tmp_path, argument, model):
    path = _write_config(tmp_path / 'config.toml')
    text = path.read_text().replace('[simulation]', f'[simulation]\ntraffic_model="{model}"')
    text = text.replace('["--collision.action", "warn"]', json.dumps([argument]))
    path.write_text(text)
    with pytest.raises(ConfigError, match='reserved SUMO option'):
        load_config(path)


def test_old_archive_missing_physics_fields_is_micro(setup):
    path = write_episode(setup)
    def strip(arrays):
        summary = json.loads(arrays['summary_json'].tobytes())
        simulation = summary['metadata']['config']['simulation']
        for key in list(simulation):
            if key == 'traffic_model' or key.startswith('meso_'):
                simulation.pop(key)
        arrays['summary_json'] = np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8)
    rewrite_npz(path, strip)
    assert archive_physics_identity(read_episode_archive(path)['metadata']) == {'traffic_model': 'micro'}
    assert dataset(setup, 'old.json').has_samples
    with pytest.raises(ValueError, match='traffic physics'):
        dataset(meso_setup(setup), 'meso_reject_old.json')


def test_frozen_meso_snapshot_cannot_relabel_old_unmarked_micro_metadata():
    metadata = {'session_protocol': 'event'}
    assert archive_physics_identity(metadata, simulation={}) == {'traffic_model': 'micro'}
    with pytest.raises(ValueError, match='physics provenance mismatch'):
        archive_physics_identity(metadata, simulation={'traffic_model': 'meso'})


def test_archive_model_label_can_accompany_custom_complete_identity():
    identity = physics_identity({'traffic_model': 'meso', 'meso_edge_length_m': 120.})
    metadata = dict(traffic_model='meso', physics_identity=identity, session_protocol='event')
    assert archive_physics_identity(metadata) == identity
    metadata['meso_edge_length_m'] = 98.
    with pytest.raises(ValueError, match='physics provenance mismatch'):
        archive_physics_identity(metadata)


def test_import_rejects_mixed_models_and_meso_parameters(setup):
    meso = meso_setup(setup)
    path = write_episode(meso)
    assert archive_physics_identity(read_episode_archive(path)['metadata']) == physics_identity(meso[0].simulation)
    assert dataset(meso, 'meso.json').has_samples
    with pytest.raises(ValueError, match='traffic physics'):
        dataset(setup, 'micro_reject_meso.json')
    with pytest.raises(ValueError, match='traffic physics'):
        dataset(meso_setup(setup, meso_edge_length_m=120.), 'length_reject.json')
    write_episode(setup, 1)
    with pytest.raises(ValueError, match='traffic physics'):
        dataset(meso, 'mixed_reject.json')


def test_archive_rejects_relabeling_and_internal_junction_state(setup):
    path = write_episode(meso_setup(setup))
    data = read_episode_archive(path)
    metadata = dict(data['metadata'], traffic_model='micro')
    with pytest.raises(ValueError, match='physics provenance mismatch'):
        EpisodeArchive(metadata)
    archive = EpisodeArchive(data['metadata'])
    state = data['states'][0]
    state['flow'] = data['flow'][0]
    state['vehicles'][0]['in_junction'] = True
    with pytest.raises(ValueError, match='zero internal junction state'):
        archive.start(state)


def test_compact_cache_keeps_meso_identity_and_legacy_cache_rejects_meso(setup, tmp_path):
    meso = meso_setup(setup)
    path, entry = cache(meso, tmp_path)
    with np.load(entry['cache_path'], allow_pickle=False) as body:
        assert json.loads(body['metadata_json'].tobytes())['physics_identity'] == physics_identity(meso[0].simulation)
    compact = CompactPretrainDataset(*meso, 2, cache_manifest_path=path)
    assert len(list(compact.iter_codec_batches(2))) == 2
    assert compact.batch_from_starts(0, [0]).flow.shape == (1, 3, 3, 8)
    with pytest.raises(ValueError, match='traffic physics'):
        CompactPretrainDataset(*setup, 2, cache_manifest_path=path)
    with pytest.raises(ValueError, match='traffic physics'):
        CompactPretrainDataset(*meso_setup(setup, meso_edge_length_m=120.), 2, cache_manifest_path=path)
    manifest = json.loads(path.read_text())
    manifest['contract'].pop('physics_identity')
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match='traffic physics'):
        CompactPretrainDataset(*meso, 2, cache_manifest_path=path)


def test_codec_flow_only_cannot_bypass_compact_body_physics(setup, tmp_path):
    meso = meso_setup(setup)
    path, entry = cache(meso, tmp_path)
    def change(arrays):
        metadata = json.loads(arrays['metadata_json'].tobytes())
        metadata['physics_identity'] = {'traffic_model': 'micro'}
        arrays['metadata_json'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    rewrite_npz(Path(entry['cache_path']), change)
    manifest = json.loads(path.read_text())
    manifest['entries'][0]['cache_content_hash'] = archive_content_hash(Path(entry['cache_path']))
    save_manifest(path, manifest)
    compact = CompactPretrainDataset(*meso, 2, cache_manifest_path=path)
    with pytest.raises(ValueError, match='compact episode traffic physics'):
        next(compact.iter_codec_batches(2))


def test_cache_builder_plan_binds_meso_physics(tmp_path, monkeypatch):
    from tools import build_pretrain_cache

    census, pilot = tmp_path / 'census', tmp_path / 'pilot'
    census.mkdir()
    pilot.mkdir()
    (census / 'census.json').write_text(json.dumps(dict(
        verified_duplicate_groups=[], frozen_at='test', unique_trajectories=0)))
    (census / 'file_rows.json').write_text(json.dumps({'rows': []}))
    (census / 'archive_paths.json').write_text('[]')
    (pilot / 'split.json').write_text(json.dumps({'holdout_groups': []}))
    monkeypatch.setattr(build_pretrain_cache, 'validate_prepared_inputs',
        lambda _: SimpleNamespace(input_hashes={'network': 'fixture'}))
    micro = configured(tmp_path)
    old_plan = build_pretrain_cache.make_plan(census, pilot, micro.source_path)
    assert 'physics_identity' not in old_plan['contract']
    meso = configured(tmp_path, 'traffic_model="meso"')
    plan = build_pretrain_cache.make_plan(census, pilot, meso.source_path)
    assert plan['contract']['physics_identity'] == physics_identity(meso.simulation)
    assert plan['identity'] != old_plan['identity']
