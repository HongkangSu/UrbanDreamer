from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import zipfile

import numpy as np
import pytest

from urban_dreamer.learning.episode_archive import EpisodeArchive, read_episode_archive, to_replay_episode
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset, archive_content_hash
from urban_dreamer.traffic.environment import NextHopDecision, WindowResult
from urban_dreamer.traffic.topology import VehicleDemand
from urban_dreamer.workflow.config import DataConfig, load_config, ConfigError


@pytest.fixture
def setup(tmp_path):
    config = load_config(Path(__file__).parents[1] / 'configs/toy_cross_bottleneck.toml')
    # This fixture exercises historical microscopic archives and manifests.
    # Shipped profiles now select MESO; keep legacy compatibility explicit.
    config = replace(config, simulation=replace(config.simulation, traffic_model='micro', backend='traci'),
                     run=replace(config.run, output_dir=tmp_path / 'output'),
                     data=DataConfig((tmp_path / 'archives',), 2))
    network = SimpleNamespace(edge_ids=('0', '1', '2'), edge_count=3,
        edge_to_index={'0': 0, '1': 1, '2': 2}, adjacency_list=((1, 2), (0, 2), (0, 1)))
    topology = SimpleNamespace(network=network, origins=('0',), destinations=('2',),
                               vehicles=(VehicleDemand('reference', 0, 0, 0.),))
    prepared = SimpleNamespace(input_hashes={'network': 'fixture'})
    return config, prepared, topology


def write_episode(setup, index=0, *, population=1, completed=False, windows=3, embedded_config=True):
    config, prepared, topology = setup
    name = f'v{index}'
    meta = dict(city=config.inputs.network_name, method='random', seed=index, episode=index,
        phase='train', input_hashes=prepared.input_hashes, total_vehicle_count=population,
        session_protocol='event', edge_ids=list(topology.network.edge_ids), step_seconds=1.,
        demand_law='independent_od_poisson', demand_duration_seconds=config.simulation.demand_duration_seconds,
        od_rates=[dict(origin_index=0, destination_index=0, rate_veh_per_second=1 / config.simulation.demand_duration_seconds)],
        demand_realization=[dict(vehicle_id=name, origin_index=0, destination_index=0, departure_second=0.)][:population])
    if embedded_config:
        meta['config'] = config.resolved_snapshot()
    def state(time, terminal=False):
        return dict(time=time, done=terminal, flow=np.full((3, 8), time, dtype=np.float32),
            vehicles=[dict(vehicle_id=name, current_edge=0, destination_edge=2,
                departure_time=0., remaining_distance=10., departed=time > 0, arrived=terminal,
                committed_next_edge=1, in_junction=False)][:population])
    archive = EpisodeArchive(meta)
    archive.start(state(0.))
    for window in range(windows):
        start = float(30 * window)
        events = (NextHopDecision(name, 0, 1, start + 1.),
                  NextHopDecision(name, 1, 2, start + 11.)) if population else ()
        counts = np.array([0, population, population])
        archive.append(WindowResult(events, counts, np.zeros(3), population * 30.,
            state(start + 30., completed and window == windows - 1), start, start + 30.))
    path = config.data.archive_roots[0] / 'random/episodes/train' / f'episode_{index:06d}.npz'
    archive.save(path, metrics={}, travel_times={name: 60.} if completed and population else {},
                 completed=completed, deadline_seconds=999.)
    return path


def dataset(setup, **kwargs):
    return OfflineEpisodeDataset(*setup, sequence_length=2, **kwargs)


def rewrite_npz(path, update):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    update(arrays)
    np.savez_compressed(path, **arrays)


def test_old_archive_events_destination_context_and_censoring(setup):
    path = write_episode(setup)
    def old(arrays):
        summary = json.loads(arrays['summary_json'].tobytes())
        summary['schema_version'] = 1
        arrays['summary_json'] = np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8)
        events = json.loads(arrays['decisions_json'].tobytes())
        for event in events:
            event.pop('destination_edge')
            event.pop('event_index')
        arrays['decisions_json'] = np.frombuffer(json.dumps(events).encode(), dtype=np.uint8)
    rewrite_npz(path, old)
    d = dataset(setup)
    batch = next(d.iter_epoch(2, np.random.default_rng(0)))
    assert batch.action_events is not None
    assert batch.action_events.destination_edge.tolist() == [2] * 8
    assert not batch.episode_complete.any()
    assert batch.total_travel_minutes.tolist() == [1.5, 1.5]
    assert d.summary['events'] == 6
    rows = to_replay_episode(path, expected_session_protocol='event', target_deadline_seconds=60.)['transitions']
    assert rows[1].context[0] == .5
    assert len(rows[0].action_events) == 2
    assert not rows[-1].terminal


def test_empty_population_and_complete_supervision(setup):
    write_episode(setup, population=0, completed=True, windows=1)
    d = dataset(setup)
    b = next(d.iter_epoch(5, np.random.default_rng(0)))
    assert b.total_vehicle_count.tolist() == [0]
    assert b.episode_complete.tolist() == [True]
    assert b.total_travel_minutes.tolist() == [0.]
    assert b.action_events.current_edge.numel() == 0


def test_duplicate_paths_cache_bound_full_epoch_and_tail(setup):
    paths = [write_episode(setup, index) for index in range(4)]
    duplicate = paths[0].parent / 'copy.npz'
    duplicate.write_bytes(paths[0].read_bytes())
    d = dataset(setup)
    batches = list(d.iter_epoch(3, np.random.default_rng(7)))
    visited = [(int(e), int(s)) for b in batches for e, s in zip(b.episode, b.start_index)]
    assert sorted(visited) == [(episode, start) for episode in range(4) for start in range(2)]
    assert d.summary['duplicate_files'] == 1
    assert d.summary['imported_episodes'] == 4
    assert d.summary['cached_episodes'] <= 2
    assert all(len(b.episode) == 2 for b in batches)
    assert 'transitions' not in json.dumps(d.state_dict())


def test_mid_epoch_restore_recompression_and_append_isolation(setup):
    paths = [write_episode(setup, index) for index in range(3)]
    d = dataset(setup)
    rng = np.random.default_rng(3)
    stream = d.iter_epoch(1, rng)
    next(stream)
    state = d.state_dict()
    rng_state = rng.bit_generator.state
    expected = [(b.episode.tolist(), b.start_index.tolist()) for b in stream]
    before = archive_content_hash(paths[0])
    with zipfile.ZipFile(paths[0]) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(paths[0], 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, content in reversed(list(members.items())):
            archive.writestr(name, content)
    assert archive_content_hash(paths[0]) == before
    write_episode(setup, 99)
    restored = dataset(setup)
    assert restored.summary['imported_episodes'] == 3
    restored.load_state_dict(state)
    restored_rng = np.random.default_rng()
    restored_rng.bit_generator.state = rng_state
    assert [(b.episode.tolist(), b.start_index.tolist())
            for b in restored.iter_epoch(1, restored_rng)] == expected


def test_content_mutation_and_count_mismatch_rejected(setup):
    path = write_episode(setup)
    d = dataset(setup)
    rewrite_npz(path, lambda a: a['action_counts'].__setitem__((0, 1), 99))
    with pytest.raises(ValueError, match='content changed'):
        d.verify_integrity()
    other = setup[0].run.output_dir / 'other_manifest.json'
    with pytest.raises(ValueError, match='reconstruction'):
        dataset(setup, manifest_path=other)


def test_frozen_collection_snapshot_fallback_and_physical_rejection(setup):
    path = write_episode(setup, embedded_config=False)
    root = setup[0].data.archive_roots[0]
    (root / 'random/protocol.json').write_text(json.dumps({'config': setup[0].resolved_snapshot()}))
    assert dataset(setup).has_samples
    wrong = replace(setup[0], simulation=replace(setup[0].simulation, vehicle_max_speed_mps=39.))
    with pytest.raises(ValueError, match='vehicle_max_speed_mps'):
        dataset((wrong, *setup[1:]), manifest_path=setup[0].run.output_dir / 'bad.json')


def test_explicit_archive_empty_list_and_config_defaults(setup, tmp_path):
    write_episode(setup)
    d = dataset(setup, archive_paths=[], manifest_path=tmp_path / 'warmup.json')
    assert not d.has_samples
    assert not d.manifest_path.exists()
    example = Path(__file__).parents[1] / 'configs/mainline.example.toml'
    parser_defaults = tmp_path / 'parser_defaults.toml'
    # Mainline execution requires a source; parser defaults remain usable by
    # standalone collectors and archived configurations without a data table.
    parser_defaults.write_text(example.read_text().split('\n[data]\n')[0])
    base = load_config(parser_defaults)
    assert base.data == DataConfig()
    assert base.simulation.session_protocol == 'event'
    assert base.simulation.observation_step_seconds == 1.
    for extra in ('[data]\ncache_episodes = 3', '[data]\narchive_roots = "bad"'):
        file = tmp_path / 'bad.toml'
        file.write_text(base.source_path.read_text() + '\n' + extra)
        with pytest.raises(ConfigError, match='data'):
            load_config(file)


def test_empty_configured_root_does_not_freeze_manifest_or_create_output(setup):
    config = setup[0]
    config.data.archive_roots[0].mkdir(parents=True)
    d = dataset(setup)
    assert not d.has_samples
    assert d.summary['discovered_files'] == 0
    assert not d.manifest_path.exists()
    assert not config.run.output_dir.exists()
    # Adding the first compatible file after an empty-source error can start a
    # fresh run; no persisted empty manifest silently excludes it.
    write_episode(setup)
    reopened = dataset(setup)
    assert reopened.has_samples
    assert reopened.summary['imported_episodes'] == 1
    assert reopened.manifest_path.is_file()


def test_raw_offline_stream_preserves_observed_arrival_metadata(setup):
    write_episode(setup,completed=True,windows=2)
    data=dataset(setup)
    batch=next(data.iter_epoch(1,np.random.default_rng(0)))
    assert batch.arrival_times==({'v0':60.},)
    assert batch.boundary_vehicles is not None
    assert batch.boundary_vehicles[0][-1]==()


def test_frozen_old_training_scope_does_not_reject_compatible_raw_physics(setup):
    path=write_episode(setup,embedded_config=False)
    config=setup[0]
    root=config.data.archive_roots[0]
    source=root/'source';(source/'configs').mkdir(parents=True)
    frozen=source/'configs'/f'{config.inputs.network_name}.toml'
    frozen.write_text(config.source_path.read_text().replace('traffic_model = "meso"', 'traffic_model = "micro"').replace(
        'transition_volume_scope = "previous_state_excluding_current_births"',
        'transition_volume_scope = "initial_cohort"'))
    (root/'manifest.json').write_text(json.dumps({'source_snapshot':str(source)}))
    before=frozen.read_bytes()
    with pytest.raises(ConfigError,match='transition_volume_scope'):
        load_config(frozen)
    assert dataset(setup).has_samples
    assert frozen.read_bytes()==before
    assert read_episode_archive(path)['source']=='real_sumo'


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('city', 'other', 'city mismatch'),
    ('phase', 'evaluation', 'training partition'),
    ('step_seconds', 2., 'one second'),
    ('edge_ids', ['2', '1', '0'], 'edge order'),
    ('input_hashes', {'network': 'wrong'}, 'hashes'),
    ('demand_law', 'fixed', 'demand law'),
])
def test_incompatible_archives_report_reason(setup, field, value, message):
    path = write_episode(setup)
    def edit(arrays):
        summary = json.loads(arrays['summary_json'].tobytes())
        summary['metadata'][field] = value
        arrays['summary_json'] = np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8)
    rewrite_npz(path, edit)
    with pytest.raises(ValueError, match=message):
        dataset(setup)


def test_future_destination_and_illegal_event_rejected(setup):
    path = write_episode(setup)
    def edit(arrays):
        events = json.loads(arrays['decisions_json'].tobytes())
        events[0]['next_edge'] = 0
        arrays['decisions_json'] = np.frombuffer(json.dumps(events).encode(), dtype=np.uint8)
    rewrite_npz(path, edit)
    with pytest.raises(ValueError, match='turn is not legal'):
        dataset(setup)


def test_same_trajectory_with_different_provenance_deduplicates(setup):
    path = write_episode(setup)
    copy = path.parent / 'another_method.npz'
    copy.write_bytes(path.read_bytes())
    def edit(arrays):
        summary = json.loads(arrays['summary_json'].tobytes())
        summary['metadata']['method'] = 'dreamer'
        summary['metrics'] = {'extra': 1.}
        arrays['summary_json'] = np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8)
    rewrite_npz(copy, edit)
    d = dataset(setup)
    assert d.summary['imported_episodes'] == 1
    assert d.summary['duplicate_files'] == 1
    rewrite_npz(copy, lambda arrays: arrays['flow'].__setitem__((0, 0, 0), 2.))
    with pytest.raises(ValueError, match='content changed'):
        d.verify_integrity()


def test_context_uses_observed_macro_deadline_for_nonmultiple_buffer(setup):
    config = replace(setup[0], simulation=replace(setup[0].simulation,
        world_model_intervals_per_episode=1, training_completion_buffer_seconds=1))
    setup = (config, *setup[1:])
    path = write_episode(setup)
    d = dataset(setup)
    assert d.target_deadline_seconds == 60.
    assert read_episode_archive(path)['deadline_seconds'] == 999.
    batch = next(d.iter_epoch(2, np.random.default_rng(0)))
    # The requested horizon is31s, but physical observations end at60s.
    # At the30s boundary imported context must therefore be exactly halfway.
    for index, start in enumerate(batch.start_index.tolist()):
        assert batch.context[index, 0, 0].item() == start * .5
        assert batch.context[index, 1, 0].item() == (start + 1) * .5
    assert d.manifest['contract']['target_deadline_seconds'] == 60.


def test_switching_sources_releases_cache_without_changing_sampler(setup, tmp_path):
    import gc
    import weakref
    paths = [write_episode(setup, index) for index in range(4)]
    sources = [dataset(setup, archive_paths=paths[:2], manifest_path=tmp_path / 'offline.json'),
               dataset(setup, archive_paths=paths[2:], manifest_path=tmp_path / 'warmup.json')]
    rng = np.random.default_rng(4)
    for epoch in range(2):
        visited = []
        for source in sources:
            before = [s.state_dict() for s in sources]
            cached_tensor = next((weakref.ref(next(iter(s._cache.values()))['transitions'][0].flow)
                                  for s in sources if s._cache), None)
            for s in sources:
                s.release_cache()
            gc.collect()
            assert sum(s.summary['cached_episodes'] for s in sources) == 0
            assert [s.state_dict() for s in sources] == before
            if cached_tensor is not None:
                assert cached_tensor() is None
            batches = list(source.iter_epoch(3, rng))
            visited.extend((int(e), int(start)) for batch in batches
                           for e, start in zip(batch.episode, batch.start_index))
            assert sum(s.summary['cached_episodes'] for s in sources) <= 2
            assert source.epochs_completed == epoch + 1
        assert sorted(visited) == [(episode, start) for episode in range(4) for start in range(2)]


def test_cross_source_dedup_preserves_files_integrity_and_restore_contract(setup, tmp_path):
    offline_path = write_episode(setup, 0)
    unique_path = write_episode(setup, 1)
    copy = tmp_path / 'episodes/warmup/episode_000000.npz'
    copy.parent.mkdir(parents=True)
    copy.write_bytes(offline_path.read_bytes())
    def provenance(arrays):
        summary = json.loads(arrays['summary_json'].tobytes())
        summary['metadata'].update(method='urban_dreamer', collection_phase='warmup')
        arrays['summary_json'] = np.frombuffer(json.dumps(summary).encode(), dtype=np.uint8)
    rewrite_npz(copy, provenance)
    offline = dataset(setup, archive_paths=[offline_path], manifest_path=tmp_path / 'offline.json')
    excluded = [entry['trajectory_hash'] for entry in offline.manifest['entries']]
    manifest = tmp_path / 'warmup.json'
    warmup = dataset(setup, archive_paths=[copy, unique_path], manifest_path=manifest,
                     excluded_trajectory_hashes=excluded)
    assert warmup.summary['discovered_files'] == 2
    assert warmup.summary['imported_episodes'] == 1
    assert warmup.summary['duplicate_files'] == 1
    assert len(warmup.manifest['files']) == 2
    assert warmup.manifest['contract']['excluded_trajectory_hashes'] == excluded
    rng = np.random.default_rng(7)
    visited = [(int(e), int(start)) for source in (offline, warmup)
               for batch in source.iter_epoch(2, rng) for e, start in zip(batch.episode, batch.start_index)]
    assert sorted(visited) == [(episode, start) for episode in range(2) for start in range(2)]
    # Equivalent repeated exclusions produce the same frozen contract and
    # retain completed sampler progress without storing source episodes.
    restored = dataset(setup, archive_paths=[copy, unique_path], manifest_path=manifest,
                       excluded_trajectory_hashes=excluded * 2)
    restored.load_state_dict(warmup.state_dict())
    assert restored.state_dict() == warmup.state_dict()
    with pytest.raises(ValueError, match='manifest contract mismatch'):
        dataset(setup, archive_paths=[copy, unique_path], manifest_path=manifest)
    rewrite_npz(copy, lambda arrays: arrays['flow'].__setitem__((0, 0, 0), 2.))
    with pytest.raises(ValueError, match='content changed'):
        restored.verify_integrity()
    before = restored.state_dict()
    with pytest.raises(ValueError, match='content changed'):
        list(restored.iter_epoch(2, np.random.default_rng(9)))
    assert restored.state_dict() == before


def test_fully_excluded_warmup_still_tracks_frozen_file(setup, tmp_path):
    path = write_episode(setup)
    digest = archive_content_hash(path, trajectory_only=True)
    d = dataset(setup, archive_paths=[path], manifest_path=tmp_path / 'excluded.json',
                excluded_trajectory_hashes=[digest])
    assert not d.has_samples
    assert d.summary['imported_episodes'] == 0
    assert d.summary['duplicate_files'] == 1
    assert len(d.manifest['files']) == 1
    assert list(d.iter_epoch(1, np.random.default_rng(0))) == []
    rewrite_npz(path, lambda arrays: arrays['flow'].__setitem__((0, 0, 0), 2.))
    with pytest.raises(ValueError, match='content changed'):
        d.verify_integrity()
    with pytest.raises(ValueError, match='content changed'):
        list(d.iter_epoch(1, np.random.default_rng(0)))
    assert d.epochs_completed == 0


def test_epoch_verifies_each_frozen_file_without_double_hashing_selected(setup, monkeypatch):
    from collections import Counter
    from urban_dreamer.learning import offline_data
    path = write_episode(setup)
    copy = path.parent / 'copy.npz'
    copy.write_bytes(path.read_bytes())
    d = dataset(setup)
    calls = []
    original = offline_data.archive_content_hash
    def tracked_hash(path, **kwargs):
        calls.append(Path(path))
        return original(path, **kwargs)
    monkeypatch.setattr(offline_data, 'archive_content_hash', tracked_hash)
    list(d.iter_epoch(1, np.random.default_rng(0)))
    assert Counter(calls) == Counter({path.resolve(): 1, copy.resolve(): 1})
