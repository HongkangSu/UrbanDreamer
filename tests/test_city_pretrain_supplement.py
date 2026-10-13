from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.prepare_city_pretrain_supplement import anchored_holdout, previous_partition, verify_old_rows
from tools import prepare_city_pretrain_supplement as supplement
from tools.prepare_city_pretrain import freeze_committed
from urban_dreamer.workflow.config import load_config


def row(path, group, **changes):
    return dict(path=path, demand_group=group, demand_hash=group, seed=42, episode=1,
                method='ac', crc_trajectory_candidate='crc', metadata_compatible=True, **changes)


def test_old_groups_keep_full_partition_with_new_methods_and_demands():
    old = [row('old/train', 'old-train'), row('old/validation', 'old-validation')]
    latest = old + [row('new/method', 'old-train')] + [row(f'new/{i}', f'g{i}') for i in range(1000)]
    holdout = anchored_holdout(latest, old, {'old-validation'})
    assert 'old-train' not in holdout
    assert 'old-validation' in holdout
    assert holdout == anchored_holdout(list(reversed(latest)), old, {'old-validation'})
    assert 120 < len(holdout) < 180
    # Future membership growth cannot change previous decisions.
    expanded = latest + [row('later', 'a-later-demand')]
    assert anchored_holdout(expanded, old, {'old-validation'}) & {r['demand_group'] for r in latest} == holdout


def test_unseen_group_partition_uses_seeded_hash_threshold_not_rank():
    old = [row('old/train', 'train'), row('old/validation', 'validation')]
    new = [row(str(i), f'new-demand-{i}') for i in range(200)]
    holdout = anchored_holdout(old + new, old, {'validation'}, seed=17)
    for r in new:
        digest = hashlib.sha256(f"city-pretrain:17:{r['demand_group']}".encode()).hexdigest()
        assert (r['demand_group'] in holdout) == (int(digest, 16) / (1 << 256) < .15)
    with pytest.raises(ValueError, match='absent'):
        anchored_holdout(old, old, {'unknown'})


def test_old_identity_accepts_container_compaction_but_rejects_semantic_changes():
    old = [row('old/path', 'demand', source_stamp={'bytes': 100})]
    fresh = deepcopy(old)
    fresh[0]['source_stamp'] = {'bytes': 20}
    verify_old_rows(fresh, old)
    for key in ('demand_group', 'demand_hash', 'seed', 'episode', 'method',
                'crc_trajectory_candidate', 'metadata_compatible'):
        changed = deepcopy(fresh)
        changed[0][key] = 'changed'
        with pytest.raises(ValueError, match='identity changed'):
            verify_old_rows(changed, old)
    with pytest.raises(ValueError, match='discarded'):
        verify_old_rows([], old)


def test_previous_partition_checks_full_cache_not_only_codec_pilot(tmp_path):
    old = [row('old/train-unseen-in-pilot', 'train'), row('old/validation', 'validation')]
    for directory in ('census', 'pilot', 'cache'):
        (tmp_path / directory).mkdir()
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value))
    save('census/file_rows.json', dict(rows=old))
    save('census/census.json', dict(unique_trajectories=2))
    save('census/archive_paths.json', [r['path'] for r in old])
    save('pilot/split.json', dict(seed=42, holdout_groups=['validation'], train=[], validation=[]))
    sources = [{**old[0], 'split': 'train'}, {**old[1], 'split': 'validation'}]
    save('cache/cache_plan.json', dict(sources=sources))
    rows, census, holdout, seed = previous_partition(tmp_path)
    assert rows == old and census['unique_trajectories'] == 2
    assert holdout == {'validation'} and seed == 42
    sources[0]['split'] = 'validation'
    save('cache/cache_plan.json', dict(sources=sources))
    with pytest.raises(ValueError, match='partition disagrees'):
        previous_partition(tmp_path)


def collection_fixture(tmp_path, monkeypatch):
    city = 'coulommiers'
    config = load_config(Path(supplement.ROOT) / 'configs/coulommiers_offline.toml')
    previous = tmp_path / 'previous'
    first, second = tmp_path / 'collection-a', tmp_path / 'collection-b'
    for root in (first, second):
        archives = root / city / 'ac/episodes/train'
        archives.mkdir(parents=True)
        for episode in (1, 2):
            (archives / f'episode_{episode:06d}.npz').touch()
    old = [row(str(first / city / 'ac/episodes/train/episode_000001.npz'), 'train', root='legacy-root'),
           row(str(tmp_path / 'old-heldout.npz'), 'validation', root='legacy-root')]
    for directory in ('census', 'pilot', 'cache'):
        (previous / directory).mkdir(parents=True)
    def save(name, value):
        (previous / name).write_text(json.dumps(value))
    save('census/file_rows.json', dict(rows=old))
    save('census/census.json', dict(unique_trajectories=2, input_hashes={'network': 'hash'}))
    save('census/archive_paths.json', [r['path'] for r in old])
    save('pilot/split.json', dict(seed=42, holdout_groups=['validation']))
    save('cache/cache_plan.json', dict(sources=[{**r, 'split': split}
                                               for r, split in zip(old, ('train', 'validation'))]))
    frozen_input = tmp_path / 'network.xml'
    frozen_input.write_text('network')
    input_hash = hashlib.sha256(frozen_input.read_bytes()).hexdigest()
    input_hashes = {label: input_hash for label in ('network_xml', 'network_agents', 'od_file', 'edge_registry')}
    save('census/census.json', dict(unique_trajectories=2, input_hashes=input_hashes))
    prepared = SimpleNamespace(input_hashes=input_hashes, network_xml=frozen_input,
                               network_agents_csv=frozen_input, od_file=frozen_input,
                               edge_registry=frozen_input)
    for root in (first, second):
        (root / 'manifest.json').write_text(json.dumps(dict(
            runs=[dict(city=city, command=['python', '--config', str(root / 'config.toml')])],
            input_hashes_by_city={city: input_hashes})))
    monkeypatch.setattr(supplement, 'load_config', lambda path: config)
    monkeypatch.setattr(supplement, 'validate_prepared_inputs', lambda actual: prepared)
    monkeypatch.setattr(supplement, 'load_traffic_topology', lambda prepared:
                        SimpleNamespace(network=SimpleNamespace(edge_ids=['edge'])))
    monkeypatch.setattr(supplement, 'export_relative', lambda actual, path: None)
    monkeypatch.setattr(supplement, 'freeze_committed', lambda root:
                        freeze_committed(root, cursor_reader=lambda run: 1))
    def scan(path, *args):
        result = next((deepcopy(r) for r in old if r['path'] == path),
                      row(path, 'train', root='incorrect-default'))
        result['crc_trajectory_candidate'] = path
        return result
    for r in old:
        r['crc_trajectory_candidate'] = r['path']
    save('census/file_rows.json', dict(rows=old))
    monkeypatch.setattr(supplement, 'scan_metadata', scan)
    return city, previous, first, second, prepared


def test_multiple_collections_keep_anchor_roots_and_committed_membership(tmp_path, monkeypatch):
    city, previous, first, second, prepared = collection_fixture(tmp_path, monkeypatch)
    exported = []
    previous_roots = set(supplement.load_config(previous / 'base_config.toml').data.archive_roots)
    monkeypatch.setattr(supplement, 'export_relative', lambda config, path: exported.append(config))
    output = tmp_path / 'supplement'
    sources = [(root, root / 'manifest.json') for root in (first, second)]
    summary = supplement.prepare(city, previous, output, collection_manifests=sources)
    assert summary['frozen_files'] == 3 and summary['new_files'] == 1
    assert all(set(config.data.archive_roots) == previous_roots | {first / city, second / city}
               for config in exported)
    frozen = json.loads((output / 'census/frozen_candidates.json').read_text())
    assert set(frozen['committed_bounds']) == {str(first), str(second)}
    for bounds in frozen['committed_bounds'].values():
        assert bounds['ac'] == dict(committed_episode=1, files=1)
    rows = json.loads((output / 'census/file_rows.json').read_text())['rows']
    assert {r['root'] for r in rows if r['path'].startswith(str(first))} == {'legacy-root'}
    assert {r['root'] for r in rows if r['path'].startswith(str(second))} == {str(second)}
    assert all('000002' not in r['path'] for r in rows)
    # Resume preserves the frozen snapshot even after checkpoint growth.
    before = (output / 'census/frozen_candidates.json').read_bytes()
    supplement.prepare(city, previous, output, collection_manifests=list(reversed(sources)))
    assert (output / 'census/frozen_candidates.json').read_bytes() == before
    with pytest.raises(ValueError, match='existing frozen supplement'):
        supplement.prepare(city, previous, output, collection_manifests=sources[:1])


def test_second_collection_input_and_physics_mismatch_is_rejected_before_freezing(tmp_path, monkeypatch):
    city, previous, first, second, prepared = collection_fixture(tmp_path, monkeypatch)
    manifest_path = second / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['input_hashes_by_city'][city] = {'network': 'different'}
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / 'supplement'
    sources = [(root, root / 'manifest.json') for root in (first, second)]
    with pytest.raises(ValueError, match='input identity differs'):
        supplement.prepare(city, previous, output, collection_manifests=sources)
    assert not output.exists()
    manifest['input_hashes_by_city'][city] = prepared.input_hashes
    manifest_path.write_text(json.dumps(manifest))
    config = supplement.load_config(previous / 'base_config.toml')
    from dataclasses import replace
    changed = replace(config, simulation=replace(config.simulation, vehicle_max_speed_mps=5.0))
    monkeypatch.setattr(supplement, 'load_config', lambda path:
                        changed if Path(path) == second / 'config.toml' else config)
    with pytest.raises(ValueError, match='physics/inputs differ'):
        supplement.prepare(city, previous, output, collection_manifests=sources)
    assert not output.exists()


def test_legacy_single_collection_and_frozen_file_remain_supported(tmp_path, monkeypatch):
    city, previous, first, second, prepared = collection_fixture(tmp_path, monkeypatch)
    output = tmp_path / 'supplement'
    summary = supplement.prepare(city, previous, output, collection=first,
                                 live_manifest=first / 'manifest.json')
    assert summary['frozen_files'] == 2
    path = output / 'census/frozen_candidates.json'
    frozen = json.loads(path.read_text())
    assert frozen['committed_bounds'] == {'ac': dict(committed_episode=1, files=1)}
    frozen.pop('collection_sources')
    path.write_text(json.dumps(frozen))
    supplement.prepare(city, previous, output, collection=first, live_manifest=first / 'manifest.json')
    with pytest.raises(ValueError, match='existing frozen supplement'):
        supplement.prepare(city, previous, output, collection=second,
                           live_manifest=second / 'manifest.json')


def test_collection_arguments_require_nonempty_unambiguous_roots(tmp_path):
    with pytest.raises(ValueError, match='at least one'):
        supplement.collection_sources(tmp_path, tmp_path / 'manifest.json', [])
    with pytest.raises(ValueError, match='only one manifest'):
        supplement.collection_sources(tmp_path, tmp_path / 'manifest.json',
                                      [(tmp_path, tmp_path / 'a.json'), (tmp_path, tmp_path / 'b.json')])


def test_cli_repeated_manifest_resolves_its_own_collection_root(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(supplement, 'prepare', lambda *args, **kwargs: calls.append(kwargs) or {})
    manifests = [tmp_path / name / 'manifest.json' for name in ('first', 'second')]
    supplement.main(['--city', 'coulommiers', '--previous-city-dir', str(tmp_path / 'previous'),
                     '--output-dir', str(tmp_path / 'output'),
                     '--collection-manifest', str(manifests[0]), '--collection-manifest', str(manifests[1])])
    assert calls == [dict(collection_manifests=[(path.parent, path) for path in manifests])]


def test_new_duplicate_sorting_first_preserves_previous_cache_source():
    from tools.prepare_city_pretrain import verify_duplicates
    old = row('/v5/original', 'demand')
    fresh = row('/two/fresh', 'demand')
    def reader(path):
        # Identical trajectories may carry different summaries/container hashes.
        return {}, dict(census_trajectory_signature='crc', trajectory_hash='trajectory',
                        content_hash='original-summary' if str(path) == old['path'] else 'new-summary')
    duplicates, unique = verify_duplicates([old, fresh], reader=reader)
    assert unique == 1 and duplicates[0]['files'][0] == fresh['path']
    anchored = supplement.preserve_previous_canonical(duplicates, [old], {})
    assert anchored[0]['files'] == [old['path'], fresh['path']]
    assert anchored[0]['trajectory_hash'] == 'trajectory'
    assert duplicates[0]['files'][0] == fresh['path']


def test_existing_noncanonical_path_cannot_replace_prior_canonical():
    old = [row('/z/canonical', 'demand'), row('/b/duplicate', 'demand')]
    census = dict(verified_duplicate_groups=[dict(trajectory_hash='same',
                                                files=['/z/canonical', '/b/duplicate'])])
    duplicates = [dict(trajectory_hash='same', files=['/a/fresh', '/b/duplicate', '/z/canonical']),
                  dict(trajectory_hash='new-only', files=['/a/new', '/b/new'])]
    anchored = supplement.preserve_previous_canonical(duplicates, old, census)
    assert anchored[0]['files'] == ['/z/canonical', '/a/fresh', '/b/duplicate']
    assert anchored[1] == duplicates[1]
    assert supplement.preserve_previous_canonical([], old, {}) == []


def test_full_supplement_anchors_duplicate_but_new_only_census_keeps_own_sources(tmp_path, monkeypatch):
    city, previous, first, second, prepared = collection_fixture(tmp_path, monkeypatch)
    old_path = str(first / city / 'ac/episodes/train/episode_000001.npz')
    fresh_path = str(second / city / 'ac/episodes/train/episode_000001.npz')
    real_verify = supplement.verify_duplicates
    def duplicates(rows):
        if {old_path, fresh_path} <= {r['path'] for r in rows}:
            return [dict(trajectory_hash='same', files=[fresh_path, old_path])], len(rows) - 1
        return real_verify(rows)
    monkeypatch.setattr(supplement, 'verify_duplicates', duplicates)
    output = tmp_path / 'supplement'
    summary = supplement.prepare(city, previous, output,
                                 collection_manifests=[(root, root / 'manifest.json')
                                                       for root in (first, second)])
    census = json.loads((output / 'census/census.json').read_text())
    assert census['verified_duplicate_groups'][0]['files'][0] == old_path
    split = json.loads((output / 'pilot/split.json').read_text())
    assert old_path in {r['path'] for r in split['train']}
    assert fresh_path not in {r['path'] for r in split['train']}
    assert summary['added_unique_trajectories'] == 0
    new_census = json.loads((output / 'new_census/census.json').read_text())
    assert new_census['unique_trajectories'] == 1 and new_census['verified_duplicate_groups'] == []


def test_distinct_old_canonical_trajectories_cannot_merge():
    old = [row('a', 'demand'), row('b', 'demand')]
    with pytest.raises(ValueError, match='merged distinct previous canonical'):
        supplement.preserve_previous_canonical([dict(trajectory_hash='same', files=['a', 'b'])], old, {})
