from copy import deepcopy
import hashlib
import json

import pytest

from tools.prepare_city_pretrain_supplement import anchored_holdout, previous_partition, verify_old_rows


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
