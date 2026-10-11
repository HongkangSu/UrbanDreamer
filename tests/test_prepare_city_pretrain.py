from collections import Counter
from pathlib import Path

import pytest

from tools.prepare_city_pretrain import freeze_committed, representative_rows, split_groups, verify_duplicates


def row(path, demand, *, method='ac', crc='candidate'):
    return dict(path=path, demand_group=demand, method=method, crc_trajectory_candidate=crc)


def test_freeze_excludes_uncommitted_tail_and_nontraining_files(tmp_path):
    for directory, episodes in [('ac/episodes/train', [1, 3, 4]), ('ac/episodes/eval', [2]),
                                ('iql/episodes/train', [1, 2])]:
        root = tmp_path / directory
        root.mkdir(parents=True)
        for index in episodes:
            (root / f'episode_{index:06d}.npz').touch()
    frozen = freeze_committed(tmp_path, cursor_reader=lambda run: {'ac': 3, 'iql': 1}[run['method']])
    assert {Path(path).parent.parent.parent.name + '/' + Path(path).name for path in frozen['paths']} == {
        'ac/episode_000001.npz', 'ac/episode_000003.npz', 'iql/episode_000001.npz'}
    assert frozen['committed_bounds']['ac'] == dict(committed_episode=3, files=2)


def test_split_uses_whole_demand_across_methods_and_is_order_independent():
    rows = [row(f'{method}/{index}', f'demand{index}', method=method)
            for method in ['ac', 'ippo', 'mamba'] for index in range(100)]
    heldout = split_groups(rows)
    assert heldout == split_groups(list(reversed(rows)))
    assert len(heldout) == 15
    train = {r['demand_group'] for r in rows if r['demand_group'] not in heldout}
    assert not train & heldout
    assert len(train) == 85


def test_representative_sampling_respects_bounds_and_balances_methods():
    rows = [row(f'{method}/{index}', f'demand{index}', method=method)
            for method in ['ac', 'ippo', 'mamba'] for index in range(100)]
    picked = representative_rows(rows, 48)
    assert len(picked) == 48
    assert Counter(r['method'] for r in picked) == {'ac': 16, 'ippo': 16, 'mamba': 16}
    assert picked == representative_rows(list(reversed(rows)), 48)


def test_crc_candidate_requires_full_sha_and_collision_does_not_deduplicate():
    rows = [row('a', 'demand'), row('b', 'demand'), row('c', 'other'), row('unique', 'unique', crc='different')]
    read = []
    def reader(path):
        read.append(str(path))
        return {}, dict(census_trajectory_signature='candidate', trajectory_hash='same' if str(path) in {'a', 'b'} else 'collision')
    groups, unique = verify_duplicates(rows, reader=reader)
    assert read == ['a', 'b', 'c']
    assert groups == [dict(trajectory_hash='same', files=['a', 'b'])]
    assert unique == 3


def test_duplicate_check_rejects_source_changes_and_conflicting_demand():
    rows = [row('a', 'first'), row('b', 'second')]
    def identity(signature):
        return lambda path: ({}, dict(census_trajectory_signature=signature, trajectory_hash='same'))
    with pytest.raises(ValueError, match='changed'):
        verify_duplicates(rows, reader=identity('changed'))
    with pytest.raises(ValueError, match='conflicting demand'):
        verify_duplicates(rows, reader=identity('candidate'))
