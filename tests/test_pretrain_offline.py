from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from tools.pretrain_offline import sampled_schedule, scheduled_batches, select_split
from urban_dreamer.learning.replay import MacroTransition


def test_split_keeps_shared_demands_and_duplicate_trajectories_out_of_holdout():
    rows = [dict(root=root, method=method, demand_group=f'demand_{episode}',
                 path=f'{root}/{method}/{episode}.npz')
            for root in ('original', 'prior') for method in ('random', 'actor')
            for episode in range(20)]
    duplicates = [dict(trajectory_hash=f'identical_{episode}',
                       files=[f'{root}/random/{episode}.npz' for root in ('original', 'prior')])
                  for episode in range(20)]
    original = deepcopy(rows)
    split = select_split(rows, duplicates, seed=42, train_per_stratum=2, validation_per_stratum=1)
    assert split == select_split(rows, duplicates, seed=42, train_per_stratum=2, validation_per_stratum=1)
    assert rows == original
    assert len(split['train']) == 8 and len(split['validation']) == 4
    assert not ({r['demand_group'] for r in split['train']} & {r['demand_group'] for r in split['validation']})
    selected = split['train'] + split['validation']
    random_demands = [r['demand_group'] for r in selected if r['method'] == 'random']
    assert len(set(random_demands)) == len(random_demands)
    for name, count in (('train', 2), ('validation', 1)):
        for root in ('original', 'prior'):
            for method in ('random', 'actor'):
                assert sum(r['root'] == root and r['method'] == method for r in split[name]) == count


def test_split_rejects_insufficient_independent_trajectories():
    rows = [dict(root='root', method='random', demand_group='one', path='one.npz')]
    with pytest.raises(ValueError, match='insufficient distinct'):
        select_split(rows, [], seed=42, train_per_stratum=1, validation_per_stratum=1)


def test_sampled_starts_span_episode_without_duplicates_and_keep_short_tail():
    dataset = SimpleNamespace(sequence_length=16,
        manifest=dict(entries=[dict(windows=127), dict(windows=10)]))
    schedule = sampled_schedule(dataset, starts_per_episode=8, batch_size=2, seed=42)
    assert schedule == sampled_schedule(dataset, starts_per_episode=8, batch_size=2, seed=42)
    starts = [start for row in schedule if row['episode'] == 0 for start in row['starts']]
    assert len(set(starts)) == len(starts) == 8
    assert min(starts) == 0 and max(starts) == 111
    assert [row['starts'] for row in schedule if row['episode'] == 1] == [[0]]


def test_scheduled_batches_load_once_per_episode_and_preserve_window_identity():
    calls = []
    def episode(index):
        calls.append(index)
        rows = [MacroTransition(flow=np.zeros((2, 8)), next_flow=np.zeros((2, 8)),
                action=np.zeros(2), departure=np.zeros(2), context=np.zeros(2),
                active_vehicle_fraction=0., time=30. * step, episode=index,
                step_index=step, action_events=()) for step in range(6)]
        return dict(transitions=rows, total_travel_minutes=0., total_vehicle_count=0)
    dataset = SimpleNamespace(sequence_length=2, _episode=episode)
    batches = list(scheduled_batches(dataset, [dict(episode=0, starts=[0, 2]),
        dict(episode=0, starts=[4]), dict(episode=1, starts=[1, 3])]))
    assert calls == [0, 1]
    assert [batch.start_index.tolist() for batch in batches] == [[0, 2], [4], [1, 3]]
    assert [batch.episode.tolist() for batch in batches] == [[0, 0], [0], [1, 1]]
    assert [batch.action_events.batch_size for batch in batches] == [2, 1, 2]
