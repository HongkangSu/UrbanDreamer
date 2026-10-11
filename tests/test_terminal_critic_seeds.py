from copy import deepcopy

import numpy as np
import pytest
import torch

from urban_dreamer.learning.replay import MacroTransition, ReplayBuffer


def make_replay(*, active=True):
    replay = ReplayBuffer(sequence_length=3)
    rows = [MacroTransition(
        flow=torch.full((1, 8), float(step)),
        next_flow=torch.full((1, 8), float(step + 1)),
        action=torch.zeros(1), departure=torch.zeros(1), context=torch.zeros(4),
        active_vehicle_fraction=1., time=300. + step * 30., step_index=step,
        vehicles=({'vehicle_id': f'observed-{step}', 'departed': active, 'arrived': False},),
        next_vehicles=({'vehicle_id': 'future', 'departed': True, 'arrived': False},),
        started_count=1, next_started_count=1,
    ) for step in range(6)]
    replay.append_episode(rows, total_vehicle_count=1, total_travel_minutes=0.)
    return replay


def test_terminal_seeds_use_observed_active_clock_and_inclusive_horizon():
    replay = make_replay()
    # Only observed t=360 and t=390 qualify. t=420 is already at the deadline.
    rng = np.random.default_rng(51)
    batch = replay.sample_terminal_seeds(80, rng, deadline=420., horizon_seconds=60.)
    expected = np.random.default_rng(51).integers(2, size=80) + 2
    assert batch is not None
    assert batch.start_index.tolist() == expected.tolist()
    assert set(batch.time.tolist()) == {360., 390.}
    assert batch.flow.shape[1] == 2
    assert batch.boundary_vehicles is None
    assert [v[0]['vehicle_id'] for v in batch.vehicles] == [f'observed-{i}' for i in expected]
    assert batch.episode_complete.tolist() == [False] * 80


def test_terminal_sampler_reproducible_without_advancing_ordinary_sampler():
    replay = make_replay()
    training_rng = np.random.default_rng(4)
    replay.sample_batch(1, training_rng)
    order, cursor = list(replay._sample_order), replay._sample_cursor
    rng_state = deepcopy(training_rng.bit_generator.state)
    left = replay.sample_terminal_seeds(10, np.random.default_rng(17), 420., 60.)
    right = replay.sample_terminal_seeds(10, np.random.default_rng(17), 420., 60.)
    assert left is not None and right is not None
    assert torch.equal(left.start_index, right.start_index)
    assert replay._sample_order == order
    assert replay._sample_cursor == cursor
    assert training_rng.bit_generator.state == rng_state
    left.vehicles[0][0]['vehicle_id'] = 'mutated'
    assert all(row.vehicles[0]['vehicle_id'].startswith('observed-') for row in replay.episodes[0])


@pytest.mark.parametrize('replay,deadline,horizon', [
    (ReplayBuffer(1), 420., 60.),
    (make_replay(active=False), 420., 60.),
    (make_replay(), 600., 30.),
    (make_replay(), 300., 60.),
])
def test_empty_terminal_pool_does_not_consume_rng(replay, deadline, horizon):
    rng = np.random.default_rng(3)
    before = deepcopy(rng.bit_generator.state)
    assert replay.sample_terminal_seeds(2, rng, deadline, horizon) is None
    assert rng.bit_generator.state == before


@pytest.mark.parametrize('name,value', [
    ('batch_size', 0), ('batch_size', True), ('batch_size', 1.5),
    ('deadline', 0.), ('deadline', -1.), ('deadline', float('nan')),
    ('deadline', float('inf')), ('deadline', True),
    ('horizon_seconds', 0.), ('horizon_seconds', -1.),
    ('horizon_seconds', float('nan')), ('horizon_seconds', float('inf')),
])
def test_terminal_sampler_rejects_invalid_arguments(name, value):
    kwargs = dict(batch_size=2, deadline=420., horizon_seconds=60.)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        make_replay().sample_terminal_seeds(rng=np.random.default_rng(0), **kwargs)
