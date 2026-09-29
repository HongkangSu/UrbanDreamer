from copy import deepcopy

import numpy as np
import pytest
import torch

from urban_dreamer.learning.replay import MacroTransition, ReplayBuffer


def transition(step, *, episode=0, terminal=False):
    return MacroTransition(
        flow=torch.full((2, 8), float(step)),
        next_flow=torch.full((2, 8), float(step + 1)),
        action=torch.tensor([float(step + 1), 0.]),
        departure=torch.tensor([0., float(step)]),
        context=torch.tensor([step / 10., 0.5]),
        active_vehicle_fraction=0.25,
        vehicles=({'vehicle_id': 'v', 'departure_time': 1.25, 'departed': True,
                   'arrived': False, 'current_edge': 'a', 'destination_edge': 'b',
                   'remaining_distance': 3.5, 'committed_next_edge': 'b',
                   'pending_decision': False, 'junction_remaining_distance': 0.,
                   'junction_speed': 10.},),
        time=30. * step, episode=episode, step_index=step, terminal=terminal,
    )


def test_explicit_transitions_preserve_alignment_and_seed():
    replay = ReplayBuffer(sequence_length=2)
    assert replay.append_episode([transition(0), transition(1, terminal=True)], total_travel_minutes=1.)
    batch = replay.sample_batch(batch_size=1, rng=np.random.default_rng(0))
    assert batch.flow.shape == (1, 3, 2, 8)
    assert batch.context.shape == (1, 2, 2)
    torch.testing.assert_close(batch.flow[0, :, 0, 0], torch.tensor([0., 1., 2.]))
    torch.testing.assert_close(batch.action[0, :, 0], torch.tensor([1., 2.]))
    torch.testing.assert_close(batch.departure[0, :, 1], torch.tensor([0., 1.]))
    assert batch.vehicles[0][0]['departure_time'] == 1.25
    assert batch.vehicles[0][0]['committed_next_edge'] == 'b'
    assert batch.vehicles[0][0]['remaining_distance'] == 3.5
    assert batch.terminal.tolist() == [[False, True]]
    assert batch.episode_complete.tolist() == [True]


def test_failed_or_incomplete_episode_cannot_enter_replay():
    replay = ReplayBuffer(sequence_length=1)
    assert not replay.append_episode([transition(0)], total_travel_minutes=1., require_complete=True)
    assert replay.episodes == ()
    with pytest.raises(ValueError, match='continu'):
        replay.append_episode([transition(0), transition(2, terminal=True)], total_travel_minutes=1.)


def test_zero_action_window_is_retained_and_multihop_counts_not_normalized():
    replay = ReplayBuffer(sequence_length=1)
    record = transition(0, terminal=True)
    record.action.zero_()
    assert replay.append_episode([record], total_travel_minutes=1.)
    assert replay.sample_batch(batch_size=1, rng=np.random.default_rng(0)).action.sum() == 0
    assert transition(4).action.sum() == 5


def test_replay_restore_preserves_sampler_and_rolls_capacity():
    replay = ReplayBuffer(sequence_length=1, max_episodes=2)
    for episode in range(3):
        replay.append_episode([transition(0, episode=episode), transition(1, episode=episode, terminal=True)], total_travel_minutes=1.)
    assert [rows[0].episode for rows in replay.episodes] == [1, 2]
    rng = np.random.default_rng(31)
    replay.sample_batch(batch_size=1, rng=rng)
    restored = ReplayBuffer(sequence_length=1, max_episodes=2)
    restored.load_state_dict(deepcopy(replay.state_dict()))
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = deepcopy(rng.bit_generator.state)
    left = replay.sample_batch(batch_size=7, rng=rng)
    right = restored.sample_batch(batch_size=7, rng=rng2)
    for field in ('flow', 'action', 'departure', 'context', 'episode', 'start_index', 'time'):
        torch.testing.assert_close(getattr(left, field), getattr(right, field))
    assert left.vehicles == right.vehicles


def test_old_replay_schema_is_rejected():
    with pytest.raises(ValueError, match='schema'):
        ReplayBuffer(sequence_length=1).load_state_dict({'kind': 'warmup'})


def test_transition_validates_shapes_and_fraction():
    values = transition(0).state_dict()
    values['context'] = torch.zeros(3)
    with pytest.raises(ValueError, match='context'):
        MacroTransition(**values)
    values = transition(0).state_dict()
    values['active_vehicle_fraction'] = 1.1
    with pytest.raises(ValueError, match='fraction'):
        MacroTransition(**values)


def test_short_completed_episode_uses_actual_transitions_without_padding():
    replay = ReplayBuffer(sequence_length=8)
    replay.append_episode([transition(0, terminal=True)], total_travel_minutes=1.)
    batch = replay.sample_batch(batch_size=2, rng=np.random.default_rng(0))
    assert batch.flow.shape == (2, 2, 2, 8)
    assert batch.action.shape == (2, 1, 2)
    assert batch.terminal.tolist() == [[True], [True]]


def test_snapshot_remains_exact_across_state_and_batch_copies():
    replay = ReplayBuffer(sequence_length=1)
    record = transition(0, terminal=True)
    replay.append_episode([record], total_travel_minutes=1.)
    state = replay.state_dict()
    restored = ReplayBuffer(sequence_length=1)
    restored.load_state_dict(state)
    state['episodes'][0]['transitions'][0]['vehicles'][0]['remaining_distance'] = 999.
    batch = restored.sample_batch(batch_size=1, rng=np.random.default_rng(0))
    assert batch.vehicles[0][0]['remaining_distance'] == 3.5
    batch.vehicles[0][0]['remaining_distance'] = 555.
    assert restored.episodes[0][0].vehicles[0]['remaining_distance'] == 3.5


def test_valid_horizon_truncation_can_be_sampled_and_restored():
    replay = ReplayBuffer(sequence_length=2)
    assert replay.append_episode([transition(0), transition(1)], total_travel_minutes=1.)
    restored = ReplayBuffer(sequence_length=2)
    restored.load_state_dict(replay.state_dict())
    batch = restored.sample_batch(batch_size=1, rng=np.random.default_rng(0))
    assert batch.terminal.tolist() == [[False, False]]
    assert batch.episode_complete.tolist() == [False]
    assert batch.flow.shape == (1, 3, 2, 8)
    assert batch.active_vehicle_fraction.tolist() == [[0.25, 0.25]]


def active_transition(step, *, active=False, fraction=0., snapshot=False, terminal=False):
    values = transition(step, terminal=terminal).state_dict()
    values['context'] = torch.tensor([step / 10., float(active)])
    values['active_vehicle_fraction'] = fraction
    values['action'] = torch.zeros(2)
    values['departure'] = torch.zeros(2)
    if not snapshot:
        values['vehicles'] = ()
    return MacroTransition(**values)


def test_active_only_excludes_empty_windows_but_retains_complete_episode():
    replay = ReplayBuffer(sequence_length=3, active_only=True)
    rows = [active_transition(index, active=index in (1, 2, 5),
                              fraction=0.25 if index == 4 else 0., terminal=index == 6)
            for index in range(7)]
    replay.append_episode(rows, total_travel_minutes=1.)
    assert len(replay) == 7
    assert replay.has_samples
    batch = replay.sample_batch(4, np.random.default_rng(3))
    assert set(batch.start_index.tolist()) == {1, 2, 4, 5}
    assert batch.action.sum() == 0  # Long-edge windows with no decisions still train.
    assert batch.flow.shape == (4, 2, 2, 8)
    for index, start in enumerate(batch.start_index.tolist()):
        assert batch.time[index] == start * 30.
        assert batch.flow[index, :, 0, 0].tolist() == [start, start + 1]


def test_active_sequences_do_not_stitch_across_empty_gaps():
    replay = ReplayBuffer(sequence_length=4, active_only=True)
    replay.append_episode([active_transition(i, active=i in (1, 2, 4, 5, 6), terminal=i == 7)
                           for i in range(8)], total_travel_minutes=1.)
    rng = np.random.default_rng(41)
    lengths = {}
    for _ in range(5):
        batch = replay.sample_batch(1, rng)
        start = batch.start_index.item()
        length = batch.action.shape[1]
        lengths[start] = length
        assert batch.flow[0, :, 0, 0].tolist() == list(range(start, start + length + 1))
        assert batch.context.shape[1] == length
    assert lengths == {1: 2, 2: 1, 4: 3, 5: 2, 6: 1}


def test_active_only_empty_replay_and_snapshot_activity():
    replay = ReplayBuffer(sequence_length=2, active_only=True)
    assert not replay.has_samples
    replay.append_episode([active_transition(0, terminal=True)], total_travel_minutes=0.)
    assert len(replay) == 1
    assert not replay.has_samples
    with pytest.raises(ValueError, match='empty'):
        replay.sample_batch(1, np.random.default_rng(0))
    replay.append_episode([active_transition(0, snapshot=True, terminal=True)], total_travel_minutes=1.)
    assert replay.has_samples


def test_active_only_sampler_restores_exactly_and_rejects_previous_schema():
    replay = ReplayBuffer(sequence_length=3, active_only=True)
    replay.append_episode([active_transition(i, active=i in (2, 3, 5, 6), terminal=i == 7)
                           for i in range(8)], total_travel_minutes=1.)
    rng = np.random.default_rng(73)
    replay.sample_batch(1, rng)
    state = deepcopy(replay.state_dict())
    restored = ReplayBuffer(sequence_length=3, active_only=True)
    restored.load_state_dict(state)
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = deepcopy(rng.bit_generator.state)
    for _ in range(3):
        left, right = replay.sample_batch(3, rng), restored.sample_batch(3, rng2)
        for name in ('flow', 'action', 'context', 'start_index', 'time'):
            torch.testing.assert_close(getattr(left, name), getattr(right, name))
    with pytest.raises(ValueError, match='configuration'):
        ReplayBuffer(sequence_length=3).load_state_dict(state)
    state['schema_version'] = 2
    with pytest.raises(ValueError, match='schema'):
        restored.load_state_dict(state)


def test_imagination_seeds_require_active_start_snapshots_without_consuming_sampler():
    replay = ReplayBuffer(sequence_length=3, active_only=True)
    rows = [active_transition(0, fraction=0.1), active_transition(1, snapshot=True),
            active_transition(2), active_transition(3, snapshot=True, terminal=True)]
    replay.append_episode(rows, total_travel_minutes=1.)
    assert replay.has_active_seeds
    rng = np.random.default_rng(71)
    replay.sample_batch(1, rng)
    before = deepcopy(replay.state_dict())
    seeds = replay.sample_active_seeds(50, rng)
    assert set(seeds.start_index.tolist()) == {1, 3}
    assert all(vehicles[0]['departed'] and not vehicles[0]['arrived'] for vehicles in seeds.vehicles)
    after = replay.state_dict()
    assert before['sample_order'] == after['sample_order']
    assert before['sample_cursor'] == after['sample_cursor']
    restored = ReplayBuffer(sequence_length=3, active_only=True)
    restored.load_state_dict(after)
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = deepcopy(rng.bit_generator.state)
    left, right = replay.sample_active_seeds(7, rng), restored.sample_active_seeds(7, rng2)
    torch.testing.assert_close(left.start_index, right.start_index)
    torch.testing.assert_close(left.flow, right.flow)


def test_exposure_without_active_start_snapshot_is_not_an_imagination_seed():
    replay = ReplayBuffer(sequence_length=1, active_only=True)
    future = active_transition(0, fraction=0.1, snapshot=True).state_dict()
    future['vehicles'][0]['departed'] = False
    arrived = active_transition(1, fraction=0.1, snapshot=True, terminal=True).state_dict()
    arrived['vehicles'][0]['arrived'] = True
    replay.append_episode([MacroTransition(**future), MacroTransition(**arrived)], total_travel_minutes=1.)
    assert replay.has_samples
    assert not replay.has_active_seeds
    with pytest.raises(ValueError, match='active vehicle'):
        replay.sample_active_seeds(1, np.random.default_rng(0))
