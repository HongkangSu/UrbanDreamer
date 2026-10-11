from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.action_events import ActionEvent
from urban_dreamer.learning.cohort_targets import prepare_transition_targets, prepare_cohort_batch, cohort_capacity_identity
from urban_dreamer.learning.episode_archive import archive_to_replay_episode, observed_arrival_times
from urban_dreamer.learning.replay import MacroTransition, ReplayBuffer, REPLAY_SCHEMA_VERSION


def vehicle(name, edge, *, junction=False, departed=True, arrived=False, **kwargs):
    return dict(vehicle_id=name, current_edge=edge, destination_edge=2, departed=departed,
                arrived=arrived, in_junction=junction, departure_time=0., **kwargs)


def cohort_replay():
    # Internal seed is a member of C0 despite not occupying an external edge.
    # Queued vehicle has an overdue scheduled departure but no observed departure.
    boundaries = [
        (vehicle('a', 0), vehicle('internal', 0, junction=True), vehicle('queued', 0, departed=False)),
        (vehicle('internal', 1), vehicle('newborn', 1), vehicle('queued', 1)),
        (vehicle('internal', 2, junction=True), vehicle('newborn', 2), vehicle('queued', 2)),
        (vehicle('newborn', 2), vehicle('queued', 2)),
    ]
    rows = []
    for index in range(3):
        flow = torch.arange(24, dtype=torch.float32).reshape(3, 8) + index
        events = ((ActionEvent('a', 0, 1., 0, 1, 2), ActionEvent('internal', 1, 11., 0, 1, 2),
                   ActionEvent('newborn', 2, 25., 0, 1, 2)) if index == 0 else
                  (ActionEvent('internal', 3, 31., 1, 2, 2), ActionEvent('queued', 4, 41., 1, 2, 2))
                  if index == 1 else ())
        rows.append(MacroTransition(flow=flow, next_flow=flow + 1., action=torch.ones(3),
            departure=torch.ones(3), context=torch.tensor([0., .5]), active_vehicle_fraction=.5,
            vehicles=boundaries[index], time=index * 30., step_index=index,
            action_events=events, next_vehicles=boundaries[-1] if index == 2 else None))
    replay = ReplayBuffer(sequence_length=3)
    replay.append_episode(rows, total_travel_minutes=3., total_vehicle_count=4,
                          arrival_times={'a': 20., 'internal': 75.})
    return replay


def test_fixed_initial_cohort_filters_newborns_and_preserves_observed_background():
    replay = cohort_replay()
    original = replay._batch_from_picks([(0, 0)])
    raw = original.flow.clone()
    actual = prepare_cohort_batch(original, capacities=[10., 20., 30.])
    assert actual.cohort_count.tolist() == [2]
    torch.testing.assert_close(actual.transition_flow[0, :, :, 2],
        torch.tensor([[.1, 0., 0.], [0., .05, 0.], [0., 0., 0.], [0., 0., 0.]]))
    torch.testing.assert_close(actual.transition_flow[..., [0, 1, 3, 4, 5, 6, 7]], raw[..., [0, 1, 3, 4, 5, 6, 7]])
    torch.testing.assert_close(original.flow, raw, atol=0., rtol=0.)
    assert original.transition_flow is None
    assert actual.transition_action_events.vehicle_ids == ('a', 'internal', 'internal')
    assert actual.transition_action_events.event_index.tolist() == [0, 1, 3]
    torch.testing.assert_close(actual.transition_action[0], torch.tensor([[0., .5, 0.], [0., 0., .25], [0., 0., 0.]]))
    # Initial a contributes 20s; internal contributes 30/30/15s. Episodes have
    # N=4, so exposure keeps N_episode instead of changing to N_cohort=2.
    torch.testing.assert_close(actual.transition_active_vehicle_fraction[0], torch.tensor([50 / 120, .25, 15 / 120]))
    assert actual.action is original.action and actual.flow is original.flow


def test_membership_is_reselected_only_at_sequence_start_and_crop_has_future_boundary():
    replay = cohort_replay()
    replay.sequence_length = 1
    middle = replay._batch_from_picks([(0, 1)])
    assert middle.boundary_vehicles[0][-1] == replay.episodes[0][2].vehicles
    actual = prepare_cohort_batch(middle, capacities=[10., 20., 30.])
    assert actual.cohort_count.tolist() == [3]
    torch.testing.assert_close(actual.transition_flow[0, 0, :, 2], torch.tensor([0., .15, 0.]))
    torch.testing.assert_close(actual.transition_flow[0, 1, :, 2], torch.tensor([0., 0., 2 / 30]))
    assert actual.transition_action_events.vehicle_ids == ('internal', 'queued')
    assert middle.vehicles[0] is middle.boundary_vehicles[0][0]


def test_empty_observed_cohort_has_zero_targets_without_using_scheduled_departures():
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    # No observed car at t0. Future observations/events remain valid telemetry.
    boundaries = (((), *batch.boundary_vehicles[0][1:]),)
    # Earlier actions must still be registry-known even though C0 is empty.
    modified = replace(batch, boundary_vehicles=boundaries,
                       arrival_times=({'a': 20., 'internal': 75.},))
    actual = prepare_cohort_batch(modified, capacities=[10., 20., 30.])
    assert actual.cohort_count.tolist() == [0]
    assert actual.transition_flow[..., 2].eq(0).all()
    assert actual.transition_action.eq(0).all()
    assert actual.transition_active_vehicle_fraction.eq(0).all()
    assert len(actual.transition_action_events) == 0


def test_missing_boundary_arrival_or_registry_metadata_fails_explicitly():
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    with pytest.raises(ValueError, match='boundary snapshots'):
        prepare_cohort_batch(replace(batch, boundary_vehicles=None), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='boundary snapshots'):
        prepare_cohort_batch(replace(batch, arrival_times=None), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='disagreement'):
        prepare_cohort_batch(replace(batch, arrival_times=({},)), capacities=[10., 20., 30.])
    bad = replace(batch.action_events, vehicle_ids=('unknown', *batch.action_events.vehicle_ids[1:]))
    with pytest.raises(ValueError, match='observed registry'):
        prepare_cohort_batch(replace(batch, action_events=bad), capacities=[10., 20., 30.])


def test_replay7_preserves_final_boundary_and_arrivals_and_rejects_old_checkpoint():
    replay = cohort_replay()
    state = replay.state_dict()
    assert state['schema_version'] == REPLAY_SCHEMA_VERSION == 7
    assert state['episodes'][0]['transitions'][0]['next_vehicles'] is None
    assert state['episodes'][0]['transitions'][-1]['next_vehicles'] is not None
    restored = ReplayBuffer(sequence_length=3)
    restored.load_state_dict(state)
    original = prepare_cohort_batch(replay._batch_from_picks([(0, 0)]), capacities=[10., 20., 30.])
    resumed = prepare_cohort_batch(restored._batch_from_picks([(0, 0)]), capacities=[10., 20., 30.])
    torch.testing.assert_close(original.transition_flow, resumed.transition_flow)
    torch.testing.assert_close(original.transition_active_vehicle_fraction, resumed.transition_active_vehicle_fraction)
    old = deepcopy(state)
    old['schema_version'] = 6
    with pytest.raises(ValueError, match='unsupported replay schema'):
        restored.load_state_dict(old)


def test_legacy_raw_replay_does_not_require_cohort_metadata():
    replay = cohort_replay()
    rows = [replace(row, next_vehicles=None) for row in replay.episodes[0]]
    raw = ReplayBuffer(sequence_length=3)
    raw.append_episode(rows, total_travel_minutes=3., total_vehicle_count=4)
    batch = raw._batch_from_picks([(0, 0)])
    assert batch.boundary_vehicles is None and batch.arrival_times is None
    assert batch.transition_flow is None and batch.flow.shape == (1, 4, 3, 8)
    with pytest.raises(ValueError, match='boundary snapshots'):
        prepare_cohort_batch(batch, capacities=[10., 20., 30.])


def test_old_archived_arrivals_use_actual_departure_not_scheduled_time():
    final = vehicle('a', 1, arrived=True, actual_departure_time=9.)
    data = dict(states=[dict(vehicles=[final])], travel_times={'a': 11.})
    assert observed_arrival_times(data) == {'a': 20.}
    final['arrival_time'] = 21.
    assert observed_arrival_times(data) == {'a': 21.}


def test_trusted_vectorized_batch_fast_path_matches_reference_without_vehicle_metadata():
    raw = cohort_replay()._batch_from_picks([(0, 0)])
    capacity = np.array([10., 20., 30.], dtype=np.float32)
    reference = prepare_cohort_batch(raw, capacities=capacity)
    # Independent vector construction: initial internal member enters road 1,
    # becomes internal at t60, and arrives at t75; newborn/queued stay excluded.
    counts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    flow = raw.flow.clone()
    flow[0, :, :, 2] = torch.from_numpy(counts / capacity)
    selected = torch.tensor([vehicle_id in {'a', 'internal'} for vehicle_id in raw.action_events.vehicle_ids])
    events = raw.action_events.select(selected)
    bins = ((events.batch_index * 3 + events.time_index) * 3 + events.next_edge).numpy()
    actions = np.bincount(bins, minlength=9).reshape(1, 3, 3) / 4.
    prepared = replace(raw, transition_flow=flow, transition_action=torch.tensor(actions, dtype=torch.float32),
        transition_action_events=events, transition_active_vehicle_fraction=torch.tensor([[50 / 120, 30 / 120, 15 / 120]]),
        cohort_count=torch.tensor([2]), transition_volume_scope='initial_cohort',
        transition_capacity_identity=cohort_capacity_identity(capacity), boundary_vehicles=None, arrival_times=None)
    actual = prepare_cohort_batch(prepared, capacities=capacity)
    assert actual is prepared
    for name in ('transition_flow', 'transition_action', 'transition_active_vehicle_fraction', 'cohort_count'):
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name))
    assert actual.transition_action_events.vehicle_ids == reference.transition_action_events.vehicle_ids
    torch.testing.assert_close(actual.transition_action_events.event_index, reference.transition_action_events.event_index)
    assert actual.transition_action_events.vehicle_ids.count('internal') == 2


def test_scope_marker_is_required_and_incomplete_or_invalid_cached_targets_fail():
    raw = cohort_replay()._batch_from_picks([(0, 0)])
    prepared = prepare_cohort_batch(raw, capacities=[10., 20., 30.])
    unmarked = replace(prepared, transition_volume_scope=None, transition_flow=raw.flow)
    recomputed = prepare_cohort_batch(unmarked, capacities=[10., 20., 30.])
    assert recomputed is not unmarked
    torch.testing.assert_close(recomputed.transition_flow, prepared.transition_flow)
    with pytest.raises(ValueError, match='capacity identity'):
        prepare_cohort_batch(prepared, capacities=[10., 20., 31.])
    with pytest.raises(ValueError, match='capacities'):
        prepare_cohort_batch(prepared, capacities=[10., 20.])
    with pytest.raises(ValueError, match='transition_action must have shape'):
        prepare_cohort_batch(replace(prepared, transition_action=None), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='integer population'):
        prepare_cohort_batch(replace(prepared, cohort_count=torch.tensor([2.])), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='exceed.*population'):
        prepare_cohort_batch(replace(prepared, cohort_count=torch.tensor([5])), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='exposure exceeds'):
        prepare_cohort_batch(replace(prepared, transition_active_vehicle_fraction=torch.ones(1, 3)), capacities=[10., 20., 30.])
    bad_volume = prepared.transition_flow.clone()
    bad_volume[0, 0, 0, 2] = 1.  # Ten external cars exceed the two-car initial cohort.
    with pytest.raises(ValueError, match='external volume exceeds'):
        prepare_cohort_batch(replace(prepared, transition_flow=bad_volume), capacities=[10., 20., 30.])
    wrong_windows = replace(prepared.transition_action_events, time_steps=4)
    with pytest.raises(ValueError, match='align with physical windows'):
        prepare_cohort_batch(replace(prepared, transition_action_events=wrong_windows), capacities=[10., 20., 30.])


def transition_batch():
    from urban_dreamer.learning.action_events import pack_action_events
    raw = cohort_replay()._batch_from_picks([(0, 0)])
    # New B appears during window0 and must become eligible in window1.
    # New C appears during window1 and is excluded only from that endpoint.
    boundaries = ((
        (vehicle('A', 0), vehicle('I', 0, junction=True)),
        (vehicle('A', 0), vehicle('I', 1), vehicle('B', 1)),
        (vehicle('I', 1, junction=True), vehicle('B', 2), vehicle('C', 2)),
        (vehicle('B', 2), vehicle('C', 1)),
    ),)
    flow = raw.flow.clone()
    flow[..., 2] = torch.tensor([[[.1, 0., 0.], [.1, .1, 0.], [0., 0., 2 / 30], [0., .05, 1 / 30]]])
    windows = [[(ActionEvent('A', 0, 1., 0, 1, 2), ActionEvent('B', 1, 21., 0, 1, 2)),
                (ActionEvent('B', 2, 31., 1, 2, 2), ActionEvent('B', 3, 41., 1, 2, 2),
                 ActionEvent('C', 4, 51., 1, 2, 2)), ()]]
    return replace(raw, flow=flow, boundary_vehicles=boundaries, arrival_times=None,
                   action_events=pack_action_events(windows), total_vehicle_count=torch.tensor([4]))


def test_each_transition_uses_full_observed_input_and_reselects_previous_window_births():
    raw = transition_batch()
    before = raw.flow.clone()
    actual = prepare_transition_targets(raw, capacities=[10., 20., 30.])
    assert actual.transition_volume_scope == 'previous_state_excluding_current_births'
    assert actual.transition_target_flow.shape == (1, 3, 3, 8)
    assert actual.transition_cohort_count.tolist() == [[2, 3, 3]]
    # I counts as a starting member even while internal, but does not add
    # endpoint road volume when internal or absent. A has arrived at t60.
    torch.testing.assert_close(actual.transition_target_flow[0, :, :, 2],
        torch.tensor([[.1, .05, 0.], [0., 0., 1 / 30], [0., .05, 1 / 30]]))
    assert actual.flow is raw.flow
    torch.testing.assert_close(actual.flow, before, rtol=0., atol=0.)
    assert actual.flow[0, 1, 1, 2] == .1  # Both I and newborn B are full input.
    assert actual.transition_target_flow[0, 1, 2, 2] == pytest.approx(1 / 30)  # B now eligible, C excluded.
    assert actual.transition_target_flow[0, 2, 1, 2] == .05  # C eligible next interval.
    torch.testing.assert_close(actual.transition_target_flow[..., [0, 1, 3, 4, 5, 6, 7]],
                               raw.flow[:, 1:, :, [0, 1, 3, 4, 5, 6, 7]])
    for name in ('action', 'action_events', 'active_vehicle_fraction', 'context', 'departure'):
        assert getattr(actual, name) is getattr(raw, name)
    assert actual.transition_action is None and actual.transition_action_events is None
    assert actual.transition_active_vehicle_fraction is None
    assert actual.action_events.vehicle_ids.count('B') == 3  # All same-car actions remain.


def test_transition_membership_uses_observed_departure_flags_not_overdue_schedules():
    raw = transition_batch()
    queued = vehicle('queued', 0, departed=False)
    boundaries = ((raw.boundary_vehicles[0][0] + (queued,),
                   raw.boundary_vehicles[0][1] + (vehicle('queued', 2),),
                   *raw.boundary_vehicles[0][2:]),)
    actual = prepare_transition_targets(replace(raw, boundary_vehicles=boundaries, total_vehicle_count=torch.tensor([5])),
                                        capacities=[10., 20., 30.])
    assert actual.transition_cohort_count[0, 0] == 2
    assert actual.transition_cohort_count[0, 1] == 4
    assert actual.transition_target_flow[0, 0, 2, 2] == 0.


def test_prepared_transition_fast_path_is_distinct_from_legacy_scope_and_checks_capacity():
    raw = transition_batch()
    reference = prepare_transition_targets(raw, capacities=[10., 20., 30.])
    cached = replace(reference, boundary_vehicles=None, arrival_times=None)
    assert prepare_transition_targets(cached, capacities=[10., 20., 30.]) is cached
    legacy_marker = replace(reference, transition_volume_scope='initial_cohort')
    recomputed = prepare_transition_targets(legacy_marker, capacities=[10., 20., 30.])
    torch.testing.assert_close(recomputed.transition_target_flow, reference.transition_target_flow)
    assert recomputed is not legacy_marker
    with pytest.raises(ValueError, match='observed boundary'):
        prepare_transition_targets(replace(cached, transition_volume_scope='initial_cohort'), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='capacity identity'):
        prepare_transition_targets(cached, capacities=[10., 20., 31.])
    with pytest.raises(ValueError, match='shape'):
        prepare_transition_targets(replace(cached, transition_target_flow=None), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='cohort counts exceed'):
        prepare_transition_targets(replace(cached, transition_cohort_count=torch.tensor([[2, 5, 3]])), capacities=[10., 20., 30.])
    with pytest.raises(ValueError, match='integers'):
        prepare_transition_targets(replace(cached, transition_cohort_count=torch.tensor([[2., 3., 3.]])), capacities=[10., 20., 30.])
    volume = cached.transition_target_flow.clone()
    volume[0, 0, 0, 2] = 1.
    with pytest.raises(ValueError, match='external volume exceeds'):
        prepare_transition_targets(replace(cached, transition_target_flow=volume), capacities=[10., 20., 30.])


def test_transition_target_cached_numpy_counts_match_reference_with_births_arrivals_and_internal_moves():
    raw = transition_batch()
    capacity = np.array([10., 20., 30.], dtype=np.float32)
    labels = np.array([[1., 1., 0.], [0., 0., 1.], [0., 1., 1.]], dtype=np.float32) / capacity
    target = raw.flow[:, 1:].clone()
    target[0, :, :, 2] = torch.from_numpy(labels)
    cached = replace(raw, boundary_vehicles=None, arrival_times=None,
        transition_target_flow=target, transition_cohort_count=torch.tensor([[2, 3, 3]]),
        transition_volume_scope='previous_state_excluding_current_births',
        transition_capacity_identity=cohort_capacity_identity(capacity))
    expected = prepare_transition_targets(raw, capacities=capacity)
    actual = prepare_transition_targets(cached, capacities=capacity)
    assert actual is cached
    torch.testing.assert_close(actual.transition_target_flow, expected.transition_target_flow)
    torch.testing.assert_close(actual.transition_cohort_count, expected.transition_cohort_count)
    assert actual.action_events is raw.action_events and actual.action is raw.action


def test_empty_interval_start_keeps_newborn_actions_and_exposure_in_both_paths():
    raw = transition_batch()
    flow = raw.flow.clone()
    flow[0, 0, :, 2] = 0.
    boundaries = (((), *raw.boundary_vehicles[0][1:]),)
    raw = replace(raw, flow=flow, boundary_vehicles=boundaries)
    reference = prepare_transition_targets(raw, capacities=[10., 20., 30.])
    assert reference.transition_cohort_count[0, 0] == 0
    assert reference.transition_target_flow[0, 0, :, 2].eq(0).all()
    assert reference.action_events.vehicle_ids[:2] == ('A', 'B')
    assert reference.active_vehicle_fraction[0, 0] > 0
    assert reference.action[0, 0].sum() > 0
    cached = replace(reference, boundary_vehicles=None, arrival_times=None)
    actual = prepare_transition_targets(cached, capacities=[10., 20., 30.])
    assert actual is cached and actual.action_events is raw.action_events
    assert actual.active_vehicle_fraction is raw.active_vehicle_fraction
