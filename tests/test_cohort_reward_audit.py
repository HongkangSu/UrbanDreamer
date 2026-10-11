from copy import deepcopy

import pytest

from tools.audit_cohort_reward import (
    RecordedPhysicalRouteReplay, RecordedRouteReplay, RouteReplayUnavailable,
    cohort_window_labels, real_cohort_label,
)
from urban_dreamer.learning.global_reward import RewardAccounting


def archive():
    seed = [
        dict(vehicle_id='old', departed=True, arrived=True, arrival_time=9., actual_departure_time=0.),
        dict(vehicle_id='a', departed=True, arrived=False, current_edge=1, destination_edge=9),
        dict(vehicle_id='b', departed=True, arrived=False, current_edge=2, destination_edge=9),
        dict(vehicle_id='future', departed=False, arrived=False),
    ]
    final = deepcopy(seed)
    final[1].update(arrived=True, arrival_time=15.5)
    final[3].update(departed=True, arrived=True, actual_departure_time=12., arrival_time=18.)
    return dict(states=[dict(time=10., vehicles=seed), dict(time=30., vehicles=final)],
                decisions=[], travel_times={})


def test_fixed_seed_cohort_fractional_arrival_and_censored_endpoint():
    row = real_cohort_label(archive(), 0, 40.)
    assert row.vehicle_ids == ('a', 'b')
    assert row.started_count == 3  # Includes old arrivals, excludes future births.
    assert row.end_time == 30.
    assert row.vehicle_seconds == 25.5
    assert row.count_integral_seconds == row.vehicle_seconds
    assert row.reward == pytest.approx(-25.5 / 180.)
    assert row.arrivals == 1


def test_windows_telescope_and_match_production_ledger_with_historical_cost():
    data = archive()
    rows = cohort_window_labels(data, 0, [15.5, 20., 30.])
    assert [r.vehicle_seconds for r in rows] == [11., 4.5, 10.]
    assert [r.arrivals for r in rows] == [1, 0, 0]
    assert all(r.count_integral_seconds == r.vehicle_seconds for r in rows)
    whole = real_cohort_label(data, 0, 30.)
    ledger = RewardAccounting(started_count=3, cumulative_vehicle_seconds=137.)
    rewards = [ledger.update(r.vehicle_seconds, r.started_count) for r in rows]
    assert rewards == pytest.approx([r.reward for r in rows])
    assert sum(rewards) == pytest.approx(whole.reward)
    assert sum(r.reward for r in rows) == pytest.approx(whole.reward)


def test_full_population_change_does_not_change_cohort_label():
    data = archive()
    before = real_cohort_label(data, 0, 30.)
    data['states'][-1]['started_vehicle_count'] = 100
    assert real_cohort_label(data, 0, 30.) == before
    full = RewardAccounting(started_count=3, cumulative_vehicle_seconds=137.)
    assert full.update(before.vehicle_seconds + 6., 4) != pytest.approx(before.reward)


def test_observed_summary_and_travel_time_fallback():
    data = archive()
    data['states'][-1]['vehicles'][1].pop('arrival_time')
    data['arrival_times'] = {'a': 15.5}
    assert real_cohort_label(data, 0, 30.).vehicle_seconds == 25.5
    data.pop('arrival_times')
    data['states'][-1]['vehicles'][1]['actual_departure_time'] = 5.
    data['travel_times']['a'] = 10.5
    assert real_cohort_label(data, 0, 30.).vehicle_seconds == 25.5


def test_empty_cohort_and_zero_duration():
    data = archive()
    assert real_cohort_label(data, 0, 10.).reward == 0.
    for v in data['states'][0]['vehicles']:
        v.update(departed=False, arrived=False)
    row = real_cohort_label(data, 0, 30.)
    assert row.started_count == 0
    assert row.vehicle_seconds == row.count_integral_seconds == row.reward == 0.


@pytest.mark.parametrize('end', [9., float('nan'), float('inf')])
def test_invalid_bounds_rejected(end):
    with pytest.raises(ValueError):
        real_cohort_label(archive(), 0, end)


class Candidates:
    def candidates(self, edge, destination):
        return (edge + 1, 9)


def event(edge, target, time, vehicle='a'):
    return dict(vehicle_id=vehicle, current_edge=edge, next_edge=target, decision_time=time)


def pending(edge=1, vehicle='a'):
    return dict(vehicle_id=vehicle, current_edge=edge, destination_edge=9)


def test_replay_keeps_seed_boundary_and_future_beyond_audit_horizon():
    data = archive()
    data['decisions'] = [event(0, 1, 9.), event(1, 2, 10.), event(2, 9, 100.)]
    replay = RecordedRouteReplay(data, 0, Candidates())
    assert replay([pending()], None, 10.)[0].next_edge == 2
    decision = replay([pending(2)], None, 11.)[0]
    assert decision.next_edge == 9 and decision.decision_time == 11.
    with pytest.raises(RouteReplayUnavailable, match='exhausted'):
        replay([pending(3)], None, 12.)


def test_replay_never_searches_ahead_to_match_later_repeated_edge():
    data = archive()
    data['decisions'] = [event(2, 3, 10.), event(1, 2, 15.)]
    replay = RecordedRouteReplay(data, 0, Candidates())
    with pytest.raises(RouteReplayUnavailable, match='ordered current edge'):
        replay([pending()], None, 10.)
    assert replay.cursor['a'] == 0


def test_invalid_next_hop_rejected_and_batch_cursor_unchanged():
    data = archive()
    data['decisions'] = [event(1, 2, 10.), event(2, 8, 10., 'b')]
    replay = RecordedRouteReplay(data, 0, Candidates())
    with pytest.raises(RouteReplayUnavailable, match='not legal'):
        replay([pending(), pending(2, 'b')], None, 10.)
    assert replay.cursor['a'] == replay.cursor['b'] == 0


def test_no_revisit_candidates_used_when_enabled():
    class NoRevisit(Candidates):
        forbid_revisit = True

        def legal_candidates(self, edge, destination, history):
            return tuple(v for v in self.candidates(edge, destination) if v not in history)

    data = archive()
    data['decisions'] = [event(1, 2, 10.)]
    replay = RecordedRouteReplay(data, 0, NoRevisit())
    with pytest.raises(RouteReplayUnavailable, match='not legal'):
        replay([{**pending(), 'visited_edges': [2, 1]}], None, 10.)


def physical_archive(*, arrived=True, pending_seed=False):
    seed = dict(vehicle_id='a', departed=True, arrived=False, current_edge=1,
                destination_edge=9, committed_next_edge=2, pending_decision=pending_seed)
    final = {**seed, 'current_edge': 9 if arrived else 3, 'arrived': arrived}
    return dict(states=[dict(time=10., vehicles=[seed]), dict(time=30., vehicles=[final])],
                decisions=[event(3, 4, 20.)])


class RouteCandidates:
    forbid_revisit = False

    def full_route(self, current, next_edge, destination):
        tails = {2: (2, 3, 9), 4: (4, 9), 5: (5, 9)}
        return (current,) + tails[next_edge]

    def candidates(self, edge, destination):
        return {1: (2, 5), 2: (3, 5), 3: (4, 9), 4: (9,), 5: (9,)}[edge]


def test_physical_replay_covers_skipped_real_observation_using_commanded_suffix():
    replay = RecordedPhysicalRouteReplay(physical_archive(), 0, RouteCandidates())
    assert replay.paths['a'] == (1, 2, 3, 4, 9)
    replay.validate_entry(pending(), 2)
    assert replay([pending(2)], None, 11.)[0].next_edge == 3
    replay.validate_entry(pending(2), 3)
    assert replay([pending(3)], None, 12.)[0].next_edge == 4
    with pytest.raises(RouteReplayUnavailable, match='physical route diverged'):
        replay.validate_entry(pending(3), 9)


def test_physical_replay_stops_at_unfinished_final_observed_edge():
    replay = RecordedPhysicalRouteReplay(physical_archive(arrived=False), 0, RouteCandidates())
    assert replay.paths['a'] == (1, 2, 3)
    replay.validate_entry(pending(), 2)
    replay.validate_entry(pending(2), 3)
    with pytest.raises(RouteReplayUnavailable, match='no recorded successor'):
        replay([pending(3)], None, 12.)


def test_physical_replay_keeps_exact_pending_seed_action():
    data = physical_archive(pending_seed=True)
    data['decisions'] = [event(1, 5, 10.)]
    replay = RecordedPhysicalRouteReplay(data, 0, RouteCandidates())
    assert replay.paths['a'] == (1, 5, 9)
    assert replay([pending()], None, 10.)[0].next_edge == 5


def test_physical_replay_rejects_archived_event_off_commanded_route():
    data = physical_archive()
    data['decisions'] = [event(5, 9, 20.)]
    with pytest.raises(RouteReplayUnavailable, match='not on commanded suffix'):
        RecordedPhysicalRouteReplay(data, 0, RouteCandidates())
