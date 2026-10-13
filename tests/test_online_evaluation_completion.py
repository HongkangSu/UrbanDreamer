"""Deadline termination must not imply that all requested trips completed."""
from types import SimpleNamespace

from tools.run_online_pilot import compare_paired, evaluation_row


def collected(*, nonarrivals, terminal, minutes=4.):
    return SimpleNamespace(total_vehicle_count=10, total_travel_minutes=minutes,
        transitions=[SimpleNamespace(terminal=terminal)], metrics={
            'evaluation_nonarrival_count': nonarrivals,
            'evaluation_completion_rate': (10-nonarrivals)/10,
            'evaluation_selection_score': minutes})


def test_deadline_terminal_keeps_cost_censored_when_trips_are_unfinished():
    row = evaluation_row(collected(nonarrivals=7, terminal=True), 0, 'demand')
    assert row['episode_terminal'] is True
    assert row['complete'] is False
    assert row['travel_time_is_censored'] is True
    assert row['travel_minutes_per_vehicle'] is None
    assert row['observed_minutes_per_vehicle'] == 4.
    assert row['arrived_count'] == 3


def test_all_arrivals_allow_complete_trip_cost():
    row = evaluation_row(collected(nonarrivals=0, terminal=True), 0, 'demand')
    assert row['complete'] is True
    assert row['episode_terminal'] is True
    assert row['travel_time_is_censored'] is False
    assert row['travel_minutes_per_vehicle'] == 4.


def test_deadline_pair_never_reports_a_completed_trip_improvement():
    before = evaluation_row(collected(nonarrivals=7, terminal=True), 0, 'demand')
    after = evaluation_row(collected(nonarrivals=5, terminal=True, minutes=3.), 0, 'demand')
    result = compare_paired([before], [after])
    assert result['pairs'][0]['completed_travel_minutes_delta'] is None
    assert result['pairs'][0]['observed_minutes_delta'] == -1.


def test_legacy_terminal_flags_cannot_make_unfinished_pairs_complete():
    before = evaluation_row(collected(nonarrivals=7, terminal=True), 0, 'demand')
    after = evaluation_row(collected(nonarrivals=5, terminal=True, minutes=3.), 0, 'demand')
    for row in (before, after):
        row['complete'] = True
        row['travel_time_is_censored'] = False
        row['travel_minutes_per_vehicle'] = row['observed_minutes_per_vehicle']
    pair = compare_paired([before], [after])['pairs'][0]
    assert pair['censored_pair'] is True
    assert pair['completed_travel_minutes_delta'] is None
