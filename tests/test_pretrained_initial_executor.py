import copy

import pytest

from tools.analyze_pretrained_initial_executor import compare_events, assert_saved_reproduction


def event(name, time=0., current=1, next_edge=2):
    return dict(vehicle_id=name, decision_time=time, current_edge=current, next_edge=next_edge)


def test_events_align_by_vehicle_occurrence_without_removing_unmatched_decisions():
    left = [event('a'), event('b'), event('a', 2.)]
    right = [event('a', 1., next_edge=3), event('a', 2.), event('c')]
    result = compare_events(left, right)
    assert result['matched_occurrences'] == 2
    assert result['next_hop_changes'] == 1
    assert result['timestamp_changes'] == 1
    assert result['source_edge_changes'] == 0
    assert result['only_left'] == result['only_right'] == 1
    assert not result['event_sequences_equal']


def test_equal_events_and_input_preservation():
    rows = [event('a')]
    original = copy.deepcopy(rows)
    result = compare_events(rows, rows)
    assert result['event_sequences_equal']
    assert result['next_hop_changes'] == result['timestamp_changes'] == 0
    assert rows == original


@pytest.mark.parametrize('field,value', [('vehicle_seconds', 60.0000000001),
    ('own_action_events', [event('a', 1.)]), ('explicit_cohort_count', [2, 0])])
def test_reproduction_rejects_cost_event_or_endpoint_count_drift(field, value):
    result = dict(vehicle_seconds=60., own_action_events=[event('a')], explicit_cohort_count=[1, 1])
    saved = dict(windows=[dict(vehicle_seconds=60., own_action_events=[event('a')])],
                 frames=[{}, dict(explicit_cohort_count=[1, 1])])
    assert_saved_reproduction(result, saved)
    result[field] = value
    with pytest.raises(ValueError, match='not reproduced exactly'):
        assert_saved_reproduction(result, saved)
