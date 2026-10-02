import pytest

from tools.plot_training_revisits import event_revisits, pooled


def event(vehicle, source, target=999):
    return {'vehicle_id': vehicle, 'current_edge': source, 'next_edge': target}


def test_counts_repeated_observed_sources_per_vehicle_not_requested_targets():
    result = event_revisits([event('a', 1), event('b', 1), event('a', 2),
                             event('a', 1), event('a', 1)])
    assert result['decision_events'] == 5
    assert result['repeated_source_events'] == 2
    assert result['event_revisit_ratio'] == .4
    assert result['vehicle_revisit_ratio'] == .5
    assert event_revisits([event('a', 1), event('b', 1)])['repeated_source_events'] == 0


def test_no_events_are_undefined_and_vehicle_names_do_not_change_metric():
    result = event_revisits([])
    assert result['event_revisit_ratio'] is None and result['vehicle_revisit_ratio'] is None
    assert event_revisits([event('x', 1), event('x', 1)]) == event_revisits([event('y', 1), event('y', 1)])
    # A new episode resets visit history even when an ID is reused.
    assert event_revisits([event('x', 1)])['event_revisit_ratio'] == 0


def test_pooling_uses_counts_and_keeps_no_event_episodes_out_of_denominators():
    a = dict(event_revisits([event('a', 1), event('a', 1)]), population=2, completion_rate=.5)
    b = dict(event_revisits([event('b', i) for i in range(8)]), population=1, completion_rate=1.)
    empty = dict(event_revisits([]), population=3, completion_rate=0.)
    result = pooled([a, b, empty])
    assert result['event_revisit_ratio'] == pytest.approx(.1)
    assert result['vehicle_revisit_ratio'] == .5
    assert result['decision_vehicle_coverage'] == pytest.approx(2/6)
    assert result['mean_completion_rate'] == .5
