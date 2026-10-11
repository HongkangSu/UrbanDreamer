"""Physical history is preserved across hidden roads, windows, and seed replay."""
from copy import deepcopy

import numpy as np
import pytest

from test_event_imagination_protocol import network, seed
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.traffic.topology import NextHopCandidates


def run(rows, topology, *, protocol='event', start=0., seconds=1., decide=None, enabled=True):
    candidates = NextHopCandidates(topology, forbid_revisit=enabled)
    if decide is None:
        def decide(pending, flow, now):
            return [NextHopDecision(row['vehicle_id'], row['current_edge'],
                candidates.legal_candidates(row['current_edge'], row['destination_edge'],
                    row.get('visited_edges', ()))[0], now) for row in pending]
    return advance_imagined_window(rows, candidates=candidates,
        edge_speeds_mps=np.ones(4) * 10., frozen_flow=np.zeros((4, 8)),
        start_time=start, interval_seconds=seconds, total_vehicle_count=max(1, len(rows)),
        decide=decide, session_protocol=protocol)


@pytest.mark.parametrize('protocol', ['event', 'legacy'])
def test_history_survives_windows_and_destination_entry(protocol):
    topology = network((10., 10., 10., 10.), graph=((1,), (2,), (3,), ()))
    rows = [seed(visited_edges=(0,))]
    first = run(rows, topology, protocol=protocol)
    assert rows[0]['visited_edges'] == (0, 1)
    assert not first.arrivals
    second = run(rows, topology, protocol=protocol, start=1., seconds=3.)
    assert rows[0]['visited_edges'] == (0, 1, 2, 3)
    assert second.arrivals == {'v': 4.}
    assert first.active_vehicle_seconds + second.active_vehicle_seconds == 4.


def test_hidden_singleton_roads_and_destination_are_all_recorded():
    rows = [seed(remaining_distance=1., visited_edges=(0,))]
    topology = network((1., 1., 1., 1.), graph=((1,), (2,), (3,), ()))
    result = run(rows, topology)
    assert not result.decisions
    assert rows[0]['visited_edges'] == (0, 1, 2, 3)
    assert result.arrivals == {'v': 1.}


@pytest.mark.parametrize('protocol', ['event', 'legacy'])
def test_policy_cannot_choose_visited_edge(protocol):
    topology = network(graph=((1,), (0, 2, 3), (3,), ()))
    rows = [seed(current_edge=1, committed_next_edge=3, pending_decision=True,
                 visited_edges=(0, 1))]
    def invalid(rows, flow, now):
        return [NextHopDecision('v', 1, 0, now)]
    with pytest.raises(ValueError, match='outside the legal candidate set'):
        run(rows, topology, protocol=protocol, decide=invalid)
    assert rows[0]['visited_edges'] == (0, 1)
    assert rows[0]['committed_next_edge'] == 3


@pytest.mark.parametrize('protocol', ['event', 'legacy'])
def test_old_seed_history_cannot_be_reset_or_omitted(protocol):
    topology = network()
    with pytest.raises(ValueError, match='requires complete visited_edges'):
        run([seed()], topology, protocol=protocol)
    with pytest.raises(ValueError, match='no no-revisit destination-reachable successor'):
        run([seed(current_edge=2, committed_next_edge=1, visited_edges=(0, 1, 2))],
            topology, protocol=protocol)
    with pytest.raises(ValueError, match='repeated physical roads'):
        run([seed(visited_edges=(0, 1, 0))], topology, protocol=protocol)


def test_pruned_fork_becomes_automatic_without_a_policy_event():
    rows = [seed(visited_edges=(0,))]
    result = run(rows, network(), seconds=3.)
    assert not result.decisions
    assert rows[0]['visited_edges'] == (0, 1, 3)
    assert rows[0]['current_edge'] == 3


def test_departure_records_origin_once_then_keeps_history():
    rows = [seed(departed=False, committed_next_edge=None, visited_edges=(),
                 departure_time=1.)]
    first = run(rows, network())
    assert rows[0]['visited_edges'] == (0,)
    np.testing.assert_array_equal(first.departure, [1., 0., 0., 0.])
    run(rows, network(), start=1., seconds=3.)
    assert rows[0]['visited_edges'] == (0, 1, 3)


@pytest.mark.parametrize('protocol', ['event', 'legacy'])
def test_default_mode_ignores_history_and_preserves_legacy_outputs(protocol):
    topology = network()
    plain = [seed()]
    with_history = deepcopy(plain)
    with_history[0]['visited_edges'] = (0, 1, 0)
    a = run(plain, topology, protocol=protocol, enabled=False, seconds=3.)
    b = run(with_history, topology, protocol=protocol, enabled=False, seconds=3.)
    history = with_history[0].pop('visited_edges')
    assert history == (0, 1, 0)
    assert plain == with_history
    assert a.decisions == b.decisions
    assert a.arrivals == b.arrivals
    assert a.cost_segments == b.cost_segments
    np.testing.assert_array_equal(a.action, b.action)


@pytest.mark.parametrize('protocol', ['event', 'legacy'])
def test_cohort_histories_are_separate_and_junction_entry_waits_for_completion(protocol):
    topology = network((10., 10., 10., 1000.), graph=((1,), (2, 3), (3,), ()),
                       junction_lengths={(0, 1): 20.})
    rows = [seed(visited_edges=(0,)),
            seed('other', current_edge=2, committed_next_edge=3,
                 visited_edges=(2,), pending_decision=False)]
    snapshots = tuple(row['visited_edges'] for row in rows)
    run(rows, topology, protocol=protocol, seconds=2.)
    assert rows[0]['in_junction']
    assert rows[0]['visited_edges'] == (0,)
    assert rows[1]['visited_edges'] == (2, 3)
    run(rows, topology, protocol=protocol, start=2., seconds=1.)
    assert rows[0]['visited_edges'] == (0, 1)
    assert rows[1]['visited_edges'] == (2, 3)
    assert snapshots == ((0,), (2,))
