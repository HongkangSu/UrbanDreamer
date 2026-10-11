from dataclasses import replace

import numpy as np
import pytest

from urban_dreamer.traffic.topology import EdgeNetwork, NextHopCandidates, TopologyError


def network():
    graph = ((1, 2, 4), (3,), (3,), (), (4,))
    return EdgeNetwork(tuple('abcde'), dict(zip('abcde', range(5))), np.ones((5, 4)),
                       np.ones(5) * 10, np.ones(5), np.array([10., 20., 30., 40., 10.]),
                       np.zeros((5, 5)), graph, np.ones(5, dtype=bool),
                       np.array([1., 2., 3., 4., 1.]))


def test_candidates_filter_unreachable_and_depend_on_destination():
    candidates = NextHopCandidates(network())
    assert candidates.candidates(0, 3) == (1, 2)
    assert candidates.candidates(0, 4) == (4,)
    assert candidates.candidates(3, 3) == ()
    assert candidates.max_candidates == 3
    with pytest.raises(TopologyError, match='reachable'):
        candidates.candidates(1, 4)


def test_distance_and_placeholder_are_not_complete_routes():
    candidates = NextHopCandidates(network())
    assert candidates.distances(3)[0] == 60
    assert candidates.distances(3)[3] == 0
    assert candidates.temporary_route(0, 1, 3) == (0, 1, 3)
    assert candidates.temporary_route(1, 3, 3) == (1, 3)
    with pytest.raises(TopologyError):
        candidates.temporary_route(0, 4, 3)


def scalar_flow_reference(network, states):
    result = np.empty((network.edge_count, 8), np.float32)
    for i, edge in enumerate(network.edge_ids):
        state = states.get(edge)
        speed = max(float(network.speed_limits_mps[i]), .1)
        capacity = max(float(network.capacities[i]), 1.)
        if state is None:
            values = (0., 1., 0., 0.)
        else:
            occupancy = float(state.get('occupancy', 0.))
            if occupancy > 1.: occupancy /= 100.
            values = (float(np.clip(occupancy, 0., 1.)),
                      float(state.get('mean_speed', speed))/speed,
                      float(state.get('vehicle_number', 0.))/capacity,
                      float(state.get('halting_number', 0.))/capacity)
        result[i, :4], result[i, 4:] = values, network.static_features[i]
    return result


@pytest.mark.parametrize('occupancy', [-2., -0., .005, .5, 1., 1.01, 100., 150., np.nan])
def test_flow_batch_conversion_preserves_percent_defaults_and_precision(occupancy):
    graph = replace(network(), speed_limits_mps=np.array([0., .02, 13.7, np.nan, 10.], np.float32),
                    capacities=np.array([0., .5, 3.7, np.nan, 10.], np.float32))
    states = {'a':dict(occupancy=occupancy, mean_speed=0., vehicle_number=2., halting_number=1.),
              'b':{}, 'c':dict(occupancy=.005, mean_speed=7.333333333, vehicle_number=7),
              'd':None, 'unregistered':dict(occupancy=99.)}
    expected = scalar_flow_reference(graph, states)
    actual = graph.extract_flow(states)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(graph.extract_flow({}), scalar_flow_reference(graph, {}))


def test_complete_route_cache_preserves_model_costs_ties_and_validates_next_edge():
    candidates = NextHopCandidates(network(), traffic_model='meso')
    assert candidates.full_route(0, 1, 3) == (0, 1, 3)
    assert candidates.full_route(0, 2, 3) == (0, 2, 3)
    for _ in range(3):
        assert candidates.full_route(0, 1, 3) == (0, 1, 3)
    with pytest.raises(TopologyError, match='invalid next edge'):
        candidates.full_route(0, 4, 3)


def test_complete_route_cache_is_bounded_and_tracks_cost_table_identity():
    candidates = NextHopCandidates(network(), traffic_model='meso')
    candidates._full_route_cache_limit = 1
    candidates.full_route(0, 1, 3)
    candidates.full_route(0, 2, 3)
    assert len(candidates._full_route_cache) == 1
    assert (1, 3) not in candidates._full_route_cache
    table = candidates.costs(3)
    replacement = table.copy()
    replacement.setflags(write=False)
    candidates.costs = lambda destination: replacement
    assert candidates.full_route(0, 2, 3) == (0, 2, 3)
    assert candidates._full_route_cache[(2, 3)][0] is replacement


def test_original_passenger_lanes_control_connectivity_and_progress(tmp_path):
    from urban_dreamer.traffic.topology import _load_connections
    path = tmp_path / 'net.xml'
    path.write_text('''<net>
      <edge id="a"><lane id="a_0" index="0" length="100" speed="10" allow="bus"/>
                   <lane id="a_1" index="1" length="0.1" speed="10" allow="passenger"/></edge>
      <edge id="b"><lane id="b_0" index="0" length="80" speed="10"/></edge>
      <edge id="c"><lane id="c_0" index="0" length="80" speed="10"/></edge>
      <edge id=":j_0"><lane id=":j_0_0" index="0" length="5" speed="8"/></edge>
      <edge id=":j_1"><lane id=":j_1_0" index="0" length="7" speed="6"/></edge>
      <connection from="a" to="b" fromLane="1" toLane="0" via=":j_0_0"/>
      <connection from="a" to="c" fromLane="0" toLane="0"/>
      <connection from=":j_0" to="b" fromLane="0" toLane="0" via=":j_1_0"/>
      <connection from=":j_1" to="b" fromLane="0" toLane="0"/>
    </net>''')
    actual = _load_connections(network(), path)
    assert actual.adjacency_list[0] == (1,)
    assert min(actual.external_lane_lengths_m.values()) == .1
    assert 'a_0' not in actual.external_lane_lengths_m
    assert actual.internal_lane_remaining_lengths_m == {':j_0_0': 12., ':j_1_0': 7.}
    assert NextHopCandidates(actual).junction(0, 1) == (12., 6.)
