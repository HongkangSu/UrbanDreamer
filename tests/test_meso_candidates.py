import numpy as np
import pytest
from types import SimpleNamespace

from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.traffic.topology import EdgeNetwork, NextHopCandidates, TopologyError
from urban_dreamer.learning.sumo_sampling import SumoEpisodeSampler


def network_with_internal_delay():
    graph = ((1, 2), (3,), (3,), ())
    return EdgeNetwork(
        edge_ids=('a', 'b', 'c', 'd'), edge_to_index=dict(zip('abcd', range(4))),
        static_features=np.ones((4, 4), np.float32), speed_limits_mps=np.ones(4, np.float32),
        capacities=np.ones(4, np.float32), lengths_m=np.full(4, 10., np.float32),
        adjacency=np.asarray([[int(j in targets) for j in range(4)] for targets in graph], np.float32),
        adjacency_list=graph, passenger_allowed=np.ones(4, bool),
        free_flow_travel_seconds=np.full(4, 10., np.float32),
        connection_lengths_m={(0, 1): 100., (1, 3): 100.},
        connection_speeds_mps={(0, 1): 1., (1, 3): 1.},
    )


def test_meso_route_costs_drop_internal_lanes_without_mutating_micro_network():
    network = network_with_internal_delay()
    micro = NextHopCandidates(network)
    old_costs = micro.costs(3).copy()
    meso = NextHopCandidates(network, traffic_model='meso')
    assert micro.candidates(0, 3) == meso.candidates(0, 3) == (1, 2)
    assert micro.costs(3)[0] == 20.
    assert micro.costs(3)[1] == 110.
    assert meso.costs(3)[1] == 10.
    assert meso.distances(3)[1] == 10.
    assert micro.distances(3)[1] == 110.
    assert meso.junction(0, 1) == (0., 1.)
    assert micro.junction(0, 1) == (100., 1.)
    np.testing.assert_array_equal(micro.costs(3), old_costs)
    assert network.connection_lengths_m == {(0, 1): 100., (1, 3): 100.}
    np.testing.assert_array_equal(meso.junction_matrices()[0], 0.)
    assert micro.junction_matrices()[0][0, 1] == 100.


def test_meso_actor_static_features_use_the_same_physical_cost_view():
    network = network_with_internal_delay()
    state = dict(current_edge=0, destination_edge=3, remaining_distance=5.)
    observations = {}
    for model in ('micro', 'meso'):
        candidates = NextHopCandidates(network, traffic_model=model)
        observations[model] = build_actor_observations(
            policy_edge_features=np.ones((4, 2), np.float32), vehicle_states=[state],
            candidates=candidates, episode_progress=.2, active_vehicle_fraction=.5)
    micro, meso = observations['micro'], observations['meso']
    np.testing.assert_array_equal(micro.mask, meso.mask)
    np.testing.assert_array_equal(micro.edge_indices, meso.edge_indices)
    # The first candidate has a costly microscopic connection both now and
    # later; MESO costs describe only the external roads.
    assert meso.features[0, 0, 4] < micro.features[0, 0, 4]
    assert meso.features[0, 0, 5] < micro.features[0, 0, 5]
    np.testing.assert_array_equal(meso.features[0, 1], micro.features[0, 1])


def test_model_views_keep_independent_static_and_route_caches():
    network = network_with_internal_delay()
    micro, meso = NextHopCandidates(network), NextHopCandidates(network, traffic_model='meso')
    assert micro._event_route_cache is not meso._event_route_cache
    assert micro._actor_static_cache is not meso._actor_static_cache
    assert micro.full_route(0, 1, 3) == meso.full_route(0, 1, 3) == (0, 1, 3)
    with pytest.raises(TopologyError, match='traffic_model'):
        NextHopCandidates(network, traffic_model='unknown')


def test_sampler_rejects_physics_mismatch_before_starting_the_session():
    with pytest.raises(ValueError, match='same traffic_model'):
        SumoEpisodeSampler(session=SimpleNamespace(traffic_model='micro'),
            candidates=NextHopCandidates(network_with_internal_delay(), traffic_model='meso'),
            deadline_seconds=30., total_vehicle_count=1)
