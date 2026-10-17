"""Both reference policies select full routes from the same restored action space."""
import numpy as np

from test_route_candidates import network
from tools.evaluate_route_shortest import select_shortest_routes
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.traffic.route_candidates import RouteCandidates


def test_static_and_dynamic_shortest_use_same_routes_and_observed_dynamic_costs():
    graph = network()
    candidates = RouteCandidates(graph, traffic_model='meso')
    states = [dict(vehicle_id='v', current_edge=0, destination_edge=4,
                   remaining_distance=graph.lengths_m[0])]
    traffic = np.zeros((graph.edge_count, 2), dtype=np.float32)
    traffic[:, 1] = 1.
    # Route 0 uses edge 2; the alternative shares first hop but uses edge 3.
    first = candidates.generate_paths(0, 4)[0]
    alternative = candidates.generate_paths(0, 4)[1]
    exclusive = set(first) - set(alternative)
    assert exclusive
    traffic[list(exclusive), 0] = 1.
    traffic[list(exclusive), 1] = .001
    observations = build_actor_observations(policy_edge_features=traffic,
        vehicle_states=states, candidates=candidates, episode_progress=0., active_vehicle_fraction=1.,
        prior_cost_mode='dynamic', prior_speed_floor_mps=.5, prior_speed_cap_mps=40.)
    static = select_shortest_routes(states, observations, None, 30., candidates=candidates, dynamic=False)
    dynamic = select_shortest_routes(states, observations, None, 30., candidates=candidates, dynamic=True)
    assert static[0].route == first
    assert dynamic[0].route == alternative
    assert static[0].next_edge == dynamic[0].next_edge == 1
    assert static[0].decision_time == dynamic[0].decision_time == 30.
