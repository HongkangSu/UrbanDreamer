from types import SimpleNamespace
import numpy as np
from urban_dreamer.learning.observations import build_actor_observations, build_vehicle_observations, aggregate_next_hop_actions


class Candidates:
    max_candidates = 3
    network = SimpleNamespace(edge_count=4, lengths_m=np.array([10.,20.,30.,40.]), free_flow_travel_seconds=np.array([1.,2.,3.,4.]))
    def candidates(self, current, destination):
        return (1,2) if destination == 3 else (2,)
    def distances(self, destination):
        return np.array([70.,40.,40.,0.]) if destination == 3 else np.array([30.,30.,0.,np.inf])
    def costs(self, destination):
        return self.distances(destination)/10
    def junction(self, current, successor):
        return 0., 1.


def test_candidate_features_use_destination_reachability_and_padding():
    states = [dict(current_edge=0,destination_edge=3,remaining_distance=5),dict(current_edge=0,destination_edge=2,remaining_distance=5)]
    result = build_actor_observations(policy_edge_features=np.ones((4,2)),vehicle_states=states,candidates=Candidates(),episode_progress=.2,active_vehicle_fraction=.5)
    assert result.features.shape == (2,3,10)
    np.testing.assert_array_equal(result.mask, [[True,True,False],[True,False,False]])
    np.testing.assert_array_equal(result.edge_indices, [[1,2,-1],[2,-1,-1]])
    np.testing.assert_allclose(result.features[0,:2,-2:], [[.2,.5],[.2,.5]])
    assert np.isfinite(result.features).all()


def test_critic_state_preserves_commitment_junction_and_future_departure():
    state = dict(current_edge=0,destination_edge=3,remaining_distance=4,committed_next_edge=2,junction_remaining_distance=3,junction_speed=2,in_junction=True,departed=True,arrived=False,departure_time=0)
    future = dict(state,departed=False,in_junction=False,departure_time=60)
    result = build_vehicle_observations(policy_edge_features=np.ones((4,2)),vehicle_states=[state,future],candidates=Candidates(),episode_progress=.2,active_vehicle_fraction=.5,current_time=30)
    assert result.shape == (2,14)
    assert not np.array_equal(result[0],result[1])
    assert np.isfinite(result).all()


def test_aggregate_counts_each_event_with_episode_denominator():
    decisions=[SimpleNamespace(next_edge=1),SimpleNamespace(next_edge=2),SimpleNamespace(next_edge=1)]
    np.testing.assert_array_equal(aggregate_next_hop_actions(decisions,edge_count=3,total_vehicle_count=2), [0,1,.5])
    np.testing.assert_array_equal(aggregate_next_hop_actions([],edge_count=3,total_vehicle_count=2), [0,0,0])


def test_critic_represents_remaining_junction_time():
    state = dict(current_edge=0,destination_edge=3,remaining_distance=0,
                 committed_next_edge=2,junction_remaining_distance=6,
                 junction_speed=1,in_junction=True,departed=True,arrived=False)
    faster = dict(state,junction_speed=2)
    result = build_vehicle_observations(policy_edge_features=np.ones((4,2)),
        vehicle_states=[state,faster],candidates=Candidates(),
        episode_progress=.2,active_vehicle_fraction=.5,current_time=30)
    np.testing.assert_allclose(result[:,7],[.2,.1])


def test_actor_static_cost_and_distance_include_initial_junction():
    class LongerJunction(Candidates):
        def junction(self, current, successor):
            return (30., 3.) if successor == 1 else (0., 1.)
    result = build_actor_observations(policy_edge_features=np.ones((4,2)),
        vehicle_states=[dict(current_edge=0,destination_edge=3,remaining_distance=5)],
        candidates=LongerJunction(),episode_progress=0.,active_vehicle_fraction=1.)
    assert result.features[0,0,4] == np.float32(np.log1p(10.+2.+4.)/np.log1p(3600.))
    assert result.features[0,0,5] == np.float32(np.log1p(30.+20.+40.)/np.log1p(10000.))
