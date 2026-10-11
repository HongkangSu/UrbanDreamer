from dataclasses import replace

import numpy as np
import pytest

from urban_dreamer.learning.observations import build_actor_observations, prepare_actor_window
from urban_dreamer.traffic.topology import NextHopCandidates
from test_urban_dreamer_observations import _real_candidates


def _fixture():
    topology, candidates = _real_candidates('toy_cross_bottleneck')
    network = topology.network
    demand = topology.vehicles[0]
    origin = network.edge_to_index[topology.origins[demand.origin_index]]
    destination = network.edge_to_index[topology.destinations[demand.destination_index]]
    state = dict(current_edge=origin, destination_edge=destination, remaining_distance=5.)
    edges = np.random.default_rng(8).random((network.edge_count, 2), dtype=np.float32)
    return candidates, state, edges


def _build(candidates, states, edges, progress=.3, population=.4):
    return build_actor_observations(policy_edge_features=edges, vehicle_states=states,
                                    candidates=candidates, episode_progress=progress,
                                    active_vehicle_fraction=population)


def test_cache_preserves_features_and_dynamic_changes_without_aliasing():
    candidates, state, edges = _fixture()
    cold = _build(candidates, [state, state], edges)
    warm = _build(candidates, [state, state], prepare_actor_window(edges, candidates.network.edge_count))
    for field in ('features', 'mask', 'edge_indices'):
        np.testing.assert_array_equal(getattr(cold, field), getattr(warm, field))
    cold.features[:] = -100
    cold.mask[:] = False
    cold.edge_indices[:] = -1
    repeated = _build(candidates, [state], edges)
    np.testing.assert_array_equal(repeated.features[0], warm.features[0])
    changed = _build(candidates, [dict(state, remaining_distance=25.)], edges * .5, .8, .9)
    mask = changed.mask[0]
    np.testing.assert_array_equal(changed.features[0, mask][:, [4, 5, 7]],
                                  warm.features[0, mask][:, [4, 5, 7]])
    np.testing.assert_array_equal(changed.features[0, mask, :2], edges[changed.edge_indices[0, mask]] * .5)
    assert (changed.features[0, mask, 6] != warm.features[0, mask, 6]).all()
    np.testing.assert_array_equal(changed.features[0, mask, 8], np.float32(.8))
    np.testing.assert_array_equal(changed.features[0, mask, 9], np.float32(.9))
    assert not hasattr(candidates, '_junction_matrices')


def test_cache_is_bounded_and_scoped_to_candidate_instance():
    candidates, state, edges = _fixture()
    candidates._actor_static_cache_limit = 1
    first = _build(candidates, [state], edges)
    choice = candidates.candidates(state['current_edge'], state['destination_edge'])[0]
    other_state = dict(state, current_edge=choice)
    _build(candidates, [other_state], edges)
    assert len(candidates._actor_static_cache) == 1
    for template in candidates._actor_static_cache.values():
        for array in (template.features, template.mask, template.edge_indices):
            assert array.flags.owndata and not array.flags.writeable
    assert next(iter(candidates._actor_static_cache)) == (choice, state['destination_edge'])
    np.testing.assert_array_equal(_build(candidates, [state], edges).features, first.features)
    network = replace(candidates.network, free_flow_travel_seconds=candidates.network.free_flow_travel_seconds * 2)
    other = NextHopCandidates(network)
    changed = _build(other, [state], edges)
    assert other._actor_static_cache is not candidates._actor_static_cache
    assert (changed.features[..., 4][changed.mask] != first.features[..., 4][first.mask]).all()


@pytest.mark.parametrize("order", ["C", "F"])
def test_prepared_window_owns_snapshot_validates_input_and_refreshes_mean(order):
    candidates, state, edges = _fixture()
    edges = np.array(edges, order=order)
    prepared = prepare_actor_window(edges, candidates.network.edge_count)
    saved = edges.copy(order="K")
    edges[:] = 9.
    np.testing.assert_array_equal(prepared.edge_features, saved)
    np.testing.assert_array_equal(prepared.global_features, saved.mean(axis=0))
    assert not prepared.edge_features.flags.writeable
    assert not prepared.global_features.flags.writeable
    observation = _build(candidates, [state], prepared)
    changed = _build(candidates, [state], prepare_actor_window(edges, candidates.network.edge_count))
    assert (changed.features[..., 2:4][changed.mask] != observation.features[..., 2:4][observation.mask]).all()
    for invalid in (np.zeros((1, 2)), np.full_like(edges, np.nan), np.full_like(edges, np.inf)):
        with pytest.raises(ValueError, match='finite with shape'):
            prepare_actor_window(invalid, candidates.network.edge_count)


def test_warm_cache_avoids_static_lookup_and_does_not_advance_rng(monkeypatch):
    candidates, state, edges = _fixture()
    prepared = prepare_actor_window(edges, candidates.network.edge_count)
    expected = _build(candidates, [state], prepared)
    def unexpected(*args, **kwargs):
        raise AssertionError('static lookup repeated on a warm pair')
    for name in ('candidates', 'costs', 'distances', 'junction'):
        monkeypatch.setattr(candidates, name, unexpected)
    before = np.random.get_state()
    actual = _build(candidates, [state], prepared)
    after = np.random.get_state()
    np.testing.assert_array_equal(expected.features, actual.features)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_prepared_empty_batch_preserves_output_shapes_and_invalid_indices():
    candidates, state, edges = _fixture()
    prepared = prepare_actor_window(edges, candidates.network.edge_count)
    empty = _build(candidates, [], prepared)
    assert empty.features.shape == (0, candidates.max_candidates, 10)
    with pytest.raises(ValueError, match='unknown current edge'):
        _build(candidates, [dict(state, current_edge=-1)], prepared)
    with pytest.raises(ValueError, match='unknown destination edge'):
        _build(candidates, [dict(state, destination_edge=candidates.network.edge_count)], prepared)
