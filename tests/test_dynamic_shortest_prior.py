"""Dynamic observed-speed priors never rewrite actor features or topology costs."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.observations import build_actor_observations, prepare_actor_window
from urban_dreamer.learning.dynamic_prior import DynamicTravelCosts
from urban_dreamer.traffic.topology import NextHopCandidates
from urban_dreamer.models.policy import NextHopActor, set_shortest_path_prior
from urban_dreamer.workflow.config import load_config, ConfigError
from test_urban_dreamer_config import _write_config


def candidates():
    network = SimpleNamespace(edge_count=4, edge_ids=('start', 'short', 'long', 'end'),
        adjacency_list=((1, 2), (3,), (3,), ()), passenger_allowed=np.ones(4, dtype=bool),
        lengths_m=np.array([10., 10., 30., 10.]), speed_limits_mps=np.full(4, 10.),
        free_flow_travel_seconds=np.array([1., 1., 3., 1.]),
        connection_lengths_m={(0, 1): 10.}, connection_speeds_mps={(0, 1): 10.})
    return NextHopCandidates(network)


def observe(c, window, **kwargs):
    return build_actor_observations(policy_edge_features=window,
        vehicle_states=[dict(current_edge=0, destination_edge=3)], candidates=c,
        episode_progress=.2, active_vehicle_fraction=.5, **kwargs)


def test_dynamic_congestion_reverses_prior_without_changing_features_or_routes():
    c = candidates()
    static = c.costs(3).copy()
    route = c.full_route(0, 1, 3)
    edges = np.ones((4, 2), dtype=np.float32)
    edges[1, 1] = 0.
    window = prepare_actor_window(edges, 4)
    ordinary = observe(c, window)
    dynamic = observe(c, window, prior_cost_mode='dynamic')
    np.testing.assert_array_equal(ordinary.features, dynamic.features)
    np.testing.assert_array_equal(ordinary.mask, dynamic.mask)
    np.testing.assert_array_equal(c.costs(3), static)
    assert c.full_route(0, 1, 3) == route
    np.testing.assert_allclose(dynamic.prior_costs_seconds, [[22., 4.]])
    actor = NextHopActor()
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
    set_shortest_path_prior(actor, 1., 1.)
    features, mask = torch.tensor(dynamic.features), torch.tensor(dynamic.mask)
    assert actor.sample_action(features, mask, deterministic=True).candidate_index.item() == 0
    assert actor.sample_action(features, mask, deterministic=True,
        prior_costs_seconds=torch.tensor(dynamic.prior_costs_seconds)).candidate_index.item() == 1


def test_snapshot_cache_reused_and_new_snapshot_updates_costs():
    c = candidates()
    edges = np.ones((4, 2), dtype=np.float32)
    window = prepare_actor_window(edges, 4)
    initial = observe(c, window, prior_cost_mode='dynamic')
    cache = next(iter(window._dynamic_prior_cache.values()))
    table = cache.tables[3]
    edges[1, 1] = 0.
    again = observe(c, window, prior_cost_mode='dynamic')
    assert cache.tables[3] is table
    np.testing.assert_array_equal(initial.prior_costs_seconds, again.prior_costs_seconds)
    later = observe(c, prepare_actor_window(edges, 4), prior_cost_mode='dynamic')
    assert later.prior_costs_seconds[0, 0] > initial.prior_costs_seconds[0, 0]


def test_empty_speed_fallback_and_stopped_occupied_floor():
    c = candidates()
    empty = np.zeros((4, 2), dtype=np.float32)
    costs = DynamicTravelCosts(c, empty, .5)
    np.testing.assert_allclose(costs.edge_seconds, c.network.free_flow_travel_seconds)
    empty[1, 0] = .1
    stopped = DynamicTravelCosts(c, empty, .5)
    assert stopped.edge_seconds[1] == 20.
    empty[1, 1] = 1.2
    faster = DynamicTravelCosts(c, empty, .5)
    assert faster.edge_seconds[1] == pytest.approx(10 / 12)


def test_reverse_costs_respect_allowed_edges_and_junctions():
    c = candidates()
    edges = np.ones((4, 2), dtype=np.float32)
    costs = DynamicTravelCosts(c, edges, .5)
    np.testing.assert_allclose(costs.to_destination(3), [3., 1., 1., 0.])
    c.network.passenger_allowed[1] = False
    blocked = DynamicTravelCosts(c, edges, .5)
    assert blocked.to_destination(3)[0] == 4.
    assert np.isinf(blocked.to_destination(3)[1])


def test_prior_costs_detached_masked_and_weight_zero_preserves_logits():
    actor = NextHopActor()
    features = torch.zeros(1, 3, 10)
    mask = torch.tensor([[True, True, False]])
    costs = torch.tensor([[4., 2., float('nan')]], requires_grad=True)
    raw = actor.forward_logits(features, mask)
    assert torch.equal(raw, actor.forward_logits(features, mask, prior_costs_seconds=costs))
    set_shortest_path_prior(actor, 1., 2.)
    biased = actor.forward_logits(features, mask, prior_costs_seconds=costs)
    torch.testing.assert_close(biased[0, :2], raw[0, :2] - torch.tensor([1., 0.]))
    biased[0, :2].sum().backward()
    assert costs.grad is None
    assert torch.isneginf(biased[0, 2])


@pytest.mark.parametrize('setting', ['sp_prior_cost_mode = "bad"',
    'sp_prior_dynamic_speed_floor_mps = 0', 'sp_prior_dynamic_speed_floor_mps = nan',
    'sp_prior_dynamic_speed_floor_mps = true'])
def test_dynamic_config_validation(tmp_path, setting):
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path / 'config.toml', extra='\n[method]\n' + setting))


def test_dynamic_fingerprint_only_when_active_and_static_defaults_compatible(tmp_path, monkeypatch):
    import urban_dreamer.workflow.config as module
    p = _write_config(tmp_path / 'config.toml', extra='\n[method]\nsp_prior_weight = 1.0\n')
    base = p.read_text()
    legacy = load_config(p)
    original = module.asdict
    def old_asdict(value):
        result = original(value)
        if isinstance(value, module.MethodConfig):
            result.pop('sp_prior_cost_mode', None)
            result.pop('sp_prior_dynamic_speed_floor_mps', None)
        return result
    with monkeypatch.context() as patch:
        patch.setattr(module, 'asdict', old_asdict)
        old = module._semantic_fingerprint(**{key: getattr(legacy, key) for key in
            ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert old == legacy.semantic_fingerprint
    p.write_text(base + 'sp_prior_cost_mode = "dynamic"\n')
    dynamic = load_config(p)
    assert dynamic.semantic_fingerprint != old
    p.write_text(p.read_text() + 'sp_prior_dynamic_speed_floor_mps = 1.0\n')
    assert load_config(p).semantic_fingerprint != dynamic.semantic_fingerprint
    p.write_text(base.replace('sp_prior_weight = 1.0', 'sp_prior_weight = 0.0'))
    inactive = load_config(p).semantic_fingerprint
    p.write_text(p.read_text() + 'sp_prior_cost_mode = "dynamic"\n')
    assert load_config(p).semantic_fingerprint == inactive


def test_vehicle_speed_cap_limits_observed_and_empty_freeflow_speeds():
    c = candidates()
    edges = np.ones((4, 2), dtype=np.float32)
    edges[:, 1] = 2.
    uncapped = DynamicTravelCosts(c, edges, .5)
    capped = DynamicTravelCosts(c, edges, .5, speed_cap_mps=5.)
    np.testing.assert_allclose(capped.edge_seconds, c.network.lengths_m / 5.)
    assert (capped.edge_seconds > uncapped.edge_seconds).all()
    empty = DynamicTravelCosts(c, np.zeros_like(edges), .5, speed_cap_mps=5.)
    np.testing.assert_allclose(empty.edge_seconds, c.network.lengths_m / 5.)
    window = prepare_actor_window(edges, 4)
    before = observe(c, window, prior_cost_mode='dynamic')
    after = observe(c, window, prior_cost_mode='dynamic', prior_speed_cap_mps=5.)
    assert len(window._dynamic_prior_cache) == 2
    np.testing.assert_array_equal(before.features, after.features)
    assert (after.prior_costs_seconds > before.prior_costs_seconds).all()


@pytest.mark.parametrize('cap', [0., -1., float('inf'), float('nan'), True, .1])
def test_dynamic_speed_cap_validation(cap):
    with pytest.raises(ValueError, match='speed cap'):
        DynamicTravelCosts(candidates(), np.ones((4, 2)), .5, speed_cap_mps=cap)


def test_prior_rejects_finite_float64_cost_overflowing_float32():
    c = candidates()
    c.network.lengths_m[1] = 1e40
    costs = DynamicTravelCosts(c, np.ones((4, 2)), .5)
    with pytest.raises(ValueError, match='FP32'):
        costs.candidates_seconds(np.array([0]), np.array([3]), np.array([[1, 2]]),
                                 np.array([[True, True]]))
