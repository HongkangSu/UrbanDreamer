"""Meso windows keep the event clock while ignoring microscopic internal links."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_event_imagination_protocol import network, policy, seed
from test_parallel_imagination import trainer_case
from urban_dreamer.learning.imagination import advance_imagined_window
from urban_dreamer.learning.imagination import clip_road_speed_fractions
from urban_dreamer.learning.parallel_imagination import (collect_parallel_imagination,
    close_parallel_imagination, _collect_window, _initialize_worker, _numpy_state)
from urban_dreamer.traffic.topology import NextHopCandidates


def advance(rows, topology=None, *, start=0., seconds=30., model='meso', **kwargs):
    topology = topology or network()
    return advance_imagined_window(rows,
        candidates=NextHopCandidates(topology, traffic_model=model),
        edge_speeds_mps=np.full(topology.edge_count, 10.),
        frozen_flow=np.zeros((topology.edge_count, 8)), start_time=start,
        interval_seconds=seconds, total_vehicle_count=max(1, len(rows)),
        decide=policy, session_protocol='event', traffic_model=model, **kwargs)


def test_meso_skips_internal_links_without_changing_shared_topology():
    topology = network(junction_lengths={(0, 1): 400.})
    meso, micro = [seed()], [seed()]
    imagined = advance(meso, topology, seconds=3.)
    original = advance(micro, topology, seconds=3., model='micro')
    assert [(item.current_edge, item.next_edge, item.decision_time)
            for item in imagined.decisions] == [(1, 3, 1.)]
    assert not original.decisions and micro[0]['in_junction']
    assert meso[0]['current_edge'] == 3 and not meso[0]['in_junction']
    assert meso[0]['junction_remaining_distance'] == 0.
    assert meso[0]['junction_speed'] == 0.
    assert meso[0]['junction_connection'] is None
    assert topology.connection_lengths_m == {(0, 1): 400.}
    assert NextHopCandidates(topology).junction(0, 1)[0] == 400.


def test_meso_boundary_defers_policy_and_keeps_precommitted_successor():
    rows = [seed(remaining_distance=300.)]
    first = advance(rows)
    assert not first.action_events
    assert rows[0]['current_edge'] == 1 and rows[0]['pending_decision']
    assert rows[0]['committed_next_edge'] == 3
    second = advance(rows, start=30., seconds=1., event_index_start=8)
    assert [(event.decision_time, event.event_index, event.current_edge, event.next_edge)
            for event in second.action_events] == [(30., 8, 1, 3)]


def test_meso_short_hidden_fork_has_no_observed_action():
    topology = network((1., 1., 1., 1000.), junction_lengths={(0, 1): 400.})
    rows = [seed(remaining_distance=1.)]
    result = advance(rows, topology, seconds=1.)
    assert not result.decisions and not result.action_events
    assert rows[0]['current_edge'] == 3
    assert rows[0]['remaining_distance'] == pytest.approx(992.)


def test_meso_keeps_route_commitment_when_window_speed_changes():
    rows = [seed(current_edge=1, remaining_distance=25., committed_next_edge=2)]
    first = advance(rows, seconds=1.)
    assert not first.decisions
    assert rows[0]['committed_next_edge'] == 2
    topology = network()
    result = advance_imagined_window(rows,
        candidates=NextHopCandidates(topology, traffic_model='meso'),
        edge_speeds_mps=np.full(4, 5.), frozen_flow=np.zeros((4, 8)),
        start_time=1., interval_seconds=2., total_vehicle_count=1,
        decide=policy, session_protocol='event', traffic_model='meso')
    assert not result.decisions
    assert rows[0]['current_edge'] == 1 and rows[0]['remaining_distance'] == 5.
    assert rows[0]['committed_next_edge'] == 2


def test_meso_suffix_uses_same_zero_junction_cost_as_candidate_policy_features():
    topology = network(graph=((1,), (2, 3), (3,), ()),
                       junction_lengths={(1, 3): 5000.})
    rows = [seed()]
    result = advance(rows, topology, seconds=1.)
    assert not result.decisions and rows[0]['pending_decision']
    assert rows[0]['committed_next_edge'] == 3
    micro = [seed()]
    advance(micro, topology, seconds=1., model='micro')
    assert micro[0]['committed_next_edge'] == 2


@pytest.mark.parametrize('model', ['micro', 'meso'])
def test_initial_route_still_omits_first_junction_cost(model):
    topology = network(graph=((1, 2), (3,), (3,), ()),
                       junction_lengths={(0, 1): 1000.})
    rows = [seed(departed=False, departure_time=1., committed_next_edge=None)]
    result = advance(rows, topology, seconds=1., model=model)
    assert not result.decisions
    assert rows[0]['committed_next_edge'] == 1


def test_meso_arrival_and_exposure_remain_observation_aligned():
    topology = network((1., 1., 1., 1.), junction_lengths={(0, 1): 1000.})
    rows = [seed(departure_time=.25, departed=False,
                 remaining_distance=1., committed_next_edge=None)]
    result = advance(rows, topology, seconds=2., stop_when_empty=True)
    assert result.arrivals == {'v': 1.}
    assert result.active_vehicle_seconds == .75
    assert result.end_time == 1.


def test_meso_records_every_hidden_external_entry_for_no_revisit_routes():
    topology = network((1., 1., 1., 1.), graph=((1,), (2,), (3,), ()),
                       junction_lengths={(0, 1): 1000., (1, 2): 1000.})
    rows = [seed(remaining_distance=1., visited_edges=(0,))]
    result = advance_imagined_window(rows,
        candidates=NextHopCandidates(topology, forbid_revisit=True, traffic_model='meso'),
        edge_speeds_mps=np.full(4, 10.), frozen_flow=np.zeros((4, 8)),
        start_time=0., interval_seconds=1., total_vehicle_count=1,
        decide=policy, session_protocol='event', traffic_model='meso')
    assert not result.decisions
    assert result.arrivals == {'v': 1.}
    assert rows[0]['visited_edges'] == (0, 1, 2, 3)


@pytest.mark.parametrize('updates', [dict(in_junction=True),
                                    dict(departed=False, in_junction=True)])
def test_meso_rejects_micro_internal_lane_seeds_before_mutation(updates):
    rows = [seed(**updates)]
    original = deepcopy(rows)
    with pytest.raises(ValueError, match='microscopic in_junction'):
        advance(rows, seconds=1.)
    assert rows == original


def test_meso_requires_event_protocol_and_matching_candidate_physics():
    kwargs = dict(vehicle_states=[seed()], edge_speeds_mps=np.full(4, 10.),
        frozen_flow=np.zeros((4, 8)), start_time=0., total_vehicle_count=1,
        interval_seconds=1., decide=policy, traffic_model='meso')
    with pytest.raises(ValueError, match='event session protocol'):
        advance_imagined_window(candidates=NextHopCandidates(network()), **kwargs)
    with pytest.raises(ValueError, match='must match'):
        advance_imagined_window(candidates=NextHopCandidates(network()),
            session_protocol='event', **kwargs)


def test_meso_serial_and_parallel_keep_event_metadata_and_objective(monkeypatch):
    trainer = trainer_case(deadline=5., rollouts=1)
    trainer.candidates = NextHopCandidates(trainer.network, traffic_model='meso')
    trainer.config.simulation.traffic_model = 'meso'
    batch = trainer.online_replay.sample_active_seeds(batch_size=1,
        rng=np.random.default_rng(124), initial_only=True)
    torch.manual_seed(434)
    actor_seed = torch.randint(0, 2**63 - 1, (1, 2))[0, 0].item()
    torch.manual_seed(actor_seed)
    expected, expected_macro = trainer._imagine_from_seed(batch, 0,
        rollout_index=0, batched_events=True)
    try:
        torch.manual_seed(434)
        actual, actual_macro = collect_parallel_imagination(trainer, 1)
        metadata = ('vehicle_id', 'event_index', 'decision_time', 'window')
        assert [[row[key] for key in metadata] for row in actual] == [
            [row[key] for key in metadata] for row in expected]
        for key in ('global_return', 'global_advantage', 'counterfactual_baseline',
                    'executed_q', 'log_prob', 'entropy', 'timeout_return'):
            torch.testing.assert_close(torch.cat([row[key] for row in actual]),
                torch.cat([row[key] for row in expected]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_macro[0], expected_macro, rtol=1e-5, atol=1e-6)
    finally:
        close_parallel_imagination(trainer)


def test_road_speed_fraction_clipping_preserves_physics_specific_floors_and_upper_limit():
    fractions = np.asarray([-.1, 0., .25, 2.])
    np.testing.assert_array_equal(clip_road_speed_fractions(fractions, 'meso'), [0., 0., .25, 1.5])
    np.testing.assert_array_equal(clip_road_speed_fractions(fractions), [.05, .05, .25, 1.5])
    with pytest.raises(ValueError, match='traffic_model'):
        clip_road_speed_fractions(fractions, 'other')


@pytest.mark.parametrize('remaining', [10., 0.])
def test_stopped_meso_road_completes_window_with_full_active_exposure(remaining):
    topology = network()
    rows = [seed(remaining_distance=remaining)]
    result = advance_imagined_window(rows,
        candidates=NextHopCandidates(topology, traffic_model='meso'),
        edge_speeds_mps=np.zeros(4), frozen_flow=np.zeros((4, 8)),
        start_time=0., interval_seconds=30., total_vehicle_count=1,
        decide=policy, session_protocol='event', traffic_model='meso', stop_when_empty=True)
    assert result.end_time == 30.
    assert result.active_vehicle_seconds == 30.
    assert not result.arrivals
    assert rows[0]['current_edge'] == (0 if remaining else 1)
    assert rows[0]['remaining_distance'] == 10.
    assert sum((segment.end_time - segment.start_time) * segment.active_count
               for segment in result.cost_segments) == 30.


@pytest.mark.parametrize('model,expected_speed', [('micro', .5), ('meso', 0.)])
def test_serial_and_worker_pass_actual_stopped_speeds_to_motion(monkeypatch, model, expected_speed):
    import urban_dreamer.learning.training as serial
    import urban_dreamer.learning.parallel_imagination as parallel
    trainer = trainer_case(deadline=5., rollouts=1)
    trainer.candidates = NextHopCandidates(trainer.network, traffic_model=model)
    trainer.config.simulation.traffic_model = model
    trainer.world_model.decode = lambda latent: latent.new_zeros(*latent.shape[:2], 4)
    batch = trainer.online_replay.sample_active_seeds(batch_size=1,
        rng=np.random.default_rng(124), initial_only=True)
    captures = []

    def capture(states, **kwargs):
        before = deepcopy(states)
        result = advance_imagined_window(states, **kwargs)
        captures.append((kwargs['edge_speeds_mps'].copy(), before, deepcopy(states), result))
        return result

    monkeypatch.setattr(serial, 'advance_imagined_window', capture)
    trainer._imagine_from_seed(batch, 0, rollout_index=0, batched_events=True)
    encoder = trainer.world_model.dynamics.event_encoder
    _initialize_worker(trainer.candidates, trainer.network.adjacency.copy(),
                       trainer.network.static_features.copy())
    monkeypatch.setattr(parallel, 'advance_imagined_window', capture)
    _collect_window(dict(version=1,
        weights=dict(actor=_numpy_state(trainer.actor), critic=_numpy_state(trainer.event_critic),
            encoder=_numpy_state(encoder), prior=(0., 30.)),
        states=deepcopy(batch.vehicles[0]),
        rng=torch.Generator().manual_seed(2).get_state().numpy(),
        decoded=np.zeros((4, 4), dtype=np.float32), latent=np.zeros((4, 64), dtype=np.float32),
        population=4, total_vehicle_count=4, cumulative_vehicle_seconds=0., deadline=5., start=0., max_speed=10., event_count=0, traffic_model=model))
    assert len(captures) == 2
    for speeds, before, after, result in captures:
        np.testing.assert_allclose(speeds, np.full(4, expected_speed), rtol=1e-7, atol=1e-8)
        assert result.end_time == 5. and result.active_vehicle_seconds == 10.
        for old, new in zip(before, after):
            assert new['current_edge'] == old['current_edge']
            assert new['remaining_distance'] == pytest.approx(old['remaining_distance'] - expected_speed * 5.)
