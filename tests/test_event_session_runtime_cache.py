"""Static route work is reused without changing live vehicle observations."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_event_session_commitments import session_at
from test_meso_event_session import meso_setup
from test_urban_dreamer_event_session import greedy
from urban_dreamer.learning.sumo_sampling import SumoEpisodeSampler
from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.topology import NextHopCandidates


def test_cached_actual_route_still_reads_live_index_and_validates_it():
    session = session_at('b', ('a', 'b', 'd'), 1)
    session._actual_routes = {'v': ('a', 'b', 'd')}
    indices = iter((1, 2, -1))
    session._connection.vehicle.getRouteIndex = lambda _: next(indices)
    session._connection.vehicle.getRoute = lambda _: pytest.fail('route copied again')
    assert session._route_commitment('v') == (1, 3)
    assert session._route_commitment('v') == (3, None)
    with pytest.raises(EnvironmentContractError, match='invalid route index'):
        session._route_commitment('v')


def test_meso_reuses_one_live_index_within_the_same_vehicle_observation():
    session = session_at('b', ('a', 'b', 'd'), 1)
    session.traffic_model = 'meso'
    session._actual_routes = {'v': ('a', 'b', 'd')}
    reads = []
    session._connection.vehicle.getRouteIndex = lambda vid: reads.append(vid) or 1
    session._observe(np.zeros(4))
    assert reads == ['v']
    assert session._vehicles['v']['current_edge'] == 1
    assert session._vehicles['v']['committed_next_edge'] == 3


@pytest.mark.parametrize('prefix', [False, True])
def test_setroute_caches_actual_accepted_prefix(prefix):
    session = session_at('b', ('a', 'b', 'd'), 1, current=1, pending=True)
    session._actual_routes = {'v': ('a', 'b', 'd')}
    actual, index, reads = [], int(prefix), []

    def assign(_, route):
        actual[:] = ['a', *route] if prefix else route

    def read_route(vid):
        reads.append(vid)
        return actual

    session._connection.vehicle.setRoute = assign
    session._connection.vehicle.getRoute = read_route
    session._connection.vehicle.getRouteIndex = lambda _: index
    session._apply_decisions(lambda rows, flow, now: [NextHopDecision('v', 1, 2, now)],
                             np.zeros((4, 4)))
    assert session._actual_routes['v'] == tuple(actual)
    assert session._route_commitment('v') == (1, 2)
    assert session._route_commitment('v') == (1, 2)
    assert reads == ['v']


def test_frozen_flow_does_not_copy_discarded_vehicle_state():
    session = session_at('b', ('a', 'b', 'd'), 1)
    session._closed = session._invalid = False
    session._started_at = 0.
    session.config = SimpleNamespace(simulation=SimpleNamespace(episode_wall_timeout_seconds=float('inf')))
    session._step_ms, session.microstep_seconds = 1000, 1.
    session.metrics.update(microsteps=0, active_vehicle_seconds=0.)
    session.travel_times = {'v': 30.}
    flow = np.zeros((4, 8))
    session._flow = lambda: flow
    copies = []
    session._state = lambda: copies.append(1) or {'flow': flow, 'time': session._time_ms / 1000.}
    session.advance_window(lambda *args: pytest.fail('empty episode asks for action'))
    assert len(copies) == 1


def test_sampler_reuses_static_candidates_and_rejects_different_topology():
    config, prepared, topology = meso_setup(0)
    candidates = NextHopCandidates(topology.network, traffic_model='meso')
    session = SumoEventSession(config, prepared, topology, backend='libsumo')
    SumoEpisodeSampler(session=session, candidates=candidates,
                       deadline_seconds=30., total_vehicle_count=0)
    assert session.candidates is candidates
    other = NextHopCandidates(replace(topology.network), traffic_model='meso')
    with pytest.raises(ValueError, match='same network'):
        SumoEpisodeSampler(session=session, candidates=other,
                           deadline_seconds=30., total_vehicle_count=0)


def test_sampler_rejects_different_revisit_rules():
    config, prepared, topology = meso_setup(0)
    session = SumoEventSession(config, prepared, topology, backend='libsumo')
    candidates = NextHopCandidates(topology.network, traffic_model='meso', forbid_revisit=True)
    with pytest.raises(ValueError, match='forbid_revisit'):
        SumoEpisodeSampler(session=session, candidates=candidates,
                           deadline_seconds=30., total_vehicle_count=0)


@pytest.mark.parametrize('traffic_model,backend,enabled', [
    ('meso', 'libsumo', True), ('meso', 'traci', False),
    ('micro', 'libsumo', False), ('micro', 'traci', False),
])
def test_direct_observation_only_for_inprocess_meso(traffic_model, backend, enabled):
    config, prepared, topology = meso_setup(0)
    config.simulation.traffic_model = traffic_model
    session = SumoEventSession(config, prepared, topology, backend=backend)
    assert session._direct_vehicle_observation is enabled


def test_direct_observation_missing_active_vehicle_keeps_contract_error():
    session = session_at('b', ('a', 'b', 'd'), 1)
    session.traffic_model = 'meso'
    session._direct_vehicle_observation = True
    session._connection.vehicle.getAllSubscriptionResults = lambda: pytest.fail('subscriptions read')

    def missing(_):
        raise RuntimeError('vehicle is gone')

    session._connection.vehicle.getRoadID = missing
    with pytest.raises(EnvironmentContractError, match='disappeared without arrival'):
        session._observe(np.zeros(4))


def test_real_meso_route_reads_only_at_departure_and_policy_replacement(monkeypatch):
    config, prepared, topology = meso_setup(8)
    candidates = NextHopCandidates(topology.network, traffic_model='meso')
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        with SumoEpisodeSampler(session=session, candidates=candidates,
                               deadline_seconds=3810., total_vehicle_count=8) as sampler:
            api = session._connection.vehicle
            actual_get_route = api.getRoute
            reads = []

            def get_route(vid):
                reads.append(vid)
                return actual_get_route(vid)

            monkeypatch.setattr(api, 'getRoute', get_route)

            def decide(rows, obs, frozen, now, index, context):
                return greedy(session)(rows, frozen, now)

            for window in sampler.windows(decide):
                for vid in session._active:
                    actual = tuple(actual_get_route(vid))
                    index = api.getRouteIndex(vid)
                    assert session._actual_routes[vid] == actual
                    expected = (session.network.edge_to_index[actual[index]],
                                session.network.edge_to_index[actual[index + 1]]
                                if index + 1 < len(actual) else None)
                    assert session._route_commitment(vid) == expected
            assert session.done and len(session.travel_times) == 8
            assert len(reads) == len(session._actual_routes) + session.metrics['events']
            assert session.metrics['microsteps'] > len(reads)


@pytest.mark.parametrize('forbid_revisit', [False, True])
def test_real_meso_subscribed_index_matches_live_after_rerouting(monkeypatch, forbid_revisit):
    import traci.constants as tc
    config, prepared, topology = meso_setup(8)
    with SumoEventSession(config, prepared, topology, backend='libsumo',
                          forbid_revisit=forbid_revisit) as session:
        session._direct_vehicle_observation = False
        session.reset()
        api = session._connection.vehicle
        live_index = api.getRouteIndex
        checked = 0
        for _ in range(500):
            monkeypatch.setattr(api, 'getRouteIndex', live_index)
            session._apply_decisions(greedy(session), session._flow())
            session._connection.simulationStep()
            session._time_ms += 1000
            # Departures are subscribed by _observe, so verify both existing
            # and newly subscribed vehicles after that call below.
            monkeypatch.setattr(api, 'getRouteIndex',
                lambda _: pytest.fail('meso observation queried a scalar route index'))
            session._observe(np.zeros(session.network.edge_count))
            subscriptions = api.getAllSubscriptionResults()
            for vid in session._active:
                assert subscriptions[vid][tc.VAR_ROUTE_INDEX] == live_index(vid)
                assert tc.VAR_LANEPOSITION in subscriptions[vid]
                assert tc.VAR_LANE_ID not in subscriptions[vid]
                checked += 1
            if session.done:
                break
        assert session.done and checked > 0 and session.metrics['events'] > 0


@pytest.mark.parametrize('forbid_revisit', [False, True])
def test_real_meso_direct_and_subscribed_observations_match_every_step(monkeypatch, forbid_revisit):
    config, prepared, topology = meso_setup(8)
    outcomes = []
    for direct in (False, True):
        with SumoEventSession(config, prepared, topology, backend='libsumo',
                              forbid_revisit=forbid_revisit) as session:
            assert session._direct_vehicle_observation is True
            session._direct_vehicle_observation = direct
            session.reset()
            observations, windows = [], []
            api, original_observe = session._connection.vehicle, session._observe
            scalar_index, original_index = [], api.getRouteIndex

            def index(vid):
                scalar_index.append(vid)
                return original_index(vid)

            def observe(departures):
                scalar_index.clear()
                original_observe(departures)
                if direct:
                    assert scalar_index == sorted(session._active)
                else:
                    assert not scalar_index
                observations.append((session._time_ms,
                    [dict(session._vehicles[vid]) for vid in sorted(session._vehicles)],
                    sorted(session._active), sorted(session._pending)))

            with monkeypatch.context() as patch:
                patch.setattr(api, 'getRouteIndex', index)
                if direct:
                    patch.setattr(api, 'subscribe', lambda *args: pytest.fail('direct path subscribed'))
                    patch.setattr(api, 'getAllSubscriptionResults',
                                  lambda: pytest.fail('direct path collected subscriptions'))
                patch.setattr(session, '_observe', observe)
                for _ in range(128):
                    if session.done:
                        break
                    result = session.advance_window(greedy(session))
                    windows.append((result.decisions, result.active_vehicle_seconds,
                        result.action_counts.tolist(), result.departure_counts.tolist(),
                        result.next_state['flow'].tolist(), result.next_state['vehicles']))
            assert session.done and session.metrics['events'] > 0
            outcomes.append((observations, windows, dict(session.travel_times),
                             session.metrics['active_vehicle_seconds']))
    assert outcomes[0] == outcomes[1]
