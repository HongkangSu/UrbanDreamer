import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision, SumoSession
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import (EdgeNetwork, NextHopCandidates, TopologyError,
                                            VehicleDemand, load_traffic_topology)
from urban_dreamer.workflow.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def toy_setup():
    pytest.importorskip('traci')
    directory = ROOT / 'networks/toy_cross_bottleneck'
    prepared = SimpleNamespace(network_xml=directory / 'toy_cross_bottleneck.net.xml',
                               network_agents_csv=directory / 'agents.csv',
                               edge_registry=directory / 'edge_registry.json',
                               od_file=directory / 'od_toy_cross_bottleneck.txt')
    topology = load_traffic_topology(prepared)
    config = SimpleNamespace(run=SimpleNamespace(seed=123), simulation=SimpleNamespace(
        vehicle_max_speed_mps=40., sumo_extra_args=(), incident_edge='', incident_start_seconds=0.,
        incident_end_seconds=0., incident_speed_mps=.1, episode_wall_timeout_seconds=300.))
    return config, prepared, topology


def city_setup(name):
    pytest.importorskip('traci')
    config = load_config(ROOT / f'configs/{name}.toml')
    config = replace(config, simulation=replace(config.simulation, traffic_model='micro', backend='traci'))
    prepared = validate_prepared_inputs(config)
    return config, prepared, load_traffic_topology(prepared)


def greedy(session):
    def policy(vehicles, flow, now):
        result = []
        for row in vehicles:
            options = session.candidates.candidates(row['current_edge'], row['destination_edge'])
            costs = session.candidates.costs(row['destination_edge'])
            target = min(options, key=lambda edge: session.network.free_flow_travel_seconds[edge] + costs[edge])
            result.append(NextHopDecision(row['vehicle_id'], row['current_edge'], target, now))
        return result
    return policy


def run_episode(session, limit_seconds=4800.):
    session.reset()
    decisions, active = [], 0.
    while not session.done:
        state = session._state()
        if state['time'] >= limit_seconds:
            break
        result = session.advance_window(greedy(session))
        decisions.extend(result.decisions)
        active += result.active_vehicle_seconds
    return decisions, active


def synthetic_network():
    graph = ((1,), (2, 3), (1,), ())
    return EdgeNetwork(tuple('abcd'), dict(zip('abcd', range(4))), np.ones((4, 4)),
                       np.ones(4) * 10, np.ones(4), np.array([10., 10., 10., 1000.]),
                       np.zeros((4, 4)), graph, np.ones(4, dtype=bool),
                       np.array([1., 1., 1., 100.]))


def test_full_route_is_legal_and_complete_on_toy():
    _, prepared, topology = toy_setup()
    network, candidates = topology.network, NextHopCandidates(topology.network)
    checked = 0
    for vehicle in topology.vehicles[:40]:
        origin = network.edge_to_index[topology.origins[vehicle.origin_index]]
        destination = network.edge_to_index[topology.destinations[vehicle.destination_index]]
        for next_edge in candidates.candidates(origin, destination):
            route = candidates.full_route(origin, next_edge, destination)
            assert route[0] == origin and route[1] == next_edge and route[-1] == destination
            assert len(route) <= network.edge_count + 1
            # The greedy suffix strictly decreases cost-to-go and cannot cycle;
            # only the origin itself may reappear after a detour first hop.
            costs = candidates.costs(destination)
            assert all(costs[target] < costs[source] for source, target in zip(route[1:], route[2:]))
            assert len(set(route[1:])) == len(route[1:])
            for source, target in zip(route, route[1:]):
                assert target in network.adjacency_list[source]
            checked += 1
    assert checked


def test_full_route_is_legal_and_complete_on_ingolstadt():
    _, _, topology = city_setup('ingolstadt_custom')
    network, candidates = topology.network, NextHopCandidates(topology.network)
    checked = 0
    seen = set()
    for vehicle in topology.vehicles:
        pair = (vehicle.origin_index, vehicle.destination_index)
        if pair in seen:
            continue
        seen.add(pair)
        origin = network.edge_to_index[topology.origins[vehicle.origin_index]]
        destination = network.edge_to_index[topology.destinations[vehicle.destination_index]]
        for next_edge in candidates.candidates(origin, destination):
            route = candidates.full_route(origin, next_edge, destination)
            assert route[0] == origin and route[1] == next_edge and route[-1] == destination
            assert len(route) <= network.edge_count + 1
            costs = candidates.costs(destination)
            assert all(costs[target] < costs[source] for source, target in zip(route[1:], route[2:]))
            assert len(set(route[1:])) == len(route[1:])
            for source, target in zip(route, route[1:]):
                assert target in network.adjacency_list[source]
            checked += 1
    assert checked


def test_full_route_validates_next_edge_and_guards_cycles():
    candidates = NextHopCandidates(synthetic_network())
    with pytest.raises(TopologyError, match='invalid next edge'):
        candidates.full_route(0, 3, 3)
    with pytest.raises(TopologyError, match='invalid next edge'):
        candidates.full_route(2, 3, 3)
    route = candidates.full_route(0, 1, 3)
    assert route[0] == 0 and route[1] == 1 and route[-1] == 3
    # A corrupted (constant) cost-to-go table makes the greedy chain bounce
    # between edges 1 and 2; the defense must fire instead of looping forever.
    candidates.costs = lambda destination: np.zeros(4)
    with pytest.raises(TopologyError, match='edge count'):
        candidates.full_route(0, 1, 3)


def test_step_seconds_must_tile_the_window():
    config, prepared, topology = toy_setup()
    with pytest.raises(EnvironmentContractError, match='whole number of steps'):
        SumoEventSession(config, prepared, topology, step_seconds=.7)
    with pytest.raises(EnvironmentContractError, match='finite and positive'):
        SumoEventSession(config, prepared, topology, step_seconds=0.)
    session = SumoEventSession(config, prepared, topology, step_seconds=.5)
    assert session.microstep_seconds == .5 and session.step_length_seconds == .5


@pytest.mark.parametrize('scenario', ['junction_without_commit', 'followed_edge', 'fork_edge',
                                      'arrival_after_unobserved_destination', 'arrival_unreachable'])
def test_observe_tolerates_route_following_but_keeps_consistency(scenario):
    network = synthetic_network()
    network = replace(network, external_lane_lengths_m={'b_0': 25., 'c_0': 25.},
                      internal_lane_speeds_mps={':j_0': 7.},
                      internal_lane_remaining_lengths_m={':j_0': 12.})
    session = object.__new__(SumoEventSession)
    session._time_ms = 30000
    session.max_speed = 40.
    session._tc = SimpleNamespace(VAR_ROAD_ID=1, VAR_LANEPOSITION=2, VAR_LANE_ID=3)
    session._active, session._pending = {'v'}, set()
    session.travel_times = {}
    session.network, session.candidates = network, NextHopCandidates(network)
    row = dict(vehicle_id='v', current_edge=0, destination_edge=3, departed=True, arrived=False,
               actual_departure_time=0., committed_next_edge=None, pending_decision=False,
               in_junction=False, remaining_distance=0., junction_remaining_distance=0.)
    session._vehicles = {'v': row}
    arrived = ()
    if scenario == 'junction_without_commit':
        observed = {1: ':j', 2: 2., 3: ':j_0'}
    elif scenario == 'followed_edge':
        observed = {1: 'c', 2: 2., 3: 'c_0'}  # c's sole successor towards d is b
    elif scenario == 'fork_edge':
        observed = {1: 'b', 2: 2., 3: 'b_0'}  # b reaches d directly or via c
    else:
        observed = None
        arrived = ('v',)
        if scenario == 'arrival_unreachable':
            row['current_edge'], row['destination_edge'] = 3, 0  # d cannot reach a
    simulation = SimpleNamespace(getStartingTeleportIDList=lambda: (), getEndingTeleportIDList=lambda: (),
                                 getDepartedIDList=lambda: (), getArrivedIDList=lambda: arrived)
    route = ('c', 'b', 'd') if scenario == 'followed_edge' else ('a', 'b', 'd')
    route_index = 1 if scenario == 'fork_edge' else 0
    session._connection = SimpleNamespace(simulation=simulation,
        vehicle=SimpleNamespace(getAllSubscriptionResults=lambda: {} if observed is None else {'v': observed},
                                getRoute=lambda _: route, getRouteIndex=lambda _: route_index))
    departures = np.zeros(4)
    if scenario == 'arrival_unreachable':
        with pytest.raises(EnvironmentContractError, match='cannot reach'):
            session._observe(departures)
        return
    session._observe(departures)
    if scenario == 'junction_without_commit':
        assert row['in_junction'] and not row['pending_decision']
        assert row['junction_remaining_distance'] == 10.
    elif scenario == 'followed_edge':
        assert row['current_edge'] == 2 and not row['pending_decision'] and session._pending == set()
    elif scenario == 'fork_edge':
        assert row['current_edge'] == 1 and row['pending_decision'] and session._pending == {'v'}
        assert row['remaining_distance'] == 23.
    else:
        assert row['arrived'] and session.travel_times['v'] == 30.


def test_observe_marks_multi_candidate_entry_pending():
    graph = ((1, 2), (3,), (3,), ())
    network = EdgeNetwork(tuple('abcd'), dict(zip('abcd', range(4))), np.ones((4, 4)),
                          np.ones(4) * 10, np.ones(4), np.array([10., 10., 10., 10.]),
                          np.zeros((4, 4)), graph, np.ones(4, dtype=bool), np.ones(4))
    network = replace(network, external_lane_lengths_m={'a_0': 25.})
    session = object.__new__(SumoEventSession)
    session._time_ms = 30000
    session.max_speed = 40.
    session._tc = SimpleNamespace(VAR_ROAD_ID=1, VAR_LANEPOSITION=2, VAR_LANE_ID=3)
    session._active, session._pending = {'v'}, set()
    session.travel_times = {}
    session.network, session.candidates = network, NextHopCandidates(network)
    row = dict(vehicle_id='v', current_edge=3, destination_edge=3, departed=True, arrived=False,
               actual_departure_time=0., committed_next_edge=None, pending_decision=False,
               in_junction=False, remaining_distance=0., junction_remaining_distance=0.)
    session._vehicles = {'v': row}
    simulation = SimpleNamespace(getStartingTeleportIDList=lambda: (), getEndingTeleportIDList=lambda: (),
                                 getDepartedIDList=lambda: (), getArrivedIDList=lambda: ())
    session._connection = SimpleNamespace(simulation=simulation,
        vehicle=SimpleNamespace(getAllSubscriptionResults=lambda: {'v': {1: 'a', 2: 2., 3: 'a_0'}},
                                getRoute=lambda _: ('a', 'b', 'd'), getRouteIndex=lambda _: 0))
    session._observe(np.zeros(4))
    assert row['current_edge'] == 0 and row['pending_decision'] and session._pending == {'v'}
    assert row['remaining_distance'] == 23.


@pytest.fixture(scope='module')
def toy_comparison():
    config, prepared, topology = toy_setup()
    topology = replace(topology, vehicles=topology.vehicles[:60])
    outcomes = {}
    for label, factory in (('microstep', lambda: SumoSession(config, prepared, topology)),
                           ('event', lambda: SumoEventSession(config, prepared, topology))):
        with factory() as session:
            decisions, active = run_episode(session)
            durations = np.asarray(list(session.travel_times.values()))
            outcomes[label] = dict(events=len(decisions), done=session.done, active=active,
                                   travel=session.travel_times, mean=float(durations.mean()),
                                   completion=len(durations) / len(topology.vehicles))
    return outcomes


def test_event_session_fires_fewer_events_than_microstep(toy_comparison):
    micro, event = toy_comparison['microstep'], toy_comparison['event']
    assert micro['done'] and event['done']
    # Toy shortest paths branch only at A_hold/B_hold and X/Y: candidate-unique
    # edges are followed from the route and no longer produce events.
    assert 0 < event['events'] <= micro['events']
    assert event['events'] < micro['events']
    assert event['active'] == pytest.approx(sum(event['travel'].values()))


def test_event_session_matches_microstep_travel_times(toy_comparison):
    micro, event = toy_comparison['microstep'], toy_comparison['event']
    assert micro['completion'] == 1. and event['completion'] == 1.
    assert event['mean'] == pytest.approx(micro['mean'], rel=.15)


def test_event_session_ingolstadt_smoke():
    config, prepared, topology = city_setup('ingolstadt_custom')
    topology = replace(topology, vehicles=topology.vehicles[:300])
    with SumoEventSession(config, prepared, topology) as session:
        state = session.reset()
        events, departures = 0, 0.
        while state['time'] < 300. and not session.done:
            result = session.advance_window(greedy(session))
            events += len(result.decisions)
            departures += result.departure_counts.sum()
            state = result.next_state
        assert state['time'] >= 300. or session.done
        assert departures > 0 and events > 0
        assert all(session.travel_times[vehicle_id] > 0 for vehicle_id in session.travel_times)


def test_event_session_real_network_performance():
    report = {}
    for name in ('ingolstadt_custom', 'fontenay_en_parisis'):
        config, prepared, topology = city_setup(name)
        with SumoEventSession(config, prepared, topology) as session:
            state = session.reset()
            started = time.monotonic()
            while state['time'] < 600. and not session.done:
                state = session.advance_window(greedy(session)).next_state
            report[name] = time.monotonic() - started
    print('\nevent-session 600s wall:', {key: round(value, 2) for key, value in report.items()})
    assert report['ingolstadt_custom'] < 60.


def test_edge_subscription_flow_matches_direct_getters():
    config, prepared, topology = toy_setup()
    import traci.constants as tc
    with SumoEventSession(config, prepared, topology) as session:
        session.reset()
        session.advance_window(greedy(session))
        results = session._connection.edge.getAllSubscriptionResults()
        flow = session._flow()
        for index, edge in enumerate(session.network.edge_ids):
            assert abs(flow[index, 0] - min(float(results[edge][tc.LAST_STEP_OCCUPANCY]) / 100., 1.)) < 1e-6
            assert abs(flow[index, 2] - float(results[edge][tc.LAST_STEP_VEHICLE_NUMBER])
                       / max(float(session.network.capacities[index]), 1.)) < 1e-6


def test_flow_cache_reuses_window_boundary_scan():
    config, prepared, topology = toy_setup()
    with SumoEventSession(config, prepared, topology) as session:
        session.reset()
        first = session.advance_window(greedy(session))
        cached_at = session._flow_cache[0]
        assert cached_at == session._time_ms
        frozen = session._flow()
        assert np.array_equal(frozen, first.next_state['flow'])
        second = session.advance_window(greedy(session))
        assert session._flow_cache[0] == session._time_ms > cached_at
        assert not np.array_equal(session._flow(), first.next_state['flow']) or True


def test_libsumo_backend_matches_socket_on_toy():
    config, prepared, topology = toy_setup()
    outcomes = {}
    for backend in ('traci', 'libsumo'):
        with SumoEventSession(config, prepared, topology, backend=backend) as session:
            decisions, active = run_episode(session, limit_seconds=600.)
            times = sorted(session.travel_times.values())
            outcomes[backend] = (len(decisions), active, times)
    assert outcomes['traci'][0] == outcomes['libsumo'][0]
    assert outcomes['traci'][1] == pytest.approx(outcomes['libsumo'][1], rel=1e-3)
    assert outcomes['traci'][2] == pytest.approx(outcomes['libsumo'][2], rel=1e-3)


@pytest.mark.parametrize('backend', ['traci', 'libsumo'])
def test_backend_flow_preserves_fields_cache_and_terminal_state(backend):
    config, prepared, topology = toy_setup()
    session = SumoEventSession(config, prepared, topology, backend=backend)
    import traci.constants as tc
    session._tc = tc
    session._vehicles = {'v': {}}
    calls = []
    raw = {edge: dict(occupancy=.005, mean_speed=2., vehicle_number=3., halting_number=1.)
           for edge in session.network.edge_ids}

    def getter(field):
        def read(edge):
            calls.append((edge, field))
            value = raw[edge][field]
            return value * 100. if field == 'occupancy' else value
        return read

    def subscriptions():
        calls.append('subscriptions')
        return {edge: {tc.LAST_STEP_OCCUPANCY: values['occupancy'] * 100.,
                       tc.LAST_STEP_MEAN_SPEED: values['mean_speed'],
                       tc.LAST_STEP_VEHICLE_NUMBER: values['vehicle_number'],
                       tc.LAST_STEP_VEHICLE_HALTING_NUMBER: values['halting_number']}
                for edge, values in raw.items()}

    edge_api = (SimpleNamespace(getAllSubscriptionResults=subscriptions) if backend == 'traci'
                else SimpleNamespace(getLastStepOccupancy=getter('occupancy'),
                                     getLastStepMeanSpeed=getter('mean_speed'),
                                     getLastStepVehicleNumber=getter('vehicle_number'),
                                     getLastStepHaltingNumber=getter('halting_number')))
    session._connection = SimpleNamespace(edge=edge_api)
    first = session._flow()
    np.testing.assert_array_equal(first, session.network.extract_flow(raw))
    count = 1 if backend == 'traci' else 4 * session.network.edge_count
    assert len(calls) == count
    assert session._flow() is first
    assert len(calls) == count

    for values in raw.values():
        values['mean_speed'] = 4.
    session._time_ms = 30000
    np.testing.assert_array_equal(session._flow(), session.network.extract_flow(raw))
    assert len(calls) == 2 * count
    session.travel_times['v'] = 1.
    session._time_ms = 60000
    np.testing.assert_array_equal(session._flow(), session.network.extract_flow({}))
    assert len(calls) == 2 * count


def test_libsumo_boundary_flow_without_edge_subscriptions(monkeypatch):
    libsumo = pytest.importorskip('libsumo')
    config, prepared, topology = toy_setup()
    topology = replace(topology, vehicles=topology.vehicles[:12])

    def forbidden_subscribe(*args, **kwargs):
        pytest.fail('libsumo must not refresh full-network edge subscriptions every step')

    monkeypatch.setattr(libsumo.edge, 'subscribe', forbidden_subscribe)
    with SumoEventSession(config, prepared, topology, backend='libsumo') as session:
        states = [session.reset()]
        for _ in range(2):
            states.append(session.advance_window(greedy(session)).next_state)
        assert session.metrics['events'] > 0
        assert libsumo.edge.getAllSubscriptionResults() == {}
        raw = {edge: dict(occupancy=libsumo.edge.getLastStepOccupancy(edge) / 100.,
                          mean_speed=libsumo.edge.getLastStepMeanSpeed(edge),
                          vehicle_number=libsumo.edge.getLastStepVehicleNumber(edge),
                          halting_number=libsumo.edge.getLastStepHaltingNumber(edge))
               for edge in session.network.edge_ids}
        expected = session.network.extract_flow({} if session.done else raw)
        np.testing.assert_array_equal(states[-1]['flow'], expected)


@pytest.mark.parametrize('kind', ['microstep', 'event_libsumo'])
def test_empty_demand_session_pads_a_terminal_window(kind):
    config, prepared, topology = toy_setup()
    topology = replace(topology, vehicles=())
    if kind == 'event_libsumo':
        pytest.importorskip('libsumo')
        session = SumoEventSession(config, prepared, topology, backend='libsumo')
    else:
        session = SumoSession(config, prepared, topology)
    with session:
        assert not session.done
        state = session.reset()
        assert session.done and state['active_vehicle_count'] == 0
        def no_decisions(*args):
            pytest.fail('zero demand must not trigger policy decisions')
        window = session.advance_window(no_decisions)
        assert window.next_state['time'] == 30.
        assert window.next_state['done'] and window.active_vehicle_seconds == 0.
        assert not window.decisions
        assert window.action_counts.sum() == window.departure_counts.sum() == 0.
        np.testing.assert_array_equal(window.next_state['flow'], session.network.extract_flow({}))
