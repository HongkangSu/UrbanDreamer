from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision, SumoSession
from urban_dreamer.traffic.topology import VehicleDemand, load_traffic_topology

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
        incident_end_seconds=0., incident_speed_mps=.1, episode_wall_timeout_seconds=120.))
    return config, prepared, topology


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


def test_live_multiple_vehicles_events_and_terminal_padding():
    config, prepared, topology = toy_setup()
    topology = replace(topology, vehicles=(VehicleDemand('v0', 0, 0, 0.), VehicleDemand('v1', 1, 1, 0.),
                                           VehicleDemand('v2', 0, 0, 30.)))
    with SumoSession(config, prepared, topology) as session:
        state = session.reset()
        assert state['time'] == 0.
        all_decisions, departures, active = [], 0, 0.
        for _ in range(20):
            result = session.advance_window(greedy(session))
            all_decisions.extend(result.decisions)
            departures += result.departure_counts.sum()
            active += result.active_vehicle_seconds
            assert all(result.start_time <= action.decision_time < result.end_time for action in result.decisions)
            assert result.action_counts.sum() == len(result.decisions)
            if session.done:
                break
        assert session.done
        assert len(session.travel_times) == 3
        assert departures == 3
        assert sum(session.travel_times.values()) == pytest.approx(active)
        assert result.end_time % 30 == 0
        assert len(all_decisions) >= 12
        assert np.all(result.next_state['flow'][:, [0, 2, 3]] == 0)
        assert all(row['current_edge'] == row['destination_edge'] for row in result.next_state['vehicles'])
        assert not any(row['pending_decision'] for row in result.next_state['vehicles'])


@pytest.mark.parametrize("failure", ["illegal_target", "execution_error"])
def test_failed_action_invalidates_episode_and_cannot_be_counted(failure, monkeypatch):
    config, prepared, topology = toy_setup()
    topology = replace(topology, vehicles=(VehicleDemand('v', 0, 0, 0.),))
    with SumoSession(config, prepared, topology) as session:
        session.reset()
        def invalid(vehicles, flow, now):
            return [NextHopDecision('v', vehicles[0]['current_edge'], -1, now)]
        if failure == 'execution_error':
            def fail_set_route(*args):
                raise RuntimeError('connection cannot be reached from current lane')
            monkeypatch.setattr(session._connection.vehicle, 'setRoute', fail_set_route)
            invalid = greedy(session)
        with pytest.raises(ValueError):
            session.advance_window(invalid)
        assert session.metrics['events'] == 0
        with pytest.raises(EnvironmentContractError, match='invalid episode'):
            session.advance_window(greedy(session))


def test_short_start_short_successor_and_placeholder_branch_override(tmp_path):
    import xml.etree.ElementTree as ET
    config, prepared, topology = toy_setup()
    tree = ET.parse(prepared.network_xml)
    for edge in tree.getroot().findall('edge'):
        if edge.get('id') in {'A_in', 'A_hold'}:
            for lane in edge.findall('lane'):
                lane.set('length', '0.10')
    short_xml = tmp_path / 'short.net.xml'
    tree.write(short_xml)
    prepared.network_xml = short_xml
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=tuple(VehicleDemand(f'short-{i}', 0, 0, i * 2.) for i in range(3)))
    with SumoSession(config, prepared, topology) as session:
        assert session.microstep_seconds == .001
        session.reset()
        hold = session.network.edge_to_index['A_hold']
        start = session.network.edge_to_index['A_in']
        destination = session.network.edge_to_index['A_out']
        placeholder = session.candidates.temporary_route(start, hold, destination)[-1]
        alternative = next(edge for edge in session.candidates.candidates(hold, destination) if edge != placeholder)
        base = greedy(session)
        def override(vehicles, flow, now):
            actions = base(vehicles, flow, now)
            return [NextHopDecision(action.vehicle_id, action.current_edge, alternative, now)
                    if action.current_edge == hold else action for action in actions]
        actions = []
        for _ in range(12):
            result = session.advance_window(override)
            actions.extend(result.decisions)
            if session.done:
                break
        assert session.done
        for vehicle in topology.vehicles:
            own = [action for action in actions if action.vehicle_id == vehicle.vehicle_id]
            assert own[0].current_edge == start
            assert own[1].current_edge == hold
            assert own[1].next_edge == alternative
            assert [action.current_edge for action in own].count(hold) == 1
        assert len(session.travel_times) == 3
