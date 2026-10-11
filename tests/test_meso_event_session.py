"""Real mesoscopic SUMO behavior, beyond checking startup arguments."""
from dataclasses import replace
from types import SimpleNamespace
from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from test_urban_dreamer_event_session import greedy, toy_setup
from test_event_session_commitments import session_at
from urban_dreamer.traffic.environment import EnvironmentContractError, NextHopDecision
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.simulation_settings import physics_identity, sumo_model_args
from urban_dreamer.traffic.topology import VehicleDemand
from urban_dreamer.traffic.topology import load_traffic_topology


def meso_setup(population=8):
    pytest.importorskip("libsumo")
    config, prepared, topology = toy_setup()
    config.simulation.traffic_model = "meso"
    return config, prepared, replace(topology, vehicles=topology.vehicles[:population])


def test_canonical_physics_defaults_and_reserved_raw_flags():
    assert physics_identity({}) == {"traffic_model": "micro"}
    assert sumo_model_args(SimpleNamespace(sumo_extra_args=())) == []
    identity = physics_identity({"traffic_model": "meso"})
    assert identity["meso_junction_control"] is True
    assert identity["meso_lane_queue"] is identity["meso_multi_queue"] is False
    assert identity["vehicle_position"] == "interpolated_external_edge"
    assert identity["junction_state"] == "zero_internal_distance"
    args = sumo_model_args({"traffic_model": "meso"})
    assert args[args.index("--meso-interpolate-pos") + 1] == "true"
    for model in ("micro", "meso"):
        for flag in ("--mesosim", "--mesosim=true", "--meso-tls-penalty=1", "--no-internal-links"):
            with pytest.raises(EnvironmentContractError, match="physical simulation contract"):
                sumo_model_args({"traffic_model": model, "sumo_extra_args": (flag,)})


@pytest.mark.parametrize("overrides", [
    {"traffic_model": "wrong"}, {"meso_interpolate_positions": False},
    {"meso_junction_control": False}, {"meso_lane_queue": True}, {"meso_multi_queue": True},
    {"meso_edge_length_m": float("nan")}, {"meso_edge_length_m": 0},
])
def test_invalid_physical_profiles_rejected(overrides):
    with pytest.raises(EnvironmentContractError):
        physics_identity({"traffic_model": "meso", **overrides})


def test_meso_internal_lane_is_invalid_instead_of_becoming_micro_state():
    session = session_at(":j", ("a", "b", "d"), 0)
    session.traffic_model = "meso"
    with pytest.raises(EnvironmentContractError, match="micro internal junction"):
        session._observe(np.zeros(4))


def test_meso_direct_session_resolves_backend_and_rejects_a_different_clock():
    config, prepared, topology = meso_setup()
    config.simulation.backend = 'libsumo'
    session = SumoEventSession(config, prepared, topology)
    assert session.backend == 'libsumo'
    with pytest.raises(EnvironmentContractError, match='one-second'):
        SumoEventSession(config, prepared, topology, step_seconds=.5)


def test_real_meso_interpolated_positions_routes_and_no_revisit_history():
    config, prepared, topology = meso_setup()
    within_segment_progress = False
    observed_kinds = set()
    previous = {}
    with SumoEventSession(config, prepared, topology, backend="libsumo", forbid_revisit=True) as session:
        session.reset()
        api = session._connection.vehicle
        for _ in range(500):
            pending = [dict(session._vehicles[vid]) for vid in sorted(session._pending)]
            for row in pending:
                assert len(session._legal_candidates(row["current_edge"], row["destination_edge"], row["visited_edges"])) > 1
                observed_kinds.add("fork")
            actions = session._apply_decisions(greedy(session), session._flow())
            for action in actions:
                assert api.getRoute(action.vehicle_id)[api.getRouteIndex(action.vehicle_id) + 1] == session.network.edge_ids[action.next_edge]
            session._connection.simulationStep()
            session._time_ms += 1000
            session._observe(np.zeros(session.network.edge_count))
            for vid in session._active:
                row = session._vehicles[vid]
                road, index = api.getRoadID(vid), api.getRouteIndex(vid)
                position, segment = api.getLanePosition(vid), api.getSegmentIndex(vid)
                assert not road.startswith(":")
                assert row["current_edge"] == session.network.edge_to_index[api.getRoute(vid)[index]]
                assert row["remaining_distance"] == pytest.approx(max(0., session._connection.lane.getLength(road + "_0") - position))
                assert not row["in_junction"]
                assert row["junction_remaining_distance"] == row["junction_speed"] == 0
                assert row["junction_connection"] is None
                assert len(row["visited_edges"]) == len(set(row["visited_edges"]))
                assert row["visited_edges"][-1] == row["current_edge"]
                if row["current_edge"] != row["destination_edge"]:
                    assert row["committed_next_edge"] == session.network.edge_to_index[api.getRoute(vid)[index + 1]]
                    if len(session._legal_candidates(row["current_edge"], row["destination_edge"], row["visited_edges"])) == 1:
                        assert not row["pending_decision"]
                        observed_kinds.add("singleton")
                else:
                    assert row["committed_next_edge"] is None
                    observed_kinds.add("destination")
                if vid in previous:
                    old_road, old_segment, old_position = previous[vid]
                    if (road, segment) == (old_road, old_segment) and position > old_position:
                        within_segment_progress = True
                previous[vid] = road, segment, position
            assert np.isfinite(session._flow()).all()
            if session.done:
                break
        assert session.done and len(session.travel_times) == len(topology.vehicles)
        assert session.metrics["physical_revisit_count"] == 0
    assert within_segment_progress, "positions must interpolate inside a fixed segment"
    assert observed_kinds == {"fork", "singleton", "destination"}


def test_real_meso_windows_departure_arrival_and_exposure_conservation():
    config, prepared, topology = meso_setup()
    departures, exposure = 0., 0.
    with SumoEventSession(config, prepared, topology, backend="libsumo") as session:
        session.reset()
        for _ in range(20):
            result = session.advance_window(greedy(session))
            assert result.end_time - result.start_time == 30.
            assert result.end_time % 30 == 0
            assert result.action_counts.sum() == len(result.decisions)
            assert all(result.start_time <= action.decision_time < result.end_time for action in result.decisions)
            assert all(action.decision_time % 1 == 0 for action in result.decisions)
            departures += result.departure_counts.sum()
            exposure += result.active_vehicle_seconds
            if session.done:
                break
        assert session.done
        assert departures == len(topology.vehicles)
        assert exposure == pytest.approx(sum(session.travel_times.values()))
        assert exposure == session.metrics["active_vehicle_seconds"]
        for row in result.next_state["vehicles"]:
            assert row["arrived"] and row["actual_departure_time"] <= row["observed_departure_time"]
            assert row["current_edge"] == row["destination_edge"]
            assert session.travel_times[row["vehicle_id"]] == row["arrival_time"] - row["actual_departure_time"]
        padded = session.advance_window(greedy(session))
        assert padded.active_vehicle_seconds == 0 and not padded.decisions


def test_real_meso_edge_incident_changes_travel_and_restores_speed():
    times = {}
    for incident in (False, True):
        config, prepared, topology = meso_setup()
        topology = replace(topology, vehicles=(VehicleDemand("incident-car", 0, 0, 0.),))
        config.simulation.incident_edge = "A_in"
        config.simulation.incident_start_seconds = 0.
        config.simulation.incident_end_seconds = 60. if incident else 0.
        config.simulation.incident_speed_mps = 1.
        with SumoEventSession(config, prepared, topology, backend="libsumo") as session:
            session.reset()
            baseline = session._connection.lane.getMaxSpeed("A_in_0")
            session._apply_incident()
            if incident:
                assert session._connection.lane.getMaxSpeed("A_in_0") == 1.
            for _ in range(20):
                session.advance_window(greedy(session))
                if session._time_ms >= 60000:
                    session._apply_incident()
                    assert session._connection.lane.getMaxSpeed("A_in_0") == baseline
                    assert not session._incident_active
                if session.done:
                    break
            assert session.done
            times[incident] = session.travel_times["incident-car"]
    assert times[True] > times[False] + 20, times


def test_real_meso_obeys_red_light_then_releases_without_internal_state(tmp_path):
    config, prepared, topology = meso_setup()
    tree = ET.parse(prepared.network_xml)
    root = tree.getroot()
    junction = next(node for node in root.findall("junction") if node.get("id") == "A1")
    junction.set("type", "traffic_light")
    for connection in root.findall("connection"):
        if connection.get("from") == "A_in":
            connection.set("tl", "A1")
            connection.set("linkIndex", "0")
    logic = ET.Element("tlLogic", id="A1", type="static", programID="0", offset="0")
    ET.SubElement(logic, "phase", duration="3600", state="r")
    root.insert(1, logic)
    prepared.network_xml = tmp_path / "red.net.xml"
    tree.write(prepared.network_xml)
    topology = replace(topology, vehicles=(VehicleDemand("red-car", 0, 0, 0.),))
    with SumoEventSession(config, prepared, topology, backend="libsumo") as session:
        session.reset()
        session.advance_window(greedy(session))
        row = session._vehicles["red-car"]
        assert not row["arrived"]
        assert row["current_edge"] == session.network.edge_to_index["A_in"]
        assert not row["in_junction"] and row["junction_remaining_distance"] == 0
        assert session.metrics["events"] == 0
        assert session._connection.vehicle.getSpeed("red-car") == 0
        session._connection.trafficlight.setRedYellowGreenState("A1", "G")
        for _ in range(20):
            session.advance_window(greedy(session))
            if session.done:
                break
        assert session.done and session.metrics["events"] > 0


def test_real_meso_skipped_fork_remains_route_followed_and_history_audited(tmp_path):
    config, prepared, topology = meso_setup()
    tree = ET.parse(prepared.network_xml)
    for edge in tree.getroot().findall("edge"):
        if edge.get("id") in {"A_in", "A_hold", "A_to_R"}:
            for lane in edge.findall("lane"):
                lane.set("length", "0.10")
    prepared.network_xml = tmp_path / "short.net.xml"
    tree.write(prepared.network_xml)
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=(VehicleDemand("short-car", 0, 0, 0.),))
    skipped_fork = topology.network.edge_to_index["A_hold"]
    with SumoEventSession(config, prepared, topology, backend="libsumo", forbid_revisit=True) as session:
        session.reset()
        actions = []
        histories = []
        for _ in range(200):
            actions.extend(session._apply_decisions(greedy(session), session._flow()))
            session._connection.simulationStep()
            session._time_ms += 1000
            session._observe(np.zeros(session.network.edge_count))
            histories.append(session._vehicles["short-car"]["visited_edges"])
            if session.done:
                break
        assert session.done
        assert skipped_fork in histories[-1]
        assert all(action.current_edge != skipped_fork for action in actions)
        assert any(len(current) - len(previous) > 1 for previous, current in zip(histories, histories[1:]))
        assert histories[-1][-1] == session._vehicles["short-car"]["destination_edge"]
        assert len(histories[-1]) == len(set(histories[-1]))


def test_real_meso_multilane_single_queue_blocks_green_turn_behind_red_turn(tmp_path):
    config, prepared, topology = meso_setup()
    tree = ET.parse(prepared.network_xml)
    root = tree.getroot()
    hold = next(edge for edge in root.findall("edge") if edge.get("id") == "A_hold")
    lane = deepcopy(hold.find("lane"))
    lane.set("id", "A_hold_1")
    lane.set("index", "1")
    hold.append(lane)
    junction = next(node for node in root.findall("junction") if node.get("id") == "A2")
    junction.set("type", "traffic_light")
    junction.set("incLanes", "A_hold_0 A_hold_1")
    for request in junction.findall("request"):
        junction.remove(request)
    for index in range(6):
        ET.SubElement(junction, "request", index=str(index), response="000000", foes="000000", cont="0")
    controlled = [connection for connection in root.findall("connection") if connection.get("from") == "A_hold"]
    states = []
    for index, connection in enumerate(controlled):
        connection.set("tl", "A2")
        connection.set("linkIndex", str(index))
        states.append("r" if connection.get("to") == "A_to_R" else "G")
        duplicate = deepcopy(connection)
        duplicate.set("fromLane", "1")
        root.append(duplicate)
    logic = ET.Element("tlLogic", id="A2", type="static", programID="0", offset="0")
    ET.SubElement(logic, "phase", duration="3600", state="".join(states))
    root.insert(1, logic)
    prepared.network_xml = tmp_path / "combined-queue.net.xml"
    tree.write(prepared.network_xml)
    topology = load_traffic_topology(prepared)
    topology = replace(topology, vehicles=(VehicleDemand("front-red", 0, 0, 0.),
                                           VehicleDemand("behind-green", 0, 0, 3.)))
    with SumoEventSession(config, prepared, topology, backend="libsumo") as session:
        session.reset()
        hold_index = session.network.edge_to_index["A_hold"]
        base = greedy(session)
        def policy(rows, flow, now):
            actions = base(rows, flow, now)
            return [NextHopDecision(action.vehicle_id, action.current_edge,
                session.network.edge_to_index["A_to_R" if action.vehicle_id == "front-red" else "A_pre_X"], now)
                if action.current_edge == hold_index else action for action in actions]
        session.advance_window(policy)
        session.advance_window(policy)
        assert session._connection.edge.getLaneNumber("A_hold") == 2
        for vid in ("front-red", "behind-green"):
            row = session._vehicles[vid]
            assert row["current_edge"] == hold_index
            assert not row["in_junction"] and not row["pending_decision"]
        assert session._vehicles["behind-green"]["committed_next_edge"] == session.network.edge_to_index["A_pre_X"]
        session._connection.trafficlight.setRedYellowGreenState("A2", "G" * len(states))
        for _ in range(20):
            session.advance_window(policy)
            if session.done:
                break
        assert session.done and len(session.travel_times) == 2
