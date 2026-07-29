from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import urban_dreamer.traffic.environment as environment_module
from urban_dreamer.traffic.environment import (
    ActionDecision,
    RouteRLSessionSpec,
    RouteRLSumoSession,
    build_live_session_spec,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class _FakeVehicleApi:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection
        self.add_calls: list[dict[str, str]] = []
        self.set_route_calls: list[tuple[str, list[str]]] = []
        self._states: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        self.add_calls.clear()
        self.set_route_calls.clear()
        self._states = {}

    def add(self, *, vehID: str, routeID: str, depart: str, typeID: str) -> None:
        route = list(self._connection.route_map[routeID])
        self.add_calls.append(
            {
                "vehID": str(vehID),
                "routeID": str(routeID),
                "depart": str(depart),
                "typeID": str(typeID),
            }
        )
        self._states[str(vehID)] = {
            "route": route,
            "road_id": route[0],
            "lane_id": f"{route[0]}_0",
            "lane_position": 80.0,
            "speed": 10.0,
            "waiting_time": 0.0,
        }

    def setRoute(self, vehicle_id: str, route_edges: list[str]) -> None:
        state = self._states[str(vehicle_id)]
        state["route"] = list(route_edges)
        state["road_id"] = str(route_edges[0])
        self.set_route_calls.append((str(vehicle_id), list(route_edges)))

    def getIDList(self) -> list[str]:
        return sorted(self._states)

    def getIDCount(self) -> int:
        return len(self._states)

    def getRoadID(self, vehicle_id: str) -> str:
        return str(self._states[str(vehicle_id)]["road_id"])

    def getSpeed(self, vehicle_id: str) -> float:
        return float(self._states[str(vehicle_id)]["speed"])

    def getWaitingTime(self, vehicle_id: str) -> float:
        return float(self._states[str(vehicle_id)]["waiting_time"])

    def getLaneID(self, vehicle_id: str) -> str:
        return str(self._states[str(vehicle_id)]["lane_id"])

    def getLanePosition(self, vehicle_id: str) -> float:
        return float(self._states[str(vehicle_id)]["lane_position"])


class _FakeEdgeApi:
    def __init__(self, vehicle_api: _FakeVehicleApi) -> None:
        self._vehicle_api = vehicle_api

    def getIDList(self) -> list[str]:
        return ["a", "b", "c"]

    def getLastStepVehicleNumber(self, edge_id: str) -> int:
        return sum(
            1
            for state in self._vehicle_api._states.values()
            if str(state["road_id"]) == str(edge_id)
        )

    def getLastStepMeanSpeed(self, edge_id: str) -> float:
        speeds = [
            float(state["speed"])
            for state in self._vehicle_api._states.values()
            if str(state["road_id"]) == str(edge_id)
        ]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def getLaneNumber(self, edge_id: str) -> int:
        return 2 if str(edge_id) == "a" else 1


class _FakeLaneApi:
    def __init__(self) -> None:
        self.lengths = {
            "a_0": 100.0,
            "a_1": 100.0,
            "b_0": 100.0,
            "c_0": 100.0,
        }
        self.links = {
            "a_0": [("b_0",)],
            "a_1": [("c_0",)],
            "b_0": [],
            "c_0": [],
        }
        self.max_speeds = {lane_id: 10.0 for lane_id in self.lengths}
        self.set_speed_calls: list[tuple[str, float]] = []

    def getLinks(self, lane_id: str) -> list[tuple[str]]:
        return list(self.links[str(lane_id)])

    def getLength(self, lane_id: str) -> float:
        return float(self.lengths[str(lane_id)])

    def getMaxSpeed(self, lane_id: str) -> float:
        return float(self.max_speeds[str(lane_id)])

    def setMaxSpeed(self, lane_id: str, speed: float) -> None:
        self.max_speeds[str(lane_id)] = float(speed)
        self.set_speed_calls.append((str(lane_id), float(speed)))


class _FakeSimulationApi:
    def __init__(self, connection: "_FakeConnection") -> None:
        self._connection = connection
        self.time = 0.0
        self.step = 0
        self._arrived: list[str] = []

    def reset(self) -> None:
        self.time = 0.0
        self.step = 0
        self._arrived = []

    def getTime(self) -> float:
        return float(self.time)

    def getArrivedIDList(self) -> list[str]:
        return list(self._arrived)

    def getMinExpectedNumber(self) -> int:
        return 0 if self.step >= 3 else 1


class _FakeConnection:
    def __init__(self, route_map: dict[str, list[str]]) -> None:
        self.route_map = route_map
        self.vehicle = _FakeVehicleApi(self)
        self.edge = _FakeEdgeApi(self.vehicle)
        self.lane = _FakeLaneApi()
        self.simulation = _FakeSimulationApi(self)
        self.loaded_commands: list[list[str]] = []
        self.closed = False

    def simulationStep(self) -> None:
        self.simulation.step += 1
        self.simulation.time = float(self.simulation.step)
        arrived: list[str] = []
        if self.simulation.step == 2 and "1" in self.vehicle._states:
            arrived.append("1")
        if self.simulation.step == 3 and "2" in self.vehicle._states:
            arrived.append("2")
        for vehicle_id in arrived:
            self.vehicle._states.pop(vehicle_id, None)
        self.simulation._arrived = arrived

    def load(self, command: list[str]) -> None:
        self.loaded_commands.append(list(command))
        self.vehicle.reset()
        self.simulation.reset()

    def close(self) -> None:
        self.closed = True


class _FakeTraciModule:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection
        self.start_commands: list[tuple[list[str], str]] = []

    def start(self, command: list[str], *, label: str) -> None:
        self.start_commands.append((list(command), str(label)))

    def getConnection(self, label: str) -> _FakeConnection:
        assert label == "live-test"
        return self._connection


def _make_live_spec(tmp_path: Path) -> RouteRLSessionSpec:
    network_dir = tmp_path / "network"
    route_dir = tmp_path / "prepared"
    records_dir = tmp_path / "records"
    network_dir.mkdir()
    route_dir.mkdir()
    records_dir.mkdir()
    (network_dir / "toy.net.xml").write_text("<net/>", encoding="utf-8")
    _write_csv(
        route_dir / "agents.csv",
        ["id", "origin", "destination", "start_time", "kind"],
        [
            {"id": "1", "origin": 0, "destination": 1, "start_time": 0, "kind": "Human"},
            {"id": "2", "origin": 0, "destination": 1, "start_time": 1, "kind": "Human"},
        ],
    )
    _write_csv(
        route_dir / "routes.csv",
        ["origins", "destinations", "path", "free_flow_time"],
        [
            {"origins": 0, "destinations": 1, "path": "a b", "free_flow_time": 1.0},
            {"origins": 0, "destinations": 1, "path": "a c", "free_flow_time": 1.2},
            {"origins": 0, "destinations": 1, "path": "a b", "free_flow_time": 1.3},
            {"origins": 0, "destinations": 1, "path": "a c", "free_flow_time": 1.4},
        ],
    )
    (route_dir / "route.rou.xml").write_text("<routes/>", encoding="utf-8")
    return RouteRLSessionSpec(
        network_name="toy",
        network_directory=network_dir,
        records_directory=records_dir,
        agents_path=route_dir / "agents.csv",
        routes_path=route_dir / "routes.csv",
        route_xml_path=route_dir / "route.rou.xml",
        fixed_machine_ids=("1",),
        fixed_human_ids=("2",),
        simulation_timesteps=30,
        seed=7,
        origins=("0",),
        destinations=("1",),
        initial_hdv_learning_state={
            "version": 1,
            "route_policy": [
                {
                    "origin": 0,
                    "destination": 1,
                    "probs": [0.0, 1.0, 0.0, 0.0],
                    "costs": [5.0, 6.0, 7.0, 8.0],
                }
            ],
        },
        sumo_extra_args=("--collision.action", "warn"),
        incident_edge="a",
        incident_start_seconds=0.0,
        incident_end_seconds=2.0,
        incident_speed_mps=1.0,
    )


def test_live_route_rl_session_manages_fixed_roles_reroutes_and_hdv_learning(
    tmp_path: Path,
) -> None:
    spec = _make_live_spec(tmp_path)
    route_map = {
        "0_1_0": ["a", "b"],
        "0_1_1": ["a", "c"],
        "0_1_2": ["a", "b"],
        "0_1_3": ["a", "c"],
    }
    fake_connection = _FakeConnection(route_map)
    fake_traci = _FakeTraciModule(fake_connection)
    captured_create_kwargs: dict[str, Any] = {}

    def _fake_traffic_environment(**kwargs: Any) -> Any:
        captured_create_kwargs.update(kwargs)
        simulator = SimpleNamespace(
            sumo_id="live-test",
            sumo_type=kwargs["simulator_parameters"]["sumo_type"],
            seed=kwargs["seed"],
            network_file_path=str(spec.network_directory / "toy.net.xml"),
            det_xml_save_path=str(spec.records_directory / "detectors.add.xml"),
            rou_xml_save_path=str(spec.records_directory / "route.rou.xml"),
            sumo_save_path=str(spec.records_directory / "sumo"),
            detectors_name=(),
            save_detectors_info=False,
            runs=0,
            timestep=0,
            waiting_vehicles={},
            sumo_connection=None,
        )
        return SimpleNamespace(
            simulator=simulator,
            human_agents=[],
            start=lambda: simulator.start(),
        )

    session = RouteRLSumoSession(
        edge_ids=("a", "b", "c"),
        backend_factory=lambda: environment_module._LiveRouteRLBackend(
            spec=spec,
            traffic_environment_cls=_fake_traffic_environment,
            traci_module=fake_traci,
        )
    )

    state = session.reset()

    assert captured_create_kwargs["create_agents"] is False
    assert captured_create_kwargs["create_paths"] is False
    assert (
        captured_create_kwargs["path_generation_parameters"]["number_of_paths"]
        == 4
    )
    assert fake_traci.start_commands == [
        (
            [
                "sumo",
                "--seed",
                "7",
                "--net-file",
                str(spec.network_directory / "toy.net.xml"),
                "--additional-files",
                f"{spec.records_directory / 'detectors.add.xml'},{spec.records_directory / 'route.rou.xml'}",
                "--no-step-log",
                "true",
                "--time-to-teleport",
                "-1",
                "--statistic-output",
                str(spec.records_directory / "sumo" / "sumo_stats_1.xml"),
                "--tripinfo-output",
                str(spec.records_directory / "sumo" / "detailed_sumo_stats_1.xml"),
                "--collision.action",
                "warn",
            ],
            "live-test",
        )
    ]
    assert [call["typeID"] for call in fake_connection.vehicle.add_calls] == [
        "AV",
        "Human",
    ]
    assert [call["routeID"] for call in fake_connection.vehicle.add_calls] == [
        "0_1_1",
        "0_1_1",
    ]
    assert (spec.records_directory / "agents.csv").read_text(
        encoding="utf-8"
    ).splitlines()[1].endswith(",AV")
    assert state["vehicles"]["1"]["edge"] == "a"

    fake_connection.vehicle._states["1"]["road_id"] = ":junction_0_0"
    internal_edge_state = session.get_traffic_state()
    assert internal_edge_state["vehicles"]["1"]["edge"] == ":junction_0_0"
    fake_connection.vehicle._states["1"]["road_id"] = "a"

    route_candidates = ((0, 1), (0, 2), (0, 1), (0, 2))
    route_to_c = ActionDecision.build(
        route_index=1,
        candidate_paths=route_candidates,
        edge_order=(0, 1, 2),
        edge_travel_times=(1.0, 1.0, 1.0),
        decision_horizon_seconds=30.0,
    )
    route_to_b = ActionDecision.build(
        route_index=0,
        candidate_paths=route_candidates,
        edge_order=(0, 1, 2),
        edge_travel_times=(1.0, 1.0, 1.0),
        decision_horizon_seconds=30.0,
    )
    session.set_routes_safely({"1": route_to_c, "2": route_to_b})
    assert fake_connection.vehicle.set_route_calls == []

    fake_connection.vehicle._states["1"]["lane_position"] = 10.0
    session.set_routes_safely({"1": route_to_c})
    assert fake_connection.vehicle.set_route_calls == [("1", ["a", "c"])]

    session.advance(3)

    assert session.last_active_vehicle_exposure == pytest.approx(5.0)
    assert session.travel_times_seconds() == {"1": 2.0, "2": 2.0}
    assert session.hdv_route_choices() == {"2": 1}
    assert session.travel_time_route_choices() == {"1": 1, "2": 1}
    assert session.is_done() is True
    assert ("a_0", 1.0) in fake_connection.lane.set_speed_calls
    assert ("a_0", 10.0) in fake_connection.lane.set_speed_calls

    updates = session.update_hdv_learning()
    learned_state = session.hdv_learning_state()

    assert updates == 2
    learned_row = learned_state["route_policy"][0]
    assert learned_row["costs"][1] < 6.0
    assert sum(learned_row["probs"]) == pytest.approx(1.0)
    assert learned_row["probs"][0] > learned_row["probs"][1]

    assert fake_connection.loaded_commands == []

    session.close()
    assert fake_connection.closed is True
    session.close()


def test_route_rl_constructor_smoke_uses_installed_signature_if_available() -> None:
    routerl = pytest.importorskip("routerl")
    traffic_environment = getattr(routerl, "TrafficEnvironment")
    signature = inspect.signature(traffic_environment)

    assert "seed" in signature.parameters
    assert "create_agents" in signature.parameters
    assert "create_paths" in signature.parameters
    assert "save_detectors_info" in signature.parameters


def test_build_live_session_spec_uses_prepared_contract_and_fixed_roles(
    tmp_path: Path,
) -> None:
    spec_fixture = _make_live_spec(tmp_path)
    post_state = tmp_path / "post.json"
    post_state.write_text(
        json.dumps(
            {
                "machine_agent_ids": ["1"],
                "human_agent_ids": ["2"],
            }
        ),
        encoding="utf-8",
    )
    initial_hdv_state = tmp_path / "hdv.json"
    initial_hdv_state.write_text(
        json.dumps({"version": 1, "agents": {}, "route_policy": []}),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        inputs=SimpleNamespace(network_name="toy"),
        run=SimpleNamespace(output_dir=tmp_path / "output", seed=17),
        simulation=SimpleNamespace(
            decision_intervals_per_episode=60,
            warmup_horizon_seconds=3600,
            warmup_completion_drain_seconds=7200,
            warmup_completion_patience_seconds=600,
            training_completion_buffer_seconds=2000,
            evaluation_horizon_seconds=9000,
            sumo_extra_args=("--collision.action", "warn"),
            incident_edge="a",
            incident_start_seconds=10.0,
            incident_end_seconds=20.0,
            incident_speed_mps=1.0,
        ),
    )
    prepared = SimpleNamespace(
        network_xml=spec_fixture.network_directory / "toy.net.xml",
        route_agents_csv=spec_fixture.agents_path,
        routes_csv=spec_fixture.routes_path,
        route_xml=spec_fixture.route_xml_path,
        post_mutation_state=post_state,
        initial_hdv_state=initial_hdv_state,
    )
    topology = SimpleNamespace(
        origins=("0",),
        destinations=("1",),
        route_count=4,
    )

    built = build_live_session_spec(
        config=config,
        prepared_inputs=prepared,
        topology=topology,
        initial_hdv_learning_state=json.loads(
            initial_hdv_state.read_text(encoding="utf-8")
        ),
        episode_counter=9,
    )

    assert built.network_name == "toy"
    assert built.fixed_machine_ids == ("1",)
    assert built.fixed_human_ids == ("2",)
    assert built.simulation_timesteps == 11_400
    assert built.episode_counter == 9
    assert built.sumo_extra_args == ("--collision.action", "warn")
    assert built.initial_hdv_learning_state["route_policy"] == []
