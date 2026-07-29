from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from urban_dreamer.traffic.routing import PreparedInputs
from urban_dreamer.traffic.topology import load_traffic_topology


def _write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepared_inputs(tmp_path: Path) -> PreparedInputs:
    network_xml = tmp_path / "toy.net.xml"
    network_agents = tmp_path / "network-agents.csv"
    od_file = tmp_path / "od_toy.txt"
    edge_registry = tmp_path / "edge_registry.json"
    route_agents = tmp_path / "route-agents.csv"
    routes_csv = tmp_path / "routes.csv"
    route_xml = tmp_path / "route.rou.xml"
    post_state = tmp_path / "post.json"
    initial_hdv_state = tmp_path / "hdv.json"
    predictor = tmp_path / "predictor.pt"

    for path in (network_xml, route_xml, initial_hdv_state, predictor):
        path.write_text("fixture", encoding="utf-8")
    od_file.write_text(
        '{"origins": ["a"], "destinations": ["d"]}',
        encoding="utf-8",
    )
    agents = [
        {
            "id": "av-1",
            "origin": 0,
            "destination": 0,
            "start_time": 3,
            "kind": "Human",
        },
        {
            "id": "hdv-1",
            "origin": 0,
            "destination": 0,
            "start_time": 7,
            "kind": "Human",
        },
    ]
    fields = ["id", "origin", "destination", "start_time", "kind"]
    _write_csv(network_agents, fields, agents)
    _write_csv(route_agents, fields, agents)
    _write_csv(
        routes_csv,
        ["origins", "destinations", "path", "free_flow_time"],
        [
            {
                "origins": 0,
                "destinations": 0,
                "path": "a b d",
                "free_flow_time": 3.0,
            },
            {
                "origins": 0,
                "destinations": 0,
                "path": "a c d",
                "free_flow_time": 4.0,
            },
        ],
    )
    edge_registry.write_text(
        json.dumps(
            {
                "network": "toy",
                "n_edges": 4,
                "edge_ids": ["a", "b", "c", "d"],
                "static_features": [
                    [0.1, 1.0, 0.5, 0.0],
                    [0.2, 0.8, 0.5, 0.0],
                    [0.3, 0.5, 0.5, 0.0],
                    [0.4, 1.0, 0.5, 0.0],
                ],
                "speed_limits": [10.0, 10.0, 5.0, 10.0],
                "capacities": [10.0, 20.0, 10.0, 10.0],
                "lengths": [10.0, 20.0, 15.0, 10.0],
                "adjacency": [
                    [0.0, 0.5, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                "passenger_allowed": [True, True, False, True],
            }
        ),
        encoding="utf-8",
    )
    post_state.write_text(
        json.dumps(
            {
                "machine_agent_ids": ["av-1"],
                "human_agent_ids": ["hdv-1"],
            }
        ),
        encoding="utf-8",
    )
    return PreparedInputs(
        network_xml=network_xml,
        network_agents_csv=network_agents,
        od_file=od_file,
        edge_registry=edge_registry,
        route_agents_csv=route_agents,
        routes_csv=routes_csv,
        route_xml=route_xml,
        post_mutation_state=post_state,
        initial_hdv_state=initial_hdv_state,
        hdv_response_checkpoint=predictor,
        automated_vehicle_count=1,
        human_vehicle_count=1,
        automated_vehicle_fraction=0.5,
        route_count=2,
        mutation_episode_index=10,
        input_hashes={},
    )


def test_load_traffic_topology_builds_one_shared_network_contract(
    tmp_path: Path,
) -> None:
    topology = load_traffic_topology(_prepared_inputs(tmp_path))

    assert topology.network.edge_ids == ("a", "b", "c", "d")
    assert topology.network.edge_to_index == {"a": 0, "b": 1, "c": 2, "d": 3}
    assert topology.network.adjacency_list == ((1,), (3,), (), ())
    np.testing.assert_allclose(
        topology.network.free_flow_travel_seconds,
        [1.0, 2.0, 3.0, 1.0],
    )
    assert topology.origins == ("a",)
    assert topology.destinations == ("d",)
    assert topology.route_paths == (((0, 1, 3), (0, 2, 3)),)
    assert topology.od_pairs == ((0, 0),)
    assert topology.od_automated_counts.tolist() == [1.0]
    assert topology.od_human_counts.tolist() == [1.0]
    assert [vehicle.vehicle_id for vehicle in topology.vehicles] == [
        "av-1",
        "hdv-1",
    ]
    assert topology.vehicles[0].is_automated is True
    assert topology.vehicles[1].is_automated is False


def test_extract_flow_uses_free_flow_defaults_and_normalizes_sumo_state(
    tmp_path: Path,
) -> None:
    network = load_traffic_topology(_prepared_inputs(tmp_path)).network

    flow = network.extract_flow(
        {
            "a": {
                "occupancy": 0.25,
                "mean_speed": 5.0,
                "vehicle_number": 5.0,
                "halting_number": 2.0,
            },
            "c": {
                "occupancy": 75.0,
                "mean_speed": 2.5,
                "vehicle_number": 2.0,
                "halting_number": 1.0,
            },
        }
    )

    assert flow.shape == (4, 8)
    np.testing.assert_allclose(flow[0, :4], [0.25, 0.5, 0.5, 0.2])
    np.testing.assert_allclose(flow[1, :4], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(flow[2, :4], [0.75, 0.5, 0.2, 0.1])
    np.testing.assert_allclose(flow[:, 4:], network.static_features)
