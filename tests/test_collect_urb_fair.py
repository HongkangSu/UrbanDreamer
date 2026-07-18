import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from scripts.collect_urb_fair import (  # noqa: E402
    _default_output_dir,
    _default_records_dir,
    _default_task_name,
    _episode_matches_selection,
    _episode_all_arrived,
    _episode_selection_metrics,
    _episode_selection_score,
    _flush_partial_collection,
    _write_collection_manifest,
)
import dreamer4.urban_routing.data_collector as data_collector_module  # noqa: E402
from dreamer4.urban_routing.data_collector import DataCollector  # noqa: E402
from dreamer4.urban_routing.data_collector import _copy_agents_csv_to_records  # noqa: E402
from dreamer4.urban_routing.routerl_connector import RouteRLConnector  # noqa: E402
import dreamer4.urban_routing.routerl_connector as routerl_connector_module  # noqa: E402


def _write_settings(tmp_path, records_dir):
    settings_path = Path(tmp_path) / "urb_fair.env"
    settings_path.write_text(
        "\n".join(
            [
                "NETWORK_NAME=ingolstadt_custom",
                f"NETWORK_DIR={records_dir}",
                f"EDGE_REGISTRY={records_dir / 'edge_registry.json'}",
                f"AGENTS_CSV={records_dir / 'agents.csv'}",
                f"ROUTE_ARTIFACTS_DIR={records_dir}",
                "RATIO_MACHINES=0.4",
                "HDV_ADAPT_EP=200",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(settings_path)


class _FakeEdge:
    def __init__(self, edge_id):
        self.edge_id = edge_id
        self.allowed = []

    def getID(self):
        return self.edge_id

    def allows(self, _vclass):
        return True

    def getAllowedOutgoing(self, _vclass):
        return list(self.allowed)


class _FakeNet:
    def __init__(self):
        self.edges = {edge_id: _FakeEdge(edge_id) for edge_id in ["a", "b", "c", "d"]}

    def connect(self, src, dst):
        self.edges[src].allowed.append(self.edges[dst])

    def getEdge(self, edge_id):
        return self.edges[edge_id]


class _FakeRouteConnector:
    def __init__(self, net):
        self.network = net


class _FakeEdgeRegistry:
    def __init__(self):
        self.n_edges = 3
        self._edge_to_idx = {"a": 0, "b": 1, "c": 2}
        self.lengths = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        self.speed_limits = np.array([10.0, 10.0, 10.0], dtype=np.float32)


class _FakeTraciVehicle:
    def __init__(self):
        self.road_id = "a"
        self.lane_id = "a_0"
        self.lane_position = 95.0
        self.route = ["a", "b", "d"]
        self.route_index = 0

    def getRoadID(self, _veh_id):
        return self.road_id

    def getLaneID(self, _veh_id):
        return self.lane_id

    def getLanePosition(self, _veh_id):
        return self.lane_position

    def getRoute(self, _veh_id):
        return list(self.route)

    def getRouteIndex(self, _veh_id):
        return self.route_index


class _FakeTraciLane:
    def __init__(self):
        self.lengths = {"a_0": 100.0, "a_1": 100.0}
        self.links = {
            "a_0": [("b_0",)],
            "a_1": [("c_0",)],
        }

    def getLinks(self, lane_id):
        return list(self.links.get(lane_id, []))

    def getLength(self, lane_id):
        return self.lengths[lane_id]


class _FakeTraciEdgeApi:
    def getLaneNumber(self, edge_id):
        return 2 if edge_id == "a" else 1


class _FakeTraci:
    def __init__(self):
        self.vehicle = _FakeTraciVehicle()
        self.lane = _FakeTraciLane()
        self.edge = _FakeTraciEdgeApi()


def test_collect_urb_fair_default_paths_are_network_scoped():
    cfg = {"NETWORK_NAME": "ingolstadt_custom"}
    assert _default_output_dir(cfg) == "data/ingolstadt_custom/collect_urb_fair"
    assert _default_task_name(cfg) == "ingolstadt_custom_urb_fair"
    assert _default_records_dir(cfg) == "results/ingolstadt_custom/collect_urb_fair_records"


def test_copy_agents_csv_to_records_uses_explicit_agents(tmp_path):
    src = tmp_path / "agents_custom.csv"
    src.write_text(
        "id,start_time,origin,destination,kind\n0,1,4,20,Human\n",
        encoding="utf-8",
    )
    records = tmp_path / "records"

    out = _copy_agents_csv_to_records(str(src), str(records))

    assert Path(out) == records / "agents.csv"
    assert (records / "agents.csv").read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def _make_role_only_collector(seed=42, random_av_fraction=0.6):
    collector = DataCollector.__new__(DataCollector)
    collector.seed = seed
    collector.av_fraction = 0.4
    collector.random_av_fraction = random_av_fraction
    collector._fixed_machine_ids = set()
    collector._fixed_human_ids = set()
    collector._av_vehicles = set()
    collector._hdv_vehicles = set()
    collector._random_avs = set()
    collector._greedy_avs = set()
    return collector


def test_fixed_roles_preserve_legacy_all_epsilon_greedy_policy_across_episodes():
    machine_ids = [f"cav_{idx:03d}" for idx in range(128)]
    human_ids = [f"hdv_{idx:03d}" for idx in range(192)]
    all_ids = machine_ids + human_ids

    collector = _make_role_only_collector()
    collector.set_fixed_roles(machine_ids, human_ids)
    collector._assign_roles(all_ids, np.random.RandomState(11))

    assert collector._random_avs == set()
    assert collector._greedy_avs == set(machine_ids)

    collector._av_vehicles.clear()
    collector._hdv_vehicles.clear()
    collector._random_avs.clear()
    collector._greedy_avs.clear()
    collector._assign_roles(list(reversed(all_ids)), np.random.RandomState(999))

    assert collector._random_avs == set()
    assert collector._greedy_avs == set(machine_ids)

    reordered = _make_role_only_collector()
    reordered.set_fixed_roles(list(reversed(machine_ids)), human_ids)
    reordered._assign_roles(all_ids, np.random.RandomState(23))
    assert reordered._random_avs == set()
    assert reordered._greedy_avs == set(machine_ids)


def test_data_collector_candidate_suffixes_filter_invalid_sumo_routes():
    net = _FakeNet()
    net.connect("a", "b")
    net.connect("b", "d")

    collector = DataCollector.__new__(DataCollector)
    collector.connector = _FakeRouteConnector(net)
    collector.n_routes = 4
    collector._veh_candidates = {
        "veh": [
            ["a", "b", "d"],
            ["a", "c", "d"],
        ]
    }

    suffixes = collector._candidate_route_suffixes("veh", "a")

    assert suffixes == [["a", "b", "d"]] * 4


def test_data_collector_reroute_safety_rejects_late_lane_impossible_turn(monkeypatch):
    net = _FakeNet()
    net.connect("a", "b")
    net.connect("a", "c")
    net.connect("b", "d")
    net.connect("c", "d")

    fake_traci = _FakeTraci()
    monkeypatch.setattr(data_collector_module, "TRACI_AVAILABLE", True)
    monkeypatch.setattr(data_collector_module, "traci", fake_traci)

    collector = DataCollector.__new__(DataCollector)
    collector.connector = _FakeRouteConnector(net)
    collector.n_routes = 4
    collector._safe_reroute_min_remaining_m = 20.0

    assert collector._route_is_safe_for_current_lane("veh", ["a", "b", "d"])
    assert not collector._route_is_safe_for_current_lane("veh", ["a", "c", "d"])

    fake_traci.vehicle.lane_position = 10.0
    assert collector._route_is_safe_for_current_lane("veh", ["a", "c", "d"])


def test_data_collector_seeds_departures_and_deduplicates_arrivals(monkeypatch):
    monkeypatch.setattr(data_collector_module, "TRACI_AVAILABLE", True)

    collector = DataCollector.__new__(DataCollector)
    collector.connector = type(
        "Connector",
        (),
        {"_agents_df": pd.DataFrame({"id": ["1", "2"], "start_time": [10.0, 20.0]})},
    )()
    collector._last_ep_travel_times = []
    collector._last_ep_route_tts = {}
    collector._veh_od = {}
    collector._hdv_route_idx = {}
    collector._av_route_idx = {}

    veh_depart = {}
    collector._seed_depart_times_from_agents(veh_depart)
    assert veh_depart == {"1": 10.0, "2": 20.0}

    veh_travel_times = {}
    collector._record_arrivals(
        veh_depart,
        veh_travel_times,
        arrival_events=[("1", 40.0), ("1", 41.0), ("2", 80.0)],
    )

    assert collector._last_ep_travel_times == [30.0, 60.0]
    assert veh_travel_times == {"1": 30.0, "2": 60.0}


def test_routerl_connector_close_handles_failed_initialization():
    connector = RouteRLConnector.__new__(RouteRLConnector)

    connector.close()

    assert connector._started is False


def test_routerl_bulk_advance_stops_when_sumo_has_no_expected_vehicles(monkeypatch):
    class _FakeSimulation:
        def __init__(self):
            self.step = 0

        def simulationStep(self):
            self.step += 1

        def getTime(self):
            return float(self.step)

        def getArrivedIDList(self):
            return ["veh0"] if self.step == 2 else []

        def getMinExpectedNumber(self):
            return 0 if self.step >= 2 else 1

    fake_sim = _FakeSimulation()
    fake_traci = type(
        "FakeTraci",
        (),
        {
            "simulationStep": fake_sim.simulationStep,
            "simulation": fake_sim,
        },
    )()
    monkeypatch.setattr(routerl_connector_module, "TRACI_AVAILABLE", True)
    monkeypatch.setattr(routerl_connector_module, "traci", fake_traci)

    connector = RouteRLConnector.__new__(RouteRLConnector)
    connector.seed = 42
    connector.current_step = 0
    connector.env = type("FakeEnv", (), {"simulator": None})()
    connector._started = False
    connector._apply_incident_for_current_time = lambda: None

    arrivals = connector.advance_steps(9000)

    assert fake_sim.step == 2
    assert connector.current_step == 2
    assert arrivals == [("veh0", 2.0)]


def test_planned_av_edge_flow_encodes_future_route_pressure_not_current_occupancy():
    collector = DataCollector.__new__(DataCollector)
    collector._edge_registry = _FakeEdgeRegistry()

    state = {
        "vehicles": {
            "v1": {"edge": "a"},
            "v2": {"edge": "b"},
        },
        "edges": {
            "a": {"mean_speed": 10.0},
            "b": {"mean_speed": 10.0},
            "c": {"mean_speed": 10.0},
        },
    }
    actions = {
        "v1": ["a", "b", "c"],
        "v2": ["b", "c"],
    }

    planned = collector._compute_planned_av_edge_flow(
        state,
        actions,
        horizon_seconds=15.0,
    )

    assert planned[0] == 0.0
    assert planned[1] == pytest.approx(0.5)
    assert planned[2] == pytest.approx(0.5)


def test_completion_aware_reward_penalizes_unfinished_vehicles():
    collector = DataCollector.__new__(DataCollector)
    collector._num_agents = 4
    collector._last_ep_travel_times = [60.0, 120.0]
    collector._global_avg_travel_time = lambda _frame: 999.0

    reward = collector._completion_aware_episode_reward(
        current_tick=300,
        max_steps=240,
        fallback_frame=np.zeros((1, 8), dtype=np.float32),
    )

    assert reward == pytest.approx(-3.25)


def test_collect_urb_fair_manifest_records_protocol_defaults(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    for name, content in (
        ("agents.csv", "id,start_time,origin,destination,kind\n0,1,0,0,Human\n"),
        ("routes.csv", "origins,destinations,path,free_flow_time\n0,0,a b,1.0\n"),
        ("route.rou.xml", "<routes></routes>\n"),
    ):
        (records_dir / name).write_text(content, encoding="utf-8")

    settings_path = _write_settings(tmp_path, records_dir)
    out_path = _write_collection_manifest(
        records_dir=records_dir,
        settings_path=settings_path,
        output_dir=tmp_path / "output",
        task_name="ing_test",
        num_episodes=7,
        warmup_episodes=200,
        post_mutation_state_path=records_dir / "post_mutation_state.json",
        hdv_state_in_path=None,
        hdv_state_out_path=records_dir / "hdv_final.json",
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["should_humans_adapt"] is False
    assert saved["num_episodes"] == 7
    assert saved["episodes_saved"] == 7
    assert saved["partial"] is False
    assert saved["warmup_episodes"] == 200
    assert saved["artifact_hashes"]["agents_csv"]


def test_collect_urb_fair_manifest_records_post_mutation_hdv_learning(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    for name, content in (
        ("agents.csv", "id,start_time,origin,destination,kind\n0,1,0,0,Human\n"),
        ("routes.csv", "origins,destinations,path,free_flow_time\n0,0,a b,1.0\n"),
        ("route.rou.xml", "<routes></routes>\n"),
    ):
        (records_dir / name).write_text(content, encoding="utf-8")

    settings_path = _write_settings(tmp_path, records_dir)
    out_path = _write_collection_manifest(
        records_dir=records_dir,
        settings_path=settings_path,
        output_dir=tmp_path / "output",
        task_name="ing_test",
        num_episodes=7,
        warmup_episodes=200,
        post_mutation_state_path=records_dir / "post_mutation_state.json",
        hdv_state_in_path=None,
        hdv_state_out_path=records_dir / "hdv_final.json",
        post_mutation_hdv_learning=True,
    )
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert saved["should_humans_adapt"] is True
    assert saved["post_mutation_hdv_learning"] is True


def test_save_intermediate_writes_demo_and_flow_shards(tmp_path):
    collector = DataCollector.__new__(DataCollector)
    collector.output_dir = Path(tmp_path)
    collector._flows_dir = Path(tmp_path) / "flows"
    collector._flows_dir.mkdir(parents=True, exist_ok=True)
    collector.shard_size = 4

    episodes = list(range(6))
    av = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    hdv = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    joint = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    rewards = [0.0 for _ in episodes]
    departs = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    contexts = [np.zeros((3,), dtype=np.float32) for _ in episodes]
    flows = [np.zeros((2, 8), dtype=np.float32) for _ in episodes]

    collector._save_intermediate(
        "sample_task",
        episodes,
        av,
        hdv,
        joint,
        rewards,
        flows,
        departs,
        contexts,
    )

    demo_path = Path(tmp_path) / "sample_task_intermediate.pt"
    shard_dir = Path(tmp_path) / "flows" / "sample_task_intermediate"
    assert demo_path.exists()
    assert shard_dir.exists()
    shard_paths = sorted(shard_dir.glob("shard_*.pt"))
    assert len(shard_paths) == 2
    saved = torch.load(demo_path, map_location="cpu", weights_only=False)
    assert tuple(saved["episode"].shape) == (6,)
    first = torch.load(shard_paths[0], map_location="cpu", weights_only=False)
    second = torch.load(shard_paths[1], map_location="cpu", weights_only=False)
    assert tuple(first["flows"].shape) == (4, 2, 8)
    assert tuple(second["flows"].shape) == (2, 2, 8)


def test_data_collector_demo_dict_accepts_pair_metadata():
    collector = DataCollector.__new__(DataCollector)
    av = [np.zeros((2,), dtype=np.float32) for _ in range(3)]
    hdv = [np.zeros((2,), dtype=np.float32) for _ in range(3)]
    joint = [np.zeros((2,), dtype=np.float32) for _ in range(3)]

    demo = collector._build_demo_dict(
        episodes=[0, 0, 0],
        av_edge_actions=av,
        hdv_edge_actions=hdv,
        joint_edge_actions=joint,
        rewards=[0.0, 0.0, 0.0],
        pair_ids=[7, 7, 7],
        variant_ids=[0, 1, 2],
    )

    assert torch.equal(demo["pair_id"], torch.tensor([7, 7, 7]))
    assert torch.equal(demo["variant_id"], torch.tensor([0, 1, 2]))
    assert torch.equal(demo["av_planned_edge_flow"], demo["av_edge_action"])
    assert int(demo["action_schema_version"].item()) == 2


def test_flush_partial_collection_writes_partial_manifest_and_artifacts(tmp_path):
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    for name, content in (
        ("agents.csv", "id,start_time,origin,destination,kind\n0,1,0,0,Human\n"),
        ("routes.csv", "origins,destinations,path,free_flow_time\n0,0,a b,1.0\n"),
        ("route.rou.xml", "<routes></routes>\n"),
    ):
        (records_dir / name).write_text(content, encoding="utf-8")

    collector = DataCollector.__new__(DataCollector)
    collector.output_dir = output_dir
    collector._flows_dir = output_dir / "flows"
    collector._flows_dir.mkdir(parents=True, exist_ok=True)
    collector.shard_size = 4

    episodes = list(range(6))
    av = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    hdv = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    joint = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    rewards = [0.0 for _ in episodes]
    departs = [np.zeros((2,), dtype=np.float32) for _ in episodes]
    contexts = [np.zeros((3,), dtype=np.float32) for _ in episodes]
    flows = [np.zeros((2, 8), dtype=np.float32) for _ in episodes]

    settings_path = _write_settings(tmp_path, records_dir)
    manifest_path = _flush_partial_collection(
        collector=collector,
        task_name="partial_task",
        records_dir=records_dir,
        settings_path=settings_path,
        output_dir=output_dir,
        requested_num_episodes=10,
        warmup_episodes=0,
        post_mutation_state_path=records_dir / "post_mutation_state.json",
        hdv_state_in_path=None,
        hdv_state_out_path=None,
        loaded_hdv_state=None,
        all_episodes=episodes,
        all_av_edge_actions=av,
        all_hdv_edge_actions=hdv,
        all_joint_edge_actions=joint,
        all_rewards=rewards,
        all_flows=flows,
        all_departs=departs,
        all_contexts=contexts,
    )

    assert manifest_path is not None
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["partial"] is True
    assert saved["num_episodes"] == 10
    assert saved["episodes_saved"] == 6
    assert (output_dir / "partial_task_intermediate.pt").exists()
    assert (output_dir / "flows" / "partial_task_intermediate" / "shard_0000.pt").exists()


def test_episode_selection_metrics_extracts_avg_tt_and_route_structure():
    collector = DataCollector.__new__(DataCollector)
    collector._last_ep_travel_times = [300.0, 360.0, 420.0]
    collector._last_ep_route_tts = {
        (0, 0, 0): [100.0],
        (0, 0, 1): [160.0],
        (1, 1, 0): [200.0],
    }
    collector._global_avg_travel_time = lambda flow: 999.0

    frame = np.zeros((2, 8), dtype=np.float32)
    frame[:, 2] = np.array([0.1, 0.01], dtype=np.float32)
    metrics = _episode_selection_metrics(collector, ep_frames=[frame])
    assert round(metrics["avg_tt_min"], 2) == 6.0
    assert metrics["route_tt_key_count"] == 3.0
    assert metrics["route_tt_od_count"] == 2.0
    assert metrics["route_tt_gap_mean_sec"] == 60.0
    assert metrics["busy_edge_count_mean"] == 1.0
    assert metrics["busy_edge_mass_mean"] == pytest.approx(0.1)


def test_episode_matches_selection_uses_band_and_route_thresholds():
    metrics = {
        "avg_tt_min": 7.5,
        "route_tt_od_count": 3.0,
        "route_tt_gap_mean_sec": 42.0,
        "busy_edge_count_mean": 1.0,
        "busy_edge_mass_mean": 0.1,
    }
    assert _episode_matches_selection(
        metrics,
        min_avg_tt_min=6.5,
        max_avg_tt_min=8.5,
        min_route_tt_od_count=2,
        min_route_tt_gap_sec=30.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=8.0,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )

    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=7.0,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=4,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=45.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=40.0,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=45.0,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=2.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=0.5,
        min_busy_edge_mass_mean=0.0,
    )
    assert not _episode_matches_selection(
        metrics,
        min_avg_tt_min=None,
        max_avg_tt_min=None,
        min_route_tt_od_count=0,
        min_route_tt_gap_sec=0.0,
        max_route_tt_gap_sec=None,
        min_busy_edge_count_mean=0.0,
        max_busy_edge_count_mean=None,
        min_busy_edge_mass_mean=0.2,
    )


def test_episode_all_arrived_requires_every_agent():
    collector = DataCollector.__new__(DataCollector)
    collector._num_agents = 3

    assert _episode_all_arrived(collector, {"n_arrived": 3.0})
    assert not _episode_all_arrived(collector, {"n_arrived": 2.0})

    collector._num_agents = 0
    assert not _episode_all_arrived(collector, {"n_arrived": 10.0})


def test_episode_selection_score_rewards_closeness_to_targets():
    metrics = {
        "avg_tt_min": 7.0,
        "route_tt_gap_mean_sec": 180.0,
        "busy_edge_count_mean": 33.0,
        "busy_edge_mass_mean": 4.8,
    }
    score_good = _episode_selection_score(
        metrics,
        target_avg_tt_min=7.0,
        target_route_tt_gap_sec=180.0,
        target_busy_edge_count_mean=33.0,
        target_busy_edge_mass_mean=4.8,
        avg_tt_tolerance_min=1.0,
        route_tt_gap_tolerance_sec=60.0,
        busy_edge_count_tolerance=3.0,
        busy_edge_mass_tolerance=1.0,
        avg_tt_weight=1.0,
        route_tt_gap_weight=1.0,
        busy_edge_count_weight=1.0,
        busy_edge_mass_weight=1.0,
    )
    score_bad = _episode_selection_score(
        {
            "avg_tt_min": 5.0,
            "route_tt_gap_mean_sec": 20.0,
            "busy_edge_count_mean": 10.0,
            "busy_edge_mass_mean": 1.0,
        },
        target_avg_tt_min=7.0,
        target_route_tt_gap_sec=180.0,
        target_busy_edge_count_mean=33.0,
        target_busy_edge_mass_mean=4.8,
        avg_tt_tolerance_min=1.0,
        route_tt_gap_tolerance_sec=60.0,
        busy_edge_count_tolerance=3.0,
        busy_edge_mass_tolerance=1.0,
        avg_tt_weight=1.0,
        route_tt_gap_weight=1.0,
        busy_edge_count_weight=1.0,
        busy_edge_mass_weight=1.0,
    )
    assert score_good == pytest.approx(1.0)
    assert score_bad == 0.0
