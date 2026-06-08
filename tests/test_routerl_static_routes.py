import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dreamer4.urban_routing import routerl_shared  # noqa: E402


class _ToyEdgeRegistry:
    def __init__(self, *, include_safe_branch: bool) -> None:
        self.edge_ids = ["a", "short", "safe", "d"]
        self._edge_to_idx = {edge: idx for idx, edge in enumerate(self.edge_ids)}
        self.lengths = np.asarray([10.0, 0.5, 5.0, 10.0], dtype=np.float32)
        self.speed_limits = np.ones((4,), dtype=np.float32)
        self.passenger_allowed = np.ones((4,), dtype=bool)
        self.adjacency = np.zeros((4, 4), dtype=np.float32)
        self.adjacency[0, 1] = 1.0
        self.adjacency[1, 3] = 1.0
        if include_safe_branch:
            self.adjacency[0, 2] = 1.0
            self.adjacency[2, 3] = 1.0


def _write_cached_routes(records_dir: Path, *, route_path: str = "a short d") -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / "routes.csv").write_text(
        "origins,destinations,path,free_flow_time\n"
        f"0,0,{route_path},1.0\n",
        encoding="utf-8",
    )
    (records_dir / "route.rou.xml").write_text("<routes></routes>\n", encoding="utf-8")
    (records_dir / "routes_meta.json").write_text(
        json.dumps(
            {
                "generator": "dijkstra_passenger",
                "network_name": "toy",
                "n_routes": 1,
                "n_origins": 1,
                "n_destinations": 1,
                "od_pairs": [[0, 0]],
            }
        ),
        encoding="utf-8",
    )


def test_static_cached_routes_replace_short_intermediate_edges(monkeypatch, tmp_path):
    records_dir = tmp_path / "records"
    _write_cached_routes(records_dir)
    monkeypatch.setenv("URB_STATIC_ROUTE_HARD_MIN_EDGE_LENGTH_M", "1.0")
    monkeypatch.setenv("URB_DIJKSTRA_HARD_MIN_EDGE_LENGTH_M", "0.0")
    monkeypatch.setattr(
        routerl_shared,
        "build_edge_registry",
        lambda _folder, _name: _ToyEdgeRegistry(include_safe_branch=True),
    )

    df = routerl_shared.generate_dijkstra_routes(
        network_folder=str(tmp_path),
        network_name="toy",
        origins=["a"],
        destinations=["d"],
        n_routes=1,
        records_folder=str(records_dir),
    )

    assert df["path"].tolist() == ["a safe d"]
    assert pd.read_csv(records_dir / "routes.csv")["path"].tolist() == ["a safe d"]
    assert "a safe d" in (records_dir / "route.rou.xml").read_text(encoding="utf-8")
    meta = json.loads((records_dir / "routes_meta.json").read_text(encoding="utf-8"))
    assert meta["hard_min_edge_length_m"] == 1.0
    assert meta["short_edge_routes_seen"] == 1
    assert meta["short_edge_routes_repaired"] == 1


def test_static_cached_routes_fail_when_all_paths_need_short_edges(monkeypatch, tmp_path):
    records_dir = tmp_path / "records"
    _write_cached_routes(records_dir)
    monkeypatch.setenv("URB_STATIC_ROUTE_HARD_MIN_EDGE_LENGTH_M", "1.0")
    monkeypatch.setenv("URB_DIJKSTRA_HARD_MIN_EDGE_LENGTH_M", "0.0")
    monkeypatch.setattr(
        routerl_shared,
        "build_edge_registry",
        lambda _folder, _name: _ToyEdgeRegistry(include_safe_branch=False),
    )

    with pytest.raises(RuntimeError, match="No 1m-safe static routes"):
        routerl_shared.generate_dijkstra_routes(
            network_folder=str(tmp_path),
            network_name="toy",
            origins=["a"],
            destinations=["d"],
            n_routes=1,
            records_folder=str(records_dir),
        )

    assert pd.read_csv(records_dir / "routes.csv")["path"].tolist() == ["a short d"]
