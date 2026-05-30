#!/usr/bin/env python3
"""Screen SUMO networks before current_to_dest/free-flow candidate experiments.

The full-cycle runner can bootstrap free-flow Dijkstra candidates before
collection. Some large networks have demand OD pairs that are present in
agents.csv but have no passenger-reachable path in the SUMO net, which later
fails inside RouteRL free-flow-time initialization. This script catches that
case without starting SUMO or writing route artifacts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import sumolib


@dataclass
class NetworkScreenResult:
    network: str
    network_dir: str
    agents: int
    demand_pairs: int
    origins: int
    destinations: int
    invalid_pairs: int
    missing_pairs: int
    passenger_blocked_pairs: int
    usable: bool
    missing_samples: list[tuple[int, int]]


def _network_name_from_net(net_xml: Path) -> str:
    name = net_xml.name
    if name.endswith(".net.xml"):
        return name[: -len(".net.xml")]
    return net_xml.stem


def _find_net_xml(network_dir: Path) -> Path:
    exact = network_dir / f"{network_dir.name}.net.xml"
    if exact.exists():
        return exact
    matches = sorted(network_dir.glob("*.net.xml"))
    if not matches:
        raise FileNotFoundError(f"No *.net.xml under {network_dir}")
    return matches[0]


def _find_od_file(network_dir: Path, network_name: str) -> Path:
    exact = network_dir / f"od_{network_name}.txt"
    if exact.exists():
        return exact
    matches = sorted(network_dir.glob("od_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No od_*.txt under {network_dir}")
    return matches[0]


def _load_demand_pairs(agents_csv: Path) -> tuple[int, set[tuple[int, int]]]:
    pairs: set[tuple[int, int]] = set()
    agents = 0
    with agents_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            agents += 1
            pairs.add((int(row["origin"]), int(row["destination"])))
    return agents, pairs


def _edge_allows_passenger(edge: object) -> bool:
    return any(lane.allows("passenger") for lane in edge.getLanes())


def _allowed_outgoing(edge: object) -> Iterable[object]:
    try:
        return edge.getAllowedOutgoing("passenger")
    except Exception:
        return edge.getOutgoing().keys()


def _reverse_passenger_graph(net: object) -> tuple[dict[str, list[str]], set[str]]:
    passenger_edges = {
        edge.getID()
        for edge in net.getEdges()
        if not edge.getID().startswith(":") and _edge_allows_passenger(edge)
    }
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in net.getEdges():
        src = edge.getID()
        if src.startswith(":") or src not in passenger_edges:
            continue
        for dst_edge in _allowed_outgoing(edge):
            dst = dst_edge.getID()
            if dst in passenger_edges:
                reverse[dst].append(src)
    return reverse, passenger_edges


def _reachable_to_destination(destination: str, reverse_graph: dict[str, list[str]]) -> set[str]:
    visited = {destination}
    queue: deque[str] = deque([destination])
    while queue:
        current = queue.popleft()
        for prev_edge in reverse_graph.get(current, ()):
            if prev_edge in visited:
                continue
            visited.add(prev_edge)
            queue.append(prev_edge)
    return visited


def screen_network(network_dir: Path) -> NetworkScreenResult:
    agents_csv = network_dir / "agents.csv"
    if not agents_csv.exists():
        raise FileNotFoundError(f"Missing agents.csv under {network_dir}")
    net_xml = _find_net_xml(network_dir)
    network_name = _network_name_from_net(net_xml)
    od_file = _find_od_file(network_dir, network_name)
    od_data = ast.literal_eval(od_file.read_text(encoding="utf-8"))
    origins = list(od_data["origins"])
    destinations = list(od_data["destinations"])
    agents, demand_pairs = _load_demand_pairs(agents_csv)

    invalid_pairs: set[tuple[int, int]] = set()
    concrete_pairs: list[tuple[int, int, str, str]] = []
    for origin_idx, dest_idx in demand_pairs:
        if (
            origin_idx < 0
            or dest_idx < 0
            or origin_idx >= len(origins)
            or dest_idx >= len(destinations)
        ):
            invalid_pairs.add((origin_idx, dest_idx))
            continue
        concrete_pairs.append((origin_idx, dest_idx, str(origins[origin_idx]), str(destinations[dest_idx])))

    net = sumolib.net.readNet(str(net_xml), withInternal=False)
    reverse_graph, passenger_edges = _reverse_passenger_graph(net)
    reachable_cache: dict[str, set[str]] = {}
    missing_pairs: list[tuple[int, int]] = []
    passenger_blocked_pairs = 0
    for origin_idx, dest_idx, origin_edge, dest_edge in concrete_pairs:
        if origin_edge not in passenger_edges or dest_edge not in passenger_edges:
            passenger_blocked_pairs += 1
            missing_pairs.append((origin_idx, dest_idx))
            continue
        reachable = reachable_cache.get(dest_edge)
        if reachable is None:
            reachable = _reachable_to_destination(dest_edge, reverse_graph)
            reachable_cache[dest_edge] = reachable
        if origin_edge not in reachable:
            missing_pairs.append((origin_idx, dest_idx))

    return NetworkScreenResult(
        network=network_dir.name,
        network_dir=str(network_dir),
        agents=agents,
        demand_pairs=len(demand_pairs),
        origins=len(origins),
        destinations=len(destinations),
        invalid_pairs=len(invalid_pairs),
        missing_pairs=len(missing_pairs),
        passenger_blocked_pairs=passenger_blocked_pairs,
        usable=(not invalid_pairs and not missing_pairs),
        missing_samples=missing_pairs[:10],
    )


def _network_dirs(args: argparse.Namespace) -> list[Path]:
    if args.network_dir:
        return [Path(item) for item in args.network_dir]
    root = Path(args.networks_root)
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "agents.csv").exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network_dir", nargs="*", help="Network directories to screen. Defaults to all networks/*.")
    parser.add_argument("--networks-root", default="networks")
    parser.add_argument("--target-agents", type=int, default=2000)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    results: list[NetworkScreenResult] = []
    failures: list[dict[str, str]] = []
    for network_dir in _network_dirs(args):
        try:
            results.append(screen_network(network_dir))
        except Exception as exc:
            failures.append({"network": str(network_dir), "error": str(exc)})

    results.sort(key=lambda item: (not item.usable, abs(item.agents - args.target_agents), item.network))
    header = (
        "network agents demand_pairs missing invalid passenger_blocked usable"
    )
    print(header)
    for item in results:
        print(
            f"{item.network:32s} {item.agents:6d} {item.demand_pairs:6d} "
            f"{item.missing_pairs:7d} {item.invalid_pairs:7d} "
            f"{item.passenger_blocked_pairs:17d} {str(item.usable):>6s}"
        )
        if item.missing_samples:
            print(f"  missing_samples={item.missing_samples}")
    for failure in failures:
        print(f"{failure['network']}: ERROR {failure['error']}")

    if args.json_output:
        payload = {
            "results": [asdict(item) for item in results],
            "failures": failures,
        }
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
