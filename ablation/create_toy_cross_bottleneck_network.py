#!/usr/bin/env python3
"""Generate a small cross-coupled SUMO routing network for online AC ablations."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
NETWORK_NAME = "toy_cross_bottleneck"
NETWORK_DIR = REPO_ROOT / "networks" / NETWORK_NAME


NODES = {
    "A0": (0.0, 120.0),
    "A1": (105.0, 120.0),
    "A2": (225.0, 120.0),
    "A3": (660.0, 120.0),
    "A4": (760.0, 120.0),
    "B0": (0.0, -120.0),
    "B1": (105.0, -120.0),
    "B2": (225.0, -120.0),
    "B3": (660.0, -120.0),
    "B4": (760.0, -120.0),
    "X0": (310.0, 45.0),
    "X1": (430.0, 45.0),
    "Y0": (310.0, -45.0),
    "Y1": (430.0, -45.0),
    "R0": (285.0, 0.0),
    "R1": (475.0, 0.0),
    "R2": (545.0, 0.0),
    "M0": (585.0, 0.0),
    "M1": (635.0, 0.0),
}


EDGES = [
    # OD-specific entry and queue-storage approaches. These edges are shared by
    # all four candidates for an OD, so mid-episode decisions can still choose
    # between competing suffixes before the first split.
    ("A_in", "A0", "A1", 80.0, 13.89, 1, "toy.fast"),
    ("A_hold", "A1", "A2", 120.0, 10.00, 1, "toy.approach"),
    ("B_in", "B0", "B1", 80.0, 13.89, 1, "toy.fast"),
    ("B_hold", "B1", "B2", 120.0, 10.00, 1, "toy.approach"),
    # First-stage route choices. X is A's nominal short bottleneck, while Y is
    # B's nominal short bottleneck; the cross choices deliberately invade the
    # other OD's bottleneck.
    ("A_pre_X", "A2", "X0", 80.0, 13.89, 1, "toy.fast"),
    ("A_pre_Y", "A2", "Y0", 95.0, 13.89, 1, "toy.fast"),
    ("A_to_R", "A2", "R0", 85.0, 11.11, 1, "toy.connector"),
    ("B_pre_Y", "B2", "Y0", 80.0, 13.89, 1, "toy.fast"),
    ("B_pre_X", "B2", "X0", 95.0, 13.89, 1, "toy.fast"),
    ("B_to_R", "B2", "R0", 85.0, 11.11, 1, "toy.connector"),
    # Coupled upstream bottlenecks. They are short enough to keep free-flow
    # routes competitive but slow and one-lane enough to create spillback.
    ("X", "X0", "X1", 150.0, 5.00, 1, "toy.bottleneck"),
    ("Y", "Y0", "Y1", 150.0, 5.00, 1, "toy.bottleneck"),
    # Direct post-bottleneck connectors can become locally bad. Route3 shares
    # the first bottleneck with route0, then can dodge this connector through
    # the relief corridor if the direct queue is worse.
    ("X_to_M", "X1", "M0", 100.0, 12.50, 1, "toy.connector"),
    ("Y_to_M", "Y1", "M0", 100.0, 12.50, 1, "toy.connector"),
    ("X_to_R", "X1", "R1", 50.0, 12.50, 1, "toy.connector"),
    ("Y_to_R", "Y1", "R1", 50.0, 12.50, 1, "toy.connector"),
    # Shared relief route and downstream merge. Route2 is no longer a private
    # bypass; both ODs and route3 can crowd this corridor before the final merge.
    ("R_shared", "R0", "R1", 210.0, 8.33, 1, "toy.relief"),
    ("R_tail", "R1", "R2", 65.0, 10.00, 1, "toy.relief"),
    ("R_to_M", "R2", "M0", 90.0, 11.11, 1, "toy.connector"),
    ("M", "M0", "M1", 100.0, 6.00, 1, "toy.merge"),
    ("M_to_A", "M1", "A3", 75.0, 13.89, 1, "toy.fast"),
    ("M_to_B", "M1", "B3", 75.0, 13.89, 1, "toy.fast"),
    ("A_out", "A3", "A4", 80.0, 13.89, 1, "toy.fast"),
    ("B_out", "B3", "B4", 80.0, 13.89, 1, "toy.fast"),
]


ROUTES = {
    (0, 0): [
        ["A_in", "A_hold", "A_pre_X", "X", "X_to_M", "M", "M_to_A", "A_out"],
        ["A_in", "A_hold", "A_pre_Y", "Y", "Y_to_M", "M", "M_to_A", "A_out"],
        ["A_in", "A_hold", "A_to_R", "R_shared", "R_tail", "R_to_M", "M", "M_to_A", "A_out"],
        ["A_in", "A_hold", "A_pre_X", "X", "X_to_R", "R_tail", "R_to_M", "M", "M_to_A", "A_out"],
    ],
    (1, 1): [
        ["B_in", "B_hold", "B_pre_Y", "Y", "Y_to_M", "M", "M_to_B", "B_out"],
        ["B_in", "B_hold", "B_pre_X", "X", "X_to_M", "M", "M_to_B", "B_out"],
        ["B_in", "B_hold", "B_to_R", "R_shared", "R_tail", "R_to_M", "M", "M_to_B", "B_out"],
        ["B_in", "B_hold", "B_pre_Y", "Y", "Y_to_R", "R_tail", "R_to_M", "M", "M_to_B", "B_out"],
    ],
}


EDGE_PARAMS = {edge_id: {"length": length, "speed": speed} for edge_id, _f, _t, length, speed, _lanes, _type in EDGES}

EDGE_SHAPES = {
    "X_to_R": "430.00,45.00 465.00,25.00 475.00,0.00",
    "Y_to_R": "430.00,-45.00 465.00,-25.00 475.00,0.00",
}


def indent(elem: ET.Element, level: int = 0) -> None:
    pad = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = pad
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = pad


def write_xml(path: Path, root: ET.Element) -> None:
    indent(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    for node_id, (x, y) in NODES.items():
        ET.SubElement(root, "node", id=node_id, x=f"{x:.2f}", y=f"{y:.2f}", type="priority")
    write_xml(path, root)


def write_types(path: Path) -> None:
    root = ET.Element("types")
    ET.SubElement(root, "type", id="toy.fast", priority="8", numLanes="1", speed="13.89")
    ET.SubElement(root, "type", id="toy.approach", priority="6", numLanes="1", speed="10.00")
    ET.SubElement(root, "type", id="toy.bottleneck", priority="4", numLanes="1", speed="5.56")
    ET.SubElement(root, "type", id="toy.connector", priority="5", numLanes="1", speed="11.11")
    ET.SubElement(root, "type", id="toy.relief", priority="6", numLanes="1", speed="8.33")
    ET.SubElement(root, "type", id="toy.merge", priority="3", numLanes="1", speed="6.00")
    write_xml(path, root)


def write_edges(path: Path) -> None:
    root = ET.Element("edges")
    for edge_id, from_node, to_node, length, speed, lanes, edge_type in EDGES:
        attrs = {
            "from": from_node,
            "to": to_node,
            "type": edge_type,
            "numLanes": str(lanes),
            "speed": f"{speed:.2f}",
            "length": f"{length:.2f}",
        }
        if edge_id in EDGE_SHAPES:
            attrs["shape"] = EDGE_SHAPES[edge_id]
        ET.SubElement(root, "edge", id=edge_id, **attrs)
    write_xml(path, root)


def write_connections(path: Path) -> None:
    root = ET.Element("connections")
    transitions = sorted({(route[i], route[i + 1]) for routes in ROUTES.values() for route in routes for i in range(len(route) - 1)})
    for from_edge, to_edge in transitions:
        ET.SubElement(root, "connection", **{"from": from_edge, "to": to_edge})
    write_xml(path, root)


def route_freeflow_seconds(route: Sequence[str]) -> float:
    total = 0.0
    for edge_id in route:
        params = EDGE_PARAMS[str(edge_id)]
        total += float(params["length"]) / max(float(params["speed"]), 0.1)
    return total


def write_routes_csv(path: Path) -> None:
    rows = []
    for (origin_idx, dest_idx), routes in sorted(ROUTES.items()):
        for route in routes:
            rows.append(
                {
                    "origins": int(origin_idx),
                    "destinations": int(dest_idx),
                    "path": " ".join(route),
                    "free_flow_time": route_freeflow_seconds(route) / 60.0,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["origins", "destinations", "path", "free_flow_time"])
        writer.writeheader()
        writer.writerows(rows)


def write_route_xml(path: Path) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="Human", color="red", guiShape="passenger/sedan")
    ET.SubElement(root, "vType", id="AV", color="yellow", guiShape="passenger/sedan")
    for (origin_idx, dest_idx), routes in sorted(ROUTES.items()):
        for route_idx, route in enumerate(routes):
            ET.SubElement(
                root,
                "route",
                id=f"{origin_idx}_{dest_idx}_{route_idx}",
                edges=" ".join(route),
            )
    write_xml(path, root)


def write_routes_meta(path: Path, *, n_agents: int) -> None:
    meta = {
        "design_version": "dynamic_game_v2",
        "generator": "manual_toy_cross_bottleneck",
        "network_name": NETWORK_NAME,
        "n_routes": 4,
        "n_origins": 2,
        "n_destinations": 2,
        "od_pairs": [[0, 0], [1, 1]],
        "n_agents": int(n_agents),
        "dynamic_game_mechanisms": [
            "OD-specific queue-storage approaches before the first split",
            "Cross-OD bottleneck invasion on route1",
            "Shared relief corridor on route2 instead of a private bypass",
            "Route3 shares the first bottleneck and then can dodge the direct post-bottleneck connector",
            "All candidates rejoin at a single downstream one-lane merge",
            "Three-wave nonstationary departure schedule with alternating A-heavy and B-heavy demand",
        ],
        "route_semantics": {
            "route0": "short own-OD bottleneck, direct downstream merge",
            "route1": "cross-OD bottleneck invasion, direct downstream merge",
            "route2": "shared relief corridor feeding the same downstream merge",
            "route3": "two-stage adaptive relief: own bottleneck then relief connector before the merge",
        },
    }
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _wave_origins(count: int, origin0_fraction: float) -> List[int]:
    count = int(count)
    if count <= 0:
        return []
    n_origin0 = int(round(float(count) * float(origin0_fraction)))
    n_origin0 = min(max(n_origin0, 0), count)
    origin_rows = []
    for origin, n_origin in ((0, n_origin0), (1, count - n_origin0)):
        for idx in range(n_origin):
            rank = (float(idx) + 0.5) / max(float(n_origin), 1.0)
            origin_rows.append((rank, origin))
    return [origin for _rank, origin in sorted(origin_rows)]


def write_agents(path: Path, *, n_agents: int, depart_span: int) -> None:
    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")
    total_agents = int(n_agents)
    rows = []
    wave_counts = [
        int(round(total_agents * 0.30)),
        int(round(total_agents * 0.35)),
    ]
    wave_counts.append(total_agents - sum(wave_counts))
    wave_specs = [
        (0.00, 0.30, wave_counts[0], 0.50),
        (0.30, 0.65, wave_counts[1], 0.72),
        (0.65, 1.00, wave_counts[2], 0.28),
    ]
    for start_frac, end_frac, count, origin0_fraction in wave_specs:
        wave_start = float(depart_span) * float(start_frac)
        wave_end = float(depart_span) * float(end_frac)
        origins = _wave_origins(int(count), float(origin0_fraction))
        for idx, group in enumerate(origins):
            denom = max(len(origins) - 1, 1)
            start = int(round(wave_start + (wave_end - wave_start) * float(idx) / float(denom)))
            rows.append(
                {
                    "origin": int(group),
                    "destination": int(group),
                    "start_time": int(start),
                    "kind": "Human",
                }
            )
    rows = sorted(rows, key=lambda row: (int(row["start_time"]), int(row["origin"])))
    for agent_id, row in enumerate(rows):
        row["id"] = int(agent_id)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "origin", "destination", "start_time", "kind"])
        writer.writeheader()
        writer.writerows(rows)


def write_od(path: Path) -> None:
    content = {
        "origins": ["A_in", "B_in"],
        "destinations": ["A_out", "B_out"],
    }
    path.write_text("{\n\"origins\" : %r,\n\"destinations\" : %r,\n}\n" % (content["origins"], content["destinations"]), encoding="utf-8")


def write_netccfg(path: Path) -> None:
    root = ET.Element("configuration")
    input_elem = ET.SubElement(root, "input")
    ET.SubElement(input_elem, "node-files", value=f"{NETWORK_NAME}.nod.xml")
    ET.SubElement(input_elem, "edge-files", value=f"{NETWORK_NAME}.edg.xml")
    ET.SubElement(input_elem, "connection-files", value=f"{NETWORK_NAME}.con.xml")
    ET.SubElement(input_elem, "type-files", value=f"{NETWORK_NAME}.typ.xml")
    output_elem = ET.SubElement(root, "output")
    ET.SubElement(output_elem, "output-file", value=f"{NETWORK_NAME}.net.xml")
    processing = ET.SubElement(root, "processing")
    ET.SubElement(processing, "no-turnarounds", value="true")
    ET.SubElement(processing, "offset.disable-normalization", value="true")
    write_xml(path, root)


def find_netconvert(explicit: str = "") -> str:
    candidates = [explicit] if explicit else []
    candidates.extend(
        [
            str(Path(sys.executable).with_name("netconvert")),
            "/home/wangwenrui/opt/miniconda3/envs/URB/bin/netconvert",
            "netconvert",
        ]
    )
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        if Path(resolved).exists() or shutil.which(resolved):
            return resolved
    return "netconvert"


def run_netconvert(network_dir: Path, *, netconvert: str) -> None:
    cmd = [
        netconvert,
        "-c",
        f"{NETWORK_NAME}.netccfg",
    ]
    subprocess.run(cmd, cwd=str(network_dir), check=True)


def build_edge_registry(network_dir: Path) -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from dreamer4.urban_routing.flow_projector import EdgeRegistry

    registry = EdgeRegistry.build(str(network_dir / f"{NETWORK_NAME}.net.xml"), network=NETWORK_NAME)
    registry.save(str(network_dir / "edge_registry.json"))


def generate(*, output_dir: Path, n_agents: int, depart_span: int, netconvert: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_nodes(output_dir / f"{NETWORK_NAME}.nod.xml")
    write_types(output_dir / f"{NETWORK_NAME}.typ.xml")
    write_edges(output_dir / f"{NETWORK_NAME}.edg.xml")
    write_connections(output_dir / f"{NETWORK_NAME}.con.xml")
    write_netccfg(output_dir / f"{NETWORK_NAME}.netccfg")
    write_od(output_dir / f"od_{NETWORK_NAME}.txt")
    write_agents(output_dir / "agents.csv", n_agents=int(n_agents), depart_span=int(depart_span))
    write_routes_csv(output_dir / "routes.csv")
    write_route_xml(output_dir / "route.rou.xml")
    write_routes_meta(output_dir / "routes_meta.json", n_agents=int(n_agents))
    run_netconvert(output_dir, netconvert=find_netconvert(netconvert))
    build_edge_registry(output_dir)
    return output_dir


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(NETWORK_DIR))
    parser.add_argument("--n-agents", type=int, default=320)
    parser.add_argument("--depart-span", type=int, default=420)
    parser.add_argument("--netconvert", default="")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out = generate(
        output_dir=Path(args.output_dir).resolve(),
        n_agents=int(args.n_agents),
        depart_span=int(args.depart_span),
        netconvert=str(args.netconvert or ""),
    )
    print(f"[toy-network] generated {NETWORK_NAME} under {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
