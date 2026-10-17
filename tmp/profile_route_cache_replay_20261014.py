"""Replay archived route queries with the frozen 16384-entry cache, without SUMO."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs/city_baselines_current_to_dest_1000_20261013_180607"
SOURCE = BATCH / "source"
sys.path.insert(0, str(SOURCE))

import numpy as np
from tools.compact_episode_archives import committed_episode
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.traffic.route_candidates import RouteCandidates


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


class ProfiledRoutes(RouteCandidates):
    def reset_metrics(self):
        self.counts = Counter()
        self.seconds = defaultdict(float)
        self.phase = "observation"

    def generate_paths(self, current_edge, destination_edge):
        hit = (current_edge, destination_edge) in self._route_candidate_cache
        self.counts[f"{self.phase}_{'hits' if hit else 'misses'}"] += 1
        if not hit and len(self._route_candidate_cache) >= self._route_candidate_cache_limit:
            self.counts["evictions"] += 1
        start = time.perf_counter()
        try:
            return super().generate_paths(current_edge, destination_edge)
        finally:
            self.seconds[f"{self.phase}_generate_paths"] += time.perf_counter() - start

    def _shortest_path(self, current, destination, costs):
        self.counts["dijkstra_calls"] += 1
        self.counts[f"{self.phase}_dijkstra_calls"] += 1
        wall = time.perf_counter()
        cpu = time.process_time()
        try:
            return super()._shortest_path(current, destination, costs)
        finally:
            elapsed = time.perf_counter() - wall
            self.seconds["dijkstra_inclusive_wall"] += elapsed
            self.seconds[f"{self.phase}_dijkstra_inclusive_wall"] += elapsed
            self.seconds["dijkstra_inclusive_cpu"] += time.process_time() - cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--reference-results", type=Path)
    args = parser.parse_args()
    label = args.city if args.capacity == 16384 else f"{args.city}_cache{args.capacity}"
    output = BATCH / "route_cache_replay_20261014" / label
    output.mkdir(parents=True, exist_ok=False)
    run_output = BATCH / args.city / "online_ac"
    boundary = committed_episode(dict(method="online_ac", output_dir=str(run_output)))
    paths = sorted(path for path in (run_output / "episodes/train").glob("episode_*.npz")
                   if int(path.stem.removeprefix("episode_")) <= boundary)[-3:]
    if args.reference_results:
        reference = json.loads(args.reference_results.read_text())
        if reference["city"] != args.city:
            raise ValueError("Reference result city mismatch")
        boundary = reference["checkpoint_boundary"]
        paths = [Path(path) for path in reference["archive_paths"]]
    if len(paths) != 3:
        raise RuntimeError("Need three checkpoint-committed training archives")
    manifest = json.loads((BATCH / "manifest.json").read_text())
    entry = next(row for row in manifest["runs"]
                 if row["city"] == args.city and row["method"] == "online_ac")
    command = entry["command"]
    config_path = Path(command[command.index("--config") + 1])
    config = load_config(config_path)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    candidates = ProfiledRoutes(topology.network, traffic_model="meso")
    frozen_default_capacity = candidates._route_candidate_cache_limit
    if args.capacity < 1:
        raise ValueError("Capacity must be positive")
    candidates._route_candidate_cache_limit = args.capacity
    results = dict(city=args.city, pid=os.getpid(), created_at=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), checkpoint_boundary=boundary,
        source=str(SOURCE), config=str(config_path), input_hashes=prepared.input_hashes,
        network_edges=topology.network.edge_count, cache_capacity=args.capacity,
        frozen_default_capacity=frozen_default_capacity,
        reference_results=str(args.reference_results) if args.reference_results else None,
        candidate_source_sha256=hashlib.sha256((SOURCE / "urban_dreamer/traffic/route_candidates.py").read_bytes()).hexdigest(),
        archive_paths=[str(path) for path in paths], episodes=[], status="running",
        protocol="Three chronological real training archives; cold empty cache first; same cache preserved across episodes; per window all sorted-vehicle observation queries then all sorted-vehicle validation queries.",
        limitations=["Only generate_paths and penalty Dijkstra are replayed; excludes SUMO, route observations/feature reductions, policy inference, graph legality, selected route application, archival writes and learner updates.",
            "Cache starts empty; does not reproduce the live trainer's earlier episodes/evaluation queries or actual live cache contents.",
            "Observation and validation query order follows archived policy decisions, not next-hop conditioning decisions; destination is selected route's last edge.",
            "Concurrent offline CPU workloads and instrumentation affect wall times; CPU times are also recorded."])
    write_json(output / "results.json", results)
    for index, path in enumerate(paths):
        decode_start = time.perf_counter()
        with zipfile.ZipFile(path) as archive:
            decisions = json.loads(np.load(archive.open("policy_decisions_json.npy"), allow_pickle=False).tobytes().decode("utf-8"))
            summary = json.loads(np.load(archive.open("summary_json.npy"), allow_pickle=False).tobytes().decode("utf-8"))
        metadata = summary["metadata"]
        if metadata["routing_mode"] != "current_to_dest" or metadata["input_hashes"] != prepared.input_hashes:
            raise ValueError("Archive routing mode or network input mismatch")
        windows = defaultdict(list)
        for decision in decisions:
            route = decision["route"]
            if not route or route[0] != decision["current_edge"]:
                raise ValueError("Invalid archived selected route")
            if decision.get("destination_edge", route[-1]) != route[-1]:
                raise ValueError("Archived route destination mismatch")
            windows[(float(decision["decision_time"]), int(decision["window_index"]))].append(decision)
        load_seconds = time.perf_counter() - decode_start
        candidates.reset_metrics()
        cache_start = len(candidates._route_candidate_cache)
        wall, cpu = time.perf_counter(), time.process_time()
        for window_number, (window_key, batch) in enumerate(sorted(windows.items()), start=1):
            ordered = sorted(batch, key=lambda row: row["vehicle_id"])
            if len({row["vehicle_id"] for row in ordered}) != len(ordered):
                raise ValueError("Duplicate policy decision in a window")
            for phase in ("observation", "validation"):
                candidates.phase = phase
                for row in ordered:
                    candidates.generate_paths(int(row["current_edge"]), int(row["route"][-1]))
            if window_number % 10 == 0:
                write_json(output / "progress.json", dict(pid=os.getpid(), episode=int(path.stem.removeprefix("episode_")),
                    completed_windows=window_number, total_windows=len(windows), cache_size=len(candidates._route_candidate_cache),
                    elapsed_wall_seconds=time.perf_counter()-wall, counts=dict(candidates.counts), seconds=dict(candidates.seconds)))
        elapsed_wall, elapsed_cpu = time.perf_counter()-wall, time.process_time()-cpu
        counts = dict(candidates.counts)
        queries = sum(counts.get(f"{phase}_{kind}", 0) for phase in ["observation", "validation"] for kind in ["hits", "misses"])
        hits = sum(counts.get(f"{phase}_hits", 0) for phase in ["observation", "validation"])
        row = dict(episode=int(path.stem.removeprefix("episode_")), archive=str(path),
            demand_sha256=metadata["demand_realization_sha256"], archive_decode_wall_seconds=load_seconds,
            policy_decisions=len(decisions), nonempty_policy_windows=len(windows),
            cache_size_start=cache_start, cache_size_end=len(candidates._route_candidate_cache),
            cache_state="cold" if index == 0 else "retained_from_preceding_episode",
            total_generate_paths_calls=queries, hit_rate=hits/queries if queries else None,
            replay_wall_seconds=elapsed_wall, replay_cpu_seconds=elapsed_cpu,
            counts=counts, seconds=dict(candidates.seconds))
        results["episodes"].append(row)
        write_json(output / "results.json", results)
        print(json.dumps(row), flush=True)
    results["status"] = "completed"
    results["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(output / "results.json", results)


if __name__ == "__main__":
    main()
