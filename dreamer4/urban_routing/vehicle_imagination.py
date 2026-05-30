"""Vehicle path construction and observation helpers for Dreamer training."""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .departure_schedule import estimate_vehicle_position
    from .flow_projector import PathTopology, split_od_agent_counts
except ImportError:
    from departure_schedule import estimate_vehicle_position
    from flow_projector import PathTopology, split_od_agent_counts


def build_lightweight_vehicle_path_topology(
    *,
    origins: List[str],
    destinations: List[str],
    n_edges: int,
    K: int,
    agents_csv: Optional[str],
    ratio_machines: float,
    edge_to_idx: Optional[Dict[str, int]] = None,
    routes_csv: Optional[str] = None,
) -> PathTopology:
    """Build the lightweight OD path topology used by vehicle-mode Dreamer."""
    n_orig = len(origins)
    n_dest = len(destinations)
    od_pairs: List[Tuple[int, int]] = []
    seen_pairs: set = set()

    def _add_pair(oi_raw: int, di_raw: int) -> None:
        if not (0 <= int(oi_raw) < n_orig and 0 <= int(di_raw) < n_dest):
            return
        key = (int(oi_raw), int(di_raw))
        if key in seen_pairs:
            return
        seen_pairs.add(key)
        od_pairs.append(key)

    if agents_csv and os.path.exists(agents_csv):
        with open(agents_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    _add_pair(int(row["origin"]), int(row["destination"]))
                except (KeyError, TypeError, ValueError):
                    continue

    if routes_csv and os.path.exists(routes_csv):
        with open(routes_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    _add_pair(
                        int(row.get("origins", row.get("origin", -1))),
                        int(row.get("destinations", row.get("destination", -1))),
                    )
                except (TypeError, ValueError):
                    continue

    if not od_pairs:
        od_pairs = [(oi, di) for oi in range(n_orig) for di in range(n_dest)]
        seen_pairs = set(od_pairs)

    od_index_map = {pair: idx for idx, pair in enumerate(od_pairs)}
    n_od = len(od_pairs)
    route_edge_indices = [[[] for _ in range(int(K))] for _ in range(n_od)]
    od_av_counts = np.zeros((n_od,), dtype=np.float32)
    od_hdv_counts = np.zeros((n_od,), dtype=np.float32)

    if routes_csv and edge_to_idx and os.path.exists(routes_csv):
        route_offsets: Dict[int, int] = {}
        with open(routes_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    oi_raw = int(row.get("origins", row.get("origin", -1)))
                    di_raw = int(row.get("destinations", row.get("destination", -1)))
                except (TypeError, ValueError):
                    continue
                od_idx = od_index_map.get((oi_raw, di_raw))
                if od_idx is None:
                    continue
                k = int(route_offsets.get(od_idx, 0))
                if k >= int(K):
                    continue
                idxs = [
                    int(edge_to_idx[e])
                    for e in str(row.get("path", "")).split()
                    if e in edge_to_idx
                ]
                if not idxs:
                    continue
                route_edge_indices[od_idx][k] = idxs
                route_offsets[od_idx] = k + 1

    if agents_csv and os.path.exists(agents_csv):
        with open(agents_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                oi_raw = int(row["origin"])
                di_raw = int(row["destination"])
                od_idx = od_index_map.get((oi_raw, di_raw))
                if od_idx is not None:
                    od_hdv_counts[od_idx] += 1.0

        if float(od_hdv_counts.sum()) > 0.0:
            od_av_counts, od_hdv_counts = split_od_agent_counts(
                od_hdv_counts,
                float(ratio_machines),
            )

    return PathTopology(
        origins=origins,
        destinations=destinations,
        K=int(K),
        route_edge_indices=route_edge_indices,
        route_masks=None,
        od_av_counts=od_av_counts,
        od_hdv_counts=od_hdv_counts,
        n_edges=int(n_edges),
        n_od=int(n_od),
        od_pairs=od_pairs,
        od_index_map=od_index_map,
    )


build_lightweight_vehicle_route_registry = build_lightweight_vehicle_path_topology


def build_diverse_paths_current_to_dest(
    adj_list: List[List[int]],
    edge_tt: np.ndarray,
    src_idx: int,
    dst_idx: int,
    K: int,
    penalty_factor: float,
) -> List[List[int]]:
    """Build K diverse shortest paths on the edge graph from current edge to destination edge."""
    src = int(src_idx)
    dst = int(dst_idx)
    K = int(max(1, K))
    if src == dst:
        return [[src] for _ in range(K)]

    weights = np.asarray(edge_tt, dtype=np.float64).copy()
    penalty = float(max(1.0, penalty_factor))
    paths: List[List[int]] = []
    seen: set = set()
    for _ in range(K):
        path = PathTopology._dijkstra(adj_list, weights, src, dst)
        if path is None or len(path) == 0:
            break
        key = tuple(int(x) for x in path)
        tries = 0
        while key in seen and tries < 3:
            for ei in path:
                weights[int(ei)] *= penalty
            path = PathTopology._dijkstra(adj_list, weights, src, dst)
            if path is None or len(path) == 0:
                break
            key = tuple(int(x) for x in path)
            tries += 1
        if path is None or len(path) == 0 or key in seen:
            break
        p_int = [int(x) for x in path]
        paths.append(p_int)
        seen.add(key)
        for ei in p_int:
            weights[int(ei)] *= penalty

    if not paths:
        paths = [[src]]
    while len(paths) < K:
        paths.append(list(paths[0]))
    return paths[:K]


# ══════════════════════════════════════════════════════════════════════════════
# Vehicle path cache for dynamic imagination
# ══════════════════════════════════════════════════════════════════════════════

class VehiclePathCache:
    """
    Pre-computes K diverse paths from every reachable edge to each unique
    destination, so that per-step imagination only needs a dict lookup
    instead of running Dijkstra per vehicle.
    """

    def __init__(
        self,
        adj_list: List[List[int]],
        edge_tt_ref: np.ndarray,
        dest_edges_unique: List[int],
        n_edges: int,
        K: int,
        penalty_factor: float = 3.0,
    ):
        self.K = int(K)
        self.n_edges = int(n_edges)
        self.dest_edges = list(dest_edges_unique)
        # cache: (src, dst) -> List[List[int]]  (K paths)
        self._cache: Dict[Tuple[int, int], List[List[int]]] = {}
        self._build(adj_list, edge_tt_ref, penalty_factor)

    def _build(
        self,
        adj_list: List[List[int]],
        edge_tt_ref: np.ndarray,
        penalty_factor: float,
    ) -> None:
        import heapq

        n = self.n_edges
        penalty = float(max(1.0, penalty_factor))

        # Build reverse adjacency list: rev_adj[v] = list of u where u→v
        rev_adj: List[List[int]] = [[] for _ in range(n)]
        for u, neighbors in enumerate(adj_list):
            for v in neighbors:
                rev_adj[v].append(u)

        def _reverse_dijkstra(
            weights: np.ndarray, dst: int,
        ) -> Tuple[np.ndarray, np.ndarray]:
            """Run Dijkstra on reversed graph from dst.
            Returns (dist, prev) arrays for ALL sources."""
            dist = np.full(n, np.inf)
            prev = np.full(n, -1, dtype=np.int64)
            dist[dst] = weights[dst]
            heap: List[tuple] = [(float(dist[dst]), dst)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v in rev_adj[u]:
                    nd = d + weights[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            return dist, prev

        def _extract_path(prev: np.ndarray, src: int, dst: int) -> Optional[List[int]]:
            """Extract path from src to dst using prev from reverse Dijkstra."""
            if prev[src] == -1 and src != dst:
                return None
            path = [src]
            u = src
            while u != dst:
                u = int(prev[u])
                if u == -1:
                    return None
                path.append(u)
            return path

        for dst in self.dest_edges:
            # Round 0: compute shortest path tree (unpenalized)
            base_weights = np.asarray(edge_tt_ref, dtype=np.float64).copy()
            dist0, prev0 = _reverse_dijkstra(base_weights, dst)

            # For each reachable source, collect K diverse paths via per-src penalty
            for src in range(n):
                if src == dst:
                    self._cache[(src, dst)] = [[src] for _ in range(self.K)]
                    continue
                if np.isinf(dist0[src]):
                    continue

                # First path from the shared shortest path tree
                path0 = _extract_path(prev0, src, dst)
                if path0 is None:
                    continue

                paths: List[List[int]] = [path0]
                seen: set = {tuple(path0)}

                # Subsequent paths: per-src penalty on individual Dijkstra
                if self.K > 1:
                    w = base_weights.copy()
                    for ei in path0:
                        w[ei] *= penalty
                    for _ in range(self.K - 1):
                        p = PathTopology._dijkstra(adj_list, w, src, dst)
                        if p is None:
                            break
                        key = tuple(p)
                        if key in seen:
                            break
                        paths.append([int(x) for x in p])
                        seen.add(key)
                        for ei in p:
                            w[int(ei)] *= penalty

                while len(paths) < self.K:
                    paths.append(list(paths[0]))
                self._cache[(src, dst)] = paths[:self.K]

    def lookup(self, src: int, dst: int) -> List[List[int]]:
        cached = self._cache.get((int(src), int(dst)))
        if cached is not None:
            return cached
        # Unreachable pair — return trivial
        return [[int(src)] for _ in range(self.K)]

    def lookup_batch(
        self,
        current_edges: np.ndarray,
        dest_edges: np.ndarray,
        done: np.ndarray,
    ) -> List[List[List[int]]]:
        """Vectorized lookup for all vehicles at once."""
        n = len(current_edges)
        batch: List[List[List[int]]] = []
        for i in range(n):
            if bool(done[i]):
                c = int(current_edges[i])
                batch.append([[c] for _ in range(self.K)])
            else:
                batch.append(self.lookup(int(current_edges[i]), int(dest_edges[i])))
        return batch


class LazyVehiclePathCache:
    """
    On-demand path cache for large networks where precomputing all (src, dst)
    pairs is too expensive. Paths are generated once per queried pair and then
    reused across later rollout steps and episodes.
    """

    def __init__(
        self,
        adj_list: List[List[int]],
        n_edges: int,
        K: int,
        penalty_factor: float = 3.0,
        max_entries: int = 250000,
    ):
        self.adj_list = adj_list
        self.n_edges = int(n_edges)
        self.K = int(K)
        self.penalty_factor = float(penalty_factor)
        self.max_entries = int(max_entries)
        self._cache: Dict[Tuple[int, int], List[List[int]]] = {}

    def _compute(self, src: int, dst: int, edge_tt: np.ndarray) -> List[List[int]]:
        return build_diverse_paths_current_to_dest(
            adj_list=self.adj_list,
            edge_tt=edge_tt,
            src_idx=int(src),
            dst_idx=int(dst),
            K=self.K,
            penalty_factor=self.penalty_factor,
        )

    def lookup(self, src: int, dst: int, edge_tt: np.ndarray) -> List[List[int]]:
        key = (int(src), int(dst))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        paths = self._compute(int(src), int(dst), edge_tt)
        if len(self._cache) < self.max_entries:
            self._cache[key] = paths
        return paths

    def lookup_batch(
        self,
        current_edges: np.ndarray,
        dest_edges: np.ndarray,
        done: np.ndarray,
        edge_tt: np.ndarray,
    ) -> List[List[List[int]]]:
        batch: List[List[List[int]]] = []
        n = len(current_edges)
        for i in range(n):
            cur = int(current_edges[i])
            if bool(done[i]):
                batch.append([[cur] for _ in range(self.K)])
            else:
                batch.append(self.lookup(cur, int(dest_edges[i]), edge_tt))
        return batch


def _od_suffix_path_batch(
    *,
    route_registry: PathTopology,
    active_avs: List[Dict[str, Any]],
    current_edges: np.ndarray,
    done: np.ndarray,
    K: int,
) -> List[List[List[int]]]:
    """Return OD-route suffix candidates matching collection-time od_suffix.

    Collection falls back to the vehicle's current SUMO route suffix whenever a
    precomputed OD candidate does not contain the current edge, then normalizes
    the bank back to K valid candidates.  In offline imagination, ``current_route``
    is the route proxy produced by the last selected action.
    """
    batch: List[List[List[int]]] = []
    n_od = int(route_registry.n_od)
    k_eff = int(K)
    for i, av in enumerate(active_avs):
        cur = int(current_edges[i])
        if bool(done[i]):
            batch.append([[cur] for _ in range(k_eff)])
            continue
        od_idx = int(av.get("od_idx", -1))
        if od_idx < 0 or od_idx >= n_od:
            batch.append([[cur] for _ in range(k_eff)])
            continue
        current_route = [int(e) for e in av.get("current_route", [])]
        fallback: List[int] = []
        if current_route:
            matches = [j for j, edge_idx in enumerate(current_route) if int(edge_idx) == cur]
            if matches:
                fallback = list(current_route[int(matches[-1]):])
        suffixes: List[List[int]] = []
        for route in route_registry.route_edge_indices[od_idx][:k_eff]:
            matches = [j for j, edge_idx in enumerate(route) if int(edge_idx) == cur]
            if not matches:
                suffixes.append(list(fallback) if fallback else [])
                continue
            suffix = list(route[int(matches[-1]):])
            if suffix and int(suffix[0]) == cur:
                suffixes.append(suffix)
            else:
                suffixes.append(list(fallback) if fallback else [])
        while len(suffixes) < k_eff:
            suffixes.append(list(fallback) if fallback else [])
        if not any(suffixes):
            batch.append([[cur] for _ in range(k_eff)])
            continue
        batch.append([list(route) for route in suffixes[:k_eff]])
    return batch


def _ensure_active_vehicle_current_edges(
    *,
    active_avs: List[Dict[str, Any]],
    current_step: int,
    avg_edge_time: float,
    step_interval: float,
    route_registry: Optional[PathTopology],
) -> None:
    """Fill current_edge for vehicles reconstructed mid-episode.

    ``rebuild_active_set`` can recreate vehicles that departed before the
    sampled imagination start step.  Those vehicles must not be treated as if
    they were still at the OD origin; otherwise current-edge route suffixes are
    built from the wrong action space.  Use route slot 0 only as a position
    prior for reconstructed vehicles.  Candidate order remains unchanged.
    """
    if not active_avs:
        return
    for av in active_avs:
        if "current_edge" in av:
            av["current_edge"] = int(av.get("current_edge", av.get("origin_edge", 0)))
            continue
        ref_route: List[int] = []
        if route_registry is not None:
            od_idx = int(av.get("od_idx", -1))
            if 0 <= od_idx < int(route_registry.n_od):
                od_routes = route_registry.route_edge_indices[od_idx]
                if od_routes:
                    ref_route = list(od_routes[0])
        if ref_route and "current_route" not in av:
            av["current_route"] = list(ref_route)
        av["current_edge"] = int(
            estimate_vehicle_position(
                av,
                int(current_step),
                float(avg_edge_time),
                float(step_interval),
                ref_route,
            )
        )


def _vehicle_obs_route_stats_mode(route_stats: str) -> str:
    mode = str(route_stats or "mean").strip().lower()
    aliases = {
        "": "mean",
        "avg": "mean",
        "average": "mean",
        "bottleneck_cost": "bottleneck",
        "route_cost": "bottleneck",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"mean", "bottleneck"}:
        raise ValueError(f"Unsupported vehicle_obs_route_stats={route_stats!r}; expected 'mean' or 'bottleneck'")
    return mode


def vehicle_route_feature_dim(d_obs: int, route_stats: str = "mean") -> int:
    """Feature width for one candidate route before global/context features."""
    d = int(d_obs)
    mode = _vehicle_obs_route_stats_mode(route_stats)
    if mode == "mean":
        return d
    # mean, max, q90, sum per dynamic feature + one normalized route-length feature.
    return 4 * d + 1


def _path_feature(
    edge_feats_dyn: np.ndarray,
    idxs: List[int],
    *,
    route_stats: str,
) -> np.ndarray:
    mode = _vehicle_obs_route_stats_mode(route_stats)
    d_dyn = int(edge_feats_dyn.shape[1])
    feat_dim = vehicle_route_feature_dim(d_dyn, mode)
    if not idxs:
        return np.zeros((feat_dim,), dtype=np.float32)

    idx_arr = np.asarray(idxs, dtype=np.int64)
    idx_arr = idx_arr[(idx_arr >= 0) & (idx_arr < int(edge_feats_dyn.shape[0]))]
    if idx_arr.size == 0:
        return np.zeros((feat_dim,), dtype=np.float32)

    vals = edge_feats_dyn[idx_arr].astype(np.float32, copy=False)
    if mode == "mean":
        return vals.mean(axis=0).astype(np.float32)

    length_norm = np.array(
        [min(float(np.log1p(idx_arr.size) / np.log1p(128.0)), 2.0)],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            vals.mean(axis=0),
            vals.max(axis=0),
            np.quantile(vals, 0.90, axis=0).astype(np.float32),
            vals.sum(axis=0),
            length_norm,
        ],
        axis=0,
    ).astype(np.float32)


def _global_route_feature(edge_feats_dyn: np.ndarray, *, route_stats: str) -> np.ndarray:
    mode = _vehicle_obs_route_stats_mode(route_stats)
    d_dyn = int(edge_feats_dyn.shape[1])
    feat_dim = vehicle_route_feature_dim(d_dyn, mode)
    if edge_feats_dyn.size == 0:
        return np.zeros((feat_dim,), dtype=np.float32)
    vals = edge_feats_dyn.astype(np.float32, copy=False)
    if mode == "mean":
        return vals.mean(axis=0).astype(np.float32)
    active = (np.abs(vals).sum(axis=1) > 1e-6).astype(np.float32)
    active_frac = np.array([float(active.mean())], dtype=np.float32)
    return np.concatenate(
        [
            vals.mean(axis=0),
            vals.max(axis=0),
            np.quantile(vals, 0.90, axis=0).astype(np.float32),
            vals.std(axis=0),
            active_frac,
        ],
        axis=0,
    ).astype(np.float32)


def build_vehicle_route_cost_obs_features(
    path_batch: List[List[List[int]]],
    edge_cost: Optional[np.ndarray],
    n_routes: int,
) -> np.ndarray:
    """Build non-leaky candidate route-cost features from current edge costs.

    Features are per candidate route: relative cost gap, log absolute cost, and
    valid-route mask.  Costs come from the current WM/SUMO state only; no
    future rollout labels or OD-specific best-route table are used.
    """
    n_agents = len(path_batch)
    k_eff = int(n_routes)
    out = np.zeros((n_agents, 3 * k_eff), dtype=np.float32)
    if n_agents <= 0 or k_eff <= 0 or edge_cost is None:
        return out

    edge_cost_arr = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
    n_edges = int(edge_cost_arr.shape[0])
    costs = np.full((n_agents, k_eff), np.nan, dtype=np.float32)
    valid = np.zeros((n_agents, k_eff), dtype=bool)

    for i, paths in enumerate(path_batch):
        for k in range(min(k_eff, len(paths))):
            idxs = [int(e) for e in paths[k] if 0 <= int(e) < n_edges]
            if not idxs:
                continue
            vals = edge_cost_arr[np.asarray(idxs, dtype=np.int64)]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            costs[i, k] = float(np.maximum(vals, 0.0).sum())
            valid[i, k] = True

    for i in range(n_agents):
        finite = np.isfinite(costs[i]) & valid[i]
        if not finite.any():
            continue
        row = costs[i].copy()
        finite_vals = row[finite]
        min_cost = float(finite_vals.min())
        scale = max(float(np.median(finite_vals)), 60.0)
        fallback = float(finite_vals.max() + scale)
        row[~finite] = fallback

        rel_gap = np.clip((row - min_cost) / scale, 0.0, 20.0)
        log_cost = np.log1p(np.maximum(row, 0.0)) / np.log1p(3600.0)
        valid_f = valid[i].astype(np.float32)
        out[i, 0:k_eff] = rel_gap.astype(np.float32)
        out[i, k_eff:2 * k_eff] = log_cost.astype(np.float32)
        out[i, 2 * k_eff:3 * k_eff] = valid_f

    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=0.0).astype(np.float32)


def build_vehicle_fleet_pressure_obs_features(
    path_batch: List[List[List[int]]],
    *,
    n_edges: int,
    n_routes: int,
    selected_route_counts: Optional[np.ndarray] = None,
    selected_edge_pressure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build same-tick fleet context features for marginal route decisions.

    Features are [route_share_0..K-1, planned_pressure_overlap_0..K-1].
    They summarize only routes selected earlier in the current decision tick,
    so they are non-leaky under a sequential actor evaluation order.
    """
    n_agents = len(path_batch)
    k_eff = int(n_routes)
    out = np.zeros((n_agents, 2 * k_eff), dtype=np.float32)
    if n_agents <= 0 or k_eff <= 0:
        return out

    counts = (
        np.asarray(selected_route_counts, dtype=np.float32).reshape(-1)
        if selected_route_counts is not None
        else np.zeros((k_eff,), dtype=np.float32)
    )
    if counts.shape[0] < k_eff:
        padded = np.zeros((k_eff,), dtype=np.float32)
        padded[: counts.shape[0]] = counts
        counts = padded
    else:
        counts = counts[:k_eff]
    denom = max(float(np.maximum(counts.sum(), 0.0)), 1.0)
    route_share = np.clip(counts / denom, 0.0, 1.0).astype(np.float32)

    pressure = (
        np.asarray(selected_edge_pressure, dtype=np.float32).reshape(-1)
        if selected_edge_pressure is not None
        else np.zeros((max(int(n_edges), 0),), dtype=np.float32)
    )
    if pressure.shape[0] < int(n_edges):
        padded = np.zeros((max(int(n_edges), 0),), dtype=np.float32)
        padded[: pressure.shape[0]] = pressure
        pressure = padded
    else:
        pressure = pressure[: max(int(n_edges), 0)]
    pressure_norm = max(denom, 1.0)

    for i, paths in enumerate(path_batch):
        out[i, :k_eff] = route_share
        for k in range(min(k_eff, len(paths))):
            idxs = sorted({int(e) for e in paths[k] if 0 <= int(e) < int(pressure.shape[0])})
            if idxs:
                out[i, k_eff + k] = float(pressure[np.asarray(idxs, dtype=np.int64)].sum()) / pressure_norm

    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=0.0).astype(np.float32)


def build_vehicle_obs_vectorized(
    edge_feats_dyn: np.ndarray,    # (E, n_dynamic)
    path_batch: List[List[List[int]]],  # [N_av][K][edge_indices]
    include_global: bool,
    route_stats: str = "mean",
    edge_cost: Optional[np.ndarray] = None,
    include_route_costs: bool = False,
    include_fleet_pressure: bool = False,
    fleet_route_counts: Optional[np.ndarray] = None,
    fleet_edge_pressure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorized version of build_vehicle_obs_from_path_batch."""
    n_agents = len(path_batch)
    if n_agents == 0:
        return np.zeros((0, 0), dtype=np.float32)
    K = len(path_batch[0])
    d_dyn = int(edge_feats_dyn.shape[1])
    route_feat_dim = vehicle_route_feature_dim(d_dyn, route_stats)
    route_cost_dim = 3 * K if bool(include_route_costs) else 0
    fleet_pressure_dim = 2 * K if bool(include_fleet_pressure) else 0
    obs_dim = K * route_feat_dim + (route_feat_dim if include_global else 0) + fleet_pressure_dim + route_cost_dim
    out = np.zeros((n_agents, obs_dim), dtype=np.float32)
    global_dyn = _global_route_feature(edge_feats_dyn, route_stats=route_stats)
    base_dim = K * route_feat_dim + (route_feat_dim if include_global else 0)

    for i in range(n_agents):
        off = 0
        for k in range(K):
            out[i, off: off + route_feat_dim] = _path_feature(
                edge_feats_dyn,
                path_batch[i][k],
                route_stats=route_stats,
            )
            off += route_feat_dim
        if include_global:
            out[i, off: off + route_feat_dim] = global_dyn
    if include_fleet_pressure:
        fleet_features = build_vehicle_fleet_pressure_obs_features(
            path_batch=path_batch,
            n_edges=int(edge_feats_dyn.shape[0]),
            n_routes=K,
            selected_route_counts=fleet_route_counts,
            selected_edge_pressure=fleet_edge_pressure,
        )
    if include_route_costs:
        cost_features = build_vehicle_route_cost_obs_features(
            path_batch=path_batch,
            edge_cost=edge_cost,
            n_routes=K,
        )
        out[:, base_dim:base_dim + route_cost_dim] = cost_features
    if include_fleet_pressure:
        fleet_start = base_dim + route_cost_dim
        out[:, fleet_start:fleet_start + fleet_pressure_dim] = fleet_features
    return out


def build_vehicle_obs_from_path_batch(
    edge_feats_dyn: np.ndarray,  # (E, n_dynamic)
    path_batch: List[List[List[int]]],  # [N_av][K][edge_idx]
    include_global: bool,
    route_stats: str = "mean",
    edge_cost: Optional[np.ndarray] = None,
    include_route_costs: bool = False,
    include_fleet_pressure: bool = False,
    fleet_route_counts: Optional[np.ndarray] = None,
    fleet_edge_pressure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build per-vehicle observations from candidate paths.
    """
    return build_vehicle_obs_vectorized(
        edge_feats_dyn=edge_feats_dyn,
        path_batch=path_batch,
        include_global=include_global,
        route_stats=route_stats,
        edge_cost=edge_cost,
        include_route_costs=include_route_costs,
        include_fleet_pressure=include_fleet_pressure,
        fleet_route_counts=fleet_route_counts,
        fleet_edge_pressure=fleet_edge_pressure,
    )


def aggregate_vehicle_actions_to_edge_distribution(
    av_vehicle_fracs: np.ndarray,   # (N_av, K)
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
    *,
    current_edges: Optional[np.ndarray] = None,
    edge_travel_times: Optional[np.ndarray] = None,
    horizon_seconds: Optional[float] = None,
) -> np.ndarray:
    """
    Aggregate per-vehicle path probabilities into planned edge-flow action.

    This mirrors collection-time ``av_planned_edge_flow``: count edges expected
    to be entered within the next decision interval, then divide by active AVs.
    """
    if av_vehicle_fracs.size == 0 or not path_batch or int(n_edges) <= 0:
        return np.zeros((int(n_edges),), dtype=np.float32)

    edge_chunks: List[np.ndarray] = []
    weight_chunks: List[np.ndarray] = []
    n_agents = min(int(av_vehicle_fracs.shape[0]), len(path_batch))
    current_arr = None if current_edges is None else np.asarray(current_edges, dtype=np.int64).reshape(-1)
    edge_tt = None
    use_horizon = edge_travel_times is not None and horizon_seconds is not None
    if use_horizon:
        edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
        if edge_tt.size < int(n_edges):
            use_horizon = False
    horizon = max(float(horizon_seconds or 0.0), 1.0)
    n_controlled = 0
    for i in range(n_agents):
        paths = path_batch[i]
        if not paths:
            continue
        k_eff = int(min(len(paths), int(n_routes), int(av_vehicle_fracs.shape[1])))
        if k_eff <= 0:
            continue
        n_controlled += 1
        probs = np.asarray(av_vehicle_fracs[i][:k_eff], dtype=np.float32)
        probs = np.clip(probs, 0.0, None)
        prob_sum = float(probs.sum())
        if prob_sum <= 1e-8:
            probs = np.ones((k_eff,), dtype=np.float32) / float(k_eff)
        else:
            probs = probs / prob_sum
        for k in range(k_eff):
            path = paths[k]
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < int(n_edges))]
            if idxs.size == 0:
                continue
            if use_horizon:
                start_pos = 0
                current_edge = None
                if current_arr is not None and i < current_arr.shape[0]:
                    current_edge = int(current_arr[i])
                    matches = np.flatnonzero(idxs == current_edge)
                    if matches.size > 0:
                        start_pos = int(matches[0])
                eta = 0.0
                keep: List[int] = []
                for pos in range(start_pos, int(idxs.size)):
                    edge_idx = int(idxs[pos])
                    if current_edge is not None and pos == start_pos and edge_idx == current_edge:
                        eta += float(edge_tt[edge_idx])
                        continue
                    if eta >= horizon:
                        break
                    keep.append(edge_idx)
                    eta += float(edge_tt[edge_idx])
                if not keep:
                    continue
                idxs = np.asarray(keep, dtype=np.int64)
            edge_chunks.append(idxs)
            weight_chunks.append(np.full((idxs.size,), float(probs[k]), dtype=np.float32))

    action_edge = np.zeros((int(n_edges),), dtype=np.float32)
    if edge_chunks:
        all_edges = np.concatenate(edge_chunks, axis=0)
        all_weights = np.concatenate(weight_chunks, axis=0)
        np.add.at(action_edge, all_edges, all_weights)

    if n_controlled > 0:
        action_edge /= float(n_controlled)
    return action_edge


def choose_vehicle_next_edge(current_edge: int, chosen_path: List[int]) -> int:
    """
    Convert chosen path into one-step next-hop edge index.
    """
    if not chosen_path:
        return int(current_edge)
    if int(chosen_path[0]) == int(current_edge):
        if len(chosen_path) >= 2:
            return int(chosen_path[1])
        return int(chosen_path[0])
    return int(chosen_path[0])


def advance_vehicle_along_path_time_budget(
    vehicle: Dict[str, Any],
    *,
    current_edge: int,
    chosen_path: List[int],
    edge_travel_times: np.ndarray,
    step_interval: float,
    n_edges: int,
    min_edge_time: float = 1e-3,
) -> None:
    """Advance one imagined vehicle along a selected suffix for one decision step.

    Offline MARL does not have SUMO's per-vehicle microscopic state.  This helper
    keeps a small deterministic proxy state by spending the current decision
    interval on the selected route suffix using WM-decoded edge travel times.
    The vehicle is marked arrived only after its time budget finishes the final
    edge in the suffix.
    """
    cur = int(current_edge)
    if int(n_edges) <= 0:
        vehicle["current_edge"] = cur
        vehicle["arrived"] = False
        return

    edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
    if edge_tt.size < int(n_edges):
        padded = np.full((int(n_edges),), float(min_edge_time), dtype=np.float32)
        padded[: edge_tt.size] = edge_tt
        edge_tt = padded
    edge_tt = np.nan_to_num(edge_tt[: int(n_edges)], nan=float(min_edge_time), posinf=float(min_edge_time))
    edge_tt = np.maximum(edge_tt, float(min_edge_time))

    path = [int(e) for e in chosen_path if 0 <= int(e) < int(n_edges)]
    if not path:
        vehicle["current_edge"] = cur
        vehicle["edge_remaining_time"] = float(edge_tt[cur]) if 0 <= cur < int(n_edges) else float(min_edge_time)
        vehicle["edge_remaining_edge"] = cur
        vehicle["arrived"] = False
        return

    matches = [idx for idx, edge_idx in enumerate(path) if int(edge_idx) == cur]
    if matches:
        pos = int(matches[-1])
    else:
        pos = 0
        cur = int(path[0])

    remaining_edge = int(vehicle.get("edge_remaining_edge", cur))
    if int(remaining_edge) == int(cur) and "edge_remaining_time" in vehicle:
        remaining = max(float(vehicle.get("edge_remaining_time", edge_tt[cur])), float(min_edge_time))
    else:
        remaining = float(edge_tt[cur]) if 0 <= cur < int(n_edges) else float(min_edge_time)

    budget = max(float(step_interval), 0.0)
    arrived = False
    while budget > 1e-6:
        if budget + 1e-6 < remaining:
            remaining -= budget
            budget = 0.0
            break

        budget -= remaining
        if pos >= len(path) - 1:
            remaining = 0.0
            arrived = True
            break

        pos += 1
        cur = int(path[pos])
        remaining = float(edge_tt[cur]) if 0 <= cur < int(n_edges) else float(min_edge_time)

    vehicle["current_edge"] = int(cur)
    vehicle["edge_remaining_time"] = float(max(remaining, 0.0))
    vehicle["edge_remaining_edge"] = int(cur)
    vehicle["arrived"] = bool(arrived)
