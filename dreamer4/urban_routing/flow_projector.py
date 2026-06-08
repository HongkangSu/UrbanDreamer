# dreamer4/urban_routing/flow_projector.py
"""
EdgeRegistry: fixed ordered list of all non-internal SUMO edges + static features + adjacency.
FlowProjector: GNN-based projector Linear(d_feat→d_spatial) with EdgeGNNLayers + decoder heads.

Both are trained jointly with the CFM dynamics model (no separate pretraining).
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import sumolib
    SUMOLIB_AVAILABLE = True
except ImportError:
    SUMOLIB_AVAILABLE = False

# Feature indices (dynamic)
IDX_OCC   = 0   # occupancy [0,1]
IDX_SPEED = 1   # mean_speed / speed_limit
IDX_VEH   = 2   # vehicle_number / capacity
IDX_HALT  = 3   # halting_number / capacity
N_DYNAMIC = 4

# Feature indices (static, appended after dynamic)
IDX_LEN   = 4   # length / max_length
IDX_SLIM  = 5   # speed_limit / max_speed_limit
IDX_LANES = 6   # num_lanes / max_lanes
IDX_TYPE  = 7   # road_type_id mapped to [0,1]
N_STATIC  = 4
D_FEAT    = N_DYNAMIC + N_STATIC   # 8

# Default free-flow speed (m/s) used when speed_limit is unknown
_DEFAULT_SPEED = 13.89   # ~50 km/h

_DIJKSTRA_EDGE_LENGTHS_BY_N: Dict[int, np.ndarray] = {}


def _register_dijkstra_edge_lengths(lengths: Optional[np.ndarray]) -> None:
    if lengths is None:
        return
    arr = np.asarray(lengths, dtype=np.float32).reshape(-1)
    if arr.size > 0:
        _DIJKSTRA_EDGE_LENGTHS_BY_N[int(arr.shape[0])] = arr.copy()


def _dijkstra_hard_min_edge_length_m() -> float:
    raw_value = os.environ.get("URB_DIJKSTRA_HARD_MIN_EDGE_LENGTH_M", "1.0")
    try:
        return max(float(raw_value), 0.0)
    except (TypeError, ValueError):
        return 1.0


def decode_nonnegative_volume(raw_volume: torch.Tensor) -> torch.Tensor:
    """
    Map the decoder's unconstrained volume logit to a physical non-negative volume.

    A hard clamp would erase gradients for negative logits, which lets the
    decoder get stuck predicting negative raw volume that becomes all-zero after
    downstream post-processing. Softplus keeps the physical output non-negative
    while still allowing the volume head to recover from negative logits.
    """
    return F.softplus(raw_volume, beta=10.0)


def split_od_agent_counts(
    total_counts: np.ndarray,
    ratio_machines: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split per-OD total counts into AV / HDV counts while preserving the exact AV total.

    Using plain rounding on sparse OD bins can lose many AVs. This helper applies a
    largest-remainder allocation so that:
      sum(od_av_counts) == int(sum(total_counts) * ratio_machines)
    """
    total = np.asarray(total_counts, dtype=np.float64).reshape(-1)
    total_agents = int(total.sum())
    n_machines = int(total_agents * float(ratio_machines))
    if total_agents <= 0 or n_machines <= 0:
        return np.zeros_like(total, dtype=np.float32), total.astype(np.float32)

    raw = total * (float(n_machines) / float(total_agents))
    av_counts = np.floor(raw).astype(np.int64)
    remainder = int(n_machines - int(av_counts.sum()))
    if remainder > 0:
        frac = raw - av_counts.astype(np.float64)
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            if av_counts[idx] < int(round(total[idx])):
                av_counts[idx] += 1

    av_counts = np.minimum(av_counts.astype(np.float64), total)
    hdv_counts = np.maximum(total - av_counts, 0.0)
    return av_counts.astype(np.float32), hdv_counts.astype(np.float32)


class EdgeRegistry:
    """
    Manages the fixed ordered edge list and static feature normalization.

    Usage:
        # Build once from net.xml and save
        reg = EdgeRegistry.build(net_xml_path)
        reg.save("edge_registry.json")

        # Load at training / collection time
        reg = EdgeRegistry.load("edge_registry.json")
        flow_matrix = reg.extract_flow(state["edges"])  # (E, 8) float32
    """

    def __init__(
        self,
        edge_ids: List[str],
        static_features: np.ndarray,  # (E, 4) float32 normalized
        speed_limits: np.ndarray,      # (E,) float32  m/s
        capacities: np.ndarray,        # (E,) float32  veh capacity estimate
        network: str = "",
        adjacency: Optional[np.ndarray] = None,
        lengths: Optional[np.ndarray] = None,  # (E,) float32 raw meters
        from_node_idx: Optional[np.ndarray] = None,  # (E,) int — PCFM Kirchhoff
        to_node_idx: Optional[np.ndarray] = None,    # (E,) int — PCFM Kirchhoff
        passenger_allowed: Optional[np.ndarray] = None,  # (E,) bool — ordinary car routing mask
    ):
        self.edge_ids = edge_ids
        self.static_features = static_features.astype(np.float32)
        self.speed_limits = speed_limits.astype(np.float32)
        self.capacities = capacities.astype(np.float32)
        self.network = network
        self._edge_to_idx: Dict[str, int] = {e: i for i, e in enumerate(edge_ids)}
        self.n_edges = len(edge_ids)
        self.adjacency = adjacency  # (E, E) float32 row-normalized, or None
        self.lengths = lengths.astype(np.float32) if lengths is not None else None
        _register_dijkstra_edge_lengths(self.lengths)
        self.from_node_idx = from_node_idx.astype(np.int64) if from_node_idx is not None else None
        self.to_node_idx = to_node_idx.astype(np.int64) if to_node_idx is not None else None
        self.passenger_allowed = (
            passenger_allowed.astype(bool)
            if passenger_allowed is not None
            else np.ones((len(edge_ids),), dtype=bool)
        )
        self._has_passenger_allowed = passenger_allowed is not None

    @classmethod
    def build(cls, net_xml_path: str, network: str = "") -> "EdgeRegistry":
        """Parse net.xml via sumolib and build the registry."""
        if not SUMOLIB_AVAILABLE:
            raise ImportError(
                "sumolib is required to build EdgeRegistry. "
                "Install via: pip install sumolib"
            )
        net = sumolib.net.readNet(net_xml_path, withInternal=False)

        # All non-internal edges, sorted by ID for determinism
        raw_edges = sorted(
            [e for e in net.getEdges() if not e.getID().startswith(":")],
            key=lambda e: e.getID(),
        )

        edge_ids   = [e.getID()        for e in raw_edges]
        lengths    = np.array([e.getLength()     for e in raw_edges], dtype=np.float32)
        speed_lims = np.array([e.getSpeed()      for e in raw_edges], dtype=np.float32)
        num_lanes  = np.array([e.getLaneNumber() for e in raw_edges], dtype=np.float32)
        passenger_allowed = np.array(
            [
                any(lane.allows("passenger") for lane in e.getLanes())
                for e in raw_edges
            ],
            dtype=bool,
        )

        # Road type → normalized integer
        type_strs    = [e.getType() for e in raw_edges]
        unique_types = sorted(set(type_strs))
        type_map     = {t: i for i, t in enumerate(unique_types)}
        denom        = max(1, len(unique_types) - 1)
        type_ids     = np.array(
            [type_map[t] / denom for t in type_strs],
            dtype=np.float32,
        )

        # Normalize static features
        max_len   = max(float(lengths.max()),   1.0)
        max_slim  = max(float(speed_lims.max()), 1.0)
        max_lanes = max(float(num_lanes.max()),  1.0)

        static = np.stack([
            lengths     / max_len,
            speed_lims  / max_slim,
            num_lanes   / max_lanes,
            type_ids,
        ], axis=1).astype(np.float32)   # (E, 4)

        # Rough capacity estimate: lanes * 10 vehicles
        capacities = np.clip(num_lanes * 10.0, 1.0, None).astype(np.float32)

        # Node indices for Kirchhoff constraint (PCFM)
        all_nodes = set()
        for e in raw_edges:
            all_nodes.add(e.getFromNode().getID())
            all_nodes.add(e.getToNode().getID())
        node_to_idx = {n: i for i, n in enumerate(sorted(all_nodes))}
        from_node_idx = np.array(
            [node_to_idx[e.getFromNode().getID()] for e in raw_edges], dtype=np.int64
        )
        to_node_idx = np.array(
            [node_to_idx[e.getToNode().getID()] for e in raw_edges], dtype=np.int64
        )

        print(f"[EdgeRegistry] Built: {len(edge_ids)} edges, {len(node_to_idx)} nodes, network='{network}'")

        edge_to_idx = {e.getID(): i for i, e in enumerate(raw_edges)}
        adjacency = cls._build_adjacency(raw_edges, edge_to_idx)

        return cls(edge_ids, static, speed_lims, capacities, network=network,
                   adjacency=adjacency, lengths=lengths,
                   from_node_idx=from_node_idx, to_node_idx=to_node_idx,
                   passenger_allowed=passenger_allowed)

    @staticmethod
    def _build_adjacency(raw_edges, edge_to_idx):
        """Line graph adjacency: edge A → edge B if SUMO has a connection from A to B."""
        E = len(raw_edges)
        adj = np.zeros((E, E), dtype=np.float32)
        for e in raw_edges:
            src_idx = edge_to_idx[e.getID()]
            # e.getOutgoing() returns {successor_edge: [Connection, ...]}
            # Only includes edges with actual SUMO connections (respects turn restrictions)
            for succ in e.getOutgoing():
                succ_id = succ.getID()
                if succ_id in edge_to_idx:
                    adj[src_idx, edge_to_idx[succ_id]] = 1.0
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj = adj / row_sum
        return adj

    def extract_flow(self, edges_state: Dict[str, dict]) -> np.ndarray:
        """
        Convert SUMO step state["edges"] to a (E, 8) float32 flow matrix.

        Dynamic features (0-3): occupancy, norm_speed, veh_norm, halt_norm.
        Static features (4-7):  pre-normalized at build time (length, speed_limit,
                                 num_lanes, road_type).
        Missing edges default to free-flow: [0, 1, 0, 0, ...static...].
        """
        out = np.empty((self.n_edges, D_FEAT), dtype=np.float32)
        for i, eid in enumerate(self.edge_ids):
            ed  = edges_state.get(eid)
            sl = float(self.speed_limits[i])
            if sl <= 0.0:
                sl = _DEFAULT_SPEED
            cap = float(self.capacities[i])
            if ed is None:
                out[i, IDX_OCC]   = 0.0
                out[i, IDX_SPEED] = 1.0   # free-flow
                out[i, IDX_VEH]   = 0.0
                out[i, IDX_HALT]  = 0.0
            else:
                occ = float(ed.get("occupancy", 0.0))
                out[i, IDX_OCC]   = min(occ, 1.0)   # clamp; traci may return [0,100] or [0,1]
                out[i, IDX_SPEED] = float(ed.get("mean_speed",      sl)) / sl
                out[i, IDX_VEH]   = float(ed.get("vehicle_number",  0))  / cap
                out[i, IDX_HALT]  = float(ed.get("halting_number",  0))  / cap
            out[i, IDX_LEN : IDX_LEN + N_STATIC] = self.static_features[i]   # copy 4 static values
        return out   # (E, 8)

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "network":         self.network,
            "n_edges":         self.n_edges,
            "edge_ids":        self.edge_ids,
            "static_features": self.static_features.tolist(),
            "speed_limits":    self.speed_limits.tolist(),
            "capacities":      self.capacities.tolist(),
        }
        if self.adjacency is not None:
            data["adjacency"] = self.adjacency.tolist()
        if self.lengths is not None:
            data["lengths"] = self.lengths.tolist()
        if self.from_node_idx is not None:
            data["from_node_idx"] = self.from_node_idx.tolist()
        if self.to_node_idx is not None:
            data["to_node_idx"] = self.to_node_idx.tolist()
        if self.passenger_allowed is not None:
            data["passenger_allowed"] = self.passenger_allowed.astype(bool).tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[EdgeRegistry] Saved → {path}  ({self.n_edges} edges)")

    def build_node_incidence(self) -> Optional[np.ndarray]:
        """Build signed node-edge incidence matrix M for internal nodes.

        M[node, edge] = +1 if edge enters node, -1 if edge leaves node.
        Only internal nodes (nodes that are both a from-node and a to-node) are
        included, so boundary (source/sink) nodes are excluded.

        Returns:
            (N_internal, E) float32 array, or None if node indices are missing.
        """
        if self.from_node_idx is None or self.to_node_idx is None:
            return None
        all_from = set(self.from_node_idx.tolist())
        all_to = set(self.to_node_idx.tolist())
        internal = sorted(all_from & all_to)
        if not internal:
            return None
        node_remap = {n: i for i, n in enumerate(internal)}
        M = np.zeros((len(internal), self.n_edges), dtype=np.float32)
        for e in range(self.n_edges):
            fn = int(self.from_node_idx[e])
            tn = int(self.to_node_idx[e])
            if fn in node_remap:
                M[node_remap[fn], e] -= 1.0  # edge leaves from_node
            if tn in node_remap:
                M[node_remap[tn], e] += 1.0  # edge enters to_node
        return M

    @classmethod
    def load(cls, path: str) -> "EdgeRegistry":
        with open(path, "r") as f:
            data = json.load(f)
        adjacency = None
        if "adjacency" in data:
            adjacency = np.array(data["adjacency"], dtype=np.float32)
        lengths = None
        if "lengths" in data:
            lengths = np.array(data["lengths"], dtype=np.float32)
        from_node_idx = None
        if "from_node_idx" in data:
            from_node_idx = np.array(data["from_node_idx"], dtype=np.int64)
        to_node_idx = None
        if "to_node_idx" in data:
            to_node_idx = np.array(data["to_node_idx"], dtype=np.int64)
        passenger_allowed = None
        if "passenger_allowed" in data:
            passenger_allowed = np.array(data["passenger_allowed"], dtype=bool)
        return cls(
            edge_ids        = data["edge_ids"],
            static_features = np.array(data["static_features"], dtype=np.float32),
            speed_limits    = np.array(data["speed_limits"],    dtype=np.float32),
            capacities      = np.array(data["capacities"],      dtype=np.float32),
            network         = data.get("network", ""),
            adjacency       = adjacency,
            lengths         = lengths,
            from_node_idx   = from_node_idx,
            to_node_idx     = to_node_idx,
            passenger_allowed = passenger_allowed,
        )


class RouteRegistry:
    """Pre-computed K routes per OD pair with edge-index mappings.

    Attributes:
        origins, destinations: List[str] edge IDs
        n_od: int  (N_origins * N_destinations for dense registries, or the
              number of active demand pairs for lightweight vehicle topologies)
        K: int     (routes per OD pair)
        route_edge_indices: List[List[List[int]]]  [od_idx][k] = [edge indices]
        route_masks: optional np.ndarray (N_od, K, E)  1/route_len for edges on route
        od_av_counts, od_hdv_counts: np.ndarray (N_od,)
    """

    def __init__(
        self,
        origins: List[str],
        destinations: List[str],
        K: int,
        route_edge_indices: List[List[List[int]]],
        route_masks: Optional[np.ndarray],
        od_av_counts: np.ndarray,
        od_hdv_counts: np.ndarray,
        n_edges: int,
        route_indicators: Optional[np.ndarray] = None,  # (N_od, K, E) binary
        n_od: Optional[int] = None,
        od_pairs: Optional[List[Tuple[int, int]]] = None,
        od_index_map: Optional[Dict[Tuple[int, int], int]] = None,
    ):
        self.origins = origins
        self.destinations = destinations
        self.n_od = int(n_od) if n_od is not None else len(origins) * len(destinations)
        self.K = K
        self.route_edge_indices = route_edge_indices
        self.od_pairs = (
            [(int(o), int(d)) for o, d in od_pairs]
            if od_pairs is not None
            else None
        )
        if od_index_map is not None:
            self.od_index_map = {
                (int(k[0]), int(k[1])): int(v)
                for k, v in od_index_map.items()
            }
        elif self.od_pairs is not None:
            self.od_index_map = {
                (int(o), int(d)): int(i)
                for i, (o, d) in enumerate(self.od_pairs)
            }
        else:
            self.od_index_map = None
        self.route_masks = (
            route_masks.astype(np.float32) if route_masks is not None else None
        )  # (N_od, K, E), optional for lightweight vehicle-mode topologies
        self.od_av_counts = od_av_counts.astype(np.float32)   # (N_od,)
        self.od_hdv_counts = od_hdv_counts.astype(np.float32) # (N_od,)
        self.n_edges = n_edges
        self._route_masks_tensor: Optional[torch.Tensor] = None
        # Binary route indicators for Stackelberg layer
        if route_indicators is not None:
            self.route_indicators = route_indicators.astype(np.float32)
        elif route_masks is None:
            self.route_indicators = None
        else:
            self.route_indicators = self._build_route_indicators()
        self._route_indicators_tensor: Optional[torch.Tensor] = None

    @staticmethod
    def _build_adj_list(adjacency: np.ndarray) -> List[List[int]]:
        """Build adjacency list from (E, E) adjacency matrix."""
        E = adjacency.shape[0]
        adj: List[List[int]] = [[] for _ in range(E)]
        rows, cols = np.nonzero(adjacency)
        for r, c in zip(rows.tolist(), cols.tolist()):
            adj[r].append(c)
        return adj

    @staticmethod
    def _build_vehicle_adj_list(
        adjacency: np.ndarray,
        allowed_edges: Optional[np.ndarray],
    ) -> List[List[int]]:
        """Build an edge adjacency list constrained to ordinary passenger edges."""
        base = RouteRegistry._build_adj_list(adjacency)
        if allowed_edges is None:
            return base
        allowed = np.asarray(allowed_edges, dtype=bool).reshape(-1)
        if allowed.shape[0] != len(base):
            return base
        out: List[List[int]] = [[] for _ in range(len(base))]
        for src, neighbors in enumerate(base):
            if not bool(allowed[src]):
                continue
            out[src] = [int(dst) for dst in neighbors if bool(allowed[int(dst)])]
        return out

    @staticmethod
    def _dijkstra(adj_list: List[List[int]], weights: np.ndarray,
                  src: int, dst: int) -> Optional[List[int]]:
        """Shortest path on edge graph via Dijkstra. Returns edge index list or None."""
        import heapq
        n = len(adj_list)
        weights_arr = np.asarray(weights)
        min_len = _dijkstra_hard_min_edge_length_m()
        edge_lengths = _DIJKSTRA_EDGE_LENGTHS_BY_N.get(n)

        def _run(blocked: Optional[np.ndarray]) -> Optional[List[int]]:
            dist = np.full(n, np.inf)
            prev = np.full(n, -1, dtype=np.int64)
            dist[src] = weights_arr[src]
            heap: List[tuple] = [(float(dist[src]), src)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                if u == dst:
                    break
                for v in adj_list[u]:
                    vi = int(v)
                    if blocked is not None and bool(blocked[vi]):
                        continue
                    nd = d + weights_arr[vi]
                    if nd < dist[vi]:
                        dist[vi] = nd
                        prev[vi] = u
                        heapq.heappush(heap, (nd, vi))
            if np.isinf(dist[dst]):
                return None
            path = []
            u = dst
            while u != -1:
                path.append(u)
                u = int(prev[u])
            return path[::-1]

        blocked_edges = None
        if min_len > 0.0 and edge_lengths is not None and edge_lengths.shape[0] >= n:
            blocked_edges = np.asarray(edge_lengths[:n] < min_len, dtype=bool)
            blocked_edges[int(src)] = False
            blocked_edges[int(dst)] = False

        path = _run(blocked_edges)
        if path is None and blocked_edges is not None:
            path = _run(None)
        if path is None:
            return None
        return path

    @classmethod
    def build(
        cls,
        net_xml_path: str,
        edge_registry: "EdgeRegistry",
        origins: List[str],
        destinations: List[str],
        K: int = 4,
        agents_csv: Optional[str] = None,
        ratio_machines: float = 0.2,
    ) -> "RouteRegistry":
        """Build RouteRegistry from net.xml using K diverse shortest paths per OD pair.

        Uses penalty-based Dijkstra: after finding each route, edges on that route
        get their weight multiplied by a penalty factor, forcing subsequent routes
        to use different edges.
        """
        if not SUMOLIB_AVAILABLE:
            raise ImportError("sumolib is required to build RouteRegistry")

        net = sumolib.net.readNet(net_xml_path, withInternal=False)
        E = edge_registry.n_edges
        N_orig = len(origins)
        N_dest = len(destinations)
        N_od = N_orig * N_dest

        # Build adjacency list from EdgeRegistry
        if edge_registry.adjacency is None:
            raise ValueError("EdgeRegistry must have adjacency for route building")
        adj_list = cls._build_vehicle_adj_list(
            edge_registry.adjacency,
            getattr(edge_registry, "passenger_allowed", None),
        )

        # Base weights: travel time = length / speed_limit (seconds)
        if edge_registry.lengths is not None:
            base_weights = edge_registry.lengths / np.maximum(
                edge_registry.speed_limits, 0.1
            )
        else:
            base_weights = np.ones(E, dtype=np.float32)

        route_edge_indices: List[List[List[int]]] = []
        route_masks = np.zeros((N_od, K, E), dtype=np.float32)
        penalty_factor = 3.0  # multiplicative penalty on used edges
        verbose_routes = N_od <= 128

        for oi, orig_id in enumerate(origins):
            for di, dest_id in enumerate(destinations):
                od_idx = oi * N_dest + di
                src_idx = edge_registry._edge_to_idx.get(orig_id)
                dst_idx = edge_registry._edge_to_idx.get(dest_id)

                routes_for_od: List[List[int]] = []
                seen_sets: List[frozenset] = []

                if src_idx is None or dst_idx is None:
                    # OD pair edge not in registry — pad with empties
                    routes_for_od = [[] for _ in range(K)]
                    route_edge_indices.append(routes_for_od)
                    continue

                weights = base_weights.copy()

                for k in range(K):
                    path = cls._dijkstra(adj_list, weights, src_idx, dst_idx)
                    if path is None:
                        break

                    path_set = frozenset(path)
                    # If duplicate of an existing route, increase penalty and retry
                    if path_set in seen_sets:
                        for idx in path:
                            weights[idx] *= penalty_factor
                        path = cls._dijkstra(adj_list, weights, src_idx, dst_idx)
                        if path is None:
                            break
                        path_set = frozenset(path)
                        if path_set in seen_sets:
                            break  # can't find more diverse routes

                    routes_for_od.append(path)
                    seen_sets.append(path_set)

                    # Penalise edges on this route for next iteration
                    for idx in path:
                        weights[idx] *= penalty_factor

                # Pad to K routes by duplicating the first
                while len(routes_for_od) < K:
                    if routes_for_od:
                        routes_for_od.append(list(routes_for_od[0]))
                    else:
                        routes_for_od.append([])

                n_distinct = len(set(frozenset(r) for r in routes_for_od if r))
                if verbose_routes:
                    print(f"  OD {od_idx:2d} ({orig_id} → {dest_id}): "
                          f"{n_distinct}/{K} distinct routes, "
                          f"lengths={[len(r) for r in routes_for_od]}")
                elif od_idx == 0 or (od_idx + 1) % 200 == 0 or (od_idx + 1) == N_od:
                    print(
                        f"  [RouteRegistry] progress {od_idx + 1}/{N_od} "
                        f"(last distinct={n_distinct}/{K}, lengths={[len(r) for r in routes_for_od]})"
                    )

                # Build masks
                for k, indices in enumerate(routes_for_od):
                    if indices:
                        w = 1.0 / len(indices)
                        for idx in indices:
                            route_masks[od_idx, k, idx] = w

                route_edge_indices.append(routes_for_od)

        # Parse agents.csv for per-OD vehicle counts
        od_av_counts = np.zeros(N_od, dtype=np.float32)
        od_hdv_counts = np.zeros(N_od, dtype=np.float32)

        if agents_csv is not None:
            import csv
            with open(agents_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    oi_raw = int(row["origin"])
                    di_raw = int(row["destination"])
                    if oi_raw < N_orig and di_raw < N_dest:
                        od_idx = oi_raw * N_dest + di_raw
                        kind = row.get("kind", "Human").lower()
                        # Initially all are human; AVs determined by ratio_machines
                        od_hdv_counts[od_idx] += 1

            # Split by ratio_machines: top ratio_machines fraction become AVs
            total_agents = od_hdv_counts.sum()
            if total_agents > 0:
                od_av_counts, od_hdv_counts = split_od_agent_counts(
                    od_hdv_counts,
                    ratio_machines,
                )

        print(f"[RouteRegistry] Built: {N_od} OD pairs, K={K}, E={E}")
        print(f"  Total AVs={int(od_av_counts.sum())}, HDVs={int(od_hdv_counts.sum())}")
        return cls(origins, destinations, K, route_edge_indices, route_masks,
                   od_av_counts, od_hdv_counts, E)

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "origins": self.origins,
            "destinations": self.destinations,
            "K": self.K,
            "n_edges": self.n_edges,
            "route_edge_indices": self.route_edge_indices,
            "route_masks": self.route_masks.tolist() if self.route_masks is not None else None,
            "od_av_counts": self.od_av_counts.tolist(),
            "od_hdv_counts": self.od_hdv_counts.tolist(),
            "route_indicators": (
                self.route_indicators.tolist()
                if self.route_indicators is not None
                else None
            ),
            "n_od": int(self.n_od),
            "od_pairs": self.od_pairs,
        }
        with open(path_obj, "w") as f:
            json.dump(data, f)
        print(f"[RouteRegistry] Saved → {path_obj}")

    @classmethod
    def load(cls, path: str, edge_registry: Optional["EdgeRegistry"] = None) -> "RouteRegistry":
        with open(path, "r") as f:
            data = json.load(f)
        route_indicators = None
        if "route_indicators" in data and data["route_indicators"] is not None:
            route_indicators = np.array(data["route_indicators"], dtype=np.float32)
        route_masks_data = data.get("route_masks")
        return cls(
            origins=data["origins"],
            destinations=data["destinations"],
            K=data["K"],
            route_edge_indices=data["route_edge_indices"],
            route_masks=(
                np.array(route_masks_data, dtype=np.float32)
                if route_masks_data is not None else None
            ),
            od_av_counts=np.array(data["od_av_counts"], dtype=np.float32),
            od_hdv_counts=np.array(data["od_hdv_counts"], dtype=np.float32),
            n_edges=data["n_edges"],
            route_indicators=route_indicators,
            n_od=data.get("n_od"),
            od_pairs=data.get("od_pairs"),
        )

    def od_index(self, origin_idx: int, dest_idx: int) -> int:
        """Return the internal OD row for an origin/destination index pair."""
        oi = int(origin_idx)
        di = int(dest_idx)
        if self.od_index_map is not None:
            return int(self.od_index_map.get((oi, di), -1))
        n_dest = max(1, len(self.destinations))
        od_idx = oi * n_dest + di
        if 0 <= od_idx < int(self.n_od):
            return int(od_idx)
        return -1

    def _build_route_indicators(self) -> np.ndarray:
        """Build binary (N_od, K, E) indicators from route_edge_indices."""
        indicators = np.zeros((self.n_od, self.K, self.n_edges), dtype=np.float32)
        for od_i, routes in enumerate(self.route_edge_indices):
            for k, edge_indices in enumerate(routes):
                for eidx in edge_indices:
                    indicators[od_i, k, eidx] = 1.0
        return indicators

    def get_route_indicators_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return cached (N_od, K, E) binary tensor."""
        if self.route_indicators is None:
            raise RuntimeError(
                "route_indicators are not available for this lightweight path topology"
            )
        if self._route_indicators_tensor is None or \
           str(self._route_indicators_tensor.device) != str(device):
            self._route_indicators_tensor = torch.from_numpy(
                self.route_indicators
            ).to(device)
        return self._route_indicators_tensor

    def get_route_masks_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return cached (N_od, K, E) tensor."""
        if self.route_masks is None:
            raise RuntimeError(
                "route_masks are not available for this lightweight path topology"
            )
        if self._route_masks_tensor is None or \
           str(self._route_masks_tensor.device) != str(device):
            self._route_masks_tensor = torch.from_numpy(
                self.route_masks
            ).to(device)
        return self._route_masks_tensor


# Internal neutral alias for the active vehicle/edge path. The underlying
# structure still stores path incidence data, but public training code should
# treat it as a path topology rather than a route-level policy interface.
PathTopology = RouteRegistry


def build_route_registry(
    network_folder: str,
    network_name: str,
    edge_registry: "EdgeRegistry",
    origins: List[str],
    destinations: List[str],
    K: int = 4,
    agents_csv: Optional[str] = None,
    ratio_machines: float = 0.2,
) -> RouteRegistry:
    """Build (or load cached) RouteRegistry. Cache: network_folder/route_registry.json

    If a cached registry exists but its AV/HDV ratio doesn't match
    *ratio_machines*, the cache is invalidated and rebuilt so that
    per-OD vehicle counts stay consistent with the caller's config.
    """
    cache = Path(network_folder) / "route_registry.json"
    if cache.exists():
        try:
            cached = RouteRegistry.load(str(cache))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(
                f"[RouteRegistry] Failed to load cached registry from {cache}: {exc}. "
                "Rebuilding from network files."
            )
        else:
            # Validate ratio: compare cached AV fraction with requested
            cached_total = float(cached.od_av_counts.sum() + cached.od_hdv_counts.sum())
            if cached_total > 0:
                cached_ratio = float(cached.od_av_counts.sum()) / cached_total
            else:
                cached_ratio = 0.0
            if abs(cached_ratio - ratio_machines) < 0.02:
                print(f"[RouteRegistry] Loading cached registry from {cache} "
                      f"(ratio={cached_ratio:.3f}, requested={ratio_machines:.3f})")
                return cached
            else:
                print(f"[RouteRegistry] Cache ratio mismatch: cached={cached_ratio:.3f}, "
                      f"requested={ratio_machines:.3f}; rebuilding.")
    net_xml = Path(network_folder) / f"{network_name}.net.xml"
    if not net_xml.exists():
        raise FileNotFoundError(f"Network file not found: {net_xml}")
    reg = RouteRegistry.build(
        str(net_xml), edge_registry, origins, destinations, K,
        agents_csv=agents_csv, ratio_machines=ratio_machines,
    )
    reg.save(str(cache))
    return reg


class EdgeGNNLayer(nn.Module):
    """Single message-passing layer on edge line graph."""
    def __init__(self, d_model: int):
        super().__init__()
        self.W_msg  = nn.Linear(d_model, d_model, bias=False)
        self.W_self = nn.Linear(d_model, d_model)
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h: (*, E, d), adj: (E, E) → (*, E, d)"""
        msg = torch.matmul(adj, self.W_msg(h))
        out = F.gelu(self.W_self(h) + msg)
        return self.norm(out + h)


class _InverseDecoderCompatibility:
    """Compatibility wrapper exposing old single-head inv_proj API."""

    def __init__(self, flow_projector: "FlowProjector"):
        self.flow_projector = flow_projector
        self.override: Optional[nn.Linear] = None

    @property
    def weight(self) -> torch.Tensor:
        if self.override is not None:
            return self.override.weight
        return self._composed_weight()

    @property
    def bias(self) -> torch.Tensor:
        if self.override is not None:
            return self.override.bias
        return self._composed_bias()

    def _composed_weight(self) -> torch.Tensor:
        fp = self.flow_projector
        shared = fp.inv_shared[0].weight
        speed = fp.inv_speed_head.weight
        volume = fp.inv_volume_head.weight
        other = fp.inv_other_head.weight
        if fp.n_dynamic == 4 and other.shape[0] >= 2:
            stacked = torch.cat([other[:1], speed, volume, other[1:2]], dim=0)
        else:
            pieces = [speed, volume]
            if other.shape[0] > 0:
                pieces.append(other)
            stacked = torch.cat(pieces, dim=0)
        return stacked @ shared

    def _composed_bias(self) -> torch.Tensor:
        fp = self.flow_projector
        hidden_bias = fp.inv_shared[0].bias
        speed_w = fp.inv_speed_head.weight
        volume_w = fp.inv_volume_head.weight
        other_w = fp.inv_other_head.weight
        speed_b = fp.inv_speed_head.bias
        volume_b = fp.inv_volume_head.bias
        other_b = fp.inv_other_head.bias
        if fp.n_dynamic == 4 and other_w.shape[0] >= 2:
            weights = torch.cat([other_w[:1], speed_w, volume_w, other_w[1:2]], dim=0)
            biases = torch.cat([other_b[:1], speed_b, volume_b, other_b[1:2]], dim=0)
        else:
            bias_pieces = [speed_b, volume_b]
            weight_pieces = [speed_w, volume_w]
            if other_w.shape[0] > 0:
                weight_pieces.append(other_w)
                bias_pieces.append(other_b)
            weights = torch.cat(weight_pieces, dim=0)
            biases = torch.cat(bias_pieces, dim=0)
        hidden = F.silu(hidden_bias)
        return biases + F.linear(hidden, weights)

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        if self.override is not None:
            return self.override(z.clone())
        return self.flow_projector.decode(z)


class FlowProjector(nn.Module):
    """
    GNN-based projector for edge flow features.

    forward:  (*, E, d_feat=8)    → (*, E, d_spatial=64)   for CFM dynamics
    decode:   (*, E, d_spatial=64) → (*, E, n_dynamic=4)   for policy obs
    """

    def __init__(
        self,
        d_feat:       int = D_FEAT,
        d_spatial:    int = 64,
        n_dynamic:    int = N_DYNAMIC,
        n_gnn_layers: int = 2,
        enable_t_all_head: bool = False,
    ):
        super().__init__()
        self.d_feat       = d_feat
        self.d_spatial    = d_spatial
        self.n_dynamic    = n_dynamic
        self.n_gnn_layers = n_gnn_layers
        self.enable_t_all_head = bool(enable_t_all_head)

        self.input_proj = nn.Linear(d_feat, d_spatial)
        self.gnn_layers = nn.ModuleList(
            [EdgeGNNLayer(d_spatial) for _ in range(n_gnn_layers)]
        )
        self.inv_shared = nn.Sequential(
            nn.Linear(d_spatial, d_spatial),
            nn.SiLU(),
        )
        self.inv_speed_head = nn.Linear(d_spatial, 1)
        self.inv_volume_head = nn.Linear(d_spatial, 1)
        self.inv_other_head = nn.Linear(d_spatial, max(0, n_dynamic - 2))
        self.route_cost_head = nn.Sequential(
            nn.Linear(d_spatial, d_spatial),
            nn.SiLU(),
            nn.Linear(d_spatial, 1),
        )
        self.inv_proj = _InverseDecoderCompatibility(self)
        self.obs_proj: Optional[nn.Linear] = None  # latent obs projection, created by init_obs_proj()
        self.t_all_head = None
        # Multi-stat pooling: [mean, max, veh-weighted-mean] → 3*d_spatial
        self._t_all_pool_dim = d_spatial * 3
        if self.enable_t_all_head:
            self.t_all_head = nn.Sequential(
                nn.Linear(self._t_all_pool_dim, d_spatial),
                nn.GELU(),
                nn.Linear(d_spatial, d_spatial // 2),
                nn.GELU(),
                nn.Linear(d_spatial // 2, 1),
            )
            _last = self.t_all_head[-1]
            nn.init.zeros_(_last.weight)
            nn.init.constant_(_last.bias, 1.5)  # softplus(1.5)≈1.74, closer to typical t_all

        nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.inv_shared[0].weight, gain=1.0)
        nn.init.zeros_(self.inv_shared[0].bias)
        nn.init.orthogonal_(self.inv_speed_head.weight, gain=1.0)
        nn.init.zeros_(self.inv_speed_head.bias)
        nn.init.orthogonal_(self.inv_volume_head.weight, gain=1.0)
        nn.init.zeros_(self.inv_volume_head.bias)
        if self.inv_other_head.out_features > 0:
            nn.init.orthogonal_(self.inv_other_head.weight, gain=1.0)
            nn.init.zeros_(self.inv_other_head.bias)
        nn.init.orthogonal_(self.route_cost_head[0].weight, gain=1.0)
        nn.init.zeros_(self.route_cost_head[0].bias)
        nn.init.zeros_(self.route_cost_head[2].weight)
        nn.init.constant_(self.route_cost_head[2].bias, float(np.log(np.expm1(1.0))))

    def set_adjacency(self, adj: torch.Tensor):
        """Register (E, E) row-normalized adjacency as a buffer."""
        self.register_buffer("adj", adj)

    def forward(
        self,
        x: torch.Tensor,
        return_inverse: bool = False,
        return_t_all: bool = False,
        return_decoded: Optional[bool] = None,
    ):
        """
        x: (*, E, d_feat) → (*, E, d_spatial)

        If return_decoded=True, also returns decoded dynamic flow channels:
            ((* ,E,d_spatial), (*,E,n_dynamic))
        return_inverse is kept as a compatibility alias for return_decoded.
        If return_t_all=True, also returns predicted episode-level t_all (minutes):
            shape (*) matching x batch/time dimensions without edge dimension.
        """
        if return_decoded is not None:
            return_inverse = bool(return_decoded)
        h = self.input_proj(x)
        if hasattr(self, "adj") and self.adj is not None:
            for layer in self.gnn_layers:
                h = layer(h, self.adj)
        decoded = self.decode(h) if return_inverse else None
        t_all = self.predict_t_all_from_z(h) if return_t_all else None
        if return_inverse and return_t_all:
            return h, decoded, t_all
        if return_inverse:
            return h, decoded
        if return_t_all:
            return h, t_all
        return h

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (*, E, d_spatial) → (*, E, n_dynamic)"""
        if self.inv_proj.override is not None:
            override = self.inv_proj.override.to(device=z.device, dtype=z.dtype)
            return override(z.clone())
        h = self.inv_shared(z)
        speed = self.inv_speed_head(h)
        volume = decode_nonnegative_volume(self.inv_volume_head(h))
        other = self.inv_other_head(h)
        if self.n_dynamic == 4:
            return torch.cat([other[..., :1], speed, volume, other[..., 1:2]], dim=-1)
        return torch.cat([speed, volume, other], dim=-1)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for decode(). New code should call decode()."""
        return self.decode(z)

    def init_obs_proj(self, d_obs: int) -> None:
        """Create latent obs projection layer (call before dynamics training)."""
        self.obs_proj = nn.Linear(self.d_spatial, d_obs)
        nn.init.orthogonal_(self.obs_proj.weight, gain=1.0)
        nn.init.zeros_(self.obs_proj.bias)

    def project_obs(self, z: torch.Tensor) -> torch.Tensor:
        """z: (*, E, d_spatial) → (*, E, d_obs). Falls back to decoder if obs_proj absent."""
        if self.obs_proj is not None:
            return self.obs_proj(z)
        return self.decode(z)

    def predict_edge_route_cost(
        self,
        z: torch.Tensor,
        free_flow_time: torch.Tensor,
    ) -> torch.Tensor:
        """Predict planner-calibrated edge travel time without changing decoder heads.

        The head predicts a positive multiplier over free-flow time. It is
        initialized to exactly 1x free-flow so old checkpoints start from the
        shortest-path baseline before route-cost distillation.
        """
        ff = free_flow_time.to(device=z.device, dtype=z.dtype)
        while ff.dim() < z.dim() - 1:
            ff = ff.unsqueeze(0)
        raw = self.route_cost_head(z).squeeze(-1)
        multiplier = F.softplus(raw).clamp_min(1e-4)
        return ff * multiplier

    @property
    def d_obs(self) -> int:
        """Observation feature dimension per edge."""
        if self.obs_proj is not None:
            return self.obs_proj.out_features
        return self.n_dynamic

    @property
    def has_t_all_head(self) -> bool:
        return self.t_all_head is not None

    def predict_t_all_from_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Predict episode-level mean travel time t_all (minutes) from projected tokens.

        Uses multi-stat pooling [mean, max, veh-weighted-mean] over edges to
        preserve more spatial information than a simple mean pool.

        Args:
            z: (*, E, d_spatial)
        Returns:
            t_all: (*) tensor, same leading dims as z except edge/spatial dims.
        """
        if self.t_all_head is None:
            raise RuntimeError("t_all_head is not enabled in FlowProjector.")

        # Decode to get vehicle counts for attention weighting
        decoded = self.decode(z)             # (*, E, n_dynamic)
        veh = decoded[..., IDX_VEH].clamp(min=0.0)  # (*, E)

        # Multi-stat pooling over edge dimension (-2)
        z_mean = z.mean(dim=-2)             # (*, d_spatial)
        z_max  = z.max(dim=-2).values       # (*, d_spatial)
        # Vehicle-weighted mean: focus on edges with traffic
        veh_w = veh / (veh.sum(dim=-1, keepdim=True) + 1e-8)  # (*, E)
        z_veh  = (z * veh_w.unsqueeze(-1)).sum(dim=-2)         # (*, d_spatial)

        pooled = torch.cat([z_mean, z_max, z_veh], dim=-1)     # (*, 3*d_spatial)
        raw = self.t_all_head(pooled).squeeze(-1)
        pred = F.softplus(raw)
        return pred

    def save(self, path: str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "d_feat":       self.d_feat,
            "d_spatial":    self.d_spatial,
            "n_dynamic":    self.n_dynamic,
            "n_gnn_layers": self.n_gnn_layers,
            "enable_t_all_head": self.enable_t_all_head,
            "t_all_pool_dim": self._t_all_pool_dim,
            "model":        self.state_dict(),
        }, path)
        print(f"[FlowProjector] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "FlowProjector":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        fp   = cls(
            d_feat       = ckpt.get("d_feat",       D_FEAT),
            d_spatial    = ckpt.get("d_spatial",    64),
            n_dynamic    = ckpt.get("n_dynamic",    N_DYNAMIC),
            n_gnn_layers = ckpt.get("n_gnn_layers", 0),
            enable_t_all_head = bool(ckpt.get("enable_t_all_head", False)),
        )
        # Auto-init obs_proj if checkpoint contains obs_proj weights
        model_sd = dict(ckpt.get("model", {}))
        if "obs_proj.weight" in model_sd and fp.obs_proj is None:
            d_obs = model_sd["obs_proj.weight"].shape[0]
            fp.init_obs_proj(d_obs)
        old_inv_weight = model_sd.pop("inv_proj.weight", None)
        old_inv_bias = model_sd.pop("inv_proj.bias", None)
        fp.load_state_dict(model_sd, strict=False)
        if old_inv_weight is not None:
            override = nn.Linear(fp.d_spatial, fp.n_dynamic)
            override.weight.data.copy_(old_inv_weight)
            if old_inv_bias is not None:
                override.bias.data.copy_(old_inv_bias)
            else:
                override.bias.data.zero_()
            fp.inv_proj.override = override
        print(f"[FlowProjector] Loaded from {path}")
        return fp.to(device)


def build_edge_registry(network_folder: str, network_name: str) -> "EdgeRegistry":
    """
    Build (or load cached) EdgeRegistry.
    Cache path: network_folder/edge_registry.json
    """
    cache = Path(network_folder) / "edge_registry.json"
    if cache.exists():
        reg = EdgeRegistry.load(str(cache))
        passenger_allowed = getattr(reg, "passenger_allowed", None)
        if (
            bool(getattr(reg, "_has_passenger_allowed", False))
            and passenger_allowed is not None
            and np.asarray(passenger_allowed).shape[0] == int(reg.n_edges)
        ):
            print(f"[EdgeRegistry] Loading cached registry from {cache}")
            return reg
        print(
            f"[EdgeRegistry] Cached registry at {cache} lacks passenger edge permissions; "
            "rebuilding from network files."
        )
    net_xml = Path(network_folder) / f"{network_name}.net.xml"
    if not net_xml.exists():
        raise FileNotFoundError(f"Network file not found: {net_xml}")
    reg = EdgeRegistry.build(str(net_xml), network=network_name)
    reg.save(str(cache))
    return reg
