"""Route, vehicle-action, and HDV distribution helpers for SUMO evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    traci = None
    TRACI_AVAILABLE = False

try:
    from dreamer4.urban_routing.vehicle_imagination import build_diverse_paths_current_to_dest
except Exception:
    try:
        from vehicle_imagination import build_diverse_paths_current_to_dest
    except Exception:
        build_diverse_paths_current_to_dest = None


def _load_av_od_route_overrides(path_value: str) -> Dict[Tuple[int, int], int]:
    if not str(path_value or "").strip():
        return {}
    path = Path(str(path_value))
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("overrides"), list):
        data = data["overrides"]

    overrides: Dict[Tuple[int, int], int] = {}
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            origin = item.get("origin", item.get("o"))
            destination = item.get("destination", item.get("dest", item.get("d")))
            action = item.get("action", item.get("route_index", item.get("route")))
            if origin is None or destination is None or action is None:
                continue
            overrides[(int(origin), int(destination))] = int(action)
        return overrides

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                action = value.get("action", value.get("route_index", value.get("route")))
            else:
                action = value
            if action is None:
                continue
            if isinstance(key, str):
                parts = key.replace(":", ",").split(",")
                if len(parts) != 2:
                    continue
                origin, destination = int(parts[0].strip()), int(parts[1].strip())
            elif isinstance(key, (list, tuple)) and len(key) == 2:
                origin, destination = int(key[0]), int(key[1])
            else:
                continue
            overrides[(origin, destination)] = int(action)
        return overrides

    raise ValueError(
        f"Unsupported --av_od_route_override_json format in {path}; expected list or object."
    )


def _parse_float_list(value: str) -> List[float]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    vals = [float(p) for p in parts]
    if any(not np.isfinite(v) for v in vals):
        raise ValueError(f"Expected finite float list, got {value!r}")
    return vals


def _edge_route_cost_matrix(
    path_batch: List[List[List[int]]],
    edge_cost: np.ndarray,
    *,
    n_routes: int,
    length_norm: str,
) -> np.ndarray:
    """Compute per-vehicle, per-route costs from an edge-indexed cost vector."""
    edge_cost_arr = np.asarray(edge_cost, dtype=np.float32).reshape(-1)
    n_agents = int(len(path_batch))
    k = int(max(n_routes, 1))
    costs = np.full((n_agents, k), np.inf, dtype=np.float32)
    use_mean = str(length_norm).strip().lower() == "mean"

    for i, paths in enumerate(path_batch):
        for route_idx, path in enumerate(list(paths)[:k]):
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < edge_cost_arr.shape[0])]
            if idxs.size == 0:
                continue
            vals = edge_cost_arr[idxs]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            costs[i, route_idx] = float(vals.mean() if use_mean else vals.sum())
    return costs


def _select_min_cost_route_indices(
    cost_matrix: np.ndarray,
    baseline_indices: np.ndarray,
    *,
    deterministic: bool,
    rng: np.random.RandomState,
    temperature: float,
) -> Tuple[np.ndarray, int]:
    """Select lowest-cost route per vehicle, falling back to baseline when costs are invalid."""
    costs = np.asarray(cost_matrix, dtype=np.float32)
    selected = np.asarray(baseline_indices, dtype=np.int64).copy()
    if costs.ndim != 2 or selected.shape[0] != costs.shape[0]:
        return selected, 0

    applied = 0
    temp = float(temperature)
    for i in range(costs.shape[0]):
        row = costs[i]
        finite = np.isfinite(row)
        if not bool(finite.any()):
            continue
        finite_indices = np.flatnonzero(finite).astype(np.int64)
        finite_costs = row[finite_indices].astype(np.float64)
        if (not deterministic) and temp > 0.0 and finite_costs.size > 1:
            logits = -(finite_costs - float(finite_costs.min())) / max(temp, 1e-6)
            logits = logits - float(logits.max())
            probs = np.exp(logits)
            denom = float(probs.sum())
            if np.isfinite(denom) and denom > 0.0:
                probs = probs / denom
                selected[i] = int(rng.choice(finite_indices, p=probs))
            else:
                selected[i] = int(finite_indices[int(np.argmin(finite_costs))])
        else:
            selected[i] = int(finite_indices[int(np.argmin(finite_costs))])
        applied += 1
    return selected, applied


def _select_min_cost_route_indices_with_planned_pressure(
    cost_matrix: np.ndarray,
    baseline_indices: np.ndarray,
    path_batch: Sequence[Sequence[Sequence[int]]],
    *,
    n_edges: int,
    deterministic: bool,
    rng: np.random.RandomState,
    temperature: float,
    pressure_penalty: float,
) -> Tuple[np.ndarray, int]:
    """Sequentially select routes with a same-tick planned-AV pressure penalty."""
    penalty = max(float(pressure_penalty), 0.0)
    if penalty <= 0.0:
        return _select_min_cost_route_indices(
            cost_matrix,
            baseline_indices,
            deterministic=deterministic,
            rng=rng,
            temperature=temperature,
        )
    costs = np.asarray(cost_matrix, dtype=np.float32)
    selected = np.asarray(baseline_indices, dtype=np.int64).copy()
    if costs.ndim != 2 or selected.shape[0] != costs.shape[0]:
        return selected, 0
    edge_pressure = np.zeros((max(int(n_edges), 0),), dtype=np.float32)
    norm = max(float(costs.shape[0]), 1.0)
    applied = 0
    temp = float(temperature)
    for i in range(costs.shape[0]):
        row = costs[i].astype(np.float64).copy()
        finite = np.isfinite(row)
        if not bool(finite.any()):
            continue
        routes = path_batch[i] if i < len(path_batch) else []
        for k in range(min(len(routes), row.shape[0])):
            idxs = [int(e) for e in routes[k] if 0 <= int(e) < edge_pressure.shape[0]]
            if idxs:
                row[k] += penalty * float(edge_pressure[idxs].sum()) / norm
        finite_indices = np.flatnonzero(finite).astype(np.int64)
        finite_costs = row[finite_indices].astype(np.float64)
        if (not deterministic) and temp > 0.0 and finite_costs.size > 1:
            logits = -(finite_costs - float(finite_costs.min())) / max(temp, 1e-6)
            logits = logits - float(logits.max())
            probs = np.exp(logits)
            denom = float(probs.sum())
            if np.isfinite(denom) and denom > 0.0:
                choice = int(rng.choice(finite_indices, p=probs / denom))
            else:
                choice = int(finite_indices[int(np.argmin(finite_costs))])
        else:
            choice = int(finite_indices[int(np.argmin(finite_costs))])
        selected[i] = choice
        if i < len(routes) and 0 <= choice < len(routes):
            for edge_idx in routes[choice]:
                e = int(edge_idx)
                if 0 <= e < edge_pressure.shape[0]:
                    edge_pressure[e] += 1.0
        applied += 1
    return selected, applied


def _select_quota_route_indices(
    route_probs: np.ndarray,
    baseline_indices: np.ndarray,
    *,
    n_routes: int,
) -> np.ndarray:
    """Deterministically realize fleet-level policy probabilities as route counts.

    Independent per-vehicle argmax collapses when one route is only slightly
    better for every vehicle. Quota selection preserves the actor's aggregate
    probability mass within the current decision tick without using route labels
    or future simulator outcomes.
    """
    probs = np.asarray(route_probs, dtype=np.float64)
    baseline = np.asarray(baseline_indices, dtype=np.int64).reshape(-1)
    n_agents = int(baseline.shape[0])
    k = max(int(n_routes), 1)
    if probs.ndim != 2 or probs.shape[0] != n_agents or n_agents == 0:
        return baseline.copy()

    probs = probs[:, :k].copy()
    probs[~np.isfinite(probs)] = 0.0
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    valid_rows = row_sums[:, 0] > 0.0
    if not bool(valid_rows.any()):
        return baseline.copy()
    probs[valid_rows] = probs[valid_rows] / row_sums[valid_rows]
    probs[~valid_rows] = 0.0
    for i in np.flatnonzero(~valid_rows):
        b = int(np.clip(baseline[i], 0, k - 1))
        probs[i, b] = 1.0

    expected = probs.sum(axis=0)
    target = np.floor(expected).astype(np.int64)
    remainder = int(n_agents - int(target.sum()))
    if remainder > 0:
        frac_order = np.argsort(-(expected - target), kind="mergesort")
        for ridx in frac_order[:remainder]:
            target[int(ridx)] += 1
    elif remainder < 0:
        frac_order = np.argsort(expected - target, kind="mergesort")
        for ridx in frac_order[: -remainder]:
            if target[int(ridx)] > 0:
                target[int(ridx)] -= 1

    selected = np.full((n_agents,), -1, dtype=np.int64)
    remaining_agents = set(range(n_agents))
    route_order = np.argsort(-target, kind="mergesort")
    for ridx in route_order:
        quota = int(target[int(ridx)])
        if quota <= 0 or not remaining_agents:
            continue
        candidates = np.asarray(sorted(remaining_agents), dtype=np.int64)
        scores = probs[candidates, int(ridx)]
        order = np.argsort(-scores, kind="mergesort")
        take = candidates[order[: min(quota, candidates.size)]]
        selected[take] = int(ridx)
        for idx in take.tolist():
            remaining_agents.discard(int(idx))

    if remaining_agents:
        for idx in sorted(remaining_agents):
            selected[int(idx)] = int(np.argmax(probs[int(idx)]))
    invalid = selected < 0
    if bool(invalid.any()):
        selected[invalid] = baseline[invalid]
    return selected.clip(0, k - 1).astype(np.int64)


def _route_cost_matrix_summary(
    cost_matrix: Optional[np.ndarray],
    n_routes: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    means = np.full((int(n_routes),), np.nan, dtype=np.float32)
    best_counts = np.zeros((int(n_routes),), dtype=np.int64)
    if cost_matrix is None:
        return means, best_counts, 0
    costs = np.asarray(cost_matrix, dtype=np.float32)
    if costs.ndim != 2:
        return means, best_counts, 0
    k_eff = min(int(n_routes), int(costs.shape[1]))
    for k in range(k_eff):
        vals = costs[:, k]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            means[k] = float(vals.mean())
    total = 0
    for row in costs:
        row = np.asarray(row[:k_eff], dtype=np.float32)
        finite = np.isfinite(row)
        if not bool(finite.any()):
            continue
        finite_indices = np.flatnonzero(finite).astype(np.int64)
        finite_costs = row[finite_indices]
        best_idx = int(finite_indices[int(np.argmin(finite_costs))])
        if 0 <= best_idx < int(n_routes):
            best_counts[best_idx] += 1
            total += 1
    return means, best_counts, int(total)


def _build_edge_adj_list(
    adjacency: Optional[np.ndarray],
    allowed_edges: Optional[np.ndarray] = None,
) -> List[List[int]]:
    if adjacency is None:
        return []
    arr = np.asarray(adjacency)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return []
    allowed = None
    if allowed_edges is not None:
        allowed = np.asarray(allowed_edges, dtype=bool).reshape(-1)
        if allowed.shape[0] != arr.shape[0]:
            allowed = None
    adj: List[List[int]] = [[] for _ in range(int(arr.shape[0]))]
    rows, cols = np.nonzero(arr)
    for r, c in zip(rows.tolist(), cols.tolist()):
        ri = int(r)
        ci = int(c)
        if allowed is not None and (not bool(allowed[ri]) or not bool(allowed[ci])):
            continue
        if 0 <= ri < len(adj):
            adj[ri].append(ci)
    return adj


def _route_batch_uniqueness_summary(route_batch: List[List[List[str]]]) -> Tuple[float, int, int]:
    if not route_batch:
        return 0.0, 0, 0
    counts: List[int] = []
    for candidates in route_batch:
        unique = {tuple(route) for route in candidates if route}
        counts.append(int(len(unique)))
    if not counts:
        return 0.0, 0, 0
    arr = np.asarray(counts, dtype=np.float32)
    return float(arr.mean()), int(arr.min()), int(arr.max())


def _aggregate_global_frac(fracs_od: np.ndarray, counts_od: np.ndarray, n_routes: int) -> np.ndarray:
    veh = fracs_od * counts_od[:, None]
    total = float(veh.sum())
    if total < 1e-8:
        return np.ones(n_routes, dtype=np.float32) / float(n_routes)
    out = veh.sum(axis=0).astype(np.float32)
    out = np.clip(out, 1e-8, None)
    out = out / out.sum()
    return out


def _normalize_prob_vector(vec: np.ndarray, n_routes: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.shape[0] != int(n_routes):
        arr = np.ones(int(n_routes), dtype=np.float32)
    arr = np.clip(arr, 1e-8, None)
    arr = arr / arr.sum()
    return arr.astype(np.float32)


def _temperature_adjust_probs(vec: np.ndarray, temperature: float, n_routes: int) -> np.ndarray:
    probs = _normalize_prob_vector(vec, n_routes)
    temp = float(temperature)
    if not np.isfinite(temp) or temp <= 0.0 or abs(temp - 1.0) < 1e-6:
        return probs
    logits = np.log(np.clip(probs.astype(np.float64), 1e-12, None)) / temp
    logits = logits - float(np.max(logits))
    out = np.exp(logits)
    out = out / max(float(out.sum()), 1e-12)
    return out.astype(np.float32)


def _safe_parse_agent_id(agent_id: Any) -> Optional[int]:
    try:
        return int(str(agent_id).split("_")[-1])
    except Exception:
        return None


def _build_av_agent_index(
    env,
    od_agent_map: Dict[int, int],
    n_od: int,
) -> Tuple[List[int], np.ndarray, Dict[int, int]]:
    """
    Build stable AV-agent index:
      - av_agent_ids: sorted integer AV IDs
      - av_od_indices: AV->OD index (for per-vehicle obs gathering)
      - av_id_to_idx: ID->row mapping in per-vehicle tensors
    """
    av_ids: List[int] = []
    for m in getattr(env, "machine_agents", []):
        aid = _safe_parse_agent_id(getattr(m, "id", ""))
        if aid is not None:
            av_ids.append(int(aid))
    av_ids = sorted(set(av_ids))

    if not av_ids:
        return [], np.zeros((0,), dtype=np.int64), {}

    n_od_safe = max(int(n_od), 1)
    av_od = np.zeros((len(av_ids),), dtype=np.int64)
    for i, aid in enumerate(av_ids):
        od_idx = int(od_agent_map.get(int(aid), 0))
        av_od[i] = int(np.clip(od_idx, 0, n_od_safe - 1))
    id_to_idx = {int(aid): int(i) for i, aid in enumerate(av_ids)}
    return av_ids, av_od, id_to_idx


def _aggregate_vehicle_fracs_to_od(
    av_vehicle_fracs: np.ndarray,  # (N_av, K)
    av_od_indices: np.ndarray,     # (N_av,)
    n_od: int,
    n_routes: int,
) -> np.ndarray:
    """
    Convert per-vehicle AV route distributions to per-OD route distributions.
    """
    n_od = max(int(n_od), 1)
    k = max(int(n_routes), 1)
    out = np.zeros((n_od, k), dtype=np.float32)
    cnt = np.zeros((n_od,), dtype=np.float32)

    if av_vehicle_fracs.size == 0 or av_od_indices.size == 0:
        return np.ones((n_od, k), dtype=np.float32) / float(k)

    for i in range(min(av_vehicle_fracs.shape[0], av_od_indices.shape[0])):
        od = int(av_od_indices[i])
        if od < 0 or od >= n_od:
            continue
        p = _normalize_prob_vector(av_vehicle_fracs[i], k)
        out[od] += p
        cnt[od] += 1.0

    uniform = np.ones((k,), dtype=np.float32) / float(k)
    for od in range(n_od):
        if cnt[od] > 0.0:
            out[od] /= float(cnt[od])
        else:
            out[od] = uniform
    return out


def _vehicle_action_edge_distribution(
    av_vehicle_fracs: np.ndarray,   # (N_av, K)
    av_od_indices: np.ndarray,      # (N_av,)
    route_incidence_np: Optional[np.ndarray],  # (N_od, K, E)
    n_edges: int,
    n_routes: int,
) -> np.ndarray:
    """
    Convert per-vehicle route distributions into expected edge-visit distribution.
    """
    out = np.zeros((int(n_edges),), dtype=np.float32)
    if (
        route_incidence_np is None
        or av_vehicle_fracs.size == 0
        or av_od_indices.size == 0
        or int(n_edges) <= 0
    ):
        return out

    n_od = int(route_incidence_np.shape[0])
    k = int(min(route_incidence_np.shape[1], int(n_routes)))
    for i in range(min(av_vehicle_fracs.shape[0], av_od_indices.shape[0])):
        od = int(av_od_indices[i])
        if od < 0 or od >= n_od:
            continue
        probs = _normalize_prob_vector(av_vehicle_fracs[i], k)
        out += np.einsum("k,ke->e", probs, route_incidence_np[od, :k], optimize=True).astype(np.float32)

    s = float(out.sum())
    if s > 1e-8:
        out /= s
    return out


def _vehicle_action_edge_distribution_from_path_batch(
    av_vehicle_fracs: np.ndarray,   # (N_av, K)
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
) -> np.ndarray:
    """
    Convert per-vehicle candidate-path distributions into expected edge visitation.
    """
    out = np.zeros((int(n_edges),), dtype=np.float32)
    if av_vehicle_fracs.size == 0 or not path_batch or int(n_edges) <= 0:
        return out

    for i in range(min(int(av_vehicle_fracs.shape[0]), len(path_batch))):
        paths = path_batch[i]
        if not paths:
            continue
        k_eff = int(min(len(paths), int(n_routes), int(av_vehicle_fracs.shape[1])))
        if k_eff <= 0:
            continue
        probs = _normalize_prob_vector(av_vehicle_fracs[i][:k_eff], k_eff)
        for k in range(k_eff):
            path = paths[k]
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < int(n_edges))]
            if idxs.size == 0:
                continue
            out[idxs] += float(probs[k])

    s = float(out.sum())
    if s > 1e-8:
        out /= s
    return out.astype(np.float32)


def _build_live_vehicle_path_batch(
    *,
    av_agent_ids: List[int],
    av_agent_name_map: Dict[int, str],
    av_od_indices: np.ndarray,
    route_registry,
    edge_to_idx: Dict[str, int],
    adj_list: List[List[int]],
    edge_tt: np.ndarray,
    n_routes: int,
    path_penalty_factor: float,
) -> List[List[List[int]]]:
    """
    Build current-edge -> destination candidate paths for each AV.
    Falls back to the OD origin edge if the vehicle is not yet active in TraCI.
    """
    path_batch: List[List[List[int]]] = []
    n_dest = max(1, len(route_registry.destinations))
    default_origin = route_registry.origins[0] if route_registry.origins else None
    active_vehicle_ids: set[str] = set()
    if TRACI_AVAILABLE:
        try:
            active_vehicle_ids = {str(v) for v in traci.vehicle.getIDList()}
        except Exception:
            active_vehicle_ids = set()

    for i, aid in enumerate(av_agent_ids):
        od = int(av_od_indices[i]) if i < len(av_od_indices) else 0
        oi = int(od // n_dest)
        di = int(od % n_dest)
        orig_id = route_registry.origins[oi] if oi < len(route_registry.origins) else default_origin
        dest_id = route_registry.destinations[di] if di < len(route_registry.destinations) else orig_id
        fallback_cur = int(edge_to_idx.get(orig_id, 0)) if orig_id is not None else 0
        dst_edge = int(edge_to_idx.get(dest_id, fallback_cur)) if dest_id is not None else fallback_cur
        cur_edge = fallback_cur

        if TRACI_AVAILABLE:
            raw_name = av_agent_name_map.get(int(aid), "")
            for cand in (raw_name, str(aid)):
                if (not cand) or (cand not in active_vehicle_ids):
                    continue
                try:
                    edge_id = str(traci.vehicle.getRoadID(cand))
                except Exception:
                    continue
                if edge_id and edge_id in edge_to_idx:
                    cur_edge = int(edge_to_idx[edge_id])
                    break

        if cur_edge == dst_edge:
            path_batch.append([[cur_edge] for _ in range(int(n_routes))])
            continue

        if build_diverse_paths_current_to_dest is None:
            path_batch.append([[cur_edge] for _ in range(int(n_routes))])
            continue

        paths = build_diverse_paths_current_to_dest(
            adj_list=adj_list,
            edge_tt=edge_tt,
            src_idx=cur_edge,
            dst_idx=dst_edge,
            K=int(n_routes),
            penalty_factor=float(path_penalty_factor),
        )
        if not paths:
            paths = [[cur_edge] for _ in range(int(n_routes))]
        while len(paths) < int(n_routes):
            paths.append(list(paths[0]))
        path_batch.append(paths[: int(n_routes)])

    return path_batch


def _build_live_vehicle_path_batch_cached(
    *,
    av_agent_ids: List[int],
    av_agent_name_map: Dict[str, str],
    av_od_indices: np.ndarray,
    route_registry,
    edge_to_idx: Dict[str, int],
    path_cache,   # VehiclePathCache
    n_routes: int,
    edge_tt: Optional[np.ndarray] = None,
) -> List[List[List[int]]]:
    """
    Cache-based version: lookup pre-computed paths instead of running Dijkstra.
    Still uses TraCI to get each vehicle's current edge for accurate routing.
    """
    path_batch: List[List[List[int]]] = []
    n_dest = max(1, len(route_registry.destinations))
    default_origin = route_registry.origins[0] if route_registry.origins else None

    active_vehicle_ids: set = set()
    if TRACI_AVAILABLE:
        try:
            active_vehicle_ids = {str(v) for v in traci.vehicle.getIDList()}
        except Exception:
            pass

    for i, aid in enumerate(av_agent_ids):
        od = int(av_od_indices[i]) if i < len(av_od_indices) else 0
        oi = int(od // n_dest)
        di = int(od % n_dest)
        orig_id = route_registry.origins[oi] if oi < len(route_registry.origins) else default_origin
        dest_id = route_registry.destinations[di] if di < len(route_registry.destinations) else orig_id
        fallback_cur = int(edge_to_idx.get(orig_id, 0)) if orig_id else 0
        dst_edge = int(edge_to_idx.get(dest_id, fallback_cur)) if dest_id else fallback_cur
        cur_edge = fallback_cur

        # TraCI current-edge lookup (same as original)
        if TRACI_AVAILABLE:
            raw_name = av_agent_name_map.get(int(aid), "")
            for cand in (raw_name, str(aid)):
                if (not cand) or (cand not in active_vehicle_ids):
                    continue
                try:
                    edge_id = str(traci.vehicle.getRoadID(cand))
                except Exception:
                    continue
                if edge_id and edge_id in edge_to_idx:
                    cur_edge = int(edge_to_idx[edge_id])
                    break

        # Cache lookup instead of Dijkstra. Lazy cache needs edge travel times.
        if edge_tt is not None:
            paths = path_cache.lookup(cur_edge, dst_edge, edge_tt)
        else:
            paths = path_cache.lookup(cur_edge, dst_edge)
        path_batch.append(paths)

    return path_batch


def _dynamic_reroute_avs(
    *,
    av_agent_ids: List[int],
    av_agent_name_map: Dict[int, str],
    av_vehicle_fracs_np: np.ndarray,   # (N_av, K)
    av_id_to_idx: Dict[int, int],
    path_batch: List[List[List[int]]],  # [N_av][K][edge_indices]
    edge_ids: List[str],               # idx -> edge name
    n_routes: int,
    deterministic: bool = False,
) -> Tuple[int, Dict[int, int]]:
    """Re-route all active AV vehicles via traci.vehicle.setRoute().

    Uses already-computed per-vehicle route probabilities and candidate paths
    from the per-step policy refresh.

    Returns:
        (n_rerouted, reroute_choices) where reroute_choices maps
        agent_int_id -> chosen route index, so that env.step() can reuse
        the same choice instead of sampling independently.
    """
    reroute_choices: Dict[int, int] = {}
    if not TRACI_AVAILABLE or av_vehicle_fracs_np is None or not path_batch:
        return 0, reroute_choices
    try:
        active_ids = {str(v) for v in traci.vehicle.getIDList()}
    except Exception:
        return 0, reroute_choices

    n_rerouted = 0
    n_av = min(len(av_agent_ids), av_vehicle_fracs_np.shape[0], len(path_batch))
    for i in range(n_av):
        aid = av_agent_ids[i]
        veh_name = av_agent_name_map.get(int(aid), "")
        if not veh_name or veh_name not in active_ids:
            continue
        try:
            cur_edge = traci.vehicle.getRoadID(veh_name)
        except Exception:
            continue
        if not cur_edge or cur_edge.startswith(":"):
            continue

        probs = _normalize_prob_vector(av_vehicle_fracs_np[i], n_routes)
        k = int(np.argmax(probs)) if deterministic else int(np.random.choice(n_routes, p=probs))

        chosen = path_batch[i][k] if k < len(path_batch[i]) else path_batch[i][0]
        if not chosen:
            continue
        edge_names = [edge_ids[idx] for idx in chosen if 0 <= idx < len(edge_ids)]
        if not edge_names or edge_names[0] != cur_edge:
            continue
        try:
            traci.vehicle.setRoute(veh_name, edge_names)
            n_rerouted += 1
            reroute_choices[int(aid)] = k
        except Exception:
            pass
    return n_rerouted, reroute_choices


def _mpc_select_action(
    *,
    wm,
    z_current: "torch.Tensor",
    policy,
    obs: "torch.Tensor",             # (N_av, obs_dim) from policy obs
    av_vehicle_fracs_np: np.ndarray,  # (N_av, K) baseline policy output
    path_batch: List[List[List[int]]],
    hdv_edge_np: Optional[np.ndarray],  # (E,) fixed HDV edge dist or None
    horizon: int,
    n_candidates: int,
    discount: float,
    n_routes: int,
    device: "torch.device",
    depart_edge_np: Optional[np.ndarray] = None,
    context_np: Optional[np.ndarray] = None,
    wm_volume_floor: float = 0.0,
) -> np.ndarray:
    """MPC lookahead: sample N candidate actions, imagine H steps, pick best.

    Returns:
        best_fracs: (N_av, K) route fractions from the best candidate.
    """
    import torch
    from dreamer4.urban_routing.rollout_buffer import RolloutBuffer

    n_av = av_vehicle_fracs_np.shape[0]
    n_edges = wm.n_edges
    K = n_routes
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32)

    # Sample N candidate action sets from policy
    candidates = [av_vehicle_fracs_np.copy()]  # include baseline
    with torch.no_grad():
        for _ in range(max(0, n_candidates - 1)):
            fracs, _, _, _, _ = policy.get_action_and_value(obs)
            candidates.append(fracs.cpu().numpy().astype(np.float32))

    best_score = -float("inf")
    best_fracs = candidates[0]

    mpc_buf = RolloutBuffer(
        max_length=2, n_spatial=wm.n_edges,
        d_spatial=wm.d_spatial, device=str(device),
    )

    for cand_fracs in candidates:
        # Convert per-vehicle fracs to edge-level action for WM
        edge_action = np.zeros((n_edges,), dtype=np.float32)
        for i in range(min(n_av, len(path_batch))):
            probs = _normalize_prob_vector(cand_fracs[i], K)
            for k in range(min(K, len(path_batch[i]))):
                path = path_batch[i][k]
                if path:
                    idxs = np.asarray(path, dtype=np.int64)
                    idxs = idxs[(idxs >= 0) & (idxs < n_edges)]
                    if idxs.size > 0:
                        edge_action[idxs] += float(probs[k])
        s = float(edge_action.sum())
        if s > 1e-8:
            edge_action /= s

        # Imagine H steps and accumulate reward
        mpc_buf.buffer.clear()
        mpc_buf.add(z_current)
        z_h = z_current
        total_reward = 0.0
        action_t = torch.from_numpy(edge_action).to(device=device, dtype=torch.float32)
        depart_t = None if depart_edge_np is None else torch.from_numpy(depart_edge_np).to(device=device, dtype=torch.float32)
        context_t = None if context_np is None else torch.from_numpy(context_np).to(device=device, dtype=torch.float32)
        for h in range(horizon):
            z_h = wm.predict_next_state(
                mpc_buf,
                action=action_t,
                depart=depart_t,
                context=context_t,
                skip_stackelberg=True,
            )
            mpc_buf.add(z_h)

            # Decode and compute BPR reward
            decode_fn = getattr(wm.flow_projector, "decode", None)
            if decode_fn is None:
                decode_fn = wm.flow_projector.inverse
            edge_dyn_h = decode_fn(z_h).detach().cpu().numpy().astype(np.float32)
            v_norm_h = np.maximum(edge_dyn_h[:, 2], 0.0)
            if float(wm_volume_floor) > 0.0:
                v_norm_h = np.where(v_norm_h >= float(wm_volume_floor), v_norm_h, 0.0)
            v_h = v_norm_h * capacities
            t_e_h = wm.bpr.numpy_forward(v_h)
            step_tt = float(np.dot(v_h, t_e_h))
            n_veh = max(float(v_h.sum()), 1.0)
            total_reward += (discount ** h) * (-step_tt / n_veh)

        if total_reward > best_score:
            best_score = total_reward
            best_fracs = cand_fracs

    return best_fracs


def _infer_hdv_global_from_models(env, n_routes: int) -> np.ndarray:
    """
    Infer HDV global route fractions from current human policies.
    Priority per human:
      1) default_action (if fixed externally)
      2) model.act(None)
      3) last_action
    """
    counts = np.full(int(n_routes), 1e-6, dtype=np.float32)
    humans = list(getattr(env, "human_agents", []))
    for human in humans:
        action = None
        da = getattr(human, "default_action", None)
        if da is not None:
            try:
                action = int(da)
            except Exception:
                action = None
        if action is None:
            model = getattr(human, "model", None)
            if model is not None and hasattr(model, "act"):
                try:
                    action = int(model.act(None))
                except Exception:
                    action = None
        if action is None:
            try:
                action = int(getattr(human, "last_action"))
            except Exception:
                action = None
        if action is not None and 0 <= int(action) < int(n_routes):
            counts[int(action)] += 1.0
    return _normalize_prob_vector(counts, int(n_routes))


def _infer_hdv_global_from_episode_actions(
    env,
    n_routes: int,
    fallback: np.ndarray,
) -> np.ndarray:
    """
    Infer HDV global route fractions from RouteRL per-episode actions already
    committed in current simulation day (`env.episode_actions`).
    """
    counts = np.zeros(int(n_routes), dtype=np.float32)
    ep_actions = getattr(env, "episode_actions", None)
    if isinstance(ep_actions, dict):
        for _, rec in ep_actions.items():
            if not isinstance(rec, dict):
                continue
            kind = str(rec.get("kind", "")).strip().lower()
            if kind not in {"human", "hdv"}:
                continue
            try:
                a = int(rec.get("action"))
            except Exception:
                continue
            if 0 <= a < int(n_routes):
                counts[a] += 1.0
    if float(counts.sum()) <= 0.0:
        return _normalize_prob_vector(fallback, int(n_routes))
    return _normalize_prob_vector(counts, int(n_routes))


def _infer_hdv_edge_distribution(
    env,
    edge_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Infer HDV edge occupancy distribution p(edge) from current in-network vehicles.

    Returns:
        (E,) float32 vector summing to 1 when HDVs are present; otherwise zeros.
    """
    n_edges = int(len(edge_to_idx))
    out = np.zeros((n_edges,), dtype=np.float32)
    if n_edges <= 0 or (not TRACI_AVAILABLE):
        return out

    human_ids: set[str] = set()
    for h in getattr(env, "human_agents", []):
        try:
            hid = str(getattr(h, "id"))
        except Exception:
            hid = ""
        if hid:
            human_ids.add(hid)

    try:
        veh_ids = traci.vehicle.getIDList()
    except Exception:
        return out

    for vid in veh_ids:
        is_hdv = str(vid) in human_ids
        if not is_hdv:
            try:
                vtype = str(traci.vehicle.getTypeID(vid)).strip().lower()
            except Exception:
                vtype = ""
            if vtype in {"human", "hdv"}:
                is_hdv = True
            elif vtype in {"machine", "av", "cav"}:
                is_hdv = False
        if not is_hdv:
            continue

        try:
            edge_id = str(traci.vehicle.getRoadID(vid))
        except Exception:
            continue
        if (not edge_id) or edge_id.startswith(":"):
            continue
        ei = edge_to_idx.get(edge_id)
        if ei is None:
            continue
        out[int(ei)] += 1.0

    s = float(out.sum())
    if s > 0.0:
        out /= s
    return out


def _collect_hdv_equilibrium_distribution(
    env,
    edge_to_idx: Dict[str, int],
    n_adapt_eps: int = 10,
    n_sample_eps: int = 3,
) -> np.ndarray:
    """
    Collect HDV equilibrium edge distribution by running adaptation episodes.

    Fixes the current AV policy, lets HDV agents (Gawron) adapt over
    n_adapt_eps episodes, then samples the HDV edge distribution over
    n_sample_eps episodes and returns the average.

    Args:
        env: RouteRL PettingZoo environment with human_agents.
        edge_to_idx: edge name → index mapping.
        n_adapt_eps: number of adaptation episodes (HDV-only learning).
        n_sample_eps: number of sampling episodes after adaptation.

    Returns:
        (E,) float32 normalized HDV edge distribution.
    """
    n_edges = len(edge_to_idx)
    if n_edges <= 0 or not TRACI_AVAILABLE:
        return np.zeros((n_edges,), dtype=np.float32)

    # Adaptation: run episodes letting HDV learn (env.step handles Gawron)
    for _ in range(n_adapt_eps):
        try:
            env.reset()
            for agent_id in env.agent_iter():
                obs_env, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action = int(np.random.randint(0, 4))  # random AV action during adaptation
                env.step(action)
        except Exception:
            pass

    # Sampling: collect HDV edge distributions
    samples = []
    for _ in range(n_sample_eps):
        try:
            env.reset()
            ep_dist = np.zeros((n_edges,), dtype=np.float32)
            n_steps = 0
            for agent_id in env.agent_iter():
                obs_env, reward, termination, truncation, info = env.last()
                if termination or truncation:
                    action = None
                else:
                    action = int(np.random.randint(0, 4))
                env.step(action)
                step_dist = _infer_hdv_edge_distribution(env, edge_to_idx)
                if step_dist.sum() > 0:
                    ep_dist += step_dist
                    n_steps += 1
            if n_steps > 0:
                ep_dist /= float(n_steps)
                samples.append(ep_dist)
        except Exception:
            pass

    if not samples:
        return np.zeros((n_edges,), dtype=np.float32)

    avg = np.mean(samples, axis=0).astype(np.float32)
    s = float(avg.sum())
    if s > 0:
        avg /= s
    return avg
