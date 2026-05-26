"""
Planner abstractions for world-model-guided routing.

Vehicle-mode planners operate on departure-time static route candidates. They
may post-process policy route probabilities before one fixed route index is
sampled for each vehicle. The resulting route plan is then used consistently by
both imagined rollouts and external SUMO evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import torch


def choose_vehicle_next_edge(current_edge: int, chosen_path: Sequence[int]) -> int:
    """Convert a path into the next-hop edge for one-step vehicle control."""
    if not chosen_path:
        return int(current_edge)
    if int(chosen_path[0]) == int(current_edge):
        if len(chosen_path) >= 2:
            return int(chosen_path[1])
        return int(chosen_path[0])
    return int(chosen_path[0])


def sample_route_indices(route_probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one route index per vehicle from route probabilities."""
    if route_probs.ndim != 2:
        raise ValueError(f"route_probs must be 2D, got shape={route_probs.shape}")
    n_agents, n_routes = route_probs.shape
    probs = np.clip(route_probs.astype(np.float32, copy=False), 1e-8, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    uniform = rng.random((n_agents,))
    cum_probs = np.cumsum(probs, axis=1)
    route_idx = (uniform[:, None] >= cum_probs).sum(axis=1).astype(np.int64)
    return np.clip(route_idx, 0, n_routes - 1)


def normalize_route_probs(route_probs: np.ndarray) -> np.ndarray:
    """Normalize route probabilities row-wise with numerical guards."""
    probs = np.asarray(route_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"route_probs must be 2D, got shape={probs.shape}")
    probs = np.clip(probs, 1e-8, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32, copy=False)


def compute_route_costs_from_edge_times(
    path_batch: List[List[List[int]]],
    edge_travel_times: np.ndarray,
) -> np.ndarray:
    """Sum edge travel times along each candidate route."""
    edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
    if edge_tt.size <= 0:
        raise ValueError("edge_travel_times must be non-empty")
    fallback = float(np.mean(edge_tt))
    n_agents = len(path_batch)
    n_routes = len(path_batch[0]) if n_agents > 0 else 0
    route_costs = np.full((n_agents, n_routes), fallback, dtype=np.float32)
    for i, paths in enumerate(path_batch):
        for k, path in enumerate(paths):
            if not path:
                continue
            idxs = np.asarray(path, dtype=np.int64)
            idxs = idxs[(idxs >= 0) & (idxs < edge_tt.shape[0])]
            if idxs.size > 0:
                route_costs[i, k] = float(edge_tt[idxs].sum())
    return route_costs


def rerank_route_probs_with_costs(
    route_probs: np.ndarray,
    route_costs: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Apply a normalized route-cost penalty on top of policy priors."""
    prior = normalize_route_probs(route_probs)
    costs = np.asarray(route_costs, dtype=np.float32)
    if costs.shape != prior.shape:
        raise ValueError(f"route_costs shape mismatch: got {costs.shape}, expected {prior.shape}")

    best = costs.min(axis=1, keepdims=True)
    gap = np.maximum(costs - best, 0.0)
    scale = np.std(costs, axis=1, keepdims=True)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    cost_penalty = gap / scale

    logits = np.log(np.clip(prior, 1e-8, None)) - cost_penalty
    temp = max(float(temperature), 1e-3)
    logits = logits / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    reranked = np.exp(logits).astype(np.float32)
    reranked = reranked / reranked.sum(axis=1, keepdims=True)
    return reranked.astype(np.float32, copy=False)


def build_route_plan_edge_distribution(
    route_indices: np.ndarray,
    path_batch: List[List[List[int]]],
    n_edges: int,
    *,
    current_edges: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
    edge_travel_times: np.ndarray | None = None,
    horizon_seconds: float | None = None,
) -> np.ndarray:
    """
    Convert fixed route choices into the planned edge-flow action used by WM.

    When edge travel times and a horizon are provided, only edges expected to be
    entered within the next decision interval are counted.  Counts are averaged
    by active vehicle, matching the pre-advance collection action semantics.
    """
    out = np.zeros((int(n_edges),), dtype=np.float32)
    if route_indices.size == 0 or not path_batch or int(n_edges) <= 0:
        return out

    edge_tt = None
    use_horizon = edge_travel_times is not None and horizon_seconds is not None
    if use_horizon:
        edge_tt = np.asarray(edge_travel_times, dtype=np.float32).reshape(-1)
        if edge_tt.size < int(n_edges):
            use_horizon = False
    horizon = max(float(horizon_seconds or 0.0), 1.0)
    current_arr = None if current_edges is None else np.asarray(current_edges, dtype=np.int64).reshape(-1)
    active_arr = None if active_mask is None else np.asarray(active_mask, dtype=bool).reshape(-1)
    n_controlled = 0

    for i in range(min(int(route_indices.shape[0]), len(path_batch))):
        if active_arr is not None and i < active_arr.shape[0] and not bool(active_arr[i]):
            continue
        paths = path_batch[i]
        if not paths:
            continue
        ridx = int(np.clip(int(route_indices[i]), 0, len(paths) - 1))
        path = paths[ridx]
        if not path:
            continue
        idxs = np.asarray(path, dtype=np.int64)
        idxs = idxs[(idxs >= 0) & (idxs < int(n_edges))]
        if idxs.size == 0:
            continue
        n_controlled += 1
        if not use_horizon:
            out[idxs] += 1.0
            continue

        start_pos = 0
        current_edge = None
        if current_arr is not None and i < current_arr.shape[0]:
            current_edge = int(current_arr[i])
            matches = np.flatnonzero(idxs == current_edge)
            if matches.size > 0:
                start_pos = int(matches[0])

        eta = 0.0
        for pos in range(start_pos, int(idxs.size)):
            edge_idx = int(idxs[pos])
            if current_edge is not None and pos == start_pos and edge_idx == current_edge:
                eta += float(edge_tt[edge_idx])
                continue
            if eta >= horizon:
                break
            out[edge_idx] += 1.0
            eta += float(edge_tt[edge_idx])

    if n_controlled > 0:
        out /= float(n_controlled)
    return out.astype(np.float32)


def estimate_edge_travel_times_from_latent(
    wm,
    z: np.ndarray | torch.Tensor,
    *,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    volume_floor: float = 0.0,
    edge_cost_source: str = "decoded_speed",
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> np.ndarray:
    """
    Decode one latent state and estimate per-edge travel times with the same
    speed-decoded surrogate used elsewhere in imagined rollouts.

    This is the critical fallback path for inference-time MPC when the
    episode-level `t_all_head` is unavailable: the planner still scores
    candidates using future latent states predicted by the world model rather
    than reverting to a purely static one-step heuristic.
    """
    if isinstance(z, np.ndarray):
        z_t = torch.from_numpy(z).to(device=wm.device, dtype=torch.float32)
    else:
        z_t = z.to(device=wm.device, dtype=torch.float32)
    if z_t.dim() != 2:
        raise ValueError(f"latent z must be (E, D), got shape={tuple(z_t.shape)}")

    with torch.inference_mode():
        edge_dyn = wm.flow_projector.decode(z_t).detach().cpu().numpy().astype(np.float32)

    if edge_dyn.ndim != 2 or edge_dyn.shape[0] != int(wm.n_edges) or edge_dyn.shape[1] < 3:
        raise ValueError(
            "flow_projector.decode(z) must return (E, >=3) dynamic features, "
            f"got shape={tuple(edge_dyn.shape)}"
        )

    lengths = np.asarray(
        wm._registry.lengths
        if getattr(wm._registry, "lengths", None) is not None
        else np.ones((int(wm.n_edges),), dtype=np.float32) * 100.0,
        dtype=np.float32,
    )
    speed_limits = np.asarray(wm._registry.speed_limits, dtype=np.float32)
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32)
    free_flow = lengths / np.maximum(speed_limits, 0.1)
    source = str(edge_cost_source or "decoded_speed").strip().lower()
    if source in {"route_head", "route_cost_head", "planner_head"}:
        free_flow_t = torch.as_tensor(free_flow, device=z_t.device, dtype=z_t.dtype)
        with torch.inference_mode():
            travel_time_t = wm.flow_projector.predict_edge_route_cost(z_t, free_flow_t)
        return travel_time_t.detach().cpu().numpy().astype(np.float32)

    if source in {"decoded_speed", "decoder_speed", "speed"}:
        min_ratio = max(float(decoded_speed_min_ratio), 1e-4)
        max_ratio = max(float(decoded_speed_max_ratio), min_ratio)
        speed_ratio = np.nan_to_num(
            edge_dyn[:, 1],
            nan=1.0,
            posinf=max_ratio,
            neginf=min_ratio,
        ).astype(np.float32)
        speed_ratio = np.clip(speed_ratio, min_ratio, max_ratio)
        excess_scale = max(float(decoded_speed_excess_scale), 0.0)
        travel_time = free_flow * (1.0 + excess_scale * ((1.0 / speed_ratio) - 1.0))
        return travel_time.astype(np.float32, copy=False)

    if source not in {"bpr_volume", "volume_bpr", "bpr"}:
        raise ValueError(
            f"Unsupported edge_cost_source={edge_cost_source!r}; "
            "expected 'bpr_volume', 'decoded_speed', or 'route_head'"
        )

    norm_volume = np.maximum(edge_dyn[:, 2], 0.0)
    if float(volume_floor) > 0.0:
        norm_volume = np.where(norm_volume >= float(volume_floor), norm_volume, 0.0)
    veh_volume = norm_volume * np.maximum(capacities, 1.0)
    travel_time = free_flow * (
        1.0
        + float(bpr_alpha)
        * (veh_volume / np.maximum(capacities, 1.0)) ** float(bpr_beta)
    )
    return travel_time.astype(np.float32, copy=False)


def score_route_plan_with_rollout_states(
    wm,
    states: Sequence[np.ndarray | torch.Tensor],
    path_batch: List[List[List[int]]],
    route_indices: np.ndarray,
    *,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    volume_floor: float = 0.0,
    edge_cost_source: str = "decoded_speed",
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> float:
    """
    Score one fixed route plan by decoding each rollout latent state into edge
    travel times and averaging the selected-route costs over the horizon.
    """
    if not states:
        raise ValueError("states must be non-empty")

    route_indices = np.asarray(route_indices, dtype=np.int64).reshape(-1)
    n_agents = route_indices.shape[0]
    row = np.arange(n_agents, dtype=np.int64)
    horizon_scores: List[float] = []

    for z_state in states:
        edge_tt = estimate_edge_travel_times_from_latent(
            wm,
            z_state,
            bpr_alpha=bpr_alpha,
            bpr_beta=bpr_beta,
            volume_floor=volume_floor,
            edge_cost_source=edge_cost_source,
            decoded_speed_min_ratio=decoded_speed_min_ratio,
            decoded_speed_max_ratio=decoded_speed_max_ratio,
            decoded_speed_excess_scale=decoded_speed_excess_scale,
        )
        route_costs = compute_route_costs_from_edge_times(path_batch, edge_tt)
        if route_costs.shape[0] != n_agents:
            raise ValueError(
                "route_costs / route_indices mismatch: "
                f"got {route_costs.shape[0]} agents vs {n_agents} indices"
            )
        selected_cost = route_costs[row, np.clip(route_indices, 0, route_costs.shape[1] - 1)]
        horizon_scores.append(float(np.mean(selected_cost)))

    return float(np.mean(horizon_scores)) if horizon_scores else float("inf")


@dataclass
class VehiclePlannerContext:
    route_probs: np.ndarray
    path_batch: List[List[List[int]]]
    current_edges: np.ndarray
    done: np.ndarray
    n_edges: int
    rng: np.random.Generator
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VehiclePlannerDecision:
    route_probs: np.ndarray
    route_indices: np.ndarray
    chosen_edges: np.ndarray
    action_edge: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)


class VehicleActionPlanner:
    """Base interface for vehicle-level planners."""

    def __init__(
        self,
        *,
        mode: str,
        n_candidates: int = 8,
        horizon: int = 3,
        temperature: float = 1.0,
    ):
        self.mode = str(mode).lower()
        self.n_candidates = int(n_candidates)
        self.horizon = int(horizon)
        self.temperature = float(temperature)

    def plan_step(self, ctx: VehiclePlannerContext) -> VehiclePlannerDecision:
        raise NotImplementedError

    def _finalize_decision(
        self,
        ctx: VehiclePlannerContext,
        route_probs: np.ndarray,
        *,
        route_indices_override: np.ndarray | None = None,
        info_extra: Dict[str, Any] | None = None,
    ) -> VehiclePlannerDecision:
        route_probs = normalize_route_probs(route_probs)
        if route_indices_override is None:
            route_indices = sample_route_indices(route_probs, ctx.rng)
        else:
            route_indices = np.asarray(route_indices_override, dtype=np.int64).reshape(-1)
            if route_indices.shape[0] != route_probs.shape[0]:
                raise ValueError(
                    "route_indices_override shape mismatch: "
                    f"got {route_indices.shape}, expected {(route_probs.shape[0],)}"
                )
            route_indices = np.clip(route_indices, 0, route_probs.shape[1] - 1)
        n_agents = route_probs.shape[0]
        chosen_edges = np.full((n_agents,), -1, dtype=np.int64)

        active_mask = ~np.asarray(ctx.done, dtype=bool)
        for i in range(n_agents):
            if not active_mask[i]:
                continue
            chosen_path = ctx.path_batch[i][int(route_indices[i])]
            chosen_edges[i] = choose_vehicle_next_edge(int(ctx.current_edges[i]), chosen_path)

        action_edge = build_route_plan_edge_distribution(
            route_indices=route_indices,
            path_batch=ctx.path_batch,
            n_edges=ctx.n_edges,
            current_edges=ctx.current_edges,
            active_mask=active_mask,
            edge_travel_times=ctx.info.get("edge_travel_times"),
            horizon_seconds=ctx.info.get("action_horizon_seconds", ctx.info.get("step_interval")),
        )

        info = {
            "planner_mode": self.mode,
            "planner_candidates": self.n_candidates,
            "planner_horizon": self.horizon,
            "active_vehicles": int(active_mask.sum()),
        }
        if info_extra:
            info.update(info_extra)

        return VehiclePlannerDecision(
            route_probs=route_probs,
            route_indices=route_indices,
            chosen_edges=chosen_edges,
            action_edge=action_edge,
            info=info,
        )


class IdentityVehiclePlanner(VehicleActionPlanner):
    """Planner that preserves the legacy policy-driven sampling behavior."""

    def __init__(self, **kwargs: Any):
        super().__init__(mode="none", **kwargs)

    def plan_step(self, ctx: VehiclePlannerContext) -> VehiclePlannerDecision:
        route_probs = np.asarray(ctx.route_probs, dtype=np.float32)
        return self._finalize_decision(
            ctx,
            route_probs,
            info_extra={
                "planner_entropy": float(
                    np.mean(-(route_probs * np.log(np.clip(route_probs, 1e-8, None))).sum(axis=1))
                ),
            },
        )


class RerankVehiclePlanner(VehicleActionPlanner):
    """
    Re-rank departure-time route candidates with world-model-derived route cost.

    Expected context info:
      - edge_travel_times: (E,) current per-edge travel time estimate
        or
      - route_costs: (N, K) precomputed route costs

    The planner keeps the policy prior and applies a normalized cost penalty
    before sampling one fixed route index per vehicle.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(mode="rerank", **kwargs)

    def plan_step(self, ctx: VehiclePlannerContext) -> VehiclePlannerDecision:
        prior = normalize_route_probs(ctx.route_probs)

        route_costs = ctx.info.get("route_costs")
        if route_costs is None:
            edge_tt = ctx.info.get("edge_travel_times")
            if edge_tt is None:
                raise ValueError(
                    "RerankVehiclePlanner requires ctx.info['edge_travel_times'] "
                    "or ctx.info['route_costs']."
                )
            route_costs = compute_route_costs_from_edge_times(ctx.path_batch, edge_tt)
        route_costs = np.asarray(route_costs, dtype=np.float32)
        reranked = rerank_route_probs_with_costs(
            prior,
            route_costs,
            temperature=self.temperature,
        )

        prior_entropy = -(prior * np.log(np.clip(prior, 1e-8, None))).sum(axis=1)
        rerank_entropy = -(reranked * np.log(np.clip(reranked, 1e-8, None))).sum(axis=1)
        gap = np.maximum(route_costs - route_costs.min(axis=1, keepdims=True), 0.0)
        greedy_prior = np.argmax(prior, axis=1)
        greedy_rerank = np.argmin(route_costs, axis=1)
        changed = float(np.mean((greedy_prior != greedy_rerank).astype(np.float32))) if prior.shape[0] > 0 else 0.0

        return self._finalize_decision(
            ctx,
            reranked,
            info_extra={
                "planner_entropy": float(np.mean(rerank_entropy)) if rerank_entropy.size > 0 else 0.0,
                "planner_prior_entropy": float(np.mean(prior_entropy)) if prior_entropy.size > 0 else 0.0,
                "planner_route_cost_mean": float(np.mean(route_costs)) if route_costs.size > 0 else float("nan"),
                "planner_route_gap_mean": float(np.mean(gap)) if gap.size > 0 else 0.0,
                "planner_prior_argmax_change_rate": changed,
            },
        )


class MPCVehiclePlanner(VehicleActionPlanner):
    """
    Sample a small set of complete route-plan candidates and choose the one with
    the lowest short-horizon world-model rollout score.

    Expected context info:
      - wm: UrbanWorldModel
      - history_buffer: RolloutBuffer containing the current latent history
      - edge_travel_times or route_costs (optional, used to build a stronger proposal)
    """

    def __init__(self, **kwargs: Any):
        super().__init__(mode="mpc", **kwargs)

    def plan_step(self, ctx: VehiclePlannerContext) -> VehiclePlannerDecision:
        wm = ctx.info.get("wm")
        history_buffer = ctx.info.get("history_buffer")
        if wm is None or history_buffer is None:
            raise ValueError(
                "MPCVehiclePlanner requires ctx.info['wm'] and ctx.info['history_buffer']."
            )

        try:
            try:
                from .planner_rollout import rollout_action_edge_sequence
            except ImportError:
                from planner_rollout import rollout_action_edge_sequence
        except Exception as exc:
            raise RuntimeError(f"Failed to import planner_rollout helper: {exc}") from exc

        prior = normalize_route_probs(ctx.route_probs)

        route_costs = ctx.info.get("route_costs")
        if route_costs is None and ctx.info.get("edge_travel_times") is not None:
            route_costs = compute_route_costs_from_edge_times(
                ctx.path_batch,
                ctx.info["edge_travel_times"],
            )
        route_costs_arr = (
            np.asarray(route_costs, dtype=np.float32)
            if route_costs is not None else None
        )
        proposal_probs = (
            rerank_route_probs_with_costs(prior, route_costs_arr, temperature=self.temperature)
            if route_costs_arr is not None else prior
        )

        n_agents, n_routes = prior.shape
        n_candidates = max(1, int(self.n_candidates))
        horizon = max(1, int(self.horizon))

        candidate_indices: List[np.ndarray] = []
        candidate_labels: List[str] = []

        candidate_indices.append(np.argmax(prior, axis=1).astype(np.int64))
        candidate_labels.append("greedy_prior")

        if route_costs_arr is not None:
            candidate_indices.append(np.argmin(route_costs_arr, axis=1).astype(np.int64))
            candidate_labels.append("greedy_cost")

        while len(candidate_indices) < n_candidates:
            sampled = sample_route_indices(proposal_probs, ctx.rng).astype(np.int64)
            candidate_indices.append(sampled)
            candidate_labels.append(f"sample_{len(candidate_indices) - 1:02d}")

        best_idx = 0
        best_score = float("inf")
        candidate_scores: List[float] = []
        candidate_score_sources: List[str] = []
        bpr_alpha = float(ctx.info.get("bpr_alpha", 0.15))
        bpr_beta = float(ctx.info.get("bpr_beta", 4.0))
        volume_floor = float(ctx.info.get("wm_volume_floor", ctx.info.get("volume_floor", 0.0)))

        for cand_idx, route_indices in enumerate(candidate_indices):
            action_edge = build_route_plan_edge_distribution(
                route_indices,
                ctx.path_batch,
                ctx.n_edges,
                current_edges=ctx.current_edges,
                active_mask=active_mask,
                edge_travel_times=ctx.info.get("edge_travel_times"),
                horizon_seconds=ctx.info.get("action_horizon_seconds", ctx.info.get("step_interval")),
            )
            rollout = rollout_action_edge_sequence(
                wm,
                history_buffer,
                [action_edge for _ in range(horizon)],
                clone_buffer=True,
            )
            if rollout.t_all_preds:
                score = float(rollout.t_all_preds[-1])
                score_source = "t_all_pred"
            elif rollout.states:
                try:
                    score = score_route_plan_with_rollout_states(
                        wm,
                        rollout.states,
                        ctx.path_batch,
                        route_indices,
                        bpr_alpha=bpr_alpha,
                        bpr_beta=bpr_beta,
                        volume_floor=volume_floor,
                        edge_cost_source=str(ctx.info.get("wm_edge_cost_source", "decoded_speed")),
                    )
                    score_source = "wm_rollout_route_cost"
                except Exception:
                    if route_costs_arr is not None and route_costs_arr.size > 0:
                        row = np.arange(n_agents, dtype=np.int64)
                        score = float(np.mean(route_costs_arr[row, route_indices]))
                        score_source = "current_route_cost_after_wm_error"
                    else:
                        score = float(cand_idx)
                        score_source = "candidate_index_after_wm_error"
            elif route_costs_arr is not None and route_costs_arr.size > 0:
                row = np.arange(n_agents, dtype=np.int64)
                score = float(np.mean(route_costs_arr[row, route_indices]))
                score_source = "current_route_cost"
            else:
                score = float(cand_idx)
                score_source = "candidate_index_fallback"
            candidate_scores.append(score)
            candidate_score_sources.append(score_source)
            if score < best_score:
                best_score = score
                best_idx = cand_idx

        chosen_route_indices = candidate_indices[best_idx]
        planner_entropy = -(proposal_probs * np.log(np.clip(proposal_probs, 1e-8, None))).sum(axis=1)
        return self._finalize_decision(
            ctx,
            proposal_probs,
            route_indices_override=chosen_route_indices,
            info_extra={
                "planner_entropy": float(np.mean(planner_entropy)) if planner_entropy.size > 0 else 0.0,
                "planner_candidate_scores": [float(x) for x in candidate_scores],
                "planner_candidate_score_sources": candidate_score_sources,
                "planner_selected_candidate": int(best_idx),
                "planner_selected_label": candidate_labels[best_idx],
                "planner_selected_score": float(best_score),
                "planner_selected_score_source": candidate_score_sources[best_idx],
                "planner_num_candidates": int(len(candidate_indices)),
            },
        )


def build_vehicle_action_planner(
    mode: str = "none",
    *,
    n_candidates: int = 8,
    horizon: int = 3,
    temperature: float = 1.0,
) -> VehicleActionPlanner:
    """
    Planner factory. Future specs will add rerank/MPC implementations here.
    """
    mode_norm = str(mode).strip().lower()
    kwargs = {
        "n_candidates": n_candidates,
        "horizon": horizon,
        "temperature": temperature,
    }
    if mode_norm in {"", "none", "identity"}:
        return IdentityVehiclePlanner(**kwargs)
    if mode_norm == "rerank":
        return RerankVehiclePlanner(**kwargs)
    if mode_norm == "mpc":
        return MPCVehiclePlanner(**kwargs)
    raise ValueError(
        f"Unknown planner_mode={mode!r}. Expected one of ['none', 'identity', 'rerank', 'mpc'] "
        "at the current planner stage."
    )
