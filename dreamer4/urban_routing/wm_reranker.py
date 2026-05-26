from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .planner import (
    build_route_plan_edge_distribution,
    estimate_edge_travel_times_from_latent,
    score_route_plan_with_rollout_states,
)
from .rollout_buffer import RolloutBuffer


@dataclass
class WMCandidateMetrics:
    candidate_id: int
    route_indices: np.ndarray
    pred_t_all: float
    pred_t_all_pressure: float
    pred_t_all_head: float
    pred_t_all_decoded: float
    pred_t_cav_proxy: float
    weighted_score: float
    gate_pass: bool
    gate_reason: str
    meta: Dict[str, Any]


def build_local_perturbation_candidates(
    route_probs: np.ndarray,
    *,
    top_k: int,
    baseline_route_indices: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Construct a baseline-centered local candidate set.

    Candidate 0 is the PPO baseline action. Additional candidates are local
    perturbations around the baseline: we identify the least-certain agents
    (smallest gap between top-1 and top-2 route probability) and, one agent at
    a time, swap that agent from its baseline route to its next-best route.

    This is intentionally *not* a global joint top-k search; it is a small,
    PPO-constrained neighborhood around the baseline action.
    """
    probs = np.asarray(route_probs, dtype=np.float32)
    if probs.ndim != 2:
        raise ValueError(f"route_probs must be 2D, got shape={probs.shape}")
    n_agents, n_routes = probs.shape
    if n_agents <= 0 or n_routes <= 0:
        return []
    rank = np.argsort(-probs, axis=1).astype(np.int64)

    if baseline_route_indices is not None:
        baseline = np.asarray(baseline_route_indices, dtype=np.int64).reshape(-1)
        if baseline.shape[0] != n_agents:
            raise ValueError(
                f"baseline_route_indices shape mismatch: got {baseline.shape}, expected {(n_agents,)}"
            )
        baseline = np.clip(baseline, 0, n_routes - 1)
    else:
        baseline = rank[:, 0].copy()

    candidates: List[np.ndarray] = [baseline.copy()]
    if int(top_k) <= 1 or n_routes <= 1:
        return candidates

    top1 = probs[np.arange(n_agents), rank[:, 0]]
    top2 = probs[np.arange(n_agents), rank[:, 1]] if n_routes > 1 else np.zeros((n_agents,), dtype=np.float32)
    uncertainty = top1 - top2
    uncertain_agents = np.argsort(uncertainty).astype(np.int64)

    for agent_idx in uncertain_agents:
        if len(candidates) >= int(top_k):
            break
        fallback_rank = rank[int(agent_idx)]
        replacement = None
        for ridx in fallback_rank:
            if int(ridx) != int(baseline[int(agent_idx)]):
                replacement = int(ridx)
                break
        if replacement is None:
            continue
        cand = baseline.copy()
        cand[int(agent_idx)] = int(replacement)
        if not any(np.array_equal(cand, prev) for prev in candidates):
            candidates.append(cand)
    return candidates


def combine_av_hdv_edge_distributions(
    av_edge: np.ndarray,
    hdv_edge: Optional[np.ndarray],
    *,
    total_av_count: float,
    total_hdv_count: float,
) -> np.ndarray:
    av = np.asarray(av_edge, dtype=np.float32).reshape(-1)
    if hdv_edge is None:
        out = av.copy()
    else:
        hdv = np.asarray(hdv_edge, dtype=np.float32).reshape(-1)
        if hdv.shape != av.shape:
            raise ValueError(f"hdv_edge shape mismatch: got {hdv.shape}, expected {av.shape}")
        out = av * float(total_av_count) + hdv * float(total_hdv_count)
    s = float(out.sum())
    if s > 1e-8:
        out /= s
    return out.astype(np.float32, copy=False)


def _fallback_t_all_proxy(
    wm,
    states: Sequence[np.ndarray | torch.Tensor],
    *,
    joint_edge: np.ndarray,
    discount: float,
    volume_floor: float = 0.0,
    edge_cost_source: str = "decoded_speed",
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> float:
    """Score predicted edge travel times under the candidate's planned pressure.

    This is action-conditioned by construction: the rollout supplies predicted
    edge costs, while ``joint_edge`` supplies the candidate AV plan plus current
    HDV pressure.  It is therefore a better candidate-ranking readout than a
    decoded all-network volume average, which can wash out small but meaningful
    AV rerouting changes.
    """
    if not states:
        return float("inf")
    joint = np.asarray(joint_edge, dtype=np.float32).reshape(-1)
    scores: List[float] = []
    weights: List[float] = []
    for i, z_state in enumerate(states):
        try:
            edge_tt = estimate_edge_travel_times_from_latent(
                wm,
                z_state,
                volume_floor=volume_floor,
                edge_cost_source=edge_cost_source,
                decoded_speed_min_ratio=decoded_speed_min_ratio,
                decoded_speed_max_ratio=decoded_speed_max_ratio,
                decoded_speed_excess_scale=decoded_speed_excess_scale,
            )
            scores.append(float(np.dot(joint, edge_tt)))
            weights.append(float(discount) ** i)
        except Exception:
            continue
    if not scores:
        return float("inf")
    weights_arr = np.asarray(weights, dtype=np.float32)
    return float(np.dot(weights_arr, np.asarray(scores, dtype=np.float32)) / max(float(weights_arr.sum()), 1e-8))


def _decoded_system_t_all_proxy(
    wm,
    states: Sequence[np.ndarray | torch.Tensor],
    *,
    discount: float,
    volume_floor: float = 0.0,
    edge_cost_source: str = "decoded_speed",
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> float:
    """Estimate system travel time from decoded rollout state, in minutes.

    The projector-level t_all head is episode-supervised and can collapse to a
    state prior.  For action comparison we need a readout that is directly tied
    to predicted per-edge speed/volume changes.
    """
    if not states:
        return float("inf")
    capacities = np.asarray(wm._registry.capacities, dtype=np.float32).reshape(-1)
    scores: List[float] = []
    weights: List[float] = []
    for i, z_state in enumerate(states):
        try:
            if isinstance(z_state, np.ndarray):
                z_t = torch.from_numpy(z_state).to(device=wm.device, dtype=torch.float32)
            else:
                z_t = z_state.to(device=wm.device, dtype=torch.float32)
            decoded = wm.flow_projector.decode(z_t).detach().cpu().numpy()
            if decoded.ndim == 3:
                decoded = decoded[0]
            volume_norm = np.clip(np.asarray(decoded[..., 2], dtype=np.float32), 0.0, None)
            volume = np.maximum(volume_norm * capacities, float(volume_floor))
            edge_tt = estimate_edge_travel_times_from_latent(
                wm,
                z_t,
                volume_floor=volume_floor,
                edge_cost_source=edge_cost_source,
                decoded_speed_min_ratio=decoded_speed_min_ratio,
                decoded_speed_max_ratio=decoded_speed_max_ratio,
                decoded_speed_excess_scale=decoded_speed_excess_scale,
            )
            denom = max(float(volume.sum()), 1e-8)
            scores.append(float(np.dot(volume, edge_tt) / denom / 60.0))
            weights.append(float(discount) ** i)
        except Exception:
            continue
    if not scores:
        return float("inf")
    weights_arr = np.asarray(weights, dtype=np.float32)
    return float(np.dot(weights_arr, np.asarray(scores, dtype=np.float32)) / max(float(weights_arr.sum()), 1e-8))


def rollout_candidate_metrics(
    *,
    wm,
    z_current: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_indices: np.ndarray,
    horizon: int,
    discount: float,
    depart_edge_np: Optional[np.ndarray],
    context_np: Optional[np.ndarray],
    hdv_edge_np: Optional[np.ndarray],
    total_av_count: float,
    total_hdv_count: float,
    volume_floor: float = 0.0,
    edge_cost_source: str = "decoded_speed",
    current_edges: Optional[np.ndarray] = None,
    action_horizon_seconds: Optional[float] = None,
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> Dict[str, Any]:
    n_edges = int(wm.n_edges)
    edge_tt_current = None
    if action_horizon_seconds is not None:
        edge_tt_current = estimate_edge_travel_times_from_latent(
            wm,
            z_current,
            volume_floor=volume_floor,
            edge_cost_source=edge_cost_source,
            decoded_speed_min_ratio=decoded_speed_min_ratio,
            decoded_speed_max_ratio=decoded_speed_max_ratio,
            decoded_speed_excess_scale=decoded_speed_excess_scale,
        )
    av_edge = build_route_plan_edge_distribution(
        route_indices,
        path_batch,
        n_edges,
        current_edges=current_edges,
        edge_travel_times=edge_tt_current,
        horizon_seconds=action_horizon_seconds,
    )
    joint_edge = combine_av_hdv_edge_distributions(
        av_edge,
        hdv_edge_np,
        total_av_count=total_av_count,
        total_hdv_count=total_hdv_count,
    )

    device = wm.device
    buf = RolloutBuffer(max_length=2, n_spatial=wm.n_edges, d_spatial=wm.d_spatial, device=str(device))
    buf.add(z_current)
    depart_t = None if depart_edge_np is None else torch.from_numpy(np.asarray(depart_edge_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
    context_t = None if context_np is None else torch.from_numpy(np.asarray(context_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
    # The graph_cond dynamics checkpoint is trained with action_key=
    # av_planned_edge_flow.  HDV edge distributions supervise the auxiliary
    # HDV head, but they are not concatenated into the dynamics action input.
    # Keep joint_edge only for scoring/fallback proxies; feeding it to dynamics
    # would both normalize away the planned-flow scale and mismatch training.
    action_t = torch.from_numpy(av_edge).to(device=device, dtype=torch.float32)

    states: List[torch.Tensor] = []
    t_all_head_preds: List[float] = []
    for _ in range(max(int(horizon), 1)):
        z_next = wm.predict_next_state(
            buf,
            action=action_t,
            depart=depart_t,
            context=context_t,
            skip_stackelberg=True,
        )
        buf.add(z_next)
        states.append(z_next.detach().cpu())
        t_pred = wm.predict_t_all_from_latent(z_next)
        if t_pred is not None and np.isfinite(t_pred):
            t_all_head_preds.append(float(t_pred))

    pred_t_cav_proxy = score_route_plan_with_rollout_states(
        wm,
        states,
        path_batch,
        np.asarray(route_indices, dtype=np.int64),
        volume_floor=volume_floor,
        edge_cost_source=edge_cost_source,
        decoded_speed_min_ratio=decoded_speed_min_ratio,
        decoded_speed_max_ratio=decoded_speed_max_ratio,
        decoded_speed_excess_scale=decoded_speed_excess_scale,
    )
    pred_t_all_head = float("nan")
    if t_all_head_preds:
        weights = np.array([float(discount) ** i for i in range(len(t_all_head_preds))], dtype=np.float32)
        pred_t_all_head = float(
            np.dot(weights, np.asarray(t_all_head_preds, dtype=np.float32)) / max(float(weights.sum()), 1e-8)
        )
    pred_t_all_pressure = _fallback_t_all_proxy(
        wm,
        states,
        joint_edge=joint_edge,
        discount=float(discount),
        volume_floor=volume_floor,
        edge_cost_source=edge_cost_source,
        decoded_speed_min_ratio=decoded_speed_min_ratio,
        decoded_speed_max_ratio=decoded_speed_max_ratio,
        decoded_speed_excess_scale=decoded_speed_excess_scale,
    )
    pred_t_all_decoded = _decoded_system_t_all_proxy(
        wm,
        states,
        discount=float(discount),
        volume_floor=volume_floor,
        edge_cost_source=edge_cost_source,
        decoded_speed_min_ratio=decoded_speed_min_ratio,
        decoded_speed_max_ratio=decoded_speed_max_ratio,
        decoded_speed_excess_scale=decoded_speed_excess_scale,
    )
    if np.isfinite(pred_t_all_pressure):
        pred_t_all = pred_t_all_pressure
    elif np.isfinite(pred_t_all_decoded):
        pred_t_all = pred_t_all_decoded
    else:
        pred_t_all = pred_t_all_head

    return {
        "route_indices": np.asarray(route_indices, dtype=np.int64),
        "av_edge": av_edge,
        "joint_edge": joint_edge,
        "pred_t_all": float(pred_t_all),
        "pred_t_all_pressure": float(pred_t_all_pressure),
        "pred_t_all_head": float(pred_t_all_head),
        "pred_t_all_decoded": float(pred_t_all_decoded),
        "pred_t_cav_proxy": float(pred_t_cav_proxy),
        "t_all_preds": t_all_head_preds,
    }


def select_candidate_from_metrics(
    candidate_metrics: Sequence[Dict[str, Any]],
    *,
    t_all_weight: float,
    t_cav_weight: float,
    cav_tolerance: float = 0.0,
) -> Dict[str, Any]:
    if not candidate_metrics:
        raise ValueError("candidate_metrics must be non-empty")

    baseline = candidate_metrics[0]
    base_t_cav_proxy = float(baseline.get("pred_t_cav_proxy", float("inf")))
    best_idx = 0
    best_score = float("inf")
    decisions: List[WMCandidateMetrics] = []

    for idx, cand in enumerate(candidate_metrics):
        pred_t_all = float(cand.get("pred_t_all", float("inf")))
        pred_t_all_pressure = float(cand.get("pred_t_all_pressure", float("nan")))
        pred_t_all_head = float(cand.get("pred_t_all_head", float("nan")))
        pred_t_all_decoded = float(cand.get("pred_t_all_decoded", float("nan")))
        pred_t_cav_proxy = float(cand.get("pred_t_cav_proxy", float("inf")))
        weighted_score = float(t_all_weight) * pred_t_all + float(t_cav_weight) * pred_t_cav_proxy
        if idx == 0:
            gate_pass = True
            gate_reason = "baseline"
        elif pred_t_cav_proxy <= base_t_cav_proxy + float(cav_tolerance):
            gate_pass = True
            gate_reason = "pass"
        else:
            gate_pass = False
            gate_reason = f"pred_t_CAV_proxy {pred_t_cav_proxy:.4f} > baseline {base_t_cav_proxy:.4f}"

        decisions.append(
            WMCandidateMetrics(
                candidate_id=idx,
                route_indices=np.asarray(cand["route_indices"], dtype=np.int64),
                pred_t_all=pred_t_all,
                pred_t_all_pressure=pred_t_all_pressure,
                pred_t_all_head=pred_t_all_head,
                pred_t_all_decoded=pred_t_all_decoded,
                pred_t_cav_proxy=pred_t_cav_proxy,
                weighted_score=weighted_score,
                gate_pass=gate_pass,
                gate_reason=gate_reason,
                meta={k: v for k, v in cand.items() if k not in {"route_indices"}},
            )
        )
        if gate_pass and weighted_score < best_score:
            best_score = weighted_score
            best_idx = idx

    return {
        "selected_idx": int(best_idx),
        "baseline_idx": 0,
        "baseline_t_cav_proxy": base_t_cav_proxy,
        "decisions": decisions,
        "selected": decisions[best_idx],
    }


def rerank_topk_candidates(
    *,
    wm,
    z_current: torch.Tensor,
    route_probs: np.ndarray,
    baseline_route_indices: np.ndarray,
    path_batch: List[List[List[int]]],
    horizon: int,
    top_k: int,
    discount: float,
    depart_edge_np: Optional[np.ndarray],
    context_np: Optional[np.ndarray],
    hdv_edge_np: Optional[np.ndarray],
    total_av_count: float,
    total_hdv_count: float,
    t_all_weight: float,
    t_cav_weight: float,
    cav_tolerance: float = 0.0,
    volume_floor: float = 0.0,
    edge_cost_source: str = "decoded_speed",
    current_edges: Optional[np.ndarray] = None,
    action_horizon_seconds: Optional[float] = None,
    decoded_speed_min_ratio: float = 0.05,
    decoded_speed_max_ratio: float = 1.50,
    decoded_speed_excess_scale: float = 1.0,
) -> Dict[str, Any]:
    candidates = build_local_perturbation_candidates(
        route_probs,
        top_k=top_k,
        baseline_route_indices=baseline_route_indices,
    )
    metrics = [
        rollout_candidate_metrics(
            wm=wm,
            z_current=z_current,
            path_batch=path_batch,
            route_indices=route_indices,
            horizon=horizon,
            discount=discount,
            depart_edge_np=depart_edge_np,
            context_np=context_np,
            hdv_edge_np=hdv_edge_np,
            total_av_count=total_av_count,
            total_hdv_count=total_hdv_count,
            volume_floor=volume_floor,
            edge_cost_source=edge_cost_source,
            current_edges=current_edges,
            action_horizon_seconds=action_horizon_seconds,
            decoded_speed_min_ratio=decoded_speed_min_ratio,
            decoded_speed_max_ratio=decoded_speed_max_ratio,
            decoded_speed_excess_scale=decoded_speed_excess_scale,
        )
        for route_indices in candidates
    ]
    return select_candidate_from_metrics(
        metrics,
        t_all_weight=t_all_weight,
        t_cav_weight=t_cav_weight,
        cav_tolerance=cav_tolerance,
    )
