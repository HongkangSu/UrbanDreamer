"""Dreamer-style reward readouts for latent vehicle imagination.

This module keeps reward/cost readout logic out of ``train_marl.py``.  The
functions here are intended to produce scalar rewards that are consumed by the
existing Dreamer lambda-return actor-critic path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from .dreamer_route_objectives import _expected_route_cost_from_paths
except ImportError:
    from dreamer_route_objectives import _expected_route_cost_from_paths


def dreamer_edge_tt_from_decoded(
    *,
    wm,
    decoded: torch.Tensor,
    z: Optional[torch.Tensor],
    edge_cost_source: str,
    decoded_speed_min_ratio: float,
    decoded_speed_max_ratio: float,
    decoded_speed_excess_scale: float = 1.0,
) -> torch.Tensor:
    """Return per-edge travel time from a decoded latent state.

    This mirrors the edge-cost readout used by the planner path, but lives in a
    reward-readout module so new Dreamer rewards do not keep growing the training
    script.
    """
    source = str(edge_cost_source or "decoded_speed").strip().lower()
    if source in {"route_head", "route_cost_head", "planner_head"}:
        if z is None:
            raise ValueError("edge_cost_source='route_head' requires latent state z.")
        lengths = torch.as_tensor(
            np.asarray(
                wm._registry.lengths
                if wm._registry.lengths is not None
                else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
                dtype=np.float32,
            ),
            device=z.device,
            dtype=z.dtype,
        )
        speed_limits = torch.as_tensor(
            np.asarray(wm._registry.speed_limits, dtype=np.float32),
            device=z.device,
            dtype=z.dtype,
        )
        free_flow = lengths / speed_limits.clamp_min(0.1)
        return wm.flow_projector.predict_edge_route_cost(z, free_flow)

    if source in {"decoded_speed", "decoder_speed", "speed"}:
        lengths = torch.as_tensor(
            np.asarray(
                wm._registry.lengths
                if wm._registry.lengths is not None
                else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
                dtype=np.float32,
            ),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        speed_limits = torch.as_tensor(
            np.asarray(wm._registry.speed_limits, dtype=np.float32),
            device=decoded.device,
            dtype=decoded.dtype,
        )
        min_ratio = max(float(decoded_speed_min_ratio), 1e-4)
        max_ratio = max(float(decoded_speed_max_ratio), min_ratio)
        speed_ratio = torch.nan_to_num(
            decoded[:, 1],
            nan=1.0,
            posinf=max_ratio,
            neginf=min_ratio,
        ).clamp(min=min_ratio, max=max_ratio)
        free_flow = lengths / speed_limits.clamp_min(0.1)
        excess_scale = max(float(decoded_speed_excess_scale), 0.0)
        return free_flow * (1.0 + excess_scale * ((1.0 / speed_ratio) - 1.0))

    if source in {"bpr_volume", "volume_bpr", "bpr"}:
        capacities = wm.bpr.capacities.to(device=decoded.device, dtype=decoded.dtype)
        v_total = torch.relu(decoded[:, 2]) * capacities
        return wm.bpr(v_total)

    raise ValueError(
        f"Unsupported edge_cost_source={edge_cost_source!r}; expected "
        "'bpr_volume', 'decoded_speed', or 'route_head'"
    )


def empty_rollout_reward_stats() -> Dict[str, float]:
    return {
        "dreamer_rollout_reward": 0.0,
        "dreamer_rollout_reward_cost": 0.0,
        "dreamer_rollout_reward_pressure_cost": 0.0,
        "dreamer_rollout_reward_cav_proxy_cost": 0.0,
    }


def mean_rollout_reward_stats(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return empty_rollout_reward_stats()
    out: Dict[str, float] = {}
    for key in empty_rollout_reward_stats().keys():
        vals = [float(item.get(key, 0.0)) for item in items if np.isfinite(float(item.get(key, 0.0)))]
        out[key] = float(np.mean(vals)) if vals else 0.0
    return out


def dreamer_rollout_cost_reward(
    *,
    wm,
    z_next: torch.Tensor,
    action_edge: torch.Tensor,
    route_probs: Optional[torch.Tensor],
    path_batch: Optional[List[List[List[int]]]],
    n_routes: int,
    avg_route_steps: float,
    reward_weight: float,
    source: str,
    t_cav_weight: float,
    cost_scale_seconds: float,
    reward_scale: float,
    invalid_route_cost_factor: float,
    edge_cost_source: str,
    decoded_speed_min_ratio: float,
    decoded_speed_max_ratio: float,
    decoded_speed_excess_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a Dreamer reward bonus from rollout cost readouts.

    The returned reward is negative cost and remains differentiable through
    ``z_next`` and soft/straight-through ``action_edge`` when the caller keeps
    the world-model rollout graph.  This is not a GRPO baseline or policy
    surrogate; it is a raw reward term for lambda-return actor-critic.
    """
    weight = float(reward_weight)
    zero = torch.zeros((), device=z_next.device, dtype=z_next.dtype)
    if weight == 0.0:
        return zero, empty_rollout_reward_stats()

    source_key = str(source or "pressure_cav").strip().lower()
    if source_key not in {"pressure", "planned_pressure", "cav_proxy", "pred_t_cav_proxy", "pressure_cav"}:
        raise ValueError(
            f"Unsupported dreamer_rollout_cost_reward_source={source!r}; expected "
            "pressure, cav_proxy, or pressure_cav"
        )

    decoded_next = wm.flow_projector.decode(z_next)
    edge_tt_next = dreamer_edge_tt_from_decoded(
        wm=wm,
        decoded=decoded_next,
        z=z_next,
        edge_cost_source=edge_cost_source,
        decoded_speed_min_ratio=decoded_speed_min_ratio,
        decoded_speed_max_ratio=decoded_speed_max_ratio,
        decoded_speed_excess_scale=decoded_speed_excess_scale,
    )

    pressure_cost = zero
    if source_key in {"pressure", "planned_pressure", "pressure_cav"}:
        pressure = torch.relu(action_edge.to(device=z_next.device, dtype=edge_tt_next.dtype))
        pressure_sum = pressure.sum()
        if bool(torch.isfinite(pressure_sum).detach().cpu().item()) and float(pressure_sum.detach().cpu().item()) > 1e-8:
            pressure_dist = pressure / pressure_sum.clamp_min(1e-8)
            pressure_cost = torch.sum(pressure_dist * edge_tt_next) * float(max(avg_route_steps, 1.0))
            pressure_cost = pressure_cost.to(device=z_next.device, dtype=z_next.dtype)

    cav_proxy_cost = zero
    if (
        source_key in {"cav_proxy", "pred_t_cav_proxy", "pressure_cav"}
        and route_probs is not None
        and path_batch is not None
    ):
        invalid_route_cost = (
            edge_tt_next.mean()
            * float(max(avg_route_steps, 1.0))
            * float(invalid_route_cost_factor)
        )
        cav_val = _expected_route_cost_from_paths(
            route_probs=route_probs,
            path_batch=path_batch,
            edge_cost=edge_tt_next,
            n_routes=int(n_routes),
            invalid_route_cost=invalid_route_cost,
        )
        if cav_val is not None:
            cav_proxy_cost = cav_val.to(device=z_next.device, dtype=z_next.dtype)

    if source_key in {"pressure", "planned_pressure"}:
        total_cost = pressure_cost
    elif source_key in {"cav_proxy", "pred_t_cav_proxy"}:
        total_cost = cav_proxy_cost
    else:
        total_cost = pressure_cost + float(t_cav_weight) * cav_proxy_cost

    scale = max(float(cost_scale_seconds), 1e-6)
    reward = -weight * total_cost / scale * float(reward_scale)
    stats = {
        "dreamer_rollout_reward": float(reward.detach().cpu().item()),
        "dreamer_rollout_reward_cost": float(total_cost.detach().cpu().item()),
        "dreamer_rollout_reward_pressure_cost": float(pressure_cost.detach().cpu().item()),
        "dreamer_rollout_reward_cav_proxy_cost": float(cav_proxy_cost.detach().cpu().item()),
    }
    return reward, stats


def dreamer_rollout_cost_reward_from_config(
    *,
    wm,
    z_next: torch.Tensor,
    action_edge: torch.Tensor,
    route_probs: Optional[torch.Tensor],
    path_batch: Optional[List[List[List[int]]]],
    n_routes: int,
    avg_route_steps: float,
    cfg,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Config adapter used by the training loop."""
    return dreamer_rollout_cost_reward(
        wm=wm,
        z_next=z_next,
        action_edge=action_edge,
        route_probs=route_probs,
        path_batch=path_batch,
        n_routes=int(n_routes),
        avg_route_steps=float(avg_route_steps),
        reward_weight=float(getattr(cfg, "dreamer_rollout_cost_reward_weight", 0.0)),
        source=str(getattr(cfg, "dreamer_rollout_cost_reward_source", "pressure_cav")),
        t_cav_weight=float(getattr(cfg, "dreamer_rollout_cost_reward_t_cav_weight", 0.0)),
        cost_scale_seconds=float(getattr(cfg, "dreamer_rollout_cost_reward_scale_seconds", 60.0)),
        reward_scale=float(getattr(cfg, "vehicle_reward_scale", 1.0)),
        invalid_route_cost_factor=float(getattr(cfg, "vehicle_path_penalty_factor", 3.0)),
        edge_cost_source=str(getattr(cfg, "wm_edge_cost_source", "decoded_speed")),
        decoded_speed_min_ratio=float(getattr(cfg, "wm_decoded_speed_min_ratio", 0.05)),
        decoded_speed_max_ratio=float(getattr(cfg, "wm_decoded_speed_max_ratio", 1.50)),
        decoded_speed_excess_scale=float(getattr(cfg, "wm_decoded_speed_excess_scale", 1.0)),
    )


def rollout_reward_metric_keys() -> Tuple[str, ...]:
    return tuple(empty_rollout_reward_stats().keys())


def rollout_reward_requires_dynamics_grad(cfg) -> bool:
    return float(getattr(cfg, "dreamer_rollout_cost_reward_weight", 0.0)) != 0.0


def add_rollout_reward_metrics(
    target: Dict[str, float],
    stats: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if stats is None:
        stats = empty_rollout_reward_stats()
    for key in rollout_reward_metric_keys():
        target[key] = float(stats.get(key, 0.0))
    return target


def add_rollout_reward_trajectory_fields(
    target: Dict[str, object],
    metrics: Dict[str, float],
) -> Dict[str, object]:
    for key in rollout_reward_metric_keys():
        target[key] = float(metrics.get(key, 0.0))
    return target


def add_rollout_reward_args(parser) -> None:
    parser.add_argument(
        "--dreamer_rollout_cost_reward_weight",
        type=float,
        default=0.0,
        help=(
            "Dreamer actor-critic raw rollout-cost reward weight. Unlike GRPO, "
            "this adds -cost directly to imagined rewards and is optimized "
            "through lambda-return."
        ),
    )
    parser.add_argument(
        "--dreamer_rollout_cost_reward_source",
        type=str,
        default="pressure_cav",
        choices=["pressure", "planned_pressure", "cav_proxy", "pred_t_cav_proxy", "pressure_cav"],
        help="Raw rollout-cost readout used as Dreamer reward: pressure, cav_proxy, or pressure_cav.",
    )
    parser.add_argument(
        "--dreamer_rollout_cost_reward_t_cav_weight",
        type=float,
        default=0.0,
        help="CAV proxy multiplier used when --dreamer_rollout_cost_reward_source=pressure_cav.",
    )
    parser.add_argument(
        "--dreamer_rollout_cost_reward_scale_seconds",
        type=float,
        default=60.0,
        help="Seconds divisor for raw rollout-cost reward before applying the reward weight.",
    )


def append_rollout_reward_stats(stats_history: Dict[str, List[float]], ep_metrics: Dict[str, float]) -> None:
    for key in rollout_reward_metric_keys():
        stats_history[key].append(float(ep_metrics.get(key, 0.0)))


def rollout_reward_diagnostics(ep_metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        key: float(ep_metrics.get(key, 0.0))
        for key in rollout_reward_metric_keys()
    }


def recent_rollout_reward_stats(stats_history: Dict[str, List[float]], log_every: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in rollout_reward_metric_keys():
        values = stats_history.get(key, [])
        out[key] = float(np.mean(values[-int(log_every):])) if values else 0.0
    return out


def format_recent_rollout_reward(stats: Dict[str, float]) -> str:
    return (
        "dreamer_rollout_reward="
        f"{float(stats.get('dreamer_rollout_reward', 0.0)):.4f} "
        "cost="
        f"{float(stats.get('dreamer_rollout_reward_cost', 0.0)):.4f}"
    )
