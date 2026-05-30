"""Policy-gradient helpers for Dreamer vehicle-route actor training.

This module keeps discrete actor-critic loss construction out of
``train_marl.py``.  It is intentionally small: the training loop owns rollout
generation, while this file owns packing sampled route decisions and computing
the score-function actor loss from Dreamer lambda-return advantages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


DISCRETE_ACTOR_CRITIC_OBJECTIVES = {
    "discrete_actor_critic",
    "discrete_ac",
    "score_function",
    "reinforce",
}


def actor_objective_key(cfg) -> str:
    return str(getattr(cfg, "dreamer_actor_objective", "direct_return") or "direct_return").strip().lower()


def uses_discrete_actor_critic(cfg) -> bool:
    return actor_objective_key(cfg) in DISCRETE_ACTOR_CRITIC_OBJECTIVES


def supports_actor_objective(name: str) -> bool:
    key = str(name or "").strip().lower()
    return key in {"direct_return", "critic_q", "direct_plus_q"} | DISCRETE_ACTOR_CRITIC_OBJECTIVES


def _masked_log_probs(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if valid.shape != logits.shape:
        raise ValueError(f"valid shape {tuple(valid.shape)} must match logits {tuple(logits.shape)}")
    large = torch.finfo(logits.dtype).max / 4.0
    safe_valid = valid.to(device=logits.device).clone()
    has_valid = safe_valid.any(dim=-1)
    if not bool(has_valid.all().detach().cpu().item()):
        safe_valid[~has_valid, 0] = True
    masked_logits = (logits / float(max(temperature, 1e-6))).masked_fill(~safe_valid, -large)
    return torch.log_softmax(masked_logits, dim=-1)


def _policy_distribution_stats(log_probs: torch.Tensor, sample_weight: torch.Tensor) -> Dict[str, float]:
    if int(log_probs.numel()) <= 0:
        return {
            "top1_prob_mean": 0.0,
            "argmax_nonzero_frac": 0.0,
            "argmax_route0_frac": 0.0,
            "argmax_route1_frac": 0.0,
            "argmax_route2_frac": 0.0,
            "argmax_route3_frac": 0.0,
            "argmax_max_route_frac": 0.0,
        }
    probs = torch.exp(log_probs.detach())
    argmax = torch.argmax(log_probs.detach(), dim=-1)
    if torch.is_tensor(sample_weight) and int(sample_weight.numel()) == int(argmax.numel()):
        weights = sample_weight.detach().to(device=log_probs.device, dtype=log_probs.dtype).clamp_min(0.0)
    else:
        weights = torch.ones((int(argmax.numel()),), device=log_probs.device, dtype=log_probs.dtype)
    weight_sum = weights.sum().clamp_min(1e-6)
    route_fracs: List[float] = []
    for route_idx in range(4):
        if int(log_probs.shape[1]) > route_idx:
            frac = ((argmax == route_idx).to(dtype=weights.dtype) * weights).sum() / weight_sum
            route_fracs.append(float(frac.detach().cpu().item()))
        else:
            route_fracs.append(0.0)
    top1 = (probs.max(dim=-1).values * weights).sum() / weight_sum
    nonzero = ((argmax != 0).to(dtype=weights.dtype) * weights).sum() / weight_sum
    return {
        "top1_prob_mean": float(top1.detach().cpu().item()),
        "argmax_nonzero_frac": float(nonzero.detach().cpu().item()),
        "argmax_route0_frac": route_fracs[0],
        "argmax_route1_frac": route_fracs[1],
        "argmax_route2_frac": route_fracs[2],
        "argmax_route3_frac": route_fracs[3],
        "argmax_max_route_frac": max(route_fracs),
    }


def _grouped_argmax_change_stats(
    *,
    old_argmax: torch.Tensor,
    new_argmax: torch.Tensor,
    sample_weight: torch.Tensor,
    group_id: torch.Tensor,
    min_samples: int = 1,
) -> Dict[str, float]:
    """Measure deterministic policy flips within stable sample groups."""
    empty = {
        "group_argmax_change_max": 0.0,
        "group_argmax_change_mean": 0.0,
        "group_argmax_change_p95": 0.0,
        "group_count": 0.0,
    }
    if not (
        torch.is_tensor(old_argmax)
        and torch.is_tensor(new_argmax)
        and torch.is_tensor(sample_weight)
        and torch.is_tensor(group_id)
    ):
        return empty
    n = min(
        int(old_argmax.reshape(-1).numel()),
        int(new_argmax.reshape(-1).numel()),
        int(sample_weight.reshape(-1).numel()),
        int(group_id.reshape(-1).numel()),
    )
    if n <= 0:
        return empty
    old_n = old_argmax.reshape(-1)[:n].to(device=new_argmax.device)
    new_n = new_argmax.reshape(-1)[:n]
    weights = sample_weight.reshape(-1)[:n].to(device=new_n.device, dtype=torch.float32).clamp_min(0.0)
    groups = group_id.reshape(-1)[:n].to(device=new_n.device, dtype=torch.long)
    finite = torch.isfinite(weights) & (groups >= 0)
    if not bool(finite.any().detach().cpu().item()):
        return empty

    groups_f = groups[finite]
    unique_groups = torch.unique(groups_f, sorted=True)
    min_count = max(int(min_samples), 1)
    group_changes: List[torch.Tensor] = []
    group_weights: List[torch.Tensor] = []
    changed = (new_n[finite] != old_n[finite]).to(dtype=torch.float32)
    weights_f = weights[finite]
    for gid in unique_groups:
        mask = groups_f == gid
        count = int(mask.to(dtype=torch.long).sum().detach().cpu().item())
        if count < min_count:
            continue
        denom = weights_f[mask].sum().clamp_min(1e-6)
        frac = (changed[mask] * weights_f[mask]).sum() / denom
        group_changes.append(frac)
        group_weights.append(denom)
    if not group_changes:
        return empty

    change_tensor = torch.stack(group_changes)
    weight_tensor = torch.stack(group_weights).clamp_min(0.0)
    weight_sum = weight_tensor.sum().clamp_min(1e-6)
    sorted_changes = torch.sort(change_tensor).values
    p95_idx = min(max(int(0.95 * (int(sorted_changes.numel()) - 1)), 0), int(sorted_changes.numel()) - 1)
    return {
        "group_argmax_change_max": float(change_tensor.max().detach().cpu().item()),
        "group_argmax_change_mean": float(((change_tensor * weight_tensor).sum() / weight_sum).detach().cpu().item()),
        "group_argmax_change_p95": float(sorted_changes[p95_idx].detach().cpu().item()),
        "group_count": float(int(change_tensor.numel())),
    }


def discrete_ac_policy_snapshot(
    actor,
    trajectories: List[Dict[str, object]],
    *,
    temperature: float,
    max_samples: int = 4096,
) -> Dict[str, object]:
    """Capture the current actor distribution on packed discrete AC samples."""
    obs_parts: List[torch.Tensor] = []
    valid_parts: List[torch.Tensor] = []
    weight_parts: List[torch.Tensor] = []
    vehicle_id_parts: List[torch.Tensor] = []
    for traj in trajectories:
        obs = traj.get("dreamer_ac_obs")
        valid = traj.get("dreamer_ac_valid")
        if not (torch.is_tensor(obs) and torch.is_tensor(valid) and obs.numel() > 0 and valid.numel() > 0):
            continue
        n = min(int(obs.shape[0]), int(valid.shape[0]))
        if n <= 0:
            continue
        obs_parts.append(obs[:n].detach())
        valid_parts.append(valid[:n].detach())
        sample_weight = traj.get("dreamer_ac_sample_weight")
        if torch.is_tensor(sample_weight) and sample_weight.numel() > 0:
            weight_n = sample_weight.reshape(-1)[:n].detach()
            if int(weight_n.numel()) == n:
                weight_parts.append(weight_n.to(device=obs.device, dtype=obs.dtype).clamp_min(0.0))
            else:
                weight_parts.append(torch.ones((n,), device=obs.device, dtype=obs.dtype))
        else:
            weight_parts.append(torch.ones((n,), device=obs.device, dtype=obs.dtype))
        vehicle_id = traj.get("dreamer_ac_vehicle_id")
        if torch.is_tensor(vehicle_id) and vehicle_id.numel() > 0:
            vehicle_id_n = vehicle_id.reshape(-1)[:n].detach()
            if int(vehicle_id_n.numel()) == n:
                vehicle_id_parts.append(vehicle_id_n.to(device=obs.device, dtype=torch.long))
            else:
                vehicle_id_parts.append(torch.full((n,), -1, device=obs.device, dtype=torch.long))
        else:
            vehicle_id_parts.append(torch.full((n,), -1, device=obs.device, dtype=torch.long))
    if not obs_parts:
        return {"sample_count": 0.0}

    obs_cat = torch.cat(obs_parts, dim=0)
    valid_cat = torch.cat(valid_parts, dim=0).to(device=obs_cat.device)
    weight_cat = torch.cat(weight_parts, dim=0).to(device=obs_cat.device, dtype=obs_cat.dtype)
    vehicle_id_cat = torch.cat(vehicle_id_parts, dim=0).to(device=obs_cat.device, dtype=torch.long)
    n_total = int(obs_cat.shape[0])
    max_n = int(max_samples)
    if max_n > 0 and n_total > max_n:
        # Deterministic stride keeps the diagnostic representative without
        # introducing another RNG source into the training loop.
        idx = torch.linspace(0, n_total - 1, steps=max_n, device=obs_cat.device).round().to(dtype=torch.long)
        obs_cat = obs_cat.index_select(0, idx)
        valid_cat = valid_cat.index_select(0, idx)
        weight_cat = weight_cat.index_select(0, idx)
        vehicle_id_cat = vehicle_id_cat.index_select(0, idx)

    with torch.no_grad():
        logits = actor.forward_logits(obs_cat)
        log_probs = _masked_log_probs(logits, valid_cat, temperature=float(temperature)).detach()
    stats = _policy_distribution_stats(log_probs, weight_cat)
    return {
        "obs": obs_cat.detach(),
        "valid": valid_cat.detach(),
        "sample_weight": weight_cat.detach(),
        "vehicle_id": vehicle_id_cat.detach(),
        "old_log_probs": log_probs,
        "old_argmax": torch.argmax(log_probs, dim=-1).detach(),
        "sample_count": float(int(obs_cat.shape[0])),
        "old_top1_prob_mean": stats["top1_prob_mean"],
        "old_argmax_nonzero_frac": stats["argmax_nonzero_frac"],
        "old_argmax_route0_frac": stats["argmax_route0_frac"],
        "old_argmax_route1_frac": stats["argmax_route1_frac"],
        "old_argmax_route2_frac": stats["argmax_route2_frac"],
        "old_argmax_route3_frac": stats["argmax_route3_frac"],
        "old_argmax_max_route_frac": stats["argmax_max_route_frac"],
    }


def discrete_ac_policy_shift_from_snapshot(
    actor,
    snapshot: Dict[str, object],
    *,
    temperature: float,
    group_min_samples: int = 1,
) -> Dict[str, float]:
    """Measure post-update policy drift from a pre-update AC snapshot."""
    obs = snapshot.get("obs")
    valid = snapshot.get("valid")
    old_log_probs = snapshot.get("old_log_probs")
    sample_weight = snapshot.get("sample_weight")
    old_argmax = snapshot.get("old_argmax")
    if not (
        torch.is_tensor(obs)
        and torch.is_tensor(valid)
        and torch.is_tensor(old_log_probs)
        and obs.numel() > 0
        and valid.numel() > 0
        and old_log_probs.numel() > 0
    ):
        return {
            "sample_count": 0.0,
            "kl": 0.0,
            "argmax_change_frac": 0.0,
            "top1_prob_mean": 0.0,
            "argmax_nonzero_frac": 0.0,
            "argmax_route0_frac": 0.0,
            "argmax_route1_frac": 0.0,
            "argmax_route2_frac": 0.0,
            "argmax_route3_frac": 0.0,
            "argmax_max_route_frac": 0.0,
            "vehicle_argmax_change_max": 0.0,
            "vehicle_argmax_change_mean": 0.0,
            "vehicle_argmax_change_p95": 0.0,
            "vehicle_group_count": 0.0,
        }
    with torch.no_grad():
        logits = actor.forward_logits(obs)
        new_log_probs = _masked_log_probs(logits, valid.to(device=logits.device), temperature=float(temperature)).detach()
        old_lp = old_log_probs.to(device=new_log_probs.device, dtype=new_log_probs.dtype)
        old_probs = torch.exp(old_lp)
        if torch.is_tensor(sample_weight) and int(sample_weight.numel()) == int(new_log_probs.shape[0]):
            weights = sample_weight.to(device=new_log_probs.device, dtype=new_log_probs.dtype).clamp_min(0.0)
        else:
            weights = torch.ones((int(new_log_probs.shape[0]),), device=new_log_probs.device, dtype=new_log_probs.dtype)
        kl_per = (old_probs * (old_lp - new_log_probs)).sum(dim=-1)
        finite = torch.isfinite(kl_per) & torch.isfinite(weights)
        if bool(finite.any().detach().cpu().item()):
            weight_sum = weights[finite].sum().clamp_min(1e-6)
            kl = (kl_per[finite] * weights[finite]).sum() / weight_sum
        else:
            kl = torch.zeros((), device=new_log_probs.device, dtype=new_log_probs.dtype)
        new_argmax = torch.argmax(new_log_probs, dim=-1)
        if torch.is_tensor(old_argmax) and int(old_argmax.numel()) == int(new_argmax.numel()):
            old_argmax_n = old_argmax.to(device=new_argmax.device)
            argmax_change = ((new_argmax != old_argmax_n).to(dtype=weights.dtype) * weights).sum() / weights.sum().clamp_min(1e-6)
        else:
            argmax_change = torch.zeros((), device=new_log_probs.device, dtype=new_log_probs.dtype)
    stats = _policy_distribution_stats(new_log_probs, weights)
    vehicle_group_stats = _grouped_argmax_change_stats(
        old_argmax=old_argmax if torch.is_tensor(old_argmax) else torch.zeros_like(new_argmax),
        new_argmax=new_argmax,
        sample_weight=weights,
        group_id=snapshot.get("vehicle_id"),
        min_samples=int(group_min_samples),
    )
    stats.update(
        {
            "sample_count": float(int(new_log_probs.shape[0])),
            "kl": float(kl.detach().cpu().item()),
            "argmax_change_frac": float(argmax_change.detach().cpu().item()),
            "vehicle_argmax_change_max": vehicle_group_stats["group_argmax_change_max"],
            "vehicle_argmax_change_mean": vehicle_group_stats["group_argmax_change_mean"],
            "vehicle_argmax_change_p95": vehicle_group_stats["group_argmax_change_p95"],
            "vehicle_group_count": vehicle_group_stats["group_count"],
        }
    )
    return stats


def append_discrete_actor_critic_samples(
    *,
    obs_steps: List[torch.Tensor],
    selected_steps: List[torch.Tensor],
    valid_steps: List[torch.Tensor],
    step_index_steps: List[torch.Tensor],
    selected_nonzero_fracs: List[float],
    obs: Optional[torch.Tensor],
    selected_idx: Optional[torch.Tensor],
    route_valid: Optional[torch.Tensor],
    step_index: int,
    route_advantage: Optional[torch.Tensor] = None,
    route_adv_steps: Optional[List[torch.Tensor]] = None,
    old_log_prob: Optional[torch.Tensor] = None,
    old_log_prob_steps: Optional[List[torch.Tensor]] = None,
    sample_weight: Optional[torch.Tensor] = None,
    sample_weight_steps: Optional[List[torch.Tensor]] = None,
    vehicle_id: Optional[torch.Tensor] = None,
    vehicle_id_steps: Optional[List[torch.Tensor]] = None,
    vehicle_reward: Optional[torch.Tensor] = None,
    vehicle_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_route_progress_reward: Optional[torch.Tensor] = None,
    vehicle_route_progress_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_route_credit_reward: Optional[torch.Tensor] = None,
    vehicle_route_credit_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_social_reward: Optional[torch.Tensor] = None,
    vehicle_social_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_route_reward_matrix: Optional[torch.Tensor] = None,
    vehicle_route_reward_matrix_steps: Optional[List[torch.Tensor]] = None,
) -> int:
    """Append valid per-vehicle sampled route decisions for actor-critic loss."""
    if obs is None or selected_idx is None or route_valid is None:
        return 0
    if not (torch.is_tensor(obs) and torch.is_tensor(selected_idx) and torch.is_tensor(route_valid)):
        return 0
    if obs.numel() <= 0 or selected_idx.numel() <= 0 or route_valid.numel() <= 0:
        return 0

    n = min(int(obs.shape[0]), int(selected_idx.reshape(-1).shape[0]), int(route_valid.shape[0]))
    if n <= 0:
        return 0
    obs_n = obs[:n]
    selected_n = selected_idx.reshape(-1)[:n].to(device=obs_n.device, dtype=torch.long)
    valid_n = route_valid[:n].to(device=obs_n.device)
    usable = valid_n.any(dim=-1)
    if not bool(usable.any().detach().cpu().item()):
        return 0

    obs_used = obs_n[usable].detach()
    selected_used = selected_n[usable].detach()
    valid_used = valid_n[usable].detach()
    obs_steps.append(obs_used)
    selected_steps.append(selected_used)
    valid_steps.append(valid_used)
    if route_adv_steps is not None:
        if torch.is_tensor(route_advantage) and route_advantage.numel() > 0:
            route_adv_n = route_advantage.reshape(-1)[:n].to(device=obs_n.device, dtype=obs_n.dtype)
            if int(route_adv_n.numel()) == n:
                route_adv_steps.append(route_adv_n[usable].detach())
            else:
                route_adv_steps.append(torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype))
        else:
            route_adv_steps.append(torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype))
    if old_log_prob_steps is not None:
        if torch.is_tensor(old_log_prob) and old_log_prob.numel() > 0:
            old_lp_n = old_log_prob.reshape(-1)[:n].to(device=obs_n.device, dtype=obs_n.dtype)
            if int(old_lp_n.numel()) == n:
                old_log_prob_steps.append(old_lp_n[usable].detach())
    if sample_weight_steps is not None:
        if torch.is_tensor(sample_weight) and sample_weight.numel() > 0:
            weight_n = sample_weight.reshape(-1)[:n].to(device=obs_n.device, dtype=obs_n.dtype)
            if int(weight_n.numel()) == n:
                sample_weight_steps.append(weight_n[usable].detach())
            else:
                sample_weight_steps.append(torch.ones((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype))
        else:
            sample_weight_steps.append(torch.ones((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype))
    if vehicle_id_steps is not None:
        if torch.is_tensor(vehicle_id) and vehicle_id.numel() > 0:
            vehicle_id_n = vehicle_id.reshape(-1)[:n].to(device=obs_n.device, dtype=torch.long)
            if int(vehicle_id_n.numel()) == n:
                vehicle_id_steps.append(vehicle_id_n[usable].detach())
            else:
                vehicle_id_steps.append(
                    torch.full(
                        (int(obs_used.shape[0]),),
                        -1,
                        device=obs_used.device,
                        dtype=torch.long,
                    )
                )
        else:
            vehicle_id_steps.append(
                torch.full(
                    (int(obs_used.shape[0]),),
                    -1,
                    device=obs_used.device,
                    dtype=torch.long,
                )
            )
    if vehicle_reward_steps is not None:
        if torch.is_tensor(vehicle_reward) and vehicle_reward.numel() > 0:
            vehicle_reward_n = vehicle_reward.reshape(-1)[:n].to(device=obs_n.device, dtype=obs_n.dtype)
            if int(vehicle_reward_n.numel()) == n:
                vehicle_reward_steps.append(vehicle_reward_n[usable].detach())
            else:
                vehicle_reward_steps.append(
                    torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
                )
        else:
            vehicle_reward_steps.append(
                torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
            )
    if vehicle_route_progress_reward_steps is not None:
        if torch.is_tensor(vehicle_route_progress_reward) and vehicle_route_progress_reward.numel() > 0:
            progress_reward_n = vehicle_route_progress_reward.reshape(-1)[:n].to(
                device=obs_n.device,
                dtype=obs_n.dtype,
            )
            if int(progress_reward_n.numel()) == n:
                vehicle_route_progress_reward_steps.append(progress_reward_n[usable].detach())
            else:
                vehicle_route_progress_reward_steps.append(
                    torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
                )
        else:
            vehicle_route_progress_reward_steps.append(
                torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
            )
    if vehicle_route_credit_reward_steps is not None:
        if torch.is_tensor(vehicle_route_credit_reward) and vehicle_route_credit_reward.numel() > 0:
            route_credit_reward_n = vehicle_route_credit_reward.reshape(-1)[:n].to(
                device=obs_n.device,
                dtype=obs_n.dtype,
            )
            if int(route_credit_reward_n.numel()) == n:
                vehicle_route_credit_reward_steps.append(route_credit_reward_n[usable].detach())
            else:
                vehicle_route_credit_reward_steps.append(
                    torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
                )
        else:
            vehicle_route_credit_reward_steps.append(
                torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
            )
    if vehicle_social_reward_steps is not None:
        if torch.is_tensor(vehicle_social_reward) and vehicle_social_reward.numel() > 0:
            social_reward_n = vehicle_social_reward.reshape(-1)[:n].to(device=obs_n.device, dtype=obs_n.dtype)
            if int(social_reward_n.numel()) == n:
                vehicle_social_reward_steps.append(social_reward_n[usable].detach())
            else:
                vehicle_social_reward_steps.append(
                    torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
                )
        else:
            vehicle_social_reward_steps.append(
                torch.zeros((int(obs_used.shape[0]),), device=obs_used.device, dtype=obs_used.dtype)
            )
    if vehicle_route_reward_matrix_steps is not None:
        if (
            torch.is_tensor(vehicle_route_reward_matrix)
            and vehicle_route_reward_matrix.numel() > 0
            and vehicle_route_reward_matrix.dim() == 2
        ):
            matrix_n = vehicle_route_reward_matrix[:n].to(device=obs_n.device, dtype=obs_n.dtype)
            if int(matrix_n.shape[0]) == n and int(matrix_n.shape[1]) == int(valid_n.shape[1]):
                vehicle_route_reward_matrix_steps.append(matrix_n[usable].detach())
            else:
                vehicle_route_reward_matrix_steps.append(
                    torch.zeros(
                        (int(obs_used.shape[0]), int(valid_used.shape[1])),
                        device=obs_used.device,
                        dtype=obs_used.dtype,
                    )
                )
        else:
            vehicle_route_reward_matrix_steps.append(
                torch.zeros(
                    (int(obs_used.shape[0]), int(valid_used.shape[1])),
                    device=obs_used.device,
                    dtype=obs_used.dtype,
                )
            )
    step_index_steps.append(
        torch.full(
            (int(obs_used.shape[0]),),
            int(step_index),
            device=obs_used.device,
            dtype=torch.long,
        )
    )
    selected_nonzero_fracs.append(
        float((selected_used != 0).to(dtype=torch.float32).mean().detach().cpu().item())
    )
    return int(obs_used.shape[0])


def pack_discrete_actor_critic_samples(
    *,
    obs_steps: List[torch.Tensor],
    selected_steps: List[torch.Tensor],
    valid_steps: List[torch.Tensor],
    step_index_steps: List[torch.Tensor],
    route_adv_steps: Optional[List[torch.Tensor]] = None,
    old_log_prob_steps: Optional[List[torch.Tensor]] = None,
    sample_weight_steps: Optional[List[torch.Tensor]] = None,
    vehicle_id_steps: Optional[List[torch.Tensor]] = None,
    vehicle_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_route_progress_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_route_credit_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_social_reward_steps: Optional[List[torch.Tensor]] = None,
    vehicle_route_reward_matrix_steps: Optional[List[torch.Tensor]] = None,
    device: torch.device,
    dtype: torch.dtype,
    obs_dim: int,
    n_routes: int,
) -> Dict[str, torch.Tensor]:
    if obs_steps:
        selected = torch.cat(selected_steps, dim=0)
        route_adv = torch.zeros((int(selected.numel()),), device=device, dtype=dtype)
        if route_adv_steps:
            route_adv_cat = torch.cat(route_adv_steps, dim=0).to(device=device, dtype=dtype)
            if int(route_adv_cat.numel()) == int(selected.numel()):
                route_adv = route_adv_cat
        old_log_prob = torch.zeros((0,), device=device, dtype=dtype)
        if old_log_prob_steps:
            old_log_prob_cat = torch.cat(old_log_prob_steps, dim=0).to(device=device, dtype=dtype)
            if int(old_log_prob_cat.numel()) == int(selected.numel()):
                old_log_prob = old_log_prob_cat
        sample_weight = torch.ones((int(selected.numel()),), device=device, dtype=dtype)
        if sample_weight_steps:
            sample_weight_cat = torch.cat(sample_weight_steps, dim=0).to(device=device, dtype=dtype)
            if int(sample_weight_cat.numel()) == int(selected.numel()):
                sample_weight = sample_weight_cat.clamp_min(0.0)
        vehicle_id = torch.full((int(selected.numel()),), -1, device=device, dtype=torch.long)
        if vehicle_id_steps:
            vehicle_id_cat = torch.cat(vehicle_id_steps, dim=0).to(device=device, dtype=torch.long)
            if int(vehicle_id_cat.numel()) == int(selected.numel()):
                vehicle_id = vehicle_id_cat
        vehicle_reward = torch.zeros((int(selected.numel()),), device=device, dtype=dtype)
        if vehicle_reward_steps:
            vehicle_reward_cat = torch.cat(vehicle_reward_steps, dim=0).to(device=device, dtype=dtype)
            if int(vehicle_reward_cat.numel()) == int(selected.numel()):
                vehicle_reward = vehicle_reward_cat
        vehicle_route_progress_reward = torch.zeros((int(selected.numel()),), device=device, dtype=dtype)
        if vehicle_route_progress_reward_steps:
            progress_reward_cat = torch.cat(vehicle_route_progress_reward_steps, dim=0).to(
                device=device,
                dtype=dtype,
            )
            if int(progress_reward_cat.numel()) == int(selected.numel()):
                vehicle_route_progress_reward = progress_reward_cat
        vehicle_route_credit_reward = torch.zeros((int(selected.numel()),), device=device, dtype=dtype)
        if vehicle_route_credit_reward_steps:
            route_credit_reward_cat = torch.cat(vehicle_route_credit_reward_steps, dim=0).to(
                device=device,
                dtype=dtype,
            )
            if int(route_credit_reward_cat.numel()) == int(selected.numel()):
                vehicle_route_credit_reward = route_credit_reward_cat
        vehicle_social_reward = torch.zeros((int(selected.numel()),), device=device, dtype=dtype)
        if vehicle_social_reward_steps:
            social_reward_cat = torch.cat(vehicle_social_reward_steps, dim=0).to(device=device, dtype=dtype)
            if int(social_reward_cat.numel()) == int(selected.numel()):
                vehicle_social_reward = social_reward_cat
        vehicle_route_reward_matrix = torch.zeros((int(selected.numel()), int(n_routes)), device=device, dtype=dtype)
        if vehicle_route_reward_matrix_steps:
            matrix_cat = torch.cat(vehicle_route_reward_matrix_steps, dim=0).to(device=device, dtype=dtype)
            if (
                int(matrix_cat.shape[0]) == int(selected.numel())
                and matrix_cat.dim() == 2
                and int(matrix_cat.shape[1]) == int(n_routes)
            ):
                vehicle_route_reward_matrix = matrix_cat
        return {
            "dreamer_ac_obs": torch.cat(obs_steps, dim=0),
            "dreamer_ac_selected": selected,
            "dreamer_ac_valid": torch.cat(valid_steps, dim=0),
            "dreamer_ac_step_index": torch.cat(step_index_steps, dim=0),
            "dreamer_ac_route_adv": route_adv,
            "dreamer_ac_old_log_prob": old_log_prob,
            "dreamer_ac_sample_weight": sample_weight,
            "dreamer_ac_vehicle_id": vehicle_id,
            "dreamer_ac_vehicle_reward": vehicle_reward,
            "dreamer_ac_vehicle_route_progress_reward": vehicle_route_progress_reward,
            "dreamer_ac_vehicle_route_credit_reward": vehicle_route_credit_reward,
            "dreamer_ac_vehicle_social_reward": vehicle_social_reward,
            "dreamer_ac_vehicle_route_reward_matrix": vehicle_route_reward_matrix,
            "dreamer_ac_sample_adv": torch.zeros((0,), device=device, dtype=dtype),
        }
    return {
        "dreamer_ac_obs": torch.zeros((0, int(obs_dim)), device=device, dtype=dtype),
        "dreamer_ac_selected": torch.zeros((0,), device=device, dtype=torch.long),
        "dreamer_ac_valid": torch.zeros((0, int(n_routes)), device=device, dtype=torch.bool),
        "dreamer_ac_step_index": torch.zeros((0,), device=device, dtype=torch.long),
        "dreamer_ac_route_adv": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_old_log_prob": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_sample_weight": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_vehicle_id": torch.zeros((0,), device=device, dtype=torch.long),
        "dreamer_ac_vehicle_reward": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_vehicle_route_progress_reward": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_vehicle_route_credit_reward": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_vehicle_social_reward": torch.zeros((0,), device=device, dtype=dtype),
        "dreamer_ac_vehicle_route_reward_matrix": torch.zeros((0, int(n_routes)), device=device, dtype=dtype),
        "dreamer_ac_sample_adv": torch.zeros((0,), device=device, dtype=dtype),
    }


def _empty_stats() -> Dict[str, float]:
    return {
        "dreamer_ac_loss": 0.0,
        "dreamer_ac_adv_mean": 0.0,
        "dreamer_ac_adv_std": 0.0,
        "dreamer_ac_adv_min": 0.0,
        "dreamer_ac_adv_max": 0.0,
        "dreamer_ac_route_adv_mean": 0.0,
        "dreamer_ac_route_adv_std": 0.0,
        "dreamer_ac_combined_adv_mean": 0.0,
        "dreamer_ac_combined_adv_std": 0.0,
        "dreamer_ac_selected_nonzero_frac": 0.0,
        "dreamer_ac_selected_route0_frac": 0.0,
        "dreamer_ac_selected_route1_frac": 0.0,
        "dreamer_ac_selected_route2_frac": 0.0,
        "dreamer_ac_selected_route3_frac": 0.0,
        "dreamer_ac_policy_argmax_nonzero_frac": 0.0,
        "dreamer_ac_policy_argmax_route0_frac": 0.0,
        "dreamer_ac_policy_argmax_route1_frac": 0.0,
        "dreamer_ac_policy_argmax_route2_frac": 0.0,
        "dreamer_ac_policy_argmax_route3_frac": 0.0,
        "dreamer_ac_policy_argmax_max_route_frac": 0.0,
        "dreamer_ac_policy_top1_prob_mean": 0.0,
        "dreamer_ac_sample_count": 0.0,
        "dreamer_ac_policy_kl": 0.0,
        "dreamer_ac_clip_frac": 0.0,
        "dreamer_ac_weight_mean": 0.0,
        "dreamer_ac_uses_sample_adv": 0.0,
        "dreamer_ac_entropy_loss": 0.0,
        "dreamer_ac_entropy_floor_loss": 0.0,
        "dreamer_ac_entropy_target": 0.0,
        "dreamer_ac_entropy_valid_target_mean": 0.0,
        "dreamer_ac_entropy_effective_route_count": 0.0,
        "dreamer_ac_entropy_cost_mask_frac": 0.0,
    }


def _cost_aware_entropy_valid_mask(
    *,
    actor,
    obs: torch.Tensor,
    valid: torch.Tensor,
    max_gap: float,
    min_routes: int,
) -> torch.Tensor:
    """Restrict entropy-floor target routes to near-best current-cost candidates.

    The mask is derived from route-cost observation features already available
    to the actor. It only changes the entropy-floor target; rewards still enter
    actor updates through critic-derived advantages.
    """
    if float(max_gap) < 0.0:
        return valid
    if not bool(getattr(actor, "include_route_costs", False)):
        return valid
    n_routes = int(getattr(actor, "n_routes", valid.shape[-1]))
    if n_routes <= 0:
        return valid
    route_cost_start_fn = getattr(actor, "_route_cost_start", None)
    if not callable(route_cost_start_fn):
        return valid
    gap_start = int(route_cost_start_fn())
    gap_end = gap_start + n_routes
    if gap_start < 0 or int(obs.shape[-1]) < gap_end:
        return valid

    route_gap = obs[..., gap_start:gap_end].to(device=valid.device, dtype=torch.float32)
    route_gap = torch.nan_to_num(route_gap, nan=1e6, posinf=1e6, neginf=1e6)
    k_eff = min(int(route_gap.shape[-1]), int(valid.shape[-1]), n_routes)
    if k_eff <= 0:
        return valid

    base_valid = valid.clone()
    near = base_valid.clone()
    near[:, :k_eff] = base_valid[:, :k_eff] & (route_gap[:, :k_eff] <= float(max_gap))

    min_keep = max(int(min_routes), 1)
    if min_keep > 1 and k_eff > 1:
        large = torch.finfo(route_gap.dtype).max / 4.0
        masked_gap = route_gap[:, :k_eff].masked_fill(~base_valid[:, :k_eff], large)
        order = torch.argsort(masked_gap, dim=-1, stable=True)
        valid_count = base_valid[:, :k_eff].sum(dim=-1)
        keep_count = torch.minimum(
            valid_count,
            torch.full_like(valid_count, min_keep),
        ).clamp_min(1)
        rank = torch.arange(k_eff, device=valid.device).unsqueeze(0).expand_as(order)
        top_keep_sorted = rank < keep_count.unsqueeze(-1)
        top_keep = torch.zeros_like(base_valid[:, :k_eff])
        top_keep.scatter_(1, order, top_keep_sorted)
        near[:, :k_eff] = near[:, :k_eff] | (top_keep & base_valid[:, :k_eff])

    no_near = ~near.any(dim=-1)
    if bool(no_near.any().detach().cpu().item()):
        near[no_near] = base_valid[no_near]
    return near


def discrete_actor_critic_policy_loss(
    *,
    actor,
    traj: Dict[str, object],
    step_advantages: torch.Tensor,
    entropy: torch.Tensor,
    entropy_coef: float,
    temperature: float,
    step_advantage_weight: float = 1.0,
    route_advantage_weight: float = 0.0,
    advantage_clip: float = 5.0,
    old_log_prob: Optional[torch.Tensor] = None,
    sample_weight: Optional[torch.Tensor] = None,
    ppo_clip_ratio: float = 0.0,
    normalize_combined_advantage: bool = False,
    entropy_target: float = 0.0,
    entropy_target_frac: float = 0.0,
    entropy_floor_coef: float = 0.0,
    entropy_cost_gap_max: float = -1.0,
    entropy_cost_min_routes: int = 1,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Return score-function actor loss for sampled discrete route actions.

    ``step_advantages`` should already be the Dreamer lambda-return advantage
    for each imagined step.  Gradients flow only through the current actor
    log-probabilities and entropy; advantages are detached.
    """
    obs = traj.get("dreamer_ac_obs")
    selected = traj.get("dreamer_ac_selected")
    valid = traj.get("dreamer_ac_valid")
    step_index = traj.get("dreamer_ac_step_index")
    route_advantage = traj.get("dreamer_ac_route_adv")
    sample_advantage = traj.get("dreamer_ac_sample_adv")
    if old_log_prob is None:
        old_log_prob = traj.get("dreamer_ac_old_log_prob")
    if sample_weight is None:
        sample_weight = traj.get("dreamer_ac_sample_weight")
    if not (
        torch.is_tensor(obs)
        and torch.is_tensor(selected)
        and torch.is_tensor(valid)
        and torch.is_tensor(step_index)
        and torch.is_tensor(step_advantages)
        and obs.numel() > 0
        and selected.numel() > 0
        and valid.numel() > 0
    ):
        return None, _empty_stats()

    n = min(
        int(obs.shape[0]),
        int(selected.reshape(-1).shape[0]),
        int(valid.shape[0]),
        int(step_index.reshape(-1).shape[0]),
    )
    if n <= 0:
        return None, _empty_stats()

    logits = actor.forward_logits(obs[:n])
    valid_n = valid[:n].to(device=logits.device)
    selected_n = selected.reshape(-1)[:n].to(device=logits.device, dtype=torch.long)
    selected_n = selected_n.clamp(0, max(int(logits.shape[1]) - 1, 0))
    step_idx = step_index.reshape(-1)[:n].to(device=logits.device, dtype=torch.long)
    step_idx = step_idx.clamp(0, max(int(step_advantages.reshape(-1).shape[0]) - 1, 0))

    log_probs = _masked_log_probs(logits, valid_n, temperature=float(temperature))
    probs = torch.exp(log_probs)
    entropy_per_sample = -(probs * log_probs).sum(dim=-1)
    rows = torch.arange(n, device=logits.device)
    selected_logp = log_probs[rows, selected_n]
    selected_valid = valid_n[rows, selected_n]
    use_sample_advantage = False
    if torch.is_tensor(sample_advantage) and sample_advantage.numel() > 0:
        sample_adv_n = sample_advantage.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype)
        if int(sample_adv_n.numel()) == n:
            adv = sample_adv_n.detach()
            use_sample_advantage = True
        else:
            adv = step_advantages.reshape(-1)[step_idx].to(device=logits.device, dtype=logits.dtype).detach()
    else:
        adv = step_advantages.reshape(-1)[step_idx].to(device=logits.device, dtype=logits.dtype).detach()
    route_adv = torch.zeros_like(adv)
    if torch.is_tensor(route_advantage) and route_advantage.numel() > 0:
        route_n = route_advantage.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()
        if int(route_n.numel()) == n:
            route_adv = route_n
    combined_adv = float(step_advantage_weight) * adv + float(route_advantage_weight) * route_adv
    if bool(normalize_combined_advantage) and int(combined_adv.numel()) > 1:
        ref = combined_adv.detach()
        combined_adv = (combined_adv - ref.mean()) / ref.std(unbiased=False).clamp_min(1e-6)
    if float(advantage_clip) > 0.0:
        combined_adv = combined_adv.clamp(min=-float(advantage_clip), max=float(advantage_clip))
    finite = selected_valid & torch.isfinite(selected_logp) & torch.isfinite(adv)
    finite = finite & torch.isfinite(combined_adv)
    sample_w = torch.ones_like(combined_adv)
    if torch.is_tensor(sample_weight) and sample_weight.numel() > 0:
        weight_n = sample_weight.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()
        if int(weight_n.numel()) == n:
            sample_w = weight_n.clamp_min(0.0)
            finite = finite & torch.isfinite(sample_w)
    old_logp_n = None
    if torch.is_tensor(old_log_prob) and old_log_prob.numel() > 0:
        old_candidate = old_log_prob.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()
        if int(old_candidate.numel()) == n:
            old_logp_n = old_candidate
            finite = finite & torch.isfinite(old_logp_n)
    if not bool(finite.any().detach().cpu().item()):
        return None, _empty_stats()

    selected_logp = selected_logp[finite]
    entropy_used = entropy_per_sample[finite]
    valid_count_used = valid_n[finite].sum(dim=-1).to(dtype=logits.dtype).clamp_min(1.0)
    adv = adv[finite]
    route_adv = route_adv[finite]
    combined_adv = combined_adv[finite]
    sample_w = sample_w[finite]
    weight_sum = sample_w.sum().clamp_min(1e-6)
    if old_logp_n is not None and float(ppo_clip_ratio) > 0.0:
        old_logp_used = old_logp_n[finite]
        ratio = torch.exp(selected_logp - old_logp_used)
        clipped_ratio = ratio.clamp(1.0 - float(ppo_clip_ratio), 1.0 + float(ppo_clip_ratio))
        surrogate = torch.minimum(ratio * combined_adv, clipped_ratio * combined_adv)
        pg_loss = -((sample_w * surrogate).sum() / weight_sum)
        approx_kl = (old_logp_used - selected_logp).detach()
        clip_frac = ((ratio - 1.0).abs() > float(ppo_clip_ratio)).to(dtype=logits.dtype)
    else:
        pg_loss = -((sample_w * combined_adv * selected_logp).sum() / weight_sum)
        approx_kl = torch.zeros_like(selected_logp)
        clip_frac = torch.zeros_like(selected_logp)
    entropy_loss = (sample_w * entropy_used).sum() / weight_sum
    entropy_floor_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    absolute_target_entropy = max(float(entropy_target), 0.0)
    entropy_valid = valid_n[finite]
    if float(entropy_cost_gap_max) >= 0.0:
        entropy_valid = _cost_aware_entropy_valid_mask(
            actor=actor,
            obs=obs[:n][finite],
            valid=entropy_valid,
            max_gap=float(entropy_cost_gap_max),
            min_routes=int(entropy_cost_min_routes),
        )
    entropy_valid_count = entropy_valid.sum(dim=-1).to(dtype=logits.dtype).clamp_min(1.0)
    frac_target_entropy = torch.zeros_like(entropy_used)
    if float(entropy_target_frac) > 0.0:
        frac = max(float(entropy_target_frac), 0.0)
        frac_target_entropy = frac * torch.log(entropy_valid_count)
    if absolute_target_entropy > 0.0:
        target_entropy_per_sample = torch.maximum(
            frac_target_entropy,
            torch.full_like(frac_target_entropy, absolute_target_entropy),
        )
    else:
        target_entropy_per_sample = frac_target_entropy
    target_entropy = (
        (sample_w * target_entropy_per_sample).sum() / weight_sum
        if int(target_entropy_per_sample.numel()) > 0
        else torch.zeros((), device=logits.device, dtype=logits.dtype)
    )
    if float(entropy_floor_coef) > 0.0:
        entropy_gap = torch.relu(target_entropy_per_sample - entropy_used)
        entropy_floor_loss = (sample_w * entropy_gap.square()).sum() / weight_sum
    loss = pg_loss - float(entropy_coef) * entropy_loss + float(entropy_floor_coef) * entropy_floor_loss
    stats = _empty_stats()
    selected_used = selected_n[finite]
    log_probs_used = log_probs[finite]
    probs_used = probs[finite]
    policy_argmax_used = torch.argmax(log_probs_used.detach(), dim=-1)
    policy_top1_prob = probs_used.detach().max(dim=-1).values
    policy_argmax_route_fracs = []
    for route_idx in range(4):
        if int(log_probs_used.shape[1]) > route_idx:
            frac = (policy_argmax_used == route_idx).to(dtype=torch.float32).mean()
            policy_argmax_route_fracs.append(float(frac.detach().cpu().item()))
        else:
            policy_argmax_route_fracs.append(0.0)
    stats.update(
        {
            "dreamer_ac_loss": float(loss.detach().cpu().item()),
            "dreamer_ac_adv_mean": float(adv.detach().mean().cpu().item()),
            "dreamer_ac_adv_std": (
                float(adv.detach().std(unbiased=False).cpu().item())
                if int(adv.numel()) > 1
                else 0.0
            ),
            "dreamer_ac_adv_min": float(adv.detach().min().cpu().item()),
            "dreamer_ac_adv_max": float(adv.detach().max().cpu().item()),
            "dreamer_ac_route_adv_mean": float(route_adv.detach().mean().cpu().item()),
            "dreamer_ac_route_adv_std": (
                float(route_adv.detach().std(unbiased=False).cpu().item())
                if int(route_adv.numel()) > 1
                else 0.0
            ),
            "dreamer_ac_combined_adv_mean": float(combined_adv.detach().mean().cpu().item()),
            "dreamer_ac_combined_adv_std": (
                float(combined_adv.detach().std(unbiased=False).cpu().item())
                if int(combined_adv.numel()) > 1
                else 0.0
            ),
            "dreamer_ac_selected_nonzero_frac": float(
                (selected_used != 0).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "dreamer_ac_selected_route0_frac": float(
                (selected_used == 0).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "dreamer_ac_selected_route1_frac": float(
                (selected_used == 1).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "dreamer_ac_selected_route2_frac": float(
                (selected_used == 2).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "dreamer_ac_selected_route3_frac": float(
                (selected_used == 3).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "dreamer_ac_policy_argmax_nonzero_frac": float(
                (policy_argmax_used != 0).to(dtype=torch.float32).mean().detach().cpu().item()
            ),
            "dreamer_ac_policy_argmax_route0_frac": policy_argmax_route_fracs[0],
            "dreamer_ac_policy_argmax_route1_frac": policy_argmax_route_fracs[1],
            "dreamer_ac_policy_argmax_route2_frac": policy_argmax_route_fracs[2],
            "dreamer_ac_policy_argmax_route3_frac": policy_argmax_route_fracs[3],
            "dreamer_ac_policy_argmax_max_route_frac": max(policy_argmax_route_fracs),
            "dreamer_ac_policy_top1_prob_mean": float(
                policy_top1_prob.mean().detach().cpu().item()
            ),
            "dreamer_ac_sample_count": float(int(adv.numel())),
            "dreamer_ac_policy_kl": float(approx_kl.mean().cpu().item()) if int(approx_kl.numel()) > 0 else 0.0,
            "dreamer_ac_clip_frac": float(clip_frac.mean().cpu().item()) if int(clip_frac.numel()) > 0 else 0.0,
            "dreamer_ac_weight_mean": float(sample_w.mean().detach().cpu().item()) if int(sample_w.numel()) > 0 else 0.0,
            "dreamer_ac_uses_sample_adv": 1.0 if use_sample_advantage else 0.0,
            "dreamer_ac_entropy_loss": float(entropy_loss.detach().cpu().item()),
            "dreamer_ac_entropy_floor_loss": float(entropy_floor_loss.detach().cpu().item()),
            "dreamer_ac_entropy_target": float(target_entropy.detach().cpu().item()),
            "dreamer_ac_entropy_valid_target_mean": float(
                frac_target_entropy.detach().mean().cpu().item()
            ) if int(frac_target_entropy.numel()) > 0 else 0.0,
            "dreamer_ac_entropy_effective_route_count": float(
                entropy_valid_count.detach().mean().cpu().item()
            ) if int(entropy_valid_count.numel()) > 0 else 0.0,
            "dreamer_ac_entropy_cost_mask_frac": float(
                (entropy_valid_count < valid_count_used).to(dtype=torch.float32).mean().detach().cpu().item()
            ) if int(entropy_valid_count.numel()) > 0 else 0.0,
        }
    )
    return loss, stats


def vehicle_route_q_actor_loss(
    *,
    actor,
    q_critic,
    traj: Dict[str, object],
    temperature: float,
    actor_weight: float = 1.0,
    advantage_clip: float = 5.0,
    normalize_advantage: bool = True,
    detach_values: bool = True,
    entropy: Optional[torch.Tensor] = None,
    entropy_coef: float = 0.0,
) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Policy loss from critic-predicted per-route vehicle Q values.

    Rewards still enter through the route-Q critic target. This helper freezes
    critic parameters and lets gradients flow only through actor probabilities.
    """
    obs = traj.get("dreamer_ac_obs")
    valid = traj.get("dreamer_ac_valid")
    step_index = traj.get("dreamer_ac_step_index")
    sample_weight = traj.get("dreamer_ac_sample_weight")
    if not (
        torch.is_tensor(obs)
        and torch.is_tensor(valid)
        and torch.is_tensor(step_index)
        and obs.numel() > 0
        and valid.numel() > 0
    ):
        return None, {
            "vehicle_route_q_actor_loss": 0.0,
            "vehicle_route_q_adv_mean": 0.0,
            "vehicle_route_q_adv_std": 0.0,
        }
    n = min(int(obs.shape[0]), int(valid.shape[0]), int(step_index.reshape(-1).shape[0]))
    if n <= 0:
        return None, {
            "vehicle_route_q_actor_loss": 0.0,
            "vehicle_route_q_adv_mean": 0.0,
            "vehicle_route_q_adv_std": 0.0,
        }

    obs_n = obs[:n]
    valid_n = valid[:n].to(device=obs_n.device)
    usable = valid_n.any(dim=-1) & torch.isfinite(obs_n).all(dim=-1)
    if not bool(usable.any().detach().cpu().item()):
        return None, {
            "vehicle_route_q_actor_loss": 0.0,
            "vehicle_route_q_adv_mean": 0.0,
            "vehicle_route_q_adv_std": 0.0,
        }
    obs_used = obs_n[usable]
    valid_used = valid_n[usable]
    logits = actor.forward_logits(obs_used)
    log_probs = _masked_log_probs(logits, valid_used, temperature=float(temperature))
    probs = torch.exp(log_probs)
    q_values = q_critic(
        obs_used,
        context=None,
        group_index=step_index.reshape(-1)[:n].to(device=obs_n.device, dtype=torch.long)[usable],
    )
    if q_values.dim() != 2 or int(q_values.shape[1]) != int(logits.shape[1]):
        raise ValueError(
            f"route-Q critic must return (N, K), got {tuple(q_values.shape)} for logits {tuple(logits.shape)}"
        )
    q_actor = q_values.detach() if bool(detach_values) else q_values
    q_actor = q_actor.to(device=logits.device, dtype=logits.dtype)
    q_actor = q_actor.masked_fill(~valid_used, 0.0)
    baseline = (probs.detach() * q_actor).sum(dim=-1, keepdim=True)
    advantage = (q_actor - baseline).masked_fill(~valid_used, 0.0)
    if bool(normalize_advantage):
        valid_adv = advantage[valid_used]
        if int(valid_adv.numel()) > 1:
            mean = valid_adv.detach().mean()
            std = valid_adv.detach().std(unbiased=False).clamp_min(1e-6)
            advantage = advantage / std
            advantage = advantage - (mean / std)
            advantage = advantage.masked_fill(~valid_used, 0.0)
    if float(advantage_clip) > 0.0:
        advantage = advantage.clamp(-float(advantage_clip), float(advantage_clip))

    per_sample_obj = (probs * advantage).sum(dim=-1)
    sample_w = torch.ones_like(per_sample_obj)
    if torch.is_tensor(sample_weight) and sample_weight.numel() > 0:
        weight_n = sample_weight.reshape(-1)[:n].to(device=obs_n.device, dtype=obs_n.dtype)
        if int(weight_n.numel()) == n:
            sample_w = weight_n[usable].to(device=per_sample_obj.device, dtype=per_sample_obj.dtype).clamp_min(0.0)
    finite = torch.isfinite(per_sample_obj) & torch.isfinite(sample_w)
    if not bool(finite.any().detach().cpu().item()):
        return None, {
            "vehicle_route_q_actor_loss": 0.0,
            "vehicle_route_q_adv_mean": 0.0,
            "vehicle_route_q_adv_std": 0.0,
        }
    weight_sum = sample_w[finite].sum().clamp_min(1e-6)
    loss = -float(actor_weight) * (sample_w[finite] * per_sample_obj[finite]).sum() / weight_sum
    if entropy is not None and torch.is_tensor(entropy) and float(entropy_coef) != 0.0:
        loss = loss - float(entropy_coef) * entropy.to(device=loss.device, dtype=loss.dtype)
    valid_adv_used = advantage[finite][valid_used[finite]]
    return loss, {
        "vehicle_route_q_actor_loss": float(loss.detach().cpu().item()),
        "vehicle_route_q_adv_mean": (
            float(valid_adv_used.detach().mean().cpu().item())
            if int(valid_adv_used.numel()) > 0 else 0.0
        ),
        "vehicle_route_q_adv_std": (
            float(valid_adv_used.detach().std(unbiased=False).cpu().item())
            if int(valid_adv_used.numel()) > 1 else 0.0
        ),
    }
