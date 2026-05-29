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
    vehicle_social_reward: Optional[torch.Tensor] = None,
    vehicle_social_reward_steps: Optional[List[torch.Tensor]] = None,
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
    vehicle_social_reward_steps: Optional[List[torch.Tensor]] = None,
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
        vehicle_social_reward = torch.zeros((int(selected.numel()),), device=device, dtype=dtype)
        if vehicle_social_reward_steps:
            social_reward_cat = torch.cat(vehicle_social_reward_steps, dim=0).to(device=device, dtype=dtype)
            if int(social_reward_cat.numel()) == int(selected.numel()):
                vehicle_social_reward = social_reward_cat
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
            "dreamer_ac_vehicle_social_reward": vehicle_social_reward,
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
        "dreamer_ac_vehicle_social_reward": torch.zeros((0,), device=device, dtype=dtype),
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
        "dreamer_ac_sample_count": 0.0,
        "dreamer_ac_policy_kl": 0.0,
        "dreamer_ac_clip_frac": 0.0,
        "dreamer_ac_weight_mean": 0.0,
        "dreamer_ac_uses_sample_adv": 0.0,
        "dreamer_ac_entropy_loss": 0.0,
        "dreamer_ac_entropy_floor_loss": 0.0,
        "dreamer_ac_entropy_target": 0.0,
        "dreamer_ac_entropy_valid_target_mean": 0.0,
    }


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
    frac_target_entropy = torch.zeros_like(entropy_used)
    if float(entropy_target_frac) > 0.0:
        frac = max(float(entropy_target_frac), 0.0)
        frac_target_entropy = frac * torch.log(valid_count_used)
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
                (selected_n[finite] != 0).to(dtype=torch.float32).mean().detach().cpu().item()
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
        }
    )
    return loss, stats
