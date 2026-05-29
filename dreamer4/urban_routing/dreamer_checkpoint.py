"""Checkpoint, resume, and RNG state helpers for Dreamer CTDE training."""

import copy
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_dreamer_checkpoint(
    cfg,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    episode,
    stats,
    extra_state=None,
    target_critic=None,
    return_normalizer=None,
    vehicle_critic=None,
    vehicle_critic_optimizer=None,
    target_vehicle_critic=None,
    vehicle_return_normalizer=None,
) -> str:
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    obj = {
        "episode": episode,
        "model_family": "dreamer_ctde",
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "policy": actor.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "stats": stats,
        "cfg": cfg.__dict__,
        "extra_state": extra_state or {},
    }
    if target_critic is not None:
        obj["target_critic"] = target_critic.state_dict()
    if return_normalizer is not None:
        obj["return_normalizer"] = return_normalizer.state_dict()
    if vehicle_critic is not None:
        obj["vehicle_critic"] = vehicle_critic.state_dict()
    if vehicle_critic_optimizer is not None:
        obj["vehicle_critic_optimizer"] = vehicle_critic_optimizer.state_dict()
    if target_vehicle_critic is not None:
        obj["target_vehicle_critic"] = target_vehicle_critic.state_dict()
    if vehicle_return_normalizer is not None:
        obj["vehicle_return_normalizer"] = vehicle_return_normalizer.state_dict()
    path = os.path.join(cfg.checkpoint_dir, f"checkpoint_ep{episode:05d}.pt")
    torch.save(obj, path)
    torch.save(obj, os.path.join(cfg.checkpoint_dir, "latest.pt"))
    return path


def _load_state_dict_with_input_padding(module, state_dict, *, module_name: str) -> bool:
    try:
        module.load_state_dict(state_dict)
        return False
    except RuntimeError as exc:
        current_state = module.state_dict()
        migrated_state = dict(state_dict)
        changed = False
        for key, saved_tensor in state_dict.items():
            current_tensor = current_state.get(key)
            if not (torch.is_tensor(saved_tensor) and torch.is_tensor(current_tensor)):
                continue
            if tuple(saved_tensor.shape) == tuple(current_tensor.shape):
                continue
            if (
                saved_tensor.dim() == 2
                and current_tensor.dim() == 2
                and int(saved_tensor.shape[0]) == int(current_tensor.shape[0])
                and int(saved_tensor.shape[1]) < int(current_tensor.shape[1])
            ):
                padded = current_tensor.detach().clone()
                padded[:, : int(saved_tensor.shape[1])] = saved_tensor.to(
                    device=padded.device,
                    dtype=padded.dtype,
                )
                migrated_state[key] = padded
                changed = True
        if not changed:
            raise exc
        module.load_state_dict(migrated_state, strict=False)
        logger.info(
            "Loaded %s checkpoint with padded input weights for newly added critic context features.",
            module_name,
        )
        return True


def load_dreamer_checkpoint(
    path,
    actor,
    critic,
    actor_optimizer=None,
    critic_optimizer=None,
    *,
    reset_actor_optimizer: bool = False,
    target_critic=None,
    return_normalizer=None,
    vehicle_critic=None,
    vehicle_critic_optimizer=None,
    target_vehicle_critic=None,
    vehicle_return_normalizer=None,
):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = ckpt.get("actor", ckpt.get("policy"))
    if actor_state is None:
        raise RuntimeError(f"checkpoint has no actor/policy state: {path}")
    actor.load_state_dict(actor_state)
    critic_state = ckpt.get("critic")
    critic_migrated = False
    if critic_state is not None:
        critic_migrated = _load_state_dict_with_input_padding(critic, critic_state, module_name="critic")
    if target_critic is not None:
        target_state = ckpt.get("target_critic", critic_state)
        target_migrated = False
        if target_state is not None:
            target_migrated = _load_state_dict_with_input_padding(
                target_critic,
                target_state,
                module_name="target_critic",
            )
        if target_migrated and critic_state is not None:
            target_critic.load_state_dict(critic.state_dict())
            logger.info("Initialized migrated target_critic from the live critic state.")
        for param in target_critic.parameters():
            param.requires_grad_(False)
        target_critic.eval()
    if return_normalizer is not None and "return_normalizer" in ckpt:
        return_normalizer.load_state_dict(ckpt.get("return_normalizer"))
    vehicle_critic_migrated = False
    if vehicle_critic is not None:
        vehicle_critic_state = ckpt.get("vehicle_critic")
        if vehicle_critic_state is not None:
            vehicle_critic_migrated = _load_state_dict_with_input_padding(
                vehicle_critic,
                vehicle_critic_state,
                module_name="vehicle_critic",
            )
    if target_vehicle_critic is not None:
        target_vehicle_state = ckpt.get("target_vehicle_critic", ckpt.get("vehicle_critic"))
        target_vehicle_migrated = False
        if target_vehicle_state is not None:
            target_vehicle_migrated = _load_state_dict_with_input_padding(
                target_vehicle_critic,
                target_vehicle_state,
                module_name="target_vehicle_critic",
            )
        if target_vehicle_migrated and vehicle_critic is not None and ckpt.get("vehicle_critic") is not None:
            target_vehicle_critic.load_state_dict(vehicle_critic.state_dict())
            logger.info("Initialized migrated target_vehicle_critic from the live vehicle critic state.")
        for param in target_vehicle_critic.parameters():
            param.requires_grad_(False)
        target_vehicle_critic.eval()
    if vehicle_return_normalizer is not None and "vehicle_return_normalizer" in ckpt:
        vehicle_return_normalizer.load_state_dict(ckpt.get("vehicle_return_normalizer"))
    if actor_optimizer is not None and "actor_optimizer" in ckpt and not reset_actor_optimizer:
        actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    if critic_optimizer is not None and "critic_optimizer" in ckpt and not critic_migrated:
        critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
    elif critic_optimizer is not None and "critic_optimizer" in ckpt and critic_migrated:
        logger.info(
            "Skipped critic optimizer state because critic input weights were migrated to a wider context-conditioned shape."
        )
    if (
        vehicle_critic_optimizer is not None
        and "vehicle_critic_optimizer" in ckpt
        and not vehicle_critic_migrated
    ):
        vehicle_critic_optimizer.load_state_dict(ckpt["vehicle_critic_optimizer"])
    elif (
        vehicle_critic_optimizer is not None
        and "vehicle_critic_optimizer" in ckpt
        and vehicle_critic_migrated
    ):
        logger.info(
            "Skipped vehicle critic optimizer state because input weights were migrated to a wider shape."
        )
    return ckpt.get("episode", 0), ckpt.get("stats", {}), dict(ckpt.get("extra_state", {}) or {})


def _move_nested_tensors(value, *, device=None, cpu: bool = False):
    if torch.is_tensor(value):
        target = "cpu" if cpu else device
        return value.detach().to(target) if target is not None else value.detach()
    if isinstance(value, list):
        return [_move_nested_tensors(item, device=device, cpu=cpu) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors(item, device=device, cpu=cpu) for item in value)
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors(item, device=device, cpu=cpu)
            for key, item in value.items()
        }
    return value


def _restore_dreamer_extra_state(extra_state, *, device):
    extra_state = dict(extra_state or {})
    pending_trajs = [
        _move_nested_tensors(traj, device=device, cpu=False)
        for traj in list(extra_state.get("pending_trajs", []))
    ]
    for traj in pending_trajs:
        traj["restored_from_checkpoint"] = True
    update_id = int(extra_state.get("update_id", 0))
    return pending_trajs, update_id, extra_state


def count_fresh_pending_trajs(pending_trajs):
    return int(sum(0 if bool(traj.get("restored_from_checkpoint", False)) else 1 for traj in pending_trajs))


def capture_rng_state(*, rng):
    state = {
        "numpy_bit_generator_state": copy.deepcopy(rng.bit_generator.state),
        "python_random_state": copy.deepcopy(random.getstate()),
        "torch_cpu_rng_state": torch.random.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(extra_state):
    extra_state = dict(extra_state or {})
    np_state = extra_state.get("numpy_bit_generator_state")
    py_state = extra_state.get("python_random_state")
    torch_cpu_state = extra_state.get("torch_cpu_rng_state")
    torch_cuda_states = extra_state.get("torch_cuda_rng_state_all")
    rng = None
    if np_state is not None:
        rng = np.random.default_rng()
        rng.bit_generator.state = np_state
    if py_state is not None:
        random.setstate(py_state)
    if torch_cpu_state is not None:
        torch.random.set_rng_state(torch_cpu_state)
    if torch_cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_states)
    return rng


def normalize_stats_history_defaults(stats_history):
    stats_history = dict(stats_history or {})
    scalar_series = (
        "ep_reward", "pg_loss", "value_loss", "entropy", "approx_kl",
        "t_av", "t_hdv", "t_all", "hdv_drift", "hdv_head_entropy",
        "adaptive_penalty",
        "terminal_t_all_pred", "terminal_bonus", "invalid_route_rate", "route_churn",
        "route_cost_aux_loss", "route_cost_aux_acc", "route_cost_aux_margin",
        "expected_cost_loss", "expected_cost_raw", "expected_cost_best_prob",
        "expected_cost_best_route0_frac", "expected_cost_nonzero_best_frac", "expected_cost_margin",
        "grpo_route_cost_loss", "grpo_route_cost_raw", "grpo_route_cost_best_prob",
        "grpo_route_cost_best_route0_frac", "grpo_route_cost_nonzero_best_frac",
        "grpo_route_cost_margin", "grpo_route_cost_adv_mean", "grpo_route_cost_adv_std",
        "grpo_rollout_loss", "grpo_rollout_cost", "grpo_rollout_adv_mean",
        "grpo_rollout_adv_std", "grpo_rollout_logprob", "grpo_rollout_selected_nonzero_frac",
        "grpo_combined_loss", "grpo_combined_pg_loss", "grpo_combined_kl_loss",
        "grpo_combined_adv_mean", "grpo_combined_adv_std", "grpo_combined_clip_frac",
        "grpo_policy_kl",
        "grpo_rollout_vehicle_cost_loss", "grpo_rollout_vehicle_cost",
        "grpo_rollout_vehicle_cost_adv_mean", "grpo_rollout_vehicle_cost_adv_std",
        "dreamer_base_reward",
        "dreamer_ac_loss", "dreamer_ac_adv_mean", "dreamer_ac_adv_std",
        "dreamer_ac_adv_min", "dreamer_ac_adv_max",
        "dreamer_ac_route_adv_mean", "dreamer_ac_route_adv_std",
        "dreamer_ac_combined_adv_mean", "dreamer_ac_combined_adv_std",
        "dreamer_ac_rollout_argmax_mix_prob",
        "dreamer_ac_rollout_argmax_frac",
        "dreamer_ac_selected_nonzero_frac",
        "dreamer_ac_selected_route0_frac", "dreamer_ac_selected_route1_frac",
        "dreamer_ac_selected_route2_frac", "dreamer_ac_selected_route3_frac",
        "dreamer_ac_policy_argmax_nonzero_frac",
        "dreamer_ac_policy_argmax_route0_frac", "dreamer_ac_policy_argmax_route1_frac",
        "dreamer_ac_policy_argmax_route2_frac", "dreamer_ac_policy_argmax_route3_frac",
        "dreamer_ac_policy_argmax_max_route_frac", "dreamer_ac_policy_top1_prob_mean",
        "dreamer_actor_gate_skipped", "dreamer_actor_gate_max_route_frac",
        "dreamer_actor_gate_route0_frac", "dreamer_actor_gate_top1_prob",
        "dreamer_actor_gate_sample_count", "dreamer_actor_gate_latched",
        "dreamer_actor_trust_region_accepted", "dreamer_actor_trust_region_kl",
        "dreamer_actor_trust_region_retries", "dreamer_actor_trust_region_sample_count",
        "dreamer_actor_trust_region_argmax_change_frac",
        "dreamer_actor_trust_region_argmax_max_route_frac",
        "dreamer_actor_trust_region_top1_prob",
        "dreamer_actor_anchor_active", "dreamer_actor_anchor_accepted",
        "dreamer_actor_anchor_kl", "dreamer_actor_anchor_argmax_change_frac",
        "dreamer_actor_anchor_argmax_max_route_frac", "dreamer_actor_anchor_top1_prob",
        "dreamer_actor_anchor_sample_count",
        "dreamer_ac_sample_count",
        "dreamer_ac_policy_kl", "dreamer_ac_clip_frac", "dreamer_ac_weight_mean",
        "dreamer_ac_uses_sample_adv", "dreamer_ac_entropy_loss",
        "dreamer_ac_entropy_floor_loss", "dreamer_ac_entropy_target",
        "dreamer_ac_entropy_valid_target_mean",
        "dreamer_ac_entropy_effective_route_count", "dreamer_ac_entropy_cost_mask_frac",
        "vehicle_value_loss", "vehicle_critic_target_mean", "vehicle_critic_target_std",
        "vehicle_critic_explained_variance", "vehicle_return_norm_mean", "vehicle_return_norm_std",
        "vehicle_reward_mean", "vehicle_reward_std",
        "vehicle_route_progress_reward_mean", "vehicle_route_progress_reward_std",
        "vehicle_route_credit_reward_mean", "vehicle_route_credit_reward_std",
        "vehicle_social_reward_mean", "vehicle_social_reward_std",
        "vehicle_return_discount",
        "critic_target_mean", "critic_target_std", "critic_explained_variance",
        "critic_return_discount",
        "return_norm_mean", "return_norm_std",
        "dreamer_rollout_reward", "dreamer_rollout_reward_cost",
        "dreamer_rollout_reward_pressure_cost", "dreamer_rollout_reward_cav_proxy_cost",
        "hdv_response_av_pressure_sum", "hdv_response_background_sum", "hdv_response_background_active_edges",
        "avg_active_av", "skipped_actor_trajs", "n_valid_paths", "n_valid_path_agents",
        "n_actor_steps", "reward_grad_steps", "entropy_grad_steps", "max_active_av",
    )
    for key in scalar_series:
        stats_history.setdefault(key, [])
    stats_history.setdefault("traj_diagnostics", [])
    stats_history.setdefault("update_diagnostics", [])
    return stats_history


def maybe_resume_dreamer_checkpoint(
    *,
    resume_ckpt,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    device,
    reset_actor_optimizer=False,
    target_critic=None,
    return_normalizer=None,
    vehicle_critic=None,
    vehicle_critic_optimizer=None,
    target_vehicle_critic=None,
    vehicle_return_normalizer=None,
    load_checkpoint_fn: Optional[Callable[..., tuple[Any, Any, Any]]] = None,
):
    if not resume_ckpt or not os.path.exists(resume_ckpt):
        return None
    ckpt_state = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    if ckpt_state.get("model_family") != "dreamer_ctde" and "actor" not in ckpt_state:
        return None
    actor_state_before = copy.deepcopy(actor.state_dict())
    critic_state_before = copy.deepcopy(critic.state_dict())
    target_critic_state_before = copy.deepcopy(target_critic.state_dict()) if target_critic is not None else None
    return_normalizer_state_before = (
        copy.deepcopy(return_normalizer.state_dict()) if return_normalizer is not None else None
    )
    vehicle_critic_state_before = copy.deepcopy(vehicle_critic.state_dict()) if vehicle_critic is not None else None
    target_vehicle_critic_state_before = (
        copy.deepcopy(target_vehicle_critic.state_dict()) if target_vehicle_critic is not None else None
    )
    vehicle_return_normalizer_state_before = (
        copy.deepcopy(vehicle_return_normalizer.state_dict()) if vehicle_return_normalizer is not None else None
    )
    actor_opt_state_before = copy.deepcopy(actor_optimizer.state_dict()) if actor_optimizer is not None else None
    critic_opt_state_before = copy.deepcopy(critic_optimizer.state_dict()) if critic_optimizer is not None else None
    vehicle_critic_opt_state_before = (
        copy.deepcopy(vehicle_critic_optimizer.state_dict())
        if vehicle_critic_optimizer is not None
        else None
    )
    try:
        loader = load_checkpoint_fn or load_dreamer_checkpoint
        start_episode, stats_history, extra_state = loader(
            resume_ckpt,
            actor,
            critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            reset_actor_optimizer=reset_actor_optimizer,
            target_critic=target_critic,
            return_normalizer=return_normalizer,
            vehicle_critic=vehicle_critic,
            vehicle_critic_optimizer=vehicle_critic_optimizer,
            target_vehicle_critic=target_vehicle_critic,
            vehicle_return_normalizer=vehicle_return_normalizer,
        )
        logger.info("Resumed Dreamer checkpoint from ep %d", start_episode)
        if reset_actor_optimizer:
            logger.info("Actor optimizer state reset on resume; actor weights were still loaded.")
        stats_history = normalize_stats_history_defaults(stats_history)
        pending_trajs, update_id, restored_extra_state = _restore_dreamer_extra_state(extra_state, device=device)
        return {
            "start_episode": int(start_episode),
            "stats_history": stats_history,
            "pending_trajs": pending_trajs,
            "update_id": int(update_id),
            "extra_state": restored_extra_state,
            "restored_rng": restore_rng_state(extra_state),
        }
    except (RuntimeError, ValueError) as exc:
        actor.load_state_dict(actor_state_before)
        critic.load_state_dict(critic_state_before)
        if target_critic is not None and target_critic_state_before is not None:
            target_critic.load_state_dict(target_critic_state_before)
            for param in target_critic.parameters():
                param.requires_grad_(False)
            target_critic.eval()
        if return_normalizer is not None and return_normalizer_state_before is not None:
            return_normalizer.load_state_dict(return_normalizer_state_before)
        if vehicle_critic is not None and vehicle_critic_state_before is not None:
            vehicle_critic.load_state_dict(vehicle_critic_state_before)
        if target_vehicle_critic is not None and target_vehicle_critic_state_before is not None:
            target_vehicle_critic.load_state_dict(target_vehicle_critic_state_before)
            for param in target_vehicle_critic.parameters():
                param.requires_grad_(False)
            target_vehicle_critic.eval()
        if (
            vehicle_return_normalizer is not None
            and vehicle_return_normalizer_state_before is not None
        ):
            vehicle_return_normalizer.load_state_dict(vehicle_return_normalizer_state_before)
        if actor_optimizer is not None and actor_opt_state_before is not None:
            actor_optimizer.load_state_dict(actor_opt_state_before)
        if critic_optimizer is not None and critic_opt_state_before is not None:
            critic_optimizer.load_state_dict(critic_opt_state_before)
        if vehicle_critic_optimizer is not None and vehicle_critic_opt_state_before is not None:
            vehicle_critic_optimizer.load_state_dict(vehicle_critic_opt_state_before)
        logger.warning(
            "Failed to resume Dreamer checkpoint %s; starting fresh. Error: %s",
            resume_ckpt,
            exc,
        )
        return None
