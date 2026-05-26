# train_dynamics.py
import os
import sys
import time
import random
import argparse
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "urban_routing"))

from model import Dynamics, ActionDepartGNNEncoder, ActionEncoder, EdgeGNNActionEncoder, route_indices_to_fracs
from flow_dataset import FlowDataset
from urban_routing.experiment_settings import add_settings_arg, apply_stage_defaults, parse_settings_arg
from urban_routing.flow_projector import EdgeRegistry, FlowProjector, RouteRegistry
from urban_routing.stackelberg import BPRLayer, StackelbergLayer, decode_bg_volume

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def seed_everything(seed: int):
    s = int(seed) % (2**32)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def worker_init_fn(worker_id: int):
    info = torch.utils.data.get_worker_info()
    seed_everything(info.seed)


def init_distributed() -> tuple[bool, int, int, int]:
    rank, world_size, local_rank = get_dist_info()
    ddp = world_size > 1
    if ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return ddp, rank, world_size, local_rank


def save_ckpt(
    path: Path,
    *,
    step: int,
    epoch: int,
    dyn_model,
    opt,
    scaler,
    args: argparse.Namespace,
    flow_projector=None,
    action_compressor: Optional[Dict[str, Any]] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "step": step,
        "epoch": epoch,
        "dynamics": (dyn_model.module.state_dict() if hasattr(dyn_model, "module") else dyn_model.state_dict()),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    if action_compressor is not None:
        obj["action_compressor"] = action_compressor
    if flow_projector is not None:
        obj["flow_projector"] = (
            flow_projector.module.state_dict() if hasattr(flow_projector, "module")
            else flow_projector.state_dict()
        )
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def load_ckpt(
    path: Path,
    *,
    dyn_model,
    opt,
    scaler,
    flow_projector=None,
    strict: bool = True,
    load_optimizer: bool = True,
    load_scaler: bool = True,
    weights_only: bool = False,
) -> tuple[int, int]:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["dynamics"]
    dyn_target = dyn_model.module if hasattr(dyn_model, "module") else dyn_model
    if not bool(strict):
        # Filter out keys with shape mismatches (strict=False only handles missing/extra keys)
        cur_sd = dyn_target.state_dict()
        filtered_state, skipped_dyn = {}, []
        for k, v in state.items():
            if k in cur_sd and cur_sd[k].shape != v.shape:
                skipped_dyn.append(f"{k}: ckpt={tuple(v.shape)} vs model={tuple(cur_sd[k].shape)}")
            else:
                filtered_state[k] = v
        if skipped_dyn:
            print(f"[load_ckpt] Dynamics: skipped {len(skipped_dyn)} shape-mismatched keys: {skipped_dyn}")
        state = filtered_state
    incompatible = dyn_target.load_state_dict(state, strict=bool(strict))
    if not bool(strict):
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        if missing or unexpected:
            print(
                f"[load_ckpt] Non-strict dynamics load: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    if bool(load_optimizer) and (not bool(weights_only)) and ("opt" in ckpt):
        try:
            opt.load_state_dict(ckpt["opt"])
        except Exception as exc:
            print(f"[load_ckpt] WARNING: optimizer state load failed, using fresh optimizer. err={exc}")
    if bool(load_scaler) and (not bool(weights_only)) and scaler is not None and ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as exc:
            print(f"[load_ckpt] WARNING: scaler state load failed, using fresh scaler. err={exc}")
    if flow_projector is not None and "flow_projector" in ckpt:
        fp_target = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
        fp_sd = ckpt["flow_projector"]
        if not bool(strict):
            # Filter out keys with shape mismatches (strict=False only handles missing/extra keys)
            cur_sd = fp_target.state_dict()
            filtered_sd = {}
            skipped = []
            for k, v in fp_sd.items():
                if k in cur_sd and cur_sd[k].shape != v.shape:
                    skipped.append(f"{k}: ckpt={tuple(v.shape)} vs model={tuple(cur_sd[k].shape)}")
                else:
                    filtered_sd[k] = v
            if skipped:
                print(f"[load_ckpt] FlowProjector: skipped {len(skipped)} shape-mismatched keys: {skipped}")
            fp_sd = filtered_sd
        incompatible_fp = fp_target.load_state_dict(
            fp_sd,
            strict=bool(strict),
        )
        if not bool(strict):
            missing_fp = getattr(incompatible_fp, "missing_keys", [])
            unexpected_fp = getattr(incompatible_fp, "unexpected_keys", [])
            if missing_fp or unexpected_fp:
                print(
                    f"[load_ckpt] Non-strict FlowProjector load: "
                    f"missing={len(missing_fp)}, unexpected={len(unexpected_fp)}"
                )
    if bool(weights_only):
        return 0, 0
    return int(ckpt.get("step", 0)), int(ckpt.get("epoch", 0))


def _fit_linear_pca(actions: torch.Tensor, n_components: int) -> Dict[str, Any]:
    """
    Fit a linear PCA projector x -> (x - mean) @ proj.

    Args:
        actions: (N, A) float tensor on CPU.
        n_components: requested output dimension.
    Returns:
        dict with mean/proj and summary stats.
    """
    if actions.dim() != 2:
        raise ValueError(f"actions must be (N,A), got {tuple(actions.shape)}")
    X = actions.to(torch.float32).clone()
    N, A = int(X.shape[0]), int(X.shape[1])
    if N < 2:
        raise ValueError(f"Need at least 2 samples to fit PCA, got N={N}")
    d = int(max(1, min(int(n_components), A, N)))

    mean = X.mean(dim=0, keepdim=True)   # (1,A)
    Xc = X - mean
    # Full SVD is reliable at this scale (N~10^3-10^4, A~10^3).
    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    proj = Vh[:d].T.contiguous()         # (A,d)

    denom = max(1, N - 1)
    eig = (S * S) / float(denom)
    total_var = float(eig.sum().item()) if eig.numel() > 0 else 0.0
    explained = float(eig[:d].sum().item()) if eig.numel() > 0 else 0.0
    explained_ratio = (explained / total_var) if total_var > 0 else 0.0

    return {
        "type": "pca_linear",
        "input_dim": A,
        "output_dim": d,
        "mean": mean.squeeze(0).contiguous(),
        "proj": proj,
        "explained_variance_ratio": explained_ratio,
        "n_samples": N,
    }


def _apply_linear_action_compressor(actions: torch.Tensor, comp: Dict[str, Any]) -> torch.Tensor:
    if comp is None:
        return actions
    ctype = str(comp.get("type", ""))
    if ctype != "pca_linear":
        raise ValueError(f"Unsupported action compressor type: {ctype}")
    mean = comp["mean"].to(device=actions.device, dtype=actions.dtype)
    proj = comp["proj"].to(device=actions.device, dtype=actions.dtype)
    in_dim = int(comp.get("input_dim", mean.numel()))
    if int(actions.shape[-1]) != in_dim:
        raise ValueError(
            f"Compressor input_dim mismatch: expected {in_dim}, got {int(actions.shape[-1])}"
        )
    flat = actions.reshape(-1, in_dim)
    out = (flat - mean.view(1, -1)) @ proj
    return out.view(*actions.shape[:-1], int(proj.shape[1]))


def _build_model_actions(
    *,
    act_raw: torch.Tensor,
    depart_b: Optional[torch.Tensor] = None,
    context_b: Optional[torch.Tensor] = None,
    action_compressor: Optional[Dict[str, Any]] = None,
    depart_compressor: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    parts = []
    if action_compressor is not None:
        parts.append(_apply_linear_action_compressor(act_raw, action_compressor))
    else:
        parts.append(act_raw)
    if depart_b is not None:
        if depart_compressor is not None:
            parts.append(_apply_linear_action_compressor(depart_b, depart_compressor))
        else:
            parts.append(depart_b)
    if context_b is not None:
        parts.append(context_b)
    model_act = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    actions = torch.zeros_like(model_act)
    actions[:, 1:] = model_act[:, :-1]
    return actions


def _build_unshifted_model_actions(
    *,
    act_raw: torch.Tensor,
    depart_b: Optional[torch.Tensor] = None,
    context_b: Optional[torch.Tensor] = None,
    action_compressor: Optional[Dict[str, Any]] = None,
    depart_compressor: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    parts = []
    if action_compressor is not None:
        parts.append(_apply_linear_action_compressor(act_raw, action_compressor))
    else:
        parts.append(act_raw)
    if depart_b is not None:
        if depart_compressor is not None:
            parts.append(_apply_linear_action_compressor(depart_b, depart_compressor))
        else:
            parts.append(depart_b)
    if context_b is not None:
        parts.append(context_b)
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _build_graph_cond_model_actions(
    *,
    act_raw: torch.Tensor,
    depart_b: Optional[torch.Tensor] = None,
    context_b: Optional[torch.Tensor] = None,
    hdv_context_b: Optional[torch.Tensor] = None,
) -> tuple:
    """Causally align graph_cond action/depart/context with state targets.

    FlowDataset returns decision fields at the same index as S_t.  Dynamics
    targets S_t, so the conditioning field for token t must be A_{t-1}.  The
    flat action path does this in _build_model_actions(); graph_cond keeps AV
    action, depart, and context as separate edge-aligned tensors, so it needs
    the same shift before entering CFM training.
    """

    def _shift(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        shifted = torch.zeros_like(x)
        if x.shape[1] > 1:
            shifted[:, 1:] = x[:, :-1]
        return shifted

    action_shifted = _shift(act_raw)
    depart_shifted = _shift(depart_b)
    context_shifted = _shift(context_b)
    hdv_shifted = _shift(hdv_context_b)
    if hdv_shifted is not None:
        return action_shifted, depart_shifted, context_shifted, hdv_shifted
    return action_shifted, depart_shifted, context_shifted


def _compose_transition_hdv_action(
    *,
    act_raw: torch.Tensor,
    hdv_target: Optional[torch.Tensor],
    mode: str,
    weight: float,
    scale: float,
) -> torch.Tensor:
    """Optionally add an HDV response background to transition actions.

    This is intentionally separate from the HDV auxiliary head.  The head
    predicts an episode-level HDV response, while the transition model only
    benefits from that signal if it is trained with the same composed action
    semantics that MARL rollout uses at inference time.
    """
    mode_l = str(mode or "none").strip().lower()
    if mode_l in {"", "none", "off", "disabled"} or hdv_target is None or float(weight) == 0.0:
        return act_raw
    if hdv_target.shape[:2] != act_raw.shape[:2] or hdv_target.shape[-1] != act_raw.shape[-1]:
        raise ValueError(
            "transition HDV action shape mismatch: "
            f"act_raw={tuple(act_raw.shape)}, hdv_target={tuple(hdv_target.shape)}"
        )
    hdv = hdv_target.to(device=act_raw.device, dtype=act_raw.dtype).clamp_min(0.0)
    if mode_l in {"episode_mean", "episode_response", "response"}:
        # FlowDataset can already return an episode-mean aux target.  Normalize
        # again here so callers can also use raw per-step hdv_edge_action data.
        hdv = hdv.mean(dim=1, keepdim=True).expand_as(hdv)
        denom = hdv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hdv = hdv / denom
    elif mode_l in {"step", "per_step"}:
        denom = hdv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hdv = hdv / denom
    elif mode_l in {"raw", "raw_step"}:
        pass
    else:
        raise ValueError(
            "--transition_hdv_action_mode must be one of none, step, raw, "
            "episode_mean, episode_response, or response; got "
            f"{mode!r}"
        )
    return act_raw + float(weight) * float(scale) * hdv


def _build_transition_hdv_context(
    *,
    act_raw: torch.Tensor,
    hdv_target: Optional[torch.Tensor],
    mode: str,
    weight: float,
    scale: float,
) -> Optional[torch.Tensor]:
    """Build a separate edge-aligned HDV response context for graph_cond dynamics."""
    mode_l = str(mode or "none").strip().lower()
    if mode_l in {"", "none", "off", "disabled"} or hdv_target is None or float(weight) == 0.0:
        return None
    if hdv_target.shape[:2] != act_raw.shape[:2] or hdv_target.shape[-1] != act_raw.shape[-1]:
        raise ValueError(
            "transition HDV context shape mismatch: "
            f"act_raw={tuple(act_raw.shape)}, hdv_target={tuple(hdv_target.shape)}"
        )
    hdv = hdv_target.to(device=act_raw.device, dtype=act_raw.dtype).clamp_min(0.0)
    if mode_l in {"episode_mean", "episode_response", "response"}:
        hdv = hdv.mean(dim=1, keepdim=True).expand_as(hdv)
    elif mode_l in {"step", "per_step"}:
        pass
    elif mode_l in {"raw", "raw_step"}:
        return float(weight) * hdv
    else:
        raise ValueError(
            "--transition_hdv_context_mode must be one of none, step, raw, "
            "episode_mean, episode_response, or response; got "
            f"{mode!r}"
        )
    denom = hdv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return float(weight) * float(scale) * (hdv / denom)


def _build_hdv_policy_condition_actions(
    *,
    mode: str,
    action_encoder_type: str,
    act_raw: torch.Tensor,
    episode_act_raw: Optional[torch.Tensor],
    depart_b: Optional[torch.Tensor],
    context_b: Optional[torch.Tensor],
    model_action_uses_depart_context: bool,
    action_compressor: Optional[Dict[str, Any]],
    depart_compressor: Optional[Dict[str, Any]],
) -> Optional[torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """Build the action condition used only by the HDV-response head.

    The transition model still trains on causally shifted per-step actions.
    The HDV head, however, is meant to predict an episode-level HDV response
    from aggregate AV pressure.  Keeping this path unshifted avoids training
    the head on a different input semantics than ``predict_episode_hdv_response``.
    """
    mode = str(mode or "step").strip().lower()
    if mode in {"", "step", "per_step", "legacy"}:
        return None
    if mode not in {"window_mean", "episode_mean", "episode_response", "response"}:
        raise ValueError(
            "--hdv_policy_action_mode must be one of step, window_mean, "
            "episode_mean, episode_response, or response; got "
            f"{mode!r}"
        )

    if mode in {"episode_mean", "episode_response", "response"} and episode_act_raw is not None:
        act_cond = episode_act_raw
    else:
        act_cond = act_raw.mean(dim=1, keepdim=True).expand_as(act_raw)
    if mode in {"episode_mean", "episode_response", "response"}:
        # predict_episode_hdv_response() is intentionally an episode AV-pressure
        # interface.  Keep depart/context zeroed here so the head is not trained
        # with conditioning fields that are absent at MARL inference time.
        dep_cond = None
        ctx_cond = None
    else:
        dep_cond = None if depart_b is None else depart_b.mean(dim=1, keepdim=True).expand_as(depart_b)
        ctx_cond = None if context_b is None else context_b.mean(dim=1, keepdim=True).expand_as(context_b)

    if action_encoder_type == "graph_cond":
        return act_cond, dep_cond, ctx_cond
    return _build_unshifted_model_actions(
        act_raw=act_cond,
        depart_b=dep_cond if model_action_uses_depart_context else None,
        context_b=ctx_cond if model_action_uses_depart_context else None,
        action_compressor=action_compressor,
        depart_compressor=depart_compressor if model_action_uses_depart_context else None,
    )


def _shuffle_model_actions(actions):
    def _shuffle_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if x.shape[0] > 1:
            return torch.roll(x, shifts=1, dims=0)
        if x.dim() >= 2 and x.shape[1] > 1:
            return torch.roll(x, shifts=1, dims=1)
        return x

    if isinstance(actions, tuple):
        return tuple(_shuffle_tensor(x) for x in actions)
    return _shuffle_tensor(actions)


def _slice_model_actions_time(actions, stop: int):
    """Slice model actions on the time axis while preserving graph_cond tuples."""
    if actions is None:
        return None
    t_stop = int(stop)
    if isinstance(actions, tuple):
        return tuple(None if a is None else a[:, :t_stop] for a in actions)
    return actions[:, :t_stop]


def _infer_model_action_dim(
    *,
    raw_action_dim: int,
    action_compressor: Optional[Dict[str, Any]] = None,
    depart_compressor: Optional[Dict[str, Any]] = None,
    depart_dim: int = 0,
    context_dim: int = 0,
) -> int:
    dim = int(action_compressor["output_dim"]) if action_compressor is not None else int(raw_action_dim)
    if depart_compressor is not None:
        dim += int(depart_compressor["output_dim"])
    else:
        dim += int(depart_dim)
    dim += int(context_dim)
    return dim


def _maybe_load_action_compressor_from_ckpt(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    comp = ckpt.get("action_compressor")
    if not isinstance(comp, dict):
        return None
    if "mean" in comp:
        comp["mean"] = torch.as_tensor(comp["mean"], dtype=torch.float32).cpu()
    if "proj" in comp:
        comp["proj"] = torch.as_tensor(comp["proj"], dtype=torch.float32).cpu()
    return comp


def _maybe_load_depart_compressor_from_ckpt(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None
    comp = ckpt.get("depart_compressor")
    if not isinstance(comp, dict):
        return None
    if "mean" in comp:
        comp["mean"] = torch.as_tensor(comp["mean"], dtype=torch.float32).cpu()
    if "proj" in comp:
        comp["proj"] = torch.as_tensor(comp["proj"], dtype=torch.float32).cpu()
    return comp


def _scheduled_sampling_prob(*, step: int, enabled: bool, warmup: int, end: float) -> float:
    if (not enabled) or warmup <= 0:
        return 0.0
    frac = min(max(float(step) / float(warmup), 0.0), 1.0)
    return float(end) * frac



def _build_rollout_targets(z1: torch.Tensor, horizon: int) -> list[torch.Tensor]:
    _, T = z1.shape[:2]
    H = max(1, min(int(horizon), T - 1))
    return [z1[:, h:h + (T - H)] for h in range(1, H + 1)]


def _build_rollout_inputs(
    act_raw: torch.Tensor,
    depart_b: Optional[torch.Tensor],
    context_b: Optional[torch.Tensor],
    *,
    horizon: int,
) -> list[dict[str, Optional[torch.Tensor]]]:
    T = act_raw.shape[1]
    H = max(1, min(int(horizon), T - 1))
    span = T - H
    out = []
    for h in range(H):
        out.append({
            "act": act_raw[:, h:h + span],
            "depart": None if depart_b is None else depart_b[:, h:h + span],
            "context": None if context_b is None else context_b[:, h:h + span],
        })
    return out


def _parse_adm_horizons(value: str | int | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),) if int(value) > 0 else ()
    text = str(value).strip()
    if not text:
        return ()
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        h = int(part)
        if h <= 0:
            raise ValueError(f"ADM horizons must be positive, got {h}")
        out.append(h)
    return tuple(sorted(set(out)))


def _build_adm_context_target_windows(z: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ADM-style direct k-step windows: repeated z_t context -> z_{t+k} target."""
    B, T = z.shape[:2]
    h = int(horizon)
    if h <= 0 or T <= h:
        raise ValueError(f"Cannot build ADM windows for T={T}, horizon={h}")
    span = T - h
    start = z[:, :span]
    target = z[:, h:h + span]
    ctx = start.unsqueeze(2).expand(B, span, h, *z.shape[2:]).reshape(B * span, h, *z.shape[2:])
    target = target.reshape(B * span, 1, *z.shape[2:])
    return ctx, target


def _build_adm_action_windows(x: Optional[torch.Tensor], horizon: int) -> Optional[torch.Tensor]:
    """Window A_t..A_{t+k-1}; repeat the last causal action for the final flow token."""
    if x is None:
        return None
    B, T = x.shape[:2]
    h = int(horizon)
    if h <= 0 or T <= h:
        raise ValueError(f"Cannot build ADM action windows for T={T}, horizon={h}")
    span = T - h
    chunks = [x[:, j:j + span] for j in range(h)]
    chunks.append(x[:, h - 1:h - 1 + span])
    return torch.stack(chunks, dim=2).reshape(B * span, h + 1, *x.shape[2:])


def _slice_adm_model_actions(actions, idx: torch.Tensor):
    if actions is None:
        return None
    if isinstance(actions, tuple):
        return tuple(None if a is None else a.index_select(0, idx) for a in actions)
    return actions.index_select(0, idx)


def _select_rollout_state(
    *,
    z_gt: torch.Tensor,
    z_pred: torch.Tensor,
    use_pred_mask: torch.Tensor,
) -> torch.Tensor:
    mask = use_pred_mask.to(device=z_gt.device).view(-1, 1, 1)
    return torch.where(mask, z_pred, z_gt)


def _edge_action_values_for_loss(
    actions: Optional[torch.Tensor | tuple],
    *,
    batch_size: int,
    time_steps: int,
    n_spatial: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    edge_actions = actions[0] if isinstance(actions, tuple) else actions
    if edge_actions is None or not isinstance(edge_actions, torch.Tensor):
        return None
    if edge_actions.dim() == 2:
        if int(edge_actions.shape[0]) not in (1, batch_size):
            return None
        edge_actions = edge_actions.unsqueeze(1)
    if edge_actions.dim() != 3:
        return None
    if int(edge_actions.shape[0]) not in (1, batch_size):
        return None
    if int(edge_actions.shape[0]) == 1 and batch_size > 1:
        edge_actions = edge_actions.expand(batch_size, -1, -1)
    action_T = int(edge_actions.shape[1])
    if action_T == 1 and time_steps > 1:
        edge_actions = edge_actions.expand(-1, time_steps, -1)
    elif action_T > time_steps:
        edge_actions = edge_actions[:, -time_steps:, :]
    elif action_T != time_steps:
        return None
    if int(edge_actions.shape[-1]) < int(n_spatial):
        return None
    return edge_actions[..., : int(n_spatial)].to(device=device, dtype=dtype)


def action_reconstruction_loss(
    *,
    action_recon: Optional[torch.Tensor],
    actions: Optional[torch.Tensor | tuple],
    n_spatial: int,
    loss_weight: float,
    action_eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = action_recon.device if action_recon is not None else torch.device("cpu")
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if action_recon is None or float(loss_weight) <= 0.0:
        return zero, {}
    if action_recon.dim() != 4 or int(action_recon.shape[-1]) != 2:
        return zero, {}

    B, T = int(action_recon.shape[0]), int(action_recon.shape[1])
    target = _edge_action_values_for_loss(
        actions,
        batch_size=B,
        time_steps=T,
        n_spatial=n_spatial,
        device=action_recon.device,
        dtype=torch.float32,
    )
    if target is None:
        return zero, {}

    logits = action_recon[..., 0].float()
    pred_value = action_recon[..., 1].float()
    active = (target.abs() > float(action_eps)).float()
    active_frac = active.mean()
    active_bce = F.binary_cross_entropy_with_logits(logits, active)
    active_count = active.sum()
    if float(active_count.item()) > 0.0:
        value_mse = ((pred_value - target.float()).pow(2) * active).sum() / active_count.clamp_min(1.0)
    else:
        value_mse = zero
    loss = float(loss_weight) * (active_bce + value_mse)
    return loss, {
        "action_recon_loss": loss.detach(),
        "action_recon_active_bce": active_bce.detach(),
        "action_recon_value_mse": value_mse.detach(),
        "action_recon_active_frac": active_frac.detach(),
    }



def cfm_loss(
    dynamics: torch.nn.Module,
    *,
    z1: torch.Tensor,                    # (B,T,Sz,Dz) clean target flow tokens
    actions: Optional[torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    act_mask: Optional[torch.Tensor],    # (A,) or None
    hdv_actions: Optional[torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    agent_tokens: Optional[torch.Tensor] = None,
    hdv_target: Optional[torch.Tensor] = None,   # (B,T,D): route(K) or edge(E)
    hdv_policy_weight: float = 0.0,
    stackelberg: Optional[StackelbergLayer] = None,
    flow_projector: Optional[torch.nn.Module] = None,
    eq_weight: float = 0.0,
    stackelberg_hdv_weight: float = 0.0,
    action_compressor: Optional[Dict[str, Any]] = None,
    av_action_dim: Optional[int] = None,
    action_reconstruction_weight: float = 0.0,
    action_reconstruction_eps: float = 1e-6,
    flow_targets: Optional[torch.Tensor] = None,
    cfm_active_weight: float = 0.0,
    cfm_busy_weight: float = 0.0,
    cfm_busy_threshold: float = 0.05,
    cfm_active_threshold: float = 0.0,
    cfm_max_weight: float = 50.0,
    volume_index: int = 2,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Conditional Flow Matching loss.

    Samples t ~ U[0,1], creates interpolated state x_t = (1-t)*x_0 + t*x_1,
    and trains the velocity field to predict (x_1 - x_0).

    When stackelberg is provided:
      - Computes joint AV+HDV action via Stackelberg equilibrium
      - Adds L_eq (fixed-point residual) as auxiliary loss
      - Optionally adds L_hdv (CE between Stackelberg HDV output and real HDV data)
    """
    device = z1.device
    B, T = z1.shape[:2]

    x0 = torch.randn_like(z1)
    t = torch.rand(B, T, device=device, dtype=torch.float32)

    t_expand = t[..., None, None]
    x_t = (1.0 - t_expand) * x0 + t_expand * z1
    target_v = z1 - x0

    want_hdv = (hdv_target is not None) and (float(hdv_policy_weight) > 0.0)
    want_action_recon = float(action_reconstruction_weight) > 0.0
    action_recon = None
    hdv_logits = None
    if want_hdv and hdv_actions is not None:
        if want_action_recon:
            v_pred, _, action_recon = dynamics(
                actions, None, None, x_t,
                act_mask=act_mask,
                agent_tokens=agent_tokens,
                flow_time=t,
                return_action_recon=True,
            )
        else:
            v_pred, _ = dynamics(
                actions, None, None, x_t,
                act_mask=act_mask,
                agent_tokens=agent_tokens,
                flow_time=t,
            )
        _, _, hdv_logits = dynamics(
            hdv_actions, None, None, x_t,
            act_mask=act_mask,
            agent_tokens=agent_tokens,
            flow_time=t,
            return_hdv_policy=True,
        )
    elif want_hdv and want_action_recon:
        v_pred, _, hdv_logits, action_recon = dynamics(
            actions, None, None, x_t,
            act_mask=act_mask,
            agent_tokens=agent_tokens,
            flow_time=t,
            return_hdv_policy=True,
            return_action_recon=True,
        )
    elif want_hdv:
        v_pred, _, hdv_logits = dynamics(
            actions, None, None, x_t,
            act_mask=act_mask,
            agent_tokens=agent_tokens,
            flow_time=t,
            return_hdv_policy=True,
        )
    elif want_action_recon:
        v_pred, _, action_recon = dynamics(
            actions, None, None, x_t,
            act_mask=act_mask,
            agent_tokens=agent_tokens,
            flow_time=t,
            return_action_recon=True,
        )
    else:
        v_pred, _ = dynamics(
            actions, None, None, x_t,
            act_mask=act_mask,
            agent_tokens=agent_tokens,
            flow_time=t,
        )

    err2 = (v_pred.float() - target_v.float()).pow(2)
    edge_mse = err2.mean(dim=-1)
    edge_weight = _cfm_target_edge_weights(
        flow_targets=flow_targets,
        ref=edge_mse,
        volume_index=volume_index,
        active_weight=cfm_active_weight,
        busy_weight=cfm_busy_weight,
        busy_threshold=cfm_busy_threshold,
        active_threshold=cfm_active_threshold,
        max_weight=cfm_max_weight,
    )
    mse_per = edge_mse.mean(dim=-1)
    flow_mse = edge_mse.mean()
    if edge_weight is None:
        weighted_flow_mse = flow_mse
        loss = weighted_flow_mse
        cfm_weight_mean = torch.ones((), device=device, dtype=torch.float32)
        cfm_active_frac = torch.zeros((), device=device, dtype=torch.float32)
        cfm_busy_frac = torch.zeros((), device=device, dtype=torch.float32)
    else:
        weighted_flow_mse = (edge_mse * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)
        loss = weighted_flow_mse
        cfm_weight_mean = edge_weight.mean()
        target_vol = flow_targets[..., int(volume_index)].to(device=device, dtype=torch.float32).clamp_min(0.0)
        cfm_active_frac = (target_vol > float(cfm_active_threshold)).float().mean()
        cfm_busy_frac = (target_vol > float(cfm_busy_threshold)).float().mean()
    hdv_ce = torch.zeros((), device=device, dtype=torch.float32)
    if want_hdv and hdv_logits is not None:
        target = hdv_target.to(device=device, dtype=torch.float32)
        if target.shape != hdv_logits.shape:
            raise ValueError(
                "HDV target/logit shape mismatch in cfm_loss: "
                f"target={tuple(target.shape)} vs logits={tuple(hdv_logits.shape)}. "
                "Check --aux_action_key and --hdv_policy_dim."
            )
        target = torch.clamp(target, min=0.0)
        target_sum = target.sum(dim=-1, keepdim=True)
        uniform = torch.full_like(target, 1.0 / float(target.shape[-1]))
        target = torch.where(target_sum > 1e-8, target / target_sum.clamp_min(1e-8), uniform)
        log_probs = F.log_softmax(hdv_logits.float(), dim=-1)
        hdv_ce = -(target * log_probs).sum(dim=-1).mean()
        loss = loss + float(hdv_policy_weight) * hdv_ce

    action_recon_loss = torch.zeros((), device=device, dtype=torch.float32)
    action_recon_active_bce = torch.zeros((), device=device, dtype=torch.float32)
    action_recon_value_mse = torch.zeros((), device=device, dtype=torch.float32)
    action_recon_active_frac = torch.zeros((), device=device, dtype=torch.float32)
    if action_recon is not None and float(action_reconstruction_weight) > 0.0:
        action_recon_loss, action_recon_aux = action_reconstruction_loss(
            action_recon=action_recon,
            actions=actions,
            n_spatial=int(action_recon.shape[-2]),
            loss_weight=float(action_reconstruction_weight),
            action_eps=float(action_reconstruction_eps),
        )
        loss = loss + action_recon_loss
        action_recon_active_bce = action_recon_aux.get("action_recon_active_bce", action_recon_active_bce)
        action_recon_value_mse = action_recon_aux.get("action_recon_value_mse", action_recon_value_mse)
        action_recon_active_frac = action_recon_aux.get("action_recon_active_frac", action_recon_active_frac)

    # --- Stackelberg auxiliary losses ---
    eq_residual_val = torch.zeros((), device=device, dtype=torch.float32)
    stk_hdv_ce = torch.zeros((), device=device, dtype=torch.float32)
    if isinstance(actions, tuple):
        stackelberg = None
    if stackelberg is not None and flow_projector is not None:
        # Use the last timestep's interpolated state as proxy for current state
        z_sample = x_t[:, -1]  # (B, E, 64)
        # Average over batch for Stackelberg computation
        z_avg = z_sample.mean(dim=0)  # (E, 64)
        bg_vol = decode_bg_volume(z_avg, flow_projector, stackelberg.capacities)  # (E,)
        # Actions: last timestep AV action → edge fracs → OD fracs
        if actions is not None:
            av_actions = actions[:, -1]
            if av_action_dim is not None and int(av_action_dim) > 0:
                av_actions = av_actions[..., :int(av_action_dim)]
            av_edge_avg = av_actions.mean(dim=0)  # (A,)
            # Decompress PCA-compressed actions back to edge-space
            if av_edge_avg.shape[0] != stackelberg.n_edges and action_compressor is not None:
                proj = action_compressor["proj"].to(device=device, dtype=av_edge_avg.dtype)  # (E, pca_dim)
                mean = action_compressor["mean"].to(device=device, dtype=av_edge_avg.dtype)  # (E,)
                av_edge_avg = av_edge_avg @ proj.T + mean  # (E,)
            if av_edge_avg.shape[0] == stackelberg.n_edges:
                # Normalize decompressed action to valid edge distribution
                av_edge_avg = av_edge_avg.clamp(min=0.0)
                av_sum = av_edge_avg.sum()
                if av_sum > 1e-8:
                    av_edge_avg = av_edge_avg / av_sum
                else:
                    av_edge_avg = torch.full_like(av_edge_avg, 1.0 / float(av_edge_avg.shape[0]))
                n_av_total = float(stackelberg.total_av_count.item())
                av_edge_vol = av_edge_avg * n_av_total
                hdv_edge, _, eq_res = stackelberg(av_edge_vol, bg_vol)
                eq_residual_val = eq_res
                if float(eq_weight) > 0.0 and eq_residual_val.isfinite():
                    loss = loss + float(eq_weight) * eq_residual_val
                # Stackelberg HDV CE against real HDV target
                if hdv_target is not None and float(stackelberg_hdv_weight) > 0.0:
                    hdv_real = hdv_target[:, -1].mean(dim=0).to(device=device, dtype=torch.float32)
                    hdv_real = hdv_real.clamp(min=0.0)
                    hdv_sum = hdv_real.sum()
                    if hdv_sum > 1e-8:
                        hdv_real = hdv_real / hdv_sum
                    else:
                        hdv_real = torch.full_like(hdv_real, 1.0 / float(hdv_real.shape[0]))
                    # Force float32 to avoid AMP float16 issues with log of small values
                    hdv_edge_f32 = hdv_edge.float().clamp(min=1e-7)
                    hdv_edge_f32 = hdv_edge_f32 / hdv_edge_f32.sum().clamp(min=1e-7)
                    hdv_real_f32 = hdv_real.float()
                    log_hdv = hdv_edge_f32.log()
                    stk_hdv_ce = -(hdv_real_f32 * log_hdv).sum()
                    if not stk_hdv_ce.isfinite():
                        stk_hdv_ce = torch.zeros((), device=device, dtype=torch.float32)
                    else:
                        loss = loss + float(stackelberg_hdv_weight) * stk_hdv_ce

    aux = {
        "flow_mse": flow_mse.detach(),
        "weighted_flow_mse": weighted_flow_mse.detach(),
        "cfm_weight_mean": cfm_weight_mean.detach(),
        "cfm_active_edge_frac": cfm_active_frac.detach(),
        "cfm_busy_edge_frac": cfm_busy_frac.detach(),
        "t_mean": t.mean().detach(),
        "hdv_policy_ce": hdv_ce.detach(),
        "action_recon_loss": action_recon_loss.detach(),
        "action_recon_active_bce": action_recon_active_bce.detach(),
        "action_recon_value_mse": action_recon_value_mse.detach(),
        "action_recon_active_frac": action_recon_active_frac.detach(),
        "eq_residual": eq_residual_val.detach(),
        "stk_hdv_ce": stk_hdv_ce.detach(),
    }
    return loss, aux


def _cfm_target_edge_weights(
    *,
    flow_targets: Optional[torch.Tensor],
    ref: torch.Tensor,
    volume_index: int,
    active_weight: float,
    busy_weight: float,
    busy_threshold: float,
    active_threshold: float,
    max_weight: float,
) -> Optional[torch.Tensor]:
    """Build edge-wise CFM velocity weights from the true next-step volume target."""
    if flow_targets is None:
        return None
    if float(active_weight) == 0.0 and float(busy_weight) == 0.0:
        return None
    vidx = int(volume_index)
    if flow_targets.dim() < 4 or vidx < 0 or vidx >= int(flow_targets.shape[-1]):
        return None
    target_vol = flow_targets[..., vidx].to(device=ref.device, dtype=torch.float32).clamp_min(0.0)
    if tuple(target_vol.shape) != tuple(ref.shape):
        return None

    active = (target_vol > float(active_threshold)).to(dtype=ref.dtype)
    busy = (target_vol > float(busy_threshold)).to(dtype=ref.dtype)
    weight = torch.ones_like(ref, dtype=ref.dtype)
    weight = weight + float(active_weight) * active + float(busy_weight) * busy
    if float(max_weight) > 0.0:
        weight = weight.clamp(max=float(max_weight))
    return weight


def one_step_spatial_loss(
    *,
    flow_targets: torch.Tensor,
    spatial_weight: float,
    decoded_dyn: Optional[torch.Tensor] = None,
    inverse_dyn: Optional[torch.Tensor] = None,
    veh_index: int = 2,
    busy_threshold: float = 0.05,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = flow_targets.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if decoded_dyn is None:
        decoded_dyn = inverse_dyn
    if decoded_dyn is None or float(spatial_weight) <= 0.0:
        return zero, {}

    dyn_dim = int(decoded_dyn.shape[-1])
    vidx = int(veh_index)
    if dyn_dim <= 0 or vidx < 0 or vidx >= dyn_dim:
        return zero, {}

    target_dyn = flow_targets[..., :dyn_dim].to(dtype=decoded_dyn.dtype)
    if vidx >= int(target_dyn.shape[-1]):
        return zero, {}

    v_pred = decoded_dyn[..., vidx].float().clamp_min(0.0)
    v_true = target_dyn[..., vidx].float().clamp_min(0.0)
    busy_mask = ((v_pred > float(busy_threshold)) | (v_true > float(busy_threshold))).float()
    busy_frac = busy_mask.mean()
    if float(busy_mask.sum().item()) <= 0.0:
        return zero, {
            "busy_edge_frac": busy_frac.detach(),
            "active_edge_precision": zero.detach(),
            "active_edge_recall": zero.detach(),
            "spatial_supervision_loss": zero.detach(),
        }

    vol_loss = (((v_pred - v_true).pow(2)) * busy_mask).sum() / busy_mask.sum().clamp_min(1.0)

    pred_active = (v_pred > float(busy_threshold)).float()
    true_active = (v_true > float(busy_threshold)).float()
    tp = (pred_active * true_active).sum()
    fp = (pred_active * (1.0 - true_active)).sum()
    fn = ((1.0 - pred_active) * true_active).sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    active_penalty = (1.0 - precision) + (1.0 - recall)

    loss = float(spatial_weight) * (vol_loss + 0.1 * active_penalty)
    aux = {
        "busy_edge_frac": busy_frac.detach(),
        "active_edge_precision": precision.detach(),
        "active_edge_recall": recall.detach(),
        "spatial_supervision_loss": loss.detach(),
        "busy_volume_mse": vol_loss.detach(),
    }
    return loss, aux


def _broadcast_edge_capacities(
    capacities: Optional[torch.Tensor],
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    if capacities is None:
        return None
    cap = capacities.to(device=ref.device, dtype=torch.float32).clamp_min(1.0)
    view_shape = [1] * (ref.dim() - 1) + [int(cap.shape[0])]
    return cap.view(*view_shape)


def _positive_pred_volume_for_loss(raw_volume: torch.Tensor) -> torch.Tensor:
    """Keep positive decoded volumes exact while preserving gradient for negatives."""
    raw = raw_volume.float()
    return torch.where(raw >= 0.0, raw, F.softplus(raw, beta=10.0))


def _positive_pred_speed_for_loss(raw_speed: torch.Tensor) -> torch.Tensor:
    """Keep positive decoded speed ratios exact while preserving gradient for negatives."""
    raw = raw_speed.float()
    return torch.where(raw >= 0.0, raw, F.softplus(raw, beta=1.0))


def _transform_decoded_targets(
    *,
    pred_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
    speed_index: int,
    volume_index: int,
    speed_target_mode: str,
    volume_target_mode: str,
    capacities: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_decoded.float()
    target = target_decoded.float()
    dyn_dim = int(pred.shape[-1])
    sidx = int(speed_index)
    vidx = int(volume_index)
    pred_out = pred
    target_out = target
    transformed = False

    speed_mode = str(speed_target_mode).lower()
    if speed_mode == "deficit" and 0 <= sidx < dyn_dim:
        if not transformed:
            pred_out = pred.clone()
            target_out = target.clone()
            transformed = True
        pred_out[..., sidx] = 1.0 - pred[..., sidx]
        target_out[..., sidx] = 1.0 - target[..., sidx]
    elif speed_mode == "slowdown" and 0 <= sidx < dyn_dim:
        pred_speed = _positive_pred_speed_for_loss(pred[..., sidx]).clamp_min(0.01)
        target_speed = target[..., sidx].clamp_min(0.1)
        if not transformed:
            pred_out = pred.clone()
            target_out = target.clone()
            transformed = True
        pred_out[..., sidx] = pred_speed.reciprocal() - 1.0
        target_out[..., sidx] = target_speed.reciprocal() - 1.0

    if str(volume_target_mode).lower() == "log1p" and 0 <= vidx < dyn_dim:
        pred_vol = _positive_pred_volume_for_loss(pred[..., vidx])
        target_vol = target[..., vidx].clamp_min(0.0)
        cap = _broadcast_edge_capacities(capacities, pred_vol)
        if cap is not None:
            pred_vol = pred_vol * cap
            target_vol = target_vol * cap
        if not transformed:
            pred_out = pred.clone()
            target_out = target.clone()
            transformed = True
        pred_out[..., vidx] = torch.log1p(pred_vol)
        target_out[..., vidx] = torch.log1p(target_vol)

    return pred_out, target_out


def _busy_edge_channel_metrics(
    *,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    pred_eval: torch.Tensor,
    target_eval: torch.Tensor,
    volume_index: int,
    speed_index: int,
    busy_threshold: float,
) -> Dict[str, torch.Tensor]:
    zero = torch.zeros((), device=pred_eval.device, dtype=torch.float32)
    dyn_dim = int(pred_eval.shape[-1])
    sidx = int(speed_index)
    vidx = int(volume_index)
    if dyn_dim <= 0 or not (0 <= vidx < dyn_dim):
        return {
            "busy_edge_frac": zero.detach(),
            "busy_speed_mse": zero.detach(),
            "busy_volume_mse": zero.detach(),
        }

    busy_mask = _busy_edge_mask(
        pred_raw=pred_raw,
        target_raw=target_raw,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
    )
    denom = busy_mask.sum().clamp_min(1.0)
    busy_frac = busy_mask.mean()

    volume_mse = (((pred_eval[..., vidx].float() - target_eval[..., vidx].float()).pow(2)) * busy_mask).sum() / denom
    if 0 <= sidx < dyn_dim:
        speed_mse = (((pred_eval[..., sidx].float() - target_eval[..., sidx].float()).pow(2)) * busy_mask).sum() / denom
    else:
        speed_mse = zero
    return {
        "busy_edge_frac": busy_frac.detach(),
        "busy_speed_mse": speed_mse.detach(),
        "busy_volume_mse": volume_mse.detach(),
    }


def _busy_edge_mask(
    *,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    volume_index: int,
    busy_threshold: float,
) -> torch.Tensor:
    vidx = int(volume_index)
    pred_vol = _positive_pred_volume_for_loss(pred_raw[..., vidx])
    target_vol = target_raw[..., vidx].float().clamp_min(0.0)
    return ((pred_vol > float(busy_threshold)) | (target_vol > float(busy_threshold))).float()


def _busy_volume_mse(
    *,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    pred_eval: torch.Tensor,
    target_eval: torch.Tensor,
    volume_index: int,
    busy_threshold: float,
) -> torch.Tensor:
    zero = torch.zeros((), device=pred_eval.device, dtype=torch.float32)
    dyn_dim = int(pred_eval.shape[-1])
    vidx = int(volume_index)
    if dyn_dim <= 0 or not (0 <= vidx < dyn_dim):
        return zero
    busy_mask = _busy_edge_mask(
        pred_raw=pred_raw,
        target_raw=target_raw,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
    )
    denom = busy_mask.sum().clamp_min(1.0)
    return (((pred_eval[..., vidx].float() - target_eval[..., vidx].float()).pow(2)) * busy_mask).sum() / denom


def _busy_presence_bce(
    *,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    volume_index: int,
    busy_threshold: float,
    margin: float,
) -> torch.Tensor:
    zero = torch.zeros((), device=pred_raw.device, dtype=torch.float32)
    dyn_dim = int(pred_raw.shape[-1])
    vidx = int(volume_index)
    if dyn_dim <= 0 or not (0 <= vidx < dyn_dim):
        return zero

    target_busy = (target_raw[..., vidx].float().clamp_min(0.0) > float(busy_threshold)).float()
    if target_busy.numel() == 0:
        return zero
    logits = (pred_raw[..., vidx].float() - float(busy_threshold)) / max(float(margin), 1e-6)

    pos = target_busy.sum()
    neg = target_busy.numel() - pos
    pos_weight = None
    if pos > 0:
        pos_weight = (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=50.0).to(device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, target_busy, pos_weight=pos_weight)


def _zero_volume_false_alarm_loss(
    *,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    volume_index: int,
    zero_target_threshold: float,
    pred_threshold: float,
    margin: float,
) -> torch.Tensor:
    zero = torch.zeros((), device=pred_raw.device, dtype=torch.float32)
    dyn_dim = int(pred_raw.shape[-1])
    vidx = int(volume_index)
    if dyn_dim <= 0 or not (0 <= vidx < dyn_dim):
        return zero

    pred_vol = _positive_pred_volume_for_loss(pred_raw[..., vidx])
    target_vol = target_raw[..., vidx].float().clamp_min(0.0)
    zero_mask = (target_vol <= float(zero_target_threshold)).float()
    denom = zero_mask.sum().clamp_min(1.0)
    scaled_excess = (pred_vol - float(pred_threshold)) / max(float(margin), 1e-6)
    false_alarm = float(margin) * F.softplus(scaled_excess)
    return (false_alarm * zero_mask).sum() / denom


def _busy_topk_rank_loss(
    *,
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    volume_index: int,
    busy_threshold: float,
    k: int,
    margin: float,
) -> torch.Tensor:
    zero = torch.zeros((), device=pred_raw.device, dtype=torch.float32)
    dyn_dim = int(pred_raw.shape[-1])
    vidx = int(volume_index)
    if dyn_dim <= 0 or not (0 <= vidx < dyn_dim):
        return zero

    pred_vol = _positive_pred_volume_for_loss(pred_raw[..., vidx]).reshape(-1, pred_raw.shape[-2])
    target_vol = target_raw[..., vidx].float().clamp_min(0.0).reshape(-1, target_raw.shape[-2])
    if pred_vol.numel() == 0:
        return zero

    losses = []
    topk = max(1, int(k))
    for pred_row, target_row in zip(pred_vol, target_vol):
        pos_idx = torch.nonzero(target_row > float(busy_threshold), as_tuple=False).flatten()
        neg_idx = torch.nonzero(target_row <= float(busy_threshold), as_tuple=False).flatten()
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        n_pos = min(topk, int(pos_idx.numel()))
        target_pos_scores = target_row[pos_idx]
        pos_keep = pos_idx[torch.topk(target_pos_scores, k=n_pos).indices]

        n_neg = min(max(topk, n_pos), int(neg_idx.numel()))
        pred_neg_scores = pred_row[neg_idx]
        neg_keep = neg_idx[torch.topk(pred_neg_scores, k=n_neg).indices]

        pos_scores = pred_row[pos_keep]
        neg_scores = pred_row[neg_keep]
        pair_losses = F.softplus(float(margin) - pos_scores[:, None] + neg_scores[None, :])
        losses.append(pair_losses.mean())

    if not losses:
        return zero
    return torch.stack(losses).mean()


def multihead_decoder_supervision_loss(
    *,
    pred_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
    full_weight: float,
    speed_weight: float,
    volume_weight: float,
    other_weight: float,
    speed_index: int = 1,
    volume_index: int = 2,
    speed_target_mode: str = "slowdown",
    volume_target_mode: str = "log1p",
    capacities: Optional[torch.Tensor] = None,
    busy_threshold: float = 0.05,
    busy_volume_weight: float = 0.0,
    busy_presence_weight: float = 0.0,
    busy_presence_margin: float = 0.05,
    topk_rank_weight: float = 0.0,
    topk_rank_k: int = 20,
    topk_rank_margin: float = 0.05,
    zero_volume_weight: float = 0.0,
    zero_volume_threshold: float = 0.05,
    zero_volume_margin: float = 0.05,
    zero_target_threshold: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    zero = torch.zeros((), device=pred_decoded.device, dtype=torch.float32)
    pred_eval, target_eval = _transform_decoded_targets(
        pred_decoded=pred_decoded,
        target_decoded=target_decoded,
        speed_index=speed_index,
        volume_index=volume_index,
        speed_target_mode=speed_target_mode,
        volume_target_mode=volume_target_mode,
        capacities=capacities,
    )
    err2 = (pred_eval - target_eval).pow(2)
    full_mse = err2.mean()

    dyn_dim = int(pred_decoded.shape[-1])
    sidx = int(speed_index)
    vidx = int(volume_index)
    speed_mse = err2[..., sidx].mean() if 0 <= sidx < dyn_dim else zero
    volume_mse = err2[..., vidx].mean() if 0 <= vidx < dyn_dim else zero
    busy_volume_mse = _busy_volume_mse(
        pred_raw=pred_decoded,
        target_raw=target_decoded,
        pred_eval=pred_eval,
        target_eval=target_eval,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
    )
    busy_presence_bce = _busy_presence_bce(
        pred_raw=pred_decoded,
        target_raw=target_decoded,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
        margin=busy_presence_margin,
    )
    topk_rank_loss = _busy_topk_rank_loss(
        pred_raw=pred_decoded,
        target_raw=target_decoded,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
        k=topk_rank_k,
        margin=topk_rank_margin,
    )
    zero_volume_loss = _zero_volume_false_alarm_loss(
        pred_raw=pred_decoded,
        target_raw=target_decoded,
        volume_index=volume_index,
        zero_target_threshold=zero_target_threshold,
        pred_threshold=zero_volume_threshold,
        margin=zero_volume_margin,
    )
    other_idx = [i for i in range(dyn_dim) if i not in (sidx, vidx)]
    other_mse = err2[..., other_idx].mean() if other_idx else zero

    loss = (
        float(full_weight) * full_mse
        + float(speed_weight) * speed_mse
        + float(volume_weight) * volume_mse
        + float(other_weight) * other_mse
        + float(busy_volume_weight) * busy_volume_mse
        + float(busy_presence_weight) * busy_presence_bce
        + float(topk_rank_weight) * topk_rank_loss
        + float(zero_volume_weight) * zero_volume_loss
    )
    aux = {
        "decoder_full_mse": full_mse.detach(),
        "decoder_speed_mse": speed_mse.detach(),
        "decoder_volume_mse": volume_mse.detach(),
        "decoder_busy_volume_weighted_mse": busy_volume_mse.detach(),
        "decoder_busy_presence_bce": busy_presence_bce.detach(),
        "decoder_topk_rank_loss": topk_rank_loss.detach(),
        "decoder_zero_volume_false_alarm_loss": zero_volume_loss.detach(),
        "decoder_other_mse": other_mse.detach(),
        "decoder_multihead_loss": loss.detach(),
    }
    aux.update(
        {
            f"decoder_{k}": v
            for k, v in _busy_edge_channel_metrics(
                pred_raw=pred_decoded,
                target_raw=target_decoded,
                pred_eval=pred_eval,
                target_eval=target_eval,
                volume_index=volume_index,
                speed_index=speed_index,
                busy_threshold=busy_threshold,
            ).items()
        }
    )
    return loss, aux



def transition_decoded_supervision_loss(
    *,
    z_pred_next: Optional[torch.Tensor],
    flow_projector: torch.nn.Module,
    flow_target_next: torch.Tensor,
    supervision_weight: float,
    speed_index: int = 1,
    volume_index: int = 2,
    speed_weight: float = 4.0,
    volume_weight: float = 2.0,
    speed_target_mode: str = "slowdown",
    volume_target_mode: str = "log1p",
    capacities: Optional[torch.Tensor] = None,
    busy_threshold: float = 0.05,
    busy_volume_weight: float = 0.0,
    busy_presence_weight: float = 0.0,
    busy_presence_margin: float = 0.05,
    topk_rank_weight: float = 0.0,
    topk_rank_k: int = 20,
    topk_rank_margin: float = 0.05,
    zero_volume_weight: float = 0.0,
    zero_volume_threshold: float = 0.05,
    zero_volume_margin: float = 0.05,
    zero_target_threshold: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Decode a predicted next latent state and supervise it against the real next flow.

    This is a minimal one-step terminal supervision term: the model first predicts the
    next latent state, then FlowProjector.decode decodes that latent into dynamic flow
    channels and compares them to the real next-step flow target.
    """
    device = flow_target_next.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if z_pred_next is None or float(supervision_weight) <= 0.0:
        return zero, {}

    decode_fn = getattr(flow_projector, "decode", None)
    if decode_fn is None:
        decode_fn = flow_projector.inverse
    pred_next_decoded = decode_fn(z_pred_next)
    dyn_dim = int(pred_next_decoded.shape[-1])
    target_dyn = flow_target_next[..., :dyn_dim].to(dtype=pred_next_decoded.dtype)
    pred_eval, target_eval = _transform_decoded_targets(
        pred_decoded=pred_next_decoded,
        target_decoded=target_dyn,
        speed_index=speed_index,
        volume_index=volume_index,
        speed_target_mode=speed_target_mode,
        volume_target_mode=volume_target_mode,
        capacities=capacities,
    )
    err2 = (pred_eval - target_eval).pow(2)

    weighted_mse = err2.mean()
    speed_mse = zero
    volume_mse = zero
    if dyn_dim > 0:
        feat_w = torch.ones((dyn_dim,), device=err2.device, dtype=err2.dtype)
        sidx = int(speed_index)
        vidx = int(volume_index)
        if 0 <= sidx < dyn_dim:
            feat_w[sidx] = float(max(1.0, speed_weight))
            speed_mse = err2[..., sidx].mean()
        if 0 <= vidx < dyn_dim:
            feat_w[vidx] = float(max(1.0, volume_weight))
            volume_mse = err2[..., vidx].mean()
        view_shape = [1] * (err2.dim() - 1) + [dyn_dim]
        weighted_mse = (err2 * feat_w.view(*view_shape)).mean()

    busy_volume_mse = _busy_volume_mse(
        pred_raw=pred_next_decoded,
        target_raw=target_dyn,
        pred_eval=pred_eval,
        target_eval=target_eval,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
    )
    busy_presence_bce = _busy_presence_bce(
        pred_raw=pred_next_decoded,
        target_raw=target_dyn,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
        margin=busy_presence_margin,
    )
    topk_rank_loss = _busy_topk_rank_loss(
        pred_raw=pred_next_decoded,
        target_raw=target_dyn,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
        k=topk_rank_k,
        margin=topk_rank_margin,
    )
    zero_volume_loss = _zero_volume_false_alarm_loss(
        pred_raw=pred_next_decoded,
        target_raw=target_dyn,
        volume_index=volume_index,
        zero_target_threshold=zero_target_threshold,
        pred_threshold=zero_volume_threshold,
        margin=zero_volume_margin,
    )

    total_mse = (
        weighted_mse
        + float(busy_volume_weight) * busy_volume_mse
        + float(busy_presence_weight) * busy_presence_bce
        + float(topk_rank_weight) * topk_rank_loss
        + float(zero_volume_weight) * zero_volume_loss
    )
    loss = float(supervision_weight) * total_mse
    aux = {
        "pred_next_decoded_supervision_loss": loss.detach(),
        "pred_next_decoded_weighted_mse": weighted_mse.detach(),
        "pred_next_decoded_total_mse": total_mse.detach(),
        "pred_next_decoded_speed_mse": speed_mse.detach(),
        "pred_next_decoded_volume_mse": volume_mse.detach(),
        "pred_next_decoded_busy_volume_weighted_mse": busy_volume_mse.detach(),
        "pred_next_decoded_busy_presence_bce": busy_presence_bce.detach(),
        "pred_next_decoded_topk_rank_loss": topk_rank_loss.detach(),
        "pred_next_decoded_zero_volume_false_alarm_loss": zero_volume_loss.detach(),
    }
    aux.update(
        {
            f"pred_next_decoded_{k}": v
            for k, v in _busy_edge_channel_metrics(
                pred_raw=pred_next_decoded,
                target_raw=target_dyn,
                pred_eval=pred_eval,
                target_eval=target_eval,
                volume_index=volume_index,
                speed_index=speed_index,
                busy_threshold=busy_threshold,
            ).items()
        }
    )
    return loss, aux


def transition_action_contrast_metric_loss(
    *,
    z_pred_next: Optional[torch.Tensor],
    flow_projector: torch.nn.Module,
    flow_target_next: torch.Tensor,
    metric: str,
    speed_index: int = 1,
    volume_index: int = 2,
    speed_target_mode: str = "slowdown",
    volume_target_mode: str = "log1p",
    capacities: Optional[torch.Tensor] = None,
    busy_threshold: float = 0.05,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable decoded metric used to rank true actions against shuffled actions."""
    device = flow_target_next.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if z_pred_next is None:
        return zero, {}

    metric_name = str(metric).strip().lower()
    decode_fn = getattr(flow_projector, "decode", None)
    if decode_fn is None:
        decode_fn = flow_projector.inverse
    pred_next_decoded = decode_fn(z_pred_next)
    dyn_dim = int(pred_next_decoded.shape[-1])
    vidx = int(volume_index)
    if vidx < 0 or vidx >= dyn_dim:
        return zero, {}

    target_dyn = flow_target_next[..., :dyn_dim].to(dtype=pred_next_decoded.dtype)
    pred_eval, target_eval = _transform_decoded_targets(
        pred_decoded=pred_next_decoded,
        target_decoded=target_dyn,
        speed_index=speed_index,
        volume_index=volume_index,
        speed_target_mode=speed_target_mode,
        volume_target_mode=volume_target_mode,
        capacities=capacities,
    )
    volume_err2 = (pred_eval[..., vidx] - target_eval[..., vidx]).pow(2)

    if metric_name == "volume":
        metric_loss = volume_err2.mean()
    elif metric_name == "active_volume":
        active_mask = (target_dyn[..., vidx] > 0.0).to(dtype=volume_err2.dtype)
        denom = active_mask.sum().clamp_min(1.0)
        metric_loss = (volume_err2 * active_mask).sum() / denom
    elif metric_name == "busy_volume":
        busy_mask = (target_dyn[..., vidx].float().clamp_min(0.0) > float(busy_threshold)).to(
            dtype=volume_err2.dtype
        )
        denom = busy_mask.sum().clamp_min(1.0)
        metric_loss = (volume_err2 * busy_mask).sum() / denom
    else:
        raise ValueError(
            f"Unsupported transition action contrast metric {metric!r}; "
            "expected 'total', 'volume', 'active_volume', or 'busy_volume'."
        )

    return metric_loss, {f"pred_next_action_contrast_{metric_name}_mse": metric_loss.detach()}


def action_local_latent_delta_loss(
    *,
    z_pred_next: Optional[torch.Tensor],
    z_prev: torch.Tensor,
    z_target_next: torch.Tensor,
    actions: Optional[torch.Tensor | tuple],
    n_spatial: int,
    loss_weight: float,
    action_eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = z_target_next.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if z_pred_next is None or float(loss_weight) <= 0.0:
        return zero, {}

    B = int(z_pred_next.shape[0])
    edge_actions = _edge_action_values_for_loss(
        actions,
        batch_size=B,
        time_steps=1,
        n_spatial=n_spatial,
        device=z_pred_next.device,
        dtype=torch.float32,
    )
    if edge_actions is None:
        return zero, {}
    mask = (edge_actions[:, -1].abs() > float(action_eps)).float().unsqueeze(-1)
    active = mask.sum()
    active_frac = mask.mean()
    if float(active.item()) <= 0.0:
        return zero, {"action_local_latent_active_frac": active_frac.detach()}

    pred_delta = z_pred_next.float() - z_prev.float()
    target_delta = z_target_next.float() - z_prev.float()
    latent_mse = ((pred_delta - target_delta).pow(2) * mask).sum() / (active * pred_delta.shape[-1]).clamp_min(1.0)
    loss = float(loss_weight) * latent_mse
    return loss, {
        "action_local_latent_loss": loss.detach(),
        "action_local_latent_mse": latent_mse.detach(),
        "action_local_latent_active_frac": active_frac.detach(),
    }


def action_volume_delta_loss(
    *,
    z_pred_next: Optional[torch.Tensor],
    z_prev: torch.Tensor,
    flow_projector: torch.nn.Module,
    flow_prev: torch.Tensor,
    flow_target_next: torch.Tensor,
    actions: Optional[torch.Tensor | tuple],
    n_spatial: int,
    loss_weight: float,
    action_eps: float = 1e-6,
    volume_index: int = 2,
    speed_index: int = 1,
    speed_target_mode: str = "slowdown",
    volume_target_mode: str = "log1p",
    capacities: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Supervise action-conditioned volume changes on edges that actually receive action.

    Absolute next-volume MSE can be minimized by state persistence. This term compares
    predicted and observed volume deltas from the same previous flow state, masked to
    active action edges so CFM receives a direct action -> delta gradient.
    """
    device = z_prev.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if z_pred_next is None or float(loss_weight) <= 0.0:
        return zero, {}

    edge_actions = _edge_action_values_for_loss(
        actions,
        batch_size=int(z_prev.shape[0]),
        time_steps=1,
        n_spatial=n_spatial,
        device=device,
        dtype=torch.float32,
    )
    if edge_actions is None:
        return zero, {"action_volume_delta_active_frac": zero.detach()}

    active_mask = (edge_actions[:, -1].abs() > float(action_eps)).float()
    active_frac = active_mask.mean()
    denom = active_mask.sum().clamp_min(1.0)
    if not bool((active_mask.sum() > 0).item()):
        return zero, {"action_volume_delta_active_frac": active_frac.detach()}

    decode_fn = getattr(flow_projector, "decode", None)
    if decode_fn is None:
        decode_fn = flow_projector.inverse
    pred_next_decoded = decode_fn(z_pred_next)
    dyn_dim = int(pred_next_decoded.shape[-1])
    vidx = int(volume_index)
    if dyn_dim <= 0 or not (0 <= vidx < dyn_dim):
        return zero, {"action_volume_delta_active_frac": active_frac.detach()}

    prev_dyn = flow_prev[..., :dyn_dim].to(device=device, dtype=pred_next_decoded.dtype)
    target_next_dyn = flow_target_next[..., :dyn_dim].to(device=device, dtype=pred_next_decoded.dtype)
    pred_eval, target_next_eval = _transform_decoded_targets(
        pred_decoded=pred_next_decoded,
        target_decoded=target_next_dyn,
        speed_index=speed_index,
        volume_index=volume_index,
        speed_target_mode=speed_target_mode,
        volume_target_mode=volume_target_mode,
        capacities=capacities,
    )
    prev_eval, _ = _transform_decoded_targets(
        pred_decoded=prev_dyn,
        target_decoded=prev_dyn,
        speed_index=speed_index,
        volume_index=volume_index,
        speed_target_mode=speed_target_mode,
        volume_target_mode=volume_target_mode,
        capacities=capacities,
    )

    pred_delta = pred_eval[..., vidx].float() - prev_eval[..., vidx].float()
    target_delta = target_next_eval[..., vidx].float() - prev_eval[..., vidx].float()
    delta_mse = ((pred_delta - target_delta).pow(2) * active_mask).sum() / denom
    loss = float(loss_weight) * delta_mse
    return loss, {
        "action_volume_delta_loss": loss.detach(),
        "action_volume_delta_mse": delta_mse.detach(),
        "action_volume_delta_active_frac": active_frac.detach(),
        "action_volume_delta_target_abs_mean": ((target_delta.abs() * active_mask).sum() / denom).detach(),
        "action_volume_delta_pred_abs_mean": ((pred_delta.abs() * active_mask).sum() / denom).detach(),
    }


def decoder_reconstruction_loss(
    *,
    decoded_dyn: Optional[torch.Tensor],
    flow_targets: torch.Tensor,
    recon_weight: float,
    speed_index: int = 1,
    speed_weight: float = 4.0,
    volume_index: int = 2,
    speed_target_mode: str = "slowdown",
    volume_target_mode: str = "log1p",
    capacities: Optional[torch.Tensor] = None,
    busy_threshold: float = 0.05,
    busy_volume_weight: float = 0.0,
    busy_presence_weight: float = 0.0,
    busy_presence_margin: float = 0.05,
    topk_rank_weight: float = 0.0,
    topk_rank_k: int = 20,
    topk_rank_margin: float = 0.05,
    zero_volume_weight: float = 0.0,
    zero_volume_threshold: float = 0.05,
    zero_volume_margin: float = 0.05,
    zero_target_threshold: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Supervise FlowProjector.decode on real flow dynamics.

    The target comes from real flow features (first n_dynamic channels), and the
    speed channel can be up-weighted to focus reconstruction on the velocity field.
    """
    device = flow_targets.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if decoded_dyn is None or float(recon_weight) <= 0.0:
        return zero, {}

    dyn_dim = int(decoded_dyn.shape[-1])
    target_dyn = flow_targets[..., :dyn_dim].to(dtype=decoded_dyn.dtype)
    pred_eval, target_eval = _transform_decoded_targets(
        pred_decoded=decoded_dyn,
        target_decoded=target_dyn,
        speed_index=speed_index,
        volume_index=volume_index,
        speed_target_mode=speed_target_mode,
        volume_target_mode=volume_target_mode,
        capacities=capacities,
    )
    err2 = (pred_eval - target_eval).pow(2)  # (B,T,E,D)
    recon_mse = err2.mean()

    weighted_mse = recon_mse
    speed_mse = zero
    volume_mse = zero
    if dyn_dim > 0:
        feat_w = torch.ones((dyn_dim,), device=err2.device, dtype=err2.dtype)
        sidx = int(speed_index)
        vidx = int(volume_index)
        if 0 <= sidx < dyn_dim:
            feat_w[sidx] = float(max(1.0, speed_weight))
            speed_mse = err2[..., sidx].mean()
        if 0 <= vidx < dyn_dim:
            volume_mse = err2[..., vidx].mean()
        view_shape = [1] * (err2.dim() - 1) + [dyn_dim]
        weighted_mse = (err2 * feat_w.view(*view_shape)).mean()

    busy_volume_mse = _busy_volume_mse(
        pred_raw=decoded_dyn,
        target_raw=target_dyn,
        pred_eval=pred_eval,
        target_eval=target_eval,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
    )
    busy_presence_bce = _busy_presence_bce(
        pred_raw=decoded_dyn,
        target_raw=target_dyn,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
        margin=busy_presence_margin,
    )
    topk_rank_loss = _busy_topk_rank_loss(
        pred_raw=decoded_dyn,
        target_raw=target_dyn,
        volume_index=volume_index,
        busy_threshold=busy_threshold,
        k=topk_rank_k,
        margin=topk_rank_margin,
    )
    zero_volume_loss = _zero_volume_false_alarm_loss(
        pred_raw=decoded_dyn,
        target_raw=target_dyn,
        volume_index=volume_index,
        zero_target_threshold=zero_target_threshold,
        pred_threshold=zero_volume_threshold,
        margin=zero_volume_margin,
    )
    total_mse = (
        weighted_mse
        + float(busy_volume_weight) * busy_volume_mse
        + float(busy_presence_weight) * busy_presence_bce
        + float(topk_rank_weight) * topk_rank_loss
        + float(zero_volume_weight) * zero_volume_loss
    )

    recon_loss = float(recon_weight) * total_mse
    aux = {
        "decoder_recon_mse": recon_mse.detach(),
        "decoder_recon_speed_mse": speed_mse.detach(),
        "decoder_recon_volume_mse": volume_mse.detach(),
        "decoder_recon_weighted_mse": weighted_mse.detach(),
        "decoder_recon_total_mse": total_mse.detach(),
        "decoder_recon_busy_volume_weighted_mse": busy_volume_mse.detach(),
        "decoder_recon_busy_presence_bce": busy_presence_bce.detach(),
        "decoder_recon_topk_rank_loss": topk_rank_loss.detach(),
        "decoder_recon_zero_volume_false_alarm_loss": zero_volume_loss.detach(),
        "decoder_recon_loss": recon_loss.detach(),
    }
    aux.update(
        {
            f"decoder_recon_{k}": v
            for k, v in _busy_edge_channel_metrics(
                pred_raw=decoded_dyn,
                target_raw=target_dyn,
                pred_eval=pred_eval,
                target_eval=target_eval,
                volume_index=volume_index,
                speed_index=speed_index,
                busy_threshold=busy_threshold,
            ).items()
        }
    )
    return recon_loss, aux


def inverse_reconstruction_loss(
    *,
    inverse_dyn: Optional[torch.Tensor] = None,
    decoded_dyn: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compatibility alias for decoder_reconstruction_loss()."""
    if decoded_dyn is None:
        decoded_dyn = inverse_dyn
    return decoder_reconstruction_loss(decoded_dyn=decoded_dyn, **kwargs)


def decoder_bpr_supervision_loss(
    *,
    decoded_dyn: Optional[torch.Tensor],
    flow_targets: torch.Tensor,
    lengths: torch.Tensor,       # (E,) edge lengths in meters
    speed_limits: torch.Tensor,  # (E,) m/s
    capacities: torch.Tensor,    # (E,) vehicle capacities
    bpr_weight: float,
    bpr_reward_weight: float = 1.0,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    veh_index: int = 2,
    target_mode: str = "relative_excess",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    BPR consistency supervision on FlowProjector.decode outputs.

    Compares only the per-edge BPR travel-time field between:
      - prediction: decoded_dyn
      - target:     real flow dynamics channels in flow_targets
    """
    device = flow_targets.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if decoded_dyn is None or float(bpr_weight) <= 0.0:
        return zero, {}

    dyn_dim = int(decoded_dyn.shape[-1])
    vidx = int(veh_index)
    if dyn_dim <= 0 or vidx < 0 or vidx >= dyn_dim:
        return zero, {}

    target_dyn = flow_targets[..., :dyn_dim].to(dtype=decoded_dyn.dtype)
    if vidx >= int(target_dyn.shape[-1]):
        return zero, {}

    cap = capacities.to(device=device, dtype=torch.float32).clamp_min(1.0).view(1, 1, -1)
    t0 = (lengths.to(device=device, dtype=torch.float32) /
          speed_limits.to(device=device, dtype=torch.float32).clamp_min(0.1)).view(1, 1, -1)

    v_pred = _positive_pred_volume_for_loss(decoded_dyn[..., vidx]) * cap
    v_true = target_dyn[..., vidx].float().clamp_min(0.0) * cap

    t_pred = t0 * (1.0 + float(bpr_alpha) * (v_pred / cap).pow(float(bpr_beta)))
    t_true = t0 * (1.0 + float(bpr_alpha) * (v_true / cap).pow(float(bpr_beta)))

    raw_bpr_edge_mse = (t_pred - t_true).pow(2).mean()

    mode = str(target_mode).lower()
    if mode == "raw_seconds":
        t_pred_eval = t_pred
        t_true_eval = t_true
    else:
        denom = t0.clamp_min(1e-6)
        t_pred_eval = (t_pred / denom) - 1.0
        t_true_eval = (t_true / denom) - 1.0
        if mode == "log_relative_excess":
            t_pred_eval = torch.log1p(t_pred_eval.clamp_min(0.0))
            t_true_eval = torch.log1p(t_true_eval.clamp_min(0.0))
        elif mode != "relative_excess":
            raise ValueError(f"Unsupported bpr target_mode: {target_mode}")

    bpr_edge_mse = (t_pred_eval - t_true_eval).pow(2).mean()

    # Keep the deprecated argument for CLI compatibility, but do not use
    # episode-level BPR reward regression in the loss anymore.
    _ = bpr_reward_weight
    weighted = bpr_edge_mse
    bpr_loss = float(bpr_weight) * weighted
    aux = {
        "bpr_edge_mse": bpr_edge_mse.detach(),
        "bpr_edge_mse_normalized": bpr_edge_mse.detach(),
        "bpr_edge_mse_raw_seconds": raw_bpr_edge_mse.detach(),
        "bpr_weighted_mse": weighted.detach(),
        "bpr_supervision_loss": bpr_loss.detach(),
    }
    return bpr_loss, aux


def _sample_route_registry_indices(
    route_edge_indices: List[List[List[int]]],
    *,
    max_samples: int,
    device: torch.device,
) -> List[List[List[int]]]:
    valid = [
        routes
        for routes in route_edge_indices
        if routes and any(len(path) > 0 for path in routes)
    ]
    if not valid:
        return []
    n = len(valid)
    k = int(max_samples)
    if k > 0 and n > k:
        idx = torch.randperm(n, device=device)[:k].detach().cpu().tolist()
        return [valid[int(i)] for i in idx]
    return valid


def _load_route_edges_from_routes_csv(
    path: str,
    edge_to_idx: Dict[str, int],
    *,
    n_routes: int,
) -> List[List[List[int]]]:
    groups: Dict[tuple[int, int], List[List[int]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                oi = int(float(row.get("origins", row.get("origin", -1))))
                di = int(float(row.get("destinations", row.get("destination", -1))))
            except (TypeError, ValueError):
                continue
            raw_path = str(row.get("path", "") or "")
            idxs = [int(edge_to_idx[e]) for e in raw_path.split() if e in edge_to_idx]
            if not idxs:
                continue
            groups.setdefault((oi, di), []).append(idxs)

    out: List[List[List[int]]] = []
    k_eff = int(max(n_routes, 1))
    for key in sorted(groups):
        routes = groups[key][:k_eff]
        if not routes:
            continue
        while len(routes) < k_eff:
            routes.append(list(routes[0]))
        out.append(routes[:k_eff])
    return out


def _route_cost_tensor_from_paths(
    *,
    edge_cost: torch.Tensor,
    sampled_routes: List[List[List[int]]],
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    length_norm: str,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if edge_cost.dim() != 2:
        raise ValueError(f"edge_cost must be 2D (N,E), got {tuple(edge_cost.shape)}")
    n_samples = len(sampled_routes)
    k_eff = int(n_routes)
    if n_samples <= 0 or k_eff <= 0:
        return None, None

    device = edge_cost.device
    dtype = edge_cost.dtype
    n_batch = int(edge_cost.shape[0])
    n_edges = int(edge_cost.shape[1])
    costs = torch.empty((n_batch, n_samples, k_eff), device=device, dtype=dtype)
    valid = torch.zeros((n_samples, k_eff), device=device, dtype=torch.bool)
    fallback = invalid_route_cost.to(device=device, dtype=dtype)
    use_mean = str(length_norm).strip().lower() == "mean"

    for od_i, routes in enumerate(sampled_routes):
        for k in range(k_eff):
            path = routes[k] if k < len(routes) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                vals = edge_cost.index_select(1, edge_idx)
                costs[:, od_i, k] = vals.mean(dim=1) if use_mean else vals.sum(dim=1)
                valid[od_i, k] = True
            else:
                costs[:, od_i, k] = fallback

    if not bool(valid.any().detach().cpu().item()):
        return None, None
    return costs, valid


def route_ranking_supervision_loss(
    *,
    decoded_dyn: Optional[torch.Tensor],
    flow_targets: torch.Tensor,
    route_edge_indices: Optional[List[List[List[int]]]],
    lengths: torch.Tensor,
    speed_limits: torch.Tensor,
    capacities: torch.Tensor,
    loss_weight: float,
    n_routes: int,
    max_od_samples: int = 64,
    temperature: float = 30.0,
    margin_weight: float = 0.0,
    margin: float = 1.0,
    gap_weight: float = 0.0,
    length_norm: str = "sum",
    target_cost_mode: str = "speed",
    non0_weight: float = 1.0,
    speed_min_ratio: float = 0.05,
    speed_max_ratio: float = 1.50,
    speed_excess_scale: float = 1.0,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    volume_index: int = 2,
    speed_index: int = 1,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Supervise route-level cost ranking induced by decoded edge costs.

    In ``bpr_volume`` mode both prediction and label are computed from edge
    volumes via the BPR travel-time surrogate used by imagined MARL.  In
    ``speed`` mode both sides use decoded/target speed ratio directly.  This
    keeps the route-ranking supervision aligned with the edge-cost source used
    by downstream planner evaluation.
    """
    device = flow_targets.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if (
        decoded_dyn is None
        or float(loss_weight) <= 0.0
        or route_edge_indices is None
        or not route_edge_indices
    ):
        return zero, {}

    dyn_dim = int(decoded_dyn.shape[-1])
    mode = str(target_cost_mode).strip().lower()
    if dyn_dim <= 0:
        return zero, {}

    sampled_routes = _sample_route_registry_indices(
        route_edge_indices,
        max_samples=int(max_od_samples),
        device=device,
    )
    if not sampled_routes:
        return zero, {}

    target_dyn = flow_targets[..., :dyn_dim].to(device=device, dtype=decoded_dyn.dtype)
    cap = capacities.to(device=device, dtype=torch.float32).clamp_min(1.0).view(1, -1)
    t0 = (
        lengths.to(device=device, dtype=torch.float32)
        / speed_limits.to(device=device, dtype=torch.float32).clamp_min(0.1)
    ).view(1, -1)

    if mode == "bpr_volume":
        vidx = int(volume_index)
        if vidx < 0 or vidx >= dyn_dim:
            return zero, {}
        pred_vol = _positive_pred_volume_for_loss(decoded_dyn[..., vidx]).reshape(-1, decoded_dyn.shape[-2]) * cap
        pred_edge_tt = t0 * (1.0 + float(bpr_alpha) * (pred_vol / cap).pow(float(bpr_beta)))
        true_vol = target_dyn[..., vidx].float().clamp_min(0.0).reshape(-1, target_dyn.shape[-2]) * cap
        true_edge_tt = t0 * (1.0 + float(bpr_alpha) * (true_vol / cap).pow(float(bpr_beta)))
    elif mode == "speed":
        sidx = int(speed_index)
        if sidx < 0 or sidx >= dyn_dim or sidx >= int(target_dyn.shape[-1]):
            return zero, {}
        min_ratio = max(float(speed_min_ratio), 1e-4)
        max_ratio = max(float(speed_max_ratio), min_ratio)
        pred_speed = _positive_pred_speed_for_loss(decoded_dyn[..., sidx]).reshape(-1, decoded_dyn.shape[-2])
        pred_speed = pred_speed.clamp(min=min_ratio, max=max_ratio)
        excess_scale = max(float(speed_excess_scale), 0.0)
        pred_edge_tt = t0 * (1.0 + excess_scale * ((1.0 / pred_speed) - 1.0))
        speed_norm = target_dyn[..., sidx].float().reshape(-1, target_dyn.shape[-2]).clamp(min=min_ratio, max=max_ratio)
        true_edge_tt = t0 * (1.0 + excess_scale * ((1.0 / speed_norm) - 1.0))
    else:
        raise ValueError(f"Unsupported route ranking target_cost_mode={target_cost_mode!r}")

    fallback_pred = pred_edge_tt.mean()
    fallback_true = true_edge_tt.mean()
    pred_costs, route_valid = _route_cost_tensor_from_paths(
        edge_cost=pred_edge_tt,
        sampled_routes=sampled_routes,
        n_routes=int(n_routes),
        invalid_route_cost=fallback_pred,
        length_norm=length_norm,
    )
    true_costs, _ = _route_cost_tensor_from_paths(
        edge_cost=true_edge_tt,
        sampled_routes=sampled_routes,
        n_routes=int(n_routes),
        invalid_route_cost=fallback_true,
        length_norm=length_norm,
    )
    if pred_costs is None or true_costs is None or route_valid is None:
        return zero, {}

    valid = route_valid.unsqueeze(0).expand(pred_costs.shape[0], -1, -1)
    flat_valid = valid.reshape(-1, valid.shape[-1])
    flat_pred = pred_costs.reshape(-1, pred_costs.shape[-1])
    flat_true = true_costs.reshape(-1, true_costs.shape[-1])
    row_mask = flat_valid.any(dim=-1)
    if not bool(row_mask.any().detach().cpu().item()):
        return zero, {}
    flat_pred = flat_pred[row_mask]
    flat_true = flat_true[row_mask]
    flat_valid = flat_valid[row_mask]

    large = torch.finfo(flat_true.dtype).max / 4.0
    masked_true = flat_true.masked_fill(~flat_valid, large)
    masked_pred = flat_pred.masked_fill(~flat_valid, large)
    best_true = torch.argmin(masked_true, dim=-1)
    best_pred = torch.argmin(masked_pred.detach(), dim=-1)
    true_non0 = (best_true > 0).float()
    row_weight = torch.ones_like(true_non0)
    if float(non0_weight) > 1.0:
        row_weight = torch.where(best_true > 0, torch.full_like(row_weight, float(non0_weight)), row_weight)

    min_true = masked_true.gather(1, best_true.unsqueeze(-1))
    normalized_true = masked_true - min_true
    temp = max(float(temperature), 1e-6)
    target_logits = -(normalized_true / temp).masked_fill(~flat_valid, -large)
    target_probs = torch.softmax(target_logits, dim=-1).detach()
    pred_logits = -(masked_pred - masked_pred.detach().amin(dim=-1, keepdim=True)) / temp
    pred_logits = pred_logits.masked_fill(~flat_valid, -large)
    ce_per = -(target_probs * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1)
    ce = (ce_per * row_weight).sum() / row_weight.sum().clamp_min(1.0)

    margin_loss = zero
    valid_counts = flat_valid.sum(dim=-1)
    if float(margin_weight) > 0.0 and bool((valid_counts > 1).any().detach().cpu().item()):
        pred_best_cost = flat_pred.gather(1, best_true.unsqueeze(-1)).squeeze(-1)
        pair_losses: List[torch.Tensor] = []
        for k in range(flat_pred.shape[1]):
            is_valid = flat_valid[:, k] & (best_true != k)
            if not bool(is_valid.any().detach().cpu().item()):
                continue
            true_gap = flat_true[:, k] - min_true.squeeze(-1)
            informative = is_valid & (true_gap > 1e-6)
            if not bool(informative.any().detach().cpu().item()):
                continue
            loss_k = F.relu(float(margin) + pred_best_cost - flat_pred[:, k])
            weight_k = row_weight[informative]
            pair_losses.append((loss_k[informative] * weight_k).sum() / weight_k.sum().clamp_min(1.0))
        if pair_losses:
            margin_loss = torch.stack(pair_losses).mean()

    gap_loss = zero
    if float(gap_weight) > 0.0 and flat_pred.shape[1] > 1:
        route_ids = torch.arange(flat_pred.shape[1], device=device)
        route0_valid = flat_valid[:, :1]
        gap_valid = flat_valid & route0_valid & (route_ids.view(1, -1) > 0)
        if bool(gap_valid.any().detach().cpu().item()):
            pred_gap = (flat_pred - flat_pred[:, :1]) / temp
            true_gap = (flat_true - flat_true[:, :1]) / temp
            gap_elem = F.smooth_l1_loss(pred_gap, true_gap.detach(), reduction="none")
            gap_weights = row_weight.unsqueeze(-1).expand_as(gap_elem)
            weighted_gap = gap_elem[gap_valid] * gap_weights[gap_valid]
            gap_loss = weighted_gap.sum() / gap_weights[gap_valid].sum().clamp_min(1.0)

    raw_loss = ce + float(margin_weight) * margin_loss + float(gap_weight) * gap_loss
    loss = float(loss_weight) * raw_loss
    non0_denom = true_non0.sum().clamp_min(1.0)
    aux = {
        "route_rank_loss": loss.detach(),
        "route_rank_ce": ce.detach(),
        "route_rank_margin_loss": margin_loss.detach(),
        "route_rank_gap_loss": gap_loss.detach(),
        "route_rank_acc": (best_pred == best_true).float().mean().detach(),
        "route_rank_true_non0_frac": true_non0.mean().detach(),
        "route_rank_pred_non0_frac": (best_pred > 0).float().mean().detach(),
        "route_rank_non0_recall": (((best_pred == best_true) & (best_true > 0)).float().sum() / non0_denom).detach(),
        "route_rank_samples": torch.as_tensor(float(flat_pred.shape[0]), device=device),
        "route_rank_od_samples": torch.as_tensor(float(len(sampled_routes)), device=device),
        "route_rank_target_speed_mode": torch.as_tensor(float(mode == "speed"), device=device),
        "route_rank_non0_weight": torch.as_tensor(float(non0_weight), device=device),
        "route_rank_gap_weight": torch.as_tensor(float(gap_weight), device=device),
    }
    return loss, aux


def route_distill_supervision_loss(
    *,
    decoded_dyn: Optional[torch.Tensor],
    route_edge_costs: Optional[torch.Tensor] = None,
    path_edge_indices: torch.Tensor,
    live_costs: torch.Tensor,
    lengths: torch.Tensor,
    speed_limits: torch.Tensor,
    capacities: torch.Tensor,
    loss_weight: float,
    temperature: float = 30.0,
    gap_weight: float = 0.0,
    cost_weight: float = 0.0,
    length_norm: str = "sum",
    pred_cost_mode: str = "speed",
    non0_weight: float = 1.0,
    speed_min_ratio: float = 0.05,
    speed_max_ratio: float = 1.50,
    speed_excess_scale: float = 1.0,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
    volume_index: int = 2,
    speed_index: int = 1,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Distill live dynamic route costs for state-aligned candidate paths.

    ``path_edge_indices`` is per-row, unlike the static route registry used by
    ``route_ranking_supervision_loss``. Each row corresponds to the same SUMO
    state as ``decoded_dyn[row]`` and carries the K dynamic candidate paths
    presented to one vehicle at that decision step.
    """
    device = live_costs.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if decoded_dyn is None and route_edge_costs is None:
        return zero, {}
    if float(loss_weight) <= 0.0:
        return zero, {}
    if path_edge_indices.numel() == 0 or live_costs.numel() == 0:
        return zero, {}

    if route_edge_costs is not None:
        pred_edge_tt = route_edge_costs
        if pred_edge_tt.dim() == 3:
            pred_edge_tt = pred_edge_tt.reshape(-1, pred_edge_tt.shape[-1])
        elif pred_edge_tt.dim() != 2:
            return zero, {}
        mode = "route_head"
    elif decoded_dyn.dim() == 4:
        edge_dyn = decoded_dyn.reshape(-1, decoded_dyn.shape[-2], decoded_dyn.shape[-1])
    elif decoded_dyn.dim() == 3:
        edge_dyn = decoded_dyn
    else:
        return zero, {}
    if route_edge_costs is not None and pred_edge_tt.shape[0] != path_edge_indices.shape[0]:
        return zero, {}
    if route_edge_costs is None and edge_dyn.shape[0] != path_edge_indices.shape[0]:
        return zero, {}

    if route_edge_costs is None:
        dyn_dim = int(edge_dyn.shape[-1])
        mode = str(pred_cost_mode).strip().lower()
        cap = capacities.to(device=device, dtype=torch.float32).clamp_min(1.0).view(1, -1)
        t0 = (
            lengths.to(device=device, dtype=torch.float32)
            / speed_limits.to(device=device, dtype=torch.float32).clamp_min(0.1)
        ).view(1, -1)
        if mode == "bpr_volume":
            vidx = int(volume_index)
            if vidx < 0 or vidx >= dyn_dim:
                return zero, {}
            pred_vol = _positive_pred_volume_for_loss(edge_dyn[..., vidx]) * cap
            pred_edge_tt = t0 * (1.0 + float(bpr_alpha) * (pred_vol / cap).pow(float(bpr_beta)))
        elif mode == "speed":
            sidx = int(speed_index)
            if sidx < 0 or sidx >= dyn_dim:
                return zero, {}
            min_ratio = max(float(speed_min_ratio), 1e-4)
            max_ratio = max(float(speed_max_ratio), min_ratio)
            pred_speed = _positive_pred_speed_for_loss(edge_dyn[..., sidx]).clamp(min=min_ratio, max=max_ratio)
            excess_scale = max(float(speed_excess_scale), 0.0)
            pred_edge_tt = t0 * (1.0 + excess_scale * ((1.0 / pred_speed) - 1.0))
        else:
            raise ValueError(f"Unsupported route distill pred_cost_mode={pred_cost_mode!r}")

    path_edge_indices = path_edge_indices.to(device=device, dtype=torch.long)
    live_costs = live_costs.to(device=device, dtype=torch.float32)
    path_valid = path_edge_indices >= 0
    safe_idx = path_edge_indices.clamp_min(0).clamp_max(pred_edge_tt.shape[-1] - 1)
    gathered = torch.gather(pred_edge_tt, dim=1, index=safe_idx.reshape(safe_idx.shape[0], -1))
    gathered = gathered.reshape_as(safe_idx).masked_fill(~path_valid, 0.0)
    pred_costs = gathered.sum(dim=-1)
    if str(length_norm).strip().lower() == "mean":
        pred_costs = pred_costs / path_valid.sum(dim=-1).clamp_min(1).to(dtype=pred_costs.dtype)
    elif str(length_norm).strip().lower() != "sum":
        raise ValueError(f"Unsupported route distill length_norm={length_norm!r}")

    route_valid = path_valid.any(dim=-1) & torch.isfinite(live_costs) & (live_costs > 0.0)
    row_mask = route_valid.any(dim=-1)
    if not bool(row_mask.any().detach().cpu().item()):
        return zero, {}
    pred = pred_costs[row_mask]
    true = live_costs[row_mask]
    valid = route_valid[row_mask]

    large = torch.finfo(true.dtype).max / 4.0
    masked_true = true.masked_fill(~valid, large)
    masked_pred = pred.masked_fill(~valid, large)
    best_true = torch.argmin(masked_true, dim=-1)
    best_pred = torch.argmin(masked_pred.detach(), dim=-1)
    true_non0 = (best_true > 0).float()
    row_weight = torch.ones_like(true_non0)
    if float(non0_weight) > 1.0:
        row_weight = torch.where(best_true > 0, torch.full_like(row_weight, float(non0_weight)), row_weight)

    temp = max(float(temperature), 1e-6)
    min_true = masked_true.gather(1, best_true.unsqueeze(-1))
    target_logits = -((masked_true - min_true) / temp).masked_fill(~valid, -large)
    target_probs = torch.softmax(target_logits, dim=-1).detach()
    pred_logits = -((masked_pred - masked_pred.detach().amin(dim=-1, keepdim=True)) / temp).masked_fill(~valid, -large)
    ce_per = -(target_probs * F.log_softmax(pred_logits, dim=-1)).sum(dim=-1)
    ce = (ce_per * row_weight).sum() / row_weight.sum().clamp_min(1.0)

    gap_loss = zero
    if float(gap_weight) > 0.0 and pred.shape[1] > 1:
        route_ids = torch.arange(pred.shape[1], device=device)
        gap_valid = valid & valid[:, :1] & (route_ids.view(1, -1) > 0)
        if bool(gap_valid.any().detach().cpu().item()):
            pred_gap = (pred - pred[:, :1]) / temp
            true_gap = (true - true[:, :1]) / temp
            gap_elem = F.smooth_l1_loss(pred_gap, true_gap.detach(), reduction="none")
            gap_weights = row_weight.unsqueeze(-1).expand_as(gap_elem)
            gap_loss = (gap_elem[gap_valid] * gap_weights[gap_valid]).sum() / gap_weights[gap_valid].sum().clamp_min(1.0)

    cost_loss = zero
    if float(cost_weight) > 0.0:
        pred_norm = pred / temp
        true_norm = true / temp
        cost_elem = F.smooth_l1_loss(pred_norm, true_norm.detach(), reduction="none")
        cost_weights = row_weight.unsqueeze(-1).expand_as(cost_elem)
        cost_loss = (cost_elem[valid] * cost_weights[valid]).sum() / cost_weights[valid].sum().clamp_min(1.0)

    raw_loss = ce + float(gap_weight) * gap_loss + float(cost_weight) * cost_loss
    loss = float(loss_weight) * raw_loss
    non0_denom = true_non0.sum().clamp_min(1.0)
    aux = {
        "route_distill_loss": loss.detach(),
        "route_distill_ce": ce.detach(),
        "route_distill_gap_loss": gap_loss.detach(),
        "route_distill_cost_loss": cost_loss.detach(),
        "route_distill_acc": (best_pred == best_true).float().mean().detach(),
        "route_distill_true_non0_frac": true_non0.mean().detach(),
        "route_distill_pred_non0_frac": (best_pred > 0).float().mean().detach(),
        "route_distill_non0_recall": (((best_pred == best_true) & (best_true > 0)).float().sum() / non0_denom).detach(),
        "route_distill_samples": torch.as_tensor(float(pred.shape[0]), device=device),
        "route_distill_gap_weight": torch.as_tensor(float(gap_weight), device=device),
        "route_distill_cost_weight": torch.as_tensor(float(cost_weight), device=device),
    }
    return loss, aux


def inverse_bpr_supervision_loss(
    *,
    inverse_dyn: Optional[torch.Tensor] = None,
    decoded_dyn: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compatibility alias for decoder_bpr_supervision_loss()."""
    if decoded_dyn is None:
        decoded_dyn = inverse_dyn
    return decoder_bpr_supervision_loss(decoded_dyn=decoded_dyn, **kwargs)


def episode_t_all_supervision_loss(
    *,
    flow_projector: torch.nn.Module,
    z_tokens: torch.Tensor,
    reward_targets: Optional[torch.Tensor],
    t_all_weight: float,
    missing_reward_value: float = 0.0,
    target_mode: str = "relative",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Supervise projector-level episode t_all head from collected reward labels.

    In the closed-loop data pipeline, reward is stored as -t_all (minutes),
    repeated over all steps in the same episode. This loss uses the final step
    label for each sequence and regresses predicted t_all directly in minutes.
    """
    device = z_tokens.device
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if reward_targets is None or float(t_all_weight) <= 0.0:
        return zero, {}

    pred_fn = getattr(flow_projector, "predict_t_all_from_z", None)
    if pred_fn is None:
        return zero, {}

    if reward_targets.dim() == 1:
        reward_last = reward_targets.to(device=device, dtype=torch.float32)
    else:
        reward_last = reward_targets[:, -1].to(device=device, dtype=torch.float32)  # (B,)
    target_t_all = -reward_last

    missing = float(missing_reward_value)
    valid = torch.isfinite(target_t_all) & (target_t_all > 0.0)
    if abs(missing) > 0.0:
        valid = valid & (reward_last != missing)
    else:
        valid = valid & (reward_last < 0.0)
    if not bool(valid.any().item()):
        return zero, {
            "t_all_supervision_loss": zero.detach(),
            "t_all_pred_mse": zero.detach(),
            "t_all_pred_mse_normalized": zero.detach(),
            "t_all_pred_mse_raw_minutes": zero.detach(),
            "t_all_target_mean": zero.detach(),
            "t_all_pred_mean": zero.detach(),
            "t_all_target_scale": zero.detach(),
            "t_all_valid_frac": zero.detach(),
        }

    pred_t_all = pred_fn(z_tokens[:, -1])  # (B,)
    pred_valid = pred_t_all[valid].float()
    target_valid = target_t_all[valid].float()
    raw_mse = (pred_valid - target_valid).pow(2).mean()

    mode = str(target_mode).lower()
    if mode == "raw_minutes":
        pred_eval = pred_valid
        target_eval = target_valid
        target_scale = torch.ones((), device=device, dtype=torch.float32)
    elif mode == "log1p":
        pred_eval = torch.log1p(pred_valid.clamp_min(0.0))
        target_eval = torch.log1p(target_valid.clamp_min(0.0))
        target_scale = torch.ones((), device=device, dtype=torch.float32)
    elif mode == "relative":
        target_scale = target_valid.mean().clamp_min(1e-6)
        pred_eval = pred_valid / target_scale
        target_eval = target_valid / target_scale
    else:
        raise ValueError(f"Unsupported t_all target_mode: {target_mode}")

    mse = (pred_eval - target_eval).pow(2).mean()
    loss = float(t_all_weight) * mse
    aux = {
        "t_all_supervision_loss": loss.detach(),
        "t_all_pred_mse": mse.detach(),
        "t_all_pred_mse_normalized": mse.detach(),
        "t_all_pred_mse_raw_minutes": raw_mse.detach(),
        "t_all_target_mean": target_valid.mean().detach(),
        "t_all_pred_mean": pred_valid.mean().detach(),
        "t_all_target_scale": target_scale.detach(),
        "t_all_valid_frac": valid.float().mean().detach(),
    }
    return loss, aux


def cfm_predict_one_timestep(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,          # (B,t,n_spatial,d_spatial)
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
    n_steps: int = 1,                   # 1=Euler, 2=Midpoint-like fixed-step
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable CFM ODE prediction for one-timestep transition supervision."""
    device = past_packed.device
    dtype = past_packed.dtype
    B, t = past_packed.shape[:2]
    n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

    if initial_state is None:
        x = torch.randn((B, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    else:
        x = initial_state.to(device=device, dtype=dtype).clone()
        if tuple(x.shape) != (B, 1, n_spatial, d_spatial):
            raise ValueError(
                "initial_state shape mismatch in cfm_predict_one_timestep: "
                f"expected {(B, 1, n_spatial, d_spatial)}, got {tuple(x.shape)}"
            )
    n_steps = max(1, int(n_steps))
    dt = 1.0 / float(n_steps)

    if act_mask is not None and act_mask.dim() == 1:
        act_mask = act_mask.view(1, 1, -1)

    for i in range(n_steps):
        t_val = float(i) * dt
        flow_time_ctx = torch.zeros(B, t, device=device, dtype=torch.float32)
        flow_time_new = torch.full((B, 1), t_val, device=device, dtype=torch.float32)
        flow_time = torch.cat([flow_time_ctx, flow_time_new], dim=1)  # (B,t+1)

        packed_seq = torch.cat([past_packed, x], dim=1)
        actions_in = _slice_model_actions_time(actions, t + 1)
        actmask_in = None if act_mask is None else act_mask[:, : t + 1]

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            v_pred, _ = dyn(
                actions_in, None, None, packed_seq,
                act_mask=actmask_in,
                agent_tokens=None,
                flow_time=flow_time,
            )
        v = v_pred[:, -1:, :, :]
        x = (x.float() + v.float() * dt).to(dtype)

    return x[:, 0]


def adm_multistep_cfm_loss(
    dynamics: torch.nn.Module,
    *,
    context_z: torch.Tensor,            # (N,k,E,D), usually repeated clean z_t
    target_z: torch.Tensor,             # (N,1,E,D), clean z_{t+k}
    actions: Optional[torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    act_mask: Optional[torch.Tensor],
    flow_targets: Optional[torch.Tensor] = None,
    cfm_active_weight: float = 0.0,
    cfm_busy_weight: float = 0.0,
    cfm_busy_threshold: float = 0.05,
    cfm_active_threshold: float = 0.0,
    cfm_max_weight: float = 50.0,
    volume_index: int = 2,
    init_mode: str = "zero",
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    ADM-style any-step CFM objective.

    The final token is a flow-matching sample for z_{t+k}; previous tokens keep
    z_t fixed while carrying the causal action sequence A_t..A_{t+k-1}.
    """
    device = target_z.device
    N, one = target_z.shape[:2]
    k = int(context_z.shape[1])
    if one != 1:
        raise ValueError(f"ADM target_z must have a singleton time axis, got {tuple(target_z.shape)}")
    if int(context_z.shape[0]) != N:
        raise ValueError("ADM context/target batch mismatch")

    mode = str(init_mode).strip().lower()
    if mode == "zero":
        x0 = torch.zeros_like(target_z)
    elif mode == "randn":
        x0 = torch.randn_like(target_z)
    else:
        raise ValueError(f"Unsupported ADM init_mode={init_mode!r}; expected 'zero' or 'randn'")

    tau = torch.rand(N, 1, device=device, dtype=torch.float32)
    tau_expand = tau[..., None, None]
    x_tau = (1.0 - tau_expand) * x0 + tau_expand * target_z
    target_v = target_z - x0

    packed = torch.cat([context_z, x_tau], dim=1)
    flow_time = torch.cat([
        torch.ones(N, k, device=device, dtype=torch.float32),
        tau,
    ], dim=1)

    v_pred, _ = dynamics(
        actions,
        None,
        None,
        packed,
        act_mask=act_mask,
        agent_tokens=None,
        flow_time=flow_time,
    )
    v_final = v_pred[:, -1:]
    err2 = (v_final.float() - target_v.float()).pow(2)
    edge_mse = err2.mean(dim=-1)
    edge_weight = _cfm_target_edge_weights(
        flow_targets=flow_targets,
        ref=edge_mse,
        volume_index=volume_index,
        active_weight=cfm_active_weight,
        busy_weight=cfm_busy_weight,
        busy_threshold=cfm_busy_threshold,
        active_threshold=cfm_active_threshold,
        max_weight=cfm_max_weight,
    )
    flow_mse = edge_mse.mean()
    if edge_weight is None:
        weighted_flow_mse = flow_mse
        cfm_weight_mean = torch.ones((), device=device, dtype=torch.float32)
        cfm_active_frac = torch.zeros((), device=device, dtype=torch.float32)
        cfm_busy_frac = torch.zeros((), device=device, dtype=torch.float32)
    else:
        weighted_flow_mse = (edge_mse * edge_weight).sum() / edge_weight.sum().clamp_min(1.0)
        cfm_weight_mean = edge_weight.mean()
        target_vol = flow_targets[..., int(volume_index)].to(device=device, dtype=torch.float32).clamp_min(0.0)
        cfm_active_frac = (target_vol > float(cfm_active_threshold)).float().mean()
        cfm_busy_frac = (target_vol > float(cfm_busy_threshold)).float().mean()
    return weighted_flow_mse, {
        "adm_flow_mse": flow_mse.detach(),
        "adm_weighted_flow_mse": weighted_flow_mse.detach(),
        "adm_cfm_weight_mean": cfm_weight_mean.detach(),
        "adm_cfm_active_edge_frac": cfm_active_frac.detach(),
        "adm_cfm_busy_edge_frac": cfm_busy_frac.detach(),
        "adm_t_mean": tau.mean().detach(),
    }


def adm_predict_k_timestep_direct(
    dyn: Dynamics,
    *,
    context_z: torch.Tensor,            # (N,k,E,D)
    actions: Optional[torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
    act_mask: Optional[torch.Tensor] = None,
    n_steps: int = 1,
    init_mode: str = "zero",
) -> torch.Tensor:
    """Differentiable direct k-step CFM sampler used for decoded ADM supervision."""
    device = context_z.device
    dtype = context_z.dtype
    N, k = context_z.shape[:2]
    n_spatial, d_spatial = context_z.shape[2], context_z.shape[3]
    mode = str(init_mode).strip().lower()
    if mode == "zero":
        x = torch.zeros((N, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    elif mode == "randn":
        x = torch.randn((N, 1, n_spatial, d_spatial), device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported ADM init_mode={init_mode!r}; expected 'zero' or 'randn'")

    n_steps = max(1, int(n_steps))
    dt = 1.0 / float(n_steps)
    for i in range(n_steps):
        t_val = float(i) * dt
        packed = torch.cat([context_z, x], dim=1)
        flow_time = torch.cat([
            torch.ones(N, k, device=device, dtype=torch.float32),
            torch.full((N, 1), t_val, device=device, dtype=torch.float32),
        ], dim=1)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            v_pred, _ = dyn(
                actions,
                None,
                None,
                packed,
                act_mask=act_mask,
                agent_tokens=None,
                flow_time=flow_time,
            )
        x = (x.float() + v_pred[:, -1:].float() * dt).to(dtype)
    return x[:, 0]


@torch.no_grad()
def cfm_sample_one_timestep(
    dyn: Dynamics,
    *,
    past_packed: torch.Tensor,          # (B,t,n_spatial,d_spatial)
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
    n_steps: int = 1,                   # 1=Euler, 2=Midpoint-like fixed-step
) -> torch.Tensor:
    """CFM ODE sampling for one-timestep prediction."""
    return cfm_predict_one_timestep(
        dyn,
        past_packed=past_packed,
        actions=actions,
        act_mask=act_mask,
        n_steps=n_steps,
    )


@torch.no_grad()
def cfm_sample_autoregressive_sequence(
    dyn: Dynamics,
    *,
    z_gt_packed: torch.Tensor,
    ctx_length: int,
    horizon: int,
    actions: Optional[torch.Tensor] = None,
    act_mask: Optional[torch.Tensor] = None,
    n_ode_steps: int = 1,
) -> torch.Tensor:
    """Autoregressive multi-step CFM prediction."""
    B, T = z_gt_packed.shape[:2]
    L = min(T, ctx_length + horizon)
    ctx_length = min(ctx_length, L - 1)
    horizon = min(horizon, L - ctx_length)

    outs = [z_gt_packed[:, t] for t in range(ctx_length)]
    for _t in range(ctx_length, ctx_length + horizon):
        past = torch.stack(outs, dim=1)
        z_next = cfm_sample_one_timestep(
            dyn,
            past_packed=past,
            actions=actions,
            act_mask=act_mask,
            n_steps=n_ode_steps,
        )
        outs.append(z_next)
    return torch.stack(outs, dim=1)


def _arg_with_legacy(args, primary: str, legacy: str, default):
    value = getattr(args, primary, None)
    if value is not None:
        return value
    legacy_value = getattr(args, legacy, None)
    if legacy_value is not None:
        return legacy_value
    return default


def train(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seed_everything(args.seed + rank)
    setattr(args, "hdv_policy_mode", "edge")

    # Dataset and DataLoader — flow mode
    if not args.edge_registry:
        raise ValueError("--edge_registry is required")

    edge_registry = EdgeRegistry.load(args.edge_registry)
    _n_edges = edge_registry.n_edges
    _d_feat  = getattr(args, "d_feat", 8)
    _lengths_np = edge_registry.lengths
    if _lengths_np is None:
        _lengths_np = np.ones((_n_edges,), dtype=np.float32) * 100.0
    _speed_np = np.asarray(edge_registry.speed_limits, dtype=np.float32)
    _cap_np = np.asarray(edge_registry.capacities, dtype=np.float32)

    # When Stackelberg is active, disable neural hdv_policy_head
    _use_stackelberg = bool(getattr(args, "use_stackelberg", False))
    if _use_stackelberg:
        if is_rank0():
            print("[info] Stackelberg mode: forcing hdv_policy_dim=0 (no neural HDV head)")
        setattr(args, "hdv_policy_dim", 0)

    if float(getattr(args, "hdv_policy_weight", 0.0)) > 0.0 and not _use_stackelberg:
        hdv_dim = int(getattr(args, "hdv_policy_dim", 0))
        if hdv_dim <= 0:
            hdv_dim = int(_n_edges)
            setattr(args, "hdv_policy_dim", hdv_dim)
            if is_rank0():
                print(
                    f"[info] auto-set --hdv_policy_dim to n_edges={hdv_dim}"
                )
        elif hdv_dim != int(_n_edges):
            raise ValueError(
                "--hdv_policy_dim must equal n_edges in edge-only mode "
                f"({int(_n_edges)}), got {hdv_dim}"
            )

    # Determine task names for each data dir
    if getattr(args, "flow_task_names", None):
        _task_names = args.flow_task_names
        _task_name  = _task_names[0] if _task_names else ""
    elif getattr(args, "flow_task_name", ""):
        _task_name  = args.flow_task_name
        _task_names = None
    else:
        _task_name  = Path(args.data_dirs[0]).name if args.data_dirs else "dynamics"
        _task_names = None

    _hdv_policy_action_mode = str(
        getattr(args, "hdv_policy_action_mode", "step") or "step"
    ).strip().lower()
    _return_episode_action_mean = _hdv_policy_action_mode in {
        "episode_mean",
        "episode_response",
        "response",
    }

    dataset = FlowDataset(
        data_dirs=args.data_dirs,
        task_name=_task_name,
        task_names=_task_names,
        seq_len=args.seq_len,
        action_key=getattr(args, "action_key", "av_planned_edge_flow"),
        aux_action_key=(getattr(args, "aux_action_key", "") or None),
        aux_target_mode=str(getattr(args, "hdv_policy_target_mode", "step")),
        return_episode_action_mean=_return_episode_action_mean,
        reward_key=(getattr(args, "reward_key", "") or None),
        paired_group_key=getattr(args, "paired_group_key", "pair_id"),
        paired_variant_key=getattr(args, "paired_variant_key", "variant_id"),
    )

    if float(getattr(args, "hdv_policy_weight", 0.0)) > 0.0:
        aux_all = getattr(dataset, "_aux_act_all", None)
        if aux_all is None:
            raise ValueError(
                "--hdv_policy_weight > 0 requires aux_action_key data in dataset; "
                f"aux_action_key={getattr(args, 'aux_action_key', '')!r}"
            )
        aux_dim = int(aux_all.shape[-1]) if aux_all.ndim > 1 else 1
        if aux_dim <= 0:
            # Dataset has the key but zero-width tensor — disable HDV supervision gracefully
            if is_rank0():
                print(
                    "[warn] HDV aux_action has zero dimension in dataset — "
                    "disabling hdv_policy_weight and hdv_policy_dim."
                )
            setattr(args, "hdv_policy_weight", 0.0)
            setattr(args, "hdv_policy_dim", 0)
        else:
            exp_dim = int(getattr(args, "hdv_policy_dim", 0))
            if exp_dim > 0 and aux_dim != exp_dim:
                raise ValueError(
                    "HDV supervision dimension mismatch: "
                    f"dataset aux dim={aux_dim}, hdv_policy_dim={exp_dim}. "
                    "Check --aux_action_key."
                )

    if (
        is_rank0()
        and float(getattr(args, "hdv_policy_weight", 0.0)) > 0.0
        and getattr(dataset, "_aux_act_all", None) is not None
    ):
        try:
            aux_all = dataset._aux_act_all.to(torch.float32)  # (N, K)
            if aux_all.ndim == 1:
                aux_all = aux_all.unsqueeze(-1)
            aux_prob = aux_all / aux_all.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            ent = -(aux_prob * aux_prob.clamp_min(1e-8).log()).sum(dim=-1)
            comp_std = aux_prob.std(dim=0)
            max_probe = min(int(aux_prob.shape[0]), 20000)
            probe = aux_prob[:max_probe]
            uniq_probe = torch.unique((probe * 10000.0).round().to(torch.int64), dim=0).shape[0]
            print(
                "HDV target stats: "
                f"N={int(aux_prob.shape[0])}, D={int(aux_prob.shape[-1])}, "
                f"entropy(mean/min/max)=({float(ent.mean()):.6f}/{float(ent.min()):.6f}/{float(ent.max()):.6f}), "
                f"comp_std_max={float(comp_std.max()):.6f}, "
                f"uniq_rows_probe={int(uniq_probe)}/{int(max_probe)}"
            )
            if float(comp_std.max()) < 1e-4 or int(uniq_probe) <= 1:
                print(
                    "[warn] HDV supervision target appears near-constant. "
                    "hdv_policy_ce may stay close to ln(D) and provide little training signal."
                )
        except Exception as exc:
            print(f"[warn] Failed to summarize HDV target stats: {exc}")

    action_compressor_state: Optional[Dict[str, Any]] = None
    depart_compressor_state: Optional[Dict[str, Any]] = None
    action_compress_mode = str(getattr(args, "action_compress", "none")).lower()
    if action_compress_mode not in {"none", "pca"}:
        raise ValueError(f"Unknown --action_compress={action_compress_mode!r}; expected 'none' or 'pca'.")

    action_encoder_type = str(getattr(args, "action_encoder_type", "mlp")).lower()
    if action_encoder_type in {"edge_gnn", "graph_cond"} and action_compress_mode != "none":
        raise ValueError(f"{action_encoder_type} requires raw edge inputs; use --action_compress none")
    if bool(getattr(args, "edge_action_injection", False)) and action_compress_mode != "none":
        raise ValueError("edge_action_injection requires raw edge inputs; use --action_compress none")

    if action_compress_mode == "pca":
        if not hasattr(dataset, "_act_all") or dataset._act_all is None:
            raise RuntimeError("FlowDataset has no _act_all tensor; cannot fit/apply PCA action compression.")
        acts_raw = dataset._act_all.to(torch.float32).cpu()  # (N, A_raw)
        raw_dim = int(acts_raw.shape[-1])
        req_dim = int(getattr(args, "action_pca_dim", 32))
        req_dim = max(1, min(req_dim, raw_dim))

        # Priority:
        # 1) explicit action_pca_ckpt
        # 2) compressor embedded in resume checkpoint
        # 3) fit from current dataset
        loaded_comp: Optional[Dict[str, Any]] = None
        pca_ckpt = str(getattr(args, "action_pca_ckpt", "") or "")
        if pca_ckpt and Path(pca_ckpt).exists():
            try:
                obj = torch.load(pca_ckpt, map_location="cpu", weights_only=False)
                comp_obj = obj.get("action_compressor", obj)
                if isinstance(comp_obj, dict) and str(comp_obj.get("type", "")) == "pca_linear":
                    loaded_comp = {
                        **comp_obj,
                        "mean": torch.as_tensor(comp_obj["mean"], dtype=torch.float32).cpu(),
                        "proj": torch.as_tensor(comp_obj["proj"], dtype=torch.float32).cpu(),
                    }
            except Exception as exc:
                if is_rank0():
                    print(f"[warn] Failed to load --action_pca_ckpt={pca_ckpt}: {exc}")

        if loaded_comp is None:
            loaded_comp = _maybe_load_action_compressor_from_ckpt(getattr(args, "resume", None))
        loaded_depart_comp = _maybe_load_depart_compressor_from_ckpt(getattr(args, "resume", None))

        if loaded_comp is not None:
            in_dim = int(loaded_comp.get("input_dim", -1))
            out_dim = int(loaded_comp.get("output_dim", -1))
            if in_dim != raw_dim:
                if is_rank0():
                    print(
                        f"[warn] Ignore loaded action compressor due to input_dim mismatch: "
                        f"loaded={in_dim}, dataset={raw_dim}"
                    )
                loaded_comp = None
            elif out_dim <= 0:
                loaded_comp = None

        if loaded_comp is None:
            loaded_comp = _fit_linear_pca(acts_raw, req_dim)
            if is_rank0():
                print(
                    "Fitted action PCA: "
                    f"input_dim={int(loaded_comp['input_dim'])}, "
                    f"output_dim={int(loaded_comp['output_dim'])}, "
                    f"explained_var={float(loaded_comp.get('explained_variance_ratio', 0.0)):.6f}"
                )
        else:
            if is_rank0():
                print(
                    "Loaded action PCA compressor: "
                    f"input_dim={int(loaded_comp.get('input_dim', -1))}, "
                    f"output_dim={int(loaded_comp.get('output_dim', -1))}"
                )

        action_compressor_state = loaded_comp
        setattr(args, "action_raw_dim", raw_dim)
        setattr(args, "action_pca_dim_effective", int(loaded_comp["output_dim"]))

        if getattr(dataset, "_depart_all", None) is not None:
            dep_raw = dataset._depart_all.to(torch.float32).cpu()
            dep_dim = int(dep_raw.shape[-1])
            dep_req_dim = int(getattr(args, "depart_pca_dim", req_dim))
            dep_req_dim = max(1, min(dep_req_dim, dep_dim))
            if loaded_depart_comp is not None:
                dep_in_dim = int(loaded_depart_comp.get("input_dim", -1))
                dep_out_dim = int(loaded_depart_comp.get("output_dim", -1))
                if dep_in_dim == dep_dim and dep_out_dim > 0:
                    depart_compressor_state = loaded_depart_comp
                    if is_rank0():
                        print(
                            "Loaded depart PCA compressor: "
                            f"input_dim={int(loaded_depart_comp.get('input_dim', -1))}, "
                            f"output_dim={int(loaded_depart_comp.get('output_dim', -1))}"
                        )
                else:
                    loaded_depart_comp = None
            if depart_compressor_state is None:
                depart_compressor_state = _fit_linear_pca(dep_raw, dep_req_dim)
                if is_rank0():
                    print(
                        "Fitted depart PCA: "
                        f"input_dim={int(depart_compressor_state['input_dim'])}, "
                        f"output_dim={int(depart_compressor_state['output_dim'])}, "
                        f"explained_var={float(depart_compressor_state.get('explained_variance_ratio', 0.0)):.6f}"
                    )
            setattr(args, "depart_raw_dim", dep_dim)
            setattr(args, "depart_pca_dim_effective", int(depart_compressor_state["output_dim"]))

        if is_rank0():
            default_path = Path(args.ckpt_dir) / "action_compressor.pt"
            out_path = Path(pca_ckpt) if pca_ckpt else default_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "action_compressor": action_compressor_state,
                "depart_compressor": depart_compressor_state,
            }, out_path)
            print(f"Saved action compressor -> {out_path}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
        worker_init_fn=worker_init_fn,
    )

    # Flow mode: no tokenizer — derive n_spatial, d_spatial from edge registry
    n_spatial     = _n_edges    # each edge = one spatial token
    d_spatial     = 64          # flow latent dimension (fixed)
    d_bottleneck  = 64          # packing_factor=1 in flow mode

    enable_t_all_head = bool(getattr(args, "enable_t_all_head", False)) or (
        float(getattr(args, "t_all_supervision_weight", 0.0)) > 0.0
    )
    flow_projector = FlowProjector(
        d_feat=_d_feat,
        d_spatial=d_spatial,
        enable_t_all_head=enable_t_all_head,
    )
    if args.flow_projector_ckpt and Path(args.flow_projector_ckpt).exists():
        fp_ckpt = torch.load(args.flow_projector_ckpt, map_location="cpu", weights_only=False)
        fp_state = dict(fp_ckpt["model"])
        # Older projector checkpoints persisted adjacency in the model state and
        # often predate optional heads such as t_all_head. Treat projector
        # warm-starts like the main resume path: allow non-strict loading when
        # requested and ignore runtime-derived adjacency.
        fp_state.pop("adj", None)
        fp_incompat = flow_projector.load_state_dict(
            fp_state,
            strict=(not bool(getattr(args, "resume_non_strict", False))),
        )
        if is_rank0():
            print(f"FlowProjector resumed from {args.flow_projector_ckpt}")
            if bool(getattr(args, "resume_non_strict", False)):
                if fp_incompat.missing_keys:
                    print(f"  FlowProjector missing keys: {sorted(fp_incompat.missing_keys)}")
                if fp_incompat.unexpected_keys:
                    print(f"  FlowProjector unexpected keys: {sorted(fp_incompat.unexpected_keys)}")
    flow_projector = flow_projector.to(device).train()
    freeze_flow_projector = bool(getattr(args, "freeze_flow_projector", False))

    # Set adjacency for GNN layers
    if edge_registry.adjacency is not None:
        flow_projector.set_adjacency(
            torch.from_numpy(edge_registry.adjacency).to(device)
        )

    assert flow_projector.d_spatial == d_spatial, (
        f"FlowProjector.d_spatial={flow_projector.d_spatial} != d_spatial={d_spatial}. "
        "Ensure --d_feat and checkpoint are consistent."
    )
    decoder_recon_weight = float(_arg_with_legacy(args, "decoder_recon_weight", "inverse_recon_weight", 0.0))
    decoder_recon_speed_weight = float(
        _arg_with_legacy(args, "decoder_recon_speed_weight", "inverse_speed_weight", 4.0)
    )
    decoder_recon_speed_index = int(
        _arg_with_legacy(args, "decoder_recon_speed_index", "inverse_speed_index", 1)
    )
    if decoder_recon_weight > 0.0:
        sidx = int(decoder_recon_speed_index)
        if (sidx < 0) or (sidx >= int(flow_projector.n_dynamic)):
            print(
                f"[warn] decoder_recon_speed_index={sidx} out of range for "
                f"n_dynamic={int(flow_projector.n_dynamic)}; speed focus disabled."
            )

    # Initialize obs_proj for latent obs if requested
    d_obs = int(getattr(args, "d_obs", 0))
    if d_obs > 0:
        flow_projector.init_obs_proj(d_obs)
        # Move newly created obs_proj to same device as flow_projector
        flow_projector.obs_proj = flow_projector.obs_proj.to(device)
        if is_rank0():
            print(f"[obs_proj] initialized: d_spatial={d_spatial} → d_obs={d_obs}")

    if freeze_flow_projector:
        flow_projector.eval()
        for p in flow_projector.parameters():
            p.requires_grad_(False)
        if is_rank0():
            print("[freeze] FlowProjector encoder/decoder parameters frozen; training dynamics/CFM only.")

    decoded_busy_volume_weight = float(getattr(args, "decoded_busy_volume_weight", 0.0))
    _decoder_recon_busy_volume_weight = _arg_with_legacy(
        args,
        "decoder_recon_busy_volume_weight",
        "inverse_busy_volume_weight",
        None,
    )
    _decoder_busy_volume_weight = getattr(args, "decoder_busy_volume_weight", None)
    _transition_busy_volume_weight = getattr(args, "transition_busy_volume_weight", None)
    decoded_busy_presence_weight = float(getattr(args, "decoded_busy_presence_weight", 0.0))
    _decoder_recon_busy_presence_weight = getattr(args, "decoder_recon_busy_presence_weight", None)
    _decoder_busy_presence_weight = getattr(args, "decoder_busy_presence_weight", None)
    decoded_topk_rank_weight = float(getattr(args, "decoded_topk_rank_weight", 0.0))
    _decoder_recon_topk_rank_weight = getattr(args, "decoder_recon_topk_rank_weight", None)
    _decoder_topk_rank_weight = getattr(args, "decoder_topk_rank_weight", None)
    decoded_zero_volume_weight = float(getattr(args, "decoded_zero_volume_weight", 0.0))
    _decoder_recon_zero_volume_weight = getattr(args, "decoder_recon_zero_volume_weight", None)
    _decoder_zero_volume_weight = getattr(args, "decoder_zero_volume_weight", None)
    _transition_zero_volume_weight = getattr(args, "transition_zero_volume_weight", None)
    decoder_recon_busy_volume_weight = (
        float(_decoder_recon_busy_volume_weight)
        if _decoder_recon_busy_volume_weight is not None
        else decoded_busy_volume_weight
    )
    decoder_busy_volume_weight = (
        float(_decoder_busy_volume_weight)
        if _decoder_busy_volume_weight is not None
        else decoded_busy_volume_weight
    )
    transition_busy_volume_weight = (
        float(_transition_busy_volume_weight)
        if _transition_busy_volume_weight is not None
        else decoded_busy_volume_weight
    )
    decoder_recon_busy_presence_weight = (
        float(_decoder_recon_busy_presence_weight)
        if _decoder_recon_busy_presence_weight is not None
        else decoded_busy_presence_weight
    )
    decoder_busy_presence_weight = (
        float(_decoder_busy_presence_weight)
        if _decoder_busy_presence_weight is not None
        else decoded_busy_presence_weight
    )
    decoder_recon_topk_rank_weight = (
        float(_decoder_recon_topk_rank_weight)
        if _decoder_recon_topk_rank_weight is not None
        else decoded_topk_rank_weight
    )
    decoder_topk_rank_weight = (
        float(_decoder_topk_rank_weight)
        if _decoder_topk_rank_weight is not None
        else decoded_topk_rank_weight
    )
    decoder_recon_zero_volume_weight = (
        float(_decoder_recon_zero_volume_weight)
        if _decoder_recon_zero_volume_weight is not None
        else decoded_zero_volume_weight
    )
    decoder_zero_volume_weight = (
        float(_decoder_zero_volume_weight)
        if _decoder_zero_volume_weight is not None
        else decoded_zero_volume_weight
    )
    transition_zero_volume_weight = (
        float(_transition_zero_volume_weight)
        if _transition_zero_volume_weight is not None
        else decoded_zero_volume_weight
    )
    transition_busy_presence_weight = float(getattr(args, "transition_busy_presence_weight", 0.0))
    decoded_busy_presence_margin = float(getattr(args, "decoded_busy_presence_margin", 0.05))
    decoded_topk_rank_k = int(getattr(args, "decoded_topk_rank_k", 20))
    decoded_topk_rank_margin = float(getattr(args, "decoded_topk_rank_margin", 0.05))
    decoded_zero_volume_threshold = float(
        getattr(args, "decoded_zero_volume_threshold", getattr(args, "decoded_busy_threshold", 0.05))
    )
    decoded_zero_volume_margin = float(getattr(args, "decoded_zero_volume_margin", 0.05))
    decoded_zero_target_threshold = float(getattr(args, "decoded_zero_target_threshold", 0.0))
    transition_topk_rank_weight = float(getattr(args, "transition_topk_rank_weight", 0.0))
    transition_topk_rank_k = int(getattr(args, "transition_topk_rank_k", 20))
    transition_topk_rank_margin = float(getattr(args, "transition_topk_rank_margin", 0.05))
    route_ranking_weight = float(getattr(args, "route_ranking_weight", 0.0))
    transition_route_ranking_weight = float(getattr(args, "transition_route_ranking_weight", 0.0))
    route_ranking_max_od = int(getattr(args, "route_ranking_max_od", 64))
    route_ranking_temperature = float(getattr(args, "route_ranking_temperature", 30.0))
    route_ranking_margin_weight = float(getattr(args, "route_ranking_margin_weight", 0.0))
    route_ranking_margin = float(getattr(args, "route_ranking_margin", 1.0))
    route_ranking_gap_weight = float(getattr(args, "route_ranking_gap_weight", 0.0))
    route_ranking_non0_weight = float(getattr(args, "route_ranking_non0_weight", 1.0))
    route_ranking_speed_min_ratio = float(getattr(args, "route_ranking_speed_min_ratio", 0.05))
    route_ranking_speed_max_ratio = float(getattr(args, "route_ranking_speed_max_ratio", 1.50))
    route_ranking_speed_excess_scale = float(getattr(args, "route_ranking_speed_excess_scale", 1.0))
    route_ranking_length_norm = str(getattr(args, "route_ranking_length_norm", "sum")).strip().lower()
    if route_ranking_length_norm not in {"sum", "mean"}:
        raise ValueError(f"Unsupported route_ranking_length_norm={route_ranking_length_norm!r}; expected 'sum' or 'mean'")
    route_ranking_target_cost = str(getattr(args, "route_ranking_target_cost", "speed")).strip().lower()
    if route_ranking_target_cost not in {"bpr_volume", "speed"}:
        raise ValueError(f"Unsupported route_ranking_target_cost={route_ranking_target_cost!r}; expected 'bpr_volume' or 'speed'")
    route_distill_weight = float(getattr(args, "route_distill_weight", 0.0))
    route_distill_pt = str(getattr(args, "route_distill_pt", "") or "")
    route_distill_batch_size = int(getattr(args, "route_distill_batch_size", max(1, min(int(args.batch_size), 64))))
    route_distill_temperature = float(getattr(args, "route_distill_temperature", route_ranking_temperature))
    route_distill_gap_weight = float(getattr(args, "route_distill_gap_weight", route_ranking_gap_weight))
    route_distill_cost_weight = float(getattr(args, "route_distill_cost_weight", 0.0))
    route_distill_non0_weight = float(getattr(args, "route_distill_non0_weight", route_ranking_non0_weight))
    route_distill_pred_cost = str(getattr(args, "route_distill_pred_cost", route_ranking_target_cost)).strip().lower()
    if route_distill_pred_cost not in {"bpr_volume", "speed", "route_head"}:
        raise ValueError(f"Unsupported route_distill_pred_cost={route_distill_pred_cost!r}; expected 'bpr_volume', 'speed', or 'route_head'")
    route_distill_length_norm = str(getattr(args, "route_distill_length_norm", route_ranking_length_norm)).strip().lower()
    if route_distill_length_norm not in {"sum", "mean"}:
        raise ValueError(f"Unsupported route_distill_length_norm={route_distill_length_norm!r}; expected 'sum' or 'mean'")
    transition_cfm_init_mode = str(getattr(args, "transition_cfm_init_mode", "randn")).strip().lower()
    if transition_cfm_init_mode not in {"randn", "zero"}:
        raise ValueError(f"Unsupported transition_cfm_init_mode={transition_cfm_init_mode!r}; expected 'randn' or 'zero'")
    cfm_active_weight = float(getattr(args, "cfm_active_weight", 0.0))
    cfm_busy_weight = float(getattr(args, "cfm_busy_weight", 0.0))
    cfm_busy_threshold = float(getattr(args, "cfm_busy_threshold", getattr(args, "decoded_busy_threshold", 0.05)))
    cfm_active_threshold = float(getattr(args, "cfm_active_threshold", 0.0))
    cfm_max_weight = float(getattr(args, "cfm_max_weight", 50.0))
    adm_cfm_horizons = _parse_adm_horizons(getattr(args, "adm_cfm_horizons", ""))
    adm_cfm_weight = float(getattr(args, "adm_cfm_weight", 0.0))
    adm_cfm_gamma = float(getattr(args, "adm_cfm_gamma", 0.7))
    adm_decoded_supervision_weight = float(getattr(args, "adm_decoded_supervision_weight", 0.0))
    adm_max_windows = int(getattr(args, "adm_max_windows", 0))
    adm_cfm_init_mode = str(getattr(args, "adm_cfm_init_mode", transition_cfm_init_mode)).strip().lower()
    if adm_cfm_init_mode not in {"randn", "zero"}:
        raise ValueError(f"Unsupported adm_cfm_init_mode={adm_cfm_init_mode!r}; expected 'randn' or 'zero'")
    if is_rank0():
        print(f"n_spatial={n_spatial} (edges), d_spatial={d_spatial}, d_feat={_d_feat}")

    # Infer action dimensionality directly from dataset key (action_key),
    # so dynamics supports both legacy K-route actions and edge-level actions.
    action_dim_flow = int(getattr(dataset, "n_actions", 0))
    if action_dim_flow <= 0:
        raise RuntimeError(
            "Failed to infer action dimension from FlowDataset. "
            "Check action_key/aux_action_key and dataset format."
        )
    model_action_uses_depart_context = action_encoder_type not in {"edge_gnn"}
    if action_encoder_type == "edge_gnn":
        model_action_dim = int(action_dim_flow)
    else:
        model_action_dim = _infer_model_action_dim(
            raw_action_dim=action_dim_flow,
            action_compressor=action_compressor_state,
            depart_compressor=depart_compressor_state,
            depart_dim=(int(dataset._depart_all.shape[-1]) if getattr(dataset, "_depart_all", None) is not None else 0),
            context_dim=(int(dataset._context_all.shape[-1]) if getattr(dataset, "_context_all", None) is not None else 0),
        )
    # Persist inferred action dim in args so UrbanWorldModel can recover it
    # robustly from checkpoint metadata.
    setattr(args, "action_dim", int(model_action_dim))
    if is_rank0():
        n_routes_cfg = int(getattr(args, "n_routes", 4))
        if n_routes_cfg != action_dim_flow:
            print(
                "[info] action dimension inferred from dataset differs from --n_routes: "
                f"action_dim={action_dim_flow}, n_routes={n_routes_cfg} "
                f"(action_key={getattr(args, 'action_key', 'av_planned_edge_flow')!r})"
            )
    action_encoder_type = str(getattr(args, "action_encoder_type", "mlp")).lower()
    if action_encoder_type == "mlp":
        action_encoder = ActionEncoder(d_model=args.d_model_dyn, action_dim=model_action_dim)
    elif action_encoder_type == "edge_gnn":
        if action_dim_flow != n_spatial:
            raise ValueError(
                "edge_gnn action encoder requires edge-level action_dim == n_edges. "
                f"Got action_dim={action_dim_flow}, n_edges={n_spatial}. "
                "Disable PCA compression or use --action_encoder_type mlp."
            )
        action_encoder = EdgeGNNActionEncoder(
            d_model=args.d_model_dyn,
            n_edges=n_spatial,
            n_static=int(edge_registry.static_features.shape[1]) if edge_registry.static_features is not None else 0,
            d_edge=int(getattr(args, "action_gnn_d_edge", 64)),
            n_gnn_layers=int(getattr(args, "action_gnn_layers", 2)),
        )
        if edge_registry.static_features is not None:
            action_encoder.set_static_features(
                torch.from_numpy(np.asarray(edge_registry.static_features, dtype=np.float32))
            )
        if edge_registry.adjacency is not None:
            action_encoder.set_adjacency(
                torch.from_numpy(np.asarray(edge_registry.adjacency, dtype=np.float32))
            )
    elif action_encoder_type == "graph_cond":
        if action_dim_flow != n_spatial:
            raise ValueError(
                "graph_cond action encoder requires edge-level action_dim == n_edges. "
                f"Got action_dim={action_dim_flow}, n_edges={n_spatial}."
            )
        action_encoder = ActionDepartGNNEncoder(
            d_model=args.d_model_dyn,
            n_edges=n_spatial,
            n_static=int(edge_registry.static_features.shape[1]) if edge_registry.static_features is not None else 0,
            d_edge=int(getattr(args, "cond_gnn_hidden_dim", 64)),
            n_gnn_layers=int(getattr(args, "cond_gnn_layers", 2)),
            context_dim=3,
            context_hidden=int(getattr(args, "cond_context_hidden_dim", 64)),
            use_hdv_context=str(getattr(args, "transition_hdv_context_mode", "none") or "none").strip().lower()
            not in {"", "none", "off", "disabled"},
        )
        if edge_registry.static_features is not None:
            action_encoder.set_static_features(
                torch.from_numpy(np.asarray(edge_registry.static_features, dtype=np.float32))
            )
        else:
            action_encoder.set_static_features(torch.zeros(n_spatial, 0, dtype=torch.float32))
        if edge_registry.adjacency is not None:
            action_encoder.set_adjacency(
                torch.from_numpy(np.asarray(edge_registry.adjacency, dtype=np.float32))
            )
    else:
        raise ValueError(
            f"Unknown --action_encoder_type={action_encoder_type!r}; expected 'mlp', 'edge_gnn', or 'graph_cond'."
        )

    # Build dynamics model
    dyn = Dynamics(
        d_model=args.d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=args.n_register,
        n_agent=args.n_agent,
        n_heads=args.n_heads,
        depth=args.dyn_depth,
        k_max=args.k_max,
        dropout=args.dropout,
        mlp_ratio=args.mlp_ratio,
        time_every=args.time_every,
        space_mode=args.space_mode,
        action_encoder=action_encoder,
        use_cfm=True,
        hdv_policy_dim=int(getattr(args, "hdv_policy_dim", 0)),
        hdv_policy_mode="edge",
        edge_action_injection=bool(getattr(args, "edge_action_injection", False)),
        edge_action_injection_eps=float(getattr(args, "edge_action_injection_eps", 1e-6)),
        edge_action_injection_scale=float(getattr(args, "edge_action_injection_scale", 1.0)),
        edge_action_value_norm=str(getattr(args, "edge_action_value_norm", "raw")),
        edge_action_id_embedding=bool(getattr(args, "edge_action_id_embedding", False)),
        edge_action_tokenizer=bool(getattr(args, "edge_action_tokenizer", False)),
        edge_action_tokenizer_fuse_encoder=bool(getattr(args, "edge_action_tokenizer_fuse_encoder", False)),
        edge_action_output_injection=bool(getattr(args, "edge_action_output_injection", False)),
        edge_action_output_from_injection=bool(getattr(args, "edge_action_output_from_injection", False)),
        action_reconstruction=(float(getattr(args, "action_reconstruction_weight", 0.0)) > 0.0),
    ).to(device)

    train_route_cost_head_only = bool(getattr(args, "train_route_cost_head_only", False))
    if train_route_cost_head_only:
        for p in dyn.parameters():
            p.requires_grad_(False)
        for p in flow_projector.parameters():
            p.requires_grad_(False)
        for p in flow_projector.route_cost_head.parameters():
            p.requires_grad_(True)
        freeze_flow_projector = False
        if is_rank0():
            print("[freeze] Training only FlowProjector.route_cost_head for planner-cost calibration.")

    # --- Edge attribute tensors (used by Stackelberg + training loop) ---
    lengths_t = torch.from_numpy(np.asarray(_lengths_np, dtype=np.float32)).to(device=device, dtype=torch.float32)
    speed_limits_t = torch.from_numpy(_speed_np).to(device=device, dtype=torch.float32)
    capacities_t = torch.from_numpy(_cap_np).to(device=device, dtype=torch.float32)

    route_ranking_route_edges: Optional[List[List[List[int]]]] = None
    if route_ranking_weight > 0.0 or transition_route_ranking_weight > 0.0:
        routes_csv_for_ranking = str(getattr(args, "route_ranking_routes_csv", "") or "")
        rr_path_for_ranking = str(getattr(args, "route_registry", "") or "")
        route_ranking_k = int(getattr(args, "n_routes", 4))
        if routes_csv_for_ranking:
            route_ranking_route_edges = _load_route_edges_from_routes_csv(
                routes_csv_for_ranking,
                getattr(edge_registry, "_edge_to_idx", {}),
                n_routes=route_ranking_k,
            )
            route_source = routes_csv_for_ranking
        elif rr_path_for_ranking:
            route_ranking_registry = RouteRegistry.load(rr_path_for_ranking)
            route_ranking_route_edges = route_ranking_registry.route_edge_indices
            route_ranking_k = int(route_ranking_registry.K)
            route_source = rr_path_for_ranking
        else:
            raise ValueError(
                "--route_registry or --route_ranking_routes_csv is required "
                "when route-ranking supervision is enabled"
            )
        if not route_ranking_route_edges:
            raise ValueError(f"No valid routes found for route-ranking supervision from {route_source}")
        if is_rank0():
            print(
                "Route ranking supervision: "
                f"decoder_weight={route_ranking_weight:.4f}, "
                f"transition_weight={transition_route_ranking_weight:.4f}, "
                f"source={route_source}, OD={len(route_ranking_route_edges)}, K={route_ranking_k}, "
                f"max_od={route_ranking_max_od}, temp={route_ranking_temperature:.3f}, "
                f"length_norm={route_ranking_length_norm}, target_cost={route_ranking_target_cost}, "
                f"non0_weight={route_ranking_non0_weight:.3f}, gap_weight={route_ranking_gap_weight:.3f}"
            )

    route_distill_loader = None
    route_distill_iter = None
    if route_distill_weight > 0.0:
        if not route_distill_pt:
            raise ValueError("--route_distill_pt is required when --route_distill_weight > 0")
        route_distill_obj = torch.load(route_distill_pt, map_location="cpu", weights_only=False)
        for key in ("flows", "path_edge_indices", "live_costs"):
            if key not in route_distill_obj:
                raise KeyError(f"route_distill_pt missing required key: {key}")
        rd_flows = torch.as_tensor(route_distill_obj["flows"], dtype=torch.float32)
        rd_paths = torch.as_tensor(route_distill_obj["path_edge_indices"], dtype=torch.long)
        rd_costs = torch.as_tensor(route_distill_obj["live_costs"], dtype=torch.float32)
        if rd_flows.dim() != 3 or rd_flows.shape[1] != n_spatial:
            raise ValueError(f"route_distill flows must have shape (N, {n_spatial}, F), got {tuple(rd_flows.shape)}")
        if rd_flows.shape[-1] != _d_feat:
            raise ValueError(f"route_distill flow feature dim mismatch: expected {_d_feat}, got {rd_flows.shape[-1]}")
        if rd_paths.dim() != 3 or rd_paths.shape[0] != rd_flows.shape[0] or rd_costs.shape[:2] != rd_paths.shape[:2]:
            raise ValueError(
                "route_distill shape mismatch: "
                f"flows={tuple(rd_flows.shape)}, paths={tuple(rd_paths.shape)}, costs={tuple(rd_costs.shape)}"
            )
        route_distill_loader = DataLoader(
            TensorDataset(rd_flows, rd_paths, rd_costs),
            batch_size=max(1, int(route_distill_batch_size)),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        route_distill_iter = iter(route_distill_loader)
        if is_rank0():
            print(
                "Route distill supervision: "
                f"weight={route_distill_weight:.4f}, source={route_distill_pt}, "
                f"N={int(rd_flows.shape[0])}, K={int(rd_paths.shape[1])}, max_path={int(rd_paths.shape[2])}, "
                f"batch={max(1, int(route_distill_batch_size))}, temp={route_distill_temperature:.3f}, "
                f"pred_cost={route_distill_pred_cost}, gap_weight={route_distill_gap_weight:.3f}, "
                f"cost_weight={route_distill_cost_weight:.3f}"
            )

    # --- Build Stackelberg layer (optional) ---
    stackelberg_layer: Optional[StackelbergLayer] = None
    if _use_stackelberg:
        rr_path = str(getattr(args, "route_registry", "") or "")
        total_av_count = float(getattr(args, "stackelberg_total_av_count", 0.0))
        total_hdv_count = float(getattr(args, "stackelberg_total_hdv_count", 0.0))
        if rr_path:
            route_reg = RouteRegistry.load(rr_path)
            if total_av_count <= 0.0:
                total_av_count = float(np.asarray(route_reg.od_av_counts, dtype=np.float32).sum())
            if total_hdv_count <= 0.0:
                total_hdv_count = float(np.asarray(route_reg.od_hdv_counts, dtype=np.float32).sum())
        if total_av_count <= 0.0 and total_hdv_count <= 0.0:
            raise ValueError(
                "Stackelberg requires total AV/HDV counts. "
                "Provide --route_registry or explicit --stackelberg_total_av_count/--stackelberg_total_hdv_count."
            )
        bpr_layer = BPRLayer(lengths_t, speed_limits_t, capacities_t).to(device)
        stackelberg_layer = StackelbergLayer(
            n_edges=n_spatial,
            bpr=bpr_layer,
            total_av_count=total_av_count,
            total_hdv_count=total_hdv_count,
            n_iters=int(getattr(args, "stackelberg_n_iters", 10)),
            damping=float(getattr(args, "stackelberg_damping", 0.5)),
        ).to(device)
        if is_rank0():
            stk_params = sum(p.numel() for p in stackelberg_layer.parameters() if p.requires_grad)
            print(
                f"StackelbergLayer: n_edges={n_spatial}, total_av={int(total_av_count)}, total_hdv={int(total_hdv_count)}, "
                f"n_iters={int(getattr(args, 'stackelberg_n_iters', 10))}, "
                f"learnable params={stk_params}"
            )

    if is_rank0():
        _decoder_full_weight = float(getattr(args, "decoder_full_weight", 1.0))
        _decoder_speed_weight = float(getattr(args, "decoder_speed_weight", 1.0))
        _decoder_volume_weight = float(getattr(args, "decoder_volume_weight", 1.5))
        _decoder_other_weight = float(getattr(args, "decoder_other_weight", 1.0))
        _decoder_head_supervision_enabled = (
            _decoder_full_weight > 0.0
            or _decoder_speed_weight > 0.0
            or _decoder_volume_weight > 0.0
            or _decoder_other_weight > 0.0
        )
        _transition_decoded_supervision_weight = float(
            getattr(args, "transition_decoded_supervision_weight", 0.0)
        )
        print(dyn)
        param_count = sum(p.numel() for p in dyn.parameters() if p.requires_grad)
        fp_trainable_count = sum(p.numel() for p in flow_projector.parameters() if p.requires_grad)
        print(f"Learnable parameters (dynamics): {param_count:,}")
        print(f"Learnable parameters (flow_projector): {fp_trainable_count:,}")
        print(
            "Action encoder: "
            f"type={action_encoder_type}, raw_action_dim={action_dim_flow}, "
            f"model_action_dim={int(getattr(args, 'action_dim', action_dim_flow))}, "
            f"compress={action_compress_mode}"
        )
        if action_compressor_state is not None:
            print(
                "Action compressor: "
                f"type={str(action_compressor_state.get('type', 'unknown'))}, "
                f"input_dim={int(action_compressor_state.get('input_dim', -1))}, "
                f"output_dim={int(action_compressor_state.get('output_dim', -1))}, "
                f"explained_var={float(action_compressor_state.get('explained_variance_ratio', 0.0)):.6f}"
            )
        if depart_compressor_state is not None:
            print(
                "Depart compressor: "
                f"type={str(depart_compressor_state.get('type', 'unknown'))}, "
                f"input_dim={int(depart_compressor_state.get('input_dim', -1))}, "
                f"output_dim={int(depart_compressor_state.get('output_dim', -1))}, "
                f"explained_var={float(depart_compressor_state.get('explained_variance_ratio', 0.0)):.6f}"
            )
        if float(getattr(args, "hdv_policy_weight", 0.0)) > 0.0:
            print(
                f"HDV policy head supervision: key={getattr(args, 'aux_action_key', '')!r}, "
                f"weight={float(args.hdv_policy_weight):.4f}, "
                f"mode=edge, "
                f"dim={int(args.hdv_policy_dim)}, "
                f"target_mode={str(getattr(args, 'hdv_policy_target_mode', 'step'))}, "
                f"action_mode={str(getattr(args, 'hdv_policy_action_mode', 'step'))}"
            )
        if str(getattr(args, "transition_hdv_action_mode", "none") or "none").strip().lower() not in {"", "none", "off", "disabled"}:
            print(
                "Transition HDV action composition: "
                f"mode={str(getattr(args, 'transition_hdv_action_mode', 'none'))}, "
                f"weight={float(getattr(args, 'transition_hdv_action_weight', 0.0)):.4f}, "
                f"scale={float(getattr(args, 'transition_hdv_action_scale', 1.0)):.4f}, "
                f"aux_key={getattr(args, 'aux_action_key', '')!r}"
            )
        if str(getattr(args, "transition_hdv_context_mode", "none") or "none").strip().lower() not in {"", "none", "off", "disabled"}:
            print(
                "Transition HDV separate context: "
                f"mode={str(getattr(args, 'transition_hdv_context_mode', 'none'))}, "
                f"weight={float(getattr(args, 'transition_hdv_context_weight', 0.0)):.4f}, "
                f"scale={float(getattr(args, 'transition_hdv_context_scale', 1.0)):.4f}, "
                f"aux_key={getattr(args, 'aux_action_key', '')!r}"
            )
        if decoder_recon_weight > 0.0:
            print(
                "Decoder reconstruction supervision: "
                f"weight={decoder_recon_weight:.4f}, "
                f"speed_idx={decoder_recon_speed_index}, "
                f"speed_weight={decoder_recon_speed_weight:.2f}, "
                f"busy_volume_weight={decoder_recon_busy_volume_weight:.2f}, "
                f"busy_presence_weight={decoder_recon_busy_presence_weight:.4f}, "
                f"topk_rank_weight={decoder_recon_topk_rank_weight:.4f}"
            )
        if float(getattr(args, "bpr_supervision_weight", 0.0)) > 0.0:
            print(
                "Decoder BPR supervision: "
                f"weight={float(args.bpr_supervision_weight):.4f}, "
                "edge_only=True, "
                f"alpha={float(getattr(args, 'bpr_alpha', 0.15)):.3f}, "
                f"beta={float(getattr(args, 'bpr_beta', 4.0)):.3f}, "
                f"veh_idx={int(getattr(args, 'bpr_veh_index', 2))}, "
                f"target_mode={str(getattr(args, 'bpr_target_mode', 'relative_excess'))!r}"
            )
        if float(getattr(args, "t_all_supervision_weight", 0.0)) > 0.0:
            print(
                "Episode t_all supervision: "
                f"weight={float(getattr(args, 't_all_supervision_weight', 0.0)):.4f}, "
                f"reward_key={str(getattr(args, 'reward_key', 'reward'))!r}, "
                f"missing_reward={float(getattr(args, 't_all_missing_reward', 0.0)):.4f}, "
                f"target_mode={str(getattr(args, 't_all_target_mode', 'relative'))!r}"
            )
        if _decoder_head_supervision_enabled:
            print(
                "Decoder multi-head supervision: "
                f"full_weight={_decoder_full_weight:.4f}, "
                f"speed_weight={_decoder_speed_weight:.4f}, "
                f"volume_weight={_decoder_volume_weight:.4f}, "
                f"other_weight={_decoder_other_weight:.4f}, "
                f"busy_volume_weight={decoder_busy_volume_weight:.4f}, "
                f"busy_presence_weight={decoder_busy_presence_weight:.4f}, "
                f"topk_rank_weight={decoder_topk_rank_weight:.4f}, "
                f"topk_rank_k={decoded_topk_rank_k}, "
                f"speed_target={str(getattr(args, 'decoded_speed_target', 'slowdown'))!r}, "
                f"volume_target={str(getattr(args, 'decoded_volume_target', 'log1p'))!r}, "
                f"busy_threshold={float(getattr(args, 'decoded_busy_threshold', 0.05)):.4f}, "
                f"busy_presence_margin={decoded_busy_presence_margin:.4f}, "
                f"topk_rank_margin={decoded_topk_rank_margin:.4f}"
            )
        if _transition_decoded_supervision_weight > 0.0:
            print(
                "Transition decoded supervision: "
                f"weight={_transition_decoded_supervision_weight:.4f}, "
                f"n_ode_steps={int(getattr(args, 'n_ode_steps', 1))}, "
                f"busy_volume_weight={transition_busy_volume_weight:.4f}, "
                f"busy_presence_weight={transition_busy_presence_weight:.4f}, "
                f"topk_rank_weight={transition_topk_rank_weight:.4f}, "
                f"topk_rank_k={transition_topk_rank_k}, "
                f"cfm_init={transition_cfm_init_mode!r}, "
                f"action_contrast_weight={float(getattr(args, 'transition_action_contrast_weight', 0.0)):.4f}, "
                f"speed_target={str(getattr(args, 'decoded_speed_target', 'slowdown'))!r}, "
                f"volume_target={str(getattr(args, 'decoded_volume_target', 'log1p'))!r}, "
                f"busy_presence_margin={decoded_busy_presence_margin:.4f}, "
                f"topk_rank_margin={transition_topk_rank_margin:.4f}, "
                f"zero_volume_weight={transition_zero_volume_weight:.4f}, "
                f"zero_volume_threshold={decoded_zero_volume_threshold:.4f}, "
                f"zero_volume_margin={decoded_zero_volume_margin:.4f}, "
                f"action_contrast_margin={float(getattr(args, 'transition_action_contrast_margin', 0.05)):.4f}, "
                f"action_contrast_metric={str(getattr(args, 'transition_action_contrast_metric', 'total'))}"
            )
        print(
            "CFM velocity edge weighting: "
            f"active={cfm_active_weight:.4f}, "
            f"busy={cfm_busy_weight:.4f}, "
            f"active_threshold={cfm_active_threshold:.4f}, "
            f"busy_threshold={cfm_busy_threshold:.4f}, "
            f"max_weight={cfm_max_weight:.2f}"
        )
        if adm_cfm_horizons and (adm_cfm_weight > 0.0 or adm_decoded_supervision_weight > 0.0):
            print(
                "ADM-v2 direct multi-step supervision: "
                f"horizons={','.join(str(h) for h in adm_cfm_horizons)}, "
                f"cfm_weight={adm_cfm_weight:.4f}, "
                f"decoded_weight={adm_decoded_supervision_weight:.4f}, "
                f"max_windows={adm_max_windows}, "
                f"gamma={adm_cfm_gamma:.4f}, "
                f"cfm_init={adm_cfm_init_mode!r}"
            )
        if bool(getattr(args, "edge_action_injection", False)):
            print(
                "Per-edge action injection: "
                f"eps={float(getattr(args, 'edge_action_injection_eps', 1e-6)):.2e}, "
                f"scale={float(getattr(args, 'edge_action_injection_scale', 1.0)):.2f}, "
                f"value_norm={str(getattr(args, 'edge_action_value_norm', 'raw'))}, "
                f"edge_id_embedding={bool(getattr(args, 'edge_action_id_embedding', False))}"
            )
        if bool(getattr(args, "edge_action_tokenizer", False)):
            print(
                "Dreamer-style edge action tokenizer: "
                f"value_norm={str(getattr(args, 'edge_action_value_norm', 'raw'))}, "
                f"recon_weight={float(getattr(args, 'action_reconstruction_weight', 0.0)):.4f}, "
                f"local_latent_weight={float(getattr(args, 'action_local_latent_weight', 0.0)):.4f}, "
                f"volume_delta_weight={float(getattr(args, 'action_volume_delta_weight', 0.0)):.4f}, "
                f"fuse_encoder={bool(getattr(args, 'edge_action_tokenizer_fuse_encoder', False))}, "
                f"output_injection={bool(getattr(args, 'edge_action_output_injection', False))}"
            )
        if _decoder_head_supervision_enabled and _transition_decoded_supervision_weight > 0.0:
            print("Combined ablation active: decoder multi-head + transition decoded supervision")

    if args.compile:
        dyn = torch.compile(dyn)

    if ddp:
        dyn = torch.nn.parallel.DistributedDataParallel(
            dyn, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )

    if ddp and not freeze_flow_projector:
        flow_projector = torch.nn.parallel.DistributedDataParallel(
            flow_projector, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )

    # Optimizer and scaler
    _opt_params = [p for p in dyn.parameters() if p.requires_grad]
    if not freeze_flow_projector:
        _opt_params += [p for p in flow_projector.parameters() if p.requires_grad]
    if stackelberg_layer is not None:
        _opt_params += [p for p in stackelberg_layer.parameters() if p.requires_grad]
    if not _opt_params:
        raise RuntimeError("No trainable parameters remain after freeze configuration.")
    opt = torch.optim.AdamW(
        _opt_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda", enabled=use_amp)

    # Initialize wandb
    if is_rank0():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            mode=str(getattr(args, "wandb_mode", "online")),
            config=vars(args),
        )

    # Resume from checkpoint
    step = 0
    start_epoch = 0
    ckpt_dir = Path(args.ckpt_dir)
    if args.resume is not None:
        step, start_epoch = load_ckpt(
            Path(args.resume),
            dyn_model=dyn,
            opt=opt,
            scaler=scaler,
            flow_projector=flow_projector,
            strict=(not bool(getattr(args, "resume_non_strict", False))),
            load_optimizer=(not bool(getattr(args, "resume_weights_only", False))),
            load_scaler=(not bool(getattr(args, "resume_weights_only", False))),
            weights_only=bool(getattr(args, "resume_weights_only", False)),
        )
        if is_rank0():
            mode = "weights-only" if bool(getattr(args, "resume_weights_only", False)) else "full-state"
            strict_mode = "strict" if (not bool(getattr(args, "resume_non_strict", False))) else "non-strict"
            print(
                f"[rank0] Resumed from {args.resume} "
                f"(mode={mode}, {strict_mode}, step={step}, epoch={start_epoch})"
            )

    # Training loop
    dyn.train()
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))
    epoch = start_epoch
    use_explicit_ddp_barrier = bool(getattr(args, "ddp_barrier", False))
    stackelberg_hdv_weight = float(getattr(args, "stackelberg_hdv_weight", 0.0))
    stackelberg_hdv_weight_decay = float(getattr(args, "stackelberg_hdv_weight_decay", 0.99))
    stackelberg_eq_weight = float(getattr(args, "stackelberg_eq_weight", 0.0))
    obs_proj_weight = float(getattr(args, "obs_proj_weight", 0.2)) if d_obs > 0 else 0.0
    decoded_speed_target = str(getattr(args, "decoded_speed_target", "slowdown")).strip().lower()
    decoded_volume_target = str(getattr(args, "decoded_volume_target", "log1p")).strip().lower()
    decoded_busy_threshold = float(getattr(args, "decoded_busy_threshold", 0.05))
    bpr_supervision_weight = float(getattr(args, "bpr_supervision_weight", 0.0))
    spatial_supervision_weight = float(getattr(args, "spatial_supervision_weight", 0.0))
    transition_decoded_supervision_weight = float(getattr(args, "transition_decoded_supervision_weight", 0.0))
    rollout_horizon = int(getattr(args, "rollout_horizon", 1))
    rollout_gamma = float(getattr(args, "rollout_gamma", 0.7))
    decoder_full_weight = float(getattr(args, "decoder_full_weight", 1.0))
    decoder_speed_weight = float(getattr(args, "decoder_speed_weight", 1.0))
    decoder_volume_weight = float(getattr(args, "decoder_volume_weight", 1.5))
    decoder_other_weight = float(getattr(args, "decoder_other_weight", 1.0))
    decoder_head_supervision_enabled = (
        decoder_full_weight > 0.0
        or decoder_speed_weight > 0.0
        or decoder_volume_weight > 0.0
        or decoder_other_weight > 0.0
    )
    bpr_reward_weight = float(getattr(args, "bpr_reward_weight", 1.0))
    bpr_alpha = float(getattr(args, "bpr_alpha", 0.15))
    bpr_beta = float(getattr(args, "bpr_beta", 4.0))
    bpr_veh_index = int(getattr(args, "bpr_veh_index", 2))
    bpr_target_mode = str(getattr(args, "bpr_target_mode", "relative_excess"))
    t_all_supervision_weight = float(getattr(args, "t_all_supervision_weight", 0.0))
    t_all_missing_reward = float(getattr(args, "t_all_missing_reward", 0.0))
    t_all_target_mode = str(getattr(args, "t_all_target_mode", "relative"))
    transition_action_contrast_weight = float(getattr(args, "transition_action_contrast_weight", 0.0))
    transition_action_contrast_margin = float(getattr(args, "transition_action_contrast_margin", 0.05))
    transition_action_contrast_metric_name = str(
        getattr(args, "transition_action_contrast_metric", "total")
    ).strip().lower()
    action_reconstruction_weight = float(getattr(args, "action_reconstruction_weight", 0.0))
    action_reconstruction_eps = float(getattr(args, "action_reconstruction_eps", 1e-6))
    action_local_latent_weight = float(getattr(args, "action_local_latent_weight", 0.0))
    action_local_latent_eps = float(getattr(args, "action_local_latent_eps", 1e-6))
    action_volume_delta_weight = float(getattr(args, "action_volume_delta_weight", 0.0))
    action_volume_delta_eps = float(getattr(args, "action_volume_delta_eps", action_local_latent_eps))

    while step < args.max_steps:
        for epoch in range(start_epoch, 10_000_000):
            if sampler is not None:
                sampler.set_epoch(epoch)

            for batch in loader:
                if step >= args.max_steps:
                    break

                # Flow mode: project flow tokens. In A12/freeze mode the projector defines
                # a fixed latent space; decoded transition losses still backprop through the
                # frozen decoder input into the CFM transition, not into projector weights.
                flows_b = batch[0].to(device, non_blocking=True)   # (B, T, E, d_feat) — from FlowDataset tuple
                decoded_dyn = None
                z_pred_next = None
                need_decoded = (
                    (decoder_recon_weight > 0.0)
                    or (bpr_supervision_weight > 0.0)
                    or (route_ranking_weight > 0.0)
                    or decoder_head_supervision_enabled
                )
                if need_decoded:
                    z1, decoded_dyn = flow_projector(flows_b, return_decoded=True)  # (B, T, E, 64), (B, T, E, 4)
                else:
                    z1 = flow_projector(flows_b)                    # (B, T, E, 64)
                act_raw = batch[1].to(device, non_blocking=True)    # (B, T, action_dim)
                # Unpack variable-length batch: (flows, act, [aux_act], [reward], [depart], [context])
                hdv_target = None
                reward_target = None
                depart_b = None
                context_b = None
                episode_act_b = None
                bi = 2  # next batch index
                if bi < len(batch):
                    b2 = batch[bi].to(device, non_blocking=True)
                    if b2.dim() >= 3:
                        hdv_target = b2   # (B, T, K) or (B, T, E)
                    else:
                        reward_target = b2  # (B, T)
                    bi += 1
                if bi < len(batch) and reward_target is None:
                    b3 = batch[bi].to(device, non_blocking=True)
                    if b3.dim() == 2:  # (B, T)
                        reward_target = b3
                    bi += 1
                batch_end = len(batch)
                if _return_episode_action_mean and batch_end > bi:
                    maybe_episode_act = batch[batch_end - 1].to(device, non_blocking=True)
                    if maybe_episode_act.dim() >= 3 and maybe_episode_act.shape[-1] == act_raw.shape[-1]:
                        episode_act_b = maybe_episode_act
                        batch_end -= 1
                # Depart and context are always the last two if present
                if bi < batch_end:
                    b_dep = batch[bi].to(device, non_blocking=True)
                    if b_dep.dim() >= 3 and b_dep.shape[-1] == n_spatial:
                        depart_b = b_dep  # (B, T, E)
                        bi += 1
                if bi < batch_end:
                    b_ctx = batch[bi].to(device, non_blocking=True)
                    if b_ctx.dim() >= 2:
                        context_b = b_ctx  # (B, T, C)
                        bi += 1

                hdv_policy_target_mode = str(getattr(args, "hdv_policy_target_mode", "step") or "step").strip().lower()
                if hdv_target is not None and hdv_policy_target_mode in {"window_mean"}:
                    hdv_target = hdv_target.mean(dim=1, keepdim=True).expand_as(hdv_target)

                route_distill_batch = None
                if route_distill_loader is not None:
                    try:
                        route_distill_batch = next(route_distill_iter)
                    except StopIteration:
                        route_distill_iter = iter(route_distill_loader)
                        route_distill_batch = next(route_distill_iter)

                act_for_transition = _compose_transition_hdv_action(
                    act_raw=act_raw,
                    hdv_target=hdv_target,
                    mode=str(getattr(args, "transition_hdv_action_mode", "none")),
                    weight=float(getattr(args, "transition_hdv_action_weight", 0.0)),
                    scale=float(getattr(args, "transition_hdv_action_scale", 1.0)),
                )
                hdv_context_for_transition = _build_transition_hdv_context(
                    act_raw=act_raw,
                    hdv_target=hdv_target,
                    mode=str(getattr(args, "transition_hdv_context_mode", "none")),
                    weight=float(getattr(args, "transition_hdv_context_weight", 0.0)),
                    scale=float(getattr(args, "transition_hdv_context_scale", 1.0)),
                )

                if action_encoder_type == "graph_cond":
                    actions = _build_graph_cond_model_actions(
                        act_raw=act_for_transition,
                        depart_b=depart_b,
                        context_b=context_b,
                        hdv_context_b=hdv_context_for_transition,
                    )
                else:
                    actions = _build_model_actions(
                        act_raw=act_for_transition,
                        depart_b=depart_b if model_action_uses_depart_context else None,
                        context_b=context_b if model_action_uses_depart_context else None,
                        action_compressor=action_compressor_state,
                        depart_compressor=depart_compressor_state if model_action_uses_depart_context else None,
                    )
                act_mask = None

                if actions is not None and not isinstance(actions, tuple):
                    actions = actions.to(device, non_blocking=True)
                hdv_policy_action_mode = str(
                    getattr(args, "hdv_policy_action_mode", "step") or "step"
                ).strip().lower()
                hdv_actions = None
                if hdv_target is not None and float(getattr(args, "hdv_policy_weight", 0.0)) > 0.0:
                    hdv_actions = _build_hdv_policy_condition_actions(
                        mode=hdv_policy_action_mode,
                        action_encoder_type=action_encoder_type,
                        act_raw=act_raw,
                        episode_act_raw=episode_act_b,
                        depart_b=depart_b,
                        context_b=context_b,
                        model_action_uses_depart_context=model_action_uses_depart_context,
                        action_compressor=action_compressor_state,
                        depart_compressor=depart_compressor_state,
                    )
                    if hdv_actions is not None and not isinstance(hdv_actions, tuple):
                        hdv_actions = hdv_actions.to(device, non_blocking=True)

                with autocast(device_type="cuda", enabled=use_amp):
                    # Get flow_projector head for Stackelberg bg_volume decoding
                    _fp_for_stk = (
                        (flow_projector.module if hasattr(flow_projector, "module") else flow_projector)
                        if stackelberg_layer is not None else None
                    )
                    if action_encoder_type == "graph_cond":
                        base_loss, aux = cfm_loss(
                            dyn.module if hasattr(dyn, "module") else dyn,
                            z1=z1,
                            actions=actions,
                            act_mask=act_mask,
                            hdv_actions=hdv_actions,
                            agent_tokens=None,
                            hdv_target=hdv_target,
                            hdv_policy_weight=float(getattr(args, "hdv_policy_weight", 0.0)),
                            stackelberg=stackelberg_layer,
                            flow_projector=_fp_for_stk,
                            eq_weight=stackelberg_eq_weight,
                            stackelberg_hdv_weight=stackelberg_hdv_weight,
                            action_compressor=action_compressor_state,
                            av_action_dim=(
                                int(action_compressor_state["output_dim"])
                                if action_compressor_state is not None else int(action_dim_flow)
                            ),
                            action_reconstruction_weight=action_reconstruction_weight,
                            action_reconstruction_eps=action_reconstruction_eps,
                            flow_targets=flows_b,
                            cfm_active_weight=cfm_active_weight,
                            cfm_busy_weight=cfm_busy_weight,
                            cfm_busy_threshold=cfm_busy_threshold,
                            cfm_active_threshold=cfm_active_threshold,
                            cfm_max_weight=cfm_max_weight,
                            volume_index=bpr_veh_index,
                        )
                    else:
                        rollout_targets = _build_rollout_targets(z1, horizon=rollout_horizon)
                        rollout_flow_targets = _build_rollout_targets(flows_b, horizon=rollout_horizon)
                        rollout_inputs = _build_rollout_inputs(act_raw, depart_b, context_b, horizon=rollout_horizon)
                        base_loss = torch.zeros((), device=device, dtype=torch.float32)
                        aux = {}
                        sched_prob = _scheduled_sampling_prob(
                            step=step,
                            enabled=bool(getattr(args, "scheduled_sampling", False)),
                            warmup=int(getattr(args, "sched_sample_warmup", 3000)),
                            end=float(getattr(args, "sched_sample_end", 0.5)),
                        )
                        for h, target_h in enumerate(rollout_targets):
                            inp_h = rollout_inputs[h]
                            flow_target_h = rollout_flow_targets[h]
                            hdv_target_h_for_transition = (
                                None if hdv_target is None
                                else hdv_target[:, h:h + inp_h["act"].shape[1]]
                            )
                            hdv_context_h_for_transition = _build_transition_hdv_context(
                                act_raw=inp_h["act"],
                                hdv_target=hdv_target_h_for_transition,
                                mode=str(getattr(args, "transition_hdv_context_mode", "none")),
                                weight=float(getattr(args, "transition_hdv_context_weight", 0.0)),
                                scale=float(getattr(args, "transition_hdv_context_scale", 1.0)),
                            )
                            actions_h = _build_model_actions(
                                act_raw=_compose_transition_hdv_action(
                                    act_raw=inp_h["act"],
                                    hdv_target=hdv_target_h_for_transition,
                                    mode=str(getattr(args, "transition_hdv_action_mode", "none")),
                                    weight=float(getattr(args, "transition_hdv_action_weight", 0.0)),
                                    scale=float(getattr(args, "transition_hdv_action_scale", 1.0)),
                                ),
                                depart_b=inp_h["depart"] if model_action_uses_depart_context else None,
                                context_b=inp_h["context"] if model_action_uses_depart_context else None,
                                action_compressor=action_compressor_state,
                                depart_compressor=depart_compressor_state if model_action_uses_depart_context else None,
                            ).to(device, non_blocking=True)
                            if action_encoder_type == "graph_cond":
                                actions_h = _build_graph_cond_model_actions(
                                    act_raw=_compose_transition_hdv_action(
                                        act_raw=inp_h["act"],
                                        hdv_target=hdv_target_h_for_transition,
                                        mode=str(getattr(args, "transition_hdv_action_mode", "none")),
                                        weight=float(getattr(args, "transition_hdv_action_weight", 0.0)),
                                        scale=float(getattr(args, "transition_hdv_action_scale", 1.0)),
                                    ),
                                    depart_b=inp_h["depart"],
                                    context_b=inp_h["context"],
                                    hdv_context_b=hdv_context_h_for_transition,
                                )
                            hdv_actions_h = None
                            if hdv_target is not None and float(getattr(args, "hdv_policy_weight", 0.0)) > 0.0:
                                hdv_actions_h = _build_hdv_policy_condition_actions(
                                    mode=hdv_policy_action_mode,
                                    action_encoder_type=action_encoder_type,
                                    act_raw=inp_h["act"],
                                    episode_act_raw=(
                                        None if episode_act_b is None
                                        else episode_act_b[:, h:h + inp_h["act"].shape[1]]
                                    ),
                                    depart_b=inp_h["depart"],
                                    context_b=inp_h["context"],
                                    model_action_uses_depart_context=model_action_uses_depart_context,
                                    action_compressor=action_compressor_state,
                                    depart_compressor=depart_compressor_state,
                                )
                                if hdv_actions_h is not None and not isinstance(hdv_actions_h, tuple):
                                    hdv_actions_h = hdv_actions_h.to(device, non_blocking=True)
                            hdv_target_h = None if hdv_target is None else hdv_target[:, h + 1:h + 1 + target_h.shape[1]]
                            target_for_loss = target_h
                            if h > 0 and sched_prob > 0.0:
                                z_pred_h = cfm_predict_one_timestep(
                                    dyn.module if hasattr(dyn, "module") else dyn,
                                    past_packed=z1[:, h - 1:h],
                                    actions=actions_h,
                                    act_mask=act_mask,
                                    n_steps=int(getattr(args, "n_ode_steps", 1)),
                                )
                                use_pred = torch.rand(z_pred_h.shape[0], device=device) < sched_prob
                                target_for_loss = _select_rollout_state(
                                    z_gt=target_h[:, 0],
                                    z_pred=z_pred_h,
                                    use_pred_mask=use_pred,
                                ).unsqueeze(1).expand_as(target_h)
                            loss_h, aux_h = cfm_loss(
                                dyn.module if hasattr(dyn, "module") else dyn,
                                z1=target_for_loss,
                                actions=actions_h,
                                act_mask=act_mask,
                                hdv_actions=hdv_actions_h,
                                agent_tokens=None,
                                hdv_target=hdv_target_h,
                                hdv_policy_weight=float(getattr(args, "hdv_policy_weight", 0.0)),
                                stackelberg=stackelberg_layer,
                                flow_projector=_fp_for_stk,
                                eq_weight=stackelberg_eq_weight,
                                stackelberg_hdv_weight=stackelberg_hdv_weight,
                                action_compressor=action_compressor_state,
                                av_action_dim=(
                                    int(action_compressor_state["output_dim"])
                                    if action_compressor_state is not None else int(action_dim_flow)
                                ),
                                action_reconstruction_weight=action_reconstruction_weight,
                                action_reconstruction_eps=action_reconstruction_eps,
                                flow_targets=flow_target_h,
                                cfm_active_weight=cfm_active_weight,
                                cfm_busy_weight=cfm_busy_weight,
                                cfm_busy_threshold=cfm_busy_threshold,
                                cfm_active_threshold=cfm_active_threshold,
                                cfm_max_weight=cfm_max_weight,
                                volume_index=bpr_veh_index,
                            )
                            base_loss = base_loss + (float(rollout_gamma) ** h) * loss_h
                            for k, v in aux_h.items():
                                aux[k] = v
                        aux["scheduled_sampling_prob"] = torch.tensor(float(sched_prob), device=device)
                    decoder_recon_loss, decoder_recon_aux = decoder_reconstruction_loss(
                        decoded_dyn=decoded_dyn,
                        flow_targets=flows_b,
                        recon_weight=decoder_recon_weight,
                        speed_index=decoder_recon_speed_index,
                        speed_weight=decoder_recon_speed_weight,
                        volume_index=bpr_veh_index,
                        speed_target_mode=decoded_speed_target,
                        volume_target_mode=decoded_volume_target,
                        capacities=capacities_t,
                        busy_threshold=decoded_busy_threshold,
                        busy_volume_weight=decoder_recon_busy_volume_weight,
                        busy_presence_weight=decoder_recon_busy_presence_weight,
                        busy_presence_margin=decoded_busy_presence_margin,
                        topk_rank_weight=decoder_recon_topk_rank_weight,
                        topk_rank_k=decoded_topk_rank_k,
                        topk_rank_margin=decoded_topk_rank_margin,
                        zero_volume_weight=decoder_recon_zero_volume_weight,
                        zero_volume_threshold=decoded_zero_volume_threshold,
                        zero_volume_margin=decoded_zero_volume_margin,
                        zero_target_threshold=decoded_zero_target_threshold,
                    )
                    bpr_sup_loss, bpr_aux = decoder_bpr_supervision_loss(
                        decoded_dyn=decoded_dyn,
                        flow_targets=flows_b,
                        lengths=lengths_t,
                        speed_limits=speed_limits_t,
                        capacities=capacities_t,
                        bpr_weight=bpr_supervision_weight,
                        bpr_reward_weight=bpr_reward_weight,
                        bpr_alpha=bpr_alpha,
                        bpr_beta=bpr_beta,
                        veh_index=bpr_veh_index,
                        target_mode=bpr_target_mode,
                    )
                    route_rank_loss, route_rank_aux = route_ranking_supervision_loss(
                        decoded_dyn=decoded_dyn,
                        flow_targets=flows_b,
                        route_edge_indices=route_ranking_route_edges,
                        lengths=lengths_t,
                        speed_limits=speed_limits_t,
                        capacities=capacities_t,
                        loss_weight=route_ranking_weight,
                        n_routes=int(getattr(args, "n_routes", 4)),
                        max_od_samples=route_ranking_max_od,
                        temperature=route_ranking_temperature,
                        margin_weight=route_ranking_margin_weight,
                        margin=route_ranking_margin,
                        gap_weight=route_ranking_gap_weight,
                        length_norm=route_ranking_length_norm,
                        target_cost_mode=route_ranking_target_cost,
                        non0_weight=route_ranking_non0_weight,
                        speed_min_ratio=route_ranking_speed_min_ratio,
                        speed_max_ratio=route_ranking_speed_max_ratio,
                        speed_excess_scale=route_ranking_speed_excess_scale,
                        bpr_alpha=bpr_alpha,
                        bpr_beta=bpr_beta,
                        volume_index=bpr_veh_index,
                        speed_index=decoder_recon_speed_index,
                    )
                    route_distill_loss = torch.zeros((), device=device, dtype=torch.float32)
                    route_distill_aux = {}
                    if route_distill_batch is not None:
                        rd_flows, rd_paths, rd_live_costs = route_distill_batch
                        rd_flows = rd_flows.to(device, non_blocking=True).unsqueeze(1)
                        rd_paths = rd_paths.to(device, non_blocking=True)
                        rd_live_costs = rd_live_costs.to(device, non_blocking=True)
                        _rd_z, rd_decoded_dyn = flow_projector(rd_flows, return_decoded=True)
                        rd_route_edge_costs = None
                        if route_distill_pred_cost == "route_head":
                            rd_free_flow = lengths_t / speed_limits_t.clamp_min(0.1)
                            rd_route_edge_costs = flow_projector.predict_edge_route_cost(_rd_z, rd_free_flow)
                        route_distill_loss, route_distill_aux = route_distill_supervision_loss(
                            decoded_dyn=rd_decoded_dyn,
                            route_edge_costs=rd_route_edge_costs,
                            path_edge_indices=rd_paths,
                            live_costs=rd_live_costs,
                            lengths=lengths_t,
                            speed_limits=speed_limits_t,
                            capacities=capacities_t,
                            loss_weight=route_distill_weight,
                            temperature=route_distill_temperature,
                            gap_weight=route_distill_gap_weight,
                            cost_weight=route_distill_cost_weight,
                            length_norm=route_distill_length_norm,
                            pred_cost_mode=route_distill_pred_cost,
                            non0_weight=route_distill_non0_weight,
                            speed_min_ratio=route_ranking_speed_min_ratio,
                            speed_max_ratio=route_ranking_speed_max_ratio,
                            speed_excess_scale=route_ranking_speed_excess_scale,
                            bpr_alpha=bpr_alpha,
                            bpr_beta=bpr_beta,
                            volume_index=bpr_veh_index,
                            speed_index=decoder_recon_speed_index,
                        )
                    spatial_sup_loss, spatial_aux = one_step_spatial_loss(
                        flow_targets=flows_b,
                        spatial_weight=spatial_supervision_weight,
                        decoded_dyn=decoded_dyn,
                        veh_index=bpr_veh_index,
                        busy_threshold=float(getattr(args, "spatial_busy_threshold", 0.05)),
                    )
                    transition_sup_loss = torch.zeros((), device=device, dtype=torch.float32)
                    transition_route_rank_loss = torch.zeros((), device=device, dtype=torch.float32)
                    transition_route_rank_aux = {}
                    transition_aux = {}
                    transition_action_contrast_loss = torch.zeros((), device=device, dtype=torch.float32)
                    action_local_latent_loss = torch.zeros((), device=device, dtype=torch.float32)
                    action_local_latent_aux = {}
                    action_volume_delta_sup_loss = torch.zeros((), device=device, dtype=torch.float32)
                    action_volume_delta_aux = {}
                    adm_multistep_sup_loss = torch.zeros((), device=device, dtype=torch.float32)
                    adm_multistep_aux = {}
                    decoder_head_loss = torch.zeros((), device=device, dtype=torch.float32)
                    decoder_head_aux = {}
                    if decoded_dyn is not None:
                        target_decoded = flows_b[..., :decoded_dyn.shape[-1]].to(dtype=decoded_dyn.dtype)
                        decoder_head_loss, decoder_head_aux = multihead_decoder_supervision_loss(
                            pred_decoded=decoded_dyn,
                            target_decoded=target_decoded,
                            full_weight=decoder_full_weight,
                            speed_weight=decoder_speed_weight,
                            volume_weight=decoder_volume_weight,
                            other_weight=decoder_other_weight,
                            speed_index=decoder_recon_speed_index,
                            volume_index=bpr_veh_index,
                            speed_target_mode=decoded_speed_target,
                            volume_target_mode=decoded_volume_target,
                            capacities=capacities_t,
                            busy_threshold=decoded_busy_threshold,
                            busy_volume_weight=decoder_busy_volume_weight,
                            busy_presence_weight=decoder_busy_presence_weight,
                            busy_presence_margin=decoded_busy_presence_margin,
                            topk_rank_weight=decoder_topk_rank_weight,
                            topk_rank_k=decoded_topk_rank_k,
                            topk_rank_margin=decoded_topk_rank_margin,
                            zero_volume_weight=decoder_zero_volume_weight,
                            zero_volume_threshold=decoded_zero_volume_threshold,
                            zero_volume_margin=decoded_zero_volume_margin,
                            zero_target_threshold=decoded_zero_target_threshold,
                        )
                    need_cfm_next_supervision = (
                        transition_decoded_supervision_weight > 0.0
                        or action_local_latent_weight > 0.0
                        or action_volume_delta_weight > 0.0
                        or transition_action_contrast_weight > 0.0
                        or transition_route_ranking_weight > 0.0
                    )
                    if need_cfm_next_supervision and flows_b.shape[1] >= 2:
                        transition_initial_state = None
                        if transition_cfm_init_mode == "zero":
                            transition_initial_state = torch.zeros(
                                (z1.shape[0], 1, z1.shape[2], z1.shape[3]),
                                device=z1.device,
                                dtype=z1.dtype,
                            )
                        elif transition_action_contrast_weight > 0.0:
                            transition_initial_state = torch.randn(
                                (z1.shape[0], 1, z1.shape[2], z1.shape[3]),
                                device=z1.device,
                                dtype=z1.dtype,
                            )
                        z_pred_next = cfm_predict_one_timestep(
                            dyn.module if hasattr(dyn, "module") else dyn,
                            past_packed=z1[:, :-1],
                            actions=actions,
                            act_mask=act_mask,
                            n_steps=int(getattr(args, "n_ode_steps", 1)),
                            initial_state=transition_initial_state,
                        )
                        if transition_decoded_supervision_weight > 0.0:
                            transition_sup_loss, transition_aux = transition_decoded_supervision_loss(
                                z_pred_next=z_pred_next,
                                flow_projector=flow_projector.module if hasattr(flow_projector, "module") else flow_projector,
                                flow_target_next=flows_b[:, -1],
                                supervision_weight=transition_decoded_supervision_weight,
                                speed_index=decoder_recon_speed_index,
                                volume_index=bpr_veh_index,
                                speed_weight=decoder_recon_speed_weight,
                                volume_weight=decoder_volume_weight,
                                speed_target_mode=decoded_speed_target,
                                volume_target_mode=decoded_volume_target,
                                capacities=capacities_t,
                                busy_threshold=decoded_busy_threshold,
                                busy_volume_weight=transition_busy_volume_weight,
                                busy_presence_weight=transition_busy_presence_weight,
                                busy_presence_margin=decoded_busy_presence_margin,
                                topk_rank_weight=transition_topk_rank_weight,
                                topk_rank_k=transition_topk_rank_k,
                                topk_rank_margin=transition_topk_rank_margin,
                                zero_volume_weight=transition_zero_volume_weight,
                                zero_volume_threshold=decoded_zero_volume_threshold,
                                zero_volume_margin=decoded_zero_volume_margin,
                                zero_target_threshold=decoded_zero_target_threshold,
                        )
                        if transition_route_ranking_weight > 0.0:
                            fp_for_route_rank = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
                            decode_fn = getattr(fp_for_route_rank, "decode", None)
                            if decode_fn is None:
                                decode_fn = fp_for_route_rank.inverse
                            transition_decoded_dyn = decode_fn(z_pred_next).unsqueeze(1)
                            transition_route_rank_loss, transition_route_rank_aux = route_ranking_supervision_loss(
                                decoded_dyn=transition_decoded_dyn,
                                flow_targets=flows_b[:, -1:].to(dtype=transition_decoded_dyn.dtype),
                                route_edge_indices=route_ranking_route_edges,
                                lengths=lengths_t,
                                speed_limits=speed_limits_t,
                                capacities=capacities_t,
                                loss_weight=transition_route_ranking_weight,
                                n_routes=int(getattr(args, "n_routes", 4)),
                                max_od_samples=route_ranking_max_od,
                                temperature=route_ranking_temperature,
                                margin_weight=route_ranking_margin_weight,
                                margin=route_ranking_margin,
                                gap_weight=route_ranking_gap_weight,
                                length_norm=route_ranking_length_norm,
                                target_cost_mode=route_ranking_target_cost,
                                non0_weight=route_ranking_non0_weight,
                                speed_min_ratio=route_ranking_speed_min_ratio,
                                speed_max_ratio=route_ranking_speed_max_ratio,
                                speed_excess_scale=route_ranking_speed_excess_scale,
                                bpr_alpha=bpr_alpha,
                                bpr_beta=bpr_beta,
                                volume_index=bpr_veh_index,
                                speed_index=decoder_recon_speed_index,
                            )
                        if transition_action_contrast_weight > 0.0:
                            if (
                                transition_action_contrast_metric_name == "total"
                                and transition_decoded_supervision_weight > 0.0
                            ):
                                real_sup_unscaled = transition_sup_loss / max(transition_decoded_supervision_weight, 1e-8)
                            elif transition_action_contrast_metric_name == "total":
                                real_sup_unscaled, contrast_true_aux = transition_decoded_supervision_loss(
                                    z_pred_next=z_pred_next,
                                    flow_projector=flow_projector.module if hasattr(flow_projector, "module") else flow_projector,
                                    flow_target_next=flows_b[:, -1],
                                    supervision_weight=1.0,
                                    speed_index=decoder_recon_speed_index,
                                    volume_index=bpr_veh_index,
                                    speed_weight=decoder_recon_speed_weight,
                                    volume_weight=decoder_volume_weight,
                                    speed_target_mode=decoded_speed_target,
                                    volume_target_mode=decoded_volume_target,
                                    capacities=capacities_t,
                                    busy_threshold=decoded_busy_threshold,
                                    busy_volume_weight=transition_busy_volume_weight,
                                    busy_presence_weight=transition_busy_presence_weight,
                                    busy_presence_margin=decoded_busy_presence_margin,
                                    topk_rank_weight=transition_topk_rank_weight,
                                    topk_rank_k=transition_topk_rank_k,
                                    topk_rank_margin=transition_topk_rank_margin,
                                    zero_volume_weight=transition_zero_volume_weight,
                                    zero_volume_threshold=decoded_zero_volume_threshold,
                                    zero_volume_margin=decoded_zero_volume_margin,
                                    zero_target_threshold=decoded_zero_target_threshold,
                                )
                                if "pred_next_decoded_total_mse" in contrast_true_aux:
                                    transition_aux["pred_next_contrast_true_total_mse"] = contrast_true_aux[
                                        "pred_next_decoded_total_mse"
                                    ]
                            else:
                                real_sup_unscaled, contrast_true_aux = transition_action_contrast_metric_loss(
                                    z_pred_next=z_pred_next,
                                    flow_projector=flow_projector.module if hasattr(flow_projector, "module") else flow_projector,
                                    flow_target_next=flows_b[:, -1],
                                    metric=transition_action_contrast_metric_name,
                                    speed_index=decoder_recon_speed_index,
                                    volume_index=bpr_veh_index,
                                    speed_target_mode=decoded_speed_target,
                                    volume_target_mode=decoded_volume_target,
                                    capacities=capacities_t,
                                    busy_threshold=decoded_busy_threshold,
                                )
                                transition_aux["pred_next_action_contrast_true_metric"] = real_sup_unscaled.detach()
                                transition_aux.update(contrast_true_aux)
                            actions_shuffled = _shuffle_model_actions(actions)
                            z_pred_next_shuffled = cfm_predict_one_timestep(
                                dyn.module if hasattr(dyn, "module") else dyn,
                                past_packed=z1[:, :-1],
                                actions=actions_shuffled,
                                act_mask=act_mask,
                                n_steps=int(getattr(args, "n_ode_steps", 1)),
                                initial_state=transition_initial_state,
                            )
                            if transition_action_contrast_metric_name == "total":
                                shuffle_sup_loss, shuffle_aux = transition_decoded_supervision_loss(
                                    z_pred_next=z_pred_next_shuffled,
                                    flow_projector=flow_projector.module if hasattr(flow_projector, "module") else flow_projector,
                                    flow_target_next=flows_b[:, -1],
                                    supervision_weight=1.0,
                                    speed_index=decoder_recon_speed_index,
                                    volume_index=bpr_veh_index,
                                    speed_weight=decoder_recon_speed_weight,
                                    volume_weight=decoder_volume_weight,
                                    speed_target_mode=decoded_speed_target,
                                    volume_target_mode=decoded_volume_target,
                                    capacities=capacities_t,
                                    busy_threshold=decoded_busy_threshold,
                                    busy_volume_weight=transition_busy_volume_weight,
                                    busy_presence_weight=transition_busy_presence_weight,
                                    busy_presence_margin=decoded_busy_presence_margin,
                                    topk_rank_weight=transition_topk_rank_weight,
                                    topk_rank_k=transition_topk_rank_k,
                                    topk_rank_margin=transition_topk_rank_margin,
                                    zero_volume_weight=transition_zero_volume_weight,
                                    zero_volume_threshold=decoded_zero_volume_threshold,
                                    zero_volume_margin=decoded_zero_volume_margin,
                                    zero_target_threshold=decoded_zero_target_threshold,
                                )
                            else:
                                shuffle_sup_loss, shuffle_aux = transition_action_contrast_metric_loss(
                                    z_pred_next=z_pred_next_shuffled,
                                    flow_projector=flow_projector.module if hasattr(flow_projector, "module") else flow_projector,
                                    flow_target_next=flows_b[:, -1],
                                    metric=transition_action_contrast_metric_name,
                                    speed_index=decoder_recon_speed_index,
                                    volume_index=bpr_veh_index,
                                    speed_target_mode=decoded_speed_target,
                                    volume_target_mode=decoded_volume_target,
                                    capacities=capacities_t,
                                    busy_threshold=decoded_busy_threshold,
                                )
                                transition_aux["pred_next_action_contrast_shuffle_metric"] = shuffle_sup_loss.detach()
                                transition_aux.update(
                                    {
                                        f"shuffled_{k}": v
                                        for k, v in shuffle_aux.items()
                                    }
                                )
                            contrast_raw = F.relu(
                                torch.as_tensor(transition_action_contrast_margin, device=device, dtype=torch.float32)
                                + real_sup_unscaled
                                - shuffle_sup_loss
                            )
                            transition_action_contrast_loss = transition_action_contrast_weight * contrast_raw
                            transition_aux["pred_next_action_contrast_loss"] = transition_action_contrast_loss.detach()
                            transition_aux["pred_next_action_contrast_raw"] = contrast_raw.detach()
                            if "pred_next_decoded_total_mse" in shuffle_aux:
                                transition_aux["pred_next_shuffled_total_mse"] = shuffle_aux["pred_next_decoded_total_mse"]
                        if action_local_latent_weight > 0.0:
                            action_local_latent_loss, action_local_latent_aux = action_local_latent_delta_loss(
                                z_pred_next=z_pred_next,
                                z_prev=z1[:, -2],
                                z_target_next=z1[:, -1],
                                actions=actions,
                                n_spatial=n_spatial,
                                loss_weight=action_local_latent_weight,
                                action_eps=action_local_latent_eps,
                            )
                        if action_volume_delta_weight > 0.0:
                            fp_for_delta = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
                            action_volume_delta_sup_loss, action_volume_delta_aux = action_volume_delta_loss(
                                z_pred_next=z_pred_next,
                                z_prev=z1[:, -2],
                                flow_projector=fp_for_delta,
                                flow_prev=flows_b[:, -2],
                                flow_target_next=flows_b[:, -1],
                                actions=actions,
                                n_spatial=n_spatial,
                                loss_weight=action_volume_delta_weight,
                                action_eps=action_volume_delta_eps,
                                volume_index=bpr_veh_index,
                                speed_index=decoder_recon_speed_index,
                                speed_target_mode=decoded_speed_target,
                                volume_target_mode=decoded_volume_target,
                                capacities=capacities_t,
                            )
                    if (
                        adm_cfm_horizons
                        and (adm_cfm_weight > 0.0 or adm_decoded_supervision_weight > 0.0)
                        and flows_b.shape[1] >= (min(adm_cfm_horizons) + 1)
                    ):
                        fp_for_adm = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
                        dyn_for_adm = dyn.module if hasattr(dyn, "module") else dyn
                        for adm_h in adm_cfm_horizons:
                            if flows_b.shape[1] <= int(adm_h):
                                continue
                            context_z_h, target_z_h = _build_adm_context_target_windows(z1, int(adm_h))
                            flow_target_h = flows_b[:, int(adm_h):].reshape(
                                target_z_h.shape[0],
                                1,
                                flows_b.shape[2],
                                flows_b.shape[3],
                            )
                            act_raw_h = _build_adm_action_windows(act_raw, int(adm_h)).to(device, non_blocking=True)
                            hdv_target_h = _build_adm_action_windows(hdv_target, int(adm_h))
                            if hdv_target_h is not None:
                                hdv_target_h = hdv_target_h.to(device, non_blocking=True)
                            act_for_transition_h = _compose_transition_hdv_action(
                                act_raw=act_raw_h,
                                hdv_target=hdv_target_h,
                                mode=str(getattr(args, "transition_hdv_action_mode", "none")),
                                weight=float(getattr(args, "transition_hdv_action_weight", 0.0)),
                                scale=float(getattr(args, "transition_hdv_action_scale", 1.0)),
                            )
                            hdv_context_h = _build_transition_hdv_context(
                                act_raw=act_raw_h,
                                hdv_target=hdv_target_h,
                                mode=str(getattr(args, "transition_hdv_context_mode", "none")),
                                weight=float(getattr(args, "transition_hdv_context_weight", 0.0)),
                                scale=float(getattr(args, "transition_hdv_context_scale", 1.0)),
                            )
                            depart_h = _build_adm_action_windows(depart_b, int(adm_h))
                            context_h = _build_adm_action_windows(context_b, int(adm_h))
                            if depart_h is not None:
                                depart_h = depart_h.to(device, non_blocking=True)
                            if context_h is not None:
                                context_h = context_h.to(device, non_blocking=True)
                            if action_encoder_type == "graph_cond":
                                actions_adm = (
                                    (act_for_transition_h, depart_h, context_h, hdv_context_h)
                                    if hdv_context_h is not None else (act_for_transition_h, depart_h, context_h)
                                )
                            else:
                                actions_adm = _build_model_actions(
                                    act_raw=act_for_transition_h,
                                    depart_b=depart_h if model_action_uses_depart_context else None,
                                    context_b=context_h if model_action_uses_depart_context else None,
                                    action_compressor=action_compressor_state,
                                    depart_compressor=depart_compressor_state if model_action_uses_depart_context else None,
                                ).to(device, non_blocking=True)
                            if adm_max_windows > 0 and context_z_h.shape[0] > adm_max_windows:
                                adm_idx = torch.randperm(context_z_h.shape[0], device=device)[:adm_max_windows]
                                context_z_h = context_z_h.index_select(0, adm_idx)
                                target_z_h = target_z_h.index_select(0, adm_idx)
                                flow_target_h = flow_target_h.index_select(0, adm_idx)
                                actions_adm = _slice_adm_model_actions(actions_adm, adm_idx)
                            horizon_scale = float(adm_cfm_gamma) ** max(0, int(adm_h) - 1)
                            if adm_cfm_weight > 0.0:
                                loss_adm_h, aux_adm_h = adm_multistep_cfm_loss(
                                    dyn_for_adm,
                                    context_z=context_z_h,
                                    target_z=target_z_h,
                                    actions=actions_adm,
                                    act_mask=None,
                                    flow_targets=flow_target_h,
                                    cfm_active_weight=cfm_active_weight,
                                    cfm_busy_weight=cfm_busy_weight,
                                    cfm_busy_threshold=cfm_busy_threshold,
                                    cfm_active_threshold=cfm_active_threshold,
                                    cfm_max_weight=cfm_max_weight,
                                    volume_index=bpr_veh_index,
                                    init_mode=adm_cfm_init_mode,
                                )
                                adm_multistep_sup_loss = adm_multistep_sup_loss + float(adm_cfm_weight) * horizon_scale * loss_adm_h
                                for k_aux, v_aux in aux_adm_h.items():
                                    adm_multistep_aux[f"adm_h{int(adm_h)}_{k_aux}"] = v_aux
                            if adm_decoded_supervision_weight > 0.0:
                                z_pred_adm_h = adm_predict_k_timestep_direct(
                                    dyn_for_adm,
                                    context_z=context_z_h,
                                    actions=actions_adm,
                                    act_mask=None,
                                    n_steps=int(getattr(args, "n_ode_steps", 1)),
                                    init_mode=adm_cfm_init_mode,
                                )
                                loss_dec_adm_h, aux_dec_adm_h = transition_decoded_supervision_loss(
                                    z_pred_next=z_pred_adm_h,
                                    flow_projector=fp_for_adm,
                                    flow_target_next=flow_target_h[:, 0],
                                    supervision_weight=float(adm_decoded_supervision_weight) * horizon_scale,
                                    speed_index=decoder_recon_speed_index,
                                    volume_index=bpr_veh_index,
                                    speed_weight=decoder_recon_speed_weight,
                                    volume_weight=decoder_volume_weight,
                                    speed_target_mode=decoded_speed_target,
                                    volume_target_mode=decoded_volume_target,
                                    capacities=capacities_t,
                                    busy_threshold=decoded_busy_threshold,
                                    busy_volume_weight=transition_busy_volume_weight,
                                    busy_presence_weight=transition_busy_presence_weight,
                                    busy_presence_margin=decoded_busy_presence_margin,
                                    topk_rank_weight=transition_topk_rank_weight,
                                    topk_rank_k=transition_topk_rank_k,
                                    topk_rank_margin=transition_topk_rank_margin,
                                    zero_volume_weight=transition_zero_volume_weight,
                                    zero_volume_threshold=decoded_zero_volume_threshold,
                                    zero_volume_margin=decoded_zero_volume_margin,
                                    zero_target_threshold=decoded_zero_target_threshold,
                                )
                                adm_multistep_sup_loss = adm_multistep_sup_loss + loss_dec_adm_h
                                for k_aux, v_aux in aux_dec_adm_h.items():
                                    adm_multistep_aux[f"adm_h{int(adm_h)}_{k_aux}"] = v_aux
                    fp_head = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
                    t_all_sup_loss, t_all_aux = episode_t_all_supervision_loss(
                        flow_projector=fp_head,
                        z_tokens=z1,
                        reward_targets=reward_target,
                        t_all_weight=t_all_supervision_weight,
                        missing_reward_value=t_all_missing_reward,
                        target_mode=t_all_target_mode,
                    )
                    loss = (
                        base_loss
                        + decoder_recon_loss
                        + bpr_sup_loss
                        + route_rank_loss
                        + route_distill_loss
                        + spatial_sup_loss
                        + decoder_head_loss
                        + transition_sup_loss
                        + transition_route_rank_loss
                        + transition_action_contrast_loss
                        + action_local_latent_loss
                        + action_volume_delta_sup_loss
                        + adm_multistep_sup_loss
                        + t_all_sup_loss
                    )
                    # obs_proj reconstruction: supervise latent obs on flow targets
                    obs_proj_loss = torch.zeros((), device=device)
                    if obs_proj_weight > 0.0 and fp_head.obs_proj is not None:
                        obs_pred = fp_head.project_obs(z1)           # (B, T, E, d_obs)
                        d_o = min(obs_pred.shape[-1], flows_b.shape[-1])
                        obs_target = flows_b[..., :d_o].to(dtype=obs_pred.dtype)
                        obs_proj_loss = obs_proj_weight * (obs_pred[..., :d_o].float() - obs_target.float()).pow(2).mean()
                        loss = loss + obs_proj_loss
                        aux["obs_proj_loss"] = obs_proj_loss.detach()
                    aux.update(decoder_recon_aux)
                    aux.update(bpr_aux)
                    aux.update(route_rank_aux)
                    aux.update(route_distill_aux)
                    aux.update(spatial_aux)
                    aux.update(decoder_head_aux)
                    aux.update(transition_aux)
                    aux.update({f"transition_{k}": v for k, v in transition_route_rank_aux.items()})
                    aux.update(action_local_latent_aux)
                    aux.update(action_volume_delta_aux)
                    aux.update(adm_multistep_aux)
                    aux.update(t_all_aux)
                    aux["loss_base"] = base_loss.detach()

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {step}: loss={loss}")

                loss_to_backprop = loss / grad_accum
                scaler.scale(loss_to_backprop).backward()

                do_step = ((step + 1) % grad_accum == 0)
                if do_step:
                    if args.grad_clip > 0:
                        scaler.unscale_(opt)
                        _clip_params = list((dyn.module if hasattr(dyn, "module") else dyn).parameters())
                        if not freeze_flow_projector:
                            _clip_params += list(flow_projector.parameters())
                        if stackelberg_layer is not None:
                            _clip_params += list(stackelberg_layer.parameters())
                        torch.nn.utils.clip_grad_norm_(_clip_params, max_norm=args.grad_clip)

                    if use_amp:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                # Evaluation / visualization
                do_eval = args.eval_every > 0 and (step % args.eval_every == 0)
                if do_eval and is_rank0():
                    # Flow mode: log loss directly (no image reconstruction eval)
                    wandb.log({"eval/flow_loss": float(loss.item())}, step=step)

                # Logging
                do_log = step % args.log_every == 0
                if do_log:
                    if is_rank0():
                        # Action shuffle loss ratio — use already-computed loss to avoid
                        # a second full forward pass that could hang under DDP
                        if actions is not None:
                            with torch.no_grad():
                                if isinstance(actions, tuple):
                                    perm = torch.randperm(actions[0].shape[0], device=actions[0].device)
                                    actions_shuffled = tuple(
                                        a[perm] if a is not None else None for a in actions
                                    )
                                else:
                                    perm = torch.randperm(actions.shape[0], device=actions.device)
                                    actions_shuffled = actions[perm]
                                if hdv_actions is not None:
                                    if isinstance(hdv_actions, tuple):
                                        hdv_actions_shuffled = tuple(
                                            a[perm] if a is not None else None for a in hdv_actions
                                        )
                                    else:
                                        hdv_actions_shuffled = hdv_actions[perm]
                                else:
                                    hdv_actions_shuffled = None
                                hdv_target_perm = hdv_target[perm] if hdv_target is not None else None
                                loss_shuffled, _ = cfm_loss(
                                    dyn.module if hasattr(dyn, "module") else dyn,
                                    z1=z1,
                                    actions=actions_shuffled,
                                    act_mask=act_mask,
                                    hdv_actions=hdv_actions_shuffled,
                                    agent_tokens=None,
                                    hdv_target=hdv_target_perm,
                                    hdv_policy_weight=float(getattr(args, "hdv_policy_weight", 0.0)),
                                    flow_targets=flows_b,
                                    cfm_active_weight=cfm_active_weight,
                                    cfm_busy_weight=cfm_busy_weight,
                                    cfm_busy_threshold=cfm_busy_threshold,
                                    cfm_active_threshold=cfm_active_threshold,
                                    cfm_max_weight=cfm_max_weight,
                                    volume_index=bpr_veh_index,
                                    av_action_dim=(
                                        int(action_compressor_state["output_dim"])
                                        if action_compressor_state is not None else int(action_dim_flow)
                                    ),
                                )
                            action_shuffle_loss_ratio = loss_shuffled / base_loss.detach().clamp_min(1e-8)
                        else:
                            action_shuffle_loss_ratio = torch.tensor(0., device=device)

                        log_dict = {
                            "loss/total": float(loss.item()),
                            "loss/base": float(aux["loss_base"].item()),
                            "loss/flow_mse": float(aux["flow_mse"].item()),
                            "stats/action_shuffle_loss_ratio": float(action_shuffle_loss_ratio.item()),
                            "lr": float(opt.param_groups[0]["lr"]),
                            "time/hrs": (time.time() - t0) / 3600.0,
                        }
                        if "weighted_flow_mse" in aux:
                            log_dict["loss/weighted_flow_mse"] = float(aux["weighted_flow_mse"].item())
                        if "cfm_weight_mean" in aux:
                            log_dict["stats/cfm_weight_mean"] = float(aux["cfm_weight_mean"].item())
                        if "cfm_active_edge_frac" in aux:
                            log_dict["stats/cfm_active_edge_frac"] = float(aux["cfm_active_edge_frac"].item())
                        if "cfm_busy_edge_frac" in aux:
                            log_dict["stats/cfm_busy_edge_frac"] = float(aux["cfm_busy_edge_frac"].item())
                        if "t_mean" in aux:
                            log_dict["stats/t_mean"] = float(aux["t_mean"].item())
                        if "hdv_policy_ce" in aux:
                            log_dict["loss/hdv_policy_ce"] = float(aux["hdv_policy_ce"].item())
                        if "eq_residual" in aux and float(aux["eq_residual"].item()) > 0:
                            log_dict["loss/eq_residual"] = float(aux["eq_residual"].item())
                        if "stk_hdv_ce" in aux and float(aux["stk_hdv_ce"].item()) > 0:
                            log_dict["loss/stk_hdv_ce"] = float(aux["stk_hdv_ce"].item())
                        if "decoder_recon_loss" in aux:
                            log_dict["loss/decoder_recon_loss"] = float(aux["decoder_recon_loss"].item())
                        if "obs_proj_loss" in aux:
                            log_dict["loss/obs_proj_loss"] = float(aux["obs_proj_loss"].item())
                        if "decoder_recon_mse" in aux:
                            log_dict["loss/decoder_recon_mse"] = float(aux["decoder_recon_mse"].item())
                        if "decoder_recon_speed_mse" in aux:
                            log_dict["loss/decoder_recon_speed_mse"] = float(aux["decoder_recon_speed_mse"].item())
                        if "decoder_recon_weighted_mse" in aux:
                            log_dict["loss/decoder_recon_weighted_mse"] = float(aux["decoder_recon_weighted_mse"].item())
                        if "decoder_recon_busy_volume_weighted_mse" in aux:
                            log_dict["loss/decoder_recon_busy_volume_weighted_mse"] = float(aux["decoder_recon_busy_volume_weighted_mse"].item())
                        if "decoder_recon_busy_presence_bce" in aux:
                            log_dict["loss/decoder_recon_busy_presence_bce"] = float(aux["decoder_recon_busy_presence_bce"].item())
                        if "decoder_recon_topk_rank_loss" in aux:
                            log_dict["loss/decoder_recon_topk_rank_loss"] = float(aux["decoder_recon_topk_rank_loss"].item())
                        if "decoder_recon_zero_volume_false_alarm_loss" in aux:
                            log_dict["loss/decoder_recon_zero_volume_false_alarm_loss"] = float(aux["decoder_recon_zero_volume_false_alarm_loss"].item())
                        if "decoder_recon_total_mse" in aux:
                            log_dict["loss/decoder_recon_total_mse"] = float(aux["decoder_recon_total_mse"].item())
                        if "bpr_supervision_loss" in aux:
                            log_dict["loss/bpr_supervision_loss"] = float(aux["bpr_supervision_loss"].item())
                        if "decoder_multihead_loss" in aux:
                            log_dict["loss/decoder_multihead_loss"] = float(aux["decoder_multihead_loss"].item())
                        if "decoder_full_mse" in aux:
                            log_dict["loss/decoder_full_mse"] = float(aux["decoder_full_mse"].item())
                        if "decoder_speed_mse" in aux:
                            log_dict["loss/decoder_speed_mse"] = float(aux["decoder_speed_mse"].item())
                        if "decoder_volume_mse" in aux:
                            log_dict["loss/decoder_volume_mse"] = float(aux["decoder_volume_mse"].item())
                        if "decoder_busy_volume_weighted_mse" in aux:
                            log_dict["loss/decoder_busy_volume_weighted_mse"] = float(aux["decoder_busy_volume_weighted_mse"].item())
                        if "decoder_busy_presence_bce" in aux:
                            log_dict["loss/decoder_busy_presence_bce"] = float(aux["decoder_busy_presence_bce"].item())
                        if "decoder_topk_rank_loss" in aux:
                            log_dict["loss/decoder_topk_rank_loss"] = float(aux["decoder_topk_rank_loss"].item())
                        if "decoder_zero_volume_false_alarm_loss" in aux:
                            log_dict["loss/decoder_zero_volume_false_alarm_loss"] = float(aux["decoder_zero_volume_false_alarm_loss"].item())
                        if "decoder_other_mse" in aux:
                            log_dict["loss/decoder_other_mse"] = float(aux["decoder_other_mse"].item())
                        if "pred_next_decoded_supervision_loss" in aux:
                            log_dict["loss/pred_next_decoded_supervision_loss"] = float(aux["pred_next_decoded_supervision_loss"].item())
                        if "pred_next_decoded_weighted_mse" in aux:
                            log_dict["loss/pred_next_decoded_weighted_mse"] = float(aux["pred_next_decoded_weighted_mse"].item())
                        if "pred_next_decoded_speed_mse" in aux:
                            log_dict["loss/pred_next_decoded_speed_mse"] = float(aux["pred_next_decoded_speed_mse"].item())
                        if "pred_next_decoded_volume_mse" in aux:
                            log_dict["loss/pred_next_decoded_volume_mse"] = float(aux["pred_next_decoded_volume_mse"].item())
                        if "pred_next_decoded_busy_volume_weighted_mse" in aux:
                            log_dict["loss/pred_next_decoded_busy_volume_weighted_mse"] = float(aux["pred_next_decoded_busy_volume_weighted_mse"].item())
                        if "pred_next_decoded_busy_presence_bce" in aux:
                            log_dict["loss/pred_next_decoded_busy_presence_bce"] = float(aux["pred_next_decoded_busy_presence_bce"].item())
                        if "pred_next_decoded_topk_rank_loss" in aux:
                            log_dict["loss/pred_next_decoded_topk_rank_loss"] = float(aux["pred_next_decoded_topk_rank_loss"].item())
                        if "pred_next_decoded_zero_volume_false_alarm_loss" in aux:
                            log_dict["loss/pred_next_decoded_zero_volume_false_alarm_loss"] = float(aux["pred_next_decoded_zero_volume_false_alarm_loss"].item())
                        if "pred_next_action_contrast_loss" in aux:
                            log_dict["loss/pred_next_action_contrast_loss"] = float(aux["pred_next_action_contrast_loss"].item())
                        if "pred_next_action_contrast_raw" in aux:
                            log_dict["loss/pred_next_action_contrast_raw"] = float(aux["pred_next_action_contrast_raw"].item())
                        if "pred_next_action_contrast_true_metric" in aux:
                            log_dict["loss/pred_next_action_contrast_true_metric"] = float(aux["pred_next_action_contrast_true_metric"].item())
                        if "pred_next_action_contrast_shuffle_metric" in aux:
                            log_dict["loss/pred_next_action_contrast_shuffle_metric"] = float(aux["pred_next_action_contrast_shuffle_metric"].item())
                        if "pred_next_shuffled_total_mse" in aux:
                            log_dict["loss/pred_next_shuffled_total_mse"] = float(aux["pred_next_shuffled_total_mse"].item())
                        if "action_volume_delta_loss" in aux:
                            log_dict["loss/action_volume_delta_loss"] = float(aux["action_volume_delta_loss"].item())
                        if "action_volume_delta_mse" in aux:
                            log_dict["loss/action_volume_delta_mse"] = float(aux["action_volume_delta_mse"].item())
                        if "action_volume_delta_active_frac" in aux:
                            log_dict["stats/action_volume_delta_active_frac"] = float(aux["action_volume_delta_active_frac"].item())
                        if "action_volume_delta_target_abs_mean" in aux:
                            log_dict["stats/action_volume_delta_target_abs_mean"] = float(aux["action_volume_delta_target_abs_mean"].item())
                        if "action_volume_delta_pred_abs_mean" in aux:
                            log_dict["stats/action_volume_delta_pred_abs_mean"] = float(aux["action_volume_delta_pred_abs_mean"].item())
                        if "pred_next_decoded_total_mse" in aux:
                            log_dict["loss/pred_next_decoded_total_mse"] = float(aux["pred_next_decoded_total_mse"].item())
                        if "decoder_recon_busy_edge_frac" in aux:
                            log_dict["stats/decoder_recon_busy_edge_frac"] = float(aux["decoder_recon_busy_edge_frac"].item())
                        if "decoder_recon_busy_speed_mse" in aux:
                            log_dict["loss/decoder_recon_busy_speed_mse"] = float(aux["decoder_recon_busy_speed_mse"].item())
                        if "decoder_recon_busy_volume_mse" in aux:
                            log_dict["loss/decoder_recon_busy_volume_mse"] = float(aux["decoder_recon_busy_volume_mse"].item())
                        if "decoder_busy_edge_frac" in aux:
                            log_dict["stats/decoder_busy_edge_frac"] = float(aux["decoder_busy_edge_frac"].item())
                        if "decoder_busy_speed_mse" in aux:
                            log_dict["loss/decoder_busy_speed_mse"] = float(aux["decoder_busy_speed_mse"].item())
                        if "decoder_busy_volume_mse" in aux:
                            log_dict["loss/decoder_busy_volume_mse"] = float(aux["decoder_busy_volume_mse"].item())
                        if "pred_next_decoded_busy_edge_frac" in aux:
                            log_dict["stats/pred_next_decoded_busy_edge_frac"] = float(aux["pred_next_decoded_busy_edge_frac"].item())
                        if "pred_next_decoded_busy_speed_mse" in aux:
                            log_dict["loss/pred_next_decoded_busy_speed_mse"] = float(aux["pred_next_decoded_busy_speed_mse"].item())
                        if "pred_next_decoded_busy_volume_mse" in aux:
                            log_dict["loss/pred_next_decoded_busy_volume_mse"] = float(aux["pred_next_decoded_busy_volume_mse"].item())
                        if "spatial_supervision_loss" in aux:
                            log_dict["loss/spatial_supervision_loss"] = float(aux["spatial_supervision_loss"].item())
                        if "busy_volume_mse" in aux:
                            log_dict["loss/busy_volume_mse"] = float(aux["busy_volume_mse"].item())
                        if "busy_edge_frac" in aux:
                            log_dict["stats/busy_edge_frac"] = float(aux["busy_edge_frac"].item())
                        if "active_edge_precision" in aux:
                            log_dict["stats/active_edge_precision"] = float(aux["active_edge_precision"].item())
                        if "active_edge_recall" in aux:
                            log_dict["stats/active_edge_recall"] = float(aux["active_edge_recall"].item())
                        if "bpr_edge_mse" in aux:
                            log_dict["loss/bpr_edge_mse"] = float(aux["bpr_edge_mse"].item())
                        if "bpr_edge_mse_normalized" in aux:
                            log_dict["loss/bpr_edge_mse_normalized"] = float(aux["bpr_edge_mse_normalized"].item())
                        if "bpr_edge_mse_raw_seconds" in aux:
                            log_dict["loss/bpr_edge_mse_raw_seconds"] = float(aux["bpr_edge_mse_raw_seconds"].item())
                        if "bpr_weighted_mse" in aux:
                            log_dict["loss/bpr_weighted_mse"] = float(aux["bpr_weighted_mse"].item())
                        if "t_all_supervision_loss" in aux:
                            log_dict["loss/t_all_supervision_loss"] = float(aux["t_all_supervision_loss"].item())
                        if "t_all_pred_mse" in aux:
                            log_dict["loss/t_all_pred_mse"] = float(aux["t_all_pred_mse"].item())
                        if "t_all_pred_mse_normalized" in aux:
                            log_dict["loss/t_all_pred_mse_normalized"] = float(aux["t_all_pred_mse_normalized"].item())
                        if "t_all_pred_mse_raw_minutes" in aux:
                            log_dict["loss/t_all_pred_mse_raw_minutes"] = float(aux["t_all_pred_mse_raw_minutes"].item())
                        if "t_all_target_mean" in aux:
                            log_dict["stats/t_all_target_mean"] = float(aux["t_all_target_mean"].item())
                        if "t_all_pred_mean" in aux:
                            log_dict["stats/t_all_pred_mean"] = float(aux["t_all_pred_mean"].item())
                        if "t_all_target_scale" in aux:
                            log_dict["stats/t_all_target_scale"] = float(aux["t_all_target_scale"].item())
                        if "t_all_valid_frac" in aux:
                            log_dict["stats/t_all_valid_frac"] = float(aux["t_all_valid_frac"].item())
                        if "scheduled_sampling_prob" in aux:
                            log_dict["stats/scheduled_sampling_prob"] = float(aux["scheduled_sampling_prob"].item())
                        log_dict["stats/rollout_horizon"] = int(rollout_horizon)
                        for k, v in aux.items():
                            if k.startswith("adm_h"):
                                namespace = "stats" if ("frac" in k or k.endswith("_mean")) else "loss"
                                log_dict[f"{namespace}/{k}"] = float(v.item())
                                continue
                            if "route_rank" in k:
                                namespace = "stats" if (
                                    k.endswith("_acc")
                                    or k.endswith("_frac")
                                    or k.endswith("_recall")
                                    or k.endswith("_samples")
                                ) else "loss"
                                log_dict[f"{namespace}/{k}"] = float(v.item())
                                continue
                            if "route_distill" in k:
                                namespace = "stats" if (
                                    k.endswith("_acc")
                                    or k.endswith("_frac")
                                    or k.endswith("_recall")
                                    or k.endswith("_samples")
                                ) else "loss"
                                log_dict[f"{namespace}/{k}"] = float(v.item())
                                continue
                            if k.startswith("det_"):
                                namespace = "stats" if k.endswith("_mae") else "loss"
                                log_dict[f"{namespace}/{k}"] = float(v.item())

                        # Log to wandb
                        wandb.log(log_dict, step=step)

                        # Log to console
                        inv_msg = ""
                        if "weighted_flow_mse" in aux and "cfm_weight_mean" in aux:
                            inv_msg += (
                                f" | cfm_w_mse={aux['weighted_flow_mse'].item():.6f}"
                                f" | cfm_w={aux['cfm_weight_mean'].item():.3f}"
                            )
                        if "decoder_recon_mse" in aux:
                            inv_msg += f" | dec_recon_mse={aux['decoder_recon_mse'].item():.6f}"
                        if "decoder_recon_speed_mse" in aux:
                            inv_msg += f" | dec_recon_speed={aux['decoder_recon_speed_mse'].item():.6f}"
                        if "decoder_recon_busy_speed_mse" in aux:
                            inv_msg += f" | dec_recon_busy_speed={aux['decoder_recon_busy_speed_mse'].item():.6f}"
                        if "decoder_recon_busy_volume_mse" in aux:
                            inv_msg += f" | dec_recon_busy_vol={aux['decoder_recon_busy_volume_mse'].item():.6f}"
                        if "decoder_recon_busy_volume_weighted_mse" in aux:
                            inv_msg += f" | dec_recon_busy_vol_w={aux['decoder_recon_busy_volume_weighted_mse'].item():.6f}"
                        if "decoder_recon_busy_presence_bce" in aux:
                            inv_msg += f" | dec_recon_busy_bce={aux['decoder_recon_busy_presence_bce'].item():.6f}"
                        if "decoder_recon_topk_rank_loss" in aux:
                            inv_msg += f" | dec_recon_topk={aux['decoder_recon_topk_rank_loss'].item():.6f}"
                        if "decoder_recon_zero_volume_false_alarm_loss" in aux:
                            inv_msg += f" | dec_recon_zero_fa={aux['decoder_recon_zero_volume_false_alarm_loss'].item():.6f}"
                        if "bpr_edge_mse" in aux:
                            inv_msg += f" | bpr_mse={aux['bpr_edge_mse'].item():.6f}"
                        if "bpr_edge_mse_raw_seconds" in aux:
                            inv_msg += f" | bpr_raw_s={aux['bpr_edge_mse_raw_seconds'].item():.6f}"
                        if "route_rank_loss" in aux:
                            inv_msg += (
                                f" | route_rank={aux['route_rank_loss'].item():.6f}"
                                f"/acc={aux.get('route_rank_acc', torch.tensor(0.0, device=device)).item():.3f}"
                                f"/non0={aux.get('route_rank_true_non0_frac', torch.tensor(0.0, device=device)).item():.3f}"
                                f"/gap={aux.get('route_rank_gap_loss', torch.tensor(0.0, device=device)).item():.3f}"
                            )
                        if "route_distill_loss" in aux:
                            inv_msg += (
                                f" | route_distill={aux['route_distill_loss'].item():.6f}"
                                f"/acc={aux.get('route_distill_acc', torch.tensor(0.0, device=device)).item():.3f}"
                                f"/non0={aux.get('route_distill_true_non0_frac', torch.tensor(0.0, device=device)).item():.3f}"
                                f"/gap={aux.get('route_distill_gap_loss', torch.tensor(0.0, device=device)).item():.3f}"
                            )
                        if "transition_route_rank_loss" in aux:
                            inv_msg += (
                                f" | tr_route_rank={aux['transition_route_rank_loss'].item():.6f}"
                                f"/acc={aux.get('transition_route_rank_acc', torch.tensor(0.0, device=device)).item():.3f}"
                                f"/non0={aux.get('transition_route_rank_true_non0_frac', torch.tensor(0.0, device=device)).item():.3f}"
                                f"/gap={aux.get('transition_route_rank_gap_loss', torch.tensor(0.0, device=device)).item():.3f}"
                            )
                        if "decoder_multihead_loss" in aux:
                            inv_msg += f" | dec_mh={aux['decoder_multihead_loss'].item():.6f}"
                        if "decoder_speed_mse" in aux:
                            inv_msg += f" | dec_speed_mse={aux['decoder_speed_mse'].item():.6f}"
                        if "decoder_volume_mse" in aux:
                            inv_msg += f" | dec_vol_mse={aux['decoder_volume_mse'].item():.6f}"
                        if "decoder_busy_speed_mse" in aux:
                            inv_msg += f" | dec_busy_speed={aux['decoder_busy_speed_mse'].item():.6f}"
                        if "decoder_busy_volume_mse" in aux:
                            inv_msg += f" | dec_busy_vol={aux['decoder_busy_volume_mse'].item():.6f}"
                        if "decoder_busy_volume_weighted_mse" in aux:
                            inv_msg += f" | dec_busy_vol_w={aux['decoder_busy_volume_weighted_mse'].item():.6f}"
                        if "decoder_busy_presence_bce" in aux:
                            inv_msg += f" | dec_busy_bce={aux['decoder_busy_presence_bce'].item():.6f}"
                        if "decoder_topk_rank_loss" in aux:
                            inv_msg += f" | dec_topk={aux['decoder_topk_rank_loss'].item():.6f}"
                        if "decoder_zero_volume_false_alarm_loss" in aux:
                            inv_msg += f" | dec_zero_fa={aux['decoder_zero_volume_false_alarm_loss'].item():.6f}"
                        if "decoder_other_mse" in aux:
                            inv_msg += f" | dec_other_mse={aux['decoder_other_mse'].item():.6f}"
                        if "pred_next_decoded_supervision_loss" in aux:
                            inv_msg += f" | pred_next_dec={aux['pred_next_decoded_supervision_loss'].item():.6f}"
                        if "pred_next_decoded_speed_mse" in aux:
                            inv_msg += f" | pred_next_speed_mse={aux['pred_next_decoded_speed_mse'].item():.6f}"
                        if "pred_next_decoded_volume_mse" in aux:
                            inv_msg += f" | pred_next_vol_mse={aux['pred_next_decoded_volume_mse'].item():.6f}"
                        if "pred_next_decoded_busy_speed_mse" in aux:
                            inv_msg += f" | pred_busy_speed={aux['pred_next_decoded_busy_speed_mse'].item():.6f}"
                        if "pred_next_decoded_busy_volume_mse" in aux:
                            inv_msg += f" | pred_busy_vol={aux['pred_next_decoded_busy_volume_mse'].item():.6f}"
                        if "pred_next_decoded_busy_volume_weighted_mse" in aux:
                            inv_msg += f" | pred_busy_vol_w={aux['pred_next_decoded_busy_volume_weighted_mse'].item():.6f}"
                        if "pred_next_decoded_busy_presence_bce" in aux:
                            inv_msg += f" | pred_busy_bce={aux['pred_next_decoded_busy_presence_bce'].item():.6f}"
                        if "pred_next_decoded_topk_rank_loss" in aux:
                            inv_msg += f" | pred_topk_rank={aux['pred_next_decoded_topk_rank_loss'].item():.6f}"
                        if "pred_next_decoded_zero_volume_false_alarm_loss" in aux:
                            inv_msg += f" | pred_zero_fa={aux['pred_next_decoded_zero_volume_false_alarm_loss'].item():.6f}"
                        if "adm_h2_adm_weighted_flow_mse" in aux:
                            inv_msg += f" | adm_h2_w={aux['adm_h2_adm_weighted_flow_mse'].item():.6f}"
                        if "adm_h4_adm_weighted_flow_mse" in aux:
                            inv_msg += f" | adm_h4_w={aux['adm_h4_adm_weighted_flow_mse'].item():.6f}"
                        if "adm_h8_adm_weighted_flow_mse" in aux:
                            inv_msg += f" | adm_h8_w={aux['adm_h8_adm_weighted_flow_mse'].item():.6f}"
                        if "pred_next_action_contrast_loss" in aux:
                            inv_msg += f" | act_contrast={aux['pred_next_action_contrast_loss'].item():.6f}"
                        if "pred_next_action_contrast_true_metric" in aux:
                            inv_msg += f" | act_true_metric={aux['pred_next_action_contrast_true_metric'].item():.6f}"
                        if "pred_next_action_contrast_shuffle_metric" in aux:
                            inv_msg += f" | act_shuffle_metric={aux['pred_next_action_contrast_shuffle_metric'].item():.6f}"
                        if "pred_next_shuffled_total_mse" in aux:
                            inv_msg += f" | pred_shuffle_total={aux['pred_next_shuffled_total_mse'].item():.6f}"
                        if "action_recon_active_bce" in aux:
                            inv_msg += f" | act_recon_bce={aux['action_recon_active_bce'].item():.6f}"
                        if "action_recon_value_mse" in aux:
                            inv_msg += f" | act_recon_val={aux['action_recon_value_mse'].item():.6f}"
                        if "action_local_latent_mse" in aux:
                            inv_msg += f" | act_latent={aux['action_local_latent_mse'].item():.6f}"
                        if "action_volume_delta_mse" in aux:
                            inv_msg += f" | act_vol_delta={aux['action_volume_delta_mse'].item():.6f}"
                        if "busy_volume_mse" in aux:
                            inv_msg += f" | busy_vol_mse={aux['busy_volume_mse'].item():.6f}"
                        if "active_edge_precision" in aux:
                            inv_msg += f" | act_prec={aux['active_edge_precision'].item():.3f}"
                        if "active_edge_recall" in aux:
                            inv_msg += f" | act_rec={aux['active_edge_recall'].item():.3f}"
                        if "t_all_pred_mse" in aux:
                            inv_msg += f" | t_all_mse={aux['t_all_pred_mse'].item():.6f}"
                        if "t_all_pred_mse_raw_minutes" in aux:
                            inv_msg += f" | t_all_raw_mse={aux['t_all_pred_mse_raw_minutes'].item():.6f}"
                        if "t_all_pred_mean" in aux:
                            inv_msg += f" | t_all_pred={aux['t_all_pred_mean'].item():.3f}"
                        if "t_all_target_mean" in aux:
                            inv_msg += f" | t_all_true={aux['t_all_target_mean'].item():.3f}"
                        if "t_all_target_scale" in aux:
                            inv_msg += f" | t_all_scale={aux['t_all_target_scale'].item():.3f}"
                        if "eq_residual" in aux and float(aux["eq_residual"].item()) > 0:
                            inv_msg += f" | eq_res={aux['eq_residual'].item():.6f}"
                        if "stk_hdv_ce" in aux and float(aux["stk_hdv_ce"].item()) > 0:
                            inv_msg += f" | stk_hdv={aux['stk_hdv_ce'].item():.4f}"
                        print(
                            f"step {step:07d} | loss={loss.item():.6f} "
                            f"| flow_mse={aux['flow_mse'].item():.6f} "
                            f"| t_mean={aux['t_mean'].item():.3f} "
                            f"| hdv_ce={aux.get('hdv_policy_ce', torch.tensor(0.0, device=device)).item():.6f} "
                            f"{inv_msg}"
                        )

                    # Optional explicit barrier after rank0-only logging.
                    # Default is OFF to avoid end-of-run stalls on some setups.
                    if ddp and use_explicit_ddp_barrier:
                        dist.barrier()

                # Checkpointing
                do_save = args.save_every > 0 and (step % args.save_every == 0)
                if do_save:
                    if is_rank0():
                        ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
                        save_ckpt(
                            ckpt_path,
                            step=step,
                            epoch=epoch,
                            dyn_model=dyn,
                            opt=opt,
                            scaler=scaler,
                            args=args,
                            flow_projector=flow_projector,
                            action_compressor=action_compressor_state,
                        )
                        latest = ckpt_dir / "latest.pt"
                        save_ckpt(
                            latest,
                            step=step,
                            epoch=epoch,
                            dyn_model=dyn,
                            opt=opt,
                            scaler=scaler,
                            args=args,
                            flow_projector=flow_projector,
                            action_compressor=action_compressor_state,
                        )
                        fp_path = ckpt_dir / "flow_projector_latest.pt"
                        _fp_inner = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
                        _fp_inner.save(str(fp_path))
                    # Optional explicit barrier after rank0-only checkpoint save.
                    if ddp and use_explicit_ddp_barrier:
                        dist.barrier()

                step += 1
                # Stop immediately once reaching max_steps, so DataLoader does
                # not fetch one extra batch before the next loop-condition check.
                if step >= args.max_steps:
                    break

            # Important: when max_steps is reached, also break the epoch loop.
            # Otherwise we keep iterating epochs and repeatedly entering the
            # dataloader, which looks like a "hang at the final step".
            if step >= args.max_steps:
                break

            # Per-epoch Stackelberg HDV weight decay
            if stackelberg_layer is not None and stackelberg_hdv_weight_decay < 1.0:
                stackelberg_hdv_weight *= stackelberg_hdv_weight_decay
                if is_rank0() and stackelberg_hdv_weight > 1e-6:
                    print(f"[epoch {epoch}] stackelberg_hdv_weight decayed to {stackelberg_hdv_weight:.6f}")

            start_epoch = epoch + 1


    # save final checkpoint
    if is_rank0():
        ckpt_path = ckpt_dir / f"step_{step:07d}.pt"
        save_ckpt(
            ckpt_path,
            step=step,
            epoch=epoch,
            dyn_model=dyn,
            opt=opt,
            scaler=scaler,
            args=args,
            flow_projector=flow_projector,
            action_compressor=action_compressor_state,
        )
        latest = ckpt_dir / "latest.pt"
        save_ckpt(
            latest,
            step=step,
            epoch=epoch,
            dyn_model=dyn,
            opt=opt,
            scaler=scaler,
            args=args,
            flow_projector=flow_projector,
            action_compressor=action_compressor_state,
        )
        fp_path = ckpt_dir / "flow_projector_latest.pt"
        _fp_inner = flow_projector.module if hasattr(flow_projector, "module") else flow_projector
        _fp_inner.save(str(fp_path))
        print(f"[rank0] Final checkpoint saved at step {step}")
        if wandb.run is not None:
            wandb.finish()
    if ddp and dist.is_initialized():
        if use_explicit_ddp_barrier:
            dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    add_settings_arg(p)

    # data
    p.add_argument("--data_dirs", type=str, nargs="+", default=[
        "/<path>/expert",
        "/<path>/mixed-small",
        "/<path>/mixed-large",
    ])
    p.add_argument("--tasks_json", type=str, default="../tasks.json")  # task metadata
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)

    # dynamics arch
    p.add_argument("--d_model_dyn", type=int, default=512)
    p.add_argument("--dyn_depth", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--time_every", type=int, default=1)

    p.add_argument("--packing_factor", type=int, default=2)

    # -- flow mode args --
    p.add_argument("--edge_registry",       type=str, default="",
                   help="Path to edge_registry.json (required)")
    p.add_argument("--flow_projector_ckpt", type=str, default="",
                   help="Resume FlowProjector from checkpoint")
    p.add_argument("--freeze_flow_projector", action="store_true", default=False,
                   help="Freeze FlowProjector encoder/decoder and train only dynamics/CFM parameters.")
    p.add_argument("--train_route_cost_head_only", action="store_true", default=False,
                   help="Freeze dynamics plus FlowProjector encoder/decoder and train only the planner route-cost head.")
    p.add_argument("--d_feat",              type=int, default=8,
                   help="Edge feature dimension (default 8)")
    p.add_argument("--flow_task_name", type=str, default="",
                   help="Task name for FlowDataset (used for single data dir or uniform name)")
    p.add_argument("--flow_task_names", type=str, nargs="+", default=None,
                   help="Per-dir task names for FlowDataset (one per --data_dirs entry)")

    p.add_argument("--n_register", type=int, default=4)
    p.add_argument("--n_agent", type=int, default=1)
    p.add_argument("--space_mode", type=str, default="wm_agent_isolated", choices=["wm_agent_isolated", "wm_agent"])

    # latent discretization
    p.add_argument("--k_max", type=int, default=8)

    # actions
    p.add_argument("--use_actions", action="store_true")
    p.add_argument("--joint_route_action", action="store_true",
                   help="Use joint AV+HDV route-fraction encoding (urban routing). Overrides action_type.")
    p.add_argument("--n_routes", type=int, default=4,
                   help="Number of alternative routes per OD pair (K). Used with --joint_route_action.")
    p.add_argument("--action_key", type=str, default="av_planned_edge_flow",
                   help=(
                       "Primary action key in dynamics data .pt. "
                       "Recommended: 'av_planned_edge_flow'. Also supported: "
                       "'joint_action', 'av_action'. Legacy aliases remain readable: "
                       "'av_edge_action'/'action_edge' for older edge-action data, "
                       "and 'action' -> 'joint_action'."
                   ))
    p.add_argument("--action_compress", type=str, default="none", choices=["none", "pca"],
                   help="Optional action compression before training. 'pca' fits/loads linear PCA projector.")
    p.add_argument("--action_pca_dim", type=int, default=32,
                   help="Target PCA dimension when --action_compress pca.")
    p.add_argument("--action_pca_ckpt", type=str, default="",
                   help="Optional path to read/write action compressor checkpoint.")
    p.add_argument("--action_encoder_type", type=str, default="mlp", choices=["mlp", "edge_gnn", "graph_cond"],
                   help="Action encoder architecture for dynamics conditioning.")
    p.add_argument("--action_gnn_layers", type=int, default=2,
                   help="Number of message-passing layers for edge_gnn action encoder.")
    p.add_argument("--action_gnn_d_edge", type=int, default=64,
                   help="Edge hidden dim for edge_gnn action encoder.")
    p.add_argument("--edge_action_injection", action="store_true", default=False,
                   help="Inject raw per-edge action values into matching spatial tokens, masked where action is zero.")
    p.add_argument("--edge_action_injection_eps", type=float, default=1e-6,
                   help="Absolute action threshold below which per-edge action injection is masked to zero.")
    p.add_argument("--edge_action_injection_scale", type=float, default=1.0,
                   help="Multiplier applied to masked per-edge action injection before adding it to spatial tokens.")
    p.add_argument("--edge_action_value_norm", type=str, default="raw", choices=["raw", "active_count"],
                   help="Per-edge action injection value transform. 'active_count' uses [active_mask, action * active_edge_count].")
    p.add_argument("--edge_action_id_embedding", action="store_true", default=False,
                   help="Add a learned edge-id embedding to active per-edge action injections.")
    p.add_argument("--edge_action_tokenizer", action="store_true", default=False,
                   help="Use Dreamer-style edge-indexed continuous action tokenizer for the global ACTION token.")
    p.add_argument("--edge_action_tokenizer_fuse_encoder", action="store_true", default=False,
                   help="When using --edge_action_tokenizer, add tokenizer global token to the normal action encoder token instead of replacing it.")
    p.add_argument("--edge_action_output_injection", action="store_true", default=False,
                   help="Project active per-edge action tokens directly into the CFM velocity output as a residual.")
    p.add_argument("--edge_action_output_from_injection", action="store_true", default=False,
                   help="Allow --edge_action_output_injection to use standard per-edge injection tokens when the tokenizer is disabled.")
    p.add_argument("--action_reconstruction_weight", type=float, default=0.0,
                   help="Weight for reconstructing active edge action mask/value from the transformer ACTION token.")
    p.add_argument("--action_reconstruction_eps", type=float, default=1e-6,
                   help="Absolute action threshold for action reconstruction active mask.")
    p.add_argument("--action_local_latent_weight", type=float, default=0.0,
                   help="Weight for CFM action-local latent delta supervision on active action edges.")
    p.add_argument("--action_local_latent_eps", type=float, default=1e-6,
                   help="Absolute action threshold for CFM action-local latent delta supervision.")
    p.add_argument("--action_volume_delta_weight", type=float, default=0.0,
                   help="Weight for decoded next-prev volume delta supervision on active action edges.")
    p.add_argument("--action_volume_delta_eps", type=float, default=1e-6,
                   help="Absolute action threshold for action-volume delta supervision.")
    p.add_argument("--cond_gnn_hidden_dim", type=int, default=64,
                   help="Hidden dim for graph_cond edge encoders.")
    p.add_argument("--cond_gnn_layers", type=int, default=2,
                   help="Number of graph layers for graph_cond.")
    p.add_argument("--cond_context_hidden_dim", type=int, default=64,
                   help="Hidden dim for graph_cond context MLP.")
    p.add_argument("--aux_action_key", type=str, default="",
                   help="Optional auxiliary action key for HDV edge-distribution supervision (recommended: 'hdv_edge_action').")
    p.add_argument("--reward_key", type=str, default="reward",
                   help="Optional reward key for episode-level supervision (default: reward).")
    p.add_argument("--paired_group_key", type=str, default="pair_id",
                   help="Optional paired/counterfactual group metadata key for A7 data. Ignored when absent.")
    p.add_argument("--paired_variant_key", type=str, default="variant_id",
                   help="Optional paired/counterfactual variant metadata key for A7 data. Ignored when absent.")
    p.add_argument("--hdv_policy_dim", type=int, default=0,
                   help="If >0, enable hdv_policy_head with this output dimension.")
    p.add_argument("--hdv_policy_weight", type=float, default=0.0,
                   help="Weight for auxiliary HDV policy cross-entropy loss (CFM mode only).")
    p.add_argument("--hdv_policy_target_mode", type=str, default="step",
                   choices=["step", "episode_mean", "window_mean", "episode_response", "response"],
                   help=(
                       "Target semantics for hdv_policy_head. step keeps legacy per-step HDV edge occupancy; "
                       "episode_mean/window_mean repeats the sequence mean target at every timestep so the head "
                       "learns an episode-level HDV response background."
                   ))
    p.add_argument("--hdv_policy_action_mode", type=str, default="step",
                   choices=["step", "window_mean", "episode_mean", "episode_response", "response"],
                   help=(
                       "Action condition used only for hdv_policy_head. step keeps the legacy per-step "
                       "conditioning; window_mean/episode_mean/episode_response/response feed an unshifted "
                       "aggregate AV pressure condition to align training with predict_episode_hdv_response."
                   ))
    p.add_argument("--transition_hdv_action_mode", type=str, default="none",
                   choices=["none", "step", "per_step", "raw", "raw_step", "episode_mean", "episode_response", "response"],
                   help=(
                       "If not none, train the transition on a composed edge action "
                       "A'=AV_action + weight*scale*HDV_aux_background. Use episode_response "
                       "with --aux_action_key hdv_edge_action to match MARL's episode-level HDV response rollout input."
                   ))
    p.add_argument("--transition_hdv_action_weight", type=float, default=0.0,
                   help="Multiplier for the HDV auxiliary background added to transition actions.")
    p.add_argument("--transition_hdv_action_scale", type=float, default=1.0,
                   help="Scale applied to the normalized HDV auxiliary background before adding it to transition actions.")
    p.add_argument("--transition_hdv_context_mode", type=str, default="none",
                   choices=["none", "step", "per_step", "raw", "raw_step", "episode_mean", "episode_response", "response"],
                   help=(
                       "If not none, pass HDV auxiliary background as a separate graph_cond context branch. "
                       "Unlike transition_hdv_action_mode, this does not add HDV response into the controllable AV action."
                   ))
    p.add_argument("--transition_hdv_context_weight", type=float, default=0.0,
                   help="Multiplier for the separate HDV response context branch.")
    p.add_argument("--transition_hdv_context_scale", type=float, default=1.0,
                   help="Scale applied to the normalized HDV response context branch.")
    p.add_argument("--decoder_recon_weight", type=float, default=None,
                   help="Weight for FlowProjector.decode reconstruction loss on real flow dynamics.")
    p.add_argument("--inverse_recon_weight", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--d_obs", type=int, default=0,
                   help="If >0, create obs_proj (d_spatial→d_obs) in FlowProjector for latent obs.")
    p.add_argument("--obs_proj_weight", type=float, default=0.2,
                   help="Weight for obs_proj reconstruction loss (requires --d_obs >0).")
    p.add_argument("--decoder_recon_speed_weight", type=float, default=None,
                   help="Extra per-channel weight for speed reconstruction in decoder reconstruction loss.")
    p.add_argument("--inverse_speed_weight", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--decoder_recon_speed_index", type=int, default=None,
                   help="Channel index treated as speed in decoder reconstruction target (default 1).")
    p.add_argument("--inverse_speed_index", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--decoded_speed_target", type=str, default="slowdown",
                   choices=["norm", "deficit", "slowdown"],
                   help="Decoded speed supervision target: raw normalized speed, "
                        "speed deficit (1-speed_norm), or slowdown (1/speed_norm - 1).")
    p.add_argument("--decoded_volume_target", type=str, default="log1p",
                   choices=["norm", "log1p"],
                   help="Decoded volume supervision target: raw veh/cap or log1p(volume). "
                        "When capacities are available, log1p uses per-edge vehicle counts.")
    p.add_argument("--decoded_busy_threshold", type=float, default=0.05,
                   help="Busy-edge threshold on raw decoded veh/cap used for masked speed/volume diagnostics.")
    p.add_argument("--decoded_busy_presence_margin", type=float, default=0.05,
                   help="Margin for busy-edge presence BCE logits: (pred_volume - threshold) / margin.")
    p.add_argument("--decoded_busy_volume_weight", type=float, default=0.0,
                   help="Global fallback weight on busy-edge volume MSE in decoded supervision losses.")
    p.add_argument("--decoded_busy_presence_weight", type=float, default=0.0,
                   help="Global fallback weight on target-busy presence BCE in decoder-side supervision losses.")
    p.add_argument("--decoded_topk_rank_weight", type=float, default=0.0,
                   help="Global fallback weight on pairwise top-k busy-edge ranking in decoder-side supervision losses.")
    p.add_argument("--decoded_topk_rank_k", type=int, default=20,
                   help="Number of target-busy positives and hard negatives used by decoder-side top-k ranking.")
    p.add_argument("--decoded_topk_rank_margin", type=float, default=0.05,
                   help="Soft margin for decoder-side top-k ranking loss in raw veh/cap units.")
    p.add_argument("--decoded_zero_volume_weight", type=float, default=0.0,
                   help="Global fallback weight for true-zero edge false-alarm volume penalty.")
    p.add_argument("--decoded_zero_volume_threshold", type=float, default=0.05,
                   help="Predicted raw volume threshold above which true-zero edges are penalized.")
    p.add_argument("--decoded_zero_volume_margin", type=float, default=0.05,
                   help="Soft margin for true-zero false-alarm penalty around --decoded_zero_volume_threshold.")
    p.add_argument("--decoded_zero_target_threshold", type=float, default=0.0,
                   help="Target raw volume threshold defining zero edges for false-alarm penalty.")
    p.add_argument("--decoder_recon_busy_volume_weight", type=float, default=None,
                   help="If set, overrides the busy-edge volume weight for decoder reconstruction supervision.")
    p.add_argument("--decoder_recon_busy_presence_weight", type=float, default=None,
                   help="If set, overrides the busy-edge presence BCE weight for decoder reconstruction supervision.")
    p.add_argument("--decoder_recon_topk_rank_weight", type=float, default=None,
                   help="If set, overrides the top-k busy-edge ranking weight for decoder reconstruction supervision.")
    p.add_argument("--decoder_recon_zero_volume_weight", type=float, default=None,
                   help="If set, overrides true-zero false-alarm penalty for decoder reconstruction supervision.")
    p.add_argument("--inverse_busy_volume_weight", type=float, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--decoder_busy_volume_weight", type=float, default=None,
                   help="If set, overrides the busy-edge volume weight for decoder multi-head supervision.")
    p.add_argument("--decoder_busy_presence_weight", type=float, default=None,
                   help="If set, overrides the busy-edge presence BCE weight for decoder multi-head supervision.")
    p.add_argument("--decoder_topk_rank_weight", type=float, default=None,
                   help="If set, overrides the top-k busy-edge ranking weight for decoder multi-head supervision.")
    p.add_argument("--decoder_zero_volume_weight", type=float, default=None,
                   help="If set, overrides true-zero false-alarm penalty for decoder multi-head supervision.")
    p.add_argument("--transition_busy_volume_weight", type=float, default=None,
                   help="If set, overrides the busy-edge volume weight for transition decoded supervision.")
    p.add_argument("--transition_busy_presence_weight", type=float, default=0.0,
                   help="Weight for target-busy edge presence BCE in transition decoded supervision.")
    p.add_argument("--transition_topk_rank_weight", type=float, default=0.0,
                   help="Weight for pairwise top-k busy-edge ranking loss in transition decoded supervision.")
    p.add_argument("--transition_topk_rank_k", type=int, default=20,
                   help="Number of target-busy positives and hard negatives used by transition top-k ranking loss.")
    p.add_argument("--transition_topk_rank_margin", type=float, default=0.05,
                   help="Soft margin for transition top-k ranking loss in raw veh/cap units.")
    p.add_argument("--transition_zero_volume_weight", type=float, default=None,
                   help="If set, overrides true-zero false-alarm penalty for transition decoded supervision.")
    p.add_argument("--transition_cfm_init_mode", type=str, default="randn", choices=["randn", "zero"],
                   help="Initial source state used by differentiable transition decoded supervision.")
    p.add_argument("--transition_action_contrast_weight", type=float, default=0.0,
                   help="Weight for contrastive loss requiring real-action transition error to beat shuffled-action error.")
    p.add_argument("--transition_action_contrast_margin", type=float, default=0.05,
                   help="Margin for transition action-contrast loss.")
    p.add_argument("--transition_action_contrast_metric", type=str, default="total",
                   choices=["total", "volume", "active_volume", "busy_volume"],
                   help="Decoded metric used by transition action-contrast loss; 'total' preserves the original behavior.")
    p.add_argument("--cfm_active_weight", type=float, default=0.0,
                   help="Extra edge weight added to CFM velocity loss where true target volume is active.")
    p.add_argument("--cfm_busy_weight", type=float, default=0.0,
                   help="Extra edge weight added to CFM velocity loss where true target volume exceeds --cfm_busy_threshold.")
    p.add_argument("--cfm_busy_threshold", type=float, default=0.05,
                   help="True target volume threshold for busy-edge CFM velocity weighting.")
    p.add_argument("--cfm_active_threshold", type=float, default=0.0,
                   help="True target volume threshold for active-edge CFM velocity weighting.")
    p.add_argument("--cfm_max_weight", type=float, default=50.0,
                   help="Maximum per-edge CFM velocity weight; set <=0 to disable clipping.")
    p.add_argument("--episode_steps", type=int, default=60,
                   help="Number of decision steps per episode (for fixed-Δt data collection).")
    p.add_argument("--scheduled_sampling", action="store_true", default=False,
                   help="Enable scheduled sampling to reduce autoregressive distribution shift.")
    p.add_argument("--sched_sample_end", type=float, default=0.5,
                   help="Final scheduled sampling probability.")
    p.add_argument("--sched_sample_warmup", type=int, default=3000,
                   help="Steps to linearly ramp scheduled sampling probability.")
    p.add_argument("--bpr_supervision_weight", type=float, default=0.0,
                   help="Weight for BPR consistency supervision based on FlowProjector.decode outputs.")
    p.add_argument("--spatial_supervision_weight", type=float, default=0.0,
                   help="Weight for one-step busy-edge spatial supervision based on FlowProjector.decode outputs.")
    p.add_argument("--transition_decoded_supervision_weight", type=float, default=0.0,
                   help="Weight for one-step decoded supervision on the terminal predicted next latent state.")
    p.add_argument("--rollout_horizon", type=int, default=1,
                   help="Teacher-forcing rollout horizon for multi-step CFM loss (1 = single-step baseline).")
    p.add_argument("--rollout_gamma", type=float, default=0.7,
                   help="Geometric decay factor for weighting farther rollout steps.")
    p.add_argument("--adm_cfm_horizons", type=str, default="",
                   help="Comma-separated ADM-style direct k-step horizons, e.g. '2,4,8'. Disabled by default.")
    p.add_argument("--adm_cfm_weight", type=float, default=0.0,
                   help="Weight for ADM-style direct k-step CFM velocity supervision.")
    p.add_argument("--adm_cfm_gamma", type=float, default=0.7,
                   help="Geometric horizon decay for ADM direct k-step losses.")
    p.add_argument("--adm_decoded_supervision_weight", type=float, default=0.0,
                   help="Weight for decoded supervision on ADM direct k-step sampled states.")
    p.add_argument("--adm_max_windows", type=int, default=0,
                   help="If >0, randomly subsample this many direct k-step windows per horizon to cap memory.")
    p.add_argument("--adm_cfm_init_mode", type=str, default="zero", choices=["randn", "zero"],
                   help="Initial source state for ADM direct k-step CFM sampling/supervision.")
    p.add_argument("--decoder_full_weight", type=float, default=1.0,
                   help="Weight for full decoded-channel reconstruction loss on FlowProjector.decode outputs.")
    p.add_argument("--decoder_speed_weight", type=float, default=1.0,
                   help="Extra per-head weight for the decoded speed channel in decoder supervision.")
    p.add_argument("--decoder_volume_weight", type=float, default=1.5,
                   help="Extra per-head weight for the decoded volume/veh channel in decoder supervision.")
    p.add_argument("--decoder_other_weight", type=float, default=1.0,
                   help="Extra per-head weight for decoded non-speed/non-volume channels in decoder supervision.")
    p.add_argument("--spatial_busy_threshold", type=float, default=0.05,
                   help="Decoded volume threshold used to define busy/active edges for spatial supervision.")
    p.add_argument("--bpr_reward_weight", type=float, default=1.0,
                   help="Deprecated compatibility flag. Episode-level BPR reward MSE is no longer used.")
    p.add_argument("--bpr_alpha", type=float, default=0.15,
                   help="BPR alpha used in decoder BPR supervision.")
    p.add_argument("--bpr_beta", type=float, default=4.0,
                   help="BPR beta used in decoder BPR supervision.")
    p.add_argument("--bpr_veh_index", type=int, default=2,
                   help="Decoded dynamic channel index interpreted as veh/cap for BPR supervision.")
    p.add_argument("--bpr_target_mode", type=str, default="relative_excess",
                   choices=["raw_seconds", "relative_excess", "log_relative_excess"],
                   help="Comparison space for BPR supervision: raw edge TT in seconds, "
                        "or normalized excess delay relative to free-flow.")
    p.add_argument("--route_ranking_weight", type=float, default=0.0,
                   help="Weight for decoder-side route-cost ranking supervision using --route_registry candidates.")
    p.add_argument("--transition_route_ranking_weight", type=float, default=0.0,
                   help="Weight for transition-predicted next-state route-cost ranking supervision.")
    p.add_argument("--route_ranking_routes_csv", type=str, default="",
                   help="Optional routes.csv source for route-ranking supervision; preferred for current RouteRL data.")
    p.add_argument("--route_ranking_max_od", type=int, default=64,
                   help="Maximum OD route groups sampled per batch for route-ranking supervision; <=0 uses all.")
    p.add_argument("--route_ranking_temperature", type=float, default=30.0,
                   help="Softmin temperature in seconds for route-ranking supervision targets.")
    p.add_argument("--route_ranking_margin_weight", type=float, default=0.0,
                   help="Optional pairwise margin term weight inside route-ranking supervision.")
    p.add_argument("--route_ranking_margin", type=float, default=1.0,
                   help="Pairwise route-ranking margin in route-cost seconds.")
    p.add_argument("--route_ranking_gap_weight", type=float, default=0.0,
                   help="Weight for route0-relative route-cost gap regression inside route-ranking supervision.")
    p.add_argument("--route_ranking_non0_weight", type=float, default=1.0,
                   help="Row weight multiplier for route-ranking samples whose true best route is non-zero.")
    p.add_argument("--route_ranking_speed_min_ratio", type=float, default=0.05,
                   help="Minimum speed/speed_limit ratio for speed-mode route-ranking costs.")
    p.add_argument("--route_ranking_speed_max_ratio", type=float, default=1.50,
                   help="Maximum speed/speed_limit ratio for speed-mode route-ranking costs.")
    p.add_argument("--route_ranking_speed_excess_scale", type=float, default=1.0,
                   help="Blend factor for speed-mode excess delay: 1.0 gives free_flow/speed_ratio; lower values damp global low-speed bias.")
    p.add_argument("--route_ranking_length_norm", type=str, default="sum", choices=["sum", "mean"],
                   help="Whether route-ranking cost sums or averages edge travel times along each route.")
    p.add_argument("--route_ranking_target_cost", type=str, default="speed", choices=["bpr_volume", "speed"],
                   help="Route-ranking label cost: BPR from true volume, or free-flow/speed_norm from true speed channel.")
    p.add_argument("--route_distill_pt", type=str, default="",
                   help="Optional state-aligned dynamic route-cost distillation artifact from tmp/build_state_aligned_route_distill.py.")
    p.add_argument("--route_distill_weight", type=float, default=0.0,
                   help="Weight for decoder-side state-aligned dynamic route-cost distillation.")
    p.add_argument("--route_distill_batch_size", type=int, default=64,
                   help="Auxiliary batch size for state-aligned route distillation samples.")
    p.add_argument("--route_distill_temperature", type=float, default=30.0,
                   help="Softmin/gap normalization temperature in seconds for route distillation.")
    p.add_argument("--route_distill_gap_weight", type=float, default=0.0,
                   help="Weight for route0-relative route-cost gap regression inside route distillation.")
    p.add_argument("--route_distill_cost_weight", type=float, default=0.0,
                   help="Weight for absolute live route-cost regression inside route distillation.")
    p.add_argument("--route_distill_non0_weight", type=float, default=1.0,
                   help="Row weight multiplier for distillation rows whose live best route is non-zero.")
    p.add_argument("--route_distill_pred_cost", type=str, default="speed", choices=["bpr_volume", "speed", "route_head"],
                   help="Predicted edge-cost source used for route distillation. route_head uses a planner-cost adapter instead of decoder speed/volume.")
    p.add_argument("--route_distill_length_norm", type=str, default="sum", choices=["sum", "mean"],
                   help="Whether route distillation costs sum or average edge travel times along each dynamic route.")
    p.add_argument("--enable_t_all_head", action="store_true",
                   help="Enable projector t_all prediction head even if supervision weight is zero.")
    p.add_argument("--t_all_supervision_weight", type=float, default=0.0,
                   help="Weight for episode-level t_all regression loss using reward labels (reward=-t_all).")
    p.add_argument("--t_all_missing_reward", type=float, default=0.0,
                   help="Reward value treated as missing label in t_all supervision (default 0.0).")
    p.add_argument("--t_all_target_mode", type=str, default="relative",
                   choices=["raw_minutes", "log1p", "relative"],
                   help="Comparison space for t_all supervision: raw minutes, "
                        "log1p(minutes), or batch-mean-normalized relative minutes.")

    # Stackelberg equilibrium layer
    p.add_argument("--use_stackelberg", action="store_true",
                   help="Enable Stackelberg equilibrium layer (replaces hdv_policy_head).")
    p.add_argument("--stackelberg_n_iters", type=int, default=10,
                   help="Number of logit dynamics iterations in Stackelberg solver.")
    p.add_argument("--stackelberg_damping", type=float, default=0.5,
                   help="Damping factor for Stackelberg logit dynamics.")
    p.add_argument("--stackelberg_total_av_count", type=float, default=0.0,
                   help="Total AV count for edge-level Stackelberg scaling. Overrides route_registry-derived count if >0.")
    p.add_argument("--stackelberg_total_hdv_count", type=float, default=0.0,
                   help="Total HDV count for edge-level Stackelberg scaling. Overrides route_registry-derived count if >0.")
    p.add_argument("--stackelberg_hdv_weight", type=float, default=0.1,
                   help="Weight for Stackelberg HDV CE auxiliary loss.")
    p.add_argument("--stackelberg_hdv_weight_decay", type=float, default=0.99,
                   help="Per-epoch decay for stackelberg_hdv_weight.")
    p.add_argument("--stackelberg_eq_weight", type=float, default=0.01,
                   help="Weight for Stackelberg fixed-point residual (L_eq) loss.")
    p.add_argument("--route_registry", type=str, default="",
                   help="Optional path to route_registry.json used only to derive total AV/HDV counts for Stackelberg.")

    # PCFM (physics-constrained flow matching) — inference-time only
    p.add_argument("--use_pcfm", action="store_true",
                   help="Enable physics-constrained CFM sampling during evaluation.")
    p.add_argument("--pcfm_lambda_bpr", type=float, default=0.05,
                   help="PCFM BPR guidance strength.")
    p.add_argument("--pcfm_lambda_kirchhoff", type=float, default=0.03,
                   help="PCFM Kirchhoff conservation guidance strength.")
    p.add_argument("--pcfm_capacity_kappa", type=float, default=1.2,
                   help="PCFM max veh/cap ratio for hard constraints.")
    p.add_argument("--pcfm_n_ode_steps", type=int, default=4,
                   help="Number of ODE steps for PCFM sampling.")

    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=10_000_000)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # eval / viz
    p.add_argument("--eval_every", type=int, default=1_000)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--eval_max_items", type=int, default=4)
    p.add_argument("--eval_ctx", type=int, default=8)
    p.add_argument("--eval_horizon", type=int, default=16)

    # logging
    p.add_argument("--log_every", type=int, default=200)

    # wandb
    p.add_argument("--wandb_project", type=str, default="dreamer4-dynamics")
    p.add_argument("--wandb_run_name", type=str, default="default")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                   choices=["online", "offline", "disabled"],
                   help="Weights & Biases mode. Respects WANDB_MODE env by default.")

    # ckpt
    p.add_argument("--ckpt_dir", type=str, default="./logs/dynamics_ckpts")
    p.add_argument("--save_every", type=int, default=10_000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--resume_non_strict", action="store_true",
                   help="Allow partial checkpoint loading (ignore missing/unexpected keys).")
    p.add_argument("--resume_weights_only", action="store_true",
                   help="Load model/projector weights only, reset optimizer/scaler/step/epoch.")

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--n_ode_steps", type=int, default=1,
                   help="ODE integration steps for CFM sampling/evaluation (1=Euler, 2=midpoint-like).")
    p.add_argument("--ddp_barrier", action="store_true",
                   help="Enable explicit DDP barriers after rank0-only logging/checkpoint/finalization.")

    settings_path = parse_settings_arg()
    apply_stage_defaults(p, settings_path, stage="dynamics")
    train(p.parse_args())
