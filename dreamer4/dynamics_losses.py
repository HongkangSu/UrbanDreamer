"""Supervision and auxiliary loss helpers for dynamics training."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


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
