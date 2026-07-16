#!/usr/bin/env python3
"""Ablate HDV context variants for one-step WM transition prediction.

This script answers a narrow question: given a trained world model and an
offline flow dataset, does the transition predictor become more accurate when
we provide different HDV edge-pressure contexts?

The available offline ``*.pt`` datasets store per-decision HDV current-edge
occupancy, not the episode-start static route decision.  The variants here
therefore evaluate current-occupancy-derived contexts:

* no/zero HDV context
* global episode-mean current occupancy
* previous/EMA episode-mean current occupancy
* oracle episode-mean current occupancy
* oracle current-step occupancy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "dreamer4") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "dreamer4"))

from dreamer4.dynamics_losses import transition_decoded_supervision_loss
from dreamer4.train_dynamics import cfm_predict_one_timestep
from dreamer4.urban_routing.urban_world_model import UrbanWorldModel


def _normalise_rows_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).clip(min=0.0)
    if arr.ndim == 1:
        denom = float(arr.sum())
        if denom <= eps:
            return np.zeros_like(arr, dtype=np.float64)
        return arr / denom
    denom = arr.sum(axis=-1, keepdims=True)
    return np.divide(arr, np.maximum(denom, eps), out=np.zeros_like(arr, dtype=np.float64))


def _causal_pair(current: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.zeros_like(current), current], dim=1)


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_numpy_2d(value: object, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"dataset is missing required field {name!r}")
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim != 2:
        raise RuntimeError(f"dataset field {name!r} must be 2D, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _as_numpy_1d(value: object, name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"dataset is missing required field {name!r}")
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim != 1:
        raise RuntimeError(f"dataset field {name!r} must be 1D, got shape {arr.shape}")
    return arr


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _infer_flow_dir(dataset_pt: Path) -> Path:
    stem = dataset_pt.stem
    candidates = [
        dataset_pt.parent / "flows" / stem,
        dataset_pt.parent.parent / "flows" / stem,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "could not infer flow shard directory; tried "
        + ", ".join(str(p) for p in candidates)
    )


def _load_flow_shards(flow_dir: Path) -> torch.Tensor:
    shards = sorted(flow_dir.glob("shard_*.pt"))
    if not shards:
        raise RuntimeError(f"no shard_*.pt files found under {flow_dir}")
    chunks: List[torch.Tensor] = []
    for shard in shards:
        obj = torch.load(shard, map_location="cpu", weights_only=False)
        flows = obj.get("flows") if isinstance(obj, dict) else obj
        if not isinstance(flows, torch.Tensor):
            flows = torch.as_tensor(flows)
        if flows.dim() != 3:
            raise RuntimeError(f"flow shard {shard} must be 3D, got {tuple(flows.shape)}")
        chunks.append(flows.to(dtype=torch.float32, device="cpu"))
    return torch.cat(chunks, dim=0)


@dataclass
class TransitionDataset:
    flows: torch.Tensor
    episode: np.ndarray
    av_action: np.ndarray
    hdv_step: np.ndarray
    depart: np.ndarray
    context: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(self.hdv_step.shape[1])


def _load_transition_dataset(dataset_pt: Path, flow_dir: Optional[Path]) -> TransitionDataset:
    raw = torch.load(dataset_pt, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected dict dataset in {dataset_pt}")
    flows = _load_flow_shards(flow_dir or _infer_flow_dir(dataset_pt))
    episode = _as_numpy_1d(raw.get("episode"), "episode").astype(np.int64, copy=False)
    av_action = _as_numpy_2d(raw.get("av_edge_action"), "av_edge_action")
    hdv_step = _as_numpy_2d(raw.get("hdv_edge_action"), "hdv_edge_action")
    depart = _as_numpy_2d(raw.get("depart"), "depart")
    context = _as_numpy_2d(raw.get("context"), "context")
    n = min(
        int(flows.shape[0]),
        int(episode.shape[0]),
        int(av_action.shape[0]),
        int(hdv_step.shape[0]),
        int(depart.shape[0]),
        int(context.shape[0]),
    )
    if n < 2:
        raise RuntimeError(f"dataset has too few aligned rows: {n}")
    return TransitionDataset(
        flows=flows[:n].contiguous(),
        episode=episode[:n],
        av_action=av_action[:n],
        hdv_step=hdv_step[:n],
        depart=depart[:n],
        context=context[:n],
    )


def _episode_means(episode: np.ndarray, hdv_step: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for ep in sorted(int(e) for e in np.unique(episode)):
        mask = episode == ep
        if not bool(mask.any()):
            continue
        out[ep] = _normalise_rows_np(hdv_step[mask].mean(axis=0))
    return out


@dataclass
class SplitInfo:
    transition_idx: np.ndarray
    train_episodes: List[int]
    test_episodes: List[int]
    cutoff_episode: Optional[int]


def _select_transitions(
    episode: np.ndarray,
    *,
    split: str,
    test_start_episode: Optional[int],
    test_fraction: float,
    seed: int,
    max_transitions: int,
) -> SplitInfo:
    all_idx = np.flatnonzero(episode[:-1] == episode[1:])
    if all_idx.size == 0:
        raise RuntimeError("no adjacent same-episode transitions found")
    all_eps = sorted(int(e) for e in np.unique(episode))
    split = str(split).strip().lower()
    cutoff_episode: Optional[int] = None
    if split == "all":
        train_eps = all_eps
        test_eps = sorted(int(e) for e in np.unique(episode[all_idx]))
        selected = all_idx
    elif split == "chronological":
        if test_start_episode is None:
            pos = int(round(len(all_eps) * (1.0 - float(test_fraction))))
            pos = min(max(pos, 1), max(len(all_eps) - 1, 1))
            cutoff_episode = all_eps[pos]
        else:
            cutoff_episode = int(test_start_episode)
        train_eps = [ep for ep in all_eps if ep < int(cutoff_episode)]
        test_eps = [ep for ep in all_eps if ep >= int(cutoff_episode)]
        selected = all_idx[np.isin(episode[all_idx], np.asarray(test_eps, dtype=np.int64))]
    elif split == "random":
        rng = random.Random(int(seed))
        candidates = sorted(int(e) for e in np.unique(episode[all_idx]))
        rng.shuffle(candidates)
        n_test = max(1, int(round(len(candidates) * float(test_fraction))))
        test_eps = sorted(candidates[:n_test])
        train_eps = sorted(ep for ep in all_eps if ep not in set(test_eps))
        selected = all_idx[np.isin(episode[all_idx], np.asarray(test_eps, dtype=np.int64))]
    else:
        raise ValueError(f"unsupported split={split!r}; expected all, chronological, or random")
    if selected.size == 0:
        raise RuntimeError(f"split {split!r} selected no transitions")
    if max_transitions > 0 and selected.size > int(max_transitions):
        rng = np.random.default_rng(int(seed))
        selected = np.sort(rng.choice(selected, size=int(max_transitions), replace=False))
    return SplitInfo(
        transition_idx=np.asarray(selected, dtype=np.int64),
        train_episodes=list(train_eps),
        test_episodes=list(test_eps),
        cutoff_episode=cutoff_episode,
    )


def _make_online_episode_predictions(
    episode_mean: Dict[int, np.ndarray],
    global_mean: np.ndarray,
    *,
    ema_alpha: float,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    previous: Dict[int, np.ndarray] = {}
    ema_pred: Dict[int, np.ndarray] = {}
    last_ep: Optional[int] = None
    ema: Optional[np.ndarray] = None
    for ep in sorted(episode_mean):
        previous[ep] = episode_mean[last_ep].copy() if last_ep is not None else global_mean.copy()
        ema_pred[ep] = ema.copy() if ema is not None else global_mean.copy()
        current = episode_mean[ep]
        if ema is None:
            ema = current.copy()
        else:
            alpha = float(ema_alpha)
            ema = alpha * current + (1.0 - alpha) * ema
            ema = _normalise_rows_np(ema)
        last_ep = ep
    return previous, ema_pred


def _checkpoint_paths(
    checkpoint: Optional[Path],
    projector_ckpt: Optional[Path],
    dynamics_ckpt: Optional[Path],
) -> Tuple[Path, Path, Dict[str, object]]:
    cfg: Dict[str, object] = {}
    projector = projector_ckpt
    dynamics = dynamics_ckpt
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(f"checkpoint is not a dict: {checkpoint}")
        cfg_obj = state.get("cfg", {})
        cfg = cfg_obj if isinstance(cfg_obj, dict) else {}
        if projector is None:
            value = state.get("wm_projector_ckpt")
            projector = Path(value) if isinstance(value, str) and value else None
        if dynamics is None:
            value = state.get("wm_dynamics_ckpt")
            dynamics = Path(value) if isinstance(value, str) and value else None
    if projector is None or dynamics is None:
        raise RuntimeError("provide --checkpoint or both --projector-ckpt and --dynamics-ckpt")
    return _resolve_path(projector), _resolve_path(dynamics), cfg


def _build_context_variants(
    ds: TransitionDataset,
    split: SplitInfo,
    *,
    ema_alpha: float,
    context_weight: float,
    context_scale: float,
    seed: int,
) -> Dict[str, Optional[np.ndarray]]:
    idx = split.transition_idx
    start_eps = ds.episode[idx].astype(np.int64, copy=False)
    hdv_norm_step = _normalise_rows_np(ds.hdv_step)
    ep_mean = _episode_means(ds.episode, ds.hdv_step)
    train_eps = [ep for ep in split.train_episodes if ep in ep_mean]
    if train_eps:
        global_mean = _normalise_rows_np(np.mean([ep_mean[ep] for ep in train_eps], axis=0))
    else:
        global_mean = _normalise_rows_np(np.mean(list(ep_mean.values()), axis=0))
    previous, ema_pred = _make_online_episode_predictions(
        ep_mean,
        global_mean,
        ema_alpha=float(ema_alpha),
    )

    def by_episode(mapping: Dict[int, np.ndarray]) -> np.ndarray:
        rows = [mapping.get(int(ep), global_mean) for ep in start_eps]
        return np.asarray(rows, dtype=np.float32)

    rng = np.random.default_rng(int(seed))
    ep_bank = np.asarray([ep_mean[ep] for ep in sorted(ep_mean)], dtype=np.float32)
    shuffled = ep_bank[rng.integers(low=0, high=int(ep_bank.shape[0]), size=int(idx.size))]

    scale = float(context_weight) * float(context_scale)
    variants: Dict[str, Optional[np.ndarray]] = {
        "no_hdv_context": None,
        "zero_hdv_context": np.zeros((int(idx.size), ds.n_edges), dtype=np.float32),
        "global_episode_mean_current_occupancy": np.tile(global_mean, (int(idx.size), 1)).astype(np.float32),
        "previous_episode_mean_current_occupancy": by_episode(previous),
        "ema_episode_mean_current_occupancy": by_episode(ema_pred),
        "shuffled_episode_mean_current_occupancy": shuffled.astype(np.float32),
        "oracle_episode_mean_current_occupancy": by_episode(ep_mean),
        "oracle_step_current_occupancy": hdv_norm_step[idx].astype(np.float32),
    }
    if abs(scale - 1.0) > 1e-12:
        for key, value in list(variants.items()):
            if value is not None:
                variants[key] = (value * scale).astype(np.float32)
    return variants


@torch.inference_mode()
def _evaluate_variant(
    wm: UrbanWorldModel,
    ds: TransitionDataset,
    idx: np.ndarray,
    hdv_context: Optional[np.ndarray],
    *,
    batch_size: int,
    cfm_init_mode: str,
    supervision_weight: float,
    speed_weight: float,
    volume_weight: float,
    busy_threshold: float,
) -> Dict[str, float]:
    device = wm.device
    flow_projector = wm.flow_projector
    dynamics = wm.dynamics
    flow_projector.eval()
    dynamics.eval()
    sums: Dict[str, float] = {}
    count = 0
    n = int(idx.size)
    for start in range(0, n, int(batch_size)):
        batch_idx = idx[start : start + int(batch_size)]
        b = int(batch_idx.size)
        flow_pair = torch.stack(
            [ds.flows[batch_idx], ds.flows[batch_idx + 1]],
            dim=1,
        ).to(device=device, dtype=torch.float32)
        action = torch.as_tensor(ds.av_action[batch_idx], device=device, dtype=torch.float32)
        depart = torch.as_tensor(ds.depart[batch_idx], device=device, dtype=torch.float32)
        context = torch.as_tensor(ds.context[batch_idx], device=device, dtype=torch.float32)
        hdv = None
        if hdv_context is not None:
            hdv = torch.as_tensor(hdv_context[start : start + b], device=device, dtype=torch.float32)

        z_pair = flow_projector(flow_pair)
        model_actions = wm._prepare_model_actions(
            batch_size=b,
            seq_len=2,
            action=_causal_pair(action),
            depart=_causal_pair(depart),
            context=_causal_pair(context),
            hdv_context=_causal_pair(hdv) if hdv is not None and bool(getattr(wm, "use_hdv_context", False)) else None,
            device=device,
            dtype=z_pair.dtype,
        )
        initial_state = None
        if str(cfm_init_mode).strip().lower() == "zero":
            initial_state = torch.zeros(
                (b, 1, int(z_pair.shape[2]), int(z_pair.shape[3])),
                device=device,
                dtype=z_pair.dtype,
            )
        z_pred_next = cfm_predict_one_timestep(
            dynamics,
            past_packed=z_pair[:, :-1],
            actions=model_actions,
            act_mask=None,
            n_steps=int(getattr(wm, "n_ode_steps", 1)),
            initial_state=initial_state,
            init_mode=str(cfm_init_mode),
        )
        loss, aux = transition_decoded_supervision_loss(
            z_pred_next=z_pred_next,
            flow_projector=flow_projector,
            flow_target_next=flow_pair[:, -1],
            supervision_weight=float(supervision_weight),
            speed_index=1,
            volume_index=2,
            speed_weight=float(speed_weight),
            volume_weight=float(volume_weight),
            busy_threshold=float(busy_threshold),
        )
        metrics = {
            "transition_loss": loss.detach(),
            "latent_mse": (z_pred_next - z_pair[:, -1]).pow(2).mean().detach(),
        }
        for key, value in aux.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metrics[key] = value.detach()
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item()) * b
        count += b
    if count <= 0:
        raise RuntimeError("no batches evaluated")
    return {key: value / float(count) for key, value in sorted(sums.items())}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pt", required=True, help="Offline decision dataset .pt path")
    parser.add_argument("--flow-dir", default=None, help="Flow shard directory; inferred from --dataset-pt if omitted")
    parser.add_argument("--edge-registry", required=True, help="Edge registry JSON used by the WM checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Online checkpoint containing wm_projector_ckpt and wm_dynamics_ckpt")
    parser.add_argument("--projector-ckpt", default=None, help="Explicit WM flow projector checkpoint")
    parser.add_argument("--dynamics-ckpt", default=None, help="Explicit WM dynamics checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cfm-init-mode", default=None, choices=["randn", "zero", "current"])
    parser.add_argument("--split", default="chronological", choices=["all", "chronological", "random"])
    parser.add_argument("--test-start-episode", type=int, default=None)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--max-transitions", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument("--supervision-weight", type=float, default=1.0)
    parser.add_argument("--speed-weight", type=float, default=4.0)
    parser.add_argument("--volume-weight", type=float, default=2.0)
    parser.add_argument("--busy-threshold", type=float, default=0.05)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_pt = _resolve_path(args.dataset_pt)
    flow_dir = _resolve_path(args.flow_dir) if args.flow_dir else None
    edge_registry = _resolve_path(args.edge_registry)
    checkpoint = _resolve_path(args.checkpoint) if args.checkpoint else None
    projector_arg = _resolve_path(args.projector_ckpt) if args.projector_ckpt else None
    dynamics_arg = _resolve_path(args.dynamics_ckpt) if args.dynamics_ckpt else None

    projector_ckpt, dynamics_ckpt, cfg = _checkpoint_paths(checkpoint, projector_arg, dynamics_arg)
    cfm_init_mode = args.cfm_init_mode or str(cfg.get("wm_cfm_init_mode", "current") or "current")
    context_weight = _safe_float(cfg.get("wm_transition_hdv_context_weight"), 1.0)
    context_scale = _safe_float(cfg.get("wm_transition_hdv_context_scale"), 1.0)

    ds = _load_transition_dataset(dataset_pt, flow_dir)
    split = _select_transitions(
        ds.episode,
        split=str(args.split),
        test_start_episode=args.test_start_episode,
        test_fraction=float(args.test_fraction),
        seed=int(args.seed),
        max_transitions=int(args.max_transitions),
    )

    wm = UrbanWorldModel(
        flow_projector_ckpt=str(projector_ckpt),
        dynamics_ckpt=str(dynamics_ckpt),
        edge_registry_path=str(edge_registry),
        device=str(args.device),
        use_amp=False,
        cfm_init_mode=str(cfm_init_mode),
    )
    variants = _build_context_variants(
        ds,
        split,
        ema_alpha=float(args.ema_alpha),
        context_weight=context_weight,
        context_scale=context_scale,
        seed=int(args.seed),
    )

    results: Dict[str, object] = {
        "dataset_pt": str(dataset_pt),
        "flow_dir": str(flow_dir or _infer_flow_dir(dataset_pt)),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "projector_ckpt": str(projector_ckpt),
        "dynamics_ckpt": str(dynamics_ckpt),
        "edge_registry": str(edge_registry),
        "device": str(args.device),
        "cfm_init_mode": str(cfm_init_mode),
        "wm_use_hdv_context": bool(getattr(wm, "use_hdv_context", False)),
        "wm_n_ode_steps": int(getattr(wm, "n_ode_steps", 1)),
        "context_weight": float(context_weight),
        "context_scale": float(context_scale),
        "split": str(args.split),
        "cutoff_episode": split.cutoff_episode,
        "n_rows": int(ds.flows.shape[0]),
        "n_edges": int(ds.n_edges),
        "n_train_episodes": int(len(split.train_episodes)),
        "n_test_episodes": int(len(split.test_episodes)),
        "n_transitions": int(split.transition_idx.size),
        "test_episode_min": int(min(split.test_episodes)) if split.test_episodes else None,
        "test_episode_max": int(max(split.test_episodes)) if split.test_episodes else None,
        "variants": {},
    }

    for name, context in variants.items():
        metrics = _evaluate_variant(
            wm,
            ds,
            split.transition_idx,
            context,
            batch_size=int(args.batch_size),
            cfm_init_mode=str(cfm_init_mode),
            supervision_weight=float(args.supervision_weight),
            speed_weight=float(args.speed_weight),
            volume_weight=float(args.volume_weight),
            busy_threshold=float(args.busy_threshold),
        )
        results["variants"][name] = metrics
        print(
            f"{name}: loss={metrics.get('transition_loss', float('nan')):.8f} "
            f"weighted_mse={metrics.get('pred_next_decoded_weighted_mse', float('nan')):.8f} "
            f"speed_mse={metrics.get('pred_next_decoded_speed_mse', float('nan')):.8f} "
            f"volume_mse={metrics.get('pred_next_decoded_volume_mse', float('nan')):.8f} "
            f"latent_mse={metrics.get('latent_mse', float('nan')):.8f}"
        )

    if args.output_json:
        output_json = _resolve_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"wrote {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
