# dreamer4/flow_dataset.py
"""
FlowDataset: world-model dataset for road-segment flow data.

Mirrors WMDataset but operates on flow shards saved as
    {"flows": Tensor(N, E, 8), dtype=float32}
instead of image frame shards.

Shard discovery path:
    <data_dir>/flows/<task_name>/shard_NNNN.pt

    Dynamics file path (actions / episode ids):
    <data_dir>/<task_name>.pt
    Preferred keys: "av_planned_edge_flow", "av_edge_action", "joint_action", "av_action", ...
    Legacy aliases: "action_edge" -> "av_edge_action", "action" -> "joint_action"

__getitem__ returns (flows, actions):
    flows:   float32 Tensor (T, E, 8)
    actions: float32 Tensor (T, n_actions)
    [episode_action_mean] optional float32 Tensor (T, n_actions), appended last
"""

import bisect
import glob
import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    """
    Flow-based world model dataset.

    Parameters
    ----------
    data_dirs : str or list[str]
        Root directories that contain both the dynamics .pt files
        (<data_dir>/<task_name>.pt) and the flow shard subdirectories
        (<data_dir>/flows/<task_name>/shard_NNNN.pt).
    task_name : str
        Name of the task to load (e.g. "ing_small").
    seq_len : int
        Number of consecutive flow snapshots per sample (default 16).
    shard_size : int
        Expected number of steps per shard file (used for index arithmetic).
        Set to 0 to auto-detect from the first shard (default 0).
    cache_mb : int
        LRU cache budget in megabytes for loaded shard tensors (default 512).
    verbose : bool
        Print progress/warning messages (default True).
    """

    def __init__(
        self,
        data_dirs: Union[str, Sequence[str]],
        task_name: str = "",
        task_names: Optional[List[str]] = None,
        seq_len: int = 16,
        shard_size: int = 0,
        cache_mb: int = 512,
        verbose: bool = True,
        action_key: str = "av_planned_edge_flow",
        aux_action_key: Optional[str] = None,
        reward_key: Optional[str] = None,
        depart_key: str = "depart",
        context_key: str = "context",
        paired_group_key: str = "pair_id",
        paired_variant_key: str = "variant_id",
        aux_target_mode: str = "step",
        return_episode_action_mean: bool = False,
    ):
        super().__init__()

        # Normalize data_dirs to list[str]
        if isinstance(data_dirs, (str, os.PathLike)):
            data_dirs = [str(data_dirs)]
        else:
            data_dirs = [str(d) for d in data_dirs]

        self.data_dirs = data_dirs
        self.task_name = task_name
        self.task_names = task_names
        self.T = int(seq_len)
        self._shard_size_hint = int(shard_size)
        self.cache_bytes = int(cache_mb) * 1024 * 1024
        self.verbose = bool(verbose)
        self._action_key = str(action_key)
        self._aux_action_key = str(aux_action_key) if aux_action_key else ""
        self._reward_key = str(reward_key) if reward_key else ""
        self._depart_key = str(depart_key) if depart_key else ""
        self._context_key = str(context_key) if context_key else ""
        self._paired_group_key = str(paired_group_key) if paired_group_key else ""
        self._paired_variant_key = str(paired_variant_key) if paired_variant_key else ""
        self._aux_target_mode = str(aux_target_mode or "step").strip().lower()
        self._return_episode_action_mean = bool(return_episode_action_mean)

        # LRU shard cache: key=(seg_idx, shard_idx) -> flows tensor (N, E, 8)
        self._cache: OrderedDict = OrderedDict()
        self._cache_nbytes: int = 0

        # Per-segment data (across all data_dirs for this task)
        self._seg_shards: List[List[str]] = []      # [seg_idx][shard_idx] -> path
        self._seg_shard_sizes: List[int] = []        # detected shard_size per segment
        self._seg_cum_frames: List[int] = []          # cumulative N_eff across segments
        # Global valid-start index structures (flat across segments)
        self._valid_starts: List[Tuple[int, int]] = []  # (seg_idx, local_start)
        self._n_edges: Optional[int] = None

        ep_offset = 0
        all_eps: List[torch.Tensor] = []
        all_acts: List[torch.Tensor] = []
        all_aux_acts: List[torch.Tensor] = []
        all_rewards: List[torch.Tensor] = []
        all_departs: List[torch.Tensor] = []
        all_contexts: List[torch.Tensor] = []
        all_pair_groups: List[torch.Tensor] = []
        all_pair_variants: List[torch.Tensor] = []
        all_episode_action_means: List[torch.Tensor] = []
        seg_num_frames: List[int] = []

        for i, dd in enumerate(self.data_dirs):
            _tname = task_names[i] if task_names else task_name
            dp = os.path.join(dd, f"{_tname}.pt")
            shard_glob = os.path.join(dd, "flows", _tname, "*shard*.pt")
            shards = sorted(glob.glob(shard_glob))

            if not os.path.exists(dp):
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: dynamics file not found ({dp})")
                continue
            if len(shards) == 0:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: no flow shards found at {shard_glob}")
                continue

            # Load dynamics file
            try:
                td = torch.load(dp, map_location="cpu", weights_only=False)
            except Exception as e:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: torch.load dynamics failed: {e}")
                continue

            try:
                ep = td["episode"].to(torch.int64).cpu()
                action_aliases = {
                    "av_planned_edge_flow": ("av_planned_edge_flow", "av_edge_action", "action_edge"),
                    "av_edge_action": ("av_edge_action", "av_planned_edge_flow", "action_edge"),
                    "action_edge": ("action_edge", "av_planned_edge_flow", "av_edge_action"),
                    "joint_action": ("joint_action", "action"),
                    "action": ("action", "joint_action"),
                }
                candidate_action_keys = action_aliases.get(self._action_key, (self._action_key,))
                act_raw = None
                resolved_action_key = None
                for candidate_key in candidate_action_keys:
                    if candidate_key in td:
                        act_raw = td[candidate_key]
                        resolved_action_key = candidate_key
                        break
                if act_raw is None:
                    available = ", ".join(sorted(map(str, td.keys())))
                    alias_hint = ""
                    if self._action_key in {"av_planned_edge_flow", "av_edge_action"}:
                        alias_hint = " Legacy aliases: 'av_edge_action', 'action_edge'."
                    elif self._action_key == "joint_action":
                        alias_hint = " Legacy alias: 'action'."
                    raise KeyError(
                        f"action key '{self._action_key}' not found in dynamics file. "
                        f"Available keys: [{available}].{alias_hint}"
                    )
                if act_raw is None:
                    raise KeyError(f"action key '{self._action_key}' resolved to None")
                act = act_raw.cpu()
                aux_act = None
                if self._aux_action_key:
                    aux_raw = td.get(self._aux_action_key)
                    if aux_raw is None:
                        raise KeyError(f"aux action key '{self._aux_action_key}' not found")
                    aux_act = aux_raw.cpu()
                rew = None
                if self._reward_key:
                    rew_raw = td.get(self._reward_key)
                    if rew_raw is None:
                        raise KeyError(f"reward key '{self._reward_key}' not found")
                    rew = rew_raw.cpu()
                # Optional depart and context (new time-step format)
                dep = td.get(self._depart_key) if self._depart_key else None
                if dep is not None:
                    dep = dep.cpu().to(torch.float32)
                ctx = td.get(self._context_key) if self._context_key else None
                if ctx is not None:
                    ctx = ctx.cpu().to(torch.float32)
                pair_group = td.get(self._paired_group_key) if self._paired_group_key else None
                if pair_group is not None:
                    pair_group = pair_group.cpu().to(torch.int64)
                pair_variant = td.get(self._paired_variant_key) if self._paired_variant_key else None
                if pair_variant is not None:
                    pair_variant = pair_variant.cpu().to(torch.int64)
            except KeyError as e:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: missing key in dynamics file: {e}")
                continue

            if act.ndim == 1:
                act = act.unsqueeze(-1)
            act = act.to(torch.float32)
            if aux_act is not None:
                if aux_act.ndim == 1:
                    aux_act = aux_act.unsqueeze(-1)
                aux_act = aux_act.to(torch.float32)
            if rew is not None:
                if rew.ndim > 1:
                    rew = rew.squeeze(-1)
                rew = rew.to(torch.float32)

            N_dyn = int(ep.shape[0])
            if act.shape[0] != N_dyn:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: episode/action length mismatch.")
                continue
            if aux_act is not None and aux_act.shape[0] != N_dyn:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: episode/aux_action length mismatch.")
                continue
            if rew is not None and rew.shape[0] != N_dyn:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: episode/reward length mismatch.")
                continue

            # Probe last shard to determine actual frames available
            try:
                last_td = torch.load(shards[-1], map_location="cpu", weights_only=False)
                last_flows = last_td["flows"]
                last_len = int(last_flows.shape[0])
                E = int(last_flows.shape[1])
            except Exception as e:
                if self.verbose:
                    print(f"[FlowDataset] Skipping {dd}: could not load last shard: {e}")
                continue

            # Detect shard_size from first shard when not provided
            if self._shard_size_hint > 0:
                shard_size_seg = self._shard_size_hint
            elif len(shards) > 1:
                try:
                    first_td = torch.load(shards[0], map_location="cpu", weights_only=False)
                    shard_size_seg = int(first_td["flows"].shape[0])
                except Exception:
                    shard_size_seg = last_len  # fallback
            else:
                shard_size_seg = last_len

            if self._shard_size_hint == 0:
                warnings.warn(
                    f"[FlowDataset] shard_size not provided; auto-detected {shard_size_seg} "
                    f"from first shard of segment {len(self._seg_shards)}. Pass shard_size= to suppress this.",
                    stacklevel=2,
                )

            N_frames_avail = (len(shards) - 1) * shard_size_seg + last_len
            N_eff = min(N_dyn, N_frames_avail)

            if N_eff < self.T:
                if self.verbose:
                    print(
                        f"[FlowDataset] Skipping {dd}: not enough frames "
                        f"(N_eff={N_eff}) for T={self.T}."
                    )
                continue

            # Validate E consistency
            if self._n_edges is None:
                self._n_edges = E
            elif self._n_edges != E:
                if self.verbose:
                    print(
                        f"[FlowDataset] Skipping {dd}: n_edges mismatch "
                        f"({E} vs {self._n_edges})."
                    )
                continue

            ep = ep[:N_eff]
            act = act[:N_eff]
            if self._return_episode_action_mean:
                episode_action_mean = self._episode_mean_action_condition(ep[:N_eff], act)
            if aux_act is not None:
                aux_act = aux_act[:N_eff]
                if self._aux_target_mode in {"episode_mean", "episode_response", "response"}:
                    aux_act = self._episode_mean_aux_target(ep[:N_eff], aux_act)
            if rew is not None:
                rew = rew[:N_eff]
            if pair_group is not None:
                pair_group = pair_group[:N_eff]
            if pair_variant is not None:
                pair_variant = pair_variant[:N_eff]

            # Make episode IDs unique across segments
            if ep.numel() > 0:
                seg_max = int(ep.max().item())
            else:
                seg_max = 0
            ep = ep + ep_offset
            ep_offset += seg_max + 1

            seg_idx = len(self._seg_shards)
            self._seg_shards.append(shards)
            self._seg_shard_sizes.append(shard_size_seg)
            all_eps.append(ep)
            all_acts.append(act)
            if self._return_episode_action_mean:
                all_episode_action_means.append(episode_action_mean)
            if self._aux_action_key:
                assert aux_act is not None
                all_aux_acts.append(aux_act)
            if self._reward_key:
                assert rew is not None
                all_rewards.append(rew)
            if dep is not None:
                all_departs.append(dep)
            if ctx is not None:
                all_contexts.append(ctx)
            if pair_group is not None:
                all_pair_groups.append(pair_group)
            if pair_variant is not None:
                all_pair_variants.append(pair_variant)
            seg_num_frames.append(N_eff)

            if self.verbose:
                print(
                    f"[FlowDataset] Loaded {dd}: "
                    f"shards={len(shards)}, N_eff={N_eff}, E={E}, "
                    f"n_actions={act.shape[-1]}, action_key={resolved_action_key}"
                )

        if len(self._seg_shards) == 0:
            raise RuntimeError(
                f"[FlowDataset] No valid segments found for task='{task_name}' "
                f"in data_dirs={data_dirs}"
            )

        # Build cumulative frame counts
        s = 0
        for nf in seg_num_frames:
            s += nf
            self._seg_cum_frames.append(s)

        # Concatenate all segments for episode/action lookup
        self._ep_all = torch.cat(all_eps, dim=0)    # (N_total,)
        self._act_all = torch.cat(all_acts, dim=0)  # (N_total, n_actions)
        self._episode_act_mean_all = (
            torch.cat(all_episode_action_means, dim=0) if self._return_episode_action_mean else None
        )
        self._aux_act_all = torch.cat(all_aux_acts, dim=0) if self._aux_action_key else None
        self._rew_all = torch.cat(all_rewards, dim=0) if self._reward_key else None
        self._depart_all = torch.cat(all_departs, dim=0) if all_departs else None  # (N_total, E)
        self._context_all = torch.cat(all_contexts, dim=0) if all_contexts else None  # (N_total, C)
        self._pair_group_all = (
            torch.cat(all_pair_groups, dim=0) if len(all_pair_groups) == len(seg_num_frames) else None
        )
        self._pair_variant_all = (
            torch.cat(all_pair_variants, dim=0) if len(all_pair_variants) == len(seg_num_frames) else None
        )

        # Build segment -> global offset map for indexing
        self._seg_offsets: List[int] = [0]
        for nf in seg_num_frames[:-1]:
            self._seg_offsets.append(self._seg_offsets[-1] + nf)

        # Compute valid starts: windows of length T that stay within one episode
        # and have no NaN actions. Actions are indexed at the same decision
        # step as frames; train_dynamics shifts them once for causal use.
        N_total = int(self._ep_all.shape[0])
        start_count = N_total - self.T + 1

        if start_count <= 0:
            raise RuntimeError(
                f"[FlowDataset] Total frames {N_total} insufficient for T={self.T}."
            )

        ep_ok = (self._ep_all[:start_count] == self._ep_all[self.T - 1 : self.T - 1 + start_count])

        act_nan = torch.isnan(self._act_all).any(dim=-1)  # (N_total,)
        step_ok = ~act_nan                                  # (N_total,)
        if self._aux_act_all is not None:
            aux_nan = torch.isnan(self._aux_act_all).any(dim=-1)
            step_ok = step_ok & (~aux_nan)
        if self._rew_all is not None:
            rew_nan = torch.isnan(self._rew_all)
            step_ok = step_ok & (~rew_nan)

        step_ok2 = step_ok                                  # (N_total,)
        cs = torch.cumsum(step_ok2.to(torch.int32), dim=0)
        end_idx = torch.arange(start_count) + (self.T - 1)
        prev_idx = torch.arange(start_count) - 1
        prev_cs = torch.zeros(start_count, dtype=cs.dtype)
        m = prev_idx >= 0
        prev_cs[m] = cs[prev_idx[m]]
        window_ok = (cs[end_idx] - prev_cs) == self.T

        valid_mask = ep_ok & window_ok
        self._valid_global_starts = valid_mask.nonzero(as_tuple=False).flatten()  # global indices

        if self._valid_global_starts.numel() == 0:
            raise RuntimeError(
                f"[FlowDataset] No valid windows for task='{task_name}' after filtering."
            )

        if self.verbose:
            print(
                f"[FlowDataset] task={task_name} "
                f"total_frames={N_total} "
                f"valid_sequences={self._valid_global_starts.numel()} "
                f"n_edges={self._n_edges} "
                f"n_actions={self._act_all.shape[-1]}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_edges(self) -> int:
        """Number of road-segment edges E (inferred from shard shape)."""
        assert self._n_edges is not None
        return self._n_edges

    @property
    def n_actions(self) -> int:
        """Dimensionality of the action vector."""
        return int(self._act_all.shape[-1])

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._valid_global_starts.numel())

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        flows : float32 Tensor (T, E, 8)
        actions : float32 Tensor (T, n_actions)
        [aux_act] : optional float32 Tensor (T, aux_dim)
        [reward] : optional float32 Tensor (T,)
        [depart] : optional float32 Tensor (T, E) — per-edge departure counts
        [context] : optional float32 Tensor (T, C) — global context features
        [episode_action_mean] : optional float32 Tensor (T, n_actions), appended last
        """
        global_start = int(self._valid_global_starts[idx].item())

        flows = self._get_flows(global_start, self.T)  # (T, E, 8)

        # Actions at the same decision indices as flows. The dynamics builder
        # performs the single causal shift needed for S_t <- A_{t-1}.
        act = self._act_all[global_start : global_start + self.T]  # (T, n_actions)
        act = torch.nan_to_num(act, nan=0.0)

        result = [flows, act]

        if self._aux_act_all is not None:
            aux_act = self._aux_act_all[global_start : global_start + self.T]
            result.append(torch.nan_to_num(aux_act, nan=0.0))

        if self._rew_all is not None:
            reward = self._rew_all[global_start : global_start + self.T]
            result.append(torch.nan_to_num(reward, nan=0.0))

        if self._depart_all is not None:
            dep = self._depart_all[global_start : global_start + self.T]
            result.append(torch.nan_to_num(dep, nan=0.0))

        if self._context_all is not None:
            ctx = self._context_all[global_start : global_start + self.T]
            result.append(torch.nan_to_num(ctx, nan=0.0))

        if self._episode_act_mean_all is not None:
            ep_act = self._episode_act_mean_all[global_start : global_start + self.T]
            result.append(torch.nan_to_num(ep_act, nan=0.0))

        return tuple(result)

    @staticmethod
    def _episode_mean_aux_target(ep: torch.Tensor, aux_act: torch.Tensor) -> torch.Tensor:
        """Replace per-step auxiliary targets with one normalized target per episode."""
        if aux_act.ndim == 1:
            aux_2d = aux_act.unsqueeze(-1)
        else:
            aux_2d = aux_act
        out = torch.empty_like(aux_2d, dtype=torch.float32)
        ep_cpu = ep.to(torch.int64).cpu()
        for ep_id in torch.unique(ep_cpu):
            mask = ep_cpu == ep_id
            vals = aux_2d[mask].to(torch.float32)
            mean = vals.clamp_min(0.0).mean(dim=0)
            total = mean.sum()
            if float(total.item()) > 1e-8:
                mean = mean / total
            out[mask] = mean
        return out if aux_act.ndim != 1 else out.squeeze(-1)

    @staticmethod
    def _episode_mean_action_condition(ep: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Repeat the raw mean action field for each full episode.

        Unlike HDV auxiliary targets, this action condition is intentionally
        not normalized.  MARL inference calls predict_episode_hdv_response()
        with an average AV pressure field, so the training condition should
        preserve that same action scale.
        """
        if act.ndim == 1:
            act_2d = act.unsqueeze(-1)
        else:
            act_2d = act
        out = torch.empty_like(act_2d, dtype=torch.float32)
        ep_cpu = ep.to(torch.int64).cpu()
        for ep_id in torch.unique(ep_cpu):
            mask = ep_cpu == ep_id
            vals = act_2d[mask].to(torch.float32)
            out[mask] = vals.mean(dim=0)
        return out if act.ndim != 1 else out.squeeze(-1)

    # ------------------------------------------------------------------
    # Internal: shard loading + LRU cache
    # ------------------------------------------------------------------

    def _get_flows(self, global_start: int, length: int) -> torch.Tensor:
        """Read `length` consecutive flow frames starting at global index `global_start`."""
        out: List[torch.Tensor] = []
        idx = int(global_start)
        remaining = int(length)

        while remaining > 0:
            # Which segment does `idx` fall in?
            seg_idx = bisect.bisect_right(self._seg_cum_frames, idx)
            seg_start = 0 if seg_idx == 0 else self._seg_cum_frames[seg_idx - 1]
            seg_end = self._seg_cum_frames[seg_idx]
            local_idx = idx - seg_start

            shard_size = self._seg_shard_sizes[seg_idx]
            shard_idx = local_idx // shard_size
            off = local_idx % shard_size

            flows_shard = self._load_shard(seg_idx, shard_idx)

            take = min(remaining, flows_shard.shape[0] - off)
            take = min(take, seg_end - idx)

            if take <= 0:
                raise RuntimeError(
                    f"[FlowDataset] Frame indexing error: "
                    f"global_idx={idx} seg_idx={seg_idx} "
                    f"local_idx={local_idx} shard_idx={shard_idx} off={off} "
                    f"shard_len={flows_shard.shape[0]}"
                )

            out.append(flows_shard[off : off + take])
            idx += take
            remaining -= take

        result = torch.cat(out, dim=0)  # (length, E, 8)
        assert result.shape[0] == length, (
            f"[FlowDataset._get_flows] expected {length} frames, got {result.shape[0]}. "
            "Possible segment boundary misalignment."
        )
        return result

    def _load_shard(self, seg_idx: int, shard_idx: int) -> torch.Tensor:
        key = (seg_idx, shard_idx)
        # LRU hit
        if key in self._cache:
            val = self._cache.pop(key)
            self._cache[key] = val
            return val

        path = self._seg_shards[seg_idx][shard_idx]
        try:
            td = torch.load(path, map_location="cpu", weights_only=False)
            flows = td["flows"].to(torch.float32)
            if flows.ndim != 3 or flows.shape[2] != 8:
                raise ValueError(
                    f"[FlowDataset] Shard {path}: expected (N, E, 8) tensor, "
                    f"got {tuple(flows.shape)}"
                )
        except Exception as e:
            warnings.warn(
                f"[FlowDataset] Failed to load shard {path}: {e}. Returning zeros."
            )
            E = self._n_edges or 1
            flows = torch.zeros(self._seg_shard_sizes[seg_idx], E, 8, dtype=torch.float32)

        # LRU eviction
        nbytes = flows.nbytes
        while self._cache_nbytes + nbytes > self.cache_bytes and len(self._cache) > 0:
            _, evicted = self._cache.popitem(last=False)
            self._cache_nbytes -= evicted.nbytes
        self._cache[key] = flows
        self._cache_nbytes += nbytes
        return flows

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "FlowDataset":
        """
        Construct from a configuration dict.

        Expected keys (mirrors WMDataset-style config):
            data_dir  : str or list[str]  — root data directories
            task_name : str               — task identifier
            seq_len   : int               — sequence length (default 16)
            shard_size: int               — shard size hint (default 0 = auto)
            cache_mb  : int               — LRU cache MB (default 512)
            verbose   : bool              — verbosity (default True)
        """
        return cls(
            data_dirs=cfg["data_dir"],
            task_name=cfg["task_name"],
            seq_len=int(cfg.get("seq_len", 16)),
            shard_size=int(cfg.get("shard_size", 0)),
            cache_mb=int(cfg.get("cache_mb", 512)),
            verbose=bool(cfg.get("verbose", True)),
            action_key=str(cfg.get("action_key", "av_planned_edge_flow")),
            aux_action_key=cfg.get("aux_action_key", None),
            reward_key=cfg.get("reward_key", None),
        )
