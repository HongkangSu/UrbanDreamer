#!/usr/bin/env python3
"""Run a TorchRL-style IQL baseline on the toy cross-bottleneck network."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import sysconfig
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import Categorical as CategoricalSpec
from torchrl.data import Composite
from torchrl.modules import MultiAgentMLP, QValueModule, SafeSequential
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "dreamer4", REPO_ROOT / "dreamer4" / "urban_routing"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _ensure_sumo_runtime_env() -> None:
    """Expose SUMO binaries/tools before importing collector code."""
    prefix = Path(sys.executable).resolve().parent.parent
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    candidate_homes = [
        Path(os.environ["SUMO_HOME"]).expanduser() if os.environ.get("SUMO_HOME") else None,
        purelib / "sumo" if purelib else None,
        prefix / "share" / "sumo",
        Path("/usr/share/sumo"),
    ]
    chosen_home: Optional[Path] = None
    for candidate in candidate_homes:
        if candidate and candidate.is_dir():
            chosen_home = candidate.resolve()
            break
    if chosen_home is not None:
        os.environ.setdefault("SUMO_HOME", str(chosen_home))
        tools_dir = chosen_home / "tools"
        if tools_dir.is_dir() and str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []
    candidate_bins = [
        prefix / "bin",
        chosen_home / "bin" if chosen_home is not None else None,
    ]
    prepend = [
        str(candidate.resolve())
        for candidate in candidate_bins
        if candidate is not None
        and candidate.is_dir()
        and str(candidate.resolve()) not in path_parts
    ]
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + path_parts)


_ensure_sumo_runtime_env()

from ablation.create_toy_cross_bottleneck_network import NETWORK_NAME as TOY_NETWORK_NAME
from ablation.create_toy_cross_bottleneck_network import generate as generate_toy_network
from ablation.online_no_cfm_actor_critic import (
    OnlineDecisionSample,
    OnlineNoCfmConfig,
    _resolve_path,
)
from ablation.run_toy_cross_bottleneck_online_ac import (
    ToyOnlineAcTrainer,
    parse_checkpoint_episodes,
    validate_paths,
)
from ablation.eval_toy_online_ac_dynamic import _mean_complete
from dreamer4.urban_routing.flow_projector import EdgeRegistry
from dreamer4.urban_routing.train_marl import vehicle_mode_obs_dim
from dreamer4.urban_routing.urb_fair_protocol import derive_fixed_roles, load_post_mutation_state


@dataclass
class ToyIqlTorchRLConfig(OnlineNoCfmConfig):
    """Toy fixed-role IQL config aligned with ``../URB/scripts/iql_torchrl.py``."""

    agent_frames_per_batch: int = 1
    n_iters: int = 0
    checkpoint_interval: int = 100
    num_epochs: int = 1
    minibatch_size: int = 32
    q_network_depth: int = 3
    q_network_num_cells: int = 64
    eps_init: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.8
    target_update_eps: float = 0.995
    av_execution_mode: str = "dynamic"

    def resolve(self) -> "ToyIqlTorchRLConfig":
        values = asdict(self)
        for key in (
            "settings",
            "output_dir",
            "network_dir",
            "edge_registry",
            "projector_ckpt",
            "dynamics_ckpt",
            "route_artifacts_dir",
            "post_mutation_state_in",
            "hdv_state_in",
            "resume_checkpoint",
        ):
            raw = values.get(key, "")
            if raw:
                values[key] = str(_resolve_path(raw))
        return ToyIqlTorchRLConfig(**values)

    @classmethod
    def from_settings(cls, settings_path: str, **overrides: Any) -> "ToyIqlTorchRLConfig":
        base_fields = set(OnlineNoCfmConfig.__dataclass_fields__.keys())
        base_overrides = {k: v for k, v in overrides.items() if k in base_fields}
        own_overrides = {k: v for k, v in overrides.items() if k not in base_fields}
        base = OnlineNoCfmConfig.from_settings(settings_path, **base_overrides).resolve()
        values = {**asdict(cls()), **asdict(base), **own_overrides}
        return cls(**values).resolve()


def _jsonable_config(cfg: ToyIqlTorchRLConfig) -> Dict[str, Any]:
    values = asdict(cfg)
    values["checkpoint_episodes"] = list(cfg.checkpoint_episodes)
    values["extra_eval_args"] = list(cfg.extra_eval_args)
    return values


def _resolve_fixed_roles_for_config(
    cfg: ToyIqlTorchRLConfig,
) -> Tuple[List[str], List[str], str]:
    post_state_in = str(getattr(cfg, "post_mutation_state_in", "") or "").strip()
    if post_state_in:
        post_state_path = Path(post_state_in).resolve()
        if post_state_path.exists():
            post_state = load_post_mutation_state(post_state_path)
            machine_ids = [str(x) for x in post_state.get("machine_agent_ids", [])]
            human_ids = [str(x) for x in post_state.get("human_agent_ids", [])]
            return machine_ids, human_ids, str(post_state_path)

    agents_csv = Path(cfg.route_artifacts_dir).resolve() / "agents.csv"
    machine_ids, human_ids = derive_fixed_roles(
        agents_csv=agents_csv,
        ratio_machines=float(cfg.ratio_machines),
        seed=int(cfg.seed),
    )
    return machine_ids, human_ids, "derived_from_agents_csv"


def _checkpoint_episodes_from_args(
    *,
    raw_episodes: str,
    train_episodes: int,
    checkpoint_interval: int,
) -> tuple[int, ...]:
    raw = str(raw_episodes or "").strip()
    if raw:
        return parse_checkpoint_episodes(raw)
    interval = int(checkpoint_interval)
    total = int(train_episodes)
    if interval <= 0 or total <= 0:
        return tuple()
    return tuple(range(interval, total + 1, interval))


class TorchRLIqlQNetwork(torch.nn.Module):
    """Independent TorchRL Q networks with fixed vehicle-to-agent slots."""

    def __init__(
        self,
        *,
        obs_dim: int,
        n_routes: int,
        machine_ids: Sequence[str],
        cfg: ToyIqlTorchRLConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_routes = int(n_routes)
        self.machine_ids = [str(x) for x in machine_ids]
        self.n_agents = int(max(len(self.machine_ids), 1))
        self.slot_by_vehicle = {veh_id: idx for idx, veh_id in enumerate(self.machine_ids)}
        self.device_ref = device
        self.epsilon = float(cfg.eps_init)
        self.include_global = bool(cfg.include_global)
        self.include_route_costs = bool(cfg.include_route_costs)
        self.include_fleet_pressure = bool(cfg.include_fleet_pressure)

        self.q_net = MultiAgentMLP(
            n_agent_inputs=int(obs_dim),
            n_agent_outputs=int(n_routes),
            n_agents=int(self.n_agents),
            centralized=False,
            share_params=False,
            device=device,
            depth=int(cfg.q_network_depth),
            num_cells=int(cfg.q_network_num_cells),
            activation_class=torch.nn.ReLU,
        )
        self.q_module = TensorDictModule(
            self.q_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "action_value")],
        )
        action_spec = Composite(
            {
                ("agents", "action"): CategoricalSpec(
                    n=int(n_routes),
                    shape=(int(self.n_agents),),
                    device=device,
                )
            },
            shape=(),
        )
        self.q_value_module = QValueModule(
            action_value_key=("agents", "action_value"),
            action_mask_key=("agents", "action_mask"),
            out_keys=[
                ("agents", "action"),
                ("agents", "action_value"),
                ("agents", "chosen_action_value"),
            ],
            spec=action_spec,
            action_space=None,
        )
        self.qnet = SafeSequential(self.q_module, self.q_value_module)
        self.to(device)

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(np.clip(float(epsilon), 0.0, 1.0))

    def slot_for_vehicle(self, veh_id: str) -> int:
        key = str(veh_id)
        if key not in self.slot_by_vehicle:
            raise KeyError(f"vehicle {key!r} is not in fixed IQL machine roles")
        return int(self.slot_by_vehicle[key])

    def _route_cost_valid_mask(self, obs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.include_route_costs:
            return None
        context_dim = 3
        cost_dim = 3 * int(self.n_routes)
        fleet_dim = 2 * int(self.n_routes) if self.include_fleet_pressure else 0
        denom = int(self.n_routes) + (1 if self.include_global else 0)
        route_and_global = int(self.obs_dim) - context_dim - cost_dim - fleet_dim
        if denom <= 0 or route_and_global <= 0 or route_and_global % denom != 0:
            return None
        route_feature_dim = route_and_global // denom
        base_feature_dim = route_feature_dim * denom
        valid_start = base_feature_dim + 2 * int(self.n_routes)
        valid_end = valid_start + int(self.n_routes)
        if valid_start < 0 or int(obs.shape[-1]) < valid_end:
            return None
        mask = obs[..., valid_start:valid_end].to(dtype=torch.bool)
        mask = mask.clone()
        mask[~mask.any(dim=-1), :] = True
        return mask

    def _group_observation_td(
        self,
        obs: torch.Tensor,
        active_ids: Sequence[str],
    ) -> Tuple[TensorDict, torch.Tensor]:
        obs_t = obs.to(device=self.device_ref, dtype=torch.float32)
        n = int(obs_t.shape[0])
        grouped = torch.zeros(
            (n, int(self.n_agents), int(self.obs_dim)),
            device=self.device_ref,
            dtype=torch.float32,
        )
        slots = torch.zeros((n,), device=self.device_ref, dtype=torch.long)
        for row, veh_id in enumerate(active_ids):
            slot = self.slot_for_vehicle(str(veh_id))
            grouped[row, slot] = obs_t[row]
            slots[row] = int(slot)
        td = TensorDict({("agents", "observation"): grouped}, batch_size=[n], device=self.device_ref)
        return td, slots

    def forward_action_values_for_active_ids(
        self,
        obs: torch.Tensor,
        active_ids: Sequence[str],
    ) -> torch.Tensor:
        td, slots = self._group_observation_td(obs, active_ids)
        out = self.q_module(td)
        q_values = out.get(("agents", "action_value"))
        rows = torch.arange(int(q_values.shape[0]), device=q_values.device)
        return q_values[rows, slots]

    def forward_logits_for_active_ids(
        self,
        obs: torch.Tensor,
        active_ids: Sequence[str],
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_t = obs.to(device=self.device_ref, dtype=torch.float32)
        q_values = self.forward_action_values_for_active_ids(obs_t, active_ids)
        if action_mask is not None:
            mask = action_mask.to(device=q_values.device, dtype=torch.bool)
        else:
            mask = self._route_cost_valid_mask(obs_t)
            if mask is not None:
                mask = mask.to(device=q_values.device, dtype=torch.bool)
        if mask is None:
            mask = torch.ones_like(q_values, dtype=torch.bool)
        if mask.shape != q_values.shape:
            raise ValueError(f"action_mask shape {tuple(mask.shape)} does not match q_values {tuple(q_values.shape)}")
        safe_mask = mask.clone()
        safe_mask[~safe_mask.any(dim=-1), :] = True
        q_values = q_values.masked_fill(~safe_mask, -1.0e9)
        eps = float(self.epsilon) if self.training else 0.0
        greedy = torch.argmax(q_values, dim=-1, keepdim=True)
        valid_count = safe_mask.sum(dim=-1, keepdim=True).to(dtype=q_values.dtype).clamp_min(1.0)
        probs = safe_mask.to(dtype=q_values.dtype) * (eps / valid_count)
        probs.scatter_add_(
            dim=-1,
            index=greedy,
            src=torch.full_like(greedy, fill_value=1.0 - eps, dtype=q_values.dtype),
        )
        return torch.log(probs.clamp_min(1e-8))


def _loss_matrix(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.shape == reference.shape:
        return value
    if value.dim() == 0:
        return value.expand_as(reference)
    if value.shape == reference.shape + (1,):
        return value.squeeze(-1)
    return value.expand_as(reference)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ref = mask.to(dtype=value.dtype, device=value.device)
    value = _loss_matrix(value, ref)
    denom = ref.sum().clamp_min(1.0)
    return (value * ref).sum() / denom


def _sample_action_mask(sample: OnlineDecisionSample, *, n_routes: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros((int(n_routes),), device=device, dtype=torch.bool)
    mask_np = np.asarray(sample.valid_mask, dtype=bool).reshape(-1)
    if int(mask_np.shape[0]) >= int(n_routes):
        mask[:] = torch.as_tensor(mask_np[: int(n_routes)], device=device, dtype=torch.bool)
    elif int(mask_np.shape[0]) > 0:
        mask[: int(mask_np.shape[0])] = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
    if not bool(mask.any().detach().cpu().item()):
        mask[:] = True
    return mask


def _sample_observation(sample: OnlineDecisionSample, *, obs_dim: int) -> np.ndarray:
    obs = np.asarray(sample.obs, dtype=np.float32).reshape(-1)
    if int(obs.shape[0]) != int(obs_dim):
        raise ValueError(f"sample obs_dim={obs.shape[0]} does not match q-network obs_dim={obs_dim}")
    return obs


def _dense_observation_and_slots(
    *,
    samples: Sequence[OnlineDecisionSample],
    indices: Sequence[int],
    q_network: TorchRLIqlQNetwork,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    n = int(len(indices))
    obs = torch.zeros(
        (n, int(q_network.n_agents), int(q_network.obs_dim)),
        device=device,
        dtype=torch.float32,
    )
    slots: List[int] = []
    for row, raw_idx in enumerate(indices):
        sample = samples[int(raw_idx)]
        slot = q_network.slot_for_vehicle(str(sample.veh_id))
        obs[row, slot] = torch.as_tensor(
            _sample_observation(sample, obs_dim=int(q_network.obs_dim)),
            device=device,
            dtype=torch.float32,
        )
        slots.append(int(slot))
    return obs, slots


def _random_minibatch_indices(
    *,
    n_samples: int,
    minibatch_size: int,
    num_batches: int,
) -> List[np.ndarray]:
    n = int(n_samples)
    batch = max(1, min(int(minibatch_size), n))
    out: List[np.ndarray] = []
    order = np.random.permutation(n)
    cursor = 0
    for _ in range(max(int(num_batches), 1)):
        if cursor + batch > n:
            order = np.random.permutation(n)
            cursor = 0
        out.append(order[cursor : cursor + batch])
        cursor += batch
    return out


def _build_iql_next_indices(samples: Sequence[OnlineDecisionSample]) -> Tuple[np.ndarray, np.ndarray]:
    next_indices = np.full((len(samples),), -1, dtype=np.int64)
    done = np.ones((len(samples),), dtype=bool)
    by_vehicle: Dict[Tuple[int, str], List[int]] = {}
    for row, sample in enumerate(samples):
        by_vehicle.setdefault((int(sample.episode), str(sample.veh_id)), []).append(row)
    for indices in by_vehicle.values():
        ordered = sorted(indices, key=lambda i: (int(samples[i].decision_idx), int(samples[i].sim_step)))
        for pos, row in enumerate(ordered[:-1]):
            next_indices[int(row)] = int(ordered[pos + 1])
            done[int(row)] = False
    return next_indices, done


def _build_iql_minibatch_td(
    *,
    samples: Sequence[OnlineDecisionSample],
    indices: Sequence[int],
    q_network: TorchRLIqlQNetwork,
    next_indices: np.ndarray,
    done_flags: np.ndarray,
    device: torch.device,
) -> TensorDict:
    n = int(len(indices))
    n_agents = int(q_network.n_agents)
    obs_dim = int(q_network.obs_dim)
    n_routes = int(q_network.n_routes)
    obs, slots = _dense_observation_and_slots(
        samples=samples,
        indices=indices,
        q_network=q_network,
        device=device,
    )
    next_obs = torch.zeros((n, n_agents, obs_dim), device=device, dtype=torch.float32)
    action = torch.zeros((n, n_agents), device=device, dtype=torch.long)
    action_mask = torch.ones((n, n_agents, n_routes), device=device, dtype=torch.bool)
    next_action_mask = torch.ones((n, n_agents, n_routes), device=device, dtype=torch.bool)
    active_mask = torch.zeros((n, n_agents), device=device, dtype=torch.bool)
    rewards = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    done = torch.ones((n, n_agents, 1), device=device, dtype=torch.bool)
    terminated = torch.ones((n, n_agents, 1), device=device, dtype=torch.bool)

    for row, (raw_idx, slot) in enumerate(zip(indices, slots)):
        sample_idx = int(raw_idx)
        sample = samples[sample_idx]
        action[row, slot] = int(np.clip(int(sample.action), 0, n_routes - 1))
        action_mask[row, slot] = _sample_action_mask(sample, n_routes=n_routes, device=device)
        active_mask[row, slot] = True
        rewards[row, slot, 0] = float(sample.reward)
        if not bool(done_flags[sample_idx]):
            next_idx = int(next_indices[sample_idx])
            next_sample = samples[next_idx]
            next_obs[row, slot] = torch.as_tensor(
                _sample_observation(next_sample, obs_dim=obs_dim),
                device=device,
                dtype=torch.float32,
            )
            next_action_mask[row, slot] = _sample_action_mask(next_sample, n_routes=n_routes, device=device)
            done[row, slot, 0] = False
            terminated[row, slot, 0] = False

    return TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): action,
            ("agents", "action_mask"): action_mask,
            ("agents", "active_mask"): active_mask,
            ("agents", "reward"): rewards,
            ("next", "agents", "observation"): next_obs,
            ("next", "agents", "action_mask"): next_action_mask,
            ("next", "agents", "reward"): rewards,
            ("next", "agents", "done"): done,
            ("next", "agents", "terminated"): terminated,
        },
        batch_size=[n],
        device=device,
    )


def _chosen_q_for_samples(
    *,
    samples: Sequence[OnlineDecisionSample],
    q_network: TorchRLIqlQNetwork,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    n = int(len(samples))
    chosen = np.zeros((n,), dtype=np.float32)
    chunk = max(1, min(int(chunk_size), n))
    q_network.eval()
    with torch.no_grad():
        for start in range(0, n, chunk):
            indices = list(range(start, min(start + chunk, n)))
            obs, slots = _dense_observation_and_slots(
                samples=samples,
                indices=indices,
                q_network=q_network,
                device=device,
            )
            action_mask = torch.ones(
                (len(indices), int(q_network.n_agents), int(q_network.n_routes)),
                device=device,
                dtype=torch.bool,
            )
            actions = torch.zeros((len(indices), int(q_network.n_agents)), device=device, dtype=torch.long)
            for row, raw_idx in enumerate(indices):
                sample = samples[int(raw_idx)]
                slot = int(slots[row])
                action_mask[row, slot] = _sample_action_mask(sample, n_routes=int(q_network.n_routes), device=device)
                actions[row, slot] = int(np.clip(int(sample.action), 0, int(q_network.n_routes) - 1))
            q_td = q_network.q_module(
                TensorDict(
                    {
                        ("agents", "observation"): obs,
                        ("agents", "action_mask"): action_mask,
                    },
                    batch_size=[len(indices)],
                    device=device,
                )
            )
            q_values = q_td.get(("agents", "action_value"))
            rows = torch.arange(len(indices), device=device)
            slot_tensor = torch.as_tensor(slots, device=device, dtype=torch.long)
            chosen[start : start + len(indices)] = (
                q_values[rows, slot_tensor]
                .gather(-1, actions[rows, slot_tensor].unsqueeze(-1))
                .squeeze(-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return chosen


def build_torchrl_iql_batch(
    *,
    samples: Sequence[OnlineDecisionSample],
    q_network: TorchRLIqlQNetwork,
    device: torch.device,
) -> TensorDict:
    n = int(len(samples))
    if n <= 0:
        raise ValueError("cannot build an IQL TorchRL batch with no samples")

    n_agents = int(q_network.n_agents)
    obs_dim = int(q_network.obs_dim)
    n_routes = int(q_network.n_routes)
    obs = torch.zeros((n, n_agents, obs_dim), device=device, dtype=torch.float32)
    next_obs = torch.zeros((n, n_agents, obs_dim), device=device, dtype=torch.float32)
    action = torch.zeros((n, n_agents), device=device, dtype=torch.long)
    action_mask = torch.ones((n, n_agents, n_routes), device=device, dtype=torch.bool)
    next_action_mask = torch.ones((n, n_agents, n_routes), device=device, dtype=torch.bool)
    active_mask = torch.zeros((n, n_agents), device=device, dtype=torch.bool)
    rewards = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    done = torch.ones((n, n_agents, 1), device=device, dtype=torch.bool)
    terminated = torch.ones((n, n_agents, 1), device=device, dtype=torch.bool)
    slot_rows: List[int] = []
    by_vehicle: Dict[Tuple[int, str], List[int]] = {}

    for row, sample in enumerate(samples):
        slot = q_network.slot_for_vehicle(str(sample.veh_id))
        slot_rows.append(int(slot))
        sample_obs = np.asarray(sample.obs, dtype=np.float32).reshape(-1)
        if int(sample_obs.shape[0]) != obs_dim:
            raise ValueError(
                f"sample obs_dim={sample_obs.shape[0]} does not match q-network obs_dim={obs_dim}"
            )
        obs[row, slot] = torch.as_tensor(sample_obs, device=device, dtype=torch.float32)
        action[row, slot] = int(np.clip(int(sample.action), 0, n_routes - 1))
        action_mask[row, slot] = _sample_action_mask(sample, n_routes=n_routes, device=device)
        active_mask[row, slot] = True
        rewards[row, slot, 0] = float(sample.reward)
        by_vehicle.setdefault((int(sample.episode), str(sample.veh_id)), []).append(row)

    for indices in by_vehicle.values():
        ordered = sorted(indices, key=lambda i: (int(samples[i].decision_idx), int(samples[i].sim_step)))
        for pos, row in enumerate(ordered):
            slot = int(slot_rows[row])
            if pos + 1 >= len(ordered):
                continue
            next_row = int(ordered[pos + 1])
            next_sample = samples[next_row]
            next_sample_obs = np.asarray(next_sample.obs, dtype=np.float32).reshape(-1)
            next_obs[row, slot] = torch.as_tensor(next_sample_obs, device=device, dtype=torch.float32)
            next_action_mask[row, slot] = _sample_action_mask(next_sample, n_routes=n_routes, device=device)
            done[row, slot, 0] = False
            terminated[row, slot, 0] = False

    return TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): action,
            ("agents", "action_mask"): action_mask,
            ("agents", "active_mask"): active_mask,
            ("agents", "reward"): rewards,
            ("next", "agents", "observation"): next_obs,
            ("next", "agents", "action_mask"): next_action_mask,
            ("next", "agents", "reward"): rewards,
            ("next", "agents", "done"): done,
            ("next", "agents", "terminated"): terminated,
        },
        batch_size=[n],
        device=device,
    )


def build_torchrl_iql_loss(
    *,
    q_network: TorchRLIqlQNetwork,
    cfg: ToyIqlTorchRLConfig,
) -> DQNLoss:
    loss_module = DQNLoss(q_network.qnet, delay_value=True, reduction="none")
    loss_module.set_keys(
        action_value=("agents", "action_value"),
        action=("agents", "action"),
        value=("agents", "chosen_action_value"),
        reward=("agents", "reward"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(ValueEstimators.TD0, gamma=float(cfg.gamma))
    return loss_module


class ToyIqlTorchRLTrainer(ToyOnlineAcTrainer):
    """Fixed-role toy trainer using TorchRL's independent Q-learning stack."""

    def __init__(self, cfg: ToyIqlTorchRLConfig):
        self.cfg = cfg.resolve()
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(self.cfg.seed))
        np.random.seed(int(self.cfg.seed))
        random.seed(int(self.cfg.seed))
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.edge_registry = EdgeRegistry.load(str(self.cfg.edge_registry))
        self.obs_dim = vehicle_mode_obs_dim(
            n_routes=int(self.cfg.n_routes),
            d_obs=8,
            include_global=bool(self.cfg.include_global),
            route_stats=str(self.cfg.route_stats),
            include_route_costs=bool(self.cfg.include_route_costs),
            include_fleet_pressure=bool(self.cfg.include_fleet_pressure),
        )
        self.machine_ids, self.human_ids, self.role_source = _resolve_fixed_roles_for_config(self.cfg)
        if not self.machine_ids:
            raise ValueError("IQL fixed-role baseline requires at least one machine agent")
        self.actor = TorchRLIqlQNetwork(
            obs_dim=int(self.obs_dim),
            n_routes=int(self.cfg.n_routes),
            machine_ids=self.machine_ids,
            cfg=self.cfg,
            device=self.device,
        )
        self.loss_module = build_torchrl_iql_loss(q_network=self.actor, cfg=self.cfg).to(self.device)
        self.target_net_updater = SoftUpdate(self.loss_module, eps=float(self.cfg.target_update_eps))
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=float(self.cfg.lr))
        self.collector = None
        self.loaded_resume_episode = self._load_resume_checkpoint()

    @staticmethod
    def _pending_update_metrics(
        *,
        n_samples: int,
        pending_episodes: int,
        epsilon: float,
    ) -> Dict[str, float]:
        return {
            "loss": 0.0,
            "q_loss": 0.0,
            "n_samples": float(n_samples),
            "replay_episodes": float(pending_episodes),
            "update_epochs": 0.0,
            "updates_per_epoch": 0.0,
            "frames_per_batch": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "chosen_q_mean": 0.0,
            "chosen_q_std": 0.0,
            "terminal_frac": 0.0,
            "epsilon": float(epsilon),
            "skipped": 1.0,
            "pending_batch": 1.0,
        }

    def _epsilon_for_episode(self, episode: int) -> float:
        total = max(int(self.cfg.train_episodes), 1)
        anneal_episodes = max(int(round(total * float(self.cfg.exploration_fraction))), 1)
        progress = min(max((int(episode) - 1) / float(anneal_episodes), 0.0), 1.0)
        return float(self.cfg.eps_init) + progress * (float(self.cfg.eps_end) - float(self.cfg.eps_init))

    def _load_resume_checkpoint(self) -> int:
        ckpt_path = str(getattr(self.cfg, "resume_checkpoint", "") or "").strip()
        if not ckpt_path:
            return 0
        path = _resolve_path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {path}")
        state = torch.load(path, map_location=self.device)
        actor_state = state.get("q_network") or state.get("actor") or state.get("policy")
        if actor_state is None:
            raise KeyError(f"resume checkpoint has no q_network/actor/policy state: {path}")
        self.actor.load_state_dict(actor_state)
        loss_state = state.get("loss_module")
        if loss_state is not None:
            self.loss_module.load_state_dict(loss_state)
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        episode = int(state.get("episode", 0) or 0)
        print(f"[toy-iql-torchrl] resumed q-network from {path} at episode {episode}", flush=True)
        return episode

    def update_from_replay(self, episode_batches: Sequence[Sequence[OnlineDecisionSample]]) -> Dict[str, float]:
        samples = [sample for batch in episode_batches for sample in batch]
        epsilon = float(getattr(self.actor, "epsilon", 0.0))
        if len(samples) < int(self.cfg.min_samples_per_update):
            return {
                "loss": 0.0,
                "q_loss": 0.0,
                "n_samples": float(len(samples)),
                "replay_episodes": float(len(episode_batches)),
                "update_epochs": 0.0,
                "epsilon": epsilon,
                "skipped": 1.0,
            }

        next_indices, done_flags = _build_iql_next_indices(samples)
        active_total = int(len(samples))
        if active_total <= 0:
            return {
                "loss": 0.0,
                "q_loss": 0.0,
                "n_samples": 0.0,
                "replay_episodes": float(len(episode_batches)),
                "update_epochs": 0.0,
                "epsilon": epsilon,
                "skipped": 1.0,
            }

        frames = int(len(samples))
        minibatch_size = min(max(int(self.cfg.minibatch_size), 1), max(frames, 1))
        frames_per_batch = max(int(self.actor.n_agents) * max(int(self.cfg.agent_frames_per_batch), 1), 1)
        updates_per_epoch = max(frames_per_batch // minibatch_size, 1)
        update_epochs = max(int(self.cfg.num_epochs), 1)
        self.actor.train()
        self.loss_module.train()

        loss_values: List[float] = []
        for _epoch in range(update_epochs):
            for batch_indices in _random_minibatch_indices(
                n_samples=frames,
                minibatch_size=minibatch_size,
                num_batches=updates_per_epoch,
            ):
                subdata = _build_iql_minibatch_td(
                    samples=samples,
                    indices=batch_indices,
                    q_network=self.actor,
                    next_indices=next_indices,
                    done_flags=done_flags,
                    device=self.device,
                )
                active = subdata.get(("agents", "active_mask")).to(device=self.device, dtype=torch.bool)
                if not bool(active.any().detach().cpu().item()):
                    continue
                loss_vals = self.loss_module(subdata)
                q_loss = _loss_matrix(loss_vals.get("loss"), active.to(dtype=torch.float32))
                loss = _masked_mean(q_loss, active)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), float(self.cfg.max_grad_norm))
                self.optimizer.step()
                loss_values.append(float(loss.detach().cpu().item()))
            self.target_net_updater.step()

        active_rewards = np.asarray([float(sample.reward) for sample in samples], dtype=np.float32)
        active_done = done_flags.astype(np.float32)
        active_q = _chosen_q_for_samples(
            samples=samples,
            q_network=self.actor,
            chunk_size=minibatch_size,
            device=self.device,
        )

        def mean(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "loss": mean(loss_values),
            "q_loss": mean(loss_values),
            "n_samples": float(active_total),
            "replay_episodes": float(len(episode_batches)),
            "update_epochs": float(update_epochs),
            "updates_per_epoch": float(updates_per_epoch),
            "frames_per_batch": float(frames_per_batch),
            "reward_mean": float(np.mean(active_rewards)) if active_rewards.size else 0.0,
            "reward_std": float(np.std(active_rewards)) if active_rewards.size > 1 else 0.0,
            "chosen_q_mean": float(np.mean(active_q)) if active_q.size else 0.0,
            "chosen_q_std": float(np.std(active_q)) if active_q.size > 1 else 0.0,
            "terminal_frac": float(np.mean(active_done)) if active_done.size else 0.0,
            "epsilon": epsilon,
            "skipped": 0.0 if loss_values else 1.0,
        }

    def save_checkpoint(self, episode: int, metrics: Optional[Dict[str, Any]] = None) -> Path:
        path = self.output_dir / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_family": "iql_torchrl",
            "episode": int(episode),
            "q_network": self.actor.state_dict(),
            "actor": self.actor.state_dict(),
            "policy": self.actor.qnet.state_dict(),
            "loss_module": self.loss_module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg": _jsonable_config(self.cfg),
            "obs_dim": int(self.obs_dim),
            "machine_agent_ids": list(self.machine_ids),
            "metrics": dict(metrics or {}),
        }
        torch.save(state, path)
        return path

    def train(self) -> Dict[str, Any]:
        collector = self._build_collector()
        ckpt_eps = {int(v) for v in self.cfg.checkpoint_episodes}
        train_rows: List[Dict[str, Any]] = []
        update_rows: List[Dict[str, Any]] = []
        checkpoint_paths: Dict[int, str] = {}
        start_episode = int(getattr(self.cfg, "resume_start_episode", 0) or 0)
        if start_episode <= 0:
            start_episode = int(self.loaded_resume_episode)
        start_episode = max(0, int(start_episode))
        start = time.time()
        try:
            pending_batches: List[List[OnlineDecisionSample]] = []
            batch_episodes = max(int(self.cfg.agent_frames_per_batch), 1)
            for ep in range(start_episode + 1, int(self.cfg.train_episodes) + 1):
                epsilon = self._epsilon_for_episode(ep)
                self.actor.train()
                self.actor.set_epsilon(epsilon)
                samples, row = collector.run_train_episode(ep, int(self.rollout_max_steps))
                pending_batches.append(list(samples))
                is_last_episode = int(ep) >= int(self.cfg.train_episodes)
                should_update = (
                    len(pending_batches) >= batch_episodes
                    or ep in ckpt_eps
                    or is_last_episode
                )
                if should_update:
                    update = self.update_from_replay(pending_batches)
                    update["pending_batch"] = 0.0
                    pending_batches = []
                else:
                    pending_samples = sum(len(batch) for batch in pending_batches)
                    update = self._pending_update_metrics(
                        n_samples=pending_samples,
                        pending_episodes=len(pending_batches),
                        epsilon=epsilon,
                    )
                row.update({f"update_{k}": v for k, v in update.items()})
                train_rows.append(row)
                update_rows.append({"episode": ep, **update})
                print(
                    f"[toy-iql-torchrl] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={len(samples)} reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"loss={update['loss']:.4f} q={update.get('chosen_q_mean', 0.0):.4f} "
                    f"eps={update.get('epsilon', epsilon):.3f} "
                    f"t_all={row.get('t_all', float('nan')):.3f} "
                    f"t_CAV={row.get('t_CAV', float('nan')):.3f} t_HDV={row.get('t_HDV', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"unfinished={int(row.get('unfinished_all', 0))} "
                    f"batch={update.get('replay_episodes', 0.0):.0f}/{batch_episodes} "
                    f"pending={int(update.get('pending_batch', 0.0))}",
                    flush=True,
                )
                if ep in ckpt_eps:
                    ckpt = self.save_checkpoint(ep, metrics=row)
                    checkpoint_paths[int(ep)] = str(ckpt)
                    print(f"[toy-iql-torchrl] saved checkpoint: {ckpt}", flush=True)
        finally:
            try:
                collector.finalize_protocol_xml_aliases()
            finally:
                close = getattr(getattr(collector, "connector", None), "close", None)
                if callable(close):
                    close()

        pd.DataFrame(train_rows).to_csv(self.output_dir / "online_train_episodes.csv", index=False)
        pd.DataFrame(update_rows).to_csv(self.output_dir / "online_train_updates.csv", index=False)
        decision_df = pd.DataFrame(getattr(collector, "_dynamic_decision_rows", []))
        if len(decision_df) > 0:
            decision_df.to_csv(self.output_dir / "online_train_decisions.csv", index=False)
        agent_df = pd.DataFrame(getattr(collector, "_dynamic_agent_decision_rows", []))
        if len(agent_df) > 0:
            agent_df.to_csv(self.output_dir / "online_train_agent_decisions.csv", index=False)
        summary = {
            "baseline": "toy_cross_bottleneck_iql_torchrl",
            "algorithm": "iql_torchrl",
            "config": _jsonable_config(self.cfg),
            "torchrl_mapping": {
                "share_parameters_policy": False,
                "centralized_q_network": False,
                "value_estimator": "td0",
                "loss": "DQNLoss",
                "target_update": "SoftUpdate",
                "agent_frames_per_batch": int(self.cfg.agent_frames_per_batch),
                "gamma": float(self.cfg.gamma),
                "eps_init": float(self.cfg.eps_init),
                "eps_end": float(self.cfg.eps_end),
                "exploration_fraction": float(self.cfg.exploration_fraction),
                "target_update_eps": float(self.cfg.target_update_eps),
            },
            "fixed_roles": {
                "source": str(self.role_source),
                "machine_count": len(self.machine_ids),
                "human_count": len(self.human_ids),
                "machine_agent_ids": list(self.machine_ids),
                "human_agent_ids": list(self.human_ids),
            },
            "obs_dim": int(self.obs_dim),
            "obs_source": "live_edge_registry_extract_flow",
            "checkpoint_paths": checkpoint_paths,
            "train_episode_csv": str(self.output_dir / "online_train_episodes.csv"),
            "train_update_csv": str(self.output_dir / "online_train_updates.csv"),
            "elapsed_sec": float(time.time() - start),
        }
        with (self.output_dir / "online_train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def evaluate(self, *, eval_episodes: int) -> Dict[str, Any]:
        collector = self._build_collector()
        collector._av_execution_mode = "dynamic"
        self.actor.eval()
        self.actor.set_epsilon(0.0)
        rows: List[Dict[str, Any]] = []
        started = time.time()
        start_episode = int(getattr(self.cfg, "resume_start_episode", 0) or 0)
        if start_episode <= 0:
            start_episode = int(self.loaded_resume_episode)
        try:
            for idx in range(int(eval_episodes)):
                episode = int(start_episode) + idx + 1
                _samples, row = collector.run_train_episode(episode, int(self.rollout_max_steps))
                row["eval_mode"] = "dynamic"
                rows.append(row)
                print(
                    f"[toy-iql-torchrl-eval] ep={idx + 1:03d}/{int(eval_episodes):03d} "
                    f"t_all={row.get('t_all', float('nan')):.3f} "
                    f"t_CAV={row.get('t_CAV', float('nan')):.3f} "
                    f"t_HDV={row.get('t_HDV', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f}",
                    flush=True,
                )
        finally:
            try:
                collector.finalize_protocol_xml_aliases()
            finally:
                close = getattr(getattr(collector, "connector", None), "close", None)
                if callable(close):
                    close()

        pd.DataFrame(rows).to_csv(self.output_dir / "dynamic_episodes.csv", index=False)
        decision_df = pd.DataFrame(getattr(collector, "_dynamic_decision_rows", []))
        if not decision_df.empty:
            decision_df.to_csv(self.output_dir / "dynamic_decisions.csv", index=False)
        agent_df = pd.DataFrame(getattr(collector, "_dynamic_agent_decision_rows", []))
        if not agent_df.empty:
            agent_df.to_csv(self.output_dir / "dynamic_agent_decisions.csv", index=False)

        summary = {
            "checkpoint": str(self.cfg.resume_checkpoint),
            "post_mutation_state_in": str(self.cfg.post_mutation_state_in),
            "hdv_state_in": str(self.cfg.hdv_state_in),
            "eval_episodes": int(eval_episodes),
            "dynamic_routing": True,
            "hdv_updates_disabled": bool(self.cfg.disable_human_learning_after_mutation),
            "human_learning_after_mutation": not bool(self.cfg.disable_human_learning_after_mutation),
            "summary": _mean_complete(rows),
            "episodes_csv": str(self.output_dir / "dynamic_episodes.csv"),
            "decisions_csv": str(self.output_dir / "dynamic_decisions.csv"),
            "agent_decisions_csv": str(self.output_dir / "dynamic_agent_decisions.csv"),
            "elapsed_sec": float(time.time() - started),
        }
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def build_config(args: argparse.Namespace) -> ToyIqlTorchRLConfig:
    network_dir = Path(args.network_dir).resolve()
    has_settings = bool(str(args.settings or "").strip())
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT
        / "results"
        / ("from_settings" if has_settings else TOY_NETWORK_NAME)
        / f"iql_torchrl_fixed_role_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    train_episodes = int(args.n_iters) if int(args.n_iters) > 0 else int(args.train_episodes)
    network_overrides = (
        {}
        if has_settings
        else dict(
            tag=TOY_NETWORK_NAME,
            network_name=TOY_NETWORK_NAME,
            network_dir=str(network_dir),
            edge_registry=str(network_dir / "edge_registry.json"),
            projector_ckpt="",
            dynamics_ckpt="",
            route_artifacts_dir=str(network_dir),
            ratio_machines=float(args.ratio_machines),
            n_routes=4,
            episode_steps=int(args.episode_steps),
            sim_buffer_after_last_depart=int(args.sim_buffer_after_last_depart),
            hdv_gawron_mode=str(args.hdv_gawron_mode),
            hdv_gawron_alpha_zero=float(args.hdv_gawron_alpha_zero),
            hdv_gawron_beta=float(args.hdv_gawron_beta),
            hdv_gawron_update_mode=str(args.hdv_gawron_update_mode),
            incident_edge=str(args.incident_edge or ""),
            incident_start=float(args.incident_start),
            incident_end=float(args.incident_end),
            incident_speed=float(args.incident_speed),
        )
    )
    overrides = dict(
        **network_overrides,
        output_dir=str(output_dir),
        post_mutation_state_in=str(getattr(args, "post_mutation_state_in", "") or ""),
        hdv_state_in=str(args.hdv_state_in or ""),
        train_episodes=int(train_episodes),
        n_iters=int(train_episodes),
        eval_episodes=int(args.eval_episodes),
        checkpoint_episodes=_checkpoint_episodes_from_args(
            raw_episodes=str(args.checkpoint_episodes or ""),
            train_episodes=int(train_episodes),
            checkpoint_interval=int(args.checkpoint_interval),
        ),
        checkpoint_interval=int(args.checkpoint_interval),
        resume_checkpoint=str(args.resume_checkpoint or ""),
        resume_start_episode=int(args.resume_start_episode),
        max_steps_per_episode=int(args.max_steps_per_episode),
        seed=int(args.seed),
        device=str(args.device),
        reward_mode=str(args.reward_mode),
        reward_temperature=float(args.reward_temperature),
        gamma=float(args.gamma),
        lr=float(args.lr),
        replay_buffer_episodes=int(args.replay_buffer_episodes),
        replay_update_epochs=int(args.num_epochs),
        replay_actor_max_age_episodes=int(args.replay_actor_max_age_episodes),
        return_horizon_steps=0,
        min_samples_per_update=1,
        max_grad_norm=float(args.max_grad_norm),
        disable_human_learning_after_mutation=False,
        sumo_time_to_teleport=str(args.sumo_time_to_teleport),
        sumo_extra_args=str(args.sumo_extra_args or ""),
        agent_frames_per_batch=int(args.agent_frames_per_batch),
        num_epochs=int(args.num_epochs),
        minibatch_size=int(args.minibatch_size),
        q_network_depth=int(args.q_network_depth),
        q_network_num_cells=int(args.q_network_num_cells),
        eps_init=float(args.eps_init),
        eps_end=float(args.eps_end),
        exploration_fraction=float(args.exploration_fraction),
        target_update_eps=float(args.target_update_eps),
        av_execution_mode=str(args.av_execution_mode),
    )
    if has_settings:
        cfg = ToyIqlTorchRLConfig.from_settings(str(args.settings), **overrides)
    else:
        cfg = ToyIqlTorchRLConfig(settings="", **overrides)
    return cfg.resolve()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default="",
        help="Optional URB settings env file used to align toy IQL with WM/no-CFM runs.",
    )
    parser.add_argument("--network-dir", default=str(REPO_ROOT / "networks" / TOY_NETWORK_NAME))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--generate-network", action="store_true", help="Generate toy network artifacts before training.")
    parser.add_argument("--n-agents", type=int, default=320)
    parser.add_argument("--depart-span", type=int, default=420)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument(
        "--checkpoint-episodes",
        default="",
        help="Comma-separated checkpoint episodes. Empty uses --checkpoint-interval.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoints every N training episodes when --checkpoint-episodes is empty; <=0 disables interval checkpoints.",
    )
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--resume-start-episode", type=int, default=0)
    parser.add_argument("--post-mutation-state-in", default="")
    parser.add_argument("--hdv-state-in", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio-machines", type=float, default=0.4)
    parser.add_argument("--episode-steps", type=int, default=60)
    parser.add_argument("--sim-buffer-after-last-depart", type=int, default=3000)
    parser.add_argument("--max-steps-per-episode", type=int, default=0)
    parser.add_argument("--reward-mode", default="capped_step_cost", choices=["progress_fraction", "capped_step_cost"])
    parser.add_argument("--reward-temperature", type=float, default=60.0)
    parser.add_argument("--replay-buffer-episodes", type=int, default=8)
    parser.add_argument("--replay-actor-max-age-episodes", type=int, default=8)
    parser.add_argument("--av-execution-mode", default="dynamic", choices=["dynamic", "first_decision_only"])
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=3.0)
    parser.add_argument("--agent-frames-per-batch", type=int, default=1)
    parser.add_argument(
        "--n-iters",
        type=int,
        default=0,
        help="URB iql_torchrl iteration alias. When >0, overrides --train-episodes.",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-network-depth", type=int, default=3)
    parser.add_argument("--q-network-num-cells", type=int, default=64)
    parser.add_argument("--eps-init", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.8)
    parser.add_argument("--target-update-eps", type=float, default=0.995)
    parser.add_argument("--hdv-gawron-mode", default="routerl")
    parser.add_argument("--hdv-gawron-alpha-zero", type=float, default=0.2)
    parser.add_argument("--hdv-gawron-beta", type=float, default=-1.5)
    parser.add_argument("--hdv-gawron-update-mode", default="expected")
    parser.add_argument("--incident-edge", default="")
    parser.add_argument("--incident-start", type=float, default=0.0)
    parser.add_argument("--incident-end", type=float, default=0.0)
    parser.add_argument("--incident-speed", type=float, default=0.0)
    parser.add_argument("--sumo-time-to-teleport", default="-1")
    parser.add_argument("--sumo-extra-args", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_plan(cfg: ToyIqlTorchRLConfig) -> Dict[str, Any]:
    return {
        "baseline": "toy_cross_bottleneck_iql_torchrl",
        "network": cfg.network_name,
        "setting_match": {
            "reward_mode": cfg.reward_mode,
            "reward_temperature": cfg.reward_temperature,
            "ratio_machines": cfg.ratio_machines,
            "episode_steps": cfg.episode_steps,
            "av_execution_mode": str(cfg.av_execution_mode),
            "av_candidate_source": str(cfg.av_candidate_source),
            "route_generator": str(cfg.route_generator),
            "path_gen_beta": float(cfg.path_gen_beta),
            "path_num_samples": int(cfg.path_num_samples),
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "hdv_state_in": cfg.hdv_state_in,
            "replay_buffer_episodes": cfg.replay_buffer_episodes,
            "replay_update_epochs": cfg.num_epochs,
            "return_horizon_steps": cfg.return_horizon_steps,
            "hdv_learning": cfg.hdv_gawron_mode,
            "human_learning_after_mutation": not bool(cfg.disable_human_learning_after_mutation),
            "incident_edge": cfg.incident_edge,
            "incident_start": cfg.incident_start,
            "incident_end": cfg.incident_end,
            "incident_speed": cfg.incident_speed,
        },
        "iql_torchrl": {
            "share_parameters_policy": False,
            "centralized_q_network": False,
            "agent_frames_per_batch": int(cfg.agent_frames_per_batch),
            "n_iters": int(cfg.n_iters or cfg.train_episodes),
            "checkpoint_interval": int(cfg.checkpoint_interval),
            "checkpoint_episodes": list(cfg.checkpoint_episodes),
            "num_epochs": int(cfg.num_epochs),
            "minibatch_size": int(cfg.minibatch_size),
            "lr": float(cfg.lr),
            "max_grad_norm": float(cfg.max_grad_norm),
            "gamma": float(cfg.gamma),
            "q_network_depth": int(cfg.q_network_depth),
            "q_network_num_cells": int(cfg.q_network_num_cells),
            "eps_init": float(cfg.eps_init),
            "eps_end": float(cfg.eps_end),
            "exploration_fraction": float(cfg.exploration_fraction),
            "target_update_eps": float(cfg.target_update_eps),
        },
        "paths": {
            "network_dir": cfg.network_dir,
            "edge_registry": cfg.edge_registry,
            "route_artifacts_dir": cfg.route_artifacts_dir,
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "output_dir": cfg.output_dir,
        },
    }


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    network_dir = Path(args.network_dir).resolve()
    if args.generate_network:
        generate_toy_network(
            output_dir=network_dir,
            n_agents=int(args.n_agents),
            depart_span=int(args.depart_span),
            netconvert="",
        )
    cfg = build_config(args)
    plan = build_plan(cfg)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    validate_paths(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "toy_iql_torchrl_config.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable_config(cfg), f, indent=2)
    with (output_dir / "toy_iql_torchrl_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    trainer = ToyIqlTorchRLTrainer(cfg)
    if args.eval_only:
        summary = trainer.evaluate(eval_episodes=int(args.eval_episodes))
        final_name = "toy_iql_torchrl_eval_summary.json"
    else:
        summary = trainer.train()
        final_name = "toy_iql_torchrl_final_summary.json"
    with (output_dir / final_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
