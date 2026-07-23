#!/usr/bin/env python3
"""Run a TorchRL-style IQL baseline on the toy cross-bottleneck network."""

from __future__ import annotations

import argparse
import json
import math
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
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import ListStorage
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

from baseline.create_toy_cross_bottleneck_network import NETWORK_NAME as TOY_NETWORK_NAME
from baseline.create_toy_cross_bottleneck_network import generate as generate_toy_network
from baseline.online_actor_critic import (
    OnlineDecisionSample,
    OnlineNoCfmConfig,
    _resolve_path,
)
from baseline.run_toy_cross_bottleneck_online_ac import (
    ToyOnlineAcTrainer,
    parse_checkpoint_episodes,
    validate_paths,
)
from baseline.eval_toy_online_ac_dynamic import _mean_complete
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
    memory_size: int = 20_000
    replay_sample_ratio: float = 0.0
    fresh_transition_epochs: int = 0
    q_network_depth: int = 3
    q_network_num_cells: int = 64
    eps_init: float = 0.99
    eps_end: float = 0.05
    exploration_fraction: float = 0.4
    epsilon_scope: str = "per_decision"
    target_update_eps: float = 0.995
    av_execution_mode: str = "dynamic"

    def resolve(self) -> "ToyIqlTorchRLConfig":
        values = asdict(self)
        if int(values["memory_size"]) <= 0:
            raise ValueError("memory_size must be positive")
        replay_sample_ratio = float(values["replay_sample_ratio"])
        if not math.isfinite(replay_sample_ratio) or replay_sample_ratio < 0.0:
            raise ValueError("replay_sample_ratio must be finite and non-negative")
        if int(values["fresh_transition_epochs"]) < 0:
            raise ValueError("fresh_transition_epochs must be non-negative")
        epsilon_scope = str(values["epsilon_scope"]).strip().lower()
        if epsilon_scope not in {"per_decision", "first_decision_per_vehicle"}:
            raise ValueError(
                "epsilon_scope must be 'per_decision' or "
                "'first_decision_per_vehicle'"
            )
        values["epsilon_scope"] = epsilon_scope
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


@dataclass(frozen=True)
class IqlReplayTransition:
    episode: int
    decision_idx: int
    veh_id: str
    obs: np.ndarray
    action: int
    action_mask: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_action_mask: np.ndarray
    done: bool
    steps_to_next_obs: int = 1


def _list_collate(samples: Sequence[IqlReplayTransition]) -> List[IqlReplayTransition]:
    return list(samples)


class IqlTransitionReplayBuffer:
    """Compact materialized replay with the official TorchRL ring/sampler semantics."""

    def __init__(self, *, memory_size: int, minibatch_size: int) -> None:
        self.memory_size = int(memory_size)
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        self._buffer = ReplayBuffer(
            storage=ListStorage(self.memory_size),
            sampler=SamplerWithoutReplacement(drop_last=False),
            batch_size=max(int(minibatch_size), 1),
            collate_fn=_list_collate,
        )

    def __len__(self) -> int:
        return int(len(self._buffer))

    @property
    def sampler_name(self) -> str:
        return type(self._buffer.sampler).__name__

    def extend(self, samples: Sequence[IqlReplayTransition]) -> None:
        values = list(samples)
        if values:
            self._buffer.extend(values)

    def snapshot(self) -> List[IqlReplayTransition]:
        if not len(self):
            return []
        return list(self._buffer.storage[:])

    def reset_sampling_cycle(self) -> None:
        """Start a new permutation after ring-buffer contents may have changed."""
        self._buffer.set_sampler(SamplerWithoutReplacement(drop_last=False))

    def sample_indices(self, *, batch_size: int) -> np.ndarray:
        if not len(self):
            return np.empty((0,), dtype=np.int64)
        _samples, info = self._buffer.sample(
            batch_size=max(int(batch_size), 1),
            return_info=True,
        )
        indices = info["index"]
        if torch.is_tensor(indices):
            indices = indices.detach().cpu().numpy()
        return np.asarray(indices, dtype=np.int64).reshape(-1)


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
        self.epsilon_scope = str(cfg.epsilon_scope)
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
        epsilon_by_row: Optional[torch.Tensor] = None,
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
        if not self.training:
            eps_by_row = torch.zeros(
                (int(q_values.shape[0]), 1),
                device=q_values.device,
                dtype=q_values.dtype,
            )
        elif epsilon_by_row is not None:
            eps_by_row = torch.as_tensor(
                epsilon_by_row,
                device=q_values.device,
                dtype=q_values.dtype,
            ).reshape(-1, 1)
            if int(eps_by_row.shape[0]) != int(q_values.shape[0]):
                raise ValueError(
                    "epsilon_by_row length does not match active observation rows"
                )
            eps_by_row = eps_by_row.clamp(0.0, 1.0)
        else:
            eps_by_row = torch.full(
                (int(q_values.shape[0]), 1),
                fill_value=float(self.epsilon),
                device=q_values.device,
                dtype=q_values.dtype,
            )
        greedy = torch.argmax(q_values, dim=-1, keepdim=True)
        valid_count = safe_mask.sum(dim=-1, keepdim=True).to(dtype=q_values.dtype).clamp_min(1.0)
        probs = safe_mask.to(dtype=q_values.dtype) * (eps_by_row / valid_count)
        probs.scatter_add_(
            dim=-1,
            index=greedy,
            src=1.0 - eps_by_row,
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


def _annotate_iql_samples_with_arrival_remaining(
    samples: Sequence[OnlineDecisionSample],
    vehicle_travel_times: Dict[str, float],
) -> None:
    """Attach the observed post-decision time-to-arrival to each sample."""
    for sample in samples:
        travel_time = float(vehicle_travel_times.get(str(sample.veh_id), float("nan")))
        elapsed = float(getattr(sample, "elapsed_time", float("nan")))
        if np.isfinite(travel_time) and np.isfinite(elapsed):
            # TraCI reports the arrival event on the tick after tripinfo's
            # timestamp. This is the same alignment used by the online-WM path.
            sample.arrival_remaining_time = float(max(travel_time - elapsed - 1.0, 0.0))
        else:
            sample.arrival_remaining_time = float("nan")


def _discounted_repeated_reward(
    base_reward: float,
    *,
    gamma: float,
    full_steps: int,
    final_fraction: float = 0.0,
) -> float:
    """Collapse repeated fixed-interval rewards into one SMDP reward."""
    value = 0.0
    discount = 1.0
    for _ in range(max(int(full_steps), 0)):
        value += discount * float(base_reward)
        discount *= float(gamma)
    fraction = float(np.clip(float(final_fraction), 0.0, 1.0))
    if fraction > 0.0:
        value += discount * float(base_reward) * fraction
    return float(value)


def _full_interval_reward(sample: OnlineDecisionSample, cfg: ToyIqlTorchRLConfig) -> float:
    """Return the configured reward for one complete decision interval."""
    mode = str(cfg.reward_mode or "").strip().lower()
    if mode in {"capped_step_cost", "capped_cost", "step_cost", "min_step_remaining"}:
        mask = np.asarray(sample.valid_mask, dtype=bool).reshape(-1)
        action = int(sample.action)
        if action < 0 or action >= int(mask.size) or not bool(mask[action]):
            return 0.0
        return float(
            -float(cfg.reward_weight)
            * float(cfg.reward_scale)
            * float(cfg.step_interval)
            / max(float(cfg.reward_temperature), 1e-6)
        )
    # Other reward modes are state dependent. Repeating the observed reward is
    # the least-assumptive SMDP approximation; the production runs use the
    # exact capped-step-cost branch above.
    return float(sample.reward)


def _decision_gap_steps(
    sample: OnlineDecisionSample,
    next_sample: OnlineDecisionSample,
    *,
    step_interval: float,
) -> int:
    decision_gap = int(next_sample.decision_idx) - int(sample.decision_idx)
    sim_gap = float(next_sample.sim_step) - float(sample.sim_step)
    sim_gap_steps = int(round(sim_gap / max(float(step_interval), 1e-6)))
    return max(1, decision_gap, sim_gap_steps)


def _materialize_iql_replay_transitions(
    samples: Sequence[OnlineDecisionSample],
    *,
    n_routes: int,
    cfg: Optional[ToyIqlTorchRLConfig] = None,
) -> List[IqlReplayTransition]:
    if not samples:
        return []
    next_indices, done_flags = _build_iql_next_indices(samples)
    transitions: List[IqlReplayTransition] = []
    for sample_idx, sample in enumerate(samples):
        obs = np.asarray(sample.obs, dtype=np.float32).reshape(-1).copy()
        action_mask = np.zeros((int(n_routes),), dtype=bool)
        raw_mask = np.asarray(sample.valid_mask, dtype=bool).reshape(-1)
        copy_len = min(int(raw_mask.shape[0]), int(n_routes))
        if copy_len > 0:
            action_mask[:copy_len] = raw_mask[:copy_len]
        if not bool(action_mask.any()):
            action_mask[:] = True

        done = bool(done_flags[sample_idx])
        steps_to_next_obs = 1
        reward = float(sample.reward)
        if done:
            if cfg is not None:
                arrival_remaining = float(
                    getattr(sample, "arrival_remaining_time", float("nan"))
                )
                if not np.isfinite(arrival_remaining):
                    # A censored tail has neither a successor state nor a
                    # trustworthy terminal return. Excluding it avoids the old
                    # optimistic one-step pseudo-terminal target.
                    continue
                interval = max(float(cfg.step_interval), 1e-6)
                full_steps = int(math.floor(max(arrival_remaining, 0.0) / interval))
                remainder = max(arrival_remaining - full_steps * interval, 0.0)
                final_fraction = remainder / interval
                steps_to_next_obs = max(full_steps + int(final_fraction > 1e-9), 1)
                reward = _discounted_repeated_reward(
                    _full_interval_reward(sample, cfg),
                    gamma=float(cfg.gamma),
                    full_steps=full_steps,
                    final_fraction=final_fraction,
                )
            next_obs = np.zeros_like(obs)
            next_action_mask = np.ones((int(n_routes),), dtype=bool)
        else:
            next_sample = samples[int(next_indices[sample_idx])]
            if cfg is not None:
                steps_to_next_obs = _decision_gap_steps(
                    sample,
                    next_sample,
                    step_interval=float(cfg.step_interval),
                )
                reward = _discounted_repeated_reward(
                    _full_interval_reward(sample, cfg),
                    gamma=float(cfg.gamma),
                    full_steps=steps_to_next_obs,
                )
            next_obs = np.asarray(next_sample.obs, dtype=np.float32).reshape(-1).copy()
            next_action_mask = np.zeros((int(n_routes),), dtype=bool)
            raw_next_mask = np.asarray(next_sample.valid_mask, dtype=bool).reshape(-1)
            next_copy_len = min(int(raw_next_mask.shape[0]), int(n_routes))
            if next_copy_len > 0:
                next_action_mask[:next_copy_len] = raw_next_mask[:next_copy_len]
            if not bool(next_action_mask.any()):
                next_action_mask[:] = True

        transitions.append(
            IqlReplayTransition(
                episode=int(sample.episode),
                decision_idx=int(sample.decision_idx),
                veh_id=str(sample.veh_id),
                obs=obs,
                action=int(sample.action),
                action_mask=action_mask,
                reward=float(reward),
                next_obs=next_obs,
                next_action_mask=next_action_mask,
                done=done,
                steps_to_next_obs=int(steps_to_next_obs),
            )
        )
    return transitions


def _build_iql_minibatch_td(
    *,
    transitions: Sequence[IqlReplayTransition],
    indices: Sequence[int],
    q_network: TorchRLIqlQNetwork,
    device: torch.device,
) -> TensorDict:
    n = int(len(indices))
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
    steps_to_next_obs = torch.ones((n, n_agents, 1), device=device, dtype=torch.long)

    for row, raw_idx in enumerate(indices):
        transition = transitions[int(raw_idx)]
        slot = q_network.slot_for_vehicle(str(transition.veh_id))
        current_obs = np.asarray(transition.obs, dtype=np.float32).reshape(-1)
        successor_obs = np.asarray(transition.next_obs, dtype=np.float32).reshape(-1)
        if int(current_obs.shape[0]) != obs_dim or int(successor_obs.shape[0]) != obs_dim:
            raise ValueError("replay transition observation does not match q-network obs_dim")
        obs[row, slot] = torch.as_tensor(current_obs, device=device, dtype=torch.float32)
        next_obs[row, slot] = torch.as_tensor(successor_obs, device=device, dtype=torch.float32)
        action[row, slot] = int(np.clip(int(transition.action), 0, n_routes - 1))
        action_mask[row, slot] = torch.as_tensor(
            transition.action_mask,
            device=device,
            dtype=torch.bool,
        )
        next_action_mask[row, slot] = torch.as_tensor(
            transition.next_action_mask,
            device=device,
            dtype=torch.bool,
        )
        active_mask[row, slot] = True
        rewards[row, slot, 0] = float(transition.reward)
        steps_to_next_obs[row, slot, 0] = max(int(transition.steps_to_next_obs), 1)
        if not bool(transition.done):
            done[row, slot, 0] = False
            terminated[row, slot, 0] = False

    return TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): action,
            ("agents", "action_mask"): action_mask,
            ("agents", "active_mask"): active_mask,
            "steps_to_next_obs": steps_to_next_obs,
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
    cfg: Optional[ToyIqlTorchRLConfig] = None,
) -> TensorDict:
    n = int(len(samples))
    if n <= 0:
        raise ValueError("cannot build an IQL TorchRL batch with no samples")
    transitions = _materialize_iql_replay_transitions(
        samples,
        n_routes=int(q_network.n_routes),
        cfg=cfg,
    )
    if not transitions:
        raise ValueError("cannot build an IQL TorchRL batch with no usable transitions")
    return _build_iql_minibatch_td(
        transitions=transitions,
        indices=np.arange(len(transitions), dtype=np.int64),
        q_network=q_network,
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


def _online_target_relative_l2(loss_module: DQNLoss) -> float:
    """Measure target lag without flattening the multi-agent parameter tensors."""
    gap_sq = 0.0
    online_sq = 0.0
    with torch.no_grad():
        online_params = loss_module.value_network_params
        target_params = loss_module.target_value_network_params
        for key in online_params.keys(True, True):
            online = online_params.get(key)
            target = target_params.get(key)
            if not torch.is_tensor(online) or not online.is_floating_point():
                continue
            gap_sq += float(torch.sum((online - target) ** 2).detach().cpu().item())
            online_sq += float(torch.sum(online**2).detach().cpu().item())
    return float(math.sqrt(gap_sq / max(online_sq, 1e-24)))


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
        self.replay_buffer = IqlTransitionReplayBuffer(
            memory_size=int(self.cfg.memory_size),
            minibatch_size=int(self.cfg.minibatch_size),
        )
        self.collector = None
        self.loaded_resume_episode = self._load_resume_checkpoint()

    def _ensure_replay_buffer(self) -> IqlTransitionReplayBuffer:
        replay_buffer = getattr(self, "replay_buffer", None)
        if replay_buffer is None:
            replay_buffer = IqlTransitionReplayBuffer(
                memory_size=int(self.cfg.memory_size),
                minibatch_size=int(self.cfg.minibatch_size),
            )
            self.replay_buffer = replay_buffer
        return replay_buffer

    @staticmethod
    def _pending_update_metrics(
        *,
        n_samples: int,
        pending_episodes: int,
        epsilon: float,
        replay_size: int = 0,
        replay_episodes: int = 0,
        replay_sample_ratio: float = 0.0,
        fresh_transition_epochs: int = 0,
    ) -> Dict[str, float]:
        return {
            "loss": 0.0,
            "q_loss": 0.0,
            "n_samples": float(n_samples),
            "new_samples": float(n_samples),
            "new_transitions": float(n_samples),
            "replay_size": float(replay_size),
            "sampled_transitions": 0.0,
            "unique_sampled_transitions": 0.0,
            "sampled_to_new_ratio": 0.0,
            "fresh_sampled_transitions": 0.0,
            "fresh_unique_transitions": 0.0,
            "fresh_coverage": 0.0,
            "replay_sampled_transitions": 0.0,
            "replay_sample_ratio": float(replay_sample_ratio),
            "fresh_transition_epochs": float(fresh_transition_epochs),
            "sample_budget_per_epoch": 0.0,
            "fresh_sample_budget_per_epoch": 0.0,
            "replay_sample_budget_per_epoch": 0.0,
            "target_sampled_transitions": 0.0,
            "fresh_retained_transitions": 0.0,
            "fresh_evicted_before_update": 0.0,
            "optimizer_steps": 0.0,
            "target_update_steps": 0.0,
            "target_old_weight_after_update": 1.0,
            "online_target_relative_l2": 0.0,
            "censored_tail_dropped": 0.0,
            "mean_steps_to_next_obs": 0.0,
            "multi_step_transition_frac": 0.0,
            "sampler_resets": 0.0,
            "replay_episodes": float(replay_episodes),
            "pending_episodes": float(pending_episodes),
            "update_epochs": 0.0,
            "replay_update_epochs": 0.0,
            "combined_update_epochs": 0.0,
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
            if self._optimizer_state_shapes_match(optimizer_state):
                self.optimizer.load_state_dict(optimizer_state)
            else:
                print(
                    "[toy-iql-torchrl] resume optimizer state shape mismatch; "
                    "continuing with a fresh optimizer",
                    flush=True,
                )
        episode = int(state.get("episode", 0) or 0)
        print(f"[toy-iql-torchrl] resumed q-network from {path} at episode {episode}", flush=True)
        return episode

    def _optimizer_state_shapes_match(self, optimizer_state: Dict[str, Any]) -> bool:
        state_by_param = optimizer_state.get("state", {})
        saved_groups = optimizer_state.get("param_groups", [])
        current_groups = self.optimizer.param_groups
        if not isinstance(state_by_param, dict) or len(saved_groups) != len(current_groups):
            return False
        for saved_group, current_group in zip(saved_groups, current_groups):
            saved_params = list(saved_group.get("params", []))
            current_params = list(current_group.get("params", []))
            if len(saved_params) != len(current_params):
                return False
            for saved_param_id, current_param in zip(saved_params, current_params):
                saved_state = state_by_param.get(saved_param_id, {})
                if not isinstance(saved_state, dict):
                    continue
                current_shape = tuple(current_param.shape)
                for value in saved_state.values():
                    if torch.is_tensor(value) and value.ndim > 0 and tuple(value.shape) != current_shape:
                        return False
        return True

    def update_from_replay(self, episode_batches: Sequence[Sequence[OnlineDecisionSample]]) -> Dict[str, float]:
        new_samples = [sample for batch in episode_batches for sample in batch]
        new_transitions = _materialize_iql_replay_transitions(
            new_samples,
            n_routes=int(self.actor.n_routes),
            cfg=self.cfg,
        )
        replay_buffer = self._ensure_replay_buffer()
        replay_buffer.extend(new_transitions)
        replay_transitions = replay_buffer.snapshot()
        epsilon = float(getattr(self.actor, "epsilon", 0.0))
        fresh_count = int(len(new_transitions))
        replay_sample_ratio = float(self.cfg.replay_sample_ratio)
        fresh_transition_epochs = int(self.cfg.fresh_transition_epochs)
        fresh_retained = min(fresh_count, int(replay_buffer.memory_size))
        fresh_evicted = max(fresh_count - int(replay_buffer.memory_size), 0)
        if len(new_samples) < int(self.cfg.min_samples_per_update) or not replay_transitions:
            return {
                "loss": 0.0,
                "q_loss": 0.0,
                "n_samples": float(len(new_samples)),
                "new_samples": float(len(new_samples)),
                "new_transitions": float(fresh_count),
                "replay_size": float(len(replay_transitions)),
                "sampled_transitions": 0.0,
                "unique_sampled_transitions": 0.0,
                "sampled_to_new_ratio": 0.0,
                "fresh_sampled_transitions": 0.0,
                "fresh_unique_transitions": 0.0,
                "fresh_coverage": 0.0,
                "replay_sampled_transitions": 0.0,
                "replay_sample_ratio": replay_sample_ratio,
                "fresh_transition_epochs": float(fresh_transition_epochs),
                "sample_budget_per_epoch": 0.0,
                "fresh_sample_budget_per_epoch": 0.0,
                "replay_sample_budget_per_epoch": 0.0,
                "target_sampled_transitions": 0.0,
                "fresh_retained_transitions": float(fresh_retained),
                "fresh_evicted_before_update": float(fresh_evicted),
                "optimizer_steps": 0.0,
                "target_update_steps": 0.0,
                "target_old_weight_after_update": 1.0,
                "online_target_relative_l2": _online_target_relative_l2(
                    self.loss_module
                ),
                "censored_tail_dropped": float(max(len(new_samples) - fresh_count, 0)),
                "mean_steps_to_next_obs": 0.0,
                "multi_step_transition_frac": 0.0,
                "sampler_resets": 0.0,
                "replay_episodes": float(len({transition.episode for transition in replay_transitions})),
                "pending_episodes": 0.0,
                "update_epochs": 0.0,
                "replay_update_epochs": 0.0,
                "combined_update_epochs": 0.0,
                "updates_per_epoch": 0.0,
                "frames_per_batch": 0.0,
                "epsilon": epsilon,
                "skipped": 1.0,
            }

        minibatch_size = max(int(self.cfg.minibatch_size), 1)
        ratio_budget = replay_sample_ratio > 0.0
        update_epochs = max(int(self.cfg.num_epochs), 1)
        fresh_coverage_mode = fresh_transition_epochs > 0
        replay_sample_budget = (
            int(math.ceil(float(fresh_count) * replay_sample_ratio)) if ratio_budget else 0
        )
        if fresh_coverage_mode:
            legacy_updates_per_epoch = 0
            replay_update_epochs = update_epochs if ratio_budget else 0
            combined_update_epochs = max(fresh_transition_epochs, replay_update_epochs)
            fresh_sample_budget = fresh_count
            frames_per_batch = fresh_sample_budget + replay_sample_budget
            target_sampled_transitions = (
                fresh_sample_budget * fresh_transition_epochs
                + replay_sample_budget * replay_update_epochs
            )
        else:
            fresh_sample_budget = 0
            replay_update_epochs = update_epochs
            combined_update_epochs = update_epochs
            if ratio_budget:
                frames_per_batch = replay_sample_budget
                legacy_updates_per_epoch = 0
                target_sampled_transitions = replay_sample_budget * update_epochs
            else:
                frames_per_batch = max(
                    int(self.actor.n_agents) * max(int(self.cfg.agent_frames_per_batch), 1),
                    1,
                )
                legacy_updates_per_epoch = frames_per_batch // minibatch_size
                target_sampled_transitions = (
                    legacy_updates_per_epoch * minibatch_size * update_epochs
                )
        self.actor.train()
        self.loss_module.train()

        loss_values: List[float] = []
        fresh_sampled_transitions = 0
        fresh_unique_indices: set[int] = set()
        replay_sampled_transitions = 0
        replay_unique_sampled_transitions = 0
        sampler_resets = 0
        optimizer_steps_per_epoch: List[int] = []

        target_update_steps = 0

        def optimize_batch(
            transitions: Sequence[IqlReplayTransition],
            batch_indices: Sequence[int],
        ) -> bool:
            nonlocal target_update_steps
            subdata = _build_iql_minibatch_td(
                transitions=transitions,
                indices=batch_indices,
                q_network=self.actor,
                device=self.device,
            )
            active = subdata.get(("agents", "active_mask")).to(
                device=self.device,
                dtype=torch.bool,
            )
            if not bool(active.any().detach().cpu().item()):
                return False
            loss_vals = self.loss_module(subdata)
            q_loss = _loss_matrix(loss_vals.get("loss"), active.to(dtype=torch.float32))
            loss = _masked_mean(q_loss, active)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.loss_module.parameters(),
                float(self.cfg.max_grad_norm),
            )
            self.optimizer.step()
            self.target_net_updater.step()
            target_update_steps += 1
            loss_values.append(float(loss.detach().cpu().item()))
            return True

        for epoch in range(combined_update_epochs):
            optimizer_steps_this_epoch = 0
            if fresh_coverage_mode and epoch < fresh_transition_epochs:
                fresh_permutation = torch.randperm(fresh_count).detach().cpu().numpy()
                for start in range(0, fresh_count, minibatch_size):
                    batch_indices = fresh_permutation[start : start + minibatch_size]
                    if optimize_batch(new_transitions, batch_indices):
                        fresh_sampled_transitions += int(len(batch_indices))
                        fresh_unique_indices.update(int(idx) for idx in batch_indices)
                        optimizer_steps_this_epoch += 1

            should_sample_replay = (
                (fresh_coverage_mode and ratio_budget and epoch < replay_update_epochs)
                or (not fresh_coverage_mode)
            )
            if should_sample_replay:
                if ratio_budget:
                    replay_buffer.reset_sampling_cycle()
                    sampler_resets += 1
                remaining_samples = (
                    int(replay_sample_budget) if ratio_budget else int(frames_per_batch)
                )
                scheduled_batches = 0
                epoch_replay_indices: set[int] = set()
                while True:
                    if ratio_budget:
                        if remaining_samples <= 0:
                            break
                        request_size = min(minibatch_size, remaining_samples)
                    else:
                        if scheduled_batches >= legacy_updates_per_epoch:
                            break
                        request_size = minibatch_size
                    batch_indices = replay_buffer.sample_indices(batch_size=request_size)
                    scheduled_batches += 1
                    if not batch_indices.size:
                        break
                    if ratio_budget:
                        remaining_samples -= int(len(batch_indices))
                    if optimize_batch(replay_transitions, batch_indices):
                        epoch_replay_indices.update(int(idx) for idx in batch_indices)
                        replay_sampled_transitions += int(len(batch_indices))
                        optimizer_steps_this_epoch += 1
                replay_unique_sampled_transitions += int(len(epoch_replay_indices))

            optimizer_steps_per_epoch.append(int(optimizer_steps_this_epoch))

        sampled_transitions = fresh_sampled_transitions + replay_sampled_transitions
        unique_sampled_transitions = (
            len(fresh_unique_indices) + replay_unique_sampled_transitions
        )

        active_rewards = np.asarray(
            [float(transition.reward) for transition in new_transitions],
            dtype=np.float32,
        )
        active_done = np.asarray([float(transition.done) for transition in new_transitions], dtype=np.float32)
        transition_steps = np.asarray(
            [float(transition.steps_to_next_obs) for transition in new_transitions],
            dtype=np.float32,
        )
        active_q = _chosen_q_for_samples(
            samples=new_samples,
            q_network=self.actor,
            chunk_size=minibatch_size,
            device=self.device,
        )

        def mean(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "loss": mean(loss_values),
            "q_loss": mean(loss_values),
            "n_samples": float(len(new_samples)),
            "new_samples": float(len(new_samples)),
            "new_transitions": float(fresh_count),
            "replay_size": float(len(replay_transitions)),
            "sampled_transitions": float(sampled_transitions),
            "unique_sampled_transitions": float(unique_sampled_transitions),
            "sampled_to_new_ratio": float(sampled_transitions / fresh_count) if fresh_count else 0.0,
            "fresh_sampled_transitions": float(fresh_sampled_transitions),
            "fresh_unique_transitions": float(len(fresh_unique_indices)),
            "fresh_coverage": (
                float(len(fresh_unique_indices) / fresh_count) if fresh_count else 0.0
            ),
            "replay_sampled_transitions": float(replay_sampled_transitions),
            "replay_sample_ratio": replay_sample_ratio,
            "fresh_transition_epochs": float(fresh_transition_epochs),
            "sample_budget_per_epoch": float(frames_per_batch),
            "fresh_sample_budget_per_epoch": float(fresh_sample_budget),
            "replay_sample_budget_per_epoch": float(replay_sample_budget),
            "target_sampled_transitions": float(target_sampled_transitions),
            "fresh_retained_transitions": float(fresh_retained),
            "fresh_evicted_before_update": float(fresh_evicted),
            "optimizer_steps": float(sum(optimizer_steps_per_epoch)),
            "target_update_steps": float(target_update_steps),
            "target_old_weight_after_update": float(
                float(self.cfg.target_update_eps) ** int(target_update_steps)
            ),
            "online_target_relative_l2": _online_target_relative_l2(self.loss_module),
            "censored_tail_dropped": float(max(len(new_samples) - fresh_count, 0)),
            "mean_steps_to_next_obs": (
                float(np.mean(transition_steps)) if transition_steps.size else 0.0
            ),
            "multi_step_transition_frac": (
                float(np.mean(transition_steps > 1.0)) if transition_steps.size else 0.0
            ),
            "sampler_resets": float(sampler_resets),
            "replay_episodes": float(len({transition.episode for transition in replay_transitions})),
            "pending_episodes": 0.0,
            "update_epochs": float(update_epochs),
            "replay_update_epochs": float(replay_update_epochs),
            "combined_update_epochs": float(combined_update_epochs),
            "updates_per_epoch": (
                mean(optimizer_steps_per_epoch)
                if fresh_coverage_mode or ratio_budget
                else float(legacy_updates_per_epoch)
            ),
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
                row.update(collector.episode_epsilon_scope_stats())
                _annotate_iql_samples_with_arrival_remaining(
                    samples,
                    dict(getattr(collector, "_last_ep_vehicle_tt", {}) or {}),
                )
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
                    replay_transitions = self._ensure_replay_buffer().snapshot()
                    update = self._pending_update_metrics(
                        n_samples=pending_samples,
                        pending_episodes=len(pending_batches),
                        epsilon=epsilon,
                        replay_size=len(replay_transitions),
                        replay_episodes=len({transition.episode for transition in replay_transitions}),
                        replay_sample_ratio=float(self.cfg.replay_sample_ratio),
                        fresh_transition_epochs=int(self.cfg.fresh_transition_epochs),
                    )
                row.update({f"update_{k}": v for k, v in update.items()})
                train_rows.append(row)
                update_rows.append({"episode": ep, **update})
                print(
                    f"[toy-iql-torchrl] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={len(samples)} reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"loss={update['loss']:.4f} q={update.get('chosen_q_mean', 0.0):.4f} "
                    f"eps={update.get('epsilon', epsilon):.3f} "
                    f"effective_eps={row.get('effective_epsilon_mean', 0.0):.4f} "
                    f"t_all={row.get('t_all', float('nan')):.3f} "
                    f"t_CAV={row.get('t_CAV', float('nan')):.3f} t_HDV={row.get('t_HDV', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"unfinished={int(row.get('unfinished_all', 0))} "
                    f"new={update.get('new_samples', 0.0):.0f} "
                    f"replay={update.get('replay_size', 0.0):.0f}/{int(self.cfg.memory_size)} "
                    f"sampled={update.get('sampled_transitions', 0.0):.0f} "
                    f"fresh={update.get('fresh_sampled_transitions', 0.0):.0f} "
                    f"fresh_coverage={update.get('fresh_coverage', 0.0):.3f} "
                    f"replay_sampled={update.get('replay_sampled_transitions', 0.0):.0f} "
                    f"pending={int(update.get('pending_batch', 0.0))} "
                    f"target={update.get('target_sampled_transitions', 0.0):.0f} "
                    f"sample_ratio={update.get('sampled_to_new_ratio', 0.0):.3f} "
                    f"evicted={update.get('fresh_evicted_before_update', 0.0):.0f} "
                    f"steps={update.get('optimizer_steps', 0.0):.0f} "
                    f"target_updates={update.get('target_update_steps', 0.0):.0f} "
                    f"target_gap={update.get('online_target_relative_l2', 0.0):.4f} "
                    f"duration={update.get('mean_steps_to_next_obs', 0.0):.3f} "
                    f"censored={update.get('censored_tail_dropped', 0.0):.0f}",
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
                "memory_size": int(self.cfg.memory_size),
                "replay_sample_ratio": float(self.cfg.replay_sample_ratio),
                "fresh_transition_epochs": int(self.cfg.fresh_transition_epochs),
                "update_budget_mode": (
                    "fresh_full_coverage_plus_replay"
                    if int(self.cfg.fresh_transition_epochs) > 0
                    and float(self.cfg.replay_sample_ratio) > 0.0
                    else "fresh_full_coverage"
                    if int(self.cfg.fresh_transition_epochs) > 0
                    else "fresh_transition_ratio"
                    if float(self.cfg.replay_sample_ratio) > 0.0
                    else "legacy_agent_frames"
                ),
                "fresh_transition_sampling": (
                    "random_without_replacement"
                    if int(self.cfg.fresh_transition_epochs) > 0
                    else "disabled"
                ),
                "target_update_cadence": (
                    "once_per_optimizer_step"
                ),
                "replay_sampler": "SamplerWithoutReplacement",
                "replay_sampler_reset_each_epoch": float(self.cfg.replay_sample_ratio) > 0.0,
                "gamma": float(self.cfg.gamma),
                "eps_init": float(self.cfg.eps_init),
                "eps_end": float(self.cfg.eps_end),
                "exploration_fraction": float(self.cfg.exploration_fraction),
                "epsilon_scope": str(self.cfg.epsilon_scope),
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
        self.cfg.disable_human_learning_after_mutation = True
        collector = self._build_collector()
        collector._av_execution_mode = "dynamic"
        rows: List[Dict[str, Any]] = []
        started = time.time()
        start_episode = int(getattr(self.cfg, "resume_start_episode", 0) or 0)
        if start_episode <= 0:
            start_episode = int(self.loaded_resume_episode)
        if str(self.cfg.real_policy_route_selection) == "sample":
            eval_epsilon = float(self.cfg.eps_end)
            self.actor.train()
        else:
            eval_epsilon = 0.0
            self.actor.eval()
        self.actor.set_epsilon(eval_epsilon)
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
            "eval_policy_route_selection": str(self.cfg.real_policy_route_selection),
            "eval_policy_executor": "valid_mask_epsilon_greedy",
            "eval_epsilon": float(eval_epsilon),
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
        disable_human_learning_after_mutation=bool(args.eval_only),
        real_policy_route_selection=(
            "sample"
            if bool(args.eval_only) and str(args.eval_policy_route_selection) == "training"
            else "argmax"
            if bool(args.eval_only)
            else "sample"
        ),
        sumo_time_to_teleport=str(args.sumo_time_to_teleport),
        sumo_extra_args=str(args.sumo_extra_args or ""),
        agent_frames_per_batch=int(args.agent_frames_per_batch),
        num_epochs=int(args.num_epochs),
        minibatch_size=int(args.minibatch_size),
        memory_size=int(args.memory_size),
        replay_sample_ratio=float(args.replay_sample_ratio),
        fresh_transition_epochs=int(args.fresh_transition_epochs),
        q_network_depth=int(args.q_network_depth),
        q_network_num_cells=int(args.q_network_num_cells),
        eps_init=float(args.eps_init),
        eps_end=float(args.eps_end),
        exploration_fraction=float(args.exploration_fraction),
        epsilon_scope=str(args.epsilon_scope),
        target_update_eps=float(args.target_update_eps),
        av_execution_mode=str(args.av_execution_mode),
    )
    if int(args.train_sim_buffer_after_last_depart) < 0:
        raise ValueError("train_sim_buffer_after_last_depart must be non-negative")
    if not bool(args.eval_only) and int(args.train_sim_buffer_after_last_depart) > 0:
        overrides["sim_buffer_after_last_depart"] = int(
            args.train_sim_buffer_after_last_depart
        )
    if bool(args.eval_only) and int(args.eval_sim_buffer_after_last_depart) > 0:
        overrides["sim_buffer_after_last_depart"] = int(args.eval_sim_buffer_after_last_depart)
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
    parser.add_argument(
        "--train-sim-buffer-after-last-depart",
        type=int,
        default=0,
        help=(
            "Training-only override applied after --settings is loaded. "
            "Use this when the settings drain is too short to observe actual arrivals."
        ),
    )
    parser.add_argument(
        "--eval-sim-buffer-after-last-depart",
        type=int,
        default=0,
        help="Evaluation-only override for the SUMO drain window after the last departure.",
    )
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
    parser.add_argument(
        "--memory-size",
        type=int,
        default=20_000,
        help="Transition replay capacity, matching the official URB IQL TensorDictReplayBuffer.",
    )
    parser.add_argument(
        "--replay-sample-ratio",
        type=float,
        default=0.0,
        help=(
            "Replay transitions sampled per fresh transition per epoch. "
            "Values >0 use the measured fresh-transition budget; 0 preserves "
            "the original n_agents*agent_frames_per_batch budget."
        ),
    )
    parser.add_argument(
        "--fresh-transition-epochs",
        type=int,
        default=0,
        help=(
            "Number of random-without-replacement passes over every transition "
            "collected for the current update. 0 preserves the legacy replay-only budget."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--q-network-depth", type=int, default=3)
    parser.add_argument("--q-network-num-cells", type=int, default=64)
    parser.add_argument("--eps-init", type=float, default=0.99)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.4)
    parser.add_argument(
        "--epsilon-scope",
        choices=["per_decision", "first_decision_per_vehicle"],
        default="per_decision",
        help=(
            "Apply epsilon at every dynamic decision, or at most once per vehicle "
            "and episode to preserve the original one-decision URB exploration scale."
        ),
    )
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
    parser.add_argument(
        "--eval-policy-route-selection",
        choices=["argmax", "training"],
        default="argmax",
        help=(
            "Evaluation executor: argmax preserves the legacy benchmark; training uses "
            "the final-episode valid-mask epsilon-greedy policy."
        ),
    )
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
            "sim_buffer_after_last_depart": cfg.sim_buffer_after_last_depart,
            "av_execution_mode": str(cfg.av_execution_mode),
            "av_candidate_source": str(cfg.av_candidate_source),
            "route_generator": str(cfg.route_generator),
            "path_gen_beta": float(cfg.path_gen_beta),
            "path_num_samples": int(cfg.path_num_samples),
            "post_mutation_state_in": cfg.post_mutation_state_in,
            "hdv_state_in": cfg.hdv_state_in,
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
            "memory_size": int(cfg.memory_size),
            "replay_sample_ratio": float(cfg.replay_sample_ratio),
            "fresh_transition_epochs": int(cfg.fresh_transition_epochs),
            "update_budget_mode": (
                "fresh_full_coverage_plus_replay"
                if int(cfg.fresh_transition_epochs) > 0
                and float(cfg.replay_sample_ratio) > 0.0
                else "fresh_full_coverage"
                if int(cfg.fresh_transition_epochs) > 0
                else "fresh_transition_ratio"
                if float(cfg.replay_sample_ratio) > 0.0
                else "legacy_agent_frames"
            ),
            "fresh_transition_sampling": (
                "random_without_replacement"
                if int(cfg.fresh_transition_epochs) > 0
                else "disabled"
            ),
            "target_update_cadence": "once_per_optimizer_step",
            "replay_sampler": "SamplerWithoutReplacement",
            "replay_sampler_reset_each_epoch": float(cfg.replay_sample_ratio) > 0.0,
            "lr": float(cfg.lr),
            "max_grad_norm": float(cfg.max_grad_norm),
            "gamma": float(cfg.gamma),
            "q_network_depth": int(cfg.q_network_depth),
            "q_network_num_cells": int(cfg.q_network_num_cells),
            "eps_init": float(cfg.eps_init),
            "eps_end": float(cfg.eps_end),
            "exploration_fraction": float(cfg.exploration_fraction),
            "epsilon_scope": str(cfg.epsilon_scope),
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
