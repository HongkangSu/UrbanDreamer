#!/usr/bin/env python3
"""Run a TorchRL-style MAPPO baseline on the toy cross-bottleneck network."""

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
from torch.distributions import Categorical
from torchrl.data import Categorical as CategoricalSpec
from torchrl.data import Composite
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss


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
class ToyMappoTorchRLConfig(OnlineNoCfmConfig):
    """Toy fixed-role MAPPO config aligned with ``../URB/scripts/mappo_torchrl.py``."""

    agent_frames_per_batch: int = 10
    n_iters: int = 0
    checkpoint_interval: int = 100
    num_epochs: int = 1
    minibatch_size: int = 32
    clip_epsilon: float = 0.2
    lmbda: float = 0.9
    entropy_eps: float = 1e-4
    normalize_advantage: bool = False
    policy_network_depth: int = 3
    policy_network_num_cells: int = 64
    critic_network_depth: int = 3
    critic_network_num_cells: int = 64
    av_execution_mode: str = "dynamic"

    def resolve(self) -> "ToyMappoTorchRLConfig":
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
        return ToyMappoTorchRLConfig(**values)

    @classmethod
    def from_settings(cls, settings_path: str, **overrides: Any) -> "ToyMappoTorchRLConfig":
        base_fields = set(OnlineNoCfmConfig.__dataclass_fields__.keys())
        base_overrides = {k: v for k, v in overrides.items() if k in base_fields}
        own_overrides = {k: v for k, v in overrides.items() if k not in base_fields}
        base = OnlineNoCfmConfig.from_settings(settings_path, **base_overrides).resolve()
        values = {**asdict(cls()), **asdict(base), **own_overrides}
        return cls(**values).resolve()


def _jsonable_config(cfg: ToyMappoTorchRLConfig) -> Dict[str, Any]:
    values = asdict(cfg)
    values["checkpoint_episodes"] = list(cfg.checkpoint_episodes)
    values["extra_eval_args"] = list(cfg.extra_eval_args)
    return values


def _resolve_fixed_roles_for_config(
    cfg: ToyMappoTorchRLConfig,
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


class TorchRLMappoActorCritic(torch.nn.Module):
    """TorchRL MAPPO actor/critic with independent policies and centralized critic."""

    def __init__(
        self,
        *,
        obs_dim: int,
        n_routes: int,
        machine_ids: Sequence[str],
        cfg: ToyMappoTorchRLConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_routes = int(n_routes)
        self.machine_ids = [str(x) for x in machine_ids]
        self.n_agents = int(max(len(self.machine_ids), 1))
        self.slot_by_vehicle = {veh_id: idx for idx, veh_id in enumerate(self.machine_ids)}
        self.device_ref = device

        self.policy_net = torch.nn.Sequential(
            MultiAgentMLP(
                n_agent_inputs=int(obs_dim),
                n_agent_outputs=int(n_routes),
                n_agents=int(self.n_agents),
                centralized=False,
                share_params=False,
                device=device,
                depth=int(cfg.policy_network_depth),
                num_cells=int(cfg.policy_network_num_cells),
                activation_class=torch.nn.Tanh,
            )
        )
        self.policy_module = TensorDictModule(
            self.policy_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "logits")],
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
        self.policy = ProbabilisticActor(
            module=self.policy_module,
            spec=action_spec,
            in_keys=[("agents", "logits")],
            out_keys=[("agents", "action")],
            distribution_class=Categorical,
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )

        self.critic_net = MultiAgentMLP(
            n_agent_inputs=int(obs_dim),
            n_agent_outputs=1,
            n_agents=int(self.n_agents),
            centralized=True,
            share_params=True,
            device=device,
            depth=int(cfg.critic_network_depth),
            num_cells=int(cfg.critic_network_num_cells),
            activation_class=torch.nn.ReLU,
        )
        self.critic = TensorDictModule(
            self.critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
        self.to(device)

    def slot_for_vehicle(self, veh_id: str) -> int:
        key = str(veh_id)
        if key not in self.slot_by_vehicle:
            raise KeyError(f"vehicle {key!r} is not in fixed MAPPO machine roles")
        return int(self.slot_by_vehicle[key])

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

    def forward_logits_for_active_ids(
        self,
        obs: torch.Tensor,
        active_ids: Sequence[str],
    ) -> torch.Tensor:
        td, slots = self._group_observation_td(obs, active_ids)
        out = self.policy_module(td)
        logits = out.get(("agents", "logits"))
        rows = torch.arange(int(logits.shape[0]), device=logits.device)
        return logits[rows, slots]


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
        raise ValueError(f"sample obs_dim={obs.shape[0]} does not match actor obs_dim={obs_dim}")
    return obs


def _dense_observation_and_slots(
    *,
    samples: Sequence[OnlineDecisionSample],
    indices: Sequence[int],
    actor_critic: TorchRLMappoActorCritic,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    n = int(len(indices))
    obs = torch.zeros(
        (n, int(actor_critic.n_agents), int(actor_critic.obs_dim)),
        device=device,
        dtype=torch.float32,
    )
    slots: List[int] = []
    for row, raw_idx in enumerate(indices):
        sample = samples[int(raw_idx)]
        slot = actor_critic.slot_for_vehicle(str(sample.veh_id))
        obs[row, slot] = torch.as_tensor(
            _sample_observation(sample, obs_dim=int(actor_critic.obs_dim)),
            device=device,
            dtype=torch.float32,
        )
        slots.append(int(slot))
    return obs, slots


def _build_mappo_minibatch_td(
    *,
    samples: Sequence[OnlineDecisionSample],
    indices: Sequence[int],
    actor_critic: TorchRLMappoActorCritic,
    advantages: np.ndarray,
    value_targets: np.ndarray,
    device: torch.device,
) -> TensorDict:
    n = int(len(indices))
    n_agents = int(actor_critic.n_agents)
    n_routes = int(actor_critic.n_routes)
    obs, slots = _dense_observation_and_slots(
        samples=samples,
        indices=indices,
        actor_critic=actor_critic,
        device=device,
    )
    action = torch.zeros((n, n_agents), device=device, dtype=torch.long)
    old_log_prob = torch.zeros((n, n_agents), device=device, dtype=torch.float32)
    action_mask = torch.zeros((n, n_agents, n_routes), device=device, dtype=torch.bool)
    active_mask = torch.zeros((n, n_agents), device=device, dtype=torch.bool)
    rewards = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    advantage = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    value_target = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)

    for row, (raw_idx, slot) in enumerate(zip(indices, slots)):
        sample_idx = int(raw_idx)
        sample = samples[sample_idx]
        action[row, slot] = int(np.clip(int(sample.action), 0, n_routes - 1))
        old_lp = float(sample.old_log_prob)
        old_log_prob[row, slot] = old_lp if np.isfinite(old_lp) else 0.0
        action_mask[row, slot] = _sample_action_mask(sample, n_routes=n_routes, device=device)
        active_mask[row, slot] = True
        rewards[row, slot, 0] = float(sample.reward)
        advantage[row, slot, 0] = float(advantages[sample_idx])
        value_target[row, slot, 0] = float(value_targets[sample_idx])

    return TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): action,
            ("agents", "sample_log_prob"): old_log_prob.detach(),
            ("agents", "action_mask"): action_mask,
            ("agents", "active_mask"): active_mask,
            ("agents", "advantage"): advantage.detach(),
            ("agents", "value_target"): value_target.detach(),
            ("agents", "reward"): rewards.detach(),
        },
        batch_size=[n],
        device=device,
    )


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


def _mappo_effective_minibatch_size(
    *,
    configured_minibatch_size: int,
    actor_critic: TorchRLMappoActorCritic,
) -> int:
    # TorchRL's centralized MultiAgentMLP materializes features that scale with
    # batch * n_agents^2 * obs_dim. Keep the forward batch bounded for real nets.
    target_centralized_elements = 16_000_000
    per_sample_elements = max(
        int(actor_critic.n_agents) * int(actor_critic.n_agents) * int(actor_critic.obs_dim),
        1,
    )
    centralized_cap = max(int(target_centralized_elements // per_sample_elements), 1)
    return max(1, min(int(configured_minibatch_size), int(centralized_cap)))


def _compute_mappo_advantages(
    *,
    samples: Sequence[OnlineDecisionSample],
    actor_critic: TorchRLMappoActorCritic,
    gamma: float,
    lmbda: float,
    normalize_advantage: bool,
    value_chunk_size: int,
    device: torch.device,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(samples))
    slots: List[int] = []
    rewards = np.zeros((n,), dtype=np.float32)
    values = np.zeros((n,), dtype=np.float32)
    by_vehicle: Dict[Tuple[int, str], List[int]] = {}
    for row, sample in enumerate(samples):
        slots.append(actor_critic.slot_for_vehicle(str(sample.veh_id)))
        rewards[row] = float(sample.reward)
        by_vehicle.setdefault((int(sample.episode), str(sample.veh_id)), []).append(row)

    chunk_size = max(1, min(int(value_chunk_size), n))
    actor_critic.eval()
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            indices = list(range(start, min(start + chunk_size, n)))
            obs, chunk_slots = _dense_observation_and_slots(
                samples=samples,
                indices=indices,
                actor_critic=actor_critic,
                device=device,
            )
            value_td = actor_critic.critic(
                TensorDict({("agents", "observation"): obs}, batch_size=[len(indices)], device=device)
            )
            value_tensor = value_td.get(("agents", "state_value")).detach()
            rows = torch.arange(len(indices), device=device)
            slot_tensor = torch.as_tensor(chunk_slots, device=device, dtype=torch.long)
            values[start : start + len(indices)] = (
                value_tensor[rows, slot_tensor, 0].detach().cpu().numpy().astype(np.float32)
            )

    advantages = np.zeros((n,), dtype=np.float32)
    value_targets = np.zeros((n,), dtype=np.float32)
    gamma_f = float(gamma)
    lmbda_f = float(lmbda)
    for indices in by_vehicle.values():
        ordered = sorted(indices, key=lambda i: (int(samples[i].decision_idx), int(samples[i].sim_step)))
        next_adv = 0.0
        next_value = 0.0
        for row in reversed(ordered):
            delta = float(rewards[row]) + gamma_f * next_value - float(values[row])
            adv = delta + gamma_f * lmbda_f * next_adv
            advantages[row] = float(adv)
            value_targets[row] = float(adv + float(values[row]))
            next_adv = float(adv)
            next_value = float(values[row])

    if bool(normalize_advantage) and n > 1:
        mean = float(np.mean(advantages))
        std = float(np.std(advantages))
        if std > 1e-6:
            advantages = ((advantages - mean) / std).astype(np.float32)

    return slots, rewards, advantages, value_targets


def build_torchrl_mappo_batch(
    *,
    samples: Sequence[OnlineDecisionSample],
    actor_critic: TorchRLMappoActorCritic,
    gamma: float,
    lmbda: float,
    normalize_advantage: bool,
    device: torch.device,
) -> TensorDict:
    n = int(len(samples))
    if n <= 0:
        raise ValueError("cannot build a MAPPO TorchRL batch with no samples")

    n_agents = int(actor_critic.n_agents)
    obs_dim = int(actor_critic.obs_dim)
    n_routes = int(actor_critic.n_routes)
    obs = torch.zeros((n, n_agents, obs_dim), device=device, dtype=torch.float32)
    action = torch.zeros((n, n_agents), device=device, dtype=torch.long)
    old_log_prob = torch.zeros((n, n_agents), device=device, dtype=torch.float32)
    action_mask = torch.zeros((n, n_agents, n_routes), device=device, dtype=torch.bool)
    active_mask = torch.zeros((n, n_agents), device=device, dtype=torch.bool)
    rewards = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    slot_rows: List[int] = []
    by_vehicle: Dict[Tuple[int, str], List[int]] = {}

    for row, sample in enumerate(samples):
        slot = actor_critic.slot_for_vehicle(str(sample.veh_id))
        slot_rows.append(int(slot))
        sample_obs = np.asarray(sample.obs, dtype=np.float32).reshape(-1)
        if int(sample_obs.shape[0]) != obs_dim:
            raise ValueError(
                f"sample obs_dim={sample_obs.shape[0]} does not match actor obs_dim={obs_dim}"
            )
        obs[row, slot] = torch.as_tensor(sample_obs, device=device, dtype=torch.float32)
        action[row, slot] = int(np.clip(int(sample.action), 0, n_routes - 1))
        old_lp = float(sample.old_log_prob)
        old_log_prob[row, slot] = old_lp if np.isfinite(old_lp) else 0.0
        mask_np = np.asarray(sample.valid_mask, dtype=bool).reshape(-1)
        if int(mask_np.shape[0]) >= n_routes:
            action_mask[row, slot] = torch.as_tensor(mask_np[:n_routes], device=device, dtype=torch.bool)
        else:
            action_mask[row, slot, : int(mask_np.shape[0])] = torch.as_tensor(mask_np, device=device, dtype=torch.bool)
        if not bool(action_mask[row, slot].any().detach().cpu().item()):
            action_mask[row, slot] = True
        active_mask[row, slot] = True
        rewards[row, slot, 0] = float(sample.reward)
        by_vehicle.setdefault((int(sample.episode), str(sample.veh_id)), []).append(row)

    td = TensorDict(
        {
            ("agents", "observation"): obs,
            ("agents", "action"): action,
            ("agents", "sample_log_prob"): old_log_prob.detach(),
            ("agents", "action_mask"): action_mask,
            ("agents", "active_mask"): active_mask,
        },
        batch_size=[n],
        device=device,
    )
    with torch.no_grad():
        value_td = actor_critic.critic(TensorDict({("agents", "observation"): obs}, batch_size=[n], device=device))
        values = value_td.get(("agents", "state_value")).detach()

    advantage = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    value_target = torch.zeros((n, n_agents, 1), device=device, dtype=torch.float32)
    gamma_f = float(gamma)
    lmbda_f = float(lmbda)
    for indices in by_vehicle.values():
        ordered = sorted(indices, key=lambda i: (int(samples[i].decision_idx), int(samples[i].sim_step)))
        next_adv = torch.zeros((), device=device, dtype=torch.float32)
        next_value = torch.zeros((), device=device, dtype=torch.float32)
        for row in reversed(ordered):
            slot = int(slot_rows[row])
            value = values[row, slot, 0]
            reward = rewards[row, slot, 0]
            delta = reward + gamma_f * next_value - value
            adv = delta + gamma_f * lmbda_f * next_adv
            advantage[row, slot, 0] = adv
            value_target[row, slot, 0] = adv + value
            next_adv = adv
            next_value = value

    if bool(normalize_advantage):
        active_adv = advantage[active_mask]
        if int(active_adv.numel()) > 1:
            mean = active_adv.mean()
            std = active_adv.std(unbiased=False).clamp_min(1e-6)
            advantage = torch.where(active_mask.unsqueeze(-1), (advantage - mean) / std, advantage)

    td.set(("agents", "advantage"), advantage.detach())
    td.set(("agents", "value_target"), value_target.detach())
    td.set(("agents", "reward"), rewards.detach())
    return td


def build_torchrl_mappo_loss(
    *,
    actor_critic: TorchRLMappoActorCritic,
    cfg: ToyMappoTorchRLConfig,
) -> ClipPPOLoss:
    loss_module = ClipPPOLoss(
        actor_network=actor_critic.policy,
        critic_network=actor_critic.critic,
        clip_epsilon=float(cfg.clip_epsilon),
        entropy_coef=float(cfg.entropy_eps),
        normalize_advantage=False,
        reduction="none",
    )
    loss_module.set_keys(
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),
    )
    return loss_module


class ToyMappoTorchRLTrainer(ToyOnlineAcTrainer):
    """Fixed-role toy trainer using TorchRL's MAPPO actor/critic/loss stack."""

    def __init__(self, cfg: ToyMappoTorchRLConfig):
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
            raise ValueError("MAPPO fixed-role baseline requires at least one machine agent")
        self.actor = TorchRLMappoActorCritic(
            obs_dim=int(self.obs_dim),
            n_routes=int(self.cfg.n_routes),
            machine_ids=self.machine_ids,
            cfg=self.cfg,
            device=self.device,
        )
        self.loss_module = build_torchrl_mappo_loss(actor_critic=self.actor, cfg=self.cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=float(self.cfg.lr))
        self.collector = None
        self.loaded_resume_episode = self._load_resume_checkpoint()

    @staticmethod
    def _pending_update_metrics(
        *,
        n_samples: int,
        pending_episodes: int,
    ) -> Dict[str, float]:
        return {
            "loss": 0.0,
            "loss_objective": 0.0,
            "loss_entropy": 0.0,
            "loss_critic": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0,
            "clip_frac": 0.0,
            "n_samples": float(n_samples),
            "replay_episodes": float(pending_episodes),
            "update_epochs": 0.0,
            "updates_per_epoch": 0.0,
            "frames_per_batch": 0.0,
            "return_mean": 0.0,
            "return_std": 0.0,
            "adv_mean": 0.0,
            "adv_std": 0.0,
            "skipped": 1.0,
            "pending_batch": 1.0,
        }

    def _load_resume_checkpoint(self) -> int:
        ckpt_path = str(getattr(self.cfg, "resume_checkpoint", "") or "").strip()
        if not ckpt_path:
            return 0
        path = _resolve_path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"resume_checkpoint does not exist: {path}")
        state = torch.load(path, map_location=self.device)
        actor_state = state.get("actor_critic") or state.get("actor") or state.get("policy")
        if actor_state is None:
            raise KeyError(f"resume checkpoint has no actor_critic/actor/policy state: {path}")
        self.actor.load_state_dict(actor_state)
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        episode = int(state.get("episode", 0) or 0)
        print(f"[toy-mappo-torchrl] resumed actor/critic from {path} at episode {episode}", flush=True)
        return episode

    def update_from_replay(self, episode_batches: Sequence[Sequence[OnlineDecisionSample]]) -> Dict[str, float]:
        samples = [sample for batch in episode_batches for sample in batch]
        if len(samples) < int(self.cfg.min_samples_per_update):
            return {
                "loss": 0.0,
                "loss_objective": 0.0,
                "loss_entropy": 0.0,
                "loss_critic": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "n_samples": float(len(samples)),
                "replay_episodes": float(len(episode_batches)),
                "update_epochs": 0.0,
                "skipped": 1.0,
            }

        frames = int(len(samples))
        configured_minibatch_size = min(max(int(self.cfg.minibatch_size), 1), max(frames, 1))
        effective_minibatch_size = _mappo_effective_minibatch_size(
            configured_minibatch_size=configured_minibatch_size,
            actor_critic=self.actor,
        )
        _slots, _rewards, advantages, value_targets = _compute_mappo_advantages(
            samples=samples,
            actor_critic=self.actor,
            gamma=float(self.cfg.gamma),
            lmbda=float(self.cfg.lmbda),
            normalize_advantage=bool(self.cfg.normalize_advantage),
            value_chunk_size=int(effective_minibatch_size),
            device=self.device,
        )
        active_total = int(len(samples))
        if active_total <= 0:
            return {
                "loss": 0.0,
                "loss_objective": 0.0,
                "loss_entropy": 0.0,
                "loss_critic": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "n_samples": 0.0,
                "replay_episodes": float(len(episode_batches)),
                "update_epochs": 0.0,
                "skipped": 1.0,
            }

        frames_per_batch = max(int(self.actor.n_agents) * max(int(self.cfg.agent_frames_per_batch), 1), 1)
        updates_per_epoch = max(frames_per_batch // configured_minibatch_size, 1)
        update_epochs = max(int(self.cfg.num_epochs), 1)
        self.actor.train()
        self.loss_module.train()

        loss_values: List[float] = []
        loss_objective_values: List[float] = []
        loss_entropy_values: List[float] = []
        loss_critic_values: List[float] = []
        entropy_values: List[float] = []
        clip_values: List[float] = []
        for _epoch in range(update_epochs):
            for batch_indices in _random_minibatch_indices(
                n_samples=frames,
                minibatch_size=effective_minibatch_size,
                num_batches=updates_per_epoch,
            ):
                subdata = _build_mappo_minibatch_td(
                    samples=samples,
                    indices=batch_indices,
                    actor_critic=self.actor,
                    advantages=advantages,
                    value_targets=value_targets,
                    device=self.device,
                )
                active = subdata.get(("agents", "active_mask")).to(device=self.device, dtype=torch.bool)
                if not bool(active.any().detach().cpu().item()):
                    continue
                loss_vals = self.loss_module(subdata)
                objective = _loss_matrix(loss_vals.get("loss_objective"), active.to(dtype=torch.float32))
                entropy_loss = _loss_matrix(loss_vals.get("loss_entropy"), active.to(dtype=torch.float32))
                critic_loss = _loss_matrix(loss_vals.get("loss_critic"), active.to(dtype=torch.float32))
                total_matrix = objective + entropy_loss + critic_loss
                loss = _masked_mean(total_matrix, active)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), float(self.cfg.max_grad_norm))
                self.optimizer.step()

                loss_values.append(float(loss.detach().cpu().item()))
                loss_objective_values.append(float(_masked_mean(objective.detach(), active).cpu().item()))
                loss_entropy_values.append(float(_masked_mean(entropy_loss.detach(), active).cpu().item()))
                loss_critic_values.append(float(_masked_mean(critic_loss.detach(), active).cpu().item()))
                entropy_tensor = loss_vals.get("entropy")
                if torch.is_tensor(entropy_tensor):
                    entropy_values.append(float(entropy_tensor.detach().mean().cpu().item()))
                clip_tensor = loss_vals.get("clip_fraction")
                if torch.is_tensor(clip_tensor):
                    clip_values.append(float(clip_tensor.detach().mean().cpu().item()))

        def mean(values: Sequence[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        return {
            "loss": mean(loss_values),
            "loss_objective": mean(loss_objective_values),
            "loss_entropy": mean(loss_entropy_values),
            "loss_critic": mean(loss_critic_values),
            "policy_loss": mean(loss_objective_values),
            "value_loss": mean(loss_critic_values),
            "entropy": mean(entropy_values),
            "clip_fraction": mean(clip_values),
            "clip_frac": mean(clip_values),
            "n_samples": float(active_total),
            "replay_episodes": float(len(episode_batches)),
            "update_epochs": float(update_epochs),
            "updates_per_epoch": float(updates_per_epoch),
            "frames_per_batch": float(frames_per_batch),
            "configured_minibatch_size": float(configured_minibatch_size),
            "effective_minibatch_size": float(effective_minibatch_size),
            "return_mean": float(np.mean(value_targets)) if value_targets.size else 0.0,
            "return_std": float(np.std(value_targets)) if value_targets.size > 1 else 0.0,
            "adv_mean": float(np.mean(advantages)) if advantages.size else 0.0,
            "adv_std": float(np.std(advantages)) if advantages.size > 1 else 0.0,
            "skipped": 0.0 if loss_values else 1.0,
        }

    def save_checkpoint(self, episode: int, metrics: Optional[Dict[str, Any]] = None) -> Path:
        path = self.output_dir / "checkpoints" / f"checkpoint_ep{int(episode):05d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_family": "mappo_torchrl",
            "episode": int(episode),
            "actor_critic": self.actor.state_dict(),
            "actor": self.actor.state_dict(),
            "policy": self.actor.policy.state_dict(),
            "critic": self.actor.critic.state_dict(),
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
                    )
                row.update({f"update_{k}": v for k, v in update.items()})
                train_rows.append(row)
                update_rows.append({"episode": ep, **update})
                print(
                    f"[toy-mappo-torchrl] ep={ep:04d}/{int(self.cfg.train_episodes):04d} "
                    f"samples={len(samples)} reward_mean={row.get('reward_mean', float('nan')):.4f} "
                    f"loss={update['loss']:.4f} objective={update.get('loss_objective', 0.0):.4f} "
                    f"critic={update.get('loss_critic', 0.0):.4f} entropy={update.get('entropy', 0.0):.4f} "
                    f"t_all={row.get('t_all', float('nan')):.3f} "
                    f"t_CAV={row.get('t_CAV', float('nan')):.3f} t_HDV={row.get('t_HDV', float('nan')):.3f} "
                    f"completion={row.get('completion_rate', float('nan')):.6f} "
                    f"unfinished={int(row.get('unfinished_all', 0))} "
                    f"batch={update.get('replay_episodes', 0.0):.0f}/{batch_episodes} "
                    f"pending={int(update.get('pending_batch', 0.0))} "
                    f"clip={update.get('clip_frac', 0.0):.3f}",
                    flush=True,
                )
                if ep in ckpt_eps:
                    ckpt = self.save_checkpoint(ep, metrics=row)
                    checkpoint_paths[int(ep)] = str(ckpt)
                    print(f"[toy-mappo-torchrl] saved checkpoint: {ckpt}", flush=True)
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
            "baseline": "toy_cross_bottleneck_mappo_torchrl",
            "algorithm": "mappo_torchrl",
            "config": _jsonable_config(self.cfg),
            "torchrl_mapping": {
                "share_parameters_policy": False,
                "share_parameters_critic": True,
                "centralized_critic": True,
                "value_estimator": "gae",
                "agent_frames_per_batch": int(self.cfg.agent_frames_per_batch),
                "clip_epsilon": float(self.cfg.clip_epsilon),
                "gamma": float(self.cfg.gamma),
                "lmbda": float(self.cfg.lmbda),
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
                    f"[toy-mappo-torchrl-eval] ep={idx + 1:03d}/{int(eval_episodes):03d} "
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


def build_config(args: argparse.Namespace) -> ToyMappoTorchRLConfig:
    network_dir = Path(args.network_dir).resolve()
    has_settings = bool(str(args.settings or "").strip())
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT
        / "results"
        / ("from_settings" if has_settings else TOY_NETWORK_NAME)
        / f"mappo_torchrl_fixed_role_{time.strftime('%Y%m%d_%H%M%S')}"
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
        entropy_coef=float(args.entropy_eps),
        ac_ppo_clip_ratio=float(args.clip_epsilon),
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
        clip_epsilon=float(args.clip_epsilon),
        lmbda=float(args.lmbda),
        entropy_eps=float(args.entropy_eps),
        normalize_advantage=bool(args.normalize_advantage),
        policy_network_depth=int(args.policy_network_depth),
        policy_network_num_cells=int(args.policy_network_num_cells),
        critic_network_depth=int(args.critic_network_depth),
        critic_network_num_cells=int(args.critic_network_num_cells),
        av_execution_mode=str(args.av_execution_mode),
    )
    if has_settings:
        cfg = ToyMappoTorchRLConfig.from_settings(str(args.settings), **overrides)
    else:
        cfg = ToyMappoTorchRLConfig(settings="", **overrides)
    return cfg.resolve()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default="",
        help="Optional URB settings env file used to align toy MAPPO with WM/no-CFM runs.",
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
    parser.add_argument("--agent-frames-per-batch", type=int, default=10)
    parser.add_argument(
        "--n-iters",
        type=int,
        default=0,
        help="URB mappo_torchrl iteration alias. When >0, overrides --train-episodes.",
    )
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.9)
    parser.add_argument("--entropy-eps", type=float, default=1e-4)
    parser.add_argument("--normalize-advantage", action="store_true")
    parser.add_argument("--policy-network-depth", type=int, default=3)
    parser.add_argument("--policy-network-num-cells", type=int, default=64)
    parser.add_argument("--critic-network-depth", type=int, default=3)
    parser.add_argument("--critic-network-num-cells", type=int, default=64)
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


def build_plan(cfg: ToyMappoTorchRLConfig) -> Dict[str, Any]:
    return {
        "baseline": "toy_cross_bottleneck_mappo_torchrl",
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
        "mappo_torchrl": {
            "share_parameters_policy": False,
            "share_parameters_critic": True,
            "centralized_critic": True,
            "agent_frames_per_batch": int(cfg.agent_frames_per_batch),
            "n_iters": int(cfg.n_iters or cfg.train_episodes),
            "checkpoint_interval": int(cfg.checkpoint_interval),
            "checkpoint_episodes": list(cfg.checkpoint_episodes),
            "num_epochs": int(cfg.num_epochs),
            "minibatch_size": int(cfg.minibatch_size),
            "lr": float(cfg.lr),
            "max_grad_norm": float(cfg.max_grad_norm),
            "clip_epsilon": float(cfg.clip_epsilon),
            "gamma": float(cfg.gamma),
            "lmbda": float(cfg.lmbda),
            "entropy_eps": float(cfg.entropy_eps),
            "normalize_advantage": bool(cfg.normalize_advantage),
            "policy_network_depth": int(cfg.policy_network_depth),
            "policy_network_num_cells": int(cfg.policy_network_num_cells),
            "critic_network_depth": int(cfg.critic_network_depth),
            "critic_network_num_cells": int(cfg.critic_network_num_cells),
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
    with (output_dir / "toy_mappo_torchrl_config.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable_config(cfg), f, indent=2)
    with (output_dir / "toy_mappo_torchrl_plan.json").open("w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    trainer = ToyMappoTorchRLTrainer(cfg)
    if args.eval_only:
        summary = trainer.evaluate(eval_episodes=int(args.eval_episodes))
        final_name = "toy_mappo_torchrl_eval_summary.json"
    else:
        summary = trainer.train()
        final_name = "toy_mappo_torchrl_final_summary.json"
    with (output_dir / final_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
