from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


ROUTE_COUNT = 4
VEHICLE_OBS_DIM = 33
LATENT_DIM = 64
CONTEXT_DIM = 3
ROUTE_FEATURE_DIM = 2
ROUTE_COST_FEATURES_PER_ROUTE = 3
FLEET_PRESSURE_FEATURES_PER_ROUTE = 2
HIDDEN_DIM = 256
HIDDEN_LAYERS = 2
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 0.95
ROUTE_COST_CAP_SECONDS = 30.0
REWARD_TEMPERATURE_SECONDS = 30.0
ROUTE_FEATURE_BLOCK = ROUTE_COUNT * ROUTE_FEATURE_DIM
GLOBAL_FEATURE_START = ROUTE_FEATURE_BLOCK
GLOBAL_FEATURE_END = GLOBAL_FEATURE_START + ROUTE_FEATURE_DIM
ROUTE_COST_START = GLOBAL_FEATURE_END
VALID_ROUTE_START = ROUTE_COST_START + 2 * ROUTE_COUNT
VALID_ROUTE_END = VALID_ROUTE_START + ROUTE_COUNT
FLEET_PRESSURE_START = ROUTE_COST_START + ROUTE_COST_FEATURES_PER_ROUTE * ROUTE_COUNT


@dataclass(frozen=True)
class ActionSelection:
    route_index: torch.Tensor
    route_one_hot: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor
    valid_route_mask: torch.Tensor


def _make_mlp(input_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width_in = int(input_dim)
    for _ in range(HIDDEN_LAYERS):
        layers.append(nn.Linear(width_in, HIDDEN_DIM))
        layers.append(nn.Tanh())
        width_in = HIDDEN_DIM
    return nn.Sequential(*layers)


class VehicleRouteActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        per_route_input_dim = (
            2 * ROUTE_FEATURE_DIM
            + ROUTE_COST_FEATURES_PER_ROUTE
            + FLEET_PRESSURE_FEATURES_PER_ROUTE
            + CONTEXT_DIM
        )
        self.route_scorer = _make_mlp(per_route_input_dim)
        self.policy_head = nn.Linear(HIDDEN_DIM, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def valid_mask_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] < VALID_ROUTE_END:
            raise ValueError(
                "obs last dim "
                f"{int(obs.shape[-1])} is too small for valid-mask slice "
                f"[{VALID_ROUTE_START}:{VALID_ROUTE_END}]"
            )
        return obs[..., VALID_ROUTE_START:VALID_ROUTE_END] > 0.0

    def _route_shared_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or int(obs.shape[-1]) != VEHICLE_OBS_DIM:
            raise ValueError(f"expected obs shape (B, {VEHICLE_OBS_DIM}), got {tuple(obs.shape)}")
        batch = int(obs.shape[0])

        route_feats = obs[:, :ROUTE_FEATURE_BLOCK].reshape(
            batch,
            ROUTE_COUNT,
            ROUTE_FEATURE_DIM,
        )
        global_feat = obs[:, GLOBAL_FEATURE_START:GLOBAL_FEATURE_END].unsqueeze(1).expand(
            batch,
            ROUTE_COUNT,
            ROUTE_FEATURE_DIM,
        )

        cost = obs[
            :,
            ROUTE_COST_START : ROUTE_COST_START
            + ROUTE_COST_FEATURES_PER_ROUTE * ROUTE_COUNT,
        ]
        gap = cost[:, :ROUTE_COUNT].unsqueeze(-1)
        log_cost = cost[:, ROUTE_COUNT : 2 * ROUTE_COUNT].unsqueeze(-1)
        valid = cost[:, 2 * ROUTE_COUNT : 3 * ROUTE_COUNT].unsqueeze(-1)

        fleet = obs[
            :,
            FLEET_PRESSURE_START : FLEET_PRESSURE_START
            + FLEET_PRESSURE_FEATURES_PER_ROUTE * ROUTE_COUNT,
        ]
        route_share = fleet[:, :ROUTE_COUNT].unsqueeze(-1)
        pressure = fleet[:, ROUTE_COUNT : 2 * ROUTE_COUNT].unsqueeze(-1)

        context = obs[:, -CONTEXT_DIM:].unsqueeze(1).expand(batch, ROUTE_COUNT, CONTEXT_DIM)
        return torch.cat(
            [route_feats, global_feat, gap, log_cost, valid, route_share, pressure, context],
            dim=-1,
        )

    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        per_route = self._route_shared_inputs(obs)
        hidden = self.route_scorer(per_route.reshape(-1, per_route.shape[-1]))
        logits = self.policy_head(hidden).reshape(obs.shape[0], ROUTE_COUNT)
        valid_mask = self.valid_mask_from_obs(obs)
        logits = logits.masked_fill(~valid_mask, -20.0)
        all_invalid = ~valid_mask.any(dim=-1, keepdim=True)
        logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
        return torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)

    def sample_action(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> ActionSelection:
        logits = self.forward_logits(obs)
        valid_mask = self.valid_mask_from_obs(obs)
        safe_valid = valid_mask.clone()
        no_valid = ~safe_valid.any(dim=-1)
        if bool(no_valid.any().item()):
            safe_valid[no_valid, 0] = True
        masked_logits = logits.masked_fill(~safe_valid, -1e9)
        dist = Categorical(logits=masked_logits)
        selected = (
            masked_logits.argmax(dim=-1)
            if deterministic
            else dist.sample()
        )
        route_one_hot = F.one_hot(selected, num_classes=ROUTE_COUNT).to(
            dtype=logits.dtype
        )
        return ActionSelection(
            route_index=selected,
            route_one_hot=route_one_hot,
            log_probability=dist.log_prob(selected),
            entropy=dist.entropy(),
            valid_route_mask=valid_mask,
        )


class CentralizedVehicleValueCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _make_mlp(VEHICLE_OBS_DIM)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 2 or int(obs.shape[-1]) != VEHICLE_OBS_DIM:
            raise ValueError(
                f"expected obs shape (N, {VEHICLE_OBS_DIM}), got {tuple(obs.shape)}"
            )
        return self.value_head(self.encoder(obs))


class CentralizedLatentValueCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = _make_mlp(LATENT_DIM + CONTEXT_DIM)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3 or int(z.shape[-1]) != LATENT_DIM:
            raise ValueError(
                f"expected z shape (B, E, {LATENT_DIM}), got {tuple(z.shape)}"
            )
        pooled = z.mean(dim=1)
        batch = int(pooled.shape[0])
        if tuple(context.shape) != (batch, CONTEXT_DIM):
            raise ValueError(
                f"expected context shape ({batch}, {CONTEXT_DIM}), got {tuple(context.shape)}"
            )
        context_in = context.to(device=pooled.device, dtype=pooled.dtype)
        return self.value_head(self.trunk(torch.cat([pooled, context_in], dim=-1)))


def lambda_return(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    *,
    terminal: torch.Tensor,
) -> torch.Tensor:
    if rewards.dim() != 1 or next_values.dim() != 1:
        raise ValueError(
            f"lambda_return expects 1D tensors, got rewards={tuple(rewards.shape)} next_values={tuple(next_values.shape)}"
        )
    if rewards.shape != next_values.shape:
        raise ValueError(
            f"rewards and next_values must match, got {tuple(rewards.shape)} and {tuple(next_values.shape)}"
        )
    terminal_mask = terminal.to(
        device=rewards.device,
        dtype=torch.bool,
    ).reshape(-1)
    if terminal_mask.shape != rewards.shape:
        raise ValueError(f"expected terminal shape {tuple(rewards.shape)}, got {tuple(terminal_mask.shape)}")

    returns = torch.zeros_like(rewards)
    running = next_values[-1]
    for idx in reversed(range(int(rewards.shape[0]))):
        if bool(terminal_mask[idx].item()):
            running = rewards[idx]
        else:
            running = rewards[idx] + DEFAULT_GAMMA * (
                (1.0 - DEFAULT_LAMBDA) * next_values[idx]
                + DEFAULT_LAMBDA * running
            )
        returns[idx] = running
    return returns


def capped_route_reward_matrix(
    route_cost_seconds: torch.Tensor,
    valid_route_mask: torch.Tensor,
) -> torch.Tensor:
    effective = torch.minimum(
        route_cost_seconds.clamp_min(0.0),
        torch.full_like(route_cost_seconds, ROUTE_COST_CAP_SECONDS),
    )
    rewards = -effective / REWARD_TEMPERATURE_SECONDS
    return torch.where(valid_route_mask, rewards, torch.zeros_like(rewards))


def lambda_returns_by_vehicle_ids(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    next_valid: torch.Tensor,
    vehicle_ids: torch.Tensor,
    *,
    terminal: torch.Tensor,
) -> torch.Tensor:
    if rewards.shape != next_values.shape:
        raise ValueError(
            f"rewards and next_values must match, got {tuple(rewards.shape)} and {tuple(next_values.shape)}"
        )
    returns = torch.zeros_like(rewards)
    if int(rewards.numel()) == 0:
        return returns

    masked_next = torch.where(
        next_valid.to(device=next_values.device, dtype=torch.bool),
        next_values,
        torch.zeros_like(next_values),
    )
    terminal_mask = terminal.to(
        device=rewards.device,
        dtype=torch.bool,
    ).reshape(-1)
    if terminal_mask.shape != rewards.shape:
        raise ValueError(
            f"expected terminal shape {tuple(rewards.shape)}, "
            f"got {tuple(terminal_mask.shape)}"
        )

    grouped: dict[int, list[int]] = {}
    ids = vehicle_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    for idx, vehicle_id in enumerate(ids[: int(rewards.numel())]):
        grouped.setdefault(int(vehicle_id), []).append(int(idx))

    for indices in grouped.values():
        idx_tensor = torch.tensor(indices, device=rewards.device, dtype=torch.long)
        returns.index_copy_(
            0,
            idx_tensor,
            lambda_return(
                rewards.index_select(0, idx_tensor),
                masked_next.index_select(0, idx_tensor),
                terminal=terminal_mask.index_select(0, idx_tensor),
            ),
        )
    return returns


__all__ = [
    "ActionSelection",
    "CentralizedLatentValueCritic",
    "CentralizedVehicleValueCritic",
    "VehicleRouteActor",
    "capped_route_reward_matrix",
    "lambda_return",
    "lambda_returns_by_vehicle_ids",
]
