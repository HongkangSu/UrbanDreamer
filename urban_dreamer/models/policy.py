from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

CANDIDATE_FEATURE_DIM = 10
VEHICLE_OBS_DIM = 14
LATENT_DIM = 64
CONTEXT_DIM = 2
HIDDEN_DIM = 256
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 0.95
REWARD_TEMPERATURE_SECONDS = 30.0


@dataclass(frozen=True)
class ActionSelection:
    candidate_index: torch.Tensor
    log_probability: torch.Tensor
    entropy: torch.Tensor


def _make_mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, HIDDEN_DIM), nn.Tanh(),
                         nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh())


def _initialize(module: nn.Module, head: nn.Linear, gain: float) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(head.weight, gain=gain)


class NextHopActor(nn.Module):
    """One shared scorer for any number of destination-reachable successor edges."""
    def __init__(self) -> None:
        super().__init__()
        self.candidate_scorer = _make_mlp(CANDIDATE_FEATURE_DIM)
        self.policy_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.policy_head, 0.01)

    def forward_logits(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != CANDIDATE_FEATURE_DIM:
            raise ValueError(f"expected candidate features (B, K, {CANDIDATE_FEATURE_DIM})")
        if mask.shape != features.shape[:2] or features.shape[1] == 0:
            raise ValueError("candidate mask must have shape (B, K) with K positive")
        mask = mask.to(device=features.device, dtype=torch.bool)
        if not bool(mask.any(dim=-1).all()):
            raise ValueError("every decision must have at least one legal candidate")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("candidate features must be finite")
        return self.policy_head(self.candidate_scorer(features)).squeeze(-1).masked_fill(~mask, -torch.inf)

    def sample_action(self, features: torch.Tensor, mask: torch.Tensor, *, deterministic: bool = False) -> ActionSelection:
        logits = self.forward_logits(features, mask)
        distribution = Categorical(logits=logits)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())


class CentralizedVehicleValueCritic(nn.Module):
    """Values arbitrary vehicle progress, including a committed turn or junction."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _make_mlp(VEHICLE_OBS_DIM)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.value_head, 1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.shape[-1] != VEHICLE_OBS_DIM:
            raise ValueError(f"expected vehicle observations (B, {VEHICLE_OBS_DIM})")
        return self.value_head(self.encoder(obs))


class CentralizedLatentValueCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = _make_mlp(LATENT_DIM + CONTEXT_DIM)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.value_head, 1.0)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != LATENT_DIM:
            raise ValueError(f"expected z shape (B, E, {LATENT_DIM})")
        if context.shape != (z.shape[0], CONTEXT_DIM):
            raise ValueError(f"expected context shape (B, {CONTEXT_DIM})")
        pooled = z.mean(dim=1)
        return self.value_head(self.trunk(torch.cat((pooled, context.to(pooled)), dim=-1)))


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
    if rewards.numel() == 0:
        return returns
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


__all__ = ["ActionSelection", "NextHopActor", "CentralizedVehicleValueCritic",
           "CentralizedLatentValueCritic", "lambda_return", "lambda_returns_by_vehicle_ids"]
