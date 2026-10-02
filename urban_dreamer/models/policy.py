from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..workflow.config import NONARRIVAL_PENALTY

CANDIDATE_FEATURE_DIM = 10
LATENT_DIM = 64
CONTEXT_DIM = 2
HIDDEN_DIM = 256
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 0.95


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


def _validate_prior(weight: float, temperature_seconds: float) -> None:
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("shortest-path prior weight must be finite and nonnegative")
    if not math.isfinite(temperature_seconds) or temperature_seconds <= 0:
        raise ValueError("shortest-path prior temperature_seconds must be finite and positive")


def set_shortest_path_prior(actor: nn.Module, weight: float, temperature_seconds: float) -> None:
    """Set nontrainable behavior settings; callers restore them from episode/config."""
    _validate_prior(weight, temperature_seconds)
    actor.prior_weight = float(weight)
    actor.prior_temperature_seconds = float(temperature_seconds)


def shortest_path_prior_weight(episode: int, *, initial_weight: float = 1.,
                              hold_episodes: int = 200, off_episode: int = 1000) -> float:
    """Hold through ``hold_episodes``, then cosine-decay to zero at ``off_episode``."""
    for name, value in (("episode", episode), ("hold_episodes", hold_episodes),
                        ("off_episode", off_episode)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    _validate_prior(initial_weight, 1.)
    if off_episode <= hold_episodes:
        raise ValueError("off_episode must be greater than hold_episodes")
    if episode <= hold_episodes:
        return float(initial_weight)
    if episode >= off_episode:
        return 0.
    progress = (episode - hold_episodes) / (off_episode - hold_episodes)
    return float(initial_weight) * 0.5 * (1. + math.cos(math.pi * progress))


def apply_shortest_path_prior(logits: torch.Tensor, features: torch.Tensor,
                             mask: torch.Tensor, *, weight: float,
                             temperature_seconds: float) -> torch.Tensor:
    """Bias legal logits by full via-candidate free-flow cost, measured in seconds.

    Feature 4 is log1p(junction + candidate edge + downstream seconds) /
    log1p(3600). Costs are detached conditions, not another learned objective.
    Input logits are already legally masked when this is called by an actor.
    """
    _validate_prior(weight, temperature_seconds)
    if weight == 0:
        return logits
    if logits.ndim != 2 or features.shape != (*logits.shape, CANDIDATE_FEATURE_DIM):
        raise ValueError("prior requires logits (B, K) and candidate features (B, K, 10)")
    if mask.shape != logits.shape or logits.shape[-1] == 0:
        raise ValueError("candidate mask must have shape (B, K) with K positive")
    legal = mask.to(device=logits.device, dtype=torch.bool)
    if not bool(legal.any(dim=-1).all()):
        raise ValueError("every decision must have at least one legal candidate")
    # Mask before exponentiation: padded candidate features must not introduce NaNs.
    encoded = features[..., 4].detach().to(logits).masked_fill(~legal, 0.)
    if not bool(torch.isfinite(encoded).all()) or bool((encoded < 0).any()):
        raise ValueError("legal free-flow cost features must be finite and nonnegative")
    costs = torch.expm1(encoded * math.log1p(3600.))
    if not bool(torch.isfinite(costs).all()) or not bool(torch.isfinite(logits[legal]).all()):
        raise ValueError("legal free-flow costs and actor logits must be finite")
    minimum = costs.masked_fill(~legal, torch.inf).amin(dim=-1, keepdim=True)
    bias = (costs - minimum).masked_fill(~legal, 0.) * (weight / temperature_seconds)
    if not bool(torch.isfinite(bias).all()):
        raise ValueError("shortest-path prior bias must be finite")
    return (logits - bias).masked_fill(~legal, -torch.inf)


class NextHopActor(nn.Module):
    """One shared scorer for any number of destination-reachable successor edges."""
    def __init__(self) -> None:
        super().__init__()
        self.candidate_scorer = _make_mlp(CANDIDATE_FEATURE_DIM)
        self.policy_head = nn.Linear(HIDDEN_DIM, 1)
        _initialize(self, self.policy_head, 0.01)
        set_shortest_path_prior(self, 0., 30.)

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
        logits = self.policy_head(self.candidate_scorer(features)).squeeze(-1).masked_fill(~mask, -torch.inf)
        return apply_shortest_path_prior(logits, features, mask, weight=self.prior_weight,
                                         temperature_seconds=self.prior_temperature_seconds)

    def sample_action(self, features: torch.Tensor, mask: torch.Tensor, *, deterministic: bool = False) -> ActionSelection:
        logits = self.forward_logits(features, mask)
        distribution = Categorical(logits=logits)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())


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


def attach_timeout_penalties(rows: list[dict], failed_vehicle_ids: set) -> None:
    """Record one terminal cost per failed vehicle and its own gamma=1 returns.

    Rows are chronological within each vehicle's trajectory. The caller must
    supply failures only at the real deadline and call separately per rollout.
    A vehicle without decisions produces no synthetic policy-training row.
    """
    last_actions = {}
    for row in rows:
        failed = row['vehicle_id'] in failed_vehicle_ids
        row['timeout_penalty'] = 0.
        row['timeout_return'] = NONARRIVAL_PENALTY if failed else 0.
        if failed:
            last_actions[row['vehicle_id']] = row
    for row in last_actions.values():
        row['timeout_penalty'] = NONARRIVAL_PENALTY


def individual_actor_advantage(global_advantages: torch.Tensor,
                               timeout_returns: torch.Tensor) -> torch.Tensor:
    """Scale shared traffic advantages before adding vehicle-specific costs.

    Never compute batch statistics over penalties: marking one vehicle failed
    must not rescale the advantage of an arrived vehicle in the same batch.
    """
    timeout_returns = timeout_returns.to(global_advantages)
    if global_advantages.ndim != 1 or timeout_returns.shape != global_advantages.shape:
        raise ValueError('global advantages and individual timeout returns must be matching vectors')
    if not torch.isfinite(global_advantages).all() or not torch.isfinite(timeout_returns).all():
        raise ValueError('actor advantages and timeout returns must be finite')
    if not global_advantages.numel():
        return global_advantages.clone()
    scale = global_advantages.std(unbiased=False).clamp_min(1.)
    return (global_advantages / scale + timeout_returns).clamp(-2.5, 2.5)


__all__ = ["ActionSelection", "NextHopActor", "CentralizedLatentValueCritic", "lambda_return",
           "attach_timeout_penalties", "individual_actor_advantage", "apply_shortest_path_prior",
           "shortest_path_prior_weight", "set_shortest_path_prior"]
