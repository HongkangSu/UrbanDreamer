from __future__ import annotations

from typing import List

import math
import numpy as np
import torch
import torch.nn as nn


class LatentGlobalCritic(nn.Module):
    """Critic-only MLP that can consume richer global latent context."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(int(n_hidden_layers) - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, critic_obs: torch.Tensor) -> torch.Tensor:
        h = self.trunk(critic_obs)
        return self.value_head(h)


def latent_global_summary(
    z_tokens: torch.Tensor,
    *,
    variant: str = "mean",
) -> torch.Tensor:
    """Build a global latent summary for critic-only conditioning.

    Supported variants:
    - ``mean``: mean-pool edge latent tokens
    - ``mean_std``: concatenate mean and per-dimension std
    """
    if z_tokens.dim() != 2:
        raise ValueError(f"expected z_tokens shape (E, D), got {tuple(z_tokens.shape)}")
    variant = str(variant).lower()
    mean = z_tokens.mean(dim=0)
    if variant == "mean":
        return mean
    if variant == "mean_std":
        std = z_tokens.std(dim=0, unbiased=False)
        return torch.cat([mean, std], dim=0)
    raise ValueError(f"unsupported latent global summary variant: {variant}")


def latent_global_summary_dim(
    d_spatial: int,
    *,
    variant: str = "mean",
) -> int:
    variant = str(variant).lower()
    if variant == "mean":
        return int(d_spatial)
    if variant == "mean_std":
        return int(d_spatial) * 2
    raise ValueError(f"unsupported latent global summary variant: {variant}")


def append_global_summary_to_obs_np(
    obs_np: np.ndarray,
    summary_np: np.ndarray,
) -> np.ndarray:
    """Tile one global summary across agents and append it to critic input."""
    obs_np = np.asarray(obs_np, dtype=np.float32)
    summary_np = np.asarray(summary_np, dtype=np.float32).reshape(-1)
    if obs_np.ndim != 2:
        raise ValueError(f"obs_np must be 2D, got shape={obs_np.shape}")
    if obs_np.shape[0] == 0:
        return np.zeros((0, obs_np.shape[1] + summary_np.shape[0]), dtype=np.float32)
    tiled = np.tile(summary_np[None, :], (obs_np.shape[0], 1))
    return np.concatenate([obs_np, tiled], axis=1).astype(np.float32, copy=False)
