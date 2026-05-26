from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.nn as nn


class VehicleRouteActor(nn.Module):
    """Vehicle-level route actor used by the Dreamer-style CTDE mainline."""

    def __init__(
        self,
        obs_dim: int,
        n_routes: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        route_cost_logit_bias: float = 0.0,
        route_cost_context_dim: int = 3,
        architecture: str = "mlp",
        include_global: bool = True,
        include_route_costs: bool = False,
        include_fleet_pressure: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_routes = int(n_routes)
        self.route_cost_logit_bias = float(route_cost_logit_bias)
        self.route_cost_context_dim = int(route_cost_context_dim)
        self.architecture = str(architecture or "mlp").strip().lower()
        self.include_global = bool(include_global)
        self.include_route_costs = bool(include_route_costs)
        self.include_fleet_pressure = bool(include_fleet_pressure)

        if self.architecture not in {"mlp", "route_shared"}:
            raise ValueError(f"Unsupported VehicleRouteActor architecture: {architecture!r}")

        if self.architecture == "route_shared":
            self.route_feature_dim = self._infer_route_feature_dim()
            shared_in_dim = self.route_feature_dim
            if self.include_global:
                shared_in_dim += self.route_feature_dim
            if self.include_fleet_pressure:
                shared_in_dim += 2
            if self.include_route_costs:
                shared_in_dim += 3
            shared_in_dim += self.route_cost_context_dim
            layers = [nn.Linear(shared_in_dim, hidden_dim), nn.Tanh()]
            for _ in range(max(int(n_hidden_layers) - 1, 0)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            self.route_scorer = nn.Sequential(*layers)
            self.policy_head = nn.Linear(hidden_dim, 1)
            self.trunk = nn.Identity()
        else:
            self.route_feature_dim = 0
            layers = [nn.Linear(self.obs_dim, hidden_dim), nn.Tanh()]
            for _ in range(max(int(n_hidden_layers) - 1, 0)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            self.trunk = nn.Sequential(*layers)
            self.policy_head = nn.Linear(hidden_dim, self.n_routes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def _infer_route_feature_dim(self) -> int:
        context_dim = self.route_cost_context_dim
        cost_dim = 3 * self.n_routes if self.include_route_costs else 0
        fleet_dim = 2 * self.n_routes if self.include_fleet_pressure else 0
        route_and_global = self.obs_dim - context_dim - cost_dim - fleet_dim
        denom = self.n_routes + (1 if self.include_global else 0)
        if denom <= 0 or route_and_global <= 0 or route_and_global % denom != 0:
            raise ValueError(
                "Cannot infer route feature dim for route_shared actor: "
                f"obs_dim={self.obs_dim}, n_routes={self.n_routes}, include_global={self.include_global}, "
                f"include_route_costs={self.include_route_costs}, include_fleet_pressure={self.include_fleet_pressure}, "
                f"context_dim={context_dim}"
            )
        return int(route_and_global // denom)

    def _effective_route_feature_dim(self) -> int:
        if int(self.route_feature_dim) > 0:
            return int(self.route_feature_dim)
        context_dim = self.route_cost_context_dim
        cost_dim = 3 * self.n_routes if self.include_route_costs else 0
        fleet_dim = 2 * self.n_routes if self.include_fleet_pressure else 0
        route_and_global = self.obs_dim - context_dim - cost_dim - fleet_dim
        denom = self.n_routes + (1 if self.include_global else 0)
        if denom <= 0 or route_and_global <= 0 or route_and_global % denom != 0:
            return 0
        return int(route_and_global // denom)

    def _base_feature_dim(self) -> int:
        return self._effective_route_feature_dim() * (self.n_routes + (1 if self.include_global else 0))

    def _route_cost_start(self) -> int:
        return self._base_feature_dim()

    def _fleet_pressure_start(self) -> int:
        return self._base_feature_dim() + (3 * self.n_routes if self.include_route_costs else 0)

    def _route_shared_inputs(self, obs: torch.Tensor) -> torch.Tensor:
        batch = int(obs.shape[0])
        K = self.n_routes
        rd = self.route_feature_dim
        off = 0
        route_feats = obs[..., off:off + K * rd].reshape(batch, K, rd)
        off += K * rd
        pieces = [route_feats]
        if self.include_global:
            global_feat = obs[..., off:off + rd].unsqueeze(1).expand(batch, K, rd)
            off += rd
            pieces.append(global_feat)
        if self.include_route_costs:
            cost_start = self._route_cost_start()
            cost = obs[..., cost_start:cost_start + 3 * K]
            gap = cost[..., :K].unsqueeze(-1)
            log_cost = cost[..., K:2 * K].unsqueeze(-1)
            valid = cost[..., 2 * K:3 * K].unsqueeze(-1)
            pieces.extend([gap, log_cost, valid])
        if self.include_fleet_pressure:
            fleet_start = self._fleet_pressure_start()
            fleet = obs[..., fleet_start:fleet_start + 2 * K]
            route_share = fleet[..., :K].unsqueeze(-1)
            pressure = fleet[..., K:2 * K].unsqueeze(-1)
            pieces.extend([route_share, pressure])
        if self.route_cost_context_dim > 0:
            context = obs[..., -self.route_cost_context_dim:].unsqueeze(1)
            context = context.expand(batch, K, self.route_cost_context_dim)
            pieces.append(context)
        return torch.cat(pieces, dim=-1)

    def forward_logits(self, obs: torch.Tensor) -> torch.Tensor:
        if self.architecture == "route_shared":
            per_route = self._route_shared_inputs(obs)
            flat = per_route.reshape(-1, per_route.shape[-1])
            h = self.route_scorer(flat)
            logits = self.policy_head(h).reshape(obs.shape[0], self.n_routes)
            if self.include_route_costs:
                valid_start = self._route_cost_start() + 2 * self.n_routes
                valid_end = valid_start + self.n_routes
                if valid_start >= 0 and obs.shape[-1] >= valid_end:
                    valid = obs[..., valid_start:valid_end].to(dtype=logits.dtype)
                    logits = logits.masked_fill(valid <= 0.0, -20.0)
                    all_invalid = valid.sum(dim=-1, keepdim=True) <= 0.0
                    logits = torch.where(all_invalid, torch.zeros_like(logits), logits)
        else:
            h = self.trunk(obs)
            logits = self.policy_head(h)
        if self.route_cost_logit_bias != 0.0:
            # When route-cost observation is enabled, the final context vector
            # follows [relative route gaps, log route costs, valid masks].
            # Subtracting the non-negative gap is a structural cost prior; the
            # neural policy still learns the residual preference.
            if self.architecture == "route_shared":
                gap_start = self._route_cost_start()
            else:
                gap_start = self._base_feature_dim()
            gap_end = gap_start + self.n_routes
            if gap_start >= 0 and obs.shape[-1] >= gap_end:
                route_gap = obs[..., gap_start:gap_end].to(dtype=logits.dtype)
                logits = logits - float(self.route_cost_logit_bias) * route_gap
        return torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)

    def get_av_frac(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        logits = self.forward_logits(obs)
        if deterministic:
            return torch.softmax(logits, dim=-1)
        noise = torch.randn_like(logits)
        return torch.softmax(logits + noise, dim=-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        raw_action: Optional[torch.Tensor] = None,
    ):
        logits = self.forward_logits(obs)
        used_logits = logits if raw_action is None else raw_action
        probs = torch.softmax(used_logits, dim=-1)
        log_probs = torch.log_softmax(used_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        fake_lp = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        fake_value = torch.zeros(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
        return probs, used_logits, fake_lp, entropy, fake_value


class CentralizedLatentValueCritic(nn.Module):
    """Centralized latent critic over pooled z, optionally conditioned on action_edge."""

    def __init__(
        self,
        d_spatial: int,
        action_dim: int = 0,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.d_spatial = int(d_spatial)
        self.action_dim = int(max(action_dim, 0))
        input_dim = self.d_spatial + self.action_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(max(int(n_hidden_layers) - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def summarize(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            return z.mean(dim=0, keepdim=True)
        if z.dim() == 3:
            return z.mean(dim=1)
        raise ValueError(f"expected z shape (E, D) or (B, E, D), got {tuple(z.shape)}")

    def _prepare_action(self, action: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        if self.action_dim <= 0:
            return torch.zeros((batch_size, 0), dtype=torch.float32)
        if action is None:
            raise ValueError("action-conditioned critic requires action input")
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() != 2 or action.shape[0] != batch_size or action.shape[1] != self.action_dim:
            raise ValueError(
                f"expected action shape ({batch_size}, {self.action_dim}), got {tuple(action.shape)}"
            )
        return action

    def forward(self, z: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        pooled = self.summarize(z)
        if self.action_dim > 0:
            action_in = self._prepare_action(action, pooled.shape[0]).to(device=pooled.device, dtype=pooled.dtype)
            critic_in = torch.cat([pooled, action_in], dim=-1)
        else:
            critic_in = pooled
        h = self.trunk(critic_in)
        return self.value_head(h)


def lambda_return(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    """Compute λ-returns for a reward sequence and next-state bootstrap values."""
    if rewards.dim() != 1 or next_values.dim() != 1:
        raise ValueError(
            f"lambda_return expects 1D tensors, got rewards={tuple(rewards.shape)} next_values={tuple(next_values.shape)}"
        )
    if rewards.shape[0] != next_values.shape[0]:
        raise ValueError(
            f"rewards and next_values length mismatch: {rewards.shape[0]} vs {next_values.shape[0]}"
        )
    ret = torch.zeros_like(rewards)
    acc = next_values[-1]
    g = float(gamma)
    lam = float(lambda_)
    for t in reversed(range(rewards.shape[0])):
        bootstrap = next_values[t]
        if t == rewards.shape[0] - 1:
            acc = rewards[t] + g * bootstrap
        else:
            acc = rewards[t] + g * ((1.0 - lam) * bootstrap + lam * acc)
        ret[t] = acc
    return ret


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def mean_last(values: Iterable[float]) -> float:
    seq = list(values)
    if not seq:
        return float("nan")
    return float(sum(seq) / len(seq))
