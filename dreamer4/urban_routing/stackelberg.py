"""
Edge-level Stackelberg response layer for the Urban World Model.

This module intentionally stays in edge space:
  - AV input: edge vehicle volumes
  - HDV response: edge distribution / edge vehicle volumes
  - travel time: shared BPR volume-delay model

No route/OD decision distribution is required in this layer.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLayer(nn.Module):
    """Learnable BPR (Bureau of Public Roads) volume-delay model.

    ``t_e = free_flow * (1 + α * (v_e / capacity)^β)``

    α and β are learnable via softplus parameterization. This module is
    shared between StackelbergLayer and the reward / observation computation
    so all BPR evaluations use the same parameters.
    """

    def __init__(
        self,
        lengths: torch.Tensor,        # (E,)
        speed_limits: torch.Tensor,   # (E,)
        capacities: torch.Tensor,     # (E,)
        alpha_init: float = 0.0,
        beta_init: float = 1.4,
    ):
        super().__init__()
        self._alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
        self._beta_raw = nn.Parameter(torch.tensor(float(beta_init)))
        self.register_buffer(
            "free_flow_time",
            lengths.float() / speed_limits.float().clamp(min=0.1),
        )
        self.register_buffer("capacities", capacities.float())

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self._alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self._beta_raw)

    def forward(self, v_e: torch.Tensor) -> torch.Tensor:
        """Compute edge travel times from edge vehicle volumes. (E,) -> (E,)"""
        return self.free_flow_time * (
            1.0 + self.alpha * (v_e / self.capacities.clamp(min=1.0)).pow(self.beta)
        )

    def numpy_forward(self, v_e: np.ndarray) -> np.ndarray:
        """Numpy version for reward / inference (no grad)."""
        ff = self.free_flow_time.detach().cpu().numpy()
        cap = np.maximum(self.capacities.detach().cpu().numpy(), 1.0)
        a = float(self.alpha.detach().cpu())
        b = float(self.beta.detach().cpu())
        return ff * (1.0 + a * (v_e / cap) ** b)


class StackelbergLayer(nn.Module):
    """
    Differentiable edge-level Stackelberg response layer.

    Given AV edge volumes and background traffic:
      1. Compute edge travel times via shared BPR parameters
      2. Solve an edge-level HDV fixed point with logit dynamics
      3. Return normalized HDV edge distribution and edge travel times
    """

    def __init__(
        self,
        n_edges: int,
        bpr: BPRLayer,
        total_av_count: float,
        total_hdv_count: float,
        n_iters: int = 10,
        damping: float = 0.5,
    ):
        super().__init__()
        self.n_edges = int(n_edges)
        self.n_iters = int(n_iters)
        self.damping = float(damping)
        self.bpr = bpr

        # Logit temperature for edge-choice dynamics.
        self._tau_raw = nn.Parameter(torch.tensor(-1.0))  # softplus -> tau ~= 0.31

        self.register_buffer(
            "total_av_count",
            torch.tensor(float(total_av_count), dtype=torch.float32),
        )
        self.register_buffer(
            "total_hdv_count",
            torch.tensor(float(total_hdv_count), dtype=torch.float32),
        )

    @property
    def alpha(self) -> torch.Tensor:
        return self.bpr.alpha

    @property
    def beta(self) -> torch.Tensor:
        return self.bpr.beta

    @property
    def capacities(self) -> torch.Tensor:
        return self.bpr.capacities

    @property
    def free_flow_time(self) -> torch.Tensor:
        return self.bpr.free_flow_time

    @property
    def tau(self) -> torch.Tensor:
        return F.softplus(self._tau_raw).clamp(min=0.01)

    def solve(
        self,
        av_edge_volume: torch.Tensor,
        bg_volume: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solve edge-level HDV equilibrium given AV edge volumes and background traffic.

        Args:
            av_edge_volume: (E,) AV vehicle counts per edge.
            bg_volume:      (E,) background edge vehicle counts.

        Returns:
            hdv_edge:     (E,) normalized HDV edge distribution.
            edge_tt:      (E,) equilibrium edge travel times.
            eq_residual:  scalar fixed-point residual ||h - softmax(-tt/tau)||^2.
        """
        tau = self.tau
        dtype = av_edge_volume.dtype
        device = av_edge_volume.device
        total_hdv_count = self.total_hdv_count.to(device=device, dtype=dtype)

        # Initialize from free-flow preference to avoid arbitrary edge bias.
        h = F.softmax(-self.free_flow_time.to(device=device, dtype=dtype) / tau, dim=0)
        t_e = self.bpr(bg_volume + av_edge_volume + h * total_hdv_count)

        for _ in range(self.n_iters):
            hdv_edge_flow = h * total_hdv_count
            v_e = bg_volume + av_edge_volume + hdv_edge_flow
            t_e = self.bpr(v_e)
            h_target = F.softmax(-t_e / tau, dim=0)
            h = (1.0 - self.damping) * h_target + self.damping * h

        h_target = F.softmax(-t_e / tau, dim=0)
        eq_residual = (h - h_target).pow(2).mean()
        hdv_edge = h / h.sum().clamp(min=1e-8)
        return hdv_edge, t_e, eq_residual

    def forward(
        self,
        av_edge_volume: torch.Tensor,
        bg_volume: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.solve(av_edge_volume, bg_volume)


def decode_bg_volume(
    z_t: torch.Tensor,
    flow_projector: "FlowProjector",
    capacities: torch.Tensor,
) -> torch.Tensor:
    """
    Decode background edge vehicle counts from latent state z_t.

    Args:
        z_t:            (E, 64) current latent state.
        flow_projector: FlowProjector instance with decode() method.
        capacities:     (E,) edge vehicle capacities.

    Returns:
        (E,) background vehicle counts per edge.
    """
    decode_fn = getattr(flow_projector, "decode", None)
    if decode_fn is None:
        decode_fn = flow_projector.inverse
    decoded = decode_fn(z_t)  # (E, 4)
    veh_ratio = decoded[:, 2].clamp(min=0.0)
    return veh_ratio * capacities
