"""
Physics-Constrained Flow Matching (PCFM)

Applies hard and soft physics constraints during CFM ODE sampling to ensure
decoded traffic states respect fundamental traffic laws:
  C1: Non-negativity and value range bounds (hard)
  C2: Vehicle conservation (hard)
  C3: Capacity limits — merged into C1 via kappa clamp (hard)
  C4: BPR speed-flow consistency (soft guidance)
  C5: Kirchhoff node flow conservation (soft guidance)

Two classes:
  PhysicsConstraints — holds constraint parameters and implements correction/guidance
  PCFMSampler       — wraps dynamics model + constraints for physics-guided ODE sampling
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PhysicsConstraints:
    """
    Container for physics constraint parameters and correction methods.

    Not an nn.Module — stores precomputed tensors and provides stateless
    constraint operations on decoded spatial states s_hat of shape (E, 4).

    Channels:
        0: occ   — occupancy         [0, 1]
        1: spd   — mean_speed/limit  [0, +inf)
        2: veh   — vehicle_number/cap [0, kappa]
        3: halt  — halting_number/cap [0, kappa]

    The lift matrix maps corrections in decoded (E,4) space back to the
    latent (E,64) space:  delta_z = delta_s @ lift.
    """

    def __init__(
        self,
        inv_proj_weight: torch.Tensor,
        capacities: torch.Tensor,
        N_total: float,
        node_incidence: Optional[torch.Tensor] = None,
        bpr_alpha: float = 0.15,
        bpr_beta: float = 4.0,
        kappa: float = 1.2,
    ):
        """
        Args:
            inv_proj_weight: (4, 64) tensor — weight matrix of the inv_proj
                linear layer (nn.Linear(64, 4).weight has shape (4, 64) in
                PyTorch convention: out_features x in_features).
            capacities: (E,) tensor — per-edge vehicle capacity estimates.
            N_total: float — total number of vehicles in the network for
                conservation constraint.
            node_incidence: optional (N_internal, E) tensor — signed incidence
                matrix for internal nodes.  Entry (n, e) is +1 if edge e enters
                node n, -1 if it leaves, 0 otherwise.  Used for Kirchhoff
                flow conservation guidance.
            bpr_alpha: BPR parameter alpha (default 0.15).
            bpr_beta: BPR parameter beta (default 4.0).
            kappa: Maximum allowed veh/cap and halt/cap ratio (default 1.2).
        """
        # W: (4, 64) — inv_proj weight
        W = inv_proj_weight.detach().float()
        self.W = W

        # Precompute lift matrix: lift = (W W^T)^{-1} W, shape (4, 64)
        # W W^T is only 4x4, cheap to invert.
        WWT = W @ W.T  # (4, 4)
        self.lift = torch.linalg.solve(WWT, W)  # (4, 64)

        self.capacities = capacities.detach().float()  # (E,)
        self.N_total = float(N_total)
        self.node_incidence = (
            node_incidence.detach().float() if node_incidence is not None else None
        )
        self.bpr_alpha = float(bpr_alpha)
        self.bpr_beta = float(bpr_beta)
        self.kappa = float(kappa)

    def to(self, device: torch.device) -> "PhysicsConstraints":
        """Move all internal tensors to the given device."""
        self.W = self.W.to(device)
        self.lift = self.lift.to(device)
        self.capacities = self.capacities.to(device)
        if self.node_incidence is not None:
            self.node_incidence = self.node_incidence.to(device)
        return self

    # ------------------------------------------------------------------
    # C1 + C2: Hard constraints
    # ------------------------------------------------------------------

    def clamp_and_conserve(
        self,
        s_hat: torch.Tensor,
        n_rounds: int = 2,
    ) -> torch.Tensor:
        """
        Apply hard physics constraints by alternating value-range clamping (C1)
        and vehicle conservation projection (C2).

        Args:
            s_hat: (E, 4) decoded spatial state.
            n_rounds: number of alternating C1+C2 rounds (default 2).

        Returns:
            Corrected s_hat of the same shape (E, 4).
        """
        s = s_hat.clone()
        for _ in range(n_rounds):
            # C1: Value-range clamp per channel
            #   occ  in [0, 1]
            #   spd  in [0, +inf)
            #   veh  in [0, kappa]
            #   halt in [0, kappa]
            s[:, 0] = s[:, 0].clamp(0.0, 1.0)
            s[:, 1] = s[:, 1].clamp(min=0.0)
            s[:, 2] = s[:, 2].clamp(0.0, self.kappa)
            s[:, 3] = s[:, 3].clamp(0.0, self.kappa)

            # C2: Vehicle conservation
            #   Total vehicles = sum_e (s[e,2] * C_e) should equal N_total.
            #   Correct by uniform shift: s[e,2] -= delta / sum(C_e)
            total_veh = (s[:, 2] * self.capacities).sum()
            delta = total_veh - self.N_total
            cap_sum = self.capacities.sum().clamp(min=1e-8)
            s[:, 2] = s[:, 2] - delta / cap_sum

        return s

    # ------------------------------------------------------------------
    # C4: BPR speed-flow consistency (soft guidance)
    # ------------------------------------------------------------------

    def bpr_guidance(self, s_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the analytical gradient of the BPR consistency loss w.r.t.
        s_hat for soft guidance.

        The BPR speed-flow relationship is:
            spd_bpr_e = 1 / (1 + alpha * max(veh_e, 0)^beta)

        Loss:
            L_BPR = sum_e (s_hat[e,1] - spd_bpr_e)^2

        Returns the gradient dL/ds_hat of shape (E, 4), computed analytically
        (faster than autograd).

        Args:
            s_hat: (E, 4) decoded spatial state.

        Returns:
            grad: (E, 4) gradient tensor.
        """
        spd = s_hat[:, 1]  # (E,)
        veh = s_hat[:, 2]  # (E,)

        veh_pos = veh.clamp(min=0.0)
        veh_pow = veh_pos.pow(self.bpr_beta)  # max(veh, 0)^beta
        denom = 1.0 + self.bpr_alpha * veh_pow  # (E,)
        spd_bpr = 1.0 / denom  # (E,)

        residual = spd - spd_bpr  # (E,)

        # dL/d_spd = 2 * (spd - spd_bpr), per edge
        grad_spd = 2.0 * residual  # (E,)

        # dL/d_veh: chain rule through spd_bpr
        #   d(spd_bpr)/d(veh) = -alpha * beta * veh^(beta-1) / (1 + alpha*veh^beta)^2
        #   but only where veh > 0 (clamp gradient)
        #   dL/d_veh = 2 * (spd - spd_bpr) * (-d(spd_bpr)/d(veh))
        #            = 2 * residual * alpha * beta * veh^(beta-1) / denom^2
        mask = (veh > 0.0).float()
        # veh^(beta-1): safe computation
        veh_safe = veh_pos.clamp(min=1e-8)
        d_spd_bpr_d_veh = (
            -self.bpr_alpha
            * self.bpr_beta
            * veh_safe.pow(self.bpr_beta - 1.0)
            / denom.pow(2)
        )
        # dL/d_veh = 2 * residual * (- d_spd_bpr/d_veh)  [note the sign]
        # L = (spd - spd_bpr)^2, dL/d_veh = 2*(spd-spd_bpr)*(-d_spd_bpr/d_veh)
        grad_veh = 2.0 * residual * (-d_spd_bpr_d_veh) * mask  # (E,)

        # Assemble full gradient (E, 4): only channels 1 (spd) and 2 (veh) are nonzero
        grad = torch.zeros_like(s_hat)
        grad[:, 1] = grad_spd
        grad[:, 2] = grad_veh

        return grad

    # ------------------------------------------------------------------
    # C5: Kirchhoff node flow conservation (soft guidance)
    # ------------------------------------------------------------------

    def kirchhoff_guidance(self, s_hat: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute the gradient of the Kirchhoff flow conservation loss w.r.t.
        s_hat for soft guidance.

        Loss:
            v_hat_e = s_hat[e, 2] * capacities[e]
            L_K = || M @ v_hat ||^2
        where M is the (N_internal, E) node incidence matrix.

        Gradient (analytical):
            dL/d_v_hat = 2 * M^T @ (M @ v_hat)          — shape (E,)
            dL/d_s_hat[:,2] = dL/d_v_hat * capacities    — chain rule

        Args:
            s_hat: (E, 4) decoded spatial state.

        Returns:
            grad: (E, 4) gradient tensor, or None if node_incidence is
                  not available.
        """
        if self.node_incidence is None:
            return None

        M = self.node_incidence  # (N_internal, E)
        v_hat = s_hat[:, 2] * self.capacities  # (E,)

        # Kirchhoff residual: flow imbalance at each internal node
        Mv = M @ v_hat  # (N_internal,)

        # Gradient w.r.t. v_hat: 2 * M^T @ (M @ v_hat)
        grad_v = 2.0 * (M.T @ Mv)  # (E,)

        # Chain to s_hat[:,2]: dv_hat/ds[:,2] = capacities
        grad_veh = grad_v * self.capacities  # (E,)

        grad = torch.zeros_like(s_hat)
        grad[:, 2] = grad_veh

        return grad


class PCFMSampler:
    """
    Physics-Constrained Flow Matching sampler.

    Wraps a dynamics model and PhysicsConstraints to perform ODE sampling
    with interleaved soft guidance (BPR, Kirchhoff) and hard projection
    (clamp + conservation) at each Euler step.

    Usage:
        constraints = PhysicsConstraints(inv_proj_weight, capacities, N_total, ...)
        sampler = PCFMSampler(dynamics, constraints)
        z_next = sampler.sample(past_tokens, actions)
    """

    def __init__(
        self,
        dynamics: nn.Module,
        physics_constraints: PhysicsConstraints,
        inv_proj: nn.Module,
        n_ode_steps: int = 4,
        lambda_bpr: float = 0.05,
        lambda_kirchhoff: float = 0.03,
    ):
        """
        Args:
            dynamics: trained Dynamics module (CFM-based).
            physics_constraints: PhysicsConstraints instance with precomputed
                lift matrix and constraint parameters.
            inv_proj: the inv_proj linear layer (nn.Linear(64, 4)) used to
                decode latent tokens to physical space.
            n_ode_steps: number of Euler integration steps (default 4).
            lambda_bpr: guidance strength for BPR consistency (default 0.05).
            lambda_kirchhoff: guidance strength for Kirchhoff conservation
                (default 0.03).
        """
        self.dynamics = dynamics
        self.pc = physics_constraints
        self.inv_proj = inv_proj
        self.n_ode_steps = max(1, int(n_ode_steps))
        self.lambda_bpr = float(lambda_bpr)
        self.lambda_kirchhoff = float(lambda_kirchhoff)

    def sample(
        self,
        past_tokens: torch.Tensor,
        actions: torch.Tensor,
        act_mask: Optional[torch.Tensor] = None,
        agent_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run physics-constrained CFM ODE sampling to predict the next latent state.

        At each Euler step:
          1. Query dynamics for the velocity field v.
          2. Decode current x to physical space s_hat via inv_proj.
          3. Compute soft guidance from BPR and Kirchhoff losses.
          4. Euler update: x += (v + guidance_lifted) * dt.
          5. Hard projection: decode, clamp_and_conserve, lift back.

        Args:
            past_tokens: (B, T, E, 64) — history of projected flow tokens.
            actions: (B, T+1, action_dim) — padded action sequence; the
                last entry is the current action.
            act_mask: optional action mask passed to dynamics.
            agent_tokens: optional agent tokens passed to dynamics.

        Returns:
            z_next: (B, 1, E, 64) — predicted next-step latent tokens.
        """
        device = past_tokens.device
        dtype = past_tokens.dtype
        B, T = past_tokens.shape[:2]
        n_spatial = past_tokens.shape[2]  # E
        d_spatial = past_tokens.shape[3]  # 64

        # Ensure constraint tensors are on the right device
        lift = self.pc.lift.to(device=device, dtype=dtype)  # (4, 64)

        # Initialize noise
        x = torch.randn(
            (B, 1, n_spatial, d_spatial), device=device, dtype=dtype
        )
        dt = 1.0 / self.n_ode_steps

        for i in range(self.n_ode_steps):
            t_val = float(i) * dt

            # Build flow_time: zeros for history, t_val for current prediction
            flow_time = torch.zeros(
                (B, T + 1), device=device, dtype=torch.float32
            )
            flow_time[:, -1] = t_val

            # Concatenate history + current noise as full input sequence
            full_seq = torch.cat([past_tokens, x], dim=1)  # (B, T+1, E, 64)

            # Dynamics forward: get velocity prediction
            t_tensor = flow_time
            v_pred, h_t = self.dynamics(
                actions,
                None,
                None,
                full_seq,
                flow_time=t_tensor,
                act_mask=act_mask,
                agent_tokens=agent_tokens,
            )
            # v_pred: (B, T+1, E, 64) — take the last timestep
            v = v_pred[:, -1:, :, :]  # (B, 1, E, 64)

            # --- Soft guidance ---
            # Decode current x to physical space for guidance computation
            # Process each batch element; guidance is computed per-sample
            guidance = torch.zeros_like(x)  # (B, 1, E, 64)

            for b in range(B):
                s_hat_b = self.inv_proj(x[b, 0].float())  # (E, 4)

                g_s = torch.zeros_like(s_hat_b)  # (E, 4)

                # C4: BPR guidance
                if self.lambda_bpr > 0.0:
                    bpr_grad = self.pc.bpr_guidance(s_hat_b)  # (E, 4)
                    g_s = g_s - self.lambda_bpr * bpr_grad

                # C5: Kirchhoff guidance
                if self.lambda_kirchhoff > 0.0:
                    k_grad = self.pc.kirchhoff_guidance(s_hat_b)  # (E, 4) or None
                    if k_grad is not None:
                        g_s = g_s - self.lambda_kirchhoff * k_grad

                # Lift guidance from (E, 4) to (E, 64)
                g_z = g_s.float() @ lift  # (E, 64)
                guidance[b, 0] = g_z.to(dtype)

            # Euler step: update x (only the predicted token)
            x = x + (v + guidance) * dt

            # --- Hard constraints ---
            # Decode, apply clamp_and_conserve, lift corrections back
            for b in range(B):
                s_hat_b = self.inv_proj(x[b, 0].float())  # (E, 4)
                s_corrected = self.pc.clamp_and_conserve(s_hat_b)  # (E, 4)
                delta_s = s_corrected - s_hat_b  # (E, 4)
                delta_z = delta_s.float() @ lift  # (E, 64)
                x[b, 0] = x[b, 0] + delta_z.to(dtype)

        return x  # (B, 1, E, 64)
