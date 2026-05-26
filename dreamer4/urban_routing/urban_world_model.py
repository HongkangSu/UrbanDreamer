"""
Urban World Model Inference Interface

Provides a high-level API for using the trained Dreamer4 world model
for urban routing tasks. Handles encoding observations, predicting next states,
and decoding predictions back to images.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import Dynamics, ActionDepartGNNEncoder, ActionEncoder, EdgeGNNActionEncoder

try:
    from .rollout_buffer import RolloutBuffer
except ImportError:
    from rollout_buffer import RolloutBuffer

try:
    from .flow_projector import FlowProjector, EdgeRegistry, PathTopology
except ImportError:
    from flow_projector import FlowProjector, EdgeRegistry, PathTopology

try:
    from .pcfm import PhysicsConstraints, PCFMSampler
except ImportError:
    from pcfm import PhysicsConstraints, PCFMSampler

try:
    from .stackelberg import BPRLayer, StackelbergLayer, decode_bg_volume
except ImportError:
    from stackelberg import BPRLayer, StackelbergLayer, decode_bg_volume


class UrbanWorldModel(nn.Module):
    @staticmethod
    def _split_action_dims(
        *,
        total_action_dim: int,
        action_compressor: Optional[Dict[str, Any]] = None,
        depart_dim_hint: int = 0,
        context_dim_hint: int = 0,
    ) -> Tuple[int, int, int]:
        act_dim = int(action_compressor.get("output_dim", 0)) if action_compressor is not None else int(total_action_dim)
        dep_dim = int(depart_dim_hint)
        ctx_dim = int(context_dim_hint)
        if act_dim + dep_dim + ctx_dim != int(total_action_dim):
            if action_compressor is None:
                act_dim = int(total_action_dim) - dep_dim - ctx_dim
            else:
                raise ValueError(
                    "Cannot split world model action dimensions: "
                    f"total={total_action_dim}, action={act_dim}, depart={dep_dim}, context={ctx_dim}"
                )
        return act_dim, dep_dim, ctx_dim

    """
    Flow-based Urban World Model — replaces image tokenizer with FlowProjector.

    State space: (E, d_feat=8) edge flow matrix per timestep.
    Dynamics:    CFM on projected flow tokens (E, d_spatial=64).
    Policy obs:  configurable source:
                 - obs_proj(z) for legacy latent observations
                 - decode(z) for semantic edge dynamics
                 - decoded volume / BPR travel time for compact routing obs
    """

    def __init__(
        self,
        flow_projector_ckpt: str,
        dynamics_ckpt:       str,
        edge_registry_path:  str,
        config:              Optional[Dict] = None,
        device:              str = "cuda",
        use_amp:             bool = True,
        pcfm_cfg:            Optional[Dict] = None,
        stackelberg_cfg:     Optional[Dict] = None,
        route_registry:      Optional["PathTopology"] = None,
        cfm_init_mode:       str = "randn",
        transition_mode:     str = "cfm",
        policy_obs_source:   str = "obs_proj",
    ):
        super().__init__()
        self.device  = torch.device(device)
        self.use_amp = use_amp and torch.cuda.is_available()
        self.set_policy_obs_source(policy_obs_source)
        self.cfm_init_mode = str(cfm_init_mode).strip().lower()
        if self.cfm_init_mode not in ("randn", "zero"):
            raise ValueError(f"Unsupported cfm_init_mode={cfm_init_mode!r}; expected 'randn' or 'zero'")
        self.transition_mode = str(transition_mode).strip().lower()
        if self.transition_mode != "cfm":
            raise ValueError(f"Unsupported transition_mode={transition_mode!r}; expected 'cfm'")

        # Load edge registry
        self._registry = EdgeRegistry.load(edge_registry_path)
        self.n_edges   = self._registry.n_edges

        # Load FlowProjector
        self.flow_projector = FlowProjector.load(flow_projector_ckpt, device=device)
        self.flow_projector.eval()

        # Set adjacency for GNN layers
        if not hasattr(self.flow_projector, 'adj') or self.flow_projector.adj is None:
            if self._registry.adjacency is not None:
                self.flow_projector.set_adjacency(
                    torch.from_numpy(self._registry.adjacency).to(self.device)
                )

        d_spatial  = self.flow_projector.d_spatial   # 64
        n_dynamic  = self.flow_projector.n_dynamic   # 4

        # Dynamics model dimensions
        self.n_spatial = self.n_edges     # each edge = one spatial token
        self.d_spatial = d_spatial
        self.n_dynamic = n_dynamic

        # Load dynamics
        dyn_state = torch.load(dynamics_ckpt, map_location=self.device, weights_only=False)
        dyn_args  = dyn_state.get("args", {})
        if hasattr(dyn_args, "__dict__"):
            dyn_args = vars(dyn_args)

        self.k_max = int(dyn_args.get("k_max", 8))
        self.use_cfm = bool(dyn_args.get("use_cfm", True))
        if not self.use_cfm:
            raise ValueError("UrbanWorldModel now supports CFM-only dynamics checkpoints (use_cfm=True).")
        self.n_ode_steps = max(1, int(dyn_args.get("n_ode_steps", 1)))
        self.hdv_policy_dim = int(dyn_args.get("hdv_policy_dim", 0))
        self.hdv_policy_mode = str(dyn_args.get("hdv_policy_mode", "edge")).lower()
        self.use_hdv_context = str(
            dyn_args.get("transition_hdv_context_mode", "none") or "none"
        ).strip().lower() not in {"", "none", "off", "disabled"}
        if self.hdv_policy_mode != "edge":
            raise ValueError("UrbanWorldModel now requires edge-level HDV head (hdv_policy_mode='edge').")
        # Infer action dim robustly:
        # 1) explicit args.action_dim (new checkpoints)
        # 2) args.n_actions (if present)
        # 3) args.n_routes (legacy checkpoints)
        # 4) fallback: inspect dynamics state_dict action encoder input dim
        action_dim = int(dyn_args.get("action_dim", dyn_args.get("n_actions", 0)))
        dyn_sd_probe = dyn_state.get("dynamics", dyn_state.get("dyn_model", dyn_state))
        if action_dim <= 0:
            action_dim = int(dyn_args.get("n_routes", 0))
        if action_dim <= 0 and isinstance(dyn_sd_probe, dict):
            for k, v in dyn_sd_probe.items():
                if str(k).endswith("action_encoder.fc1.weight") and hasattr(v, "shape"):
                    if len(v.shape) == 2 and int(v.shape[1]) > 0:
                        action_dim = int(v.shape[1])
                        break
        if action_dim <= 0:
            raise ValueError(
                "Cannot infer dynamics action_dim from checkpoint args/state_dict. "
                "Expected args.action_dim or action_encoder.fc1.weight in state_dict."
            )
        self.action_dim = int(action_dim)
        self.action_encoder_type = str(dyn_args.get("action_encoder_type", "mlp")).lower()
        self.depart_dim = int(dyn_args.get("depart_pca_dim_effective", 0) or 0)
        self.context_dim = 0
        self.raw_action_dim = int(self.action_dim)
        if self.depart_dim <= 0 and self.action_dim > self.n_edges:
            rem = int(self.action_dim) - int(self.n_edges)
            if rem >= self.n_edges:
                self.depart_dim = int(self.n_edges)
                self.context_dim = max(0, rem - int(self.n_edges))
                self.raw_action_dim = int(self.n_edges)

        # Optional action compressor (e.g., linear PCA) saved by train_dynamics.
        self.action_compressor: Optional[Dict[str, Any]] = None
        self.action_compressor_input_dim: Optional[int] = None
        self.action_compressor_output_dim: Optional[int] = None
        comp = dyn_state.get("action_compressor")
        if isinstance(comp, dict):
            ctype = str(comp.get("type", ""))
            if ctype == "pca_linear":
                mean = torch.as_tensor(comp.get("mean"), dtype=torch.float32, device=self.device)
                proj = torch.as_tensor(comp.get("proj"), dtype=torch.float32, device=self.device)
                in_dim = int(comp.get("input_dim", mean.numel()))
                out_dim = int(comp.get("output_dim", proj.shape[1] if proj.ndim == 2 else -1))
                if proj.ndim != 2 or proj.shape[0] != in_dim or proj.shape[1] != out_dim:
                    raise ValueError(
                        "Invalid action_compressor in dynamics checkpoint: "
                        f"proj_shape={tuple(proj.shape)}, input_dim={in_dim}, output_dim={out_dim}"
                    )
                self.action_compressor = {
                    "type": "pca_linear",
                    "mean": mean,
                    "proj": proj,
                    "input_dim": in_dim,
                    "output_dim": out_dim,
                    "explained_variance_ratio": float(comp.get("explained_variance_ratio", 0.0)),
                }
                self.action_compressor_input_dim = in_dim
                self.action_compressor_output_dim = out_dim
                self.raw_action_dim = out_dim
                remaining_dim = int(self.action_dim) - int(out_dim) - int(self.depart_dim)
                self.context_dim = max(0, remaining_dim)
                _, self.depart_dim, self.context_dim = self._split_action_dims(
                    total_action_dim=self.action_dim,
                    action_compressor=self.action_compressor,
                    depart_dim_hint=self.depart_dim,
                    context_dim_hint=self.context_dim,
                )
        assert int(dyn_args.get("packing_factor", 1)) == 1, (
            f"UrbanWorldModel requires packing_factor=1, got {dyn_args.get('packing_factor')}. "
            "Ensure the dynamics checkpoint was trained with packing_factor=1."
        )

        if self.action_encoder_type == "mlp":
            action_encoder = ActionEncoder(
                d_model=int(dyn_args.get("d_model_dyn", 256)),
                action_dim=self.action_dim,
            )
        elif self.action_encoder_type == "edge_gnn":
            if self.action_dim != self.n_edges:
                raise ValueError(
                    "edge_gnn checkpoint requires action_dim == n_edges, "
                    f"got action_dim={self.action_dim}, n_edges={self.n_edges}"
                )
            action_encoder = EdgeGNNActionEncoder(
                d_model=int(dyn_args.get("d_model_dyn", 256)),
                n_edges=self.n_edges,
                n_static=int(self._registry.static_features.shape[1])
                if self._registry.static_features is not None else 0,
                d_edge=int(dyn_args.get("action_gnn_d_edge", 64)),
                n_gnn_layers=int(dyn_args.get("action_gnn_layers", 2)),
            )
            if self._registry.static_features is not None:
                action_encoder.set_static_features(
                    torch.from_numpy(self._registry.static_features).to(torch.float32)
                )
            if self._registry.adjacency is not None:
                action_encoder.set_adjacency(
                    torch.from_numpy(self._registry.adjacency).to(torch.float32)
                )
        elif self.action_encoder_type == "graph_cond":
            graph_action_dim = int(self.n_edges)
            if self.raw_action_dim not in (0, graph_action_dim):
                raise ValueError(
                    "graph_cond checkpoint requires raw action dim == n_edges, "
                    f"got raw_action_dim={self.raw_action_dim}, n_edges={self.n_edges}"
                )
            self.raw_action_dim = graph_action_dim
            if self.depart_dim <= 0:
                self.depart_dim = graph_action_dim
            if self.context_dim <= 0:
                self.context_dim = 3
            expected_action_dim = self.raw_action_dim + self.depart_dim + self.context_dim
            if self.action_dim != expected_action_dim:
                raise ValueError(
                    "graph_cond checkpoint action layout mismatch: "
                    f"expected {expected_action_dim} (= action {self.raw_action_dim} + depart {self.depart_dim} + "
                    f"context {self.context_dim}), got {self.action_dim}"
                )
            action_encoder = ActionDepartGNNEncoder(
                d_model=int(dyn_args.get("d_model_dyn", 256)),
                n_edges=self.n_edges,
                n_static=int(self._registry.static_features.shape[1])
                if self._registry.static_features is not None else 0,
                d_edge=int(dyn_args.get("cond_gnn_hidden_dim", 64)),
                n_gnn_layers=int(dyn_args.get("cond_gnn_layers", 2)),
                context_dim=int(self.context_dim if self.context_dim > 0 else 3),
                context_hidden=int(dyn_args.get("cond_context_hidden_dim", 64)),
                use_hdv_context=bool(self.use_hdv_context),
            )
            if self._registry.static_features is not None:
                action_encoder.set_static_features(
                    torch.from_numpy(self._registry.static_features).to(torch.float32)
                )
            if self._registry.adjacency is not None:
                action_encoder.set_adjacency(
                    torch.from_numpy(self._registry.adjacency).to(torch.float32)
                )
        else:
            raise ValueError(
                f"Unknown action_encoder_type in checkpoint: {self.action_encoder_type!r}"
            )
        self.dynamics = Dynamics(
            d_model      = int(dyn_args.get("d_model_dyn", 256)),
            d_bottleneck = d_spatial,        # packing_factor=1 in flow mode
            d_spatial    = d_spatial,
            n_spatial    = self.n_spatial,
            n_register   = int(dyn_args.get("n_register", 4)),
            n_agent      = int(dyn_args.get("n_agent", 1)),
            n_heads      = int(dyn_args.get("n_heads", 4)),
            depth        = int(dyn_args.get("dyn_depth", 6)),
            k_max        = self.k_max,
            dropout      = 0.0,
            mlp_ratio    = float(dyn_args.get("mlp_ratio", 4.0)),
            time_every   = int(dyn_args.get("time_every", 1)),
            space_mode   = str(dyn_args.get("space_mode", "wm_agent_isolated")),
            action_encoder = action_encoder,
            use_cfm      = self.use_cfm,
            hdv_policy_dim = self.hdv_policy_dim,
            hdv_policy_mode = self.hdv_policy_mode,
            edge_action_injection = bool(dyn_args.get("edge_action_injection", False)),
            edge_action_injection_eps = float(dyn_args.get("edge_action_injection_eps", 1e-6)),
            edge_action_injection_scale = float(dyn_args.get("edge_action_injection_scale", 1.0)),
            edge_action_value_norm = str(dyn_args.get("edge_action_value_norm", "raw")),
            edge_action_id_embedding = bool(dyn_args.get("edge_action_id_embedding", False)),
            edge_action_tokenizer = bool(dyn_args.get("edge_action_tokenizer", False)),
            edge_action_tokenizer_fuse_encoder = bool(dyn_args.get("edge_action_tokenizer_fuse_encoder", False)),
            edge_action_output_injection = bool(dyn_args.get("edge_action_output_injection", False)),
            edge_action_output_from_injection = bool(dyn_args.get("edge_action_output_from_injection", False)),
            action_reconstruction = bool(float(dyn_args.get("action_reconstruction_weight", 0.0)) > 0.0),
        ).to(self.device).eval()

        dyn_sd = dyn_state.get("dynamics", dyn_state.get("dyn_model", dyn_state))
        if not isinstance(dyn_sd, dict):
            raise ValueError(
                f"Cannot find dynamics state_dict in checkpoint keys: {list(dyn_state.keys())}"
            )
        dyn_sd = {k.replace("module.", "", 1): v for k, v in dyn_sd.items()}
        self.dynamics.load_state_dict(dyn_sd, strict=True)

        for p in self.parameters():
            p.requires_grad = False

        self.config = config or {}
        self.schedule = "cfm"
        self.schedule_d = None
        self.sched = None

        # --- PCFM sampler (optional) ---
        self.pcfm_sampler: Optional[PCFMSampler] = None
        if pcfm_cfg is not None and pcfm_cfg.get("enabled", False):
            inv_proj_weight = self.flow_projector.inv_proj.weight.detach()  # (4, 64)
            capacities_t = torch.from_numpy(
                np.asarray(self._registry.capacities, dtype=np.float32)
            ).to(self.device)
            node_incidence = None
            if hasattr(self._registry, "build_node_incidence"):
                try:
                    ni_np = self._registry.build_node_incidence()
                    if ni_np is not None:
                        node_incidence = torch.from_numpy(ni_np.astype(np.float32)).to(self.device)
                except Exception:
                    pass
            N_total = float(pcfm_cfg.get("N_total", float(capacities_t.sum().item()) * 0.3))
            pc = PhysicsConstraints(
                inv_proj_weight=inv_proj_weight,
                capacities=capacities_t,
                N_total=N_total,
                node_incidence=node_incidence,
                bpr_alpha=float(pcfm_cfg.get("bpr_alpha", 0.15)),
                bpr_beta=float(pcfm_cfg.get("bpr_beta", 4.0)),
                kappa=float(pcfm_cfg.get("kappa", 1.2)),
            ).to(self.device)
            self.pcfm_sampler = PCFMSampler(
                dynamics=self.dynamics,
                physics_constraints=pc,
                inv_proj=self.flow_projector.inv_proj,
                n_ode_steps=int(pcfm_cfg.get("n_ode_steps", 4)),
                lambda_bpr=float(pcfm_cfg.get("lambda_bpr", 0.05)),
                lambda_kirchhoff=float(pcfm_cfg.get("lambda_kirchhoff", 0.03)),
            )

        # --- BPR layer (shared by Stackelberg + reward) ---
        lengths_t = torch.from_numpy(
            np.asarray(self._registry.lengths, dtype=np.float32)
        ).to(self.device)
        speed_limits_t = torch.from_numpy(
            np.asarray(self._registry.speed_limits, dtype=np.float32)
        ).to(self.device)
        capacities_t = torch.from_numpy(
            np.asarray(self._registry.capacities, dtype=np.float32)
        ).to(self.device)
        self.bpr = BPRLayer(lengths_t, speed_limits_t, capacities_t).to(self.device)

        # --- Stackelberg equilibrium layer (optional) ---
        self.stackelberg: Optional[StackelbergLayer] = None
        self._route_registry = route_registry
        if stackelberg_cfg is not None and stackelberg_cfg.get("enabled", False):
            total_av_count = float(stackelberg_cfg.get("total_av_count", 0.0))
            total_hdv_count = float(stackelberg_cfg.get("total_hdv_count", 0.0))
            if route_registry is not None:
                if total_av_count <= 0.0:
                    total_av_count = float(np.asarray(route_registry.od_av_counts, dtype=np.float32).sum())
                if total_hdv_count <= 0.0:
                    total_hdv_count = float(np.asarray(route_registry.od_hdv_counts, dtype=np.float32).sum())
            if total_av_count <= 0.0 and total_hdv_count <= 0.0:
                raise ValueError("Stackelberg requires total AV/HDV vehicle counts.")
            self.stackelberg = StackelbergLayer(
                n_edges=self.n_edges,
                bpr=self.bpr,
                total_av_count=total_av_count,
                total_hdv_count=total_hdv_count,
                n_iters=int(stackelberg_cfg.get("n_iters", 10)),
                damping=float(stackelberg_cfg.get("damping", 0.5)),
            ).to(self.device)

        print(
            f"UrbanWorldModel (CFM): n_edges={self.n_edges}, d_spatial={d_spatial}, "
            f"action_dim={self.action_dim}, k_max={self.k_max}, "
            f"n_ode_steps={self.n_ode_steps}, hdv_policy_dim={self.hdv_policy_dim}, "
            f"hdv_policy_mode={self.hdv_policy_mode}, action_encoder={self.action_encoder_type}, "
            f"hdv_context={self.use_hdv_context}, "
            f"transition_mode={self.transition_mode}, "
            f"policy_obs_source={self.policy_obs_source}, "
            f"pcfm={self.pcfm_sampler is not None}, stackelberg={self.stackelberg is not None}"
        )
        if self.action_compressor is not None:
            print(
                "UrbanWorldModel action compressor: "
                f"type={self.action_compressor.get('type', 'unknown')}, "
                f"input_dim={self.action_compressor_input_dim}, "
                f"output_dim={self.action_compressor_output_dim}, "
                f"explained_var={float(self.action_compressor.get('explained_variance_ratio', 0.0)):.6f}"
            )

    @torch.inference_mode()
    def encode_observation(self, state) -> torch.Tensor:
        """
        Encode SUMO step state to projected flow tokens for dynamics.

        Args:
            state: either a dict with key "edges" (RouteRLConnector / SUMOConnector state)
                   or a raw (E, 8) float32 numpy array of pre-extracted flow features

        Returns:
            Projected tokens of shape (n_spatial=E, d_spatial=64)
        """
        if isinstance(state, np.ndarray):
            flow_np = state                                             # (E, 8) already
        else:
            flow_np = self._registry.extract_flow(state.get("edges", {}))  # (E, 8)
        flow_t  = torch.from_numpy(flow_np).float().to(self.device)    # (E, 8)
        proj    = self.flow_projector(flow_t)                           # (E, 64)
        return proj   # (E, 64) — stored in RolloutBuffer

    @property
    def d_obs(self) -> int:
        """Observation feature dimension per edge (d_obs or n_dynamic fallback)."""
        source = getattr(self, "policy_obs_source", "obs_proj")
        if source == "decoder":
            return int(self.flow_projector.n_dynamic)
        if source == "decoder_volume":
            return 1
        if source == "decoder_volume_tt":
            return 2
        if source == "decoder_volume_tt_route_head":
            return 3
        return self.flow_projector.d_obs

    def set_policy_obs_source(self, source: str) -> None:
        source_norm = str(source or "obs_proj").strip().lower()
        aliases = {
            "project_obs": "obs_proj",
            "obs": "obs_proj",
            "latent": "obs_proj",
            "decode": "decoder",
            "decoded": "decoder",
            "volume": "decoder_volume",
            "decoder_vol": "decoder_volume",
            "volume_tt": "decoder_volume_tt",
            "decoder_bpr": "decoder_volume_tt",
            "decoder_route_head": "decoder_volume_tt_route_head",
            "route_head": "decoder_volume_tt_route_head",
        }
        source_norm = aliases.get(source_norm, source_norm)
        if source_norm not in ("obs_proj", "decoder", "decoder_volume", "decoder_volume_tt", "decoder_volume_tt_route_head"):
            raise ValueError(
                f"Unsupported policy_obs_source={source!r}; expected "
                "'obs_proj', 'decoder', 'decoder_volume', 'decoder_volume_tt', "
                "or 'decoder_volume_tt_route_head'"
            )
        self.policy_obs_source = source_norm

    def latent_edge_features(self, z: torch.Tensor) -> torch.Tensor:
        """z: (E, d_spatial) → (E, d_obs) for policy observations."""
        source = getattr(self, "policy_obs_source", "obs_proj")
        if source.startswith("decoder"):
            decoded = self.flow_projector.decode(z)
            if source == "decoder":
                return decoded
            volume_norm = torch.relu(decoded[..., 2:3])
            if source == "decoder_volume":
                return volume_norm
            v_e = volume_norm[..., 0] * self.bpr.capacities.to(device=z.device, dtype=decoded.dtype)
            tt_min = self.bpr(v_e).unsqueeze(-1) / 60.0
            if source == "decoder_volume_tt_route_head":
                lengths = torch.as_tensor(
                    self._registry.lengths,
                    device=z.device,
                    dtype=decoded.dtype,
                )
                speed_limits = torch.as_tensor(
                    self._registry.speed_limits,
                    device=z.device,
                    dtype=decoded.dtype,
                )
                free_flow = lengths / speed_limits.clamp_min(0.1)
                route_head_tt_min = self.flow_projector.predict_edge_route_cost(z, free_flow).unsqueeze(-1) / 60.0
                return torch.cat(
                    [volume_norm, tt_min.to(dtype=decoded.dtype), route_head_tt_min.to(dtype=decoded.dtype)],
                    dim=-1,
                )
            return torch.cat([volume_norm, tt_min.to(dtype=decoded.dtype)], dim=-1)
        return self.flow_projector.project_obs(z)

    def policy_obs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert projected tokens to policy observation vector.

        Args:
            z: (E, 64) projected flow tokens (output of encode_observation)

        Returns:
            (E * d_obs,) float tensor for GaussianRoutePolicy
        """
        dyn = self.latent_edge_features(z)     # (E, d_obs)
        return dyn.flatten()                   # (E*d_obs,)

    def local_policy_obs(
        self,
        z: torch.Tensor,
        route_masks: torch.Tensor,
        prev_av_fracs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hdv_fracs: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute per-OD-pair local observations for mean field MARL.

        Args:
            z: (E, d_spatial=64) projected flow tokens
            route_masks: (N_od, K, E) route mask tensor
            prev_av_fracs: optional (N_od, K) previous-step AV route fractions
            hdv_fracs: optional (N_od, K) current HDV route fractions

        Returns:
            Base observation:
              (N_od, K*n_dynamic + n_dynamic) = (N_od, K*4+4)
            If intent tensors are provided, append them in order:
              [..., prev_av_fracs, hdv_fracs] -> (N_od, K*4+4+K+K)
        """
        edge_feats = self.latent_edge_features(z)                          # (E, d_obs)
        global_summary = edge_feats.mean(dim=0)                           # (d_obs,)
        local = torch.einsum('oke,ed->okd', route_masks, edge_feats)      # (N_od, K, d_obs)
        local_flat = local.reshape(route_masks.shape[0], -1)              # (N_od, K*d_obs)
        global_exp = global_summary.unsqueeze(0).expand(local_flat.shape[0], -1)  # (N_od, 4)
        obs_parts = [local_flat, global_exp]

        if prev_av_fracs is not None:
            prev_t = torch.as_tensor(
                prev_av_fracs, device=edge_feats.device, dtype=edge_feats.dtype
            )
            if prev_t.dim() == 1:
                prev_t = prev_t.unsqueeze(0).expand(route_masks.shape[0], -1)
            if prev_t.shape[0] != route_masks.shape[0]:
                raise ValueError(
                    f"prev_av_fracs batch mismatch: expected {route_masks.shape[0]}, got {prev_t.shape[0]}"
                )
            obs_parts.append(prev_t)

        if hdv_fracs is not None:
            hdv_t = torch.as_tensor(
                hdv_fracs, device=edge_feats.device, dtype=edge_feats.dtype
            )
            if hdv_t.dim() == 1:
                hdv_t = hdv_t.unsqueeze(0).expand(route_masks.shape[0], -1)
            if hdv_t.shape[0] != route_masks.shape[0]:
                raise ValueError(
                    f"hdv_fracs batch mismatch: expected {route_masks.shape[0]}, got {hdv_t.shape[0]}"
                )
            obs_parts.append(hdv_t)

        return torch.cat(obs_parts, dim=-1)

    def _compress_action_if_needed(self, action: torch.Tensor) -> torch.Tensor:
        """
        Map raw action vector to dynamics action_dim when a compressor is attached.
        """
        in_dim = int(action.shape[-1])
        if in_dim == int(getattr(self, "raw_action_dim", self.action_dim)):
            return action

        comp = self.action_compressor
        if comp is None:
            raise ValueError(
                f"Action dimension mismatch: world model expects {self.action_dim}, got {in_dim}. "
                "No action compressor found in dynamics checkpoint."
            )

        ctype = str(comp.get("type", ""))
        if ctype != "pca_linear":
            raise ValueError(f"Unsupported action compressor type: {ctype}")

        c_in = int(comp.get("input_dim", -1))
        c_out = int(comp.get("output_dim", -1))
        if in_dim != c_in:
            raise ValueError(
                f"Action compressor input mismatch: expected {c_in}, got {in_dim}"
            )
        if c_out != int(self.action_compressor_output_dim or c_out):
            raise ValueError(
                f"Action compressor output mismatch: expected {self.action_compressor_output_dim}, got {c_out}"
            )

        mean = comp["mean"].to(device=action.device, dtype=action.dtype).view(1, -1)
        proj = comp["proj"].to(device=action.device, dtype=action.dtype)  # (A_in, A_out)
        flat = action.reshape(-1, in_dim)
        out = (flat - mean) @ proj
        return out.view(*action.shape[:-1], c_out)

    def _build_graph_cond_action_tuple(
        self,
        *,
        batch_size: int,
        seq_len: int,
        action: torch.Tensor,
        depart: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        hdv_context: Optional[torch.Tensor] = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, ...]:
        def _ensure_seq(name: str, tensor: torch.Tensor, expected_last_dim: int) -> torch.Tensor:
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() == 2:
                if int(tensor.shape[0]) not in (1, batch_size):
                    raise ValueError(
                        f"{name} batch mismatch: expected batch size {batch_size}, got {int(tensor.shape[0])}"
                    )
                if int(tensor.shape[-1]) != expected_last_dim:
                    raise ValueError(
                        f"{name} feature dim mismatch: expected {expected_last_dim}, got {int(tensor.shape[-1])}"
                    )
                if int(tensor.shape[0]) == 1 and batch_size > 1:
                    tensor = tensor.expand(batch_size, -1)
                tensor = tensor.unsqueeze(1)
                if seq_len > 1:
                    padded = torch.zeros(
                        (batch_size, seq_len, expected_last_dim),
                        device=tensor.device,
                        dtype=tensor.dtype,
                    )
                    padded[:, -1, :] = tensor[:, 0, :]
                    tensor = padded
            elif tensor.dim() == 3:
                if int(tensor.shape[0]) not in (1, batch_size):
                    raise ValueError(
                        f"{name} batch mismatch: expected batch size {batch_size}, got {int(tensor.shape[0])}"
                    )
                if int(tensor.shape[1]) not in (1, seq_len):
                    raise ValueError(
                        f"{name} sequence mismatch: expected seq len {seq_len}, got {int(tensor.shape[1])}"
                    )
                if int(tensor.shape[-1]) != expected_last_dim:
                    raise ValueError(
                        f"{name} feature dim mismatch: expected {expected_last_dim}, got {int(tensor.shape[-1])}"
                    )
                if int(tensor.shape[0]) == 1 and batch_size > 1:
                    tensor = tensor.expand(batch_size, -1, -1)
                if int(tensor.shape[1]) == 1 and seq_len > 1:
                    tensor = tensor.expand(-1, seq_len, -1)
            else:
                raise ValueError(f"{name} must be 1D, 2D, or 3D; got shape {tuple(tensor.shape)}")
            return tensor.to(device=device, dtype=dtype)

        action_dim = int(getattr(self, "raw_action_dim", action.shape[-1]))
        depart_dim = int(getattr(self, "depart_dim", depart.shape[-1] if depart is not None else action_dim))
        context_dim = int(getattr(self, "context_dim", context.shape[-1] if context is not None else 3))

        action_seq = _ensure_seq("action", action, action_dim)
        if depart is None:
            depart_seq = torch.zeros(
                (batch_size, seq_len, depart_dim),
                device=device,
                dtype=dtype,
            )
        else:
            depart_seq = _ensure_seq("depart", depart, depart_dim)

        if context is None:
            context_seq = torch.zeros(
                (batch_size, seq_len, context_dim),
                device=device,
                dtype=dtype,
            )
        else:
            context_seq = _ensure_seq("context", context, context_dim)

        if self.use_hdv_context:
            if hdv_context is None:
                hdv_seq = torch.zeros(
                    (batch_size, seq_len, action_dim),
                    device=device,
                    dtype=dtype,
                )
            else:
                hdv_seq = _ensure_seq("hdv_context", hdv_context, action_dim)
            return action_seq, depart_seq, context_seq, hdv_seq
        return action_seq, depart_seq, context_seq

    def _prepare_model_actions(
        self,
        *,
        batch_size: int,
        seq_len: int,
        action: torch.Tensor,
        depart: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        hdv_context: Optional[torch.Tensor] = None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.action_encoder_type == "graph_cond":
            return self._build_graph_cond_action_tuple(
                batch_size=batch_size,
                seq_len=seq_len,
                action=action,
                depart=depart,
                context=context,
                hdv_context=hdv_context,
                device=device,
                dtype=dtype,
            )
        if self.action_encoder_type == "edge_gnn":
            depart = None
            context = None
        action_prepared = self._prepare_action_tensor(action, device=device, dtype=dtype)
        depart_prepared = None
        if depart is not None:
            depart_prepared = self._prepare_action_tensor(depart, device=device, dtype=dtype)
        context_prepared = None
        if context is not None:
            context_prepared = context.to(device=device, dtype=dtype) if isinstance(context, torch.Tensor) else torch.as_tensor(context, device=device, dtype=dtype)
            if context_prepared.dim() == 1:
                context_prepared = context_prepared.unsqueeze(0)
        full = self._build_full_action_vector(action=action_prepared, depart=depart_prepared, context=context_prepared)
        if full.dim() == 1:
            full = full.unsqueeze(0)
        if full.dim() == 2:
            if int(full.shape[0]) not in (1, batch_size):
                raise ValueError(
                    f"action batch mismatch: expected batch size {batch_size}, got {int(full.shape[0])}"
                )
            if int(full.shape[0]) == 1 and batch_size > 1:
                full = full.expand(batch_size, -1)
            full = full.unsqueeze(1)
            if seq_len > 1:
                full = full.expand(-1, seq_len, -1)
        elif full.dim() == 3:
            if int(full.shape[0]) not in (1, batch_size):
                raise ValueError(
                    f"action batch mismatch: expected batch size {batch_size}, got {int(full.shape[0])}"
                )
            if int(full.shape[1]) not in (1, seq_len):
                raise ValueError(
                    f"action sequence mismatch: expected seq len {seq_len}, got {int(full.shape[1])}"
                )
            if int(full.shape[0]) == 1 and batch_size > 1:
                full = full.expand(batch_size, -1, -1)
            if int(full.shape[1]) == 1 and seq_len > 1:
                full = full.expand(-1, seq_len, -1)
        else:
            raise ValueError(f"full action tensor must be 1D, 2D, or 3D; got shape {tuple(full.shape)}")
        return full.to(device=device, dtype=dtype)

    def _build_full_action_vector(
        self,
        *,
        action: torch.Tensor,
        depart: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        parts = [action]
        if depart is not None:
            parts.append(depart)
        if context is not None:
            parts.append(context)
        full = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        if int(full.shape[-1]) != int(self.action_dim):
            raise ValueError(
                f"Full action dimension mismatch: expected {self.action_dim}, got {int(full.shape[-1])}"
            )
        return full

    def _prepare_action_tensor(
        self,
        action: Union[np.ndarray, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action)
        else:
            action_t = action
        action_t = action_t.to(device=device)
        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0)
        action_t = action_t.to(dtype=dtype)
        action_t = self._compress_action_if_needed(action_t)
        return action_t

    @torch.inference_mode()
    def predict_next_state(
        self,
        history_buffer: RolloutBuffer,
        action: Union[np.ndarray, torch.Tensor],
        depart: Optional[Union[np.ndarray, torch.Tensor]] = None,
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
        hdv_context: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_info: bool = False,
        skip_stackelberg: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Predict next latent state using CFM sampling.

        When Stackelberg is enabled, the AV edge action is combined with
        the HDV equilibrium response to form a joint action for dynamics.

        Args:
            depart: optional departure schedule vector to concatenate to
                the action before dynamics forward. Shape (E,) or (1, E).
            context: optional context vector (e.g. time-of-day, demand
                level) to concatenate to the action. Shape (C,) or (1, C).
            return_info: if True, also return info dict with Stackelberg
                outputs (hdv_edge, eq_residual, edge_tt).
            skip_stackelberg: if True, treat *action* as already-combined
                joint action and bypass Stackelberg.  Use when HDV
                equilibrium was pre-computed once per episode.
        """
        if history_buffer.is_empty():
            raise ValueError("History buffer is empty.")

        past_packed = history_buffer.get_past_packed(batch_size=1)   # (1, t, E, 64)
        B, t = past_packed.shape[:2]

        act_dtype = past_packed.dtype          # match latent dtype (float32/16)
        action = self._prepare_action_tensor(action, device=self.device, dtype=act_dtype)

        depart_t = None
        if depart is not None:
            if isinstance(depart, np.ndarray):
                depart_t = torch.from_numpy(depart).to(device=self.device, dtype=act_dtype)
            elif isinstance(depart, (list, tuple)):
                depart_t = torch.tensor(depart, device=self.device, dtype=act_dtype)
            else:
                depart_t = depart.to(device=self.device, dtype=act_dtype)
            if depart_t.dim() == 1:
                depart_t = depart_t.unsqueeze(0)

        context_t = None
        if context is not None:
            if isinstance(context, np.ndarray):
                context_t = torch.from_numpy(context).to(device=self.device, dtype=act_dtype)
            elif isinstance(context, (list, tuple)):
                context_t = torch.tensor(context, device=self.device, dtype=act_dtype)
            else:
                context_t = context.to(device=self.device, dtype=act_dtype)
            if context_t.dim() == 1:
                context_t = context_t.unsqueeze(0)

        hdv_context_t = None
        if hdv_context is not None:
            if isinstance(hdv_context, np.ndarray):
                hdv_context_t = torch.from_numpy(hdv_context).to(device=self.device, dtype=act_dtype)
            elif isinstance(hdv_context, (list, tuple)):
                hdv_context_t = torch.tensor(hdv_context, device=self.device, dtype=act_dtype)
            else:
                hdv_context_t = hdv_context.to(device=self.device, dtype=act_dtype)
            if hdv_context_t.dim() == 1:
                hdv_context_t = hdv_context_t.unsqueeze(0)

        # Keep raw graph_cond inputs separate; other encoders use flat concatenation.
        actions = self._prepare_model_actions(
            batch_size=B,
            seq_len=t + 1,
            action=action,
            depart=depart_t,
            context=context_t,
            hdv_context=hdv_context_t,
            device=self.device,
            dtype=act_dtype,
        )

        info: Dict[str, Any] = {}
        # Stackelberg: combine AV action with HDV equilibrium
        if not skip_stackelberg and self.stackelberg is not None:
            if return_info:
                action, stk_info = self._apply_stackelberg(
                    action, past_packed[:, -1], return_info=True,
                )
                info.update(stk_info)
                actions = self._prepare_model_actions(
                    batch_size=B,
                    seq_len=t + 1,
                    action=action,
                    depart=depart_t,
                    context=context_t,
                    hdv_context=hdv_context_t,
                    device=self.device,
                    dtype=act_dtype,
                )
            else:
                action = self._apply_stackelberg(action, past_packed[:, -1])
                actions = self._prepare_model_actions(
                    batch_size=B,
                    seq_len=t + 1,
                    action=action,
                    depart=depart_t,
                    context=context_t,
                    hdv_context=hdv_context_t,
                    device=self.device,
                    dtype=act_dtype,
                )

        z_next = self._predict_one_step(past_packed, actions)   # (B, 1, E, 64)
        z_out = z_next[0, 0]   # (E, 64)
        if return_info:
            return z_out, info
        return z_out

    @torch.inference_mode()
    def _predict_one_step(self, past_packed, actions):
        return self._cfm_sample_one_step(past_packed, actions)

    @torch.inference_mode()
    def _cfm_sample_one_step(self, past_packed, actions):
        """CFM ODE sampling for one timestep prediction.

        Dispatches to PCFMSampler when physics-constrained sampling is enabled.
        """
        if self.pcfm_sampler is not None:
            sample_actions = actions
            if isinstance(actions, tuple):
                sample_actions = tuple(a[:, : past_packed.shape[1] + 1] for a in actions)
            else:
                sample_actions = actions[:, : past_packed.shape[1] + 1]
            return self.pcfm_sampler.sample(
                past_packed, sample_actions,
            )

        device = past_packed.device
        dtype  = past_packed.dtype
        B, t   = past_packed.shape[:2]
        n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]

        x = self._init_cfm_state(B, n_spatial, d_spatial, device=device, dtype=dtype)
        dt = 1.0 / float(max(1, self.n_ode_steps))
        if isinstance(actions, tuple):
            actions_in = tuple(a[:, :t + 1] for a in actions)
        else:
            actions_in = actions[:, :t + 1]

        for i in range(max(1, self.n_ode_steps)):
            t_val = float(i) * dt
            flow_time = torch.zeros((B, t + 1), device=device, dtype=torch.float32)
            flow_time[:, -1] = t_val

            packed_seq = torch.cat([past_packed, x], dim=1)
            with torch.autocast(device_type=device.type, enabled=(self.use_amp and device.type == "cuda")):
                v_pred, _ = self.dynamics(
                    actions_in, None, None, packed_seq,
                    act_mask=None, agent_tokens=None,
                    flow_time=flow_time,
                )
            v = v_pred[:, -1:, :, :]
            x = (x.float() + v.float() * dt).to(dtype)

        return x

    @torch.inference_mode()
    def predict_next_state_batch(
        self,
        z_current: torch.Tensor,    # (E, D)
        actions: torch.Tensor,      # (N, E) batch of joint edge actions
        departs: Optional[torch.Tensor] = None,   # (N, E)
        contexts: Optional[torch.Tensor] = None,   # (N, C)
        hdv_contexts: Optional[torch.Tensor] = None,   # (N, E)
    ) -> torch.Tensor:
        """Predict next states for N candidate actions in parallel.

        Args:
            z_current: (E, D) current latent state.
            actions:   (N, E) batch of normalized edge-level joint actions.
            departs:   optional (N, E) departure schedule vectors.
            contexts:  optional (N, C) context vectors.

        Returns:
            (N, E, D) predicted next latent states.
        """
        N = actions.shape[0]
        E, D = z_current.shape
        dtype = z_current.dtype

        # Keep graph_cond inputs as a tuple; other encoders use flat action vectors.
        actions_full = self._prepare_model_actions(
            batch_size=N,
            seq_len=2,
            action=actions,
            depart=departs,
            context=contexts,
            hdv_context=hdv_contexts,
            device=self.device,
            dtype=dtype,
        )

        past_packed = z_current.unsqueeze(0).unsqueeze(0).expand(N, 1, E, D)  # (N, 1, E, D)

        z_next = self._predict_one_step(past_packed, actions_full)  # (N, 1, E, D)
        return z_next[:, 0]  # (N, E, D)

    def predict_next_state_differentiable(
        self,
        z_current: torch.Tensor,
        action: torch.Tensor,
        past_packed: Optional[torch.Tensor] = None,
        depart: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        hdv_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Differentiable one-step prediction for BPTT.

        Only supported in CFM mode; keeps gradients through dynamics forward.

        Returns:
            z_next: (E, D) predicted next-step latent tokens.
            info: dict with optional keys like "hdv_edge" (E,) when Stackelberg is active.
        """
        if past_packed is None:
            past_packed = z_current.unsqueeze(0).unsqueeze(0)  # (1,1,E,D)
        if past_packed.dim() != 4:
            raise ValueError(f"past_packed must be (B,t,E,D), got {tuple(past_packed.shape)}")

        B, t = past_packed.shape[:2]
        device = past_packed.device
        dtype = past_packed.dtype
        info: Dict[str, Any] = {}

        action = self._prepare_action_tensor(action, device=device, dtype=dtype)

        # Stackelberg: combine AV action with HDV equilibrium (differentiable)
        if self.stackelberg is not None:
            action, stk_info = self._apply_stackelberg(
                action, past_packed[:, -1], return_info=True,
            )
            info.update(stk_info)

        model_actions = self._prepare_model_actions(
            batch_size=B,
            seq_len=t + 1,
            action=action,
            depart=depart,
            context=context,
            hdv_context=hdv_context,
            device=device,
            dtype=dtype,
        )

        n_spatial, d_spatial = past_packed.shape[2], past_packed.shape[3]
        x = self._init_cfm_state(B, n_spatial, d_spatial, device=device, dtype=dtype)
        flow_time = torch.zeros((B, t + 1), device=device, dtype=torch.float32)

        packed_seq = torch.cat([past_packed, x], dim=1)

        def _forward_dynamics(
            actions_t: torch.Tensor,
            packed_t: torch.Tensor,
            flow_time_t: torch.Tensor,
        ) -> torch.Tensor:
            v_out, _ = self.dynamics(
                actions_t, None, None, packed_t,
                act_mask=None, agent_tokens=None,
                flow_time=flow_time_t,
            )
            return v_out

        checkpoint_enabled = str(os.environ.get("URB_WM_DIFF_CHECKPOINT", "1")).lower() not in {
            "0", "false", "no", "off"
        }
        can_checkpoint = (
            checkpoint_enabled
            and torch.is_grad_enabled()
            and isinstance(model_actions, torch.Tensor)
            and (model_actions.requires_grad or packed_seq.requires_grad)
        )
        if can_checkpoint:
            from torch.utils.checkpoint import checkpoint

            v_pred = checkpoint(
                _forward_dynamics,
                model_actions,
                packed_seq,
                flow_time,
                use_reentrant=False,
            )
        else:
            v_pred = _forward_dynamics(model_actions, packed_seq, flow_time)
        z_next = x + v_pred[:, -1:, :, :]
        return z_next[0, 0], info

    def _init_cfm_state(
        self,
        batch_size: int,
        n_spatial: int,
        d_spatial: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = (int(batch_size), 1, int(n_spatial), int(d_spatial))
        if self.cfm_init_mode == "zero":
            return torch.zeros(shape, device=device, dtype=dtype)
        return torch.randn(shape, device=device, dtype=dtype)

    def _apply_stackelberg(
        self,
        action: torch.Tensor,
        z_last: torch.Tensor,
        return_info: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Combine AV edge action with Stackelberg HDV equilibrium response.

        Args:
            action: (1, E) or (E,) AV edge distribution.
            z_last: (1, E, 64) or (E, 64) last latent state for bg_volume decoding.
            return_info: if True, also return info dict with edge-level equilibrium details.

        Returns:
            joint_action: (1, E) normalized AV+HDV edge distribution.
            info (optional): dict with "hdv_edge", "eq_residual", "edge_tt".
        """
        stk = self.stackelberg

        # AV edge distribution → edge vehicle counts (no OD decomposition)
        av_edge = action.view(-1)  # (E,) normalized
        n_av_total = float(stk.total_av_count.item())
        n_hdv_total = float(stk.total_hdv_count.item())
        av_edge_volume = av_edge * n_av_total  # (E,) vehicle counts

        # Decode background volume from last latent state
        z_2d = z_last.view(-1, z_last.shape[-1])  # (E, 64)
        bg_volume = decode_bg_volume(z_2d, self.flow_projector, stk.capacities)  # (E,)

        # Solve HDV equilibrium
        hdv_edge, edge_tt, eq_residual = stk(av_edge_volume, bg_volume)

        # Joint action: combine AV/HDV edge volumes using real vehicle counts.
        joint = av_edge_volume + hdv_edge * n_hdv_total
        joint = joint / joint.sum().clamp(min=1e-8)
        joint_action = joint.unsqueeze(0).to(dtype=action.dtype)  # (1, E)

        if return_info:
            info = {
                "hdv_edge": hdv_edge.detach(),     # (E,)
                "eq_residual": eq_residual,        # scalar (keep grad for L_eq)
                "edge_tt": edge_tt.detach(),       # (E,)
            }
            return joint_action, info
        return joint_action

    @torch.inference_mode()
    def compute_hdv_equilibrium(
        self,
        av_action: Union[np.ndarray, torch.Tensor],
        z_current: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Compute HDV edge distribution once per episode via Stackelberg.

        HDVs choose routes at episode start and stay fixed.  Call this once
        before the step loop, then combine AV / HDV edge distributions using
        their real vehicle counts before calling
        ``predict_next_state(..., skip_stackelberg=True)``.

        Returns:
            hdv_edge: (E,) normalized HDV edge distribution, or None if
            Stackelberg is not available.
        """
        if self.stackelberg is None:
            return None
        stk = self.stackelberg

        if isinstance(av_action, np.ndarray):
            av_action_t = torch.from_numpy(av_action)
        else:
            av_action_t = av_action
        av_action_t = av_action_t.to(device=self.device, dtype=torch.float32)
        av_edge = av_action_t.view(-1)  # (E,) normalized raw edge distribution
        if int(av_edge.numel()) != int(stk.n_edges):
            raise ValueError(
                f"compute_hdv_equilibrium expects raw edge action of length {stk.n_edges}, got {int(av_edge.numel())}"
            )
        n_av_total = float(stk.total_av_count.item())
        av_edge_volume = av_edge * n_av_total  # (E,) vehicle counts

        if z_current is not None:
            z_2d = z_current.view(-1, z_current.shape[-1]).to(device=self.device, dtype=torch.float32)
            bg_volume = decode_bg_volume(z_2d, self.flow_projector, stk.capacities)
        else:
            bg_volume = torch.zeros(stk.n_edges, device=self.device, dtype=torch.float32)
        hdv_edge, _edge_tt, _eq_residual = stk(av_edge_volume, bg_volume)
        return hdv_edge.detach()  # (E,) normalized

    @torch.inference_mode()
    def predict_hdv_policy(
        self,
        history_buffer: RolloutBuffer,
        action: Union[np.ndarray, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Predict HDV policy distribution for the next step from hdv_policy_head.

        Returns:
            probs: (D,) tensor on CPU if head exists, else None.
            D=K in route mode, D=E in edge mode.
        """
        if self.hdv_policy_dim <= 0:
            return None
        if history_buffer.is_empty():
            raise ValueError("History buffer is empty.")

        past_packed = history_buffer.get_past_packed(batch_size=1)   # (1, t, E, 64)
        B, t = past_packed.shape[:2]

        act_dtype = past_packed.dtype
        action = self._prepare_action_tensor(action, device=self.device, dtype=act_dtype)
        action_dim = action.shape[-1]
        actions = torch.zeros((B, t + 1, action_dim), device=self.device, dtype=act_dtype)
        actions[:, -1] = action

        x_query = torch.zeros((B, 1, self.n_spatial, self.d_spatial), device=self.device, dtype=act_dtype)
        packed_seq = torch.cat([past_packed, x_query], dim=1)

        flow_time = torch.zeros((B, t + 1), device=self.device, dtype=torch.float32)
        _, _, hdv_logits = self.dynamics(
            actions, None, None, packed_seq,
            act_mask=None, agent_tokens=None,
            flow_time=flow_time, return_hdv_policy=True,
        )

        if hdv_logits is None:
            return None
        probs = torch.softmax(hdv_logits[:, -1, :], dim=-1)
        return probs[0].detach().cpu()

    @torch.inference_mode()
    def predict_episode_hdv_response(
        self,
        episode_av_pressure: Union[np.ndarray, torch.Tensor],
        *,
        z_context: Optional[Union[np.ndarray, torch.Tensor]] = None,
        depart: Optional[Union[np.ndarray, torch.Tensor]] = None,
        context: Optional[Union[np.ndarray, torch.Tensor]] = None,
        normalize: bool = True,
    ) -> Optional[torch.Tensor]:
        """Predict an episode-level HDV edge response from aggregated AV pressure.

        This is the inference surface for the final HDV-response semantics:
        ``episode_av_pressure`` summarizes the whole AV policy/action trajectory
        for an imagined episode, and the head returns one converged HDV edge
        background distribution.  Older checkpoints whose HDV head was trained
        on per-step occupancy can still execute this method, but they should be
        retrained with an episode-level target before being trusted.
        """
        if self.hdv_policy_dim <= 0:
            return None
        if z_context is None:
            z = torch.zeros(
                (self.n_spatial, self.d_spatial),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            z = torch.as_tensor(z_context, device=self.device, dtype=torch.float32)
            if z.dim() == 3:
                z = z[0]
            if z.dim() != 2:
                raise ValueError(f"z_context must be (E,D) or (1,E,D), got {tuple(z.shape)}")
        packed_seq = z.unsqueeze(0).unsqueeze(0)
        action = torch.as_tensor(episode_av_pressure, device=self.device, dtype=packed_seq.dtype)
        if action.dim() > 1:
            action = action.reshape(-1)
        model_actions = self._prepare_model_actions(
            batch_size=1,
            seq_len=1,
            action=action,
            depart=None if depart is None else torch.as_tensor(depart, device=self.device, dtype=packed_seq.dtype),
            context=None if context is None else torch.as_tensor(context, device=self.device, dtype=packed_seq.dtype),
            device=self.device,
            dtype=packed_seq.dtype,
        )
        flow_time = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
        _, _, hdv_logits = self.dynamics(
            model_actions, None, None, packed_seq,
            act_mask=None, agent_tokens=None,
            flow_time=flow_time, return_hdv_policy=True,
        )
        if hdv_logits is None:
            return None
        if normalize:
            return torch.softmax(hdv_logits[0, -1].float(), dim=-1).detach().cpu()
        return hdv_logits[0, -1].float().detach().cpu()

    @torch.inference_mode()
    def predict_t_all_from_latent(self, z: Union[np.ndarray, torch.Tensor]) -> Optional[float]:
        """
        Predict episode-level t_all (minutes) directly from projected latent tokens.

        Args:
            z: (E, d_spatial) latent flow tokens.
        Returns:
            Scalar t_all prediction in minutes, or None if head is unavailable.
        """
        pred_fn = getattr(self.flow_projector, "predict_t_all_from_z", None)
        if pred_fn is None:
            return None
        try:
            if isinstance(z, np.ndarray):
                z_t = torch.from_numpy(z).to(device=self.device, dtype=torch.float32)
            else:
                z_t = z.to(device=self.device)
            if z_t.dim() == 2:
                z_t = z_t.unsqueeze(0)
            pred = pred_fn(z_t).reshape(-1)[0]
            out = float(pred.detach().cpu().item())
            if np.isfinite(out):
                return out
        except Exception:
            return None
        return None
