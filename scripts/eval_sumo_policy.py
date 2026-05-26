"""Policy checkpoint helpers for SUMO evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from dreamer4.urban_routing.dreamer_actor_critic import VehicleRouteActor
except Exception:
    try:
        from dreamer_actor_critic import VehicleRouteActor
    except Exception:
        VehicleRouteActor = None


def load_policy(
    ckpt_path: str,
    obs_dim:   int,
    n_routes:  int,
    device:    str,
) -> nn.Module:
    """Load trained policy from a Dreamer4 MARL checkpoint. Auto-detects policy type."""
    print(f"  Loading policy checkpoint: {ckpt_path}")
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})
    model_family = str(state.get("model_family", cfg_dict.get("training_family", ""))).strip().lower()
    if model_family != "dreamer_ctde":
        raise RuntimeError(f"Unsupported policy checkpoint family {model_family!r}; expected dreamer_ctde.")

    hidden_dim      = int(cfg_dict.get("hidden_dim",     256))
    n_hidden_layers = int(cfg_dict.get("n_hidden_layers", 2))

    if VehicleRouteActor is None:
        raise RuntimeError("VehicleRouteActor import unavailable for Dreamer checkpoint loading.")
    route_cost_logit_bias = (
        float(cfg_dict.get("vehicle_actor_route_cost_logit_bias", 0.0))
        if bool(cfg_dict.get("vehicle_obs_include_route_costs", False))
        else 0.0
    )
    policy = VehicleRouteActor(
        obs_dim,
        n_routes,
        hidden_dim,
        n_hidden_layers,
        route_cost_logit_bias=route_cost_logit_bias,
        architecture=str(cfg_dict.get("vehicle_actor_architecture", "mlp")),
        include_global=bool(cfg_dict.get("vehicle_obs_include_global", True)),
        include_route_costs=bool(cfg_dict.get("vehicle_obs_include_route_costs", False)),
        include_fleet_pressure=bool(cfg_dict.get("vehicle_obs_include_fleet_pressure", False)),
    ).to(device)
    actor_state = state.get("actor", state.get("policy"))
    if actor_state is None:
        raise RuntimeError(f"Dreamer checkpoint missing actor state: {ckpt_path}")
    policy.load_state_dict(actor_state)
    policy.eval()
    episode_saved = state.get("episode", "?")
    print(f"  VehicleRouteActor loaded (saved at episode {episode_saved})")
    return policy
