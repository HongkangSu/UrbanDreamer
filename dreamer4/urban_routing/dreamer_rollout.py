from __future__ import annotations

from typing import List, Optional

import torch


def aggregate_route_probs_to_action_edge(
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
    *,
    current_edges: Optional[torch.Tensor] = None,
    edge_travel_times: Optional[torch.Tensor] = None,
    horizon_seconds: Optional[float] = None,
) -> torch.Tensor:
    """Differentiably aggregate route probabilities into planned edge flow.

    The output matches collection-time ``av_planned_edge_flow``: expected future
    edge entries within the decision horizon, averaged by active vehicle.  Edge
    selection itself is discrete, so gradients flow through route probabilities
    but not through the horizon cutoff.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    device = route_probs.device
    dtype = route_probs.dtype
    action_edge = torch.zeros(int(n_edges), device=device, dtype=dtype)
    current_cpu = None
    if current_edges is not None:
        current_cpu = current_edges.detach().to(device="cpu", dtype=torch.long).view(-1)
    edge_tt_cpu = None
    use_horizon = edge_travel_times is not None and horizon_seconds is not None
    if use_horizon:
        edge_tt_cpu = edge_travel_times.detach().to(device="cpu", dtype=torch.float32).view(-1)
        if int(edge_tt_cpu.numel()) < int(n_edges):
            use_horizon = False
    horizon = max(float(horizon_seconds or 0.0), 1.0)

    n_agents = min(int(route_probs.shape[0]), len(path_batch))
    n_controlled = 0
    for i in range(n_agents):
        paths = path_batch[i]
        k_eff = min(len(paths), int(n_routes), int(route_probs.shape[1]))
        if k_eff <= 0:
            continue
        n_controlled += 1
        for k in range(k_eff):
            raw_idxs = [int(e) for e in paths[k] if 0 <= int(e) < int(n_edges)]
            if use_horizon and raw_idxs:
                start_pos = 0
                current_edge = None
                if current_cpu is not None and i < int(current_cpu.numel()):
                    current_edge = int(current_cpu[i].item())
                    for pos, edge_idx in enumerate(raw_idxs):
                        if int(edge_idx) == current_edge:
                            start_pos = int(pos)
                            break
                eta = 0.0
                idxs = []
                for pos in range(start_pos, len(raw_idxs)):
                    edge_idx = int(raw_idxs[pos])
                    if current_edge is not None and pos == start_pos and edge_idx == current_edge:
                        eta += float(edge_tt_cpu[edge_idx].item())
                        continue
                    if eta >= horizon:
                        break
                    idxs.append(edge_idx)
                    eta += float(edge_tt_cpu[edge_idx].item())
            else:
                idxs = raw_idxs
            if not idxs:
                continue
            edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
            weights = route_probs[i, k].expand(edge_idx.shape[0])
            action_edge = action_edge.index_add(0, edge_idx, weights)

    if n_controlled > 0:
        action_edge = action_edge / float(n_controlled)
    return action_edge


def build_history_packed(
    latent_history: List[torch.Tensor],
    history_length: int,
) -> torch.Tensor:
    """Build differentiable past_packed tensor from recent latent history."""
    if not latent_history:
        raise ValueError("latent_history must be non-empty")
    hist = latent_history[-max(1, int(history_length)) :]
    return torch.stack(hist, dim=0).unsqueeze(0)
