"""Route-cost and GRPO objective helpers for Dreamer CTDE training."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _vehicle_potential_progress_terms_from_paths(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    edge_cost_current: torch.Tensor,
    edge_cost_next: torch.Tensor,
    n_routes: int,
    invalid_route_cost_current: torch.Tensor,
    invalid_route_cost_next: torch.Tensor,
    temperature: float,
    step_seconds: float,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Vehicle-level potential terms for progress reward.

    Current potential is a soft shortest-route cost from the vehicle's current
    edge to its destination.  Next potential advances each candidate route by
    one decision-step time budget, then takes the actor-distribution-weighted
    remaining suffix cost.  This makes route choices affect the progress signal
    through the vehicle's predicted position, not only through congestion-field
    changes in the world model.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if edge_cost_current.dim() != 1 or edge_cost_next.dim() != 1:
        raise ValueError("edge costs must be 1D tensors")

    current_costs, route_valid = _route_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost=edge_cost_current,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost_current,
    )
    if current_costs is None or route_valid is None:
        return None, None

    device = route_probs.device
    dtype = route_probs.dtype
    current_costs = current_costs.to(device=device, dtype=dtype)
    route_valid = route_valid.to(device=device)
    fallback_cost = invalid_route_cost_current.to(device=device, dtype=dtype)
    valid_any = route_valid.any(dim=1)
    if not bool(valid_any.any().detach().cpu().item()):
        return None, None

    large_cost = fallback_cost + current_costs.detach().abs().max().clamp_min(1.0) * 1000.0
    masked_current = torch.where(route_valid, current_costs, large_cost)
    temp = float(temperature)
    if temp <= 1e-8:
        current_potential = masked_current.min(dim=1).values
    else:
        tau = torch.as_tensor(temp, device=device, dtype=dtype)
        current_potential = -tau * torch.logsumexp(-masked_current / tau, dim=1)

    suffix_costs = _advanced_suffix_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost_current=edge_cost_current,
        edge_cost_next=edge_cost_next,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost_next,
        step_seconds=step_seconds,
    )
    if suffix_costs is None:
        return None, None
    suffix_costs = suffix_costs.to(device=device, dtype=dtype)

    k_eff = min(int(route_probs.shape[1]), int(suffix_costs.shape[1]))
    probs = route_probs[: int(suffix_costs.shape[0]), :k_eff].to(device=device, dtype=dtype)
    valid = route_valid[: int(suffix_costs.shape[0]), :k_eff]
    probs = torch.where(valid, torch.relu(probs), torch.zeros_like(probs))
    prob_sum = probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    next_potential = torch.sum((probs / prob_sum) * suffix_costs[:, :k_eff], dim=1)

    mask = valid_any[: int(next_potential.shape[0])]
    return current_potential[: int(next_potential.shape[0])][mask].mean(), next_potential[mask].mean()


def _advanced_suffix_cost_matrix_from_paths(
    *,
    path_batch: List[List[List[int]]],
    edge_cost_current: torch.Tensor,
    edge_cost_next: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    step_seconds: float,
) -> Optional[torch.Tensor]:
    """Return route suffix costs after moving along each candidate path.

    The suffix start is chosen with detached current edge travel times because
    it is a discrete vehicle-position update.  The returned costs still depend
    on ``edge_cost_next``, so gradients flow through the next-state travel-time
    prediction.
    """
    if edge_cost_next.dim() != 1:
        raise ValueError(f"edge_cost_next must be 1D, got {tuple(edge_cost_next.shape)}")

    n_agents = len(path_batch)
    k_eff = int(n_routes)
    if n_agents <= 0 or k_eff <= 0:
        return None

    device = edge_cost_next.device
    dtype = edge_cost_next.dtype
    n_edges = int(edge_cost_next.shape[0])
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    current_np = edge_cost_current.detach().cpu().numpy()
    costs = torch.empty((n_agents, k_eff), device=device, dtype=dtype)

    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if not idxs:
                costs[i, k] = fallback_cost
                continue
            suffix = _advance_path_by_seconds(idxs, current_np, float(step_seconds), n_edges)
            if not suffix:
                costs[i, k] = torch.zeros((), device=device, dtype=dtype)
                continue
            edge_idx = torch.tensor(suffix, device=device, dtype=torch.long)
            costs[i, k] = edge_cost_next.index_select(0, edge_idx).sum().to(dtype=dtype)

    return costs


def _advance_path_by_seconds(
    path: List[int],
    edge_cost_seconds: np.ndarray,
    step_seconds: float,
    n_edges: int,
) -> List[int]:
    remaining = max(float(step_seconds), 0.0)
    valid_path = [int(e) for e in path if 0 <= int(e) < int(n_edges)]
    if not valid_path or remaining <= 1e-8:
        return valid_path

    for pos, edge_idx in enumerate(valid_path):
        edge_seconds = float(edge_cost_seconds[int(edge_idx)])
        if not np.isfinite(edge_seconds) or edge_seconds <= 1e-8:
            edge_seconds = 1e-8
        if remaining < edge_seconds:
            return valid_path[pos:]
        remaining -= edge_seconds
    return []


def _softmin_route_cost_from_paths(
    *,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    temperature: float,
) -> Optional[torch.Tensor]:
    """Policy-independent soft shortest-path cost over current candidates.

    This defines the per-agent potential R_i(s_k): each active vehicle only
    looks at candidate paths from its current edge to its destination, then
    takes a soft minimum over candidate route costs.  The actor probabilities
    are intentionally not used here, so progress reward measures state
    improvement instead of rewarding the current policy's own route mixture.
    """
    route_costs, route_valid = _route_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost=edge_cost,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost,
    )
    if route_costs is None or route_valid is None:
        return None

    dtype = route_costs.dtype
    device = route_costs.device
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    valid_any = route_valid.any(dim=1)
    if not bool(valid_any.any().detach().cpu().item()):
        return None

    large_cost = fallback_cost + route_costs.detach().abs().max().clamp_min(1.0) * 1000.0
    masked_costs = torch.where(route_valid, route_costs, large_cost)

    temp = float(temperature)
    if temp <= 1e-8:
        per_agent = masked_costs.min(dim=1).values
    else:
        tau = torch.as_tensor(temp, device=device, dtype=dtype)
        per_agent = -tau * torch.logsumexp(-masked_costs / tau, dim=1)
    per_agent = torch.where(valid_any, per_agent, fallback_cost)
    return per_agent[valid_any].mean()


def _expected_route_cost_from_paths(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Expected candidate-path cost under the actor distribution.

    This uses only current candidate route topology plus WM-predicted edge costs.
    It deliberately does not encode any externally observed best route label.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if edge_cost.dim() != 1:
        raise ValueError(f"edge_cost must be 1D, got {tuple(edge_cost.shape)}")

    n_agents = min(int(route_probs.shape[0]), len(path_batch))
    k_eff = min(int(route_probs.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    n_edges = int(edge_cost.shape[0])
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    agent_costs: List[torch.Tensor] = []

    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        costs: List[torch.Tensor] = []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                costs.append(edge_cost.index_select(0, edge_idx).sum().to(dtype=dtype))
            else:
                costs.append(fallback_cost)
        route_costs = torch.stack(costs)
        probs = route_probs[i, :k_eff]
        prob_sum = probs.sum().clamp_min(1e-8)
        agent_costs.append(torch.sum((probs / prob_sum) * route_costs))

    return torch.stack(agent_costs).mean()


def _expected_route_cost_from_cost_matrix(
    *,
    route_probs: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Expected candidate-route cost from a precomputed cost matrix."""
    n = min(int(route_probs.shape[0]), int(route_costs.shape[0]), int(route_valid.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_costs.shape[1]), int(route_valid.shape[1]))
    if n <= 0 or k <= 0:
        return None
    probs = route_probs[:n, :k].to(device=route_costs.device, dtype=route_costs.dtype)
    costs = route_costs[:n, :k]
    valid = route_valid[:n, :k].to(device=route_costs.device)
    agent_mask = valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None
    probs = torch.where(valid, torch.relu(probs), torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    per_agent = (probs * costs.masked_fill(~valid, 0.0)).sum(dim=-1)
    return per_agent[agent_mask].mean()


def _expected_route_regret_from_cost_matrix(
    *,
    route_probs: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
) -> Optional[torch.Tensor]:
    """Expected per-agent route regret from a precomputed cost matrix."""
    n = min(int(route_probs.shape[0]), int(route_costs.shape[0]), int(route_valid.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_costs.shape[1]), int(route_valid.shape[1]))
    if n <= 0 or k <= 0:
        return None
    probs = route_probs[:n, :k].to(device=route_costs.device, dtype=route_costs.dtype)
    valid = route_valid[:n, :k].to(device=route_costs.device)
    costs = route_costs[:n, :k].to(device=route_costs.device, dtype=route_costs.dtype)
    agent_mask = valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None

    probs = torch.where(valid, torch.relu(probs), torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    large = torch.finfo(costs.dtype).max / 4.0
    masked_costs = costs.masked_fill(~valid, large)
    min_cost = masked_costs.min(dim=-1, keepdim=True).values
    regret = (masked_costs - min_cost) / float(max(temperature, 1e-6))
    regret = regret.masked_fill(~valid, 0.0)
    per_agent = (probs * regret).sum(dim=-1)
    return per_agent[agent_mask].mean()


def _expected_route_regret_from_paths(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
    temperature: float,
) -> Optional[torch.Tensor]:
    """Expected per-agent route regret normalized by a seconds temperature."""
    route_costs, route_valid = _route_cost_matrix_from_paths(
        path_batch=path_batch,
        edge_cost=edge_cost,
        n_routes=n_routes,
        invalid_route_cost=invalid_route_cost,
    )
    if route_costs is None or route_valid is None:
        return None
    n = min(int(route_probs.shape[0]), int(route_costs.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_costs.shape[1]))
    if n <= 0 or k <= 0:
        return None

    return _expected_route_regret_from_cost_matrix(
        route_probs=route_probs[:n, :k],
        route_costs=route_costs[:n, :k],
        route_valid=route_valid[:n, :k],
        temperature=float(temperature),
    )


def _route_cost_matrix_from_paths(
    *,
    path_batch: List[List[List[int]]],
    edge_cost: torch.Tensor,
    n_routes: int,
    invalid_route_cost: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return per-agent candidate route costs and a valid-path mask.

    Costs are computed from the current imagined state and candidate path
    topology only.  No observed best-route labels or future SUMO outcomes are
    used.
    """
    if edge_cost.dim() != 1:
        raise ValueError(f"edge_cost must be 1D, got {tuple(edge_cost.shape)}")

    n_agents = len(path_batch)
    k_eff = int(n_routes)
    if n_agents <= 0 or k_eff <= 0:
        return None, None

    device = edge_cost.device
    dtype = edge_cost.dtype
    n_edges = int(edge_cost.shape[0])
    fallback_cost = invalid_route_cost.to(device=device, dtype=dtype)
    costs = torch.empty((n_agents, k_eff), device=device, dtype=dtype)
    valid = torch.zeros((n_agents, k_eff), device=device, dtype=torch.bool)

    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = [int(e) for e in path if 0 <= int(e) < n_edges]
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                costs[i, k] = edge_cost.index_select(0, edge_idx).sum().to(dtype=dtype)
                valid[i, k] = True
            else:
                costs[i, k] = fallback_cost

    if not bool(valid.any().detach().cpu().item()):
        return None, None
    return costs, valid


def _route_cost_policy_aux_loss(
    *,
    logits: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
    balance_best_classes: bool = False,
    hard_target: bool = False,
    nonzero_margin: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Cross-entropy from actor logits to current best route slots.

    Candidate slot order is preserved by callers.  The target route is the
    argmin within those original slots after masking invalid candidates; this
    function never reorders or renumbers routes.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(
            f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}"
        )

    n = min(int(logits.shape[0]), int(route_costs.shape[0]))
    k = min(int(logits.shape[1]), int(route_costs.shape[1]))
    if n <= 0 or k <= 0:
        return None, {"route_cost_aux_acc": 0.0, "route_cost_aux_margin": 0.0}

    logits = logits[:n, :k]
    route_costs = route_costs[:n, :k].to(device=logits.device, dtype=logits.dtype)
    route_valid = route_valid[:n, :k].to(device=logits.device)
    agent_mask = route_valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None, {"route_cost_aux_acc": 0.0, "route_cost_aux_margin": 0.0}

    logits = logits[agent_mask]
    route_costs = route_costs[agent_mask]
    route_valid = route_valid[agent_mask]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(min=-50.0, max=50.0)
    large = torch.finfo(route_costs.dtype).max / 4.0
    masked_costs = route_costs.masked_fill(~route_valid, large)
    best = torch.argmin(masked_costs, dim=-1)

    masked_logits = logits.masked_fill(~route_valid, -large)
    if bool(hard_target):
        per_agent_loss = F.cross_entropy(masked_logits, best, reduction="none")
    else:
        min_cost = masked_costs.gather(1, best.unsqueeze(-1))
        normalized_costs = masked_costs - min_cost
        temp = float(max(temperature, 1e-6))
        target_logits = -(normalized_costs / temp).masked_fill(~route_valid, large)
        target = torch.softmax(target_logits, dim=-1).detach()
        log_probs = F.log_softmax(masked_logits, dim=-1)
        per_agent_loss = -(target * log_probs).sum(dim=-1)
    if bool(balance_best_classes):
        counts = torch.bincount(best, minlength=k).to(device=logits.device, dtype=logits.dtype)
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = counts[present].sum() / (present.to(logits.dtype).sum().clamp_min(1.0) * counts[present])
        sample_weights = weights.index_select(0, best).detach()
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
        loss = (per_agent_loss * sample_weights).mean()
    else:
        loss = per_agent_loss.mean()

    margin_value = float(nonzero_margin)
    nonzero_best_mask = best != 0
    if margin_value > 0.0 and k > 1 and bool(nonzero_best_mask.any().detach().cpu().item()):
        best_logits = masked_logits.gather(1, best.unsqueeze(-1)).squeeze(-1)
        route0_logits = masked_logits[:, 0]
        margin_terms = F.relu(route0_logits - best_logits + margin_value)[nonzero_best_mask]
        if margin_terms.numel() > 0:
            loss = loss + margin_terms.mean()

    pred = torch.argmax(masked_logits.detach(), dim=-1)
    acc = float((pred == best).float().mean().detach().cpu().item())
    nonzero_best_frac = float((best != 0).float().mean().detach().cpu().item())
    valid_counts = route_valid.sum(dim=-1)
    if bool((valid_counts > 1).any().detach().cpu().item()):
        top2 = torch.topk(masked_costs, k=2, largest=False, dim=-1).values
        margin_t = (top2[:, 1] - top2[:, 0])[valid_counts > 1]
        margin = float(margin_t.mean().detach().cpu().item()) if margin_t.numel() > 0 else 0.0
    else:
        margin = 0.0
    return loss, {
        "route_cost_aux_acc": acc,
        "route_cost_aux_margin": margin,
        "route_cost_aux_nonzero_best_frac": nonzero_best_frac,
    }


def _planned_av_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
    detach_pressure: bool = True,
) -> Optional[torch.Tensor]:
    """Return per-route overlap with the current expected planned AV edge load.

    Optional edge weights turn pure path-overlap pressure into a WM-weighted
    externality proxy, e.g. planned load on edges with high decoded delay.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if route_valid.dim() != 2:
        raise ValueError(f"route_valid must be 2D, got {tuple(route_valid.shape)}")

    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]
    probs = torch.relu(route_probs[:n_agents, :k_eff])
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    probs = probs * valid.to(dtype=dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    edge_pressure = torch.zeros((int(n_edges),), device=device, dtype=dtype)
    route_edge_indices: List[List[List[int]]] = []
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        per_agent: List[List[int]] = []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            per_agent.append(idxs)
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                edge_pressure.index_add_(0, edge_idx, probs[i, k].expand(len(idxs)))
        route_edge_indices.append(per_agent)
    edge_pressure = edge_pressure / float(max(n_agents, 1))
    if bool(detach_pressure):
        edge_pressure = edge_pressure.detach()

    pressure_costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        for k in range(k_eff):
            idxs = route_edge_indices[i][k]
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                pressure_values = edge_pressure.index_select(0, edge_idx)
                if edge_weights is not None:
                    pressure_values = pressure_values * edge_weights.index_select(0, edge_idx)
                cost_value = pressure_values.sum()
                if bool(normalize_by_path_length):
                    cost_value = cost_value / float(max(len(idxs), 1))
                pressure_costs[i, k] = cost_value
    return pressure_costs


def _marginal_av_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    pressure_route_probs: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
    detach_pressure: bool = True,
) -> Optional[torch.Tensor]:
    """Return per-route overlap with other vehicles' planned AV edge load.

    This is a COMA/difference-reward style externality proxy.  Unlike
    ``_planned_av_pressure_cost_matrix``, it removes vehicle i's expected edge
    contribution before scoring vehicle i's candidates, so the cost is
    state/edge-specific rather than a global route-slot share rule.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if route_valid.dim() != 2:
        raise ValueError(f"route_valid must be 2D, got {tuple(route_valid.shape)}")
    if pressure_route_probs is not None and pressure_route_probs.dim() != 2:
        raise ValueError(
            f"pressure_route_probs must be 2D, got {tuple(pressure_route_probs.shape)}"
        )

    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    if pressure_route_probs is not None:
        n_agents = min(n_agents, int(pressure_route_probs.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if pressure_route_probs is not None:
        k_eff = min(k_eff, int(pressure_route_probs.shape[1]))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]

    pressure_source = route_probs if pressure_route_probs is None else pressure_route_probs
    probs = torch.relu(pressure_source[:n_agents, :k_eff].to(device=device, dtype=dtype))
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    probs = probs * valid.to(dtype=dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    edge_pressure = torch.zeros((int(n_edges),), device=device, dtype=dtype)
    agent_edge_pressure = torch.zeros((n_agents, int(n_edges)), device=device, dtype=dtype)
    route_edge_indices: List[List[List[int]]] = []
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        per_agent: List[List[int]] = []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            per_agent.append(idxs)
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                contribution = probs[i, k].expand(len(idxs))
                edge_pressure.index_add_(0, edge_idx, contribution)
                agent_edge_pressure[i].index_add_(0, edge_idx, contribution)
        route_edge_indices.append(per_agent)

    denom = float(max(n_agents, 1))
    edge_pressure = edge_pressure / denom
    agent_edge_pressure = agent_edge_pressure / denom
    if bool(detach_pressure):
        edge_pressure = edge_pressure.detach()
        agent_edge_pressure = agent_edge_pressure.detach()

    pressure_costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        other_pressure = (edge_pressure - agent_edge_pressure[i]).clamp_min(0.0)
        for k in range(k_eff):
            idxs = route_edge_indices[i][k]
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                pressure_values = other_pressure.index_select(0, edge_idx)
                if edge_weights is not None:
                    pressure_values = pressure_values * edge_weights.index_select(0, edge_idx)
                cost_value = pressure_values.sum()
                if bool(normalize_by_path_length):
                    cost_value = cost_value / float(max(len(idxs), 1))
                pressure_costs[i, k] = cost_value
    return pressure_costs


def _counterfactual_fleet_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    edge_tt: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
    detach_pressure: bool = True,
) -> Optional[torch.Tensor]:
    """Return COMA-style route costs aligned with fleet pressure reward.

    For each vehicle i and candidate route k, fixed other-vehicle route
    probabilities define the counterfactual planned edge pressure after forcing
    i to k.  The cost is the same fleet-level planned-pressure readout used by
    the rollout reward, rather than a local suffix travel-time label.
    """
    if route_probs.dim() != 2:
        raise ValueError(f"route_probs must be 2D, got {tuple(route_probs.shape)}")
    if route_valid.dim() != 2:
        raise ValueError(f"route_valid must be 2D, got {tuple(route_valid.shape)}")
    if edge_tt.dim() != 1:
        raise ValueError(f"edge_tt must be 1D, got {tuple(edge_tt.shape)}")

    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    probs = torch.relu(route_probs[:n_agents, :k_eff].to(device=device, dtype=dtype))
    probs = probs * valid.to(dtype=dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if bool(detach_pressure):
        probs = probs.detach()

    edge_tt = torch.nan_to_num(
        edge_tt.to(device=device, dtype=dtype).reshape(-1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    if int(edge_tt.numel()) < int(n_edges):
        padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
        padded[: int(edge_tt.numel())] = edge_tt
        edge_tt = padded
    else:
        edge_tt = edge_tt[: int(n_edges)]
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]
        edge_cost = edge_tt * edge_weights
    else:
        edge_cost = edge_tt

    route_masks = torch.zeros((n_agents, k_eff, int(n_edges)), device=device, dtype=dtype)
    valid_counts = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            if not idxs or not bool(valid[i, k].detach().cpu().item()):
                continue
            edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
            route_masks[i, k, edge_idx] = 1.0
            valid_counts[i, k] = float(len(idxs))
    if bool(normalize_by_path_length):
        route_masks = route_masks / valid_counts.clamp_min(1.0).unsqueeze(-1)

    agent_expected_pressure = torch.einsum("ik,ike->ie", probs, route_masks)
    fleet_pressure = agent_expected_pressure.sum(dim=0) / float(max(n_agents, 1))
    if bool(detach_pressure):
        fleet_pressure = fleet_pressure.detach()
        agent_expected_pressure = agent_expected_pressure.detach()

    costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        other_pressure = fleet_pressure - agent_expected_pressure[i] / float(max(n_agents, 1))
        other_pressure = other_pressure.clamp_min(0.0)
        for k in range(k_eff):
            if not bool(valid[i, k].detach().cpu().item()):
                continue
            cf_pressure = other_pressure + route_masks[i, k] / float(max(n_agents, 1))
            pressure_sum = cf_pressure.sum().clamp_min(1e-8)
            pressure_dist = cf_pressure / pressure_sum
            costs[i, k] = torch.sum(pressure_dist * edge_cost)
    return costs.masked_fill(~valid, 0.0)


def _sequential_marginal_av_pressure_cost_matrix(
    *,
    route_probs: torch.Tensor,
    path_batch: List[List[List[int]]],
    route_valid: torch.Tensor,
    n_edges: int,
    n_routes: int,
    edge_weights: Optional[torch.Tensor] = None,
    normalize_by_path_length: bool = False,
) -> Optional[torch.Tensor]:
    """Return candidate overlap with previously fixed fleet route pressure.

    This matches the vehicle actor's sequential fleet-pressure observation:
    vehicle i is scored against the routes selected by vehicles < i, then its
    own argmax route is added to the pressure field for later vehicles.
    """
    if route_probs.dim() != 2 or route_valid.dim() != 2:
        return None
    n_agents = min(int(route_probs.shape[0]), len(path_batch), int(route_valid.shape[0]))
    k_eff = min(int(route_probs.shape[1]), int(route_valid.shape[1]), int(n_routes))
    if n_agents <= 0 or k_eff <= 0 or int(n_edges) <= 0:
        return None

    device = route_probs.device
    dtype = route_probs.dtype
    valid = route_valid[:n_agents, :k_eff].to(device=device)
    if edge_weights is not None:
        edge_weights = torch.nan_to_num(
            edge_weights.to(device=device, dtype=dtype).reshape(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        if int(edge_weights.numel()) < int(n_edges):
            padded = torch.zeros((int(n_edges),), device=device, dtype=dtype)
            padded[: int(edge_weights.numel())] = edge_weights
            edge_weights = padded
        else:
            edge_weights = edge_weights[: int(n_edges)]

    pressure = torch.zeros((int(n_edges),), device=device, dtype=dtype)
    costs = torch.zeros((n_agents, k_eff), device=device, dtype=dtype)
    for i in range(n_agents):
        paths = path_batch[i] if i < len(path_batch) else []
        for k in range(k_eff):
            path = paths[k] if k < len(paths) else []
            idxs = sorted({int(e) for e in path if 0 <= int(e) < int(n_edges)})
            if idxs and bool(valid[i, k].detach().cpu().item()):
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                pressure_values = pressure.index_select(0, edge_idx)
                if edge_weights is not None:
                    pressure_values = pressure_values * edge_weights.index_select(0, edge_idx)
                cost_value = pressure_values.sum()
                if bool(normalize_by_path_length):
                    cost_value = cost_value / float(max(len(idxs), 1))
                costs[i, k] = cost_value

        valid_i = valid[i]
        if bool(valid_i.any().detach().cpu().item()):
            large_neg = torch.finfo(dtype).min / 4.0
            logits_i = route_probs[i, :k_eff].masked_fill(~valid_i, large_neg)
            selected = int(torch.argmax(logits_i.detach()).cpu().item())
            if selected < len(paths):
                idxs = sorted({int(e) for e in paths[selected] if 0 <= int(e) < int(n_edges)})
                if idxs:
                    edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                    pressure.index_add_(0, edge_idx, torch.ones((len(idxs),), device=device, dtype=dtype))
    denom = torch.arange(n_agents, device=device, dtype=dtype).clamp_min(1.0).unsqueeze(-1)
    return costs / denom


def _route_share_penalty_cost_matrix(
    *,
    route_probs: torch.Tensor,
    route_valid: torch.Tensor,
    cap: float,
    detach_share: bool = True,
) -> Optional[torch.Tensor]:
    """Return per-candidate marginal cost from fleet-level route-slot share.

    This is a lightweight externality proxy: if the actor currently places too
    much fleet probability on candidate slot k, all candidates in that slot get
    a positive marginal cost. It keeps the original slot order intact and does
    not construct labels.
    """
    if route_probs.dim() != 2 or route_valid.dim() != 2:
        return None
    n = min(int(route_probs.shape[0]), int(route_valid.shape[0]))
    k = min(int(route_probs.shape[1]), int(route_valid.shape[1]))
    if n <= 0 or k <= 0:
        return None

    probs = torch.relu(route_probs[:n, :k])
    valid = route_valid[:n, :k].to(device=route_probs.device)
    probs = probs * valid.to(dtype=probs.dtype)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    fleet_share = probs.mean(dim=0)
    excess = (fleet_share - float(max(0.0, min(float(cap), 1.0)))).clamp_min(0.0)
    if bool(detach_share):
        excess = excess.detach()
    return excess.unsqueeze(0).expand(n, k).masked_fill(~valid, 0.0)


def _planned_av_pressure_edge_weights(
    *,
    wm: UrbanWorldModel,
    edge_tt: torch.Tensor,
    v_total: torch.Tensor,
    capacities: torch.Tensor,
    mode: str,
) -> Optional[torch.Tensor]:
    """Edge weights for turning planned route overlap into WM congestion pressure."""
    mode_key = str(mode or "overlap").strip().lower()
    if mode_key in {"", "overlap", "plain", "none"}:
        return None
    if mode_key in {"volume_ratio", "v_over_c"}:
        return (v_total / capacities.clamp_min(1.0)).clamp_min(0.0)
    if mode_key in {"excess_volume_ratio", "volume_excess"}:
        return (v_total / capacities.clamp_min(1.0) - 1.0).clamp_min(0.0)
    if mode_key in {"tt_ratio", "travel_time_ratio"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6)).clamp_min(0.0)
    if mode_key in {"excess_tt_ratio", "tt_excess", "wm_excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6) - 1.0).clamp_min(0.0)
    if mode_key in {"clipped_excess_tt_ratio", "clipped_tt_excess", "wm_clipped_excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6) - 1.0).clamp(0.0, 3.0)
    if mode_key in {"threshold_excess_tt_ratio", "thresholded_excess_tt_ratio", "wm_threshold_excess_tt"}:
        free_flow = _free_flow_edge_tt_tensor(wm, device=edge_tt.device, dtype=edge_tt.dtype)
        return (edge_tt / free_flow.clamp_min(1e-6) - 1.25).clamp(0.0, 3.0)
    raise ValueError(f"Unsupported planned AV pressure mode: {mode!r}")


def _marginal_pressure_reference_probs(
    route_probs: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    reference: str,
) -> torch.Tensor:
    """Return detached route probabilities used as the fixed fleet baseline."""
    ref = str(reference or "soft").strip().lower()
    valid = route_valid[: route_probs.shape[0], : route_probs.shape[1]].to(device=route_probs.device)
    if ref in {"", "soft", "policy", "expected"}:
        probs = torch.relu(route_probs) * valid.to(dtype=route_probs.dtype)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs.detach()
    if ref in {"argmax", "deterministic", "hard"}:
        large_neg = torch.finfo(route_probs.dtype).min / 4.0
        masked_logits = route_probs.masked_fill(~valid, large_neg)
        best = torch.argmax(masked_logits, dim=-1)
        one_hot = F.one_hot(best, num_classes=int(route_probs.shape[1])).to(
            device=route_probs.device,
            dtype=route_probs.dtype,
        )
        one_hot = one_hot * valid.to(dtype=route_probs.dtype)
        return one_hot.detach()
    raise ValueError(f"Unsupported marginal AV pressure reference: {reference!r}")


def _free_flow_edge_tt_tensor(
    wm: UrbanWorldModel,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    lengths = torch.as_tensor(
        np.asarray(
            wm._registry.lengths if wm._registry.lengths is not None
            else np.ones((wm.n_edges,), dtype=np.float32) * 100.0,
            dtype=np.float32,
        ),
        device=device,
        dtype=dtype,
    )
    speed_limits = torch.as_tensor(
        np.asarray(wm._registry.speed_limits, dtype=np.float32),
        device=device,
        dtype=dtype,
    )
    return lengths / speed_limits.clamp_min(0.1)


def _expected_cost_policy_loss(
    *,
    logits: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
    detach_cost: bool = True,
    balance_best_classes: bool = False,
    pairwise: bool = False,
    pairwise_margin: float = 0.0,
    route0_margin_weight: float = 0.0,
    route0_margin: float = 0.0,
    route0_margin_min_gap: float = 0.0,
    nonzero_best_boost: float = 0.0,
    nonzero_best_min_gap: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Direct actor objective for minimizing E_pi[C_i(a)].

    This is not a distillation/CE target: no best-route label is constructed.
    The loss is the actor's expected normalized route cost under its own
    probabilities while preserving the original candidate slot order.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(
            f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}"
        )

    n = min(int(logits.shape[0]), int(route_costs.shape[0]))
    k = min(int(logits.shape[1]), int(route_costs.shape[1]))
    empty_stats = {
        "expected_cost_loss": 0.0,
        "expected_cost_raw": 0.0,
        "expected_cost_best_prob": 0.0,
        "expected_cost_best_route0_frac": 0.0,
        "expected_cost_nonzero_best_frac": 0.0,
        "expected_cost_margin": 0.0,
        "expected_cost_route0_margin_loss": 0.0,
        "expected_cost_route0_margin_frac": 0.0,
    }
    if n <= 0 or k <= 0:
        return None, empty_stats

    logits = logits[:n, :k]
    route_costs = route_costs[:n, :k].to(device=logits.device, dtype=logits.dtype)
    route_valid = route_valid[:n, :k].to(device=logits.device)
    agent_mask = route_valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None, empty_stats

    logits = logits[agent_mask]
    route_costs = route_costs[agent_mask]
    route_valid = route_valid[agent_mask]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(min=-50.0, max=50.0)
    large = torch.finfo(route_costs.dtype).max / 4.0
    masked_costs = route_costs.masked_fill(~route_valid, large)
    masked_logits = logits.masked_fill(~route_valid, -large)
    probs = torch.softmax(masked_logits, dim=-1)
    best = torch.argmin(masked_costs, dim=-1)
    min_cost = masked_costs.gather(1, best.unsqueeze(-1))
    valid_counts = route_valid.sum(dim=-1)
    if k > 1:
        top2 = torch.topk(masked_costs, k=2, largest=False, dim=-1).values
        best_gap = top2[:, 1] - top2[:, 0]
    else:
        best_gap = torch.zeros_like(min_cost.squeeze(-1))
    scaled_costs = (masked_costs - min_cost) / float(max(temperature, 1e-6))
    scaled_costs = scaled_costs.masked_fill(~route_valid, 0.0)
    raw_expected = (probs * masked_costs.masked_fill(~route_valid, 0.0)).sum(dim=-1)
    if bool(detach_cost):
        scaled_costs = scaled_costs.detach()
    if bool(pairwise):
        cost_i = masked_costs.unsqueeze(2)
        cost_j = masked_costs.unsqueeze(1)
        valid_pairs = route_valid.unsqueeze(2) & route_valid.unsqueeze(1) & (cost_i < cost_j)
        cost_gap = ((cost_j - cost_i) / float(max(temperature, 1e-6))).clamp_min(0.0)
        cost_gap = cost_gap.masked_fill(~valid_pairs, 0.0)
        if bool(detach_cost):
            cost_gap = cost_gap.detach()
        logit_diff = masked_logits.unsqueeze(2) - masked_logits.unsqueeze(1)
        margin = float(max(pairwise_margin, 0.0))
        pair_losses = F.softplus(-(logit_diff - margin)) * cost_gap
        gap_sum = cost_gap.sum(dim=(1, 2))
        pair_sum = pair_losses.sum(dim=(1, 2))
        expected_fallback = (probs * scaled_costs).sum(dim=-1)
        per_agent_loss = torch.where(gap_sum > 1e-8, pair_sum / gap_sum.clamp_min(1e-8), expected_fallback)
    else:
        per_agent_loss = (probs * scaled_costs).sum(dim=-1)
    route0_margin_losses = torch.zeros_like(per_agent_loss)
    r0_margin_weight = float(max(route0_margin_weight, 0.0))
    if r0_margin_weight > 0.0 and k > 1:
        route0_valid = route_valid[:, 0]
        best_valid = route_valid.gather(1, best.unsqueeze(-1)).squeeze(-1)
        route0_cost = masked_costs[:, 0]
        best_cost = masked_costs.gather(1, best.unsqueeze(-1)).squeeze(-1)
        route0_gap = route0_cost - best_cost
        margin_mask = (
            (best != 0)
            & route0_valid
            & best_valid
            & (route0_gap >= float(max(route0_margin_min_gap, 0.0)))
        )
        if bool(margin_mask.any().detach().cpu().item()):
            best_logits = masked_logits.gather(1, best.unsqueeze(-1)).squeeze(-1)
            route0_logits = masked_logits[:, 0]
            route0_margin_losses = F.relu(
                route0_logits - best_logits + float(max(route0_margin, 0.0))
            )
            route0_margin_losses = route0_margin_losses * margin_mask.to(dtype=route0_margin_losses.dtype)
            route0_margin_losses = route0_margin_losses / margin_mask.to(dtype=route0_margin_losses.dtype).mean().clamp_min(1e-8)
            per_agent_loss = per_agent_loss + r0_margin_weight * route0_margin_losses
    boost = float(max(nonzero_best_boost, 0.0))
    min_gap = float(max(nonzero_best_min_gap, 0.0))
    if boost > 0.0:
        nonzero_high_conf = (best != 0) & (best_gap >= min_gap) & (valid_counts > 1)
        if bool(nonzero_high_conf.any().detach().cpu().item()):
            sample_weights = torch.ones_like(per_agent_loss)
            sample_weights = sample_weights + boost * nonzero_high_conf.to(dtype=sample_weights.dtype)
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
            per_agent_loss = per_agent_loss * sample_weights
    if bool(balance_best_classes):
        counts = torch.bincount(best, minlength=k).to(device=logits.device, dtype=logits.dtype)
        present = counts > 0
        weights = torch.zeros_like(counts)
        weights[present] = counts[present].sum() / (present.to(logits.dtype).sum().clamp_min(1.0) * counts[present])
        sample_weights = weights.index_select(0, best).detach()
        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
        per_agent_loss = per_agent_loss * sample_weights
    stats = {
        "expected_cost_loss": float(per_agent_loss.detach().mean().cpu().item()),
        "expected_cost_raw": float(raw_expected.detach().mean().cpu().item()),
        "expected_cost_best_prob": float(
            probs.gather(1, best.unsqueeze(-1)).detach().mean().cpu().item()
        ),
        "expected_cost_best_route0_frac": float((best == 0).float().mean().detach().cpu().item()),
        "expected_cost_nonzero_best_frac": float((best != 0).float().mean().detach().cpu().item()),
        "expected_cost_route0_margin_loss": float(route0_margin_losses.detach().mean().cpu().item()),
        "expected_cost_route0_margin_frac": float(
            ((best != 0) & route_valid[:, 0]).float().mean().detach().cpu().item()
        ) if k > 1 else 0.0,
    }
    if bool((valid_counts > 1).any().detach().cpu().item()):
        margin_t = (top2[:, 1] - top2[:, 0])[valid_counts > 1]
        stats["expected_cost_margin"] = float(margin_t.mean().detach().cpu().item()) if margin_t.numel() > 0 else 0.0
    else:
        stats["expected_cost_margin"] = 0.0
    return per_agent_loss.mean(), stats


def _grpo_route_cost_policy_loss(
    *,
    logits: torch.Tensor,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    temperature: float,
    normalize_advantages: bool = True,
    detach_cost: bool = True,
    pairwise_weight: float = 0.0,
    pairwise_margin: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """Per-vehicle route-group policy loss from relative route costs.

    Each active vehicle's valid candidate routes form one GRPO-style group.
    The advantage is the within-vehicle mean route cost minus the candidate
    cost, so lower-cost routes get positive advantage and higher-cost routes
    get negative advantage.  This keeps original route slots intact and does
    not build a hard best-route supervision label.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(
            f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}"
        )

    n = min(int(logits.shape[0]), int(route_costs.shape[0]))
    k = min(int(logits.shape[1]), int(route_costs.shape[1]))
    empty_stats = {
        "grpo_route_cost_loss": 0.0,
        "grpo_route_cost_raw": 0.0,
        "grpo_route_cost_best_prob": 0.0,
        "grpo_route_cost_best_route0_frac": 0.0,
        "grpo_route_cost_nonzero_best_frac": 0.0,
        "grpo_route_cost_margin": 0.0,
        "grpo_route_cost_adv_mean": 0.0,
        "grpo_route_cost_adv_std": 0.0,
        "grpo_route_cost_pairwise_loss": 0.0,
    }
    if n <= 0 or k <= 0:
        return None, empty_stats

    logits = logits[:n, :k]
    route_costs = route_costs[:n, :k].to(device=logits.device, dtype=logits.dtype)
    route_valid = route_valid[:n, :k].to(device=logits.device)
    agent_mask = route_valid.any(dim=-1)
    if not bool(agent_mask.any().detach().cpu().item()):
        return None, empty_stats

    logits = logits[agent_mask]
    route_costs = route_costs[agent_mask]
    route_valid = route_valid[agent_mask]
    large = torch.finfo(route_costs.dtype).max / 4.0
    masked_costs = route_costs.masked_fill(~route_valid, large)
    masked_logits = logits.masked_fill(~route_valid, -large)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    probs = torch.softmax(masked_logits, dim=-1)

    valid_f = route_valid.to(dtype=route_costs.dtype)
    valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    cost_sum = (route_costs.masked_fill(~route_valid, 0.0) * valid_f).sum(dim=-1, keepdim=True)
    group_mean = cost_sum / valid_count
    advantages = (group_mean - route_costs) / float(max(temperature, 1e-6))
    advantages = advantages.masked_fill(~route_valid, 0.0)
    if bool(normalize_advantages):
        centered_cost = (route_costs - group_mean).masked_fill(~route_valid, 0.0)
        group_std = torch.sqrt((centered_cost.square() * valid_f).sum(dim=-1, keepdim=True) / valid_count).clamp_min(1e-6)
        advantages = ((group_mean - route_costs) / group_std).masked_fill(~route_valid, 0.0)
    if bool(detach_cost):
        advantages = advantages.detach()

    group_weights = valid_f / valid_count
    per_agent_loss = -(group_weights * advantages * log_probs).sum(dim=-1)
    pairwise_losses = torch.zeros_like(per_agent_loss)
    pw_weight = float(max(pairwise_weight, 0.0))
    if pw_weight > 0.0 and k > 1:
        cost_low = masked_costs.unsqueeze(2)
        cost_high = masked_costs.unsqueeze(1)
        valid_pairs = (
            route_valid.unsqueeze(2)
            & route_valid.unsqueeze(1)
            & (cost_low < cost_high)
        )
        if bool(valid_pairs.any().detach().cpu().item()):
            cost_gap = ((cost_high - cost_low) / float(max(temperature, 1e-6))).clamp_min(0.0)
            if bool(normalize_advantages):
                centered_cost = (route_costs - group_mean).masked_fill(~route_valid, 0.0)
                group_std = torch.sqrt(
                    (centered_cost.square() * valid_f).sum(dim=-1, keepdim=True)
                    / valid_count
                ).clamp_min(1e-6)
                cost_gap = ((cost_high - cost_low) / group_std.unsqueeze(2)).clamp_min(0.0)
            cost_gap = cost_gap.masked_fill(~valid_pairs, 0.0)
            if bool(detach_cost):
                cost_gap = cost_gap.detach()
            logit_low = masked_logits.unsqueeze(2)
            logit_high = masked_logits.unsqueeze(1)
            margin = float(max(pairwise_margin, 0.0))
            pair_loss = F.softplus(-(logit_low - logit_high - margin)) * cost_gap
            pair_sum = pair_loss.sum(dim=(1, 2))
            pair_weight_sum = cost_gap.sum(dim=(1, 2)).clamp_min(1e-8)
            pairwise_losses = pair_sum / pair_weight_sum
            per_agent_loss = per_agent_loss + pw_weight * pairwise_losses
    best = torch.argmin(masked_costs, dim=-1)
    raw_expected = (probs * route_costs.masked_fill(~route_valid, 0.0)).sum(dim=-1)
    adv_valid = advantages.detach()[route_valid]
    stats = {
        "grpo_route_cost_loss": float(per_agent_loss.detach().mean().cpu().item()),
        "grpo_route_cost_raw": float(raw_expected.detach().mean().cpu().item()),
        "grpo_route_cost_best_prob": float(
            probs.gather(1, best.unsqueeze(-1)).detach().mean().cpu().item()
        ),
        "grpo_route_cost_best_route0_frac": float((best == 0).float().mean().detach().cpu().item()),
        "grpo_route_cost_nonzero_best_frac": float((best != 0).float().mean().detach().cpu().item()),
        "grpo_route_cost_adv_mean": float(adv_valid.mean().detach().cpu().item()) if adv_valid.numel() > 0 else 0.0,
        "grpo_route_cost_adv_std": float(adv_valid.std(unbiased=False).detach().cpu().item()) if adv_valid.numel() > 0 else 0.0,
        "grpo_route_cost_pairwise_loss": float(pairwise_losses.detach().mean().cpu().item()),
    }
    valid_counts = route_valid.sum(dim=-1)
    if bool((valid_counts > 1).any().detach().cpu().item()):
        top2 = torch.topk(masked_costs, k=2, largest=False, dim=-1).values
        margin_t = (top2[:, 1] - top2[:, 0])[valid_counts > 1]
        stats["grpo_route_cost_margin"] = float(margin_t.mean().detach().cpu().item()) if margin_t.numel() > 0 else 0.0
    else:
        stats["grpo_route_cost_margin"] = 0.0
    return per_agent_loss.mean(), stats


def _dreamer_action_route_probs(
    logits: torch.Tensor,
    *,
    mode: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return route probabilities used for the imagined environment action.

    ``expected`` preserves the old fully-soft action. Straight-through modes
    use a discrete forward route choice to match SUMO execution while keeping
    the softmax gradient for actor learning.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"st_sample", "straight_through_stochastic"}:
        mode_norm = "straight_through_sample"
    elif mode_norm in {"st_argmax", "straight_through_deterministic"}:
        mode_norm = "straight_through_argmax"
    if mode_norm not in {"expected", "straight_through_sample", "straight_through_argmax"}:
        raise ValueError(f"unsupported dreamer_action_mode={mode!r}")

    temp = max(float(temperature), 1e-6)
    soft_probs = torch.softmax(logits / temp, dim=-1)
    if mode_norm == "expected":
        return soft_probs

    if mode_norm == "straight_through_sample":
        idx = torch.multinomial(soft_probs.detach().clamp_min(0.0), num_samples=1).squeeze(-1)
    else:
        idx = torch.argmax(soft_probs.detach(), dim=-1)
    hard_probs = torch.zeros_like(soft_probs)
    hard_probs.scatter_(1, idx.unsqueeze(-1), 1.0)
    return hard_probs + soft_probs - soft_probs.detach()


def _valid_route_mask_from_paths(
    path_batch: List[List[List[int]]],
    *,
    n_routes: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((len(path_batch), int(n_routes)), device=device, dtype=torch.bool)
    for i, paths in enumerate(path_batch):
        k_eff = min(len(paths), int(n_routes))
        for k in range(k_eff):
            if paths[k]:
                mask[i, k] = True
    return mask


def _sample_masked_route_actions(
    logits: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    temperature: float,
    mode: str = "sample",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select route actions from valid candidates and return hard probs/log-probs.

    The hard one-hot route probabilities drive the imagined environment.  The
    selected log-probabilities are used by score-function losses.  ``sample``
    keeps the historical stochastic rollout behavior; ``argmax`` matches
    deterministic SUMO policy execution while still training through the
    selected action log-probability.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_valid.shape != logits.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match logits {tuple(logits.shape)}")
    mode_norm = str(mode or "sample").strip().lower()
    if mode_norm in {"stochastic", "sampled", "sampling"}:
        mode_norm = "sample"
    elif mode_norm in {"deterministic", "greedy", "eval", "auto"}:
        mode_norm = "argmax"
    if mode_norm not in {"sample", "argmax"}:
        raise ValueError(f"unsupported route action selection mode={mode!r}")
    large = torch.finfo(logits.dtype).max / 4.0
    valid = route_valid.to(device=logits.device)
    agent_has_valid = valid.any(dim=-1)
    safe_valid = valid.clone()
    if not bool(agent_has_valid.all().detach().cpu().item()):
        safe_valid[~agent_has_valid, 0] = True
    masked_logits = (logits / float(max(temperature, 1e-6))).masked_fill(~safe_valid, -large)
    dist = torch.distributions.Categorical(logits=masked_logits)
    if mode_norm == "argmax":
        selected = torch.argmax(masked_logits.detach(), dim=-1)
    else:
        selected = dist.sample()
    log_probs = dist.log_prob(selected)
    hard_probs = torch.zeros_like(logits)
    hard_probs.scatter_(1, selected.unsqueeze(-1), 1.0)
    log_probs = log_probs.masked_fill(~agent_has_valid, 0.0)
    return hard_probs, selected, log_probs


def _masked_route_log_probs(
    logits: torch.Tensor,
    route_valid: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Log probabilities over valid route candidates."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_valid.shape != logits.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match logits {tuple(logits.shape)}")
    large = torch.finfo(logits.dtype).max / 4.0
    valid = route_valid.to(device=logits.device)
    safe_valid = valid.clone()
    agent_has_valid = safe_valid.any(dim=-1)
    if not bool(agent_has_valid.all().detach().cpu().item()):
        safe_valid[~agent_has_valid, 0] = True
    masked_logits = (logits / float(max(temperature, 1e-6))).masked_fill(~safe_valid, -large)
    return torch.log_softmax(masked_logits, dim=-1)


def _route_credit_base_cost_matrix(
    *,
    route_tt_costs: torch.Tensor,
    route_valid: torch.Tensor,
    mode: str,
    step_interval: float,
    elapsed_time: Optional[torch.Tensor] = None,
    remaining_tt_eps: float = 1.0,
    progress_cap: float = 1.0,
) -> torch.Tensor:
    """Convert route suffix travel times into the local AC credit base.

    The default ``progress_fraction`` mode scores a decision by the fraction of
    the current trip's elapsed-plus-remaining time budget that can actually be
    covered before the next decision: ``-min(step_interval, remaining_route_tt)
    / (elapsed_time + remaining_route_tt)``.  This keeps the local credit as a
    per-step trip progress signal without over-rewarding short final suffixes
    where the vehicle reaches its destination before the full decision interval
    elapses.
    """
    mode_key = str(mode or "progress_fraction").strip().lower()
    if mode_key in {"", "full_path", "remaining_tt", "route_tt", "legacy"}:
        return route_tt_costs
    if mode_key in {
        "progress_fraction",
        "elapsed_progress",
        "trip_progress_fraction",
        "remaining_tt_inverse",
        "neg_inverse_remaining_tt",
        "step_progress",
    }:
        eps = float(max(remaining_tt_eps, 1e-6))
        delta_t = float(max(step_interval, 1e-6))
        cap = float(progress_cap)
        if elapsed_time is None:
            elapsed = torch.zeros(
                (route_tt_costs.shape[0], 1),
                device=route_tt_costs.device,
                dtype=route_tt_costs.dtype,
            )
        else:
            elapsed = elapsed_time.to(device=route_tt_costs.device, dtype=route_tt_costs.dtype)
            if elapsed.dim() == 0:
                elapsed = elapsed.reshape(1, 1).expand(route_tt_costs.shape[0], 1)
            elif elapsed.dim() == 1:
                elapsed = elapsed.reshape(-1, 1)
            elif elapsed.dim() == 2:
                pass
            else:
                raise ValueError(f"elapsed_time must be scalar, 1D, or 2D, got {tuple(elapsed.shape)}")
            if elapsed.shape[0] == 1 and route_tt_costs.shape[0] != 1:
                elapsed = elapsed.expand(route_tt_costs.shape[0], elapsed.shape[1])
            if elapsed.shape[0] != route_tt_costs.shape[0]:
                raise ValueError(
                    f"elapsed_time first dimension {elapsed.shape[0]} must match "
                    f"route_tt_costs rows {route_tt_costs.shape[0]}"
                )
            if elapsed.shape[1] not in {1, route_tt_costs.shape[1]}:
                raise ValueError(
                    f"elapsed_time second dimension {elapsed.shape[1]} must be 1 "
                    f"or match route count {route_tt_costs.shape[1]}"
                )
        remaining_tt = route_tt_costs.clamp_min(0.0)
        total_trip_tt = (remaining_tt + elapsed.clamp_min(0.0)).clamp_min(eps)
        effective_step = torch.minimum(
            torch.full_like(total_trip_tt, delta_t),
            remaining_tt,
        )
        progress = effective_step / total_trip_tt
        if cap > 0.0:
            progress = progress.clamp(max=cap)
        return (-progress).masked_fill(~route_valid.to(device=route_tt_costs.device), 0.0)
    if mode_key in {"none", "zero", "disabled"}:
        return torch.zeros_like(route_tt_costs)
    raise ValueError(
        f"Unsupported route credit base cost mode {mode!r}; expected "
        "progress_fraction, full_path, or none"
    )


def _selected_route_cost_advantages(
    *,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    selected_idx: torch.Tensor,
    temperature: float,
    normalize_advantages: bool,
) -> torch.Tensor:
    """Return per-sampled-route relative route-cost advantages."""
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}")
    n = min(int(route_costs.shape[0]), int(route_valid.shape[0]), int(selected_idx.reshape(-1).shape[0]))
    if n <= 0:
        return torch.zeros((0,), device=route_costs.device, dtype=route_costs.dtype)
    k = int(route_costs.shape[1])
    costs = route_costs[:n]
    valid = route_valid[:n].to(device=costs.device)
    selected = selected_idx.reshape(-1)[:n].to(device=costs.device, dtype=torch.long).clamp(0, max(k - 1, 0))
    rows = torch.arange(n, device=costs.device)
    valid_f = valid.to(dtype=costs.dtype)
    valid_count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    cost_sum = (costs.masked_fill(~valid, 0.0) * valid_f).sum(dim=-1, keepdim=True)
    group_mean = cost_sum / valid_count
    selected_cost = costs[rows, selected]
    adv = group_mean.squeeze(-1) - selected_cost
    if bool(normalize_advantages):
        centered = (costs - group_mean).masked_fill(~valid, 0.0)
        group_std = torch.sqrt((centered.square() * valid_f).sum(dim=-1) / valid_count.squeeze(-1)).clamp_min(1e-6)
        adv = adv / group_std
    else:
        adv = adv / float(max(temperature, 1e-6))
    selected_valid = valid[rows, selected]
    return adv.masked_fill(~selected_valid, 0.0)


def _combined_grpo_policy_loss(
    *,
    logits: torch.Tensor,
    selected_idx: torch.Tensor,
    old_log_probs: torch.Tensor,
    route_valid: torch.Tensor,
    route_advantages: Optional[torch.Tensor],
    rollout_advantage: Union[float, torch.Tensor] = 0.0,
    route_weight: float = 1.0,
    rollout_weight: float = 0.0,
    temperature: float = 1.0,
    policy_clip_ratio: float = 0.2,
    advantage_clip: float = 5.0,
    kl_coef: float = 0.0,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    """PPO-style policy-gradient loss on route-cost plus rollout advantages.

    The selected action and old log-probability come from the imagined rollout.
    The current log-probability is recomputed from the actor during update, so
    route-cost and rollout signals share one on-policy score-function update
    instead of being applied as separate auxiliary losses.
    """
    empty_stats = {
        "grpo_combined_loss": 0.0,
        "grpo_combined_pg_loss": 0.0,
        "grpo_combined_kl_loss": 0.0,
        "grpo_combined_adv_mean": 0.0,
        "grpo_combined_adv_std": 0.0,
        "grpo_combined_clip_frac": 0.0,
        "grpo_policy_kl": 0.0,
    }
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if route_valid.shape != logits.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match logits {tuple(logits.shape)}")

    n = min(
        int(logits.shape[0]),
        int(route_valid.shape[0]),
        int(selected_idx.reshape(-1).shape[0]),
        int(old_log_probs.reshape(-1).shape[0]),
    )
    if route_advantages is not None:
        n = min(n, int(route_advantages.reshape(-1).shape[0]))
    if n <= 0:
        return None, empty_stats

    logits = logits[:n]
    valid = route_valid[:n].to(device=logits.device)
    selected = selected_idx.reshape(-1)[:n].to(device=logits.device, dtype=torch.long)
    selected = selected.clamp(0, max(int(logits.shape[1]) - 1, 0))
    old_logp = old_log_probs.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()
    if route_advantages is None:
        route_adv = torch.zeros((n,), device=logits.device, dtype=logits.dtype)
    else:
        route_adv = route_advantages.reshape(-1)[:n].to(device=logits.device, dtype=logits.dtype).detach()

    rows = torch.arange(n, device=logits.device)
    selected_valid = valid[rows, selected]
    current_logp_all = _masked_route_log_probs(
        logits,
        valid,
        temperature=float(max(temperature, 1e-6)),
    )
    current_logp = current_logp_all[rows, selected]
    rollout_adv = torch.as_tensor(
        rollout_advantage,
        device=logits.device,
        dtype=logits.dtype,
    ).reshape(-1)
    rollout_scalar = rollout_adv[0] if int(rollout_adv.numel()) > 0 else torch.zeros((), device=logits.device, dtype=logits.dtype)
    combined_adv = (
        float(route_weight) * route_adv
        + float(rollout_weight) * rollout_scalar.detach()
    )
    finite = (
        selected_valid
        & torch.isfinite(current_logp)
        & torch.isfinite(old_logp)
        & torch.isfinite(combined_adv)
    )
    if not bool(finite.any().detach().cpu().item()):
        return None, empty_stats

    current_logp = current_logp[finite]
    old_logp = old_logp[finite]
    combined_adv = combined_adv[finite].detach()
    adv_clip = float(advantage_clip)
    if adv_clip > 0.0:
        combined_adv = combined_adv.clamp(min=-adv_clip, max=adv_clip)

    log_ratio = current_logp - old_logp
    ratio = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
    clip_eps = float(max(policy_clip_ratio, 0.0))
    if clip_eps > 0.0:
        clipped_ratio = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
        pg_terms = torch.minimum(ratio * combined_adv, clipped_ratio * combined_adv)
        clip_frac = (torch.abs(ratio - 1.0) > clip_eps).to(dtype=logits.dtype).mean()
    else:
        pg_terms = ratio * combined_adv
        clip_frac = torch.zeros((), device=logits.device, dtype=logits.dtype)
    pg_loss = -pg_terms.mean()
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    kl_loss = float(max(kl_coef, 0.0)) * approx_kl
    loss = pg_loss + kl_loss

    stats = {
        "grpo_combined_loss": float(loss.detach().cpu().item()),
        "grpo_combined_pg_loss": float(pg_loss.detach().cpu().item()),
        "grpo_combined_kl_loss": float(kl_loss.detach().cpu().item()),
        "grpo_combined_adv_mean": float(combined_adv.detach().mean().cpu().item()),
        "grpo_combined_adv_std": float(
            combined_adv.detach().std(unbiased=False).cpu().item()
        ) if int(combined_adv.numel()) > 1 else 0.0,
        "grpo_combined_clip_frac": float(clip_frac.detach().cpu().item()),
        "grpo_policy_kl": float(approx_kl.detach().cpu().item()),
    }
    return loss, stats


def _rollout_group_relative_advantages(
    costs: Union[torch.Tensor, List[float]],
    group_ids: Optional[List[int]],
    *,
    normalize: bool,
) -> torch.Tensor:
    """Return cost-minimizing advantages, preferably within same-start groups."""
    costs_t = torch.as_tensor(costs, dtype=torch.float32)
    if costs_t.numel() == 0:
        return costs_t
    if group_ids is None or len(group_ids) != int(costs_t.numel()):
        group_ids = [0 for _ in range(int(costs_t.numel()))]

    advantages = torch.zeros_like(costs_t)
    used_group_baseline = False
    for gid in sorted(set(int(g) for g in group_ids)):
        idx = [i for i, g in enumerate(group_ids) if int(g) == gid]
        if len(idx) < 2:
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=costs_t.device)
        group_costs = costs_t.index_select(0, idx_t)
        group_adv = group_costs.mean() - group_costs
        if normalize:
            group_adv = group_adv / group_costs.std(unbiased=False).clamp_min(1e-6)
        advantages.index_copy_(0, idx_t, group_adv)
        used_group_baseline = True

    if used_group_baseline:
        return advantages

    advantages = costs_t.mean() - costs_t
    if normalize and int(costs_t.numel()) > 1:
        advantages = advantages / costs_t.std(unbiased=False).clamp_min(1e-6)
    return advantages


def _rollout_baseline_relative_advantages(
    costs: Union[torch.Tensor, List[float]],
    baseline_costs: Union[torch.Tensor, List[float]],
    group_ids: Optional[List[int]],
    *,
    normalize: bool,
) -> torch.Tensor:
    """Return advantages against same-state counterfactual rollout costs."""
    costs_t = torch.as_tensor(costs, dtype=torch.float32)
    baseline_t = torch.as_tensor(baseline_costs, dtype=torch.float32)
    if costs_t.shape != baseline_t.shape:
        raise ValueError(f"costs shape {tuple(costs_t.shape)} must match baseline {tuple(baseline_t.shape)}")
    if costs_t.numel() == 0:
        return costs_t

    advantages = baseline_t - costs_t
    if not normalize:
        return advantages
    if group_ids is None or len(group_ids) != int(costs_t.numel()):
        scale = costs_t.std(unbiased=False).clamp_min(1e-6) if int(costs_t.numel()) > 1 else torch.ones_like(costs_t)
        return advantages / scale

    scaled = torch.zeros_like(advantages)
    for gid in sorted(set(int(g) for g in group_ids)):
        idx = [i for i, g in enumerate(group_ids) if int(g) == gid]
        if not idx:
            continue
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=costs_t.device)
        group_costs = costs_t.index_select(0, idx_t)
        if len(idx) > 1:
            scale = group_costs.std(unbiased=False).clamp_min(1e-6)
        else:
            scale = torch.ones((), dtype=costs_t.dtype, device=costs_t.device)
        scaled.index_copy_(0, idx_t, advantages.index_select(0, idx_t) / scale)
    return scaled


def _sequential_planned_pressure_greedy_action_probs(
    *,
    route_costs: torch.Tensor,
    route_valid: torch.Tensor,
    path_batch: List[List[List[int]]],
    n_edges: int,
    n_routes: int,
    pressure_weight: float,
) -> torch.Tensor:
    """Greedily select route-cost routes with sequential planned-AV pressure."""
    if route_costs.dim() != 2:
        raise ValueError(f"route_costs must be 2D, got {tuple(route_costs.shape)}")
    if route_valid.shape != route_costs.shape:
        raise ValueError(f"route_valid shape {tuple(route_valid.shape)} must match route_costs {tuple(route_costs.shape)}")
    n_agents = min(int(route_costs.shape[0]), len(path_batch))
    k_eff = min(int(route_costs.shape[1]), int(n_routes))
    action_probs = torch.zeros_like(route_costs)
    if n_agents <= 0 or k_eff <= 0:
        return action_probs

    device = route_costs.device
    dtype = route_costs.dtype
    edge_pressure = torch.zeros((max(int(n_edges), 0),), device=device, dtype=dtype)
    norm = float(max(n_agents, 1))
    large = torch.finfo(dtype).max / 4.0
    penalty = max(float(pressure_weight), 0.0)

    for i in range(n_agents):
        valid_i = route_valid[i, :k_eff].to(device=device)
        if not bool(valid_i.any().detach().cpu().item()):
            continue
        row = route_costs[i, :k_eff].to(device=device, dtype=dtype).clone()
        if penalty > 0.0 and int(edge_pressure.numel()) > 0:
            routes = path_batch[i] if i < len(path_batch) else []
            for k in range(min(k_eff, len(routes))):
                if not bool(valid_i[k].detach().cpu().item()):
                    continue
                idxs = sorted({int(e) for e in routes[k] if 0 <= int(e) < int(edge_pressure.numel())})
                if idxs:
                    edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                    row[k] = row[k] + penalty * edge_pressure.index_select(0, edge_idx).sum() / norm
        masked = row.masked_fill(~valid_i, large)
        choice = int(torch.argmin(masked).detach().cpu().item())
        action_probs[i, choice] = 1.0
        routes = path_batch[i] if i < len(path_batch) else []
        if choice < len(routes) and int(edge_pressure.numel()) > 0:
            idxs = sorted({int(e) for e in routes[choice] if 0 <= int(e) < int(edge_pressure.numel())})
            if idxs:
                edge_idx = torch.tensor(idxs, device=device, dtype=torch.long)
                edge_pressure.index_add_(0, edge_idx, torch.ones((len(idxs),), device=device, dtype=dtype))
    return action_probs


def _standardize_with_detached_stats(
    values: torch.Tensor,
    *,
    center: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize values while keeping gradients through the original tensor."""
    if values.numel() <= 1:
        return values
    ref = values.detach()
    std = ref.std(unbiased=False).clamp_min(float(eps))
    if bool(center):
        return (values - ref.mean()) / std
    return values / std


def _mean_path_invalid_rate(
    path_batch: List[List[List[int]]],
    n_routes: int,
) -> float:
    total = 0
    invalid = 0
    for agent_paths in path_batch:
        k_eff = min(len(agent_paths), int(n_routes))
        for k in range(k_eff):
            total += 1
            if not agent_paths[k]:
                invalid += 1
    if total <= 0:
        return 0.0
    return float(invalid) / float(total)


def _route_prob_churn(
    prev_probs: Optional[torch.Tensor],
    cur_probs: torch.Tensor,
) -> float:
    if prev_probs is None:
        return 0.0
    n = min(int(prev_probs.shape[0]), int(cur_probs.shape[0]))
    if n <= 0:
        return 0.0
    churn = (cur_probs[:n].detach() - prev_probs[:n]).abs().mean().item()
    return float(churn)
