"""
Shared Gawron-style update utilities.

This module unifies two HDV route-choice update backends:

1) "replicator" (existing project behavior)
   log-space probability update with travel-time advantage.

2) "routerl" compatibility mode
   cost-memory based update aligned with RouteRL's Gawron spirit:
   - action probabilities derived from exp(beta * cost)
   - cost adapted toward observed route travel times
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def normalize_probs(probs: np.ndarray, min_prob: float = 0.0) -> np.ndarray:
    """Normalize a probability vector with optional floor."""
    out = np.asarray(probs, dtype=np.float32).copy()
    if out.ndim != 1:
        raise ValueError(f"Expected 1D probs, got shape={out.shape}")
    if out.size == 0:
        return out
    out = np.clip(out, 0.0, None)
    if min_prob > 0.0:
        out = np.maximum(out, float(min_prob))
    s = float(out.sum())
    if s <= 0.0:
        out[:] = 1.0 / float(out.size)
        return out
    out /= s
    return out


def route_tt_from_joint_frac(
    joint_frac: np.ndarray,
    k: int,
    bpr_alpha: float = 0.15,
    bpr_beta: float = 4.0,
) -> np.ndarray:
    """BPR-style per-route travel-time proxy from route load fractions."""
    jf = np.asarray(joint_frac, dtype=np.float32)
    if jf.ndim != 1:
        raise ValueError(f"Expected 1D joint_frac, got shape={jf.shape}")
    return 1.0 + float(bpr_alpha) * (float(k) * jf) ** float(bpr_beta)


def routerl_probs_from_cost(cost: np.ndarray, beta: float) -> np.ndarray:
    """
    Convert RouteRL-style route cost memory into route-choice probabilities.

    In RouteRL, smaller cost is preferred and beta is typically negative.
    """
    c = np.asarray(cost, dtype=np.float32)
    if c.ndim != 1:
        raise ValueError(f"Expected 1D cost, got shape={c.shape}")
    logits = float(beta) * c
    logits = logits - float(logits.max())
    probs = np.exp(logits)
    return normalize_probs(probs)


def gawron_step_from_route_tt(
    probs: np.ndarray,
    route_tt: np.ndarray,
    mode: str = "replicator",
    lr: float = 0.1,
    cost: Optional[np.ndarray] = None,
    alpha_zero: float = 0.2,
    beta: float = -1.5,
    update_mode: str = "expected",
    min_prob: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    One HDV update step.

    Returns:
      new_probs, new_cost_or_none
    """
    mode = str(mode).lower()
    update_mode = str(update_mode).lower()

    p = normalize_probs(np.asarray(probs, dtype=np.float32), min_prob=0.0)
    tt = np.asarray(route_tt, dtype=np.float32)
    if tt.ndim != 1 or p.ndim != 1 or tt.shape[0] != p.shape[0]:
        raise ValueError(f"Shape mismatch: probs={p.shape}, route_tt={tt.shape}")

    if mode == "replicator":
        avg_tt = float(np.dot(p, tt))
        log_p = np.log(p + 1e-10) + float(lr) * (avg_tt - tt)
        log_p = log_p - float(log_p.max())
        new_p = np.exp(log_p)
        new_p = normalize_probs(new_p, min_prob=min_prob)
        return new_p.astype(np.float32), None

    if mode == "routerl":
        if cost is None:
            c = tt.copy()
        else:
            c = np.asarray(cost, dtype=np.float32).copy()
            if c.ndim != 1 or c.shape[0] != p.shape[0]:
                raise ValueError(f"Cost shape mismatch: cost={c.shape}, probs={p.shape}")

        # RouteRL-compatible choice probabilities from current cost memory.
        p_from_cost = routerl_probs_from_cost(c, beta=float(beta))

        if update_mode == "sample":
            if rng is None:
                rng = np.random.default_rng()
            a = int(rng.choice(c.shape[0], p=p_from_cost))
            c[a] = (1.0 - float(alpha_zero)) * c[a] + float(alpha_zero) * tt[a]
        elif update_mode == "expected":
            # Expected update of RouteRL's selected-action update.
            c = c + float(alpha_zero) * p_from_cost * (tt - c)
        else:
            raise ValueError(f"Unknown update_mode='{update_mode}', expected 'expected' or 'sample'")

        new_p = routerl_probs_from_cost(c, beta=float(beta))
        new_p = normalize_probs(new_p, min_prob=min_prob)
        return new_p.astype(np.float32), c.astype(np.float32)

    raise ValueError(f"Unknown gawron mode='{mode}', expected 'replicator' or 'routerl'")

