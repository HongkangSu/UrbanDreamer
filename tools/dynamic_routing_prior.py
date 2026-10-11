"""Residual actor conditioned on the routing costs saved at decision time.

Costs are explicit inputs so optimization cannot accidentally query later traffic.
The driver owns the 30-second observation clock and must store each cost row with
the corresponding action. Candidate feature 4 keeps its original static meaning.
"""
from __future__ import annotations

import math

import torch
from torch.distributions import Categorical

from urban_dreamer.models.policy import ActionSelection, NextHopActor


def apply_dynamic_routing_prior(
    logits: torch.Tensor,
    mask: torch.Tensor,
    prior_cost_seconds: torch.Tensor,
    *,
    weight: float = 1.,
    temperature_seconds: float = 30.,
) -> torch.Tensor:
    """Add detached, minimum-centered legal route costs to residual logits."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('dynamic prior weight must be finite and nonnegative')
    if not math.isfinite(temperature_seconds) or temperature_seconds <= 0:
        raise ValueError('dynamic prior temperature must be finite and positive')
    if logits.ndim != 2 or logits.shape[-1] == 0:
        raise ValueError('dynamic prior requires logits (B, K) with K positive')
    if mask.shape != logits.shape or prior_cost_seconds.shape != logits.shape:
        raise ValueError('mask and prior costs must match logits (B, K)')
    legal = mask.to(device=logits.device, dtype=torch.bool)
    if not bool(legal.any(dim=-1).all()):
        raise ValueError('every decision must have at least one legal candidate')
    costs = prior_cost_seconds.detach().to(logits).masked_fill(~legal, 0.)
    if not bool(torch.isfinite(costs).all()) or bool((costs < 0).any()):
        raise ValueError('legal dynamic route costs must be finite and nonnegative')
    if not bool(torch.isfinite(logits[legal]).all()):
        raise ValueError('legal residual logits must be finite')
    minimum = costs.masked_fill(~legal, torch.inf).amin(dim=-1, keepdim=True)
    bias = (costs - minimum).masked_fill(~legal, 0.) * (weight / temperature_seconds)
    if not bool(torch.isfinite(bias).all()):
        raise ValueError('dynamic prior bias must be finite')
    return (logits - bias).masked_fill(~legal, -torch.inf)


class DynamicPriorActor(NextHopActor):
    """Learn an additive correction to current shortest-time routing.

    The zero initial residual makes argmax equal the cheapest legal candidate,
    using candidate order to break ties. Sampling remains stochastic.
    """

    def __init__(self, *, prior_weight: float = 1., temperature_seconds: float = 30.) -> None:
        super().__init__()
        if not math.isfinite(prior_weight) or prior_weight < 0:
            raise ValueError('dynamic prior weight must be finite and nonnegative')
        if not math.isfinite(temperature_seconds) or temperature_seconds <= 0:
            raise ValueError('dynamic prior temperature must be finite and positive')
        self.dynamic_prior_weight = float(prior_weight)
        self.dynamic_prior_temperature_seconds = float(temperature_seconds)
        torch.nn.init.zeros_(self.policy_head.weight)
        torch.nn.init.zeros_(self.policy_head.bias)

    def forward_logits(self, features: torch.Tensor, mask: torch.Tensor,
                       prior_cost_seconds: torch.Tensor) -> torch.Tensor:
        if self.prior_weight != 0:
            raise ValueError('static prior must remain disabled for dynamic routing')
        residual = super().forward_logits(features, mask)
        return apply_dynamic_routing_prior(
            residual, mask, prior_cost_seconds, weight=self.dynamic_prior_weight,
            temperature_seconds=self.dynamic_prior_temperature_seconds,
        )

    def sample_action(self, features: torch.Tensor, mask: torch.Tensor,
                      prior_cost_seconds: torch.Tensor, *,
                      deterministic: bool = False) -> ActionSelection:
        logits = self.forward_logits(features, mask, prior_cost_seconds)
        distribution = Categorical(logits=logits)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())
