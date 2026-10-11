import pytest
import torch

from tools.dynamic_routing_prior import DynamicPriorActor, apply_dynamic_routing_prior


def test_zero_residual_matches_shortest_cost_and_ties_without_changing_features():
    actor = DynamicPriorActor()
    features = torch.randn(3, 3, 10)
    original = features.clone()
    mask = torch.tensor([[True, True, False], [True, True, True], [False, True, True]])
    costs = torch.tensor([[100., 50., float('nan')], [20., 20., 21.], [0., 100., 90.]])
    selected = actor.sample_action(features, mask, costs, deterministic=True)
    assert selected.candidate_index.tolist() == [1, 0, 2]
    assert torch.isfinite(selected.log_probability).all()
    assert torch.isfinite(selected.entropy).all()
    assert torch.equal(features, original)
    assert actor.prior_weight == 0


def test_bias_is_relative_masked_and_detached():
    logits = torch.tensor([[2., 3., 4.]], requires_grad=True)
    costs = torch.tensor([[60., 90., float('inf')]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    result = apply_dynamic_routing_prior(logits, mask, costs, weight=2., temperature_seconds=30.)
    torch.testing.assert_close(result, torch.tensor([[2., 1., -float('inf')]]))
    result[mask].sum().backward()
    assert costs.grad is None
    torch.testing.assert_close(logits.grad, torch.tensor([[1., 1., 0.]]))


def test_actor_learns_residual_without_prior_gradient():
    torch.manual_seed(11)
    actor = DynamicPriorActor()
    features = torch.randn(2, 3, 10)
    mask = torch.ones(2, 3, dtype=torch.bool)
    costs = torch.tensor([[10., 20., 30.], [30., 20., 10.]], requires_grad=True)
    optimizer = torch.optim.SGD(actor.parameters(), lr=.01)
    targets = torch.tensor([2, 0])
    for step in range(2):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(actor.forward_logits(features, mask, costs), targets).backward()
        assert costs.grad is None
        assert actor.policy_head.weight.grad.abs().sum() > 0
        trunk_gradient = sum(parameter.grad.abs().sum() for parameter in actor.candidate_scorer.parameters())
        assert (trunk_gradient == 0) if step == 0 else (trunk_gradient > 0)
        optimizer.step()


def test_stored_decision_costs_are_used_instead_of_later_costs():
    actor = DynamicPriorActor()
    features = torch.zeros(1, 2, 10)
    mask = torch.ones(1, 2, dtype=torch.bool)
    observed_costs = torch.tensor([[10., 90.]])
    stored_costs = observed_costs.clone()
    before = actor.forward_logits(features, mask, stored_costs).detach().clone()
    observed_costs.copy_(torch.tensor([[90., 10.]]))
    torch.testing.assert_close(actor.forward_logits(features, mask, stored_costs), before)
    assert actor.sample_action(features, mask, observed_costs, deterministic=True).candidate_index.item() == 1
    assert actor.sample_action(features, mask, stored_costs, deterministic=True).candidate_index.item() == 0


@pytest.mark.parametrize('weight,temperature', [(-1., 30.), (float('nan'), 30.), (1., 0.), (1., float('inf'))])
def test_invalid_configuration_rejected(weight, temperature):
    with pytest.raises(ValueError):
        DynamicPriorActor(prior_weight=weight, temperature_seconds=temperature)


@pytest.mark.parametrize('costs,mask', [([[float('nan'), 1.]], [[True, True]]),
                                     ([[-1., 1.]], [[True, True]]),
                                     ([[0., 1.]], [[False, False]])])
def test_invalid_legal_costs_and_empty_decisions_rejected_even_at_zero_weight(costs, mask):
    with pytest.raises(ValueError):
        apply_dynamic_routing_prior(torch.zeros(1, 2), torch.tensor(mask), torch.tensor(costs), weight=0.)


def test_rejects_accidental_double_static_prior():
    actor = DynamicPriorActor()
    actor.prior_weight = 1.
    with pytest.raises(ValueError, match='static prior'):
        actor.forward_logits(torch.zeros(1, 2, 10), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2))
