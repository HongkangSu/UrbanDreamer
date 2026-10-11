"""Entropy strength must follow episode-population and actor-advantage units."""
import math

import pytest
import torch

from urban_dreamer.models.policy import population_scaled_entropy


def test_population_and_advantage_scale_apply_per_event_without_gradient_leak():
    entropy = torch.tensor([.2, .4], requires_grad=True)
    population = torch.tensor([100., 1000.], requires_grad=True)
    advantages = torch.tensor([-3., 3.], requires_grad=True)
    bonus, stats = population_scaled_entropy(entropy, population, advantages)
    expected = torch.tensor([.002 / 300., .002 / 3000.])
    torch.testing.assert_close(bonus, entropy * expected)
    bonus.sum().backward()
    torch.testing.assert_close(entropy.grad, expected)
    assert population.grad is advantages.grad is None
    assert stats['entropy_advantage_scale'].item() == 3.
    assert all(not value.requires_grad and value.grad_fn is None for value in stats.values())


def test_single_event_scale_floor_and_scalar_population():
    bonus, stats = population_scaled_entropy(torch.tensor([.7]), torch.tensor(1000), torch.tensor([9.]))
    torch.testing.assert_close(bonus, torch.tensor([.7 * .002 / 1000]))
    assert stats['entropy_advantage_scale'].item() == 1.


@pytest.mark.parametrize('mode,coefficient', [('legacy', .002), ('off', 0.), ('annealed', .002)])
def test_control_modes_preserve_legacy_or_remove_entropy(mode, coefficient):
    entropy = torch.tensor([.2, .4], requires_grad=True)
    bonus, _ = population_scaled_entropy(entropy, torch.tensor([100, 1000]),
                                         torch.tensor([-100., 100.]), mode=mode)
    torch.testing.assert_close(bonus, coefficient * entropy)
    bonus.sum().backward()
    torch.testing.assert_close(entropy.grad, torch.full_like(entropy, coefficient))


def test_population_scaling_preserves_faster_route_gradient_despite_exploration():
    gradients = {}
    # One vehicle saves 30 seconds in a 1,000-car episode. Exact expectation
    # isolates entropy's objective effect without noisy sampled returns or a Q.
    for mode in ('legacy', 'population', 'off'):
        logit = torch.tensor(math.log(9.), requires_grad=True)
        probability = logit.sigmoid()
        distribution = torch.distributions.Bernoulli(probs=probability)
        entropy, _ = population_scaled_entropy(distribution.entropy()[None], torch.tensor([1000]),
                                               torch.tensor([0.]), mode=mode)
        objective = probability * (30. / (1000. * 60.)) + entropy.mean()
        objective.backward()
        gradients[mode] = logit.grad.item()
    assert gradients['legacy'] < 0.
    assert 0. < gradients['population'] < gradients['off']
    assert gradients['off'] == pytest.approx(.9 * .1 * .0005)


def test_illegal_candidate_entropy_has_no_gradient_and_conditions_stay_detached():
    logits = torch.tensor([[1., 50., 0.]], requires_grad=True)
    legal = torch.tensor([[True, False, True]])
    distribution = torch.distributions.Categorical(logits=logits.masked_fill(~legal, -torch.inf))
    advantages = torch.tensor([.3], requires_grad=True)
    bonus, _ = population_scaled_entropy(distribution.entropy(), torch.tensor([1000]), advantages)
    bonus.sum().backward()
    assert logits.grad[0, 1] == 0.
    assert torch.isfinite(logits.grad).all()
    assert advantages.grad is None


@pytest.mark.parametrize('population', [0., -1., .5, float('nan'), float('inf')])
def test_invalid_populations_are_rejected_without_division(population):
    with pytest.raises(ValueError, match='positive integers'):
        population_scaled_entropy(torch.tensor([.3]), torch.tensor([population]), torch.tensor([.1]))


def test_empty_event_batch_is_finite_without_std_warning():
    bonus, stats = population_scaled_entropy(torch.empty(0), torch.empty(0), torch.empty(0))
    assert bonus.shape == (0,)
    assert all(torch.isfinite(value) for value in stats.values())


def test_mismatched_event_axes_are_rejected():
    with pytest.raises(ValueError, match='align per event'):
        population_scaled_entropy(torch.ones(2), torch.tensor([1000]), torch.ones(2))


@pytest.mark.parametrize('coefficient', [-.1, float('nan'), float('inf')])
def test_invalid_coefficient_rejected(coefficient):
    with pytest.raises(ValueError, match='coefficient'):
        population_scaled_entropy(torch.ones(1), torch.ones(1), torch.ones(1), base_coef=coefficient)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match='mode'):
        population_scaled_entropy(torch.ones(1), torch.ones(1), torch.ones(1), mode='automatic')


def test_annealed_coefficient_has_no_population_or_advantage_scaling():
    from urban_dreamer.learning.entropy_schedule import entropy_schedule_profile
    from urban_dreamer.workflow.config import MethodConfig

    coefficient = entropy_schedule_profile(MethodConfig(actor_entropy_mode='annealed'),
        completed_episodes=4000, total_episodes=5000)['coefficient']
    entropies = torch.tensor([.2, .4], requires_grad=True)
    advantages = torch.tensor([-100., 100.], requires_grad=True)
    populations = torch.tensor([100., 3000.], requires_grad=True)
    bonus, stats = population_scaled_entropy(entropies, populations, advantages,
                                            mode='annealed', base_coef=coefficient)
    torch.testing.assert_close(bonus, entropies * .0001)
    bonus.sum().backward()
    torch.testing.assert_close(entropies.grad, torch.full_like(entropies, .0001))
    assert populations.grad is advantages.grad is None
    assert stats['entropy_coefficient_min'] == stats['entropy_coefficient_max']
