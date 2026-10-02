import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from urban_dreamer.learning.baselines.gnac import GraphNextHopActor
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.models.policy import (
    NextHopActor, apply_shortest_path_prior, set_shortest_path_prior,
    shortest_path_prior_weight,
)


def features_for_costs(costs):
    costs = torch.as_tensor(costs, dtype=torch.float64)
    features = torch.zeros((*costs.shape, 10), dtype=costs.dtype)
    features[..., 4] = torch.log1p(costs) / math.log1p(3600.)
    return features


def test_full_cost_from_observations_includes_candidate_and_junction():
    class Candidates:
        max_candidates = 2
        network = SimpleNamespace(edge_count=4, lengths_m=np.array([10., 20., 30., 40.]),
                                  free_flow_travel_seconds=np.array([1., 2., 3., 4.]))

        def candidates(self, current, destination):
            return (1, 2)

        def distances(self, destination):
            return np.array([70., 40., 40., 0.])

        def costs(self, destination):
            return np.array([7., 4., 4., 0.])

        def junction(self, current, successor):
            return (30., 3.) if successor == 1 else (0., 1.)

    obs = build_actor_observations(policy_edge_features=np.ones((4, 2)),
        vehicle_states=[dict(current_edge=0, destination_edge=3, remaining_distance=5)],
        candidates=Candidates(), episode_progress=0., active_vehicle_fraction=1.)
    result = apply_shortest_path_prior(torch.zeros(1, 2), torch.from_numpy(obs.features),
        torch.from_numpy(obs.mask), weight=1., temperature_seconds=3.)
    # Equal downstream costs still differ: (10 + 2 + 4) - (0 + 3 + 4) = 9 s.
    torch.testing.assert_close(result, torch.tensor([[-3., 0.]]))
    assert Categorical(logits=result).probs[0, 0] > 0


def test_prior_odds_mask_and_cost_offset_invariance():
    features = features_for_costs([[100., 130., 0.]])
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0.3, -0.2, -torch.inf]], dtype=torch.float64)
    result = apply_shortest_path_prior(logits, features, mask, weight=2., temperature_seconds=30.)
    assert torch.isneginf(result[0, 2])
    probs = Categorical(logits=result).probs
    assert float(probs[0, 1] / probs[0, 0]) == pytest.approx(math.exp(-2.5))
    shifted = apply_shortest_path_prior(logits, features_for_costs([[500., 530., 0.]]), mask,
                                       weight=2., temperature_seconds=30.)
    torch.testing.assert_close(result, shifted)
    features[0, 2, 4] = torch.nan
    torch.testing.assert_close(result, apply_shortest_path_prior(logits, features, mask,
                                          weight=2., temperature_seconds=30.))


def test_zero_weight_is_exact_identity_even_for_unused_cost_features():
    logits = torch.tensor([[1., -torch.inf]])
    result = apply_shortest_path_prior(logits, torch.full((1, 2, 10), torch.nan),
                                      torch.tensor([[True, False]]),
                                      weight=0., temperature_seconds=30.)
    assert result is logits


def test_biased_policy_gradient_and_detached_prior():
    logits = torch.tensor([[0.1, 0.4]], dtype=torch.float64, requires_grad=True)
    features = features_for_costs([[20., 50.]]).requires_grad_()
    result = apply_shortest_path_prior(logits, features, torch.ones(1, 2, dtype=torch.bool),
                                      weight=1., temperature_seconds=30.)
    distribution = Categorical(logits=result)
    (-distribution.log_prob(torch.tensor([1]))).sum().backward()
    expected = distribution.probs.detach().clone()
    expected[0, 1] -= 1
    torch.testing.assert_close(logits.grad, expected)
    assert torch.isfinite(logits.grad).all()
    assert features.grad is None


@pytest.mark.parametrize('graph', [False, True])
def test_actor_sampling_uses_same_prior_and_old_state_dict_remains_loadable(graph):
    torch.manual_seed(18)
    actor = GraphNextHopActor(np.eye(3, dtype=np.float32)) if graph else NextHopActor()
    features = features_for_costs([[10., 100., 0.]]).float()
    mask = torch.tensor([[True, True, False]])
    args = (features, mask)
    if graph:
        args += (torch.tensor([[1, 2, -1]]), torch.tensor([0]), torch.zeros(3, 8))
    original_state = {key: value.clone() for key, value in actor.state_dict().items()}
    original_parameters = sum(p.numel() for p in actor.parameters())
    original_logits = actor.forward_logits(*args)
    if not graph:
        old_logits = actor.policy_head(actor.candidate_scorer(features)).squeeze(-1).masked_fill(~mask, -torch.inf)
        assert torch.equal(original_logits, old_logits)
    set_shortest_path_prior(actor, 1., 30.)
    biased = actor.forward_logits(*args)
    torch.testing.assert_close(biased[:, :2], original_logits[:, :2] - torch.tensor([[0., 3.]]))
    selection = actor.sample_action(*args)
    expected = Categorical(logits=biased)
    torch.testing.assert_close(selection.log_probability, expected.log_prob(selection.candidate_index))
    torch.testing.assert_close(selection.entropy, expected.entropy())
    (-selection.log_probability.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
    assert actor.state_dict().keys() == original_state.keys()
    assert sum(p.numel() for p in actor.parameters()) == original_parameters
    actor.load_state_dict(original_state, strict=True)
    set_shortest_path_prior(actor, 0., 30.)
    assert torch.equal(actor.forward_logits(*args), original_logits)


def test_cosine_schedule_boundaries_and_resume():
    assert shortest_path_prior_weight(0) == shortest_path_prior_weight(200) == 1.
    assert shortest_path_prior_weight(600) == pytest.approx(0.5)
    assert shortest_path_prior_weight(1000) == shortest_path_prior_weight(5000) == 0.
    full = [shortest_path_prior_weight(ep) for ep in range(1500)]
    resumed = [shortest_path_prior_weight(ep) for ep in range(473, 1500)]
    assert full[473:] == resumed
    assert all(first >= second for first, second in zip(full, full[1:]))


@pytest.mark.parametrize('kwargs', [dict(episode=-1), dict(episode=1.5), dict(episode=True),
    dict(episode=1, initial_weight=float('nan')), dict(episode=1, initial_weight=-1),
    dict(episode=1, hold_episodes=-1), dict(episode=1, off_episode=200)])
def test_schedule_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        shortest_path_prior_weight(**kwargs)


@pytest.mark.parametrize('weight,temperature', [(-1, 30), (float('inf'), 30),
    (float('nan'), 30), (1, 0), (0, -1), (1, float('nan'))])
def test_setter_rejects_invalid_settings(weight, temperature):
    with pytest.raises(ValueError):
        set_shortest_path_prior(SimpleNamespace(), weight, temperature)


@pytest.mark.parametrize('bad_cost', [-1., float('nan'), float('inf'), 100.])
def test_prior_rejects_invalid_or_overflowing_legal_costs(bad_cost):
    features = torch.zeros(1, 1, 10)
    features[..., 4] = bad_cost
    with pytest.raises(ValueError):
        apply_shortest_path_prior(torch.zeros(1, 1), features, torch.ones(1, 1, dtype=torch.bool),
                                 weight=1., temperature_seconds=30.)


def test_prior_rejects_empty_legal_set():
    with pytest.raises(ValueError, match='at least one legal'):
        apply_shortest_path_prior(torch.zeros(1, 2), features_for_costs([[1., 2.]]),
                                 torch.zeros(1, 2, dtype=torch.bool),
                                 weight=1., temperature_seconds=30.)
