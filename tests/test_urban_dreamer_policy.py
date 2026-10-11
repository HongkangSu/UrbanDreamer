import pytest
import torch

from urban_dreamer.models.policy import (
    CANDIDATE_FEATURE_DIM, NextHopActor, CentralizedLatentValueCritic, lambda_return,
)


def test_variable_candidates_mask_singleton_and_permutation_equivariance():
    actor = NextHopActor()
    features = torch.randn(3, 7, CANDIDATE_FEATURE_DIM)
    mask = torch.tensor([[True]*7, [False, True]+[False]*5, [True,False]*3+[True]])
    sample = actor.sample_action(features, mask)
    assert mask.gather(1, sample.candidate_index[:, None]).all()
    assert sample.candidate_index[1] == 1
    assert sample.entropy[1] == 0
    permutation = torch.tensor([3, 0, 6, 2, 1, 5, 4])
    torch.testing.assert_close(actor.forward_logits(features[:,permutation], mask[:,permutation]), actor.forward_logits(features, mask)[:,permutation])
    assert actor.sample_action(features[:, :1], torch.ones(3,1,dtype=torch.bool)).candidate_index.eq(0).all()


def test_all_invalid_is_error_not_an_arbitrary_action():
    with pytest.raises(ValueError, match='legal candidate'):
        NextHopActor().sample_action(torch.zeros(1,2,CANDIDATE_FEATURE_DIM), torch.zeros(1,2,dtype=torch.bool))


def test_global_critic_supports_two_context_fields():
    assert CentralizedLatentValueCritic()(torch.zeros(2,3,64), torch.zeros(2,4)).shape == (2,1)


def test_lambda_return_retains_bootstrap_at_truncated_horizon():
    actual = lambda_return(torch.tensor([-2.0]), torch.tensor([-3.0]), terminal=torch.tensor([False]))
    torch.testing.assert_close(actual, torch.tensor([-5.0]))
    terminal = lambda_return(torch.tensor([-2.0]), torch.tensor([-3.0]), terminal=torch.tensor([True]))
    torch.testing.assert_close(terminal, torch.tensor([-2.0]))


def test_policy_samples_for_training_and_uses_argmax_for_evaluation(monkeypatch):
    actor = NextHopActor()
    logits = torch.tensor([[2., 0., -torch.inf]]).expand(512, -1)
    monkeypatch.setattr(actor, 'forward_logits', lambda features, mask: logits)
    features = torch.zeros(512, 3, CANDIDATE_FEATURE_DIM)
    mask = torch.tensor([[True, True, False]]).expand(512, -1)
    torch.manual_seed(42)
    sampled = actor.sample_action(features, mask, deterministic=False)
    assert set(sampled.candidate_index.tolist()) == {0, 1}
    chosen = actor.sample_action(features, mask, deterministic=True)
    assert chosen.candidate_index.eq(0).all()


def test_vehicle_reward_interfaces_are_removed():
    from urban_dreamer.models import policy
    assert not hasattr(policy, 'CentralizedVehicleValueCritic')
    assert not hasattr(policy, 'lambda_returns_by_vehicle_ids')


def test_individual_timeout_is_charged_once_and_only_returns_to_its_own_actions():
    from urban_dreamer.models.policy import attach_timeout_penalties
    rows = [dict(vehicle_id='failed'), dict(vehicle_id='arrived'),
            dict(vehicle_id='failed'), dict(vehicle_id='arrived')]
    attach_timeout_penalties(rows, {'failed', 'never_departed'})
    assert [row['timeout_penalty'] for row in rows] == [0., 0., -5000., 0.]
    assert [row['timeout_return'] for row in rows] == [-5000., 0., -5000., 0.]
    assert sum(row['timeout_penalty'] for row in rows) == -5000.
    attach_timeout_penalties(rows, set())
    assert all(row['timeout_penalty'] == row['timeout_return'] == 0. for row in rows)


def test_individual_timeout_does_not_change_arrived_advantage_normalization():
    from urban_dreamer.models.policy import individual_actor_advantage
    global_advantages = torch.tensor([-.4, .3, -.2, .1])
    original = individual_actor_advantage(global_advantages, torch.zeros(4))
    penalized = individual_actor_advantage(global_advantages, torch.tensor([-5000., 0., -5000., 0.]))
    torch.testing.assert_close(original[[1, 3]], penalized[[1, 3]], rtol=0., atol=0.)
    assert (penalized[[0, 2]] == -2.5).all()
