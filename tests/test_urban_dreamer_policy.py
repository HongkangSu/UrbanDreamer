import pytest
import torch

from urban_dreamer.models.policy import (
    CANDIDATE_FEATURE_DIM, VEHICLE_OBS_DIM, NextHopActor,
    CentralizedVehicleValueCritic, CentralizedLatentValueCritic,
    lambda_return, lambda_returns_by_vehicle_ids,
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


def test_critics_support_independent_vehicle_state_and_two_context_fields():
    assert CentralizedVehicleValueCritic()(torch.zeros(2,VEHICLE_OBS_DIM)).shape == (2,1)
    assert CentralizedLatentValueCritic()(torch.zeros(2,3,64), torch.zeros(2,2)).shape == (2,1)


def test_lambda_return_retains_bootstrap_at_truncated_horizon():
    actual = lambda_return(torch.tensor([-2.0]), torch.tensor([-3.0]), terminal=torch.tensor([False]))
    torch.testing.assert_close(actual, torch.tensor([-5.0]))
    terminal = lambda_return(torch.tensor([-2.0]), torch.tensor([-3.0]), terminal=torch.tensor([True]))
    torch.testing.assert_close(terminal, torch.tensor([-2.0]))


def test_vehicle_returns_do_not_mix_interleaved_event_sequences():
    returns = lambda_returns_by_vehicle_ids(torch.tensor([1.,100.,2.,3.,200.]), torch.tensor([5.,50.,6.,7.,80.]), torch.tensor([True,False,True,True,False]), torch.tensor([0,1,0,0,1]), terminal=torch.tensor([False,True,False,True,True]))
    torch.testing.assert_close(returns, torch.tensor([6.1425,100.,5.15,3.,200.]))
