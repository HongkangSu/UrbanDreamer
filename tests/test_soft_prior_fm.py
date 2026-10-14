import numpy as np
import pytest
import torch

from tools.evaluate_soft_prior_fm import MeanOverridePrior,summaries,score
from tools.component_prediction_metrics import evaluate_prediction_samples
from urban_dreamer.models.world_model import LowFrequencyGraphPrior,ConditionEncoding


def native():
    torch.manual_seed(11)
    graph=torch.tensor([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    return LowFrequencyGraphPrior(graph,rank=2,prior_filter='hard',prior_action_conditioned=True).eval()


def test_native_clone_and_all_mean_changes_preserve_original_noise():
    prior=native();z=torch.randn(4,1,3,64);noise=torch.randn(4,1,2,64)
    condition=ConditionEncoding(torch.randn(4,1,3,256))
    clone=MeanOverridePrior(prior)
    torch.testing.assert_close(prior(z,condition),clone(z,condition),rtol=0,atol=0)
    variants=[clone,MeanOverridePrior(prior,zero=True),MeanOverridePrior(prior,operator=np.eye(3),state_only=True)]
    for variant in variants:
        torch.testing.assert_close(variant.sample_noise(z,noise),prior.sample_noise(z,noise),rtol=0,atol=0)
        torch.testing.assert_close(variant.basis,prior.basis,rtol=0,atol=0)
        assert variant.noise_scale==prior.noise_scale and variant.rank==prior.rank


def test_state_head_override_is_action_invariant_and_changes_only_output_weights():
    prior=native();coefficients=np.zeros((65,64));coefficients[-1]=2
    modified=MeanOverridePrior(prior,operator=np.eye(3),coefficients=coefficients,state_only=True)
    z=torch.randn(2,1,3,64)
    a=ConditionEncoding(torch.randn(2,1,3,256));b=ConditionEncoding(a.edge_tokens*100)
    torch.testing.assert_close(modified(z,a),modified(z,b),rtol=0,atol=0)
    torch.testing.assert_close(modified(z,a),torch.full_like(z,2),rtol=0,atol=0)
    for name,param in prior.named_parameters():
        if not name.startswith('output.'):
            torch.testing.assert_close(param,dict(modified.named_parameters())[name],rtol=0,atol=0)


def test_zero_mean_still_has_nonzero_original_noise():
    prior=native();modified=MeanOverridePrior(prior,zero=True)
    z=torch.randn(1,1,3,64);condition=ConditionEncoding(torch.randn(1,1,3,256))
    assert not bool(modified(z,condition).any())
    assert bool(modified.sample_noise(z,torch.ones(1,1,2,64)).any())


def test_point_metrics_use_mean_decoded_samples_and_crps_has_pair_term():
    samples=np.array([[[[0.,0.]]],[[[2.,2.]]]])
    target=np.array([[[1.,1.]]])
    row=evaluate_prediction_samples(samples,target,scales=np.ones(2))['states'][0]
    assert row['count_mae_all']==0 and row['count_rmse_all']==0
    assert row['count_sample_mae_all']==1
    assert row['count_crps_all']==pytest.approx(.5)
    assert row['count_coverage90_all']==1


def test_invalid_override_dimensions_rejected():
    prior=native()
    with pytest.raises(ValueError):MeanOverridePrior(prior,operator=np.eye(2))
    with pytest.raises(ValueError):MeanOverridePrior(prior,coefficients=np.zeros((3,64)))


def test_summary_pools_rmse_from_mean_prediction_mse_including_paired_contrast():
    rows=[]
    for state in (0,1):
        for arm,error in [('native',state+1),('modified',state+3)]:
            draws=np.full((8,1,1,2),1.+error)
            metadata=dict(condition_index=state,arm=arm,steps=100,horizon=1,demand_group='group',episode='episode')
            rows.append(score(draws,np.ones((1,1,2)),np.ones(2),10,np.array([20.]),metadata))
    result=summaries(rows,41)
    modified=next(r for r in result['arms'] if r['arm']=='modified')
    assert modified['metrics']['count_rmse_all']['estimate']==pytest.approx(np.sqrt(12.5))
    delta=result['intervention_minus_native'][0]['metrics']['count_rmse_all']['estimate']
    assert delta==pytest.approx(np.sqrt(12.5)-np.sqrt(2.5))
