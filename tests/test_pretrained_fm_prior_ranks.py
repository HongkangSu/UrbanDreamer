import numpy as np
import pytest
import torch

from tools.evaluate_pretrained_fm_prior_ranks import ARMS, METRICS, SCOPES, ProjectionRankPrior, paired_summaries
from urban_dreamer.models.world_model import ConditionEncoding, LowFrequencyGraphPrior


def fixture_prior():
    torch.manual_seed(76)
    graph=torch.ones(70,70)-torch.eye(70)
    prior=LowFrequencyGraphPrior(graph,rank=16,noise_scale=.15).eval()
    with torch.no_grad():
        prior.output.weight.normal_(std=.1)
        prior.output.bias.normal_(std=.1)
    basis=torch.linalg.qr(torch.randn(70,70)).Q
    return prior,basis


def test_noop_prior_matches_native_state_dict_and_forward_bitwise():
    prior,_=fixture_prior()
    cloned=ProjectionRankPrior(prior)
    assert cloned.rank==cloned.prediction_rank==16
    assert cloned.state_dict().keys()==prior.state_dict().keys()
    assert all(torch.equal(v,cloned.state_dict()[k]) for k,v in prior.state_dict().items())
    assert cloned.projection_basis is None
    z=torch.randn(2,1,70,64)
    condition=ConditionEncoding(torch.randn(2,1,70,256))
    assert torch.equal(prior(z,condition),cloned(z,condition))
    assert not any(p.requires_grad for p in cloned.parameters())


@pytest.mark.parametrize('rank',[32,64])
def test_projection_override_keeps_exact_native_noise_coordinates(rank):
    prior,basis=fixture_prior()
    alternative=ProjectionRankPrior(prior,basis[:,:rank])
    z=torch.randn(3,1,70,64)
    coefficient=torch.randn(3,1,16,64)
    assert alternative.rank==16
    assert alternative.prediction_rank==rank
    assert torch.equal(alternative.sample_noise(z,coefficient),prior.sample_noise(z,coefficient))
    assert torch.equal(alternative.basis,prior.basis)
    assert alternative.noise_scale==prior.noise_scale
    assert alternative.state_dict().keys()==prior.state_dict().keys()
    with pytest.raises(ValueError,match='low-frequency coefficient shape'):
        alternative.sample_noise(z,torch.randn(3,1,rank,64))


def test_nested_projector_is_psd_idempotent_and_preserves_dtype():
    prior,basis=fixture_prior()
    low=ProjectionRankPrior(prior,basis[:,:32])
    high=ProjectionRankPrior(prior,basis[:,:64])
    value=torch.randn(3,70,64,dtype=torch.float64)
    projected=high.project(value)
    assert projected.dtype==torch.float64
    torch.testing.assert_close(high.project(projected),projected,atol=2e-6,rtol=2e-6)
    assert torch.sum(value*projected)>=0
    assert torch.sum(low.project(value)**2)<=torch.sum(projected**2)
    torch.testing.assert_close(high.project(low.project(value)),low.project(value),atol=2e-6,rtol=2e-6)


def test_refit_augmented_coefficients_load_and_only_output_parameters_change():
    prior,basis=fixture_prior()
    coefficients=np.random.default_rng(7).normal(size=(65,64))
    refit=ProjectionRankPrior(prior,basis[:,:64],coefficients)
    np.testing.assert_array_equal(refit.output.weight.detach().numpy(),coefficients[:-1].T.astype(np.float32))
    np.testing.assert_array_equal(refit.output.bias.detach().numpy(),coefficients[-1].astype(np.float32))
    for key,value in prior.state_dict().items():
        if not key.startswith('output.'):
            assert torch.equal(value,refit.state_dict()[key])
    with pytest.raises(ValueError,match='65,64'):
        refit.load_output_coefficients(np.zeros((64,64)))


def test_mean_intervention_changes_coarse_mean_while_noise_is_controlled():
    prior,basis=fixture_prior()
    alternative=ProjectionRankPrior(prior,basis[:,:64])
    z=torch.randn(2,1,70,64)
    condition=ConditionEncoding(torch.randn(2,1,70,256))
    noise=torch.randn(2,1,16,64)
    native_mean=prior(z,condition)
    alternative_mean=alternative(z,condition)
    assert not torch.equal(native_mean,alternative_mean)
    native_start=z+native_mean+prior.sample_noise(z,noise)
    alternative_start=z+alternative_mean+alternative.sample_noise(z,noise)
    torch.testing.assert_close(alternative_start-native_start,alternative_mean-native_mean,atol=2e-6,rtol=2e-6)
    assert torch.equal(prior.sample_noise(z,noise),alternative.sample_noise(z,noise))


def test_paired_bootstrap_pairs_conditions_and_resamples_demand_groups():
    rows=[]
    for index,group in enumerate(('a','a','b','b')):
        for scope in SCOPES:
            for arm in ARMS:
                value=index+(scope=='full_factual')+(arm!='native16')*.25
                rows.append(dict(condition_index=index,demand_group=group,episode=group,condition_seed=index,
                    control_scope=scope,arm=arm,**{metric:value for metric in METRICS}))
    result=paired_summaries(rows,repetitions=100)
    for scope in SCOPES:
        for arm in ARMS[1:]:
            value=result['intervention_minus_native'][scope][arm]['count_mae_all']
            assert value['estimate']==pytest.approx(.25)
            assert value['demand_groups']==2
            assert value['ci_low']==value['ci_high']==pytest.approx(.25)
    assert result['full_minus_c0']['native16']['count_mae_all']['estimate']==pytest.approx(1.)
    with pytest.raises(ValueError,match='every condition'):
        paired_summaries(rows[:-1])
