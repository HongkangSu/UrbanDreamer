import numpy as np
import pytest
import torch

from tools.evaluate_pretrained_prior_ranks import (
    aggregate_groups, frozen_hidden, metric_sums, solve_ridge, spectral_design, subspace_geometry, summarize,
)
from urban_dreamer.models.world_model import ConditionEncoding, LowFrequencyGraphPrior


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_native_prior_algebra_and_projected_bias(dtype):
    torch.manual_seed(4)
    prior = LowFrequencyGraphPrior(torch.ones(5, 5)-torch.eye(5), rank=2).to(dtype=dtype)
    with torch.no_grad():
        prior.output.weight.normal_()
        prior.output.bias.normal_()
    z = torch.randn(3, 1, 5, 64, dtype=dtype)
    condition = ConditionEncoding(torch.randn(3, 1, 5, 256, dtype=dtype))
    hidden = frozen_hidden(prior, z, condition)
    torch.testing.assert_close(prior(z, condition), prior.project(prior.output(hidden)))
    x, y = spectral_design(hidden[:,0].detach().numpy(), np.zeros((3,5,64)), prior.basis.numpy())
    weights = np.concatenate((prior.output.weight.detach().numpy().T, prior.output.bias.detach().numpy()[None]))
    expected = prior.basis.numpy() @ (x @ weights)
    np.testing.assert_allclose(prior(z, condition)[:,0].detach().numpy(), expected, atol=2e-5, rtol=2e-5)
    assert x.dtype == y.dtype == np.float64


def test_closed_form_head_is_ridge_optimum_and_streamable():
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(100,5)), rng.normal(size=(100,3))
    gram = sum(chunk.T@chunk for chunk in np.array_split(x,4))
    cross = sum(a.T@b for a,b in zip(np.array_split(x,4),np.array_split(y,4)))
    w, proof = solve_ridge(gram,cross,np.square(y).sum())
    direct = np.linalg.lstsq(np.vstack((x,np.sqrt(proof['ridge'])*np.eye(5))),
        np.vstack((y,np.zeros((5,3)))),rcond=None)[0]
    np.testing.assert_allclose(w,direct,atol=1e-13)
    np.testing.assert_allclose((gram+np.eye(5)*proof['ridge'])@w,cross,atol=1e-13)
    assert proof['normal_equation_relative_residual'] < 1e-12
    assert proof['training_projected_sse'] == pytest.approx(np.square(x@w-y).sum())


def test_demand_group_aggregation_combines_episodes_before_bootstrap():
    records = [dict(split='validation',arm='refit_output',rank=8,demand_group=group,episode=episode,
        windows=1,total_delta_energy=energy,captured_energy=energy/2,full_sse=energy*.8,
        projected_sse=energy*.3,prediction_energy=energy*.1,dot_product=energy*.15,
        count_absolute_error=1.,count_elements=2,count_occupied_absolute_error=1.,count_occupied_elements=1,
        speed_absolute_error=2.,speed_elements=2,speed_occupied_absolute_error=2.,speed_occupied_elements=1)
        for group,episode,energy in [('a',1,10.),('a',2,20.),('b',3,100.)]]
    groups=aggregate_groups(records)
    assert len(groups)==2
    assert next(r for r in groups if r['demand_group']=='a')['total_delta_energy']==30
    _,summary=summarize(records,repetitions=100)
    assert summary[0]['groups']==2
    assert summary[0]['windows']==3
    assert summary[0]['full_normalized_sse']==pytest.approx(.8)
    assert summary[0]['captured_energy_fraction']==pytest.approx(.5)


def test_nested_projection_energy_bound_and_full_decomposition():
    rng=np.random.default_rng(5)
    basis=np.linalg.qr(rng.normal(size=(7,7)))[0]
    hidden,delta=rng.normal(size=(2,7,4)),rng.normal(size=(2,7,4))
    x,y=spectral_design(hidden,delta,basis)
    assert np.all(np.diff(np.cumsum(np.square(y).sum(-1),axis=-1),axis=-1)>=0)
    for rank in (1,3,7):
        projected=basis[:,:rank]@y[:,:rank]
        oracle=metric_sums(delta,projected,y[:,:rank])
        assert oracle['captured_energy']<=oracle['total_delta_energy']+1e-12
        assert oracle['full_sse']==pytest.approx(oracle['total_delta_energy']-oracle['captured_energy'],abs=1e-12)
        assert oracle['projected_sse']==pytest.approx(0,abs=1e-12)
        coefficients=rng.normal(size=(5,4))
        prediction=basis[:,:rank]@(x[:,:rank]@coefficients)
        metrics=metric_sums(delta,prediction,y[:,:rank])
        assert metrics['projected_sse']==pytest.approx(np.square(x[:,:rank]@coefficients-y[:,:rank]).sum())


def test_native_rank_limit_and_projection_precision():
    prior=LowFrequencyGraphPrior(torch.ones(7,7)-torch.eye(7),rank=7)
    assert prior.rank==6
    assert prior.basis.shape==(7,6)
    values=torch.randn(2,7,64,dtype=torch.float64)
    assert prior.project(values).dtype==torch.float64
    torch.testing.assert_close(prior.project(prior.project(values)),prior.project(values),atol=2e-6,rtol=2e-6)


def test_subspace_equivalence_normalizes_float32_basis_without_mutation():
    rng=np.random.default_rng(88)
    exact=np.linalg.qr(rng.normal(size=(11,3)))[0]
    approximate=(exact*1.0001).astype(np.float32)
    saved=approximate.copy()
    geometry=subspace_geometry(exact,approximate)
    assert geometry['native_subspace_overlap']==pytest.approx(1.,abs=1e-12)
    assert 0<=geometry['native_subspace_overlap']<=1
    assert geometry['native_subspace_chordal_distance_squared']>=0
    np.testing.assert_array_equal(approximate,saved)


def test_zero_energy_has_undefined_ratios_and_zero_fit():
    weights,proof=solve_ridge(np.zeros((3,3)),np.zeros((3,2)),0.)
    np.testing.assert_array_equal(weights,0)
    assert proof['training_projected_sse']==0
    assert proof['gram_condition'] is None
    metrics=metric_sums(np.zeros((1,2,3)),np.zeros((1,2,3)),np.zeros((1,1,3)))
    assert all(value==0 for value in metrics.values())


def test_nonfinite_spectral_inputs_rejected():
    with pytest.raises(ValueError,match='finite'):
        spectral_design(np.full((1,2,3),np.nan),np.zeros((1,2,3)),np.eye(2))
