import numpy as np
import pytest

from tools.evaluate_soft_background import (spectral_moments,weighted_moments,solve_weighted,
    sufficient_sse,spectral_sums,pareto,confirmation_selection)


@pytest.mark.parametrize('supervision',['filtered','full'])
def test_spectral_weighted_regression_matches_direct_filter(supervision):
    rng=np.random.default_rng(4)
    basis,_=np.linalg.qr(rng.normal(size=(6,6)))
    gains=np.linspace(.08,1,6);operator=(basis*gains)@basis.T
    x=rng.normal(size=(8,6,4));y=rng.normal(size=(8,6,3))
    xs=np.einsum('nr,tnc->trc',basis,x);ys=np.einsum('nr,tnc->trc',basis,y)
    moments=spectral_moments(xs,ys);weighted=weighted_moments(moments,gains,supervision)
    xf=(operator@x).reshape(-1,4);yf=(operator@y if supervision=='filtered' else y).reshape(-1,3)
    np.testing.assert_allclose(weighted['gram'],xf.T@xf,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(weighted['cross'],xf.T@yf,rtol=1e-12,atol=1e-12)
    assert weighted['energy']==pytest.approx(float(np.square(yf).sum()))
    head,_,_=solve_weighted(moments,gains,supervision)
    prediction=(operator@x)@head
    target=operator@y if supervision=='filtered' else y
    assert sufficient_sse(head,weighted)==pytest.approx(float(np.square(target-prediction).sum()),rel=1e-12)
    np.testing.assert_allclose(operator@(x@head),prediction,rtol=1e-12,atol=1e-12)


def test_tau_zero_losses_and_solutions_coincide_exactly():
    rng=np.random.default_rng(3);moments=spectral_moments(rng.normal(size=(7,6,4)),rng.normal(size=(7,6,2)))
    a,_,_=solve_weighted(moments,np.ones(6),'filtered')
    b,_,_=solve_weighted(moments,np.ones(6),'full')
    np.testing.assert_array_equal(a,b)


def test_full_target_compensates_shrinkage_when_filter_is_scalar():
    rng=np.random.default_rng(8);x=rng.normal(size=(100,6,2));true=np.array([[2.],[-1.]])
    y=x@true;moments=spectral_moments(x,y);gains=np.full(6,.2)
    filtered,_,_=solve_weighted(moments,gains,'filtered')
    full,_,_=solve_weighted(moments,gains,'full')
    np.testing.assert_allclose(full,filtered/.2,rtol=1e-12)
    assert np.square(y-gains[None,:,None]*(x@full)).sum()<1e-8
    assert np.square(y-gains[None,:,None]*(x@filtered)).sum()>100


def test_non_idempotent_filter_target_loss_is_not_filtered_error_loss():
    rng=np.random.default_rng(1);target=rng.normal(size=(3,6,2));prediction=rng.normal(size=target.shape)
    sums,bands=spectral_sums(target,prediction,np.linspace(.1,.9,6),np.arange(6))
    assert sums['filtered_target_sse']!=pytest.approx(sums['filtered_error_sse'])
    assert bands.sum()==pytest.approx(sums['full_sse'])


def test_parseval_matches_spatial_error():
    rng=np.random.default_rng(19);basis,_=np.linalg.qr(rng.normal(size=(6,6)))
    target=rng.normal(size=(2,6,3));prediction=rng.normal(size=target.shape)
    spectra=[np.einsum('nr,tnc->trc',basis,a) for a in (target,prediction)]
    sums,_=spectral_sums(*spectra,np.ones(6),np.arange(6))
    assert sums['full_sse']==pytest.approx(np.square(target-prediction).sum(),rel=1e-12)


def test_pareto_preserves_tradeoffs_and_equal_points():
    rows=[dict(candidate='a',x=1,y=3),dict(candidate='b',x=2,y=2),dict(candidate='c',x=3,y=3),dict(candidate='d',x=1,y=3)]
    assert pareto(rows,['x','y'])==['a','b','d']


@pytest.mark.parametrize('bad',['other',''])
def test_invalid_supervision_rejected(bad):
    moments=spectral_moments(np.ones((2,3,2)),np.ones((2,3,1)))
    with pytest.raises(ValueError):weighted_moments(moments,np.ones(3),bad)


def test_confirmation_groups_exclude_all_previous_train_and_screen_groups():
    entries=[dict(split='validation',demand_group=str(i),cache_path=f'{i}/{j}') for i in range(10) for j in range(2)]
    entries.append(dict(split='train',demand_group='train-only',cache_path='other'))
    previous=dict(train=[dict(demand_group='0')],validation=[dict(demand_group='1')])
    a=confirmation_selection(entries,previous,6,51)
    b=confirmation_selection(entries,previous,6,51)
    assert a==b
    assert len({r['demand_group'] for r in a})==6
    assert not ({r['demand_group'] for r in a}&{'0','1','train-only'})


def test_confirmation_requires_enough_unused_groups():
    with pytest.raises(ValueError):
        confirmation_selection([],dict(train=[],validation=[]),3,1)
