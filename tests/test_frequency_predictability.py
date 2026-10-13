import numpy as np
import pytest

from tools.evaluate_frequency_predictability import augmented, band_statistics, frequency_bands, summarize


def test_frequency_bands_preserve_cutoff_eigenspace_and_cover_endpoints():
    values=np.array([-1e-12,0.,.1-1e-10,.1+1e-10,.3,1.,1.8,2.])
    bins=frequency_bands(values)
    assert bins.tolist()==[0,0,1,1,2,4,5,5]


@pytest.mark.parametrize('values',[[.3,.1],[-.1,1.],[],[float('nan')],[2.1]])
def test_frequency_bands_reject_invalid_spectra(values):
    with pytest.raises(ValueError):frequency_bands(values)


def test_band_statistics_parseval_and_mean_baseline():
    rng=np.random.default_rng(3)
    basis,_=np.linalg.qr(rng.normal(size=(6,6)))
    target=rng.normal(size=(3,6,2));prediction=target*.75;mean=target.mean(axis=0)
    stats=band_statistics(target,prediction,mean,basis,np.arange(6))
    np.testing.assert_allclose(stats[:,0].sum(axis=0),np.square(target).sum(axis=(0,1)))
    np.testing.assert_allclose(stats[:,1].sum(axis=0),np.square(target*.25).sum(axis=(0,1)))
    np.testing.assert_allclose(stats[:,2].sum(axis=0),np.square(target-mean).sum(axis=(0,1)))


def test_augmentation_only_adds_bias():
    hidden=np.arange(24).reshape(2,3,4)
    x=augmented(hidden)
    np.testing.assert_array_equal(x[...,:4],hidden)
    np.testing.assert_array_equal(x[...,4],np.ones((2,3)))


def test_skill_is_pooled_and_interval_sign_flips():
    groups=[]
    for group in range(3):
        for band in range(6):
            groups.append(dict(space='latent',band=band,modes=1,windows=1,demand_group=str(group),
                               zero_sse=group+1.,prediction_sse=.5*(group+1),mean_sse=.25*(group+1),total_energy=6*(group+1.)))
    rows=summarize(groups,7)
    assert rows[0]['skill_vs_zero']==pytest.approx(.5)
    assert rows[0]['skill_vs_train_mean']==pytest.approx(-1.)
    assert rows[0]['energy_fraction']==pytest.approx(1/6)
    assert rows[0]['skill_vs_train_mean_ci95_low']==pytest.approx(-1.)
