import numpy as np
import torch

from tools.diagnose_noise_training import make_schedule, paired_rng, identity


def test_paired_rng_independent_of_global_rng_and_arm():
    first=paired_rng(11,3,4,3,16)
    torch.manual_seed(999);torch.randn(100)
    second=paired_rng(11,3,4,3,16)
    assert all(torch.equal(a,b) for a,b in zip(first,second))
    assert not torch.equal(first[1],paired_rng(11,4,4,3,16)[1])


def test_paired_schedule_reuses_identical_training_rows():
    entries=[dict(windows=20,path=str(i),content_hash=str(i),cache_content_hash=str(i),
                  demand_group=str(i),method='toy') for i in range(6)]
    first=make_schedule(entries,5,4,3,42)
    np.random.seed(9);np.random.randn(100)
    second=make_schedule(entries,5,4,3,42)
    assert first==second
    assert all(len(set(row['starts']))==4 for row in first)
    assert all(0<=start<18 for row in first for start in row['starts'])


def test_plan_identity_does_not_depend_on_dictionary_insertion_order():
    assert identity(dict(a=1,b=2))==identity(dict(b=2,a=1))


def test_spectral_noise_scale_is_per_coefficient_not_per_road():
    basis=torch.linalg.qr(torch.randn(11,3,dtype=torch.float64)).Q
    actual=(.1*basis).square().sum(-1).mean()
    assert torch.allclose(actual,torch.tensor(.1**2*3/11,dtype=torch.float64),atol=1e-15,rtol=1e-12)
    assert .1*(16/853)**.5<.014
