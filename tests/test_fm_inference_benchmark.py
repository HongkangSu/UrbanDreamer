"""Physical units and multi-step reference accounting for Euler inference probes."""
import pytest
import torch
from tools.benchmark_fm_inference import quality, PROFILES


def record(offset=0.):
    decoded=torch.zeros(1,2,8)
    decoded[...,1]=offset
    decoded[...,2]=offset
    return dict(latent=torch.ones(3,1,2,64)+offset,variance=torch.ones(1,2)*(1+offset),
                decoded=decoded,readout=torch.ones(3,1)+offset)


def test_quality_compares_each_step_in_physical_units():
    rows=quality([record(.1),record(.2)],[record(),record()],torch.tensor([10.,20.]),torch.tensor([5.,15.]))
    assert [r['step'] for r in rows]==[1,2]
    assert rows[0]['count_mae_vs_reference']==pytest.approx(1.5)
    assert rows[1]['speed_mae_vs_reference']==pytest.approx(2.)
    assert rows[0]['variance_ratio']==pytest.approx(1.1)
    assert rows[0]['latent']['finite']
    assert rows[1]['readout']['relative_l2']==pytest.approx(.2)


def test_euler_optimization_profiles_keep_fp32_reference_separate():
    assert PROFILES['baseline']==(16,32768,16384,'fp32',False)
    assert PROFILES['compiled'][-2:]==('fp32',True)
    assert PROFILES['bf16'][-2:]==('bf16',False)


def test_quality_rejects_shortened_or_empty_rollouts():
    with pytest.raises(ValueError,match="same nonempty"):
        quality([record()],[record(),record()],torch.ones(2),torch.ones(2))
    with pytest.raises(ValueError,match="same nonempty"):
        quality([],[],torch.ones(2),torch.ones(2))
