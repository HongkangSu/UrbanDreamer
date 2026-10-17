import importlib.util
from pathlib import Path
import numpy as np
import pytest
SPEC=importlib.util.spec_from_file_location('tau_eval',Path(__file__).resolve().parents[1]/'tools/evaluate_tau_condition.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

def test_soft_filter_preserves_constant_and_smoothly_attenuates_all_other_modes():
    q,_=np.linalg.qr(np.random.default_rng(7).normal(size=(5,5)))
    eig=np.arange(5,dtype=float)
    np.testing.assert_allclose(m.soft_operator(q,eig,0),np.eye(5),atol=1e-14)
    for tau in [.01,.1,1,100]:
        op=m.soft_operator(q,eig,tau)
        np.testing.assert_allclose(op@q[:,0],q[:,0],atol=1e-14)
        gains=np.diag(q.T@op@q)
        np.testing.assert_allclose(gains,1/(1+tau*eig),atol=1e-14)
        assert np.all(gains>0)

@pytest.mark.parametrize('tau',[-1,np.inf,np.nan])
def test_invalid_tau_rejected(tau):
    with pytest.raises(ValueError):m.soft_operator(np.eye(2),np.arange(2),tau)

def test_validation_selection_never_requires_test_fields():
    study=m.ROOT/'outputs/source_condition_controls_20261015/v1'
    split=m.read(study/'split.json')
    safe={k:v for k,v in split.items() if 'test' not in k}
    conditions=m.validation_conditions(safe)
    assert len(conditions)==32
    assert len({x['demand_group'] for x in conditions})==16
    assert all(x['role']=='validation' for x in conditions)

def test_tau_changes_only_background_and_keeps_graph_noise_identical():
    import torch
    from urban_dreamer.models.world_model import LowFrequencyGraphPrior
    torch.manual_seed(4)
    graph=torch.tensor([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])
    native=LowFrequencyGraphPrior(graph,rank=2)
    with torch.no_grad():native.output.weight.normal_()
    identity=m.MeanOverridePrior(native,operator=np.eye(3),state_only=True)
    smooth=m.MeanOverridePrior(native,operator=np.ones((3,3))/3,state_only=True)
    z=torch.randn(2,1,3,64);noise=torch.randn(2,1,2,64)
    torch.testing.assert_close(identity.sample_noise(z,noise),smooth.sample_noise(z,noise),rtol=0,atol=0)
    assert not torch.allclose(identity(z,None),smooth(z,None))
    assert identity.filter_input is False and smooth.filter_input is False
