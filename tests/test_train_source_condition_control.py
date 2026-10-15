import json
import numpy as np
import pytest
import torch

from tools.train_source_condition_control import gaussian_training_noise, install_prior
from urban_dreamer.models.world_model import Dynamics
from tools.source_condition_controls import StandardGaussianPrior


def test_gaussian_cache_uses_full_independent_draws_and_training_rms(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();output=tmp_path/'output';output.mkdir()
    schedule=[dict(draws=[dict(noise_seed=17),dict(noise_seed=29)])]
    (parent/'draw_schedule.json').write_text(json.dumps(schedule))
    z=np.ones((2,1,3,64),dtype=np.float32)*100
    arrays=dict(z=z,target=np.ones_like(z)*2,
                example_window=np.array([1,0]),time=np.array([[.2],[.7]],dtype=np.float32))
    noise,audit=gaussian_training_noise(parent,output,arrays)
    expected=torch.stack([torch.randn(1,3,64,generator=torch.Generator().manual_seed(s)) for s in (17,29)]).numpy()
    np.testing.assert_array_equal(noise,expected)
    relative=(1-arrays['time'][:,:,None,None])*expected+arrays['time'][:,:,None,None]*2
    assert audit['relative']==pytest.approx(np.sqrt(np.square(relative.astype(np.float64)).mean()))
    assert noise.shape==(2,1,3,64) and len(audit['sha256'])==64
    with pytest.raises(ValueError,match='preserve existing'):
        gaussian_training_noise(parent,output,arrays)


@pytest.mark.parametrize('arm',['A','G'])
def test_no_background_variants_do_not_require_fitted_prior_files(arm,tmp_path):
    class Model:pass
    model=Model()
    model.dynamics=Dynamics(adjacency=torch.eye(3),static_features=torch.ones(3,5),low_frequency_rank=2)
    native=model.dynamics.graph_prior
    install_prior(model,arm,tmp_path/'missing_basis',tmp_path/'missing_head')
    if arm=='G':
        assert isinstance(model.dynamics.graph_prior,StandardGaussianPrior)
        assert model.dynamics.graph_prior.native is native
        assert model.dynamics.graph_prior.rank==3
    else:
        assert model.dynamics.graph_prior.zero_mean
        assert model.dynamics.graph_prior.rank==2
