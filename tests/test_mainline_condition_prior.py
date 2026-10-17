"""The shipped configuration uses the same conditional flow formula as study B."""
import torch

from test_urban_dreamer_training import make_trainer
from tools.source_condition_controls import source_context
from tools.train_direct_context_fm import DirectResidual
from urban_dreamer.models.world_model import Dynamics


def test_mainline_trainer_constructs_and_optimizes_condition_only_background(tmp_path):
    trainer = make_trainer(tmp_path)
    dynamics = trainer.world_model.dynamics
    assert dynamics.prior_usage == 'condition_only'
    assert dynamics.graph_prior.prior_filter == 'soft'
    assert dynamics.graph_prior.prior_filter_strength == .1
    assert not dynamics.graph_prior.prior_action_conditioned
    assert not dynamics.graph_prior.filter_input
    wanted = {id(p) for p in dynamics.condition_residual.parameters()}
    for name in ('world_model_initial', 'world_model_online'):
        actual = {id(p) for g in trainer.optimizers[name].param_groups for p in g['params']}
        assert wanted <= actual
    state = trainer.state_dict()
    clone = make_trainer(tmp_path/'restored')
    clone.load_state_dict(state)
    for name, value in dynamics.state_dict().items():
        torch.testing.assert_close(value, clone.world_model.dynamics.state_dict()[name], rtol=0, atol=0)


def test_native_condition_path_matches_experimental_B_transition_and_disagreement():
    torch.manual_seed(191)
    kwargs = dict(adjacency=torch.tensor([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]]),
                  static_features=torch.ones(3, 5), low_frequency_rank=2,
                  prior_filter='soft', prior_filter_strength=.1, prior_action_conditioned=False)
    legacy = Dynamics(**kwargs).eval()
    native = Dynamics(**kwargs, prior_usage='condition_only').eval()
    head = DirectResidual().eval()
    with torch.no_grad():
        legacy.graph_prior.output.weight.normal_(std=.03)
        legacy.flow_head.weight.normal_(std=.02)
        head.output.weight.normal_(std=.01)
        state = native.state_dict()
        for name,value in legacy.state_dict().items():state[name].copy_(value)
        native.condition_residual.load_state_dict(head.state_dict())
        native.condition_rms_z.fill_(1.1)
        native.condition_rms_background.fill_(.1)
        native.condition_rms_relative.fill_(.12)
    scales = dict(z=1.1,b=.1,relative=.12)
    z = torch.randn(2, 3, 64)
    action = torch.zeros(2, 3)
    noises = torch.randn(3, 2, 2, 64)
    expected = []
    with torch.no_grad(), source_context(legacy, head, 'B', scales):
        for noise in noises:
            expected.append(legacy.transition(z, action, noise=noise, steps=4))
    with torch.no_grad():
        actual, _, _ = native.latent_disagreement(z, action, n_samples=3, noise=noises, steps=4)
    torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-5, atol=2e-6)
