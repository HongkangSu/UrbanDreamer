import copy
from dataclasses import replace

import torch

from tools.pretrain_tail_execution import enable_pretrain_tail_execution
from urban_dreamer.models.world_model import ConditionEncoding


def test_tail_execution_preserves_outputs_loss_gradients_and_sample_count(tmp_path):
    from test_urban_dreamer_training import make_trainer

    torch.set_num_threads(1)
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, resources=replace(trainer.config.resources,
        world_model_initialization_batch_size=64))
    with torch.no_grad():
        trainer.world_model.dynamics.flow_head.weight.normal_(std=.01)
        trainer.world_model.dynamics.condition_residual.output.weight.normal_(std=.01)
    reference = copy.deepcopy(trainer.world_model.dynamics)
    enable_pretrain_tail_execution(trainer, max_batch=32)
    edges = reference.n_edges
    torch.manual_seed(27)
    action = torch.zeros(63, 1, edges)
    latent = torch.randn(63, 1, edges, 64)
    flow_time = torch.rand(63, 1)
    condition = ConditionEncoding(torch.randn(63, 1, edges, 256),
        reference_tokens=torch.randn(63, 1, edges, 256),
        current_state=torch.randn_like(latent), background=torch.randn_like(latent))
    expected = reference(action, packed_latents=latent, flow_time=flow_time, condition=condition).latent_velocity
    actual = trainer.world_model.dynamics(action, packed_latents=latent, flow_time=flow_time, condition=condition).latent_velocity
    assert actual.shape[0] == 63
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    expected_loss, actual_loss = expected.square().mean(), actual.square().mean()
    torch.testing.assert_close(actual_loss, expected_loss)
    expected_loss.backward()
    actual_loss.backward()
    for (name, a), (_, b) in zip(reference.named_parameters(), trainer.world_model.dynamics.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6, msg=name)
