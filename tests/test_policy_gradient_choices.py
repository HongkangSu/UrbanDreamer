"""Forced routing actions must not dilute or move the actor optimizer."""
import numpy as np
import torch

from urban_dreamer.learning.baselines.common import policy_gradient_step


def update(parameter, optimizer, rows, advantages):
    def logits(batch):
        mask = torch.as_tensor(np.stack([row['mask'] for row in batch]))
        return parameter.expand(len(batch), -1).masked_fill(~mask, -torch.inf)
    return policy_gradient_step(compute_logits=logits, actor_parameters=[parameter],
        actor_optimizer=optimizer, rows=rows, advantages=torch.tensor(advantages),
        entropy_coef=0., include_individual_timeout=False)


def test_forced_episode_does_not_apply_stale_adam_momentum():
    parameter = torch.nn.Parameter(torch.tensor([0., 0.]))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    update(parameter, optimizer, [dict(action=0, mask=[True, True])], [1.])
    before = parameter.detach().clone()
    step = optimizer.state[parameter]['step'].clone()
    metrics = update(parameter, optimizer, [dict(action=0, mask=[True, False])], [1.])
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[parameter]['step'], step)
    assert metrics['actor_loss'] == 0.


def test_forced_rows_do_not_dilute_choice_gradient_or_advantage_scale():
    choice = dict(action=0, mask=[True, True])
    forced = dict(action=0, mask=[True, False])
    alone = torch.nn.Parameter(torch.zeros(2))
    mixed = torch.nn.Parameter(torch.zeros(2))
    update(alone, torch.optim.SGD([alone], lr=.1), [choice], [1.])
    update(mixed, torch.optim.SGD([mixed], lr=.1), [choice, forced, forced], [1., -100., 100.])
    torch.testing.assert_close(alone, mixed)


def test_zero_objective_preserves_adam_state_after_learning():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=.01)
    rows = [dict(action=0, mask=[True, True])]
    update(parameter, optimizer, rows, [1.])
    before = parameter.detach().clone()
    step = optimizer.state[parameter]['step'].clone()
    metrics = update(parameter, optimizer, rows, [0.])
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state[parameter]['step'], step)
    assert metrics['actor_loss'] == 0.
