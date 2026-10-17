"""Global reward direction and full-batch gradients survive bounded updates."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines.common import policy_gradient_step


def step(parameter, advantages, *, normalize=True, batch_size=None, choices=None):
    rows = [dict(action=i % 2, mask=np.array([True, True])) for i in range(len(advantages))]
    if choices is not None:
        for row, choice in zip(rows, choices):
            row['mask'] = np.array([True, bool(choice)])
            if not choice:
                row['action'] = 0
    optimizer = torch.optim.SGD([parameter], lr=.1)
    seen = []
    def logits(batch):
        seen.append(len(batch))
        mask = torch.as_tensor(np.stack([r['mask'] for r in batch]))
        return parameter.expand(len(batch), -1).masked_fill(~mask, -torch.inf)
    result = policy_gradient_step(compute_logits=logits, actor_parameters=[parameter],
        actor_optimizer=optimizer, rows=rows, advantages=advantages, entropy_coef=0.,
        include_individual_timeout=False, normalize_global_advantages=normalize,
        minibatch_size=batch_size)
    return result, seen


def test_global_advantage_common_offset_does_not_erase_relative_signal():
    ordinary = torch.nn.Parameter(torch.zeros(2))
    biased = torch.nn.Parameter(torch.zeros(2))
    step(ordinary, [1., -1.])
    step(biased, [-10., -12.])
    torch.testing.assert_close(ordinary, biased)
    assert biased[0] > 0 and biased[1] < 0


def test_global_scale_invariance_and_forced_rows():
    ordinary = torch.nn.Parameter(torch.zeros(2))
    scaled = torch.nn.Parameter(torch.zeros(2))
    forced = torch.nn.Parameter(torch.zeros(2))
    step(ordinary, [2., -1.])
    step(scaled, [200., -100.])
    step(forced, [2., -1., -1000., 1000.], choices=[1, 1, 0, 0])
    torch.testing.assert_close(ordinary, scaled)
    torch.testing.assert_close(ordinary, forced)


def test_single_choice_sample_preserves_its_baseline_signal():
    parameter = torch.nn.Parameter(torch.zeros(2))
    step(parameter, [-10.])
    assert parameter[0] < 0 and parameter[1] > 0


@pytest.mark.parametrize('normalize', [False, True])
def test_unequal_minibatches_accumulate_one_full_batch_gradient(normalize):
    full = torch.nn.Parameter(torch.tensor([.3, -.2]))
    chunked = torch.nn.Parameter(full.detach().clone())
    advantages = [1., -2., .3, 4., -.4, -1., 2.]
    a, _ = step(full, advantages, normalize=normalize)
    b, sizes = step(chunked, advantages, normalize=normalize, batch_size=3)
    torch.testing.assert_close(full, chunked)
    assert b['actor_loss'] == pytest.approx(a['actor_loss'], abs=1e-7)
    assert sizes == [3, 3, 1]


def test_constant_global_batch_does_not_apply_stale_adam_momentum():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([parameter], lr=.1)
    rows = [dict(action=0, mask=np.ones(2, bool)), dict(action=1, mask=np.ones(2, bool))]
    kwargs = dict(compute_logits=lambda batch: parameter.expand(len(batch), -1),
        actor_parameters=[parameter], actor_optimizer=optimizer, rows=rows,
        entropy_coef=0., include_individual_timeout=False, normalize_global_advantages=True)
    policy_gradient_step(**kwargs, advantages=[1., -1.])
    before = parameter.detach().clone()
    saved = deepcopy(optimizer.state_dict())
    policy_gradient_step(**kwargs, advantages=[-10., -10.])
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict()['state'][0]['step'], saved['state'][0]['step'])
