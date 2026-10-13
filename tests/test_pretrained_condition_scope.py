from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.evaluate_pretrained_condition_scope import (
    FIELDS, assert_reference, paired_summary, parser, run, score,
)
from tools.evaluate_pretrained_distribution import trajectory_bank
from urban_dreamer.learning.action_events import pack_action_events


def row(index, group, arm, value):
    return dict(condition_index=index, condition_seed=index+10,
                demand_group=group, episode='episode', control_scope=arm,
                **{field: value for field in FIELDS})


def test_reference_reproduction_rejects_wrong_shape_nonfinite_and_changed_bank():
    expected = np.ones((8, 1, 2, 2))
    assert assert_reference(expected.copy(), expected)['bit_identical']
    assert not assert_reference(expected+1e-6, expected)['bit_identical']
    with pytest.raises(ValueError, match='shape/finiteness'):
        assert_reference(expected[:4], expected)
    with pytest.raises(ValueError, match='shape/finiteness'):
        assert_reference(expected*np.nan, expected)
    with pytest.raises(AssertionError):
        assert_reference(expected+.01, expected)


def test_paired_bootstrap_weights_demands_equally_and_pairs_before_aggregation():
    rows = [row(0,'a','c0',2), row(0,'a','full_factual',1),
            row(1,'a','c0',4), row(1,'a','full_factual',3),
            row(2,'b','c0',10), row(2,'b','full_factual',12)]
    summary = paired_summary(rows)
    field = FIELDS[0]
    assert summary['arms']['c0'][field]['estimate'] == 6.5
    assert summary['full_minus_c0'][field]['estimate'] == .5
    assert summary['full_minus_c0'][field]['demand_groups'] == 2
    assert summary == paired_summary(rows)


def test_pairing_fails_if_arm_missing_duplicated_or_identity_differs():
    a = row(0, 'a', 'c0', 1)
    b = row(0, 'a', 'full_factual', 2)
    with pytest.raises(ValueError, match='both paired'):
        paired_summary([a])
    with pytest.raises(ValueError, match='duplicate'):
        paired_summary([a, a, b])
    with pytest.raises(ValueError, match='identity'):
        paired_summary([a, dict(b, demand_group='b')])
    with pytest.raises(ValueError, match='availability'):
        paired_summary([a, dict(b, **{FIELDS[0]:None})])


def test_scoring_masks_derive_from_same_truth_not_control_arm():
    target = np.array([[[1.,2.],[0.,3.]]])
    draws = np.repeat(np.array([[[[1.,2.],[9.,8.]]]]), 8, 0)
    details = dict(condition_index=0,demand_group='g',episode='e')
    first = score(draws, target, [1.,1.], 1, np.array([20.,20.]), details)
    second = score(draws+.5, target, [1.,1.], 1, np.array([20.,20.]), details)
    assert first['count_mae_occupied'] == 0
    assert second['count_mae_occupied'] == .5
    assert first['count_mae_all'] == 4.5
    assert first['count_crps_u_occupied'] == 0


class ControlToy:
    def __init__(self):
        self.dynamics = SimpleNamespace(graph_prior=SimpleNamespace(rank=1))
        self.noises = []

    def transition(self, current, actions, *, noise, action_events, steps, solver):
        self.noises.append(noise.clone())
        return current + noise + actions[...,None]

    def decode(self, latent):
        result = latent.new_zeros((*latent.shape[:-1], 4))
        result[...,2] = latent[...,0]
        result[...,1] = latent[...,1]
        return result


def test_control_intervention_keeps_identical_draw_noise_and_initial_state():
    initial = torch.zeros(1,2,2)
    common = dict(initial=initial, events=pack_action_events([[()]]), seed=123,
        capacities=torch.ones(2),limits=torch.ones(2), samples=8,chunk=4,horizons=(1,),steps=100)
    c0, full = ControlToy(), ControlToy()
    left = trajectory_bank(c0, actions=torch.zeros(1,1,2), **common)[1]
    right = trajectory_bank(full, actions=torch.ones(1,1,2), **common)[1]
    assert torch.count_nonzero(initial) == 0
    for a,b in zip(c0.noises,full.noises,strict=True):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    np.testing.assert_allclose(right-left,1,atol=2e-7)


def arguments(tmp_path):
    return parser().parse_args(['--experiment-dir',str(tmp_path/'experiment'),
        '--reference-dir',str(tmp_path/'reference'),'--runtime-manifest',str(tmp_path/'manifest'),
        '--output-dir',str(tmp_path/'output')])


def test_existing_output_is_never_overwritten(tmp_path):
    args = arguments(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(ValueError,match='preserve'):
        run(args)


def test_undertrained_checkpoint_rejected_before_gpu_or_runtime_loading(tmp_path):
    args = arguments(tmp_path)
    checkpoint = args.experiment_dir/'snapshot/checkpoints/latest.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'undertrained')
    with pytest.raises(ValueError,match='completed pretrained'):
        run(args)
    assert not args.output_dir.exists()
