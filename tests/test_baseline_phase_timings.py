"""Wall timings partition work without adding device synchronization or changing learning."""
import copy

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.dreamer import DreamerMethod
from urban_dreamer.learning.baselines.mbpo import MbpoMethod
from test_toy_baselines_model_based import make_args, synthetic_window, active_snapshot


@pytest.mark.parametrize('method_class', [DreamerMethod, MbpoMethod])
def test_empty_update_has_zero_phase_timings(tmp_path, method_class):
    runner = common.EpisodeRunner(make_args(tmp_path))
    method = method_class(runner.args, runner)
    metrics = method.on_episode([], [], episode=1)
    assert {key: value for key, value in metrics.items() if key.endswith('_seconds')} == {
        'world_model_seconds': 0., 'imagination_seconds': 0., 'actor_update_seconds': 0.,
        'critic_update_seconds': 0., 'learning_seconds': 0.}


def test_mbpo_update_phase_wall_clock_boundaries(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.mbpo as mbpo
    runner = common.EpisodeRunner(make_args(tmp_path, method='mbpo'))
    method = MbpoMethod(runner.args, runner)
    clock = iter(range(10))
    monkeypatch.setattr(mbpo.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(method, '_train_model', lambda: {'model_nll': 3.})
    monkeypatch.setattr(mbpo, 'global_critic_update', lambda **kw: (torch.zeros(1), {}))
    monkeypatch.setattr(method, '_generate_model_rows', lambda windows: 7)
    monkeypatch.setattr(method, '_policy_updates', lambda rows, adv: {'actor_loss': 2.})
    window = synthetic_window(runner, np.random.default_rng(0), time=0., terminal=False)
    metrics = method.on_episode([], [window], episode=1)
    assert metrics['world_model_seconds'] == metrics['imagination_seconds'] == 1.
    assert metrics['critic_update_seconds'] == metrics['actor_update_seconds'] == 1.
    assert metrics['learning_seconds'] == 9.
    assert metrics['imagined_events'] == 7 and metrics['actor_loss'] == 2.


def test_dreamer_pass_timings_sum_across_passes(tmp_path, monkeypatch):
    import urban_dreamer.learning.baselines.dreamer as dreamer
    runner = common.EpisodeRunner(make_args(tmp_path))
    method = DreamerMethod(runner.args, runner)
    window = synthetic_window(runner, np.random.default_rng(0), time=0., terminal=False,
                              vehicles=(active_snapshot(runner),))
    monkeypatch.setattr(method, '_train_world_model', lambda: {'world_model_loss': 3.})
    monkeypatch.setattr(method, '_imagination_pass', lambda: dict(latent_critic_loss=1., actor_loss=2.,
        imagined_events=4., imagination_seconds=5., actor_update_seconds=6., critic_update_seconds=7.))
    clock = iter(range(4))
    monkeypatch.setattr(dreamer.time, 'monotonic', lambda: next(clock))
    metrics = method.on_episode([], [window], episode=1)
    assert metrics['world_model_seconds'] == 1. and metrics['learning_seconds'] == 3.
    assert metrics['imagined_events'] == 4.  # historical mean across passes is retained
    assert metrics['imagination_seconds'] == dreamer.IMAGINATION_PASSES * 5.
    assert metrics['actor_update_seconds'] == dreamer.IMAGINATION_PASSES * 6.
    assert metrics['critic_update_seconds'] == dreamer.IMAGINATION_PASSES * 7.


@pytest.mark.parametrize('method_class', [DreamerMethod, MbpoMethod])
@pytest.mark.parametrize('tensor_advantages', [False, True])
def test_unused_flow_removal_preserves_exact_actor_update(tmp_path, method_class, tensor_advantages):
    runner = common.EpisodeRunner(make_args(tmp_path))
    original = method_class(runner.args, runner)
    optimized = copy.deepcopy(original)
    rng = np.random.default_rng(50)
    rows = [dict(features=rng.random((3, 10), dtype=np.float32), mask=np.array([True, True, False]),
                 action=i % 2, timeout_return=-float(i)) for i in range(5)]
    legacy_rows = [dict(row, flow=runner.network.extract_flow({}).copy()) for row in rows]
    advantages = [-.2, .5, -.7, .1, -.3]
    def update(method, records, advantage_values):
        return common.policy_gradient_step(compute_logits=method._compute_logits,
            actor_parameters=method.actor.parameters(), actor_optimizer=method.actor_optimizer,
            rows=records, advantages=advantage_values)
    before_rng = torch.get_rng_state().clone()
    old = update(original, legacy_rows, advantages)
    new = update(optimized, rows, torch.tensor(advantages) if tensor_advantages else advantages)
    assert old == new
    assert torch.equal(before_rng, torch.get_rng_state())
    for old_parameter, new_parameter in zip(original.actor.parameters(), optimized.actor.parameters()):
        assert torch.equal(old_parameter, new_parameter)
        if old_parameter.grad is None:
            # The stable score gauge omits the unidentifiable common head bias.
            assert new_parameter.grad is None
        else:
            assert torch.equal(old_parameter.grad, new_parameter.grad)
