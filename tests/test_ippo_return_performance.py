"""Regression reference for asynchronous IPPO team returns."""
from collections import defaultdict

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines.common import global_reward_integral
from urban_dreamer.models.policy import DEFAULT_LAMBDA


def reference_targets(rows, values, windows, deadline):
    integrated, end = global_reward_integral(windows, deadline=deadline)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row['vehicle_id']].append(index)
    targets = torch.empty_like(values)
    for indices in groups.values():
        indices.sort(key=lambda index: float(rows[index]['time']))
        bounds = [float(rows[index]['time']) for index in indices] + [end]
        rewards = torch.tensor([integrated(b) - integrated(a)
                                for a, b in zip(bounds, bounds[1:])], device=values.device)
        # Independent scalar reference: traces decay per physical 30 seconds,
        # with no extra decay for duplicate event timestamps.
        running = rewards[-1]
        targets[indices[-1]] = running
        for offset in reversed(range(len(indices) - 1)):
            trace = values.new_tensor(DEFAULT_LAMBDA ** ((bounds[offset + 1] - bounds[offset]) / 30.))
            running = rewards[offset] + (1. - trace) * values[indices[offset + 1]] + trace * running
            targets[indices[offset]] = running
    return targets


def make_case(lengths=(1, 2, 9, 100, 2500)):
    rng = np.random.default_rng(817)
    rows = [dict(vehicle_id=vehicle, time=float(time), reward=float('nan'), timeout_return=-1e12)
            for vehicle, length in enumerate(lengths)
            for time in rng.integers(0, 3811, size=length)]
    rng.shuffle(rows)
    windows = [dict(start_time=start, end_time=min(start + 30, 3810), reward=-rng.random() * 5)
               for start in range(0, 3810, 30)]
    values = torch.tensor(rng.normal(size=len(rows)), dtype=torch.float32)
    return rows, values, windows


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bulk_targets_match_original_chronology_ties_and_long_routes(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from urban_dreamer.learning.baselines.ppo import _local_lambda_targets
    rows, values, windows = make_case()
    values = values.to(device)
    expected = reference_targets(rows, values, windows, 3810)
    values.requires_grad_(True)
    actual = _local_lambda_targets(rows, values, windows, deadline=3810)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
    assert actual.device == values.device and actual.dtype == values.dtype
    assert not actual.requires_grad


def test_empty_targets_and_complete_team_tail():
    from urban_dreamer.learning.baselines.ppo import _local_lambda_targets
    windows = [dict(start_time=0., end_time=30., reward=-30.),
               dict(start_time=30., end_time=90., reward=-120.)]
    assert _local_lambda_targets([], torch.empty(0), windows, deadline=90).numel() == 0
    rows = [dict(vehicle_id='early', time=15.), dict(vehicle_id='last', time=90.)]
    torch.testing.assert_close(_local_lambda_targets(rows, torch.tensor([999., 999.]), windows,
                                                     deadline=90), torch.tensor([-135., 0.]))


def test_bulk_integration_rejects_outside_and_nonfinite_timestamps():
    from urban_dreamer.learning.baselines.ppo import _local_lambda_targets
    windows = [dict(start_time=0., end_time=30., reward=-30.)]
    for timestamp in [-1., 31., float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='decision time outside'):
            _local_lambda_targets([dict(vehicle_id=0, time=timestamp)], torch.zeros(1), windows,
                                  deadline=30)


@pytest.mark.parametrize('legacy', [False, True])
def test_integral_bulk_matches_scalar_and_legacy_bounds(legacy):
    windows = [dict(start_time=0., end_time=30., reward=-3.),
               dict(start_time=30., end_time=45., reward=-6.)]
    if legacy:
        windows = [dict(obs=[window['start_time'] / 45, 0],
                        next_obs=[window['end_time'] / 45, 0], reward=window['reward'])
                   for window in windows]
    integrated, end = global_reward_integral(windows, deadline=45)
    times = np.array([-0.0005, 0., 7., 30., 40., end, end + 0.0005])
    np.testing.assert_array_equal(integrated(times), [integrated(time) for time in times])
    assert isinstance(integrated(7.), float)
    assert integrated(np.empty(0)).shape == (0,)
    np.testing.assert_allclose(integrated(times), [0., 0., -.7, -3., -7., -9., -9.])
