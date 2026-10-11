import importlib.util
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('routing_cost_observer', ROOT / 'tools/routing_cost_observer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PeriodicCostObserver = module.PeriodicCostObserver


def test_samples_physical_grid_without_intermediate_getters():
    observer = PeriodicCostObserver(30, [1, 2])
    reader = Mock(side_effect=[[3, 4], [5, 6], [7, 8]])
    sampled = []
    for tick in range(61):
        if observer.observe(tick, reader):
            sampled.append(tick)
        assert not observer.observe(tick, reader)
        assert observer.timestamp == tick // 30 * 30
    assert sampled == observer.sample_times == [0, 30, 60]
    assert reader.call_count == observer.query_count == 3
    assert observer.query_seconds >= 0
    np.testing.assert_array_equal(observer.values, [7, 8])


def test_snapshot_is_independent_and_cannot_be_modified():
    costs = np.array([3., 4.])
    observer = PeriodicCostObserver(30, [1, 2])
    observer.observe(0, lambda: costs)
    first = observer.values
    costs[:] = 8
    np.testing.assert_array_equal(first, [3, 4])
    with pytest.raises(ValueError):
        first[0] = 9
    with pytest.raises(ValueError):
        first.setflags(write=True)
    observer.observe(30, lambda: costs)
    np.testing.assert_array_equal(first, [3, 4])
    np.testing.assert_array_equal(observer.values, [8, 8])


def test_missing_initial_or_scheduled_tick_fails_without_query():
    observer = PeriodicCostObserver(30, [1])
    reader = Mock(return_value=[2])
    with pytest.raises(RuntimeError, match='time 0'):
        _ = observer.values
    with pytest.raises(ValueError, match='time 0'):
        observer.observe(1, reader)
    reader.assert_not_called()
    observer.observe(0, reader)
    observer.observe(29, reader)
    with pytest.raises(ValueError, match='time 30'):
        observer.observe(31, reader)
    assert reader.call_count == 1


def test_invalid_values_use_original_free_flow_and_reset_new_episode():
    free_flow = np.array([1., 2., 3., 4., 5., 6.])
    observer = PeriodicCostObserver(30, free_flow)
    free_flow[:] = 99
    observer.observe(0, lambda: [np.nan, np.inf, -np.inf, 0, -2, 7])
    np.testing.assert_array_equal(observer.values, [1, 2, 3, 4, 5, 7])
    assert observer.invalid_cost_count == 5
    observer.observe(30, lambda: np.ones(6))
    observer.reset()
    assert observer.timestamp is None
    assert observer.sample_times == []
    assert observer.query_count == observer.invalid_cost_count == 0
    assert observer.query_seconds == 0
    observer.observe(0, lambda: np.ones(6) * 2)
    assert observer.sample_times == [0]


@pytest.mark.parametrize('period', [True, np.bool_(False), 0, -1, 30., 1.5, np.nan, '30'])
def test_invalid_period(period):
    with pytest.raises(ValueError, match='positive integer'):
        PeriodicCostObserver(period, [1])


@pytest.mark.parametrize('now', [True, -1, 0.5, np.inf, np.nan, None])
def test_invalid_clock(now):
    observer = PeriodicCostObserver(30, [1])
    reader = Mock()
    with pytest.raises(ValueError, match='clock'):
        observer.observe(now, reader)
    reader.assert_not_called()


def test_backward_clock_and_wrong_shape():
    observer = PeriodicCostObserver(30, [1, 2])
    with pytest.raises(ValueError, match='one value per'):
        observer.observe(0, lambda: [1])
    observer.observe(0, lambda: [2, 3])
    observer.observe(10, lambda: pytest.fail('Unexpected getter'))
    with pytest.raises(ValueError, match='monotonic'):
        observer.observe(9, lambda: pytest.fail('Unexpected getter'))


def test_one_second_period_samples_each_tick():
    observer = PeriodicCostObserver(np.int64(1), [1])
    for tick in range(4):
        assert observer.observe(tick, lambda: [tick + 1])
    assert observer.sample_times == [0, 1, 2, 3]
