import pytest

from tools.benchmark_precision_mae import aggregate_scores


def row(occupied, slow, error):
    return dict(road_observations=10, trajectories=1, occupied_roads=occupied, slow_roads=slow,
        speed_mae_mps=error, occupied_speed_mae_mps=error if occupied else None,
        slow_speed_mae_mps=error if slow else None,
        occupied_count_mae_vehicles=error if occupied else None,
        total_count_mae_per_trajectory=error)


def test_masked_mae_uses_road_counts_not_case_average():
    result = aggregate_scores([row(9, 3, 1), row(1, 1, 9)])
    assert result['occupied_speed_mae_mps'] == pytest.approx(1.8)
    assert result['slow_speed_mae_mps'] == pytest.approx(3)
    assert result['occupied_count_mae_vehicles'] == pytest.approx(1.8)
    assert result['speed_mae_mps'] == pytest.approx(5)
    assert result['total_count_mae_per_trajectory'] == pytest.approx(5)


def test_absent_slow_roads_remain_unavailable():
    result = aggregate_scores([row(0, 0, 2), row(3, 0, 1)])
    assert result['slow_speed_mae_mps'] is None
    assert result['slow_roads'] == 0
    assert result['occupied_speed_mae_mps'] == 1
    assert result['road_observations'] == 20
