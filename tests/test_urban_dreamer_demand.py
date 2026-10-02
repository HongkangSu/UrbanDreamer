import collections

import numpy as np
import pytest

from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.topology import VehicleDemand


def reference_vehicles():
    return tuple(VehicleDemand(str(i), 0 if i < 2 else 1, 1 if i < 2 else 0, float(i))
                 for i in range(6))


def test_poisson_od_rates_and_reproducible_episode_identities():
    model = PoissonDemand.from_vehicles(reference_vehicles(), duration_seconds=60.)
    assert model.od_rates == ((0, 1, 2. / 60.), (1, 0, 4. / 60.))
    assert model.expected_vehicle_count == 6.
    first = model.sample(seed=42, episode_index=0)
    assert first == model.sample(seed=42, episode_index=0)
    second = model.sample(seed=42, episode_index=1)
    assert first != second
    assert not ({v.vehicle_id for v in first} & {v.vehicle_id for v in second})
    assert len({v.vehicle_id for v in first}) == len(first)
    assert all(0. <= v.departure_second < 60. for v in first + second)
    assert all((v.origin_index, v.destination_index) in {(0, 1), (1, 0)} for v in first + second)
    assert [v.departure_second for v in first] == sorted(v.departure_second for v in first)


def test_independent_poisson_od_counts_and_uniform_time_increments():
    model = PoissonDemand.from_vehicles(reference_vehicles(), duration_seconds=60.)
    counts, early, empty = [], [], 0
    for episode in range(2000):
        vehicles = model.sample(seed=72, episode_index=episode)
        od = collections.Counter((v.origin_index, v.destination_index) for v in vehicles)
        counts.append([od[(0, 1)], od[(1, 0)]])
        early.append(sum(v.departure_second < 30. for v in vehicles))
        empty += not vehicles
    counts = np.asarray(counts)
    np.testing.assert_allclose(counts.mean(axis=0), [2., 4.], atol=.12)
    np.testing.assert_allclose(counts.var(axis=0), [2., 4.], atol=.35)
    assert abs(np.corrcoef(counts.T)[0, 1]) < .08
    assert abs(np.mean(early) - 3.) < .12
    assert empty > 0  # Do not condition the Poisson distribution on N > 0.


def test_demand_sampling_does_not_consume_global_numpy_rng():
    model = PoissonDemand.from_vehicles(reference_vehicles(), duration_seconds=60.)
    np.random.seed(42)
    expected = np.random.random(4)
    np.random.seed(42)
    model.sample(seed=42, episode_index=0)
    np.testing.assert_array_equal(np.random.random(4), expected)


@pytest.mark.parametrize('duration', [0., -1., float('nan'), float('inf')])
def test_invalid_poisson_duration(duration):
    with pytest.raises(ValueError, match='duration'):
        PoissonDemand.from_vehicles(reference_vehicles(), duration_seconds=duration)
