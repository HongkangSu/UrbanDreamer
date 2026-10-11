from collections import Counter
from dataclasses import dataclass
import hashlib
import math

import pytest

from tools.routing_demand_profiles import PROFILES, ReTimedDemand, od_cohort, profile_metadata, retime_vehicles
from urban_dreamer.traffic.demand import PoissonDemand
from urban_dreamer.traffic.topology import VehicleDemand


@pytest.mark.parametrize('profile', PROFILES)
def test_preserves_vehicles_od_and_within_od_order(profile):
    base = PoissonDemand(((0, 1, .4), (1, 2, .3)), 1800.)
    original = base.sample(seed=42, episode_index=3)
    result = retime_vehicles(original, 1800., profile)
    identity = lambda rows: Counter((v.vehicle_id, v.origin_index, v.destination_index) for v in rows)
    assert identity(result) == identity(original)
    assert all(0. <= v.departure_second < 1800. for v in result)
    assert result == tuple(sorted(result, key=lambda v: (v.departure_second, v.vehicle_id)))
    for od in [(0, 1), (1, 2)]:
        ids = lambda rows: [v.vehicle_id for v in rows if (v.origin_index, v.destination_index) == od]
        assert ids(result) == ids(original)
    assert result == retime_vehicles(tuple(reversed(original)), 1800., profile)
    assert original == base.sample(seed=42, episode_index=3)


def test_peaks_exact_inverse_cdf_and_expected_interval_counts():
    # Uniform midpoints give the exact integer mass allocation for these rates.
    original = tuple(VehicleDemand(str(i), 0, 1, i + .5) for i in range(1800))
    result = retime_vehicles(original, 1800., 'peaks')
    bounds = (0., 300., 600., 900., 1200., 1800.)
    counts = [sum(a <= v.departure_second < b for v in result) for a, b in zip(bounds, bounds[1:])]
    assert counts == [120, 750, 120, 630, 180]
    samples = [VehicleDemand(str(t), 0, 1, float(t)) for t in (0, 60, 495, 930, 1305, 1710)]
    times = [v.departure_second for v in retime_vehicles(samples, 1800., 'peaks')]
    assert times == pytest.approx([0., 150., 450., 750., 1050., 1500.])


def test_stagger_cohorts_and_future_vehicle_independence():
    ods = {}
    for origin in range(100):
        cohort = od_cohort(origin, 101)
        assert cohort == int(hashlib.sha256(f'{origin}:101'.encode()).hexdigest(), 16) % 2
        ods.setdefault(cohort, origin)
    for cohort, origin in ods.items():
        original = tuple(VehicleDemand(f'{origin}_{i}', origin, 101, i + .5) for i in range(1800))
        result = retime_vehicles(original, 1800., 'od_stagger')
        bounds = (0., 300., 600., 900., 1200., 1800.)
        counts = [sum(a <= v.departure_second < b for v in result) for a, b in zip(bounds, bounds[1:])]
        assert counts == ([120, 960, 120, 300, 300] if cohort == 0 else [120, 300, 120, 960, 300])
        extra = VehicleDemand('extra', 500, 501, 1700.)
        combined = retime_vehicles((*original, extra), 1800., 'od_stagger')
        assert tuple(v for v in combined if v.vehicle_id != 'extra') == result
    assert set(ods) == {0, 1}


@pytest.mark.parametrize('profile', PROFILES)
def test_empty_boundary_scaling_and_metadata(profile):
    assert retime_vehicles((), 1800., profile) == ()
    original = (VehicleDemand('first', 0, 1, 0.), VehicleDemand('last', 0, 1, math.nextafter(60., 0.)))
    result = retime_vehicles(original, 60., profile)
    assert result[0].departure_second == (20. if profile == 'burst' else 0.)
    assert 0. <= result[-1].departure_second < 60.
    metadata = profile_metadata(profile, 60.)
    assert metadata['time_bounds_seconds'] == pytest.approx([0, 10, 20, 30, 40, 60])
    assert metadata['additional_rng_draws'] == 0


def test_wrapper_keeps_seed_stream_and_all_extra_dataclass_fields():
    base = PoissonDemand(((0, 1, 1.),), 60.)
    wrapped = ReTimedDemand(base, 'peaks')
    assert (wrapped.duration_seconds, wrapped.od_rates, wrapped.expected_vehicle_count) == (60., base.od_rates, 60.)
    assert wrapped.sample(seed=12, episode_index=0) == retime_vehicles(base.sample(seed=12, episode_index=0), 60., 'peaks')
    assert ReTimedDemand(base, 'original').sample(seed=12, episode_index=0) == base.sample(seed=12, episode_index=0)

    @dataclass(frozen=True)
    class ExtraVehicle(VehicleDemand):
        vehicle_type: str = 'car'

    vehicle = ExtraVehicle('v', 0, 1, 1., 'bus')
    changed, = retime_vehicles((vehicle,), 60., 'peaks')
    assert type(changed) is ExtraVehicle and changed.vehicle_type == 'bus'
    assert vehicle.departure_second == 1.


@pytest.mark.parametrize('departure', [-1., 60., float('nan'), float('inf')])
def test_invalid_departures_rejected(departure):
    with pytest.raises(ValueError, match='departure'):
        retime_vehicles((VehicleDemand('v', 0, 1, departure),), 60., 'peaks')


@pytest.mark.parametrize('duration', [0., -1., float('nan'), float('inf')])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError, match='duration'):
        retime_vehicles((), duration, 'original')


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match='profile'):
        ReTimedDemand(PoissonDemand((), 60.), 'missing')


def test_burst_preserves_population_and_handles_zero_rate_intervals():
    base = tuple(VehicleDemand(str(i), i % 2, 3, float(t))
                 for i, t in enumerate([0., 450., 900., 1350., math.nextafter(1800., 0.)]))
    result = retime_vehicles(base, 1800., 'burst')
    assert [v.vehicle_id for v in result] == [v.vehicle_id for v in base]
    assert all(600. <= v.departure_second < 900. for v in result)
    assert [v.departure_second for v in result[:4]] == pytest.approx([600., 675., 750., 825.])
    assert profile_metadata('burst', 1800.)['cohort_rate_multipliers'] == [[0., 0., 6., 0., 0.]]
