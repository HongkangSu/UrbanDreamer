"""Paired, environment-only demand retiming without changing vehicles or OD.

An original uniform departure quantile is mapped through an intensity CDF.
Unlike resampling or rank transforms, each vehicle's new time is independent
of the presence of other vehicles. No demand information is sent to policies.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Sequence, TypeVar

from urban_dreamer.traffic.demand import PoissonDemand


PROFILES = ('original', 'peaks', 'od_stagger', 'burst')
_BOUNDS = (0., 1. / 6., 1. / 3., .5, 2. / 3., 1.)
_PEAKS = (.4, 2.5, .4, 2.1, .3)
_BURST = (0., 0., 6., 0., 0.)
_STAGGER = ((.4, 3.2, .4, 1., .5), (.4, 1., .4, 3.2, .5))
Vehicle = TypeVar('Vehicle')


def _validate(profile: str, duration_seconds: float) -> None:
    if profile not in PROFILES:
        raise ValueError(f'unknown demand profile {profile!r}; expected {PROFILES}')
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('duration_seconds must be finite and positive')


def od_cohort(origin_index: int, destination_index: int) -> int:
    """Stable OD-only cohort, independent of demand counts, order, and seed."""
    digest = hashlib.sha256(f'{origin_index}:{destination_index}'.encode('utf-8')).digest()
    return int.from_bytes(digest, 'big') % 2


def profile_metadata(profile: str, duration_seconds: float) -> dict:
    """Serializable generation contract; this belongs in archives, not observations."""
    _validate(profile, duration_seconds)
    weights = {'original': ((1.,) * 5,), 'peaks': (_PEAKS,), 'od_stagger': _STAGGER, 'burst': (_BURST,)}[profile]
    return {
        'profile': profile,
        'duration_seconds': float(duration_seconds),
        'departure_transform': 't_new = duration * inverse_CDF(t_original / duration)',
        'time_bounds_seconds': [float(x * duration_seconds) for x in _BOUNDS],
        'cohort_rate_multipliers': [list(row) for row in weights],
        'normalization': 'interval probability proportional to duration times rate multiplier',
        'cohort_rule': 'int(sha256(UTF8(origin_index:destination_index)), 16) % 2' if profile == 'od_stagger' else None,
        'demand_law': 'independent_poisson' if profile == 'original' else 'independent_nonhomogeneous_poisson',
        'preserves': ['vehicle_count', 'vehicle_id', 'origin_index', 'destination_index', 'within_od_departure_order'],
        'additional_rng_draws': 0,
    }


def _inverse_cdf(quantile: float, rates: tuple[float, ...]) -> float:
    masses = tuple((right - left) * rate for left, right, rate in zip(_BOUNDS, _BOUNDS[1:], rates))
    remaining = quantile * math.fsum(masses)
    for left, right, rate, mass in zip(_BOUNDS, _BOUNDS[1:], rates, masses):
        if remaining < mass:
            return min(right, left + remaining / rate)
        remaining -= mass
    # Rounding near the upper boundary must never move a departure outside T.
    return math.nextafter(1., 0.)


def retime_vehicles(vehicles: Sequence[Vehicle], duration_seconds: float, profile: str) -> tuple[Vehicle, ...]:
    """Retain every dataclass field except departure_second; do not mutate inputs."""
    _validate(profile, duration_seconds)
    output = []
    for vehicle in vehicles:
        departure = float(vehicle.departure_second)
        if not math.isfinite(departure) or not 0. <= departure < duration_seconds:
            raise ValueError('vehicle departure_second must be finite and in [0, duration_seconds)')
        if profile == 'original':
            output.append(vehicle)
            continue
        rates = (_BURST if profile == 'burst' else _PEAKS if profile == 'peaks'
                 else _STAGGER[od_cohort(vehicle.origin_index, vehicle.destination_index)])
        new_time = _inverse_cdf(departure / duration_seconds, rates) * duration_seconds
        active_end = duration_seconds * _BOUNDS[1 + max(i for i, rate in enumerate(rates) if rate > 0)]
        new_time = min(new_time, math.nextafter(float(active_end), 0.))
        output.append(replace(vehicle, departure_second=new_time))
    return tuple(sorted(output, key=lambda vehicle: (vehicle.departure_second, vehicle.vehicle_id)))


@dataclass(frozen=True)
class ReTimedDemand:
    """Wrap the existing generator, preserving its seed stream and OD counts."""

    base: PoissonDemand
    profile: str

    def __post_init__(self) -> None:
        _validate(self.profile, self.duration_seconds)

    @property
    def duration_seconds(self) -> float:
        return self.base.duration_seconds

    @property
    def expected_vehicle_count(self) -> float:
        return self.base.expected_vehicle_count

    @property
    def od_rates(self) -> tuple[tuple[int, int, float], ...]:
        # Average rates: temporal variation is described by profile_metadata.
        return self.base.od_rates

    def sample(self, *, seed: int, episode_index: int):
        return retime_vehicles(self.base.sample(seed=seed, episode_index=episode_index), self.duration_seconds, self.profile)
