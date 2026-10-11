"""Freeze observed road costs on an episode-relative physical clock grid."""
from __future__ import annotations

import math
from numbers import Integral
from time import perf_counter

import numpy as np


class PeriodicCostObserver:
    """Sample at 0, period, 2 * period, independently of routing decisions.

    Call ``observe`` at each physical second, before any decisions at that time.
    A skipped sampling tick is an error: current costs cannot reconstruct a past
    observation. Repeated calls at the same physical time are harmless.
    """

    def __init__(self, period_seconds: int, free_flow_costs):
        if isinstance(period_seconds, (bool, np.bool_)) or not isinstance(period_seconds, Integral) or period_seconds <= 0:
            raise ValueError('Observation period must be a positive integer number of seconds')
        costs = np.array(free_flow_costs, dtype=np.float64, copy=True)
        if costs.ndim != 1 or not np.all(np.isfinite(costs)) or np.any(costs <= 0):
            raise ValueError('Free-flow costs must be a finite positive vector')
        self.period_seconds = int(period_seconds)
        self._free_flow_costs = costs
        self.reset()

    def reset(self):
        self._values = None
        self._last_time = None
        self._next_sample = 0
        self.timestamp = None
        self.sample_times = []
        self.query_seconds = 0.0
        self.query_count = 0
        self.invalid_cost_count = 0

    @property
    def values(self):
        if self._values is None:
            raise RuntimeError('Road costs must first be observed at time 0')
        return self._values

    def observe(self, now, read_costs) -> bool:
        if isinstance(now, (bool, np.bool_)):
            raise ValueError('Observation clock must be finite nonnegative integer seconds')
        try:
            physical_time = float(now)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Observation clock must be finite nonnegative integer seconds') from exc
        if not math.isfinite(physical_time) or physical_time < 0 or not physical_time.is_integer():
            raise ValueError('Observation clock must be finite nonnegative integer seconds')
        tick = int(physical_time)
        if self._last_time is not None and tick < self._last_time:
            raise ValueError('Observation clock must be monotonic; reset between episodes')
        if tick > self._next_sample:
            raise ValueError(f'Missed mandatory observation at time {self._next_sample}; current time is {tick}')
        if tick < self._next_sample:
            self._last_time = tick
            return False

        started = perf_counter()
        try:
            costs = np.array(read_costs(), dtype=np.float64, copy=True)
        finally:
            self.query_seconds += perf_counter() - started
            self.query_count += 1
        if costs.shape != self._free_flow_costs.shape:
            raise ValueError('Observed costs must contain one value per free-flow road cost')
        invalid = ~np.isfinite(costs) | (costs <= 0)
        costs[invalid] = self._free_flow_costs[invalid]
        self.invalid_cost_count += int(np.count_nonzero(invalid))
        # A bytes backing store prevents callers from re-enabling writes, too.
        self._values = np.frombuffer(costs.tobytes(), dtype=np.float64)
        self.timestamp = physical_time
        self.sample_times.append(physical_time)
        self._last_time = tick
        self._next_sample += self.period_seconds
        return True
