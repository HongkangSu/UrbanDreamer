"""Fixed-cohort labels and strict retrospective route replay for reward audits.

Future archived decisions are diagnostic controls, not a deployable policy or
an on-policy model test. No subsequent births contribute to these labels;
their congestion effects on the real cohort remain in the observed outcome.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

from urban_dreamer.learning.episode_archive import observed_arrival_times
from urban_dreamer.learning.global_reward import started_count_from_state
from urban_dreamer.traffic.environment import NextHopDecision


@dataclass(frozen=True)
class CohortLabel:
    start_time: float
    end_time: float
    started_count: int
    vehicle_ids: tuple[str, ...]
    vehicle_seconds: float
    count_integral_seconds: float
    reward: float
    arrivals: int


def _cohort(archive: dict[str, Any], seed_index: int):
    if seed_index < 0 or seed_index >= len(archive['states']):
        raise ValueError('seed index outside archive')
    state = archive['states'][seed_index]
    ids = tuple(str(v['vehicle_id']) for v in state['vehicles']
                if v.get('departed', False) and not v.get('arrived', False))
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate cohort vehicle ID')
    count = started_count_from_state(state)
    if count < len(ids):
        raise ValueError('started count smaller than active cohort')
    return state, ids, count


def _arrivals(archive: dict[str, Any]) -> dict[str, float]:
    # Prefer boundary observations; permit an explicit observed summary for
    # lightweight archives, but never infer arrival from scheduled departure.
    result = {str(k): float(v) for k, v in archive.get('arrival_times', {}).items()}
    final = archive['states'][-1]['vehicles']
    missing = [v for v in final if v.get('arrived') and str(v['vehicle_id']) not in result]
    if missing:
        observed = observed_arrival_times({**archive, 'states': [{'vehicles': missing}]})
        result.update(observed)
    for v in final:
        if v.get('arrived') and v.get('arrival_time') is not None:
            key, timestamp = str(v['vehicle_id']), float(v['arrival_time'])
            if key in result and not math.isclose(result[key], timestamp, abs_tol=1e-7):
                raise ValueError('arrival summary contradicts observed arrival')
            result[key] = timestamp
    if any(not math.isfinite(t) or t < 0 for t in result.values()):
        raise ValueError('invalid observed arrival time')
    return result


def _label(archive, seed_index, start_time, end_time) -> CohortLabel:
    state, ids, count = _cohort(archive, seed_index)
    seed_time = float(state['time'])
    last = float(archive['states'][-1]['time'])
    if not math.isfinite(start_time) or not math.isfinite(end_time):
        raise ValueError('audit boundaries must be finite')
    end_time = min(float(end_time), last)
    if not seed_time <= start_time <= end_time:
        raise ValueError('interval must lie after seed within observed archive')
    arrivals = _arrivals(archive)
    if any(arrivals.get(v, math.inf) < seed_time for v in ids):
        raise ValueError('active seed vehicle arrived before seed')
    # Per-vehicle observed lifetimes and independent piecewise population
    # integral must agree, including fractional arrivals and censored trips.
    exposure = sum(max(0., min(end_time, arrivals.get(v, end_time)) - start_time)
                   for v in ids)
    cuts = sorted({start_time, end_time} | {
        arrivals[v] for v in ids if v in arrivals and start_time < arrivals[v] < end_time})
    integral = sum((right - left) * sum(arrivals.get(v, math.inf) > left for v in ids)
                   for left, right in zip(cuts, cuts[1:]))
    return CohortLabel(float(start_time), end_time, count, ids, exposure, integral,
                       -exposure / (60. * max(count, 1)),
                       sum(start_time < arrivals.get(v, math.inf) <= end_time for v in ids))


def real_cohort_label(archive: dict[str, Any], seed_index: int, end_time: float) -> CohortLabel:
    """Undiscounted real physical reward from seed to bounded observed endpoint."""
    return _label(archive, seed_index, float(archive['states'][seed_index]['time']), end_time)


def cohort_window_labels(archive: dict[str, Any], seed_index: int,
                         end_times) -> list[CohortLabel]:
    """Disjoint windows retaining the original seed cohort and denominator."""
    start = float(archive['states'][seed_index]['time'])
    rows = []
    for end in end_times:
        row = _label(archive, seed_index, start, float(end))
        rows.append(row)
        start = row.end_time
    return rows


class RouteReplayUnavailable(ValueError):
    """Archived action sequence cannot control this imagined trajectory."""


class RecordedRouteReplay:
    """Strict ordered per-vehicle action replay; never searches ahead/falls back.

    Archive decisions occupy [window_start, window_end), so an action exactly
    at the seed timestamp has not yet been applied to that boundary snapshot.
    Keep all later archived actions, even beyond the comparison endpoint: a
    faster imagined vehicle may reach a later recorded turn before that time.
    """
    def __init__(self, archive: dict[str, Any], seed_index: int, candidates: Any):
        state, ids, _ = _cohort(archive, seed_index)
        self.start_time = float(state['time'])
        self.candidates = candidates
        self.events = defaultdict(list)
        self.cursor = defaultdict(int)
        for event in archive['decisions']:
            key = str(event['vehicle_id'])
            if key in ids and float(event['decision_time']) >= self.start_time:
                self.events[key].append(event)
        for events in self.events.values():
            if any(float(a['decision_time']) > float(b['decision_time'])
                   for a, b in zip(events, events[1:])):
                raise ValueError('archived vehicle decisions are not ordered')

    def __call__(self, pending, flow, now):
        if now < self.start_time:
            raise RouteReplayUnavailable('replay called before seed')
        decisions = []
        keys = [str(v['vehicle_id']) for v in pending]
        if len(set(keys)) != len(keys):
            raise RouteReplayUnavailable('duplicate pending vehicle')
        for state, key in zip(pending, keys):
            index = self.cursor[key]
            events = self.events[key]
            if index >= len(events):
                raise RouteReplayUnavailable(f'{key}: archived decisions exhausted')
            event = events[index]
            edge, destination = int(state['current_edge']), int(state['destination_edge'])
            if int(event['current_edge']) != edge:
                raise RouteReplayUnavailable(
                    f'{key}: ordered current edge mismatch at imagined time {float(now):g}; '
                    f'imagined edge={edge}, next recorded edge={event["current_edge"]}, '
                    f'recorded time={event["decision_time"]}, cursor={index}')
            if event.get('destination_edge') is not None and int(event['destination_edge']) != destination:
                raise RouteReplayUnavailable(f'{key}: destination mismatch')
            if getattr(self.candidates, 'forbid_revisit', False):
                legal = self.candidates.legal_candidates(edge, destination, state['visited_edges'])
            else:
                legal = self.candidates.candidates(edge, destination)
            target = int(event['next_edge'])
            if target not in legal:
                raise RouteReplayUnavailable(f'{key}: recorded next edge is not legal')
            decisions.append(NextHopDecision(key, edge, target, float(now)))
        # Failed batches leave every cursor untouched.
        for key in keys:
            self.cursor[key] += 1
        return decisions


class RecordedPhysicalRouteReplay:
    """Reconstruct commanded physical itineraries, including unobserved forks.

    Requires the diagnostic caller to call ``validate_entry(state, next_edge)``
    on EVERY physical external edge entry. This detects turns skipped between
    imagined observations; callback matching alone cannot establish route
    equality. No-revisit route reconstruction is intentionally unsupported.
    """
    def __init__(self, archive, seed_index, candidates):
        if getattr(candidates, 'forbid_revisit', False):
            raise RouteReplayUnavailable('physical replay requires unconstrained full-route semantics')
        seed, ids, _ = _cohort(archive, seed_index)
        self.start_time = float(seed['time'])
        self.candidates = candidates
        self.paths = {}
        self.cursor = {key: 0 for key in ids}
        finals = {str(v['vehicle_id']): v for v in archive['states'][-1]['vehicles']}
        events = defaultdict(list)
        for event in archive['decisions']:
            key = str(event['vehicle_id'])
            if key in self.cursor and float(event['decision_time']) >= self.start_time:
                events[key].append(event)
        for state in seed['vehicles']:
            key = str(state['vehicle_id'])
            if key not in self.cursor:
                continue
            current, destination = int(state['current_edge']), int(state['destination_edge'])
            committed = state.get('committed_next_edge')
            records = events[key]
            if current == destination:
                suffix = [current]
            elif committed is not None:
                suffix = list(candidates.full_route(current, int(committed), destination))
            elif records and float(records[0]['decision_time']) == self.start_time:
                suffix = [current]
            else:
                raise RouteReplayUnavailable(f'{key}: seed lacks commanded next edge')
            prefix = []
            previous_time = self.start_time
            for index, event in enumerate(records):
                timestamp, edge = float(event['decision_time']), int(event['current_edge'])
                if timestamp < previous_time:
                    raise RouteReplayUnavailable(f'{key}: unordered archived decisions')
                # Only a pending action at the seed can replace the route at
                # suffix[0]; later observations require physical edge entry.
                at_seed = (index == 0 and timestamp == self.start_time
                           and state.get('pending_decision', False))
                start = 0 if at_seed else 1
                try:
                    position = suffix.index(edge, start)
                except ValueError as error:
                    raise RouteReplayUnavailable(
                        f'{key}: recorded edge {edge} at {timestamp:g} not on commanded suffix') from error
                prefix.extend(suffix[:position])
                suffix = list(candidates.full_route(edge, int(event['next_edge']), destination))
                previous_time = timestamp
            final = finals[key]
            final_edge = int(final['current_edge'])
            if final.get('arrived'):
                if final_edge != destination:
                    raise RouteReplayUnavailable(f'{key}: final arrival not on destination')
                path = prefix + suffix
            else:
                try:
                    position = suffix.index(final_edge)
                except ValueError as error:
                    raise RouteReplayUnavailable(f'{key}: final observed edge not on commanded suffix') from error
                # Beyond this observed edge, future actor choices are unknown.
                path = prefix + suffix[:position + 1]
            self.paths[key] = tuple(path)

    def validate_entry(self, state, next_edge):
        key = str(state['vehicle_id'])
        path, index = self.paths[key], self.cursor[key]
        if index + 1 >= len(path):
            raise RouteReplayUnavailable(f'{key}: physical itinerary exhausted')
        if int(next_edge) != path[index + 1]:
            raise RouteReplayUnavailable(
                f'{key}: physical route diverged: entered {next_edge}, expected {path[index + 1]}')
        self.cursor[key] = index + 1

    def __call__(self, pending, flow, now):
        decisions = []
        for state in pending:
            key = str(state['vehicle_id'])
            path, index = self.paths[key], self.cursor[key]
            edge = int(state['current_edge'])
            if path[index] != edge:
                raise RouteReplayUnavailable(f'{key}: callback missing physical entry validation')
            if index + 1 >= len(path):
                raise RouteReplayUnavailable(f'{key}: physical itinerary has no recorded successor')
            target = path[index + 1]
            if target not in self.candidates.candidates(edge, int(state['destination_edge'])):
                raise RouteReplayUnavailable(f'{key}: reconstructed successor not legal')
            decisions.append(NextHopDecision(key, edge, target, float(now)))
        return decisions
