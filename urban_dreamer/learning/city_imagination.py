"""Opt-in causal city surrogate; frozen road speeds plus chronological queues.

One FIFO queue per external edge, constant discharge/receiving headways and a
constant conflicting-movement clearance are declared approximations. They are
not SUMO segment capacity, microscopic gap acceptance, or full meso physics.
Only the supplied active cohort exists. Static controls are sampled at physical
crossing times; actions, arrival publication and costs retain the 1 s clock.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np

from . import imagination as runtime
from .action_events import ActionEvent
from .imagination import ActiveCostSegment, ImaginedWindow
from ..traffic.movement_controls import UnsupportedControl

_EPS = 1e-9


class ChronologicalJunctionArbiter:
    """Persistent per-rollout FIFO and movement reservations.

    Keep this object across windows. ``city_queue_order`` is an authoritative
    per-edge rank supplied by the caller. Otherwise unequal spatial positions
    initialize an explicitly approximate order; equal positions require the
    explicit ``uncertain_order`` seeded mode. Vehicle IDs only canonicalize RNG
    draws, and never purport to establish physical queue order.
    """

    def __init__(self, controls: Any, *, seed: int, uncertain_order: bool = False,
                 discharge_headway_seconds: float = 1.,
                 receiving_headway_seconds: float = 1.,
                 conflict_clearance_seconds: float = 1.):
        values = (discharge_headway_seconds, receiving_headway_seconds,
                  conflict_clearance_seconds)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('surrogate headways and clearance must be finite and positive')
        self.controls = controls
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.uncertain_order = bool(uncertain_order)
        self.discharge_headway, self.receiving_headway, self.clearance = map(float, values)
        self.queues: dict[int, list[dict[str, Any]]] = {}
        self.source_ready: dict[int, float] = {}
        self.target_ready: dict[int, float] = {}
        self.reservations: list[tuple[Any, float]] = []
        self.audit: list[dict[str, Any]] = []
        self._cohort: set[str] | None = None
        self._rank = 0
        self.end_time: float | None = None

    def prepare(self, states, now):
        cohort = {str(s['vehicle_id']) for s in states}
        if self._cohort is not None:
            if cohort != self._cohort or abs(now - self.end_time) > _EPS:
                raise ValueError('persistent arbiter requires the same cohort and contiguous windows')
            expected = {str(s['vehicle_id']) for q in self.queues.values() for s in q}
            actual = {str(s['vehicle_id']) for s in states if not s.get('arrived', False)}
            if expected != actual:
                raise ValueError('cohort state changed outside the persistent arbiter')
            # The caller may reconstruct state dictionaries between windows.
            by_id = {str(s['vehicle_id']): s for s in states}
            for edge, queue in self.queues.items():
                replacement = [by_id[str(s['vehicle_id'])] for s in queue]
                if any(int(s['current_edge']) != edge for s in replacement):
                    raise ValueError('cohort edges changed outside the persistent arbiter')
                self.queues[edge] = replacement
            return
        if any('_city_queue_rank' in state for state in states):
            raise ValueError('existing city FIFO state requires its persistent arbiter and reservations')
        grouped = defaultdict(list)
        for state in states:
            if not state.get('arrived', False):
                grouped[int(state['current_edge'])].append(state)
        prepared = {}
        for edge, rows in grouped.items():
            if all('city_queue_order' in s for s in rows):
                if len({s['city_queue_order'] for s in rows}) != len(rows):
                    raise ValueError('authoritative city_queue_order ranks must be unique per edge')
                ordered = sorted(rows, key=lambda s: s['city_queue_order'])
                kind = 'authoritative'
            elif any('city_queue_order' in s for s in rows):
                raise ValueError('authoritative queue order must cover every seed on its edge')
            else:
                positions = defaultdict(list)
                for state in rows:
                    positions[float(state['remaining_distance'])].append(state)
                ordered = []
                kind = 'spatial_approximation'
                for distance in sorted(positions):
                    tied = positions[distance]
                    if len(tied) > 1:
                        if not self.uncertain_order:
                            raise ValueError('tied positions require authoritative city_queue_order or explicit uncertain_order')
                        tied = sorted(tied, key=lambda s: str(s['vehicle_id']))
                        tied = [tied[i] for i in self.rng.permutation(len(tied))]
                        kind = 'seeded_uncertain_order'
                    ordered.extend(tied)
            prepared[edge] = (ordered, kind)
        self._cohort = cohort
        for edge, (rows, kind) in prepared.items():
            self.queues[edge] = rows
            self.audit.append(dict(event='queue_initialization', edge=edge, kind=kind,
                                   vehicles=[str(s['vehicle_id']) for s in rows], seed=self.seed))
            for state in rows:
                self._stamp(state, edge, kind)

    def _stamp(self, state, edge, kind):
        state['_city_queue_rank'] = self._rank
        state['_city_queue_edge'] = edge
        state['_city_queue_order_kind'] = kind
        self._rank += 1

    @staticmethod
    def _conflicts(a, b):
        return (a.junction_id == b.junction_id and
                (b.request_index in a.conflicts_with or a.request_index in b.conflicts_with))

    def movement(self, state):
        successor = state.get('committed_next_edge')
        if successor is None:
            return None
        pair = (int(state['current_edge']), int(successor))
        try:
            movement = self.controls.movements[pair]
        except KeyError as exc:
            raise UnsupportedControl(f'unregistered movement {pair}') from exc
        return movement

    def grants(self, now, speeds, *, control_time=None):
        """Gather all queue heads before any physical entry mutates the queues."""
        control_time = now if control_time is None else control_time
        self.reservations = [(m, until) for m, until in self.reservations if until > now + _EPS]
        heads = []
        for edge, queue in self.queues.items():
            if not queue:
                continue
            state = queue[0]
            distance, speed = float(state['remaining_distance']), float(speeds[edge])
            eta = now if distance <= _EPS else now + distance / speed if speed > 0 else math.inf
            movement = self.movement(state)
            kind = 'protected' if movement is None else self.controls.classification(movement, control_time)
            heads.append((state, movement, kind, eta))
        ready = []
        for state, movement, kind, eta in heads:
            source = int(state['current_edge'])
            if eta > now + _EPS or kind == 'blocked' or self.source_ready.get(source, -math.inf) > now + _EPS:
                continue
            if movement is not None:
                if self.target_ready.get(movement.target_edge, -math.inf) > now + _EPS:
                    continue
                if any(self._conflicts(movement, other) for other, _ in self.reservations):
                    continue
            ready.append((state, movement, kind))
        blockers = {}
        for state, movement, kind in ready:
            waiting_for = set()
            if movement is not None and kind in {'conditional', 'yield'}:
                for other, other_movement, other_kind, eta in heads:
                    if other is state or other_movement is None or other_kind == 'blocked':
                        continue
                    if eta > now + self.clearance + _EPS or not self._conflicts(movement, other_movement):
                        continue
                    if (other_kind == 'protected' or
                            (movement.junction_id == other_movement.junction_id and
                             other_movement.request_index in movement.yields_to)):
                        waiting_for.add(str(other['vehicle_id']))
            blockers[str(state['vehicle_id'])] = waiting_for
        available = [row for row in ready if not blockers[str(row[0]['vehicle_id'])]]
        if not available and ready:
            # Release only a closed ready priority cycle; never override a red,
            # a reservation, or an approaching priority vehicle outside it.
            pool = {str(row[0]['vehicle_id']) for row in ready}
            while True:
                reduced = {name for name in pool if blockers[name] and blockers[name] <= pool}
                if reduced == pool:
                    break
                pool = reduced
            if pool:
                # Choose a sink strongly connected component. A simple cycle
                # may still yield to another cycle, and breaking that upstream
                # cycle would incorrectly ignore an external priority head.
                reachable = {}
                for name in pool:
                    reached, stack = set(), [name]
                    while stack:
                        other = stack.pop()
                        if other not in reached:
                            reached.add(other)
                            stack.extend(blockers[other])
                    reachable[name] = reached
                sink_members = set()
                components = {}
                for name in sorted(pool):
                    cycle = {other for other in reachable[name] if name in reachable[other]}
                    if all(blockers[other] <= cycle for other in cycle):
                        sink_members.update(cycle)
                        components[name] = sorted(cycle)
                if not sink_members:
                    raise AssertionError('closed finite priority graph has no sink component')
                candidates = sorted((row for row in ready if str(row[0]['vehicle_id']) in sink_members),
                                    key=lambda row: str(row[0]['vehicle_id']))
                chosen = candidates[int(self.rng.integers(len(candidates)))]
                available = [chosen]
                self.audit.append(dict(event='priority_cycle_break', time=now,
                    contenders=sorted(sink_members), cycle=components[str(chosen[0]['vehicle_id'])],
                    winner=str(chosen[0]['vehicle_id']), seed=self.seed))
        # Randomize equal-time claims explicitly, retaining protected priority.
        available.sort(key=lambda row: str(row[0]['vehicle_id']))
        if len(available) > 1:
            available = [available[i] for i in self.rng.permutation(len(available))]
            available.sort(key=lambda row: row[2] != 'protected')
        winners = []
        for row in available:
            state, movement, _ = row
            contested = [other for other in winners if movement is not None and other[1] is not None
                         and (movement.target_edge == other[1].target_edge or self._conflicts(movement, other[1]))]
            if contested:
                self.audit.append(dict(event='equal_time_tie', time=now, seed=self.seed,
                    winner=str(contested[0][0]['vehicle_id']), deferred=str(state['vehicle_id'])))
                continue
            winners.append(row)
        return winners

    def release(self, state, movement, now):
        source = int(state['current_edge'])
        if self.queues[source][0] is not state:
            raise AssertionError('only a FIFO head may cross')
        self.queues[source].pop(0)
        self.source_ready[source] = now + self.discharge_headway
        if movement is not None:
            self.target_ready[movement.target_edge] = now + self.receiving_headway
            self.reservations.append((movement, now + self.clearance))
        self.audit.append(dict(event='crossing' if movement else 'physical_arrival', time=now,
            vehicle_id=str(state['vehicle_id']), source_edge=source,
            target_edge=None if movement is None else movement.target_edge))

    def entered(self, state):
        edge = int(state['current_edge'])
        self.queues.setdefault(edge, []).append(state)
        self._stamp(state, edge, 'chronological_entry')

    def next_time(self, now, limit, speeds):
        events = [limit, math.floor(now + _EPS) + 1.]
        events.extend(t for t in self.source_ready.values() if t > now + _EPS)
        events.extend(t for t in self.target_ready.values() if t > now + _EPS)
        events.extend(t for _, t in self.reservations if t > now + _EPS)
        for edge, queue in self.queues.items():
            if queue and speeds[edge] > 0 and queue[0]['remaining_distance'] > _EPS:
                events.append(now + float(queue[0]['remaining_distance']) / speeds[edge])
        return min(events)


def advance_city_imagined_window(
    vehicle_states: Sequence[dict[str, Any]], *, candidates: Any,
    edge_speeds_mps: np.ndarray, frozen_flow: np.ndarray, start_time: float,
    total_vehicle_count: int,
    decide: Callable[[list[dict[str, Any]], np.ndarray, float], Sequence[Any]],
    junction_arbiter: ChronologicalJunctionArbiter,
    interval_seconds: float = 30., observation_step_seconds: float | None = None,
    vehicle_max_speed_mps: float | None = None, stop_when_empty: bool = False,
    session_protocol: str = 'event', event_index_start: int = 0,
    traffic_model: str = 'meso',
) -> ImaginedWindow:
    """Advance only a fixed active cohort, with no insertion/future demand input."""
    if session_protocol != 'event' or traffic_model != 'meso' or getattr(candidates, 'traffic_model', None) != 'meso':
        raise ValueError('city imagination requires meso candidates and event session protocol')
    if observation_step_seconds not in (None, 1.):
        raise ValueError('city imagination requires one-second observations')
    end = float(start_time + interval_seconds)
    if (not math.isfinite(start_time) or not math.isfinite(end) or interval_seconds <= 0
            or abs(start_time - round(start_time)) > _EPS
            or abs(interval_seconds - round(interval_seconds)) > _EPS):
        raise ValueError('city windows require finite whole-second boundaries')
    if total_vehicle_count <= 0 or event_index_start < 0 or int(event_index_start) != event_index_start:
        raise ValueError('positive population and nonnegative integer event index required')
    network = candidates.network
    speeds = np.asarray(edge_speeds_mps, dtype=float)
    if speeds.shape != (network.edge_count,) or not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError('one finite non-negative speed per edge required')
    if vehicle_max_speed_mps is not None:
        if not math.isfinite(vehicle_max_speed_mps) or vehicle_max_speed_mps <= 0:
            raise ValueError('vehicle_max_speed_mps must be finite and positive')
        speeds = np.minimum(speeds, vehicle_max_speed_mps)
    states = sorted(vehicle_states, key=lambda s: str(s['vehicle_id']))
    if len({str(s['vehicle_id']) for s in states}) != len(states):
        raise ValueError('vehicle IDs must be unique')
    for state in states:
        if not state.get('departed', False):
            raise ValueError('city imagination accepts only the active seed cohort; no future births')
        if state.get('in_junction', False):
            raise ValueError('city meso imagination cannot accept microscopic in_junction seeds')
        if not math.isfinite(float(state['remaining_distance'])) or state['remaining_distance'] < 0:
            raise ValueError('remaining route distances must be finite and non-negative')
    runtime._prepare_no_revisit_history(states, candidates)

    def needs_decision(state):
        return (int(state['current_edge']) != int(state['destination_edge'])
                and len(runtime._legal_candidates(state, candidates)) > 1)

    def route_successor(state):
        edge, destination = int(state['current_edge']), int(state['destination_edge'])
        if edge == destination:
            return None
        if getattr(candidates, 'forbid_revisit', False):
            route = state.get('no_revisit_route', ())
            if route and route[0] != edge:
                if len(route) < 2 or route[1] != edge:
                    raise ValueError('no-revisit route commitment disagrees with physical entry')
                route = route[1:]
                state['no_revisit_route'] = route
            if route:
                return int(route[1])
        choices, costs = runtime._legal_candidates(state, candidates), candidates.costs(destination)
        successor = min(choices, key=lambda target: (float(network.free_flow_travel_seconds[target]) + costs[target], target))
        if getattr(candidates, 'forbid_revisit', False):
            state['no_revisit_route'] = candidates.full_route(edge, successor, destination, state['visited_edges'])
        return int(successor)

    for state in states:
        if state.get('arrived', False):
            continue
        state.update(junction_remaining_distance=0., junction_speed=0., junction_connection=None)
        if not needs_decision(state):
            state['pending_decision'] = False
        if state['current_edge'] != state['destination_edge']:
            successor = state.get('committed_next_edge')
            if successor is None:
                if not state.get('pending_decision', False):
                    raise ValueError('non-pending event seed has no committed next hop')
                state['committed_next_edge'] = route_successor(state)
            elif int(successor) not in runtime._legal_candidates(state, candidates):
                raise ValueError('event seed has an illegal committed next hop')
            if getattr(candidates, 'forbid_revisit', False):
                state['no_revisit_route'] = candidates.full_route(int(state['current_edge']),
                    int(state['committed_next_edge']), int(state['destination_edge']), state['visited_edges'])
        else:
            state['committed_next_edge'] = None
        movement = junction_arbiter.movement(state)
        if movement is not None:
            junction_arbiter.controls.classification(movement, start_time)
    junction_arbiter.prepare(states, float(start_time))
    action = np.zeros(network.edge_count, dtype=np.float32)
    decisions, action_events, segments, arrivals = [], [], [], {}
    time = float(start_time)
    while time < end - _EPS:
        active = [s for s in states if not s.get('arrived', False)]
        if stop_when_empty and not active:
            break
        pending = [s for s in active if s.get('pending_decision', False)]
        if pending:
            selected = list(decide(pending, frozen_flow, time))
            by_id = {str(item.vehicle_id): item for item in selected}
            if len(by_id) != len(selected) or set(by_id) != {str(s['vehicle_id']) for s in pending}:
                raise ValueError('policy must return exactly one decision per pending vehicle')
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                if int(item.current_edge) != int(state['current_edge']) or int(item.next_edge) not in runtime._legal_candidates(state, candidates):
                    raise ValueError('policy selected a next hop outside the legal candidate set')
                if not math.isfinite(float(item.decision_time)) or abs(float(item.decision_time) - time) > _EPS:
                    raise ValueError('decision timestamp must match its observation')
                junction_arbiter.controls.classification((int(item.current_edge), int(item.next_edge)), time)
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                state['committed_next_edge'] = int(item.next_edge)
                state['pending_decision'] = False
                if getattr(candidates, 'forbid_revisit', False):
                    state['no_revisit_route'] = candidates.full_route(int(state['current_edge']),
                        int(item.next_edge), int(state['destination_edge']), state['visited_edges'])
                action[int(item.next_edge)] += 1.
                decisions.append(item)
                action_events.append(ActionEvent(str(item.vehicle_id), int(event_index_start) + len(action_events),
                    time, int(item.current_edge), int(item.next_edge), int(state['destination_edge'])))
        previous = {str(s['vehicle_id']): int(s['current_edge']) for s in active}
        physical_finished = set()
        observed = time + 1.
        physical = time
        while True:
            # Endpoint physical completions belong to the step just executed.
            # A control intervention at the new integer tick takes effect only
            # after that observation/policy boundary, not retroactively in it.
            control_time = np.nextafter(observed, -math.inf) if physical >= observed - _EPS else physical
            grants = junction_arbiter.grants(physical, speeds, control_time=control_time)
            for state, movement, _ in grants:
                junction_arbiter.release(state, movement, physical)
                if movement is None:
                    physical_finished.add(str(state['vehicle_id']))
                    continue
                successor = int(state['committed_next_edge'])
                runtime._record_physical_entry(state, successor, candidates)
                state['current_edge'] = successor
                state['remaining_distance'] = float(network.lengths_m[successor])
                if not math.isfinite(state['remaining_distance']) or state['remaining_distance'] <= 0:
                    raise ValueError('external edge lengths must be finite and positive')
                state['committed_next_edge'] = route_successor(state)
                junction_arbiter.entered(state)
            if grants:
                # A newly exposed FIFO head is considered at this same instant,
                # after the complete simultaneous batch has been committed.
                continue
            if physical >= observed - _EPS:
                break
            next_time = junction_arbiter.next_time(physical, observed, speeds)
            duration = next_time - physical
            if duration <= 0:
                raise RuntimeError('city physical clock failed to advance')
            for state in active:
                if str(state['vehicle_id']) not in physical_finished:
                    state['remaining_distance'] = max(0., float(state['remaining_distance']) -
                                                       speeds[int(state['current_edge'])] * duration)
            physical = next_time
        for state in active:
            name = str(state['vehicle_id'])
            if name in physical_finished:
                state.update(arrived=True, arrival_time=observed, pending_decision=False,
                             committed_next_edge=None, remaining_distance=0., in_junction=False)
                arrivals[name] = observed
            elif previous[name] != int(state['current_edge']):
                state['pending_decision'] = needs_decision(state)
        segments.append(ActiveCostSegment(time, observed, len(active)))
        time = observed
    junction_arbiter.end_time = time
    cost = sum((s.end_time - s.start_time) * s.active_count for s in segments)
    return ImaginedWindow(action / total_vehicle_count, np.zeros_like(action), tuple(decisions),
                         arrivals, float(cost), tuple(segments), time, tuple(action_events))
