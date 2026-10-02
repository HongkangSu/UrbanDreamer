from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActiveCostSegment:
    start_time: float
    end_time: float
    active_count: int


@dataclass(frozen=True)
class ImaginedWindow:
    action: np.ndarray
    departure: np.ndarray
    decisions: tuple[Any, ...]
    arrivals: dict[str, float]
    active_vehicle_seconds: float
    cost_segments: tuple[ActiveCostSegment, ...]
    end_time: float


def advance_imagined_window(
    vehicle_states: Sequence[dict[str, Any]], *, candidates: Any,
    edge_speeds_mps: np.ndarray, frozen_flow: np.ndarray, start_time: float,
    total_vehicle_count: int,
    decide: Callable[[list[dict[str, Any]], np.ndarray, float], Sequence[Any]],
    interval_seconds: float = 30.0,
    observation_step_seconds: float = 0.001,
    vehicle_max_speed_mps: float | None = None,
    stop_when_empty: bool = False,
) -> ImaginedWindow:
    """Advance physical events through a window; policy events at its end stay pending.

    States are mutated in place, with distances (rather than cached traversal
    times) preserving progress under the next window's newly decoded speeds.
    A committed successor remains attached throughout the external edge and its
    junction connection. Only arrival on that successor creates a new decision.
    """
    network = candidates.network
    speeds = np.asarray(edge_speeds_mps, dtype=np.float64)
    if speeds.shape != (network.edge_count,) or not np.isfinite(speeds).all() or (speeds < 0).any():
        raise ValueError('edge_speeds_mps must contain one finite non-negative speed per edge')
    if total_vehicle_count <= 0 or interval_seconds <= 0 or not np.isfinite(start_time + interval_seconds):
        raise ValueError('positive population and finite positive interval required')
    if not np.isfinite(observation_step_seconds) or observation_step_seconds <= 0:
        raise ValueError('observation_step_seconds must be finite and positive')
    max_speed = float('inf') if vehicle_max_speed_mps is None else float(vehicle_max_speed_mps)
    if max_speed <= 0 or (vehicle_max_speed_mps is not None and not np.isfinite(max_speed)):
        raise ValueError('vehicle_max_speed_mps must be finite and positive')
    speeds = np.minimum(speeds, max_speed)
    states = sorted(vehicle_states, key=lambda state: str(state['vehicle_id']))
    for state in states:
        if state.get('in_junction', False):
            junction_speed = float(state['junction_speed'])
            if not np.isfinite(junction_speed) or junction_speed <= 0:
                raise ValueError('in-transit junction speed must be finite and positive')
            state['junction_speed'] = min(junction_speed, max_speed)
    if len({str(s['vehicle_id']) for s in states}) != len(states):
        raise ValueError('vehicle IDs must be unique')
    action = np.zeros(network.edge_count, dtype=np.float32)
    departure = np.zeros_like(action)
    decisions: list[Any] = []
    arrivals: dict[str, float] = {}
    active_seconds = 0.0
    cost_segments: list[ActiveCostSegment] = []
    time, end = float(start_time), float(start_time + interval_seconds)
    tolerance = 1e-9
    # An undeparted seed is physically absent at S_t even when its scheduled
    # departure is overdue (e.g. queued SUMO insertion). Insert it on the next
    # observation, so the resulting appearance is represented in D_t.
    departure_events = {
        str(state['vehicle_id']): (start_time + observation_step_seconds
            if float(state['departure_time']) <= start_time
            else float(state['departure_time']))
        for state in states if not state.get('departed', False)
    }

    def active(state: dict[str, Any]) -> bool:
        return bool(state.get('departed', False)) and not bool(state.get('arrived', False))

    def enter_successor(state: dict[str, Any]) -> None:
        successor = state.get('committed_next_edge')
        if successor is None:
            raise ValueError('vehicle exhausted its edge without a committed next hop')
        state['current_edge'] = int(successor)
        state['remaining_distance'] = float(network.lengths_m[int(successor)])
        state['committed_next_edge'] = None
        state['in_junction'] = False
        state['junction_remaining_distance'] = 0.0
        state['pending_decision'] = int(successor) != int(state['destination_edge'])

    while True:
        # Complete physical events at the endpoint before assigning the window.
        for state in states:
            if not state.get('departed', False) and not state.get('arrived', False) and departure_events[str(state['vehicle_id'])] <= time + tolerance:
                state['departed'] = True
                state['observed_departure_time'] = time
                state['pending_decision'] = int(state['current_edge']) != int(state['destination_edge'])
                if time > start_time + tolerance:
                    departure[int(state['current_edge'])] += 1.0
            if not active(state):
                continue
            if state.get('in_junction', False):
                if float(state['junction_remaining_distance']) <= tolerance:
                    enter_successor(state)
            elif float(state['remaining_distance']) <= tolerance:
                if int(state['current_edge']) == int(state['destination_edge']):
                    state['arrived'] = True
                    state['remaining_distance'] = 0.0
                    state['pending_decision'] = False
                    state['arrival_time'] = time
                    arrivals[str(state['vehicle_id'])] = time
                elif not state.get('pending_decision', False):
                    successor = state.get('committed_next_edge')
                    if successor is None:
                        raise ValueError('vehicle exhausted its edge without a committed next hop')
                    length, speed = candidates.junction(int(state['current_edge']), int(successor))
                    if length < 0 or speed <= 0 or not np.isfinite(length + speed):
                        raise ValueError('junction length and speed must be finite and physically valid')
                    state['in_junction'] = True
                    state['junction_remaining_distance'] = float(length)
                    state['junction_speed'] = min(float(speed), max_speed)
                    if length <= tolerance:
                        enter_successor(state)
        if time >= end - tolerance:
            break
        if stop_when_empty and all(state.get('arrived', False) for state in states):
            break

        pending = [s for s in states if active(s) and s.get('pending_decision', False)]
        if pending:
            selected = list(decide(pending, frozen_flow, time))
            by_id = {str(item.vehicle_id): item for item in selected}
            if len(by_id) != len(selected) or set(by_id) != {str(s['vehicle_id']) for s in pending}:
                raise ValueError('policy must return exactly one decision per pending vehicle')
            # Validate the complete batch before mutating commitments or counts.
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                legal = candidates.candidates(int(state['current_edge']), int(state['destination_edge']))
                if int(item.current_edge) != int(state['current_edge']) or int(item.next_edge) not in legal:
                    raise ValueError('policy selected a next hop outside the legal candidate set')
                if abs(float(item.decision_time) - time) > tolerance:
                    raise ValueError('decision timestamp must match its physical event')
            for state in pending:
                item = by_id[str(state['vehicle_id'])]
                state['committed_next_edge'] = int(item.next_edge)
                state['pending_decision'] = False
                action[int(item.next_edge)] += 1.0
                decisions.append(item)

        duration = end - time
        for state in states:
            if not state.get('departed', False) and not state.get('arrived', False):
                duration = min(duration, max(0., departure_events[str(state['vehicle_id'])] - time))
            elif active(state):
                in_junction = bool(state.get('in_junction', False))
                distance = float(state['junction_remaining_distance'] if in_junction else state['remaining_distance'])
                speed = float(state['junction_speed'] if in_junction else speeds[int(state['current_edge'])])
                if speed > 0:
                    duration = min(duration, max(distance, 0.) / speed)
        if duration <= tolerance:
            # Zero-distance committed edges need another physical event pass.
            if any(active(s) and not s.get('in_junction', False) and float(s['remaining_distance']) <= tolerance for s in states):
                if any(float(network.lengths_m[int(s['current_edge'])]) <= 0 for s in states if active(s)):
                    raise ValueError('zero-length external edges cannot be advanced safely')
                continue
            # A residual distance within one tolerance-step of completing its
            # edge or junction traversal (distance <= speed * tolerance) is
            # invisible to the physical pass but still stalls the clock; snap
            # it to completion. Runs that never raised never enter this branch.
            repaired = False
            for state in states:
                if not active(state):
                    continue
                in_junction = bool(state.get('in_junction', False))
                key = 'junction_remaining_distance' if in_junction else 'remaining_distance'
                speed = float(state['junction_speed'] if in_junction else speeds[int(state['current_edge'])])
                distance = float(state[key])
                if 0. < distance and speed > 0 and distance / speed <= tolerance:
                    state[key] = 0.
                    repaired = True
            if repaired:
                continue
            raise ValueError('imagined vehicle clock failed to advance')
        active_count = sum(active(s) for s in states)
        cost_segments.append(ActiveCostSegment(time, min(time + duration, end), active_count))
        active_seconds += active_count * duration
        for state in states:
            if not active(state):
                continue
            if state.get('in_junction', False):
                state['junction_remaining_distance'] = max(0., float(state['junction_remaining_distance']) - float(state['junction_speed']) * duration)
            else:
                state['remaining_distance'] = max(0., float(state['remaining_distance']) - float(speeds[int(state['current_edge'])]) * duration)
        time = min(time + duration, end)

    return ImaginedWindow(action / total_vehicle_count, departure / total_vehicle_count,
                          tuple(decisions), arrivals, float(active_seconds),
                          tuple(cost_segments), time)
