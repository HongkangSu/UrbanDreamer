"""Opt-in stop-line/FIFO constraints, not a complete SUMO meso simulator.

These constraints omit segment capacity, discharge headways and priority-junction
arbitration. Initial spatial order approximates FIFO unless an authoritative
queue order is supplied. Equal positions and ambiguous same-step merges fail
explicitly. Signal inputs must be declared schedules or current observations,
never future vehicle outcomes.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol
import xml.etree.ElementTree as ET


class UnsupportedConstraint(ValueError):
    pass


class SignalProvider(Protocol):
    def permits(self, current_edge: int, next_edge: int, timestamp: float) -> bool: ...


class StaticSignalSchedule:
    """Single fixed XML program per TLS, integral ticks, no external overrides.

    Caller must guarantee no additional-file/WAUT/TraCI program overrides. Only
    protected G and blocked r/y are supported; permissive g needs arbitration.
    Offset follows SUMO's cycle position (t - offset) modulo period. Lookup uses
    the current simulation tick, not the lagging boundary getState observation.
    """
    def __init__(self, programs, links):
        self.programs, self.links = programs, links

    @classmethod
    def from_network(cls, path: str | Path, edge_ids, *, no_external_overrides: bool = False):
        if not no_external_overrides:
            raise UnsupportedConstraint('static signals require declared absence of external/runtime overrides')
        root = ET.parse(path).getroot()
        if root.findall('.//WAUT') or root.findall('.//waut'):
            raise UnsupportedConstraint('WAUT schedules unsupported')
        programs = {}
        for logic in root.findall('tlLogic'):
            name = logic.attrib['id']
            if name in programs or logic.get('type', 'static') != 'static':
                raise UnsupportedConstraint('only one static program per TLS is supported')
            offset = float(logic.get('offset', '0'))
            if not math.isfinite(offset) or not offset.is_integer():
                raise UnsupportedConstraint('static TLS offset must use whole-second ticks')
            phases = []
            for phase in logic.findall('phase'):
                duration = float(phase.attrib['duration'])
                state = phase.attrib['state']
                if ('next' in phase.attrib or not math.isfinite(duration) or duration <= 0
                        or not duration.is_integer() or set(state) - set('Gry')):
                    raise UnsupportedConstraint('unsupported TLS phase order, duration, or signal permissions')
                phases.append((int(duration), state))
            if not phases or not phases[0][1] or len({len(state) for _, state in phases}) != 1:
                raise UnsupportedConstraint('TLS phase states must have consistent nonzero width')
            programs[name] = (int(offset), tuple(phases))
        indices = {edge: index for index, edge in enumerate(edge_ids)}
        links = {}
        for link in root.findall('connection'):
            if link.get('from') not in indices or link.get('to') not in indices or 'tl' not in link.attrib:
                continue
            tls, signal = link.attrib['tl'], int(link.attrib['linkIndex'])
            if tls not in programs or signal < 0 or signal >= len(programs[tls][1][0][1]):
                raise UnsupportedConstraint('controlled connection has missing program or signal index')
            key = (indices[link.attrib['from']], indices[link.attrib['to']])
            links.setdefault(key, set()).add((tls, signal))
        return cls(programs, links)

    def permits(self, current_edge, next_edge, timestamp):
        permissions = set()
        for tls, index in self.links.get((current_edge, next_edge), ()):
            offset, phases = self.programs[tls]
            position = (math.floor(timestamp) - offset) % sum(duration for duration, _ in phases)
            for duration, state in phases:
                if position < duration:
                    permissions.add(state[index] == 'G')
                    break
                position -= duration
        if len(permissions) > 1:
            raise UnsupportedConstraint('lane-dependent signal permissions cannot be resolved from edge pair')
        return next(iter(permissions), True)


class MesoReleaseConstraint:
    """Stop-line permission plus edge FIFO; blocked cars remain on source edge.

    Zero-gap queues are permitted: this is an exit-order constraint, not a
    microscopic following-distance model. An explicit fifo_order must come from
    authoritative queue telemetry, not arbitrary IDs or inferred departure times.
    """
    def __init__(self, signals: SignalProvider, *, fifo_order=None):
        self.signals = signals
        self.fifo_order = fifo_order
        self.queues = {}
        self.states = {}
        self.step_entries = {}
        self.blocked_signal = self.blocked_fifo = 0

    @staticmethod
    def _active(state):
        return state.get('departed', False) and not state.get('arrived', False)

    def prepare(self, states):
        if any(not state.get('departed', False) and not state.get('arrived', False) for state in states):
            raise UnsupportedConstraint('release constraints require a fixed active cohort; no future births')
        self.states = {str(state['vehicle_id']): state for state in states}
        if len(self.states) != len(states):
            raise UnsupportedConstraint('duplicate vehicle IDs')
        groups = {}
        for key, state in self.states.items():
            if self._active(state):
                groups.setdefault(int(state['current_edge']), []).append(key)
        for edge, members in groups.items():
            previous = [key for key in self.queues.get(edge, ()) if key in members]
            if previous:
                if set(previous) != set(members):
                    raise UnsupportedConstraint('new seed members require a fresh constraint or explicit entry events')
                self.queues[edge] = previous
                continue
            if self.fifo_order is not None and edge in self.fifo_order:
                order = list(map(str, self.fifo_order[edge]))
                if len(set(order)) != len(order) or set(order) != set(members):
                    raise UnsupportedConstraint('explicit FIFO order must cover exactly this road cohort')
            else:
                order = sorted(members, key=lambda key: float(self.states[key]['remaining_distance']))
                distances = [float(self.states[key]['remaining_distance']) for key in order]
                if any(abs(a-b) <= 1e-9 for a, b in zip(distances, distances[1:])):
                    raise UnsupportedConstraint('equal-position vehicles require authoritative FIFO order')
            self.queues[edge] = order
        self._prune()

    def _prune(self):
        for edge, queue in self.queues.items():
            self.queues[edge] = [key for key in queue if self._active(self.states[key])
                                and int(self.states[key]['current_edge']) == edge]

    def before_step(self):
        self._prune()
        self.step_entries = {}

    def movement_order(self, indices, states):
        positions = {key: (edge, index) for edge, queue in self.queues.items() for index, key in enumerate(queue)}
        return sorted(indices, key=lambda index: positions.get(str(states[index]['vehicle_id']), (math.inf, index)))

    def permits_exit(self, state, next_edge, timestamp):
        self._prune()
        edge, key = int(state['current_edge']), str(state['vehicle_id'])
        queue = self.queues.get(edge, ())
        if not queue or queue[0] != key:
            self.blocked_fifo += 1
            return False
        if next_edge is not None and not self.signals.permits(edge, int(next_edge), timestamp):
            self.blocked_signal += 1
            return False
        return True

    def entered(self, state, source_edge, timestamp):
        target, key = int(state['current_edge']), str(state['vehicle_id'])
        previous = self.step_entries.setdefault(target, [])
        if previous and any(source != source_edge for source, _, _ in previous):
            raise UnsupportedConstraint('same-step multi-upstream merge needs explicit arbitration')
        previous.append((source_edge, timestamp, key))
        self._prune()
        if key not in self.queues.setdefault(target, []):
            self.queues[target].append(key)
