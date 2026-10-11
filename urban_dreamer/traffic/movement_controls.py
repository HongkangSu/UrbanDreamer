"""Static city controls for single-queue SUMO meso execution.

This module describes control, not gap acceptance or queue dynamics. Request
indices follow SUMO's incoming-lane / outgoing-connection ordering, independently
of TLS indices. Representative links match MESegment::getLink in SUMO 1.27.1:
queue lane 0 first, then other lanes, first connection to the requested edge.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
import xml.etree.ElementTree as ET

from .topology import _lane_allows_passenger


class UnsupportedControl(ValueError):
    """A movement has no declared, supported control interpretation."""


@dataclass(frozen=True)
class MovementControl:
    source_edge: int
    target_edge: int
    junction_id: str
    junction_type: str
    request_index: int
    tls_id: str | None
    tls_index: int | None
    off_state: str
    yields_to: frozenset[int]
    conflicts_with: frozenset[int]
    from_lane: int
    to_lane: int
    support: str = 'exact_control'
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class _Program:
    offset: int
    phases: tuple[tuple[int, str], ...]

    def state(self, index: int, tick: float) -> str:
        position = (math.floor(tick) - self.offset) % sum(d for d, _ in self.phases)
        for duration, states in self.phases:
            if position < duration:
                return states[index]
            position -= duration
        raise AssertionError('invalid phase cycle')


def _bits(value: str, width: int) -> frozenset[int]:
    if len(value) != width or set(value) - {'0', '1'}:
        raise ValueError('request bitset must have one binary bit per junction connection')
    return frozenset(i for i, bit in enumerate(reversed(value)) if bit == '1')


def _program(logic: ET.Element) -> _Program:
    if logic.get('type', 'static') != 'static':
        raise UnsupportedControl('non-static TLS program')
    offset = float(logic.get('offset', '0'))
    phases = []
    if not math.isfinite(offset) or not offset.is_integer():
        raise UnsupportedControl('TLS offset requires integral seconds')
    for phase in logic.findall('phase'):
        duration, state = float(phase.attrib['duration']), phase.attrib['state']
        if ('next' in phase.attrib or not math.isfinite(duration) or duration <= 0
                or not duration.is_integer() or not state or set(state) - set('Ggry')):
            raise UnsupportedControl('unsupported TLS phase duration, order, or state')
        phases.append((int(duration), state))
    if not phases or len({len(state) for _, state in phases}) != 1:
        raise UnsupportedControl('TLS phase widths must be consistent and nonzero')
    return _Program(int(offset), tuple(phases))


class CityControlTable:
    """Immutable movement lookup with explicit control support accounting.

``no_external_overrides`` declares no additional-file/WAUT/TraCI TLS changes.
``verified_road_only_no_trains`` additionally asserts that railway traffic and
rail-controller changes were excluded and the open road state verified. The
latter remains a labeled approximation, never an inferred static program.
"""

    def __init__(self, movements, programs, inventory):
        self.movements = MappingProxyType(dict(movements))
        self.programs = MappingProxyType(dict(programs))
        self._inventory = inventory

    @classmethod
    def from_network(cls, path: str | Path, edge_ids, *, no_external_overrides: bool = False,
                     verified_road_only_no_trains: bool = False):
        if not no_external_overrides:
            raise UnsupportedControl('declare absence of additional-file and runtime control overrides')
        path = Path(path)
        root = ET.parse(path).getroot()
        if root.findall('.//WAUT') or root.findall('.//waut'):
            raise UnsupportedControl('WAUT schedules require a separate control provider')
        edge_ids = tuple(edge_ids)
        indices = {edge: i for i, edge in enumerate(edge_ids)}
        if len(indices) != len(edge_ids):
            raise ValueError('duplicate registered edge IDs')
        edges = {edge.attrib['id']: edge for edge in root.findall('edge')}
        if set(indices) - set(edges):
            raise ValueError('registered edge IDs absent from network')
        lanes = {(name, int(lane.attrib['index'])): lane
                 for name, edge in edges.items() for lane in edge.findall('lane')}
        lane_keys = {lane.attrib['id']: key for key, lane in lanes.items()}
        junctions = {j.attrib['id']: j for j in root.findall('junction')}
        by_lane = defaultdict(list)
        connections = root.findall('connection')
        for connection in connections:
            by_lane[(connection.attrib['from'], int(connection.attrib['fromLane']))].append(connection)

        # Equivalent to sumolib Node.getLinkIndex; never use via/intLanes order.
        request_indices, requests = {}, {}
        for name, junction in junctions.items():
            if junction.get('type') == 'internal':
                continue
            ordered = []
            for lane_id in junction.get('incLanes', '').split():
                if lane_id not in lane_keys:
                    raise ValueError(f'unknown incoming lane {lane_id}')
                for connection in by_lane[lane_keys[lane_id]]:
                    source_function = edges[connection.attrib['from']].get('function', '')
                    target_function = edges[connection.attrib['to']].get('function', '')
                    if target_function == 'walkingarea' or (source_function == 'walkingarea'
                                                           and target_function != 'crossing'):
                        continue
                    ordered.append(connection)
            rows = junction.findall('request')
            if not ordered and not rows:
                continue
            row_map = {int(row.attrib['index']): row for row in rows}
            if len(row_map) != len(rows) or set(row_map) != set(range(len(ordered))):
                raise ValueError(f'incomplete request indices at junction {name}')
            for index, connection in enumerate(ordered):
                row = row_map[index]
                response = _bits(row.attrib['response'], len(ordered))
                foes = _bits(row.attrib['foes'], len(ordered))
                if not response <= foes:
                    raise ValueError(f'response outside foes at junction {name}')
                request_indices[id(connection)] = index
                requests[(name, index)] = (response, foes)

        programs, program_errors = {}, {}
        logics = defaultdict(list)
        for logic in root.findall('tlLogic'):
            logics[logic.attrib['id']].append(logic)
        for name, entries in logics.items():
            try:
                if len(entries) != 1:
                    raise UnsupportedControl('multiple TLS programs without declared selection')
                programs[name] = _program(entries[0])
            except UnsupportedControl as exc:
                program_errors[name] = str(exc)

        by_pair, passenger_connections = defaultdict(list), []
        for connection in connections:
            source, target = connection.attrib['from'], connection.attrib['to']
            if source not in indices or target not in indices:
                continue
            if source.startswith(':') or target.startswith(':'):
                continue
            by_pair[(source, target)].append(connection)
            if all(_lane_allows_passenger(lanes[(connection.attrib[side], int(connection.attrib[side+'Lane']))])
                   for side in ('from', 'to')):
                passenger_connections.append(connection)
        eligible_pairs = {(c.attrib['from'], c.attrib['to']) for c in passenger_connections}
        movements = {}
        for source, target in sorted(eligible_pairs):
            # Python's stable minimum preserves XML order within the same lane.
            connection = min(by_pair[(source, target)], key=lambda c: int(c.attrib['fromLane']))
            name = edges[source].attrib['to']
            junction_type = junctions[name].attrib['type']
            if id(connection) not in request_indices:
                raise ValueError(f'connection missing request index: {source} -> {target}')
            index = request_indices[id(connection)]
            response, foes = requests[(name, index)]
            tls = connection.get('tl') or None
            tls_index = int(connection.attrib['linkIndex']) if tls else None
            off_state = connection.attrib['state']
            support, reason = 'exact_control', None
            if junction_type == 'rail_crossing':
                if verified_road_only_no_trains:
                    support = 'road_only_rail_approximation'
                else:
                    support, reason = 'unsupported', 'rail crossing requires observed controller or verified road-only declaration'
            elif junction_type not in {'priority', 'right_before_left', 'traffic_light', 'zipper'}:
                support, reason = 'unsupported', f'unhandled junction type {junction_type}'
            elif tls:
                reason = program_errors.get(tls)
                if tls not in programs and reason is None:
                    reason = 'controlled connection has no static TLS program'
                if reason is None and (tls_index < 0 or tls_index >= len(programs[tls].phases[0][1])):
                    reason = 'controlled connection has invalid TLS index'
                if reason:
                    support = 'unsupported'
            elif off_state not in {'M', 'm', '=', 'Z', 'O', 'o'}:
                support, reason = 'unsupported', f'unhandled uncontrolled link state {off_state}'
            pair = (indices[source], indices[target])
            movements[pair] = MovementControl(*pair, name, junction_type, index, tls, tls_index,
                off_state, response, foes, int(connection.attrib['fromLane']),
                int(connection.attrib['toLane']), support, reason)

        inventory = dict(network_path=str(path.resolve()), network_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            edge_ids=edge_ids, no_external_overrides=True,
            verified_road_only_no_trains=verified_road_only_no_trains,
            passenger_lane_connections=len(passenger_connections),
            request_indices_validated=len(request_indices),
            static_program_errors=program_errors,
            passenger_connections_with_static_tls=sum(c.get('tl') in programs for c in passenger_connections),
            passenger_connections_with_g=sum(any(state[int(c.attrib['linkIndex'])] == 'g'
                for _, state in programs[c.attrib['tl']].phases) for c in passenger_connections
                if c.get('tl') in programs and 0 <= int(c.attrib['linkIndex']) < len(programs[c.attrib['tl']].phases[0][1])),
            passenger_connections_without_tls_program=sum(bool(c.get('tl')) and c.get('tl') not in programs
                                                          for c in passenger_connections))
        return cls(movements, programs, inventory)

    def state(self, movement: MovementControl | tuple[int, int], tick: float) -> str:
        if not math.isfinite(tick):
            raise ValueError('simulation tick must be finite')
        if not isinstance(movement, MovementControl):
            try:
                movement = self.movements[movement]
            except KeyError as exc:
                raise UnsupportedControl(f'unregistered movement {movement}') from exc
        if self.movements.get((movement.source_edge, movement.target_edge)) != movement:
            raise UnsupportedControl('movement does not belong to this table')
        if movement.support == 'unsupported':
            raise UnsupportedControl(movement.unsupported_reason)
        if movement.support == 'road_only_rail_approximation':
            return 'G'
        if movement.tls_id:
            return self.programs[movement.tls_id].state(movement.tls_index, tick)
        return movement.off_state

    def classification(self, movement: MovementControl | tuple[int, int], tick: float) -> str:
        state = self.state(movement, tick)
        return {'G': 'protected', 'M': 'protected', 'O': 'protected', 'r': 'blocked',
                'g': 'conditional', 'y': 'conditional', 'm': 'yield', '=': 'yield',
                'o': 'yield', 'Z': 'zipper'}[state]

    def preflight(self) -> dict:
        rows = []
        upstream = defaultdict(set)
        for movement in self.movements.values():
            row = asdict(movement)
            row['yields_to'], row['conflicts_with'] = sorted(movement.yields_to), sorted(movement.conflicts_with)
            row['source_id'] = self._inventory['edge_ids'][movement.source_edge]
            row['target_id'] = self._inventory['edge_ids'][movement.target_edge]
            rows.append(row)
            upstream[movement.target_edge].add(movement.source_edge)
        junctions = {m.junction_id: m.junction_type for m in self.movements.values()}
        inventory = {key: value for key, value in self._inventory.items() if key != 'edge_ids'}
        return dict(inventory, schema_version=1, semantics='static control only; conditional gap acceptance is not modeled',
            movement_count=len(rows), support_counts=dict(Counter(m.support for m in self.movements.values())),
            junction_types=dict(Counter(junctions.values())), merge_targets=sum(len(v) > 1 for v in upstream.values()),
            programs={name: dict(offset=p.offset, phases=p.phases) for name, p in self.programs.items()},
            movements=rows)
