"""DTA baseline: simulation-driven MSA over BPR marginal-cost paths.

Approximate system-optimum dynamic traffic assignment. Each training episode
doubles as one assignment iteration: realized window flows update per-edge
travel-time and volume estimates, volumes feed a BPR marginal-cost toll
(free_flow * alpha * (beta + 1) * (x / capacity)^beta, floored at the realized
travel time), every OD pair recomputes an all-or-nothing shortest path under
those marginal costs, and path fractions move toward it with a
method-of-successive-averages step. Fresh Poisson vehicles are allocated by
independent categorical draws keyed by episode seed, their own ID and OD.
Assignments never depend on future demand counts or departure timetables;
evaluation does not advance training randomness.
"""
from __future__ import annotations

import heapq
import hashlib
import math

import numpy as np

from . import METHODS
from ...traffic.topology import VehicleDemand

BPR_ALPHA = 0.15
BPR_BETA = 4.0
VOLUME_EMA = 0.5
MAX_PATHS_PER_OD = 8
CONVERGENCE_PATIENCE = 10
MIN_MEAN_SPEED_MPS = 0.5


class DtaMethod:
    """Approximate system-optimum DTA via MSA on simulation-estimated marginal costs.

    In current-to-destination mode, assignment is constrained to the same four
    static routes as the actor. At later decision boundaries an unavailable
    assigned suffix falls back to the cheapest current candidate, so executed
    path fractions can differ from the original OD assignment fractions.
    """

    needs_window_records = True
    needs_rows = False
    needs_row_flow = False

    def __init__(self, args, runner):
        if args is None or runner is None:
            # test_toy_baselines.py freezes this placeholder registry contract.
            raise NotImplementedError('method not implemented yet')
        self.args, self.runner = args, runner
        self.route_mode = getattr(runner.candidates, 'routing_mode', None) == 'current_to_dest'
        network = runner.network
        topology = runner.topology
        self.predecessors = [[] for _ in network.edge_ids]
        for source, targets in enumerate(network.adjacency_list):
            for target in targets:
                self.predecessors[target].append(source)
        self.edge_cost = network.free_flow_travel_seconds.astype(np.float64).copy()
        self.realized_seconds = self.edge_cost.copy()
        self.volume_ema = np.zeros(network.edge_count, dtype=np.float64)
        self.od_groups: dict[tuple[int, int], dict] = {}
        for vehicle in topology.vehicles:
            origin = network.edge_to_index[topology.origins[vehicle.origin_index]]
            destination = network.edge_to_index[topology.destinations[vehicle.destination_index]]
            group = self.od_groups.setdefault((origin, destination), {'vehicles': [], 'path_fractions': {}})
            group['vehicles'].append(vehicle)
        self._cost_to_go: dict[int, np.ndarray] = {}
        for (origin, destination), group in self.od_groups.items():
            group['vehicles'] = tuple(sorted(group['vehicles'],
                                             key=lambda v: (v.departure_second, v.vehicle_id)))
            if self.route_mode:
                path = self._best_candidate_path(origin, destination)
            else:
                path, dist = self._shortest_path(origin, destination, self.edge_cost)
                self._cost_to_go[destination] = dist
            group['path_fractions'] = {path: 1.0}
        self.assignment: dict[str, tuple[int, ...]] = {}
        self.poisson_mode = False
        self.episode_seed = None
        self.episode_groups = {}
        self.route_assignment_follow_count = 0
        self.route_assignment_fallback_count = 0
        self._rebuild_assignments()
        self.iteration = 0
        self.stable_count = 0
        self.converged = False
        self.rng = np.random.default_rng(args.seed)

    def _junction_seconds(self, source, target):
        length, speed = self.runner.candidates.junction(source, target)
        return float(length) / max(float(speed), 0.1)

    def _dijkstra(self, destination, edge_cost):
        """Reverse Dijkstra to ``destination``; edge (u->v) costs edge_cost[v] + junction(u, v)."""
        dist = np.full(self.runner.network.edge_count, np.inf)
        dist[destination] = 0.
        best_next = {}
        queue = [(0., destination)]
        while queue:
            total, target = heapq.heappop(queue)
            if total != dist[target]:
                continue
            for source in self.predecessors[target]:
                cost = total + float(edge_cost[target]) + self._junction_seconds(source, target)
                if cost < dist[source]:
                    dist[source] = cost
                    best_next[source] = target
                    heapq.heappush(queue, (cost, source))
        return dist, best_next

    @staticmethod
    def _extract_path(origin, destination, dist, best_next):
        if not math.isfinite(dist[origin]):
            raise ValueError(f'no path from edge {origin} to edge {destination}')
        path = [origin]
        while path[-1] != destination:
            path.append(best_next[path[-1]])
        return tuple(path)

    def _shortest_path(self, origin, destination, edge_cost):
        """Shortest path under ``edge_cost`` plus the full cost-to-go table to ``destination``."""
        dist, best_next = self._dijkstra(destination, edge_cost)
        return self._extract_path(origin, destination, dist, best_next), dist

    def _path_cost(self, path, edge_cost):
        return float(sum(float(edge_cost[target]) + self._junction_seconds(source, target)
                         for source, target in zip(path, path[1:])))

    def _best_candidate_path(self, origin, destination):
        """Constrain assignment to the actor's static four-route action pool."""
        return min(self.runner.candidates.generate_paths(origin, destination),
                   key=lambda path: self._path_cost(path, self.edge_cost))

    def begin_episode(self, episode_topology, *, seed):
        """Assign fresh vehicles without changing learned fractions, costs or RNG.

        Each vehicle draws independently from its OD's learned path fractions.
        The stable hash uses only the episode seed, its own ID and its OD;
        adding or reordering future arrivals cannot change an existing choice.
        Precomputing this mapping is execution convenience, not optimization
        against a known future demand timetable.
        """
        groups = {}
        network = self.runner.network
        for vehicle in episode_topology.vehicles:
            key = (network.edge_to_index[episode_topology.origins[vehicle.origin_index]],
                   network.edge_to_index[episode_topology.destinations[vehicle.destination_index]])
            if key not in self.od_groups:
                raise ValueError(f'episode demand contains an unknown reference OD: {key}')
            groups.setdefault(key, []).append(vehicle)
        self.episode_groups = {
            key: tuple(sorted(vehicles, key=lambda v: (v.departure_second, v.vehicle_id)))
            for key, vehicles in groups.items()}
        self.poisson_mode = True
        self.episode_seed = int(seed)
        self.stable_count = 0
        self.converged = False
        self.assignment = {}
        self.route_assignment_follow_count = 0
        self.route_assignment_fallback_count = 0
        self._rebuild_assignments()

    def _rebuild_assignments(self):
        """Allocate current vehicles; changed count only describes this realization."""
        changes = 0
        assignment = {}
        for key, group in sorted(self.od_groups.items()):
            vehicles = self.episode_groups.get(key, ()) if self.poisson_mode else group['vehicles']
            fractions = group['path_fractions']
            paths = sorted(fractions)
            weights = np.asarray([fractions[path] for path in paths], dtype=np.float64)
            weights = weights / max(float(weights.sum()), 1e-12)
            cumulative = np.cumsum(weights)
            count = len(vehicles)
            prefix = f'dta:{self.episode_seed}:{key[0]}:{key[1]}:'.encode()
            for slot, vehicle in enumerate(vehicles):
                if self.poisson_mode:
                    digest = hashlib.sha256(prefix + vehicle.vehicle_id.encode()).digest()
                    # 53 bits keep the resulting double strictly below one.
                    quantile = (int.from_bytes(digest[:8], 'big') >> 11) / (1 << 53)
                else:
                    quantile = (slot + 0.5) / count
                index = int(np.searchsorted(cumulative, quantile, side='right'))
                path = paths[min(index, len(paths) - 1)]
                if self.assignment.get(vehicle.vehicle_id) != path:
                    changes += 1
                assignment[vehicle.vehicle_id] = path
        self.assignment = assignment
        return changes

    def select(self, vehicles, actor_obs, frozen_flow, now, *, explore=True):
        """Follow the assigned path; fall back to marginal-cost greedy when off path."""
        indices = []
        routes = getattr(actor_obs, 'routes', None)
        for row, vehicle in enumerate(vehicles):
            legal = np.flatnonzero(actor_obs.mask[row])
            edge_indices = actor_obs.edge_indices[row]
            path = self.assignment.get(vehicle['vehicle_id'])
            chosen = None
            if routes is not None:
                if path is not None:
                    try:
                        suffix = path[path.index(int(vehicle['current_edge'])):]
                    except ValueError:
                        suffix = None
                    for column in legal:
                        if tuple(routes[row][column]) == suffix:
                            chosen = int(column)
                            break
                if chosen is None:
                    chosen = int(min(legal, key=lambda column:
                        self._path_cost(routes[row][column], self.edge_cost)))
                    self.route_assignment_fallback_count += 1
                else:
                    self.route_assignment_follow_count += 1
                indices.append(chosen)
                continue
            if path is not None:
                try:
                    position = path.index(int(vehicle['current_edge']))
                except ValueError:
                    position = -1
                if 0 <= position < len(path) - 1:
                    nxt = path[position + 1]
                    for column in legal:
                        if int(edge_indices[column]) == nxt:
                            chosen = int(column)
                            break
            if chosen is None:
                destination = int(vehicle['destination_edge'])
                cost_to_go = self._cost_to_go.get(destination)
                if cost_to_go is None:
                    cost_to_go = self.runner.candidates.costs(destination)
                chosen = int(min(legal, key=lambda column: (
                    self.edge_cost[int(edge_indices[column])] + cost_to_go[int(edge_indices[column])],
                    int(edge_indices[column]))))
            indices.append(chosen)
        return indices

    def on_episode(self, rows, windows, *, episode):
        """One DTA iteration: estimate realized costs, toll, AON, MSA step, convergence."""
        self.iteration += 1
        network = self.runner.network
        edge_count = network.edge_count
        speed_samples: list[list[float]] = [[] for _ in range(edge_count)]
        volume_samples: list[list[float]] = [[] for _ in range(edge_count)]
        for window in windows:
            next_flow = np.asarray(window['next_flow'], dtype=np.float64)
            for edge in range(edge_count):
                if next_flow[edge, 2] > 0:
                    speed_samples[edge].append(float(next_flow[edge, 1]) * float(network.speed_limits_mps[edge]))
                    volume_samples[edge].append(float(next_flow[edge, 2]) * float(network.capacities[edge]))
        measured_volume = self.volume_ema.copy()
        for edge in range(edge_count):
            if speed_samples[edge]:
                mean_speed = float(np.mean(speed_samples[edge]))
                self.realized_seconds[edge] = float(network.lengths_m[edge]) / max(mean_speed, MIN_MEAN_SPEED_MPS)
                measured_volume[edge] = float(np.mean(volume_samples[edge]))
        self.volume_ema = VOLUME_EMA * self.volume_ema + (1. - VOLUME_EMA) * measured_volume
        ratio = np.clip(self.volume_ema / np.maximum(network.capacities, 1e-9), 0., None)
        marginal = network.free_flow_travel_seconds * (1. + BPR_ALPHA * (BPR_BETA + 1.) * ratio ** BPR_BETA)
        self.edge_cost = np.maximum(marginal, self.realized_seconds)
        cost_to_go = {}
        aon_paths = {}
        for origin, destination in self.od_groups:
            if self.route_mode:
                aon_paths[(origin, destination)] = self._best_candidate_path(origin, destination)
                continue
            if destination not in cost_to_go:
                cost_to_go[destination] = self._dijkstra(destination, self.edge_cost)
            dist, best_next = cost_to_go[destination]
            aon_paths[(origin, destination)] = self._extract_path(origin, destination, dist, best_next)
        self._cost_to_go = {destination: tables[0] for destination, tables in cost_to_go.items()}
        eta = 1. / self.iteration
        weighted_fraction_change, max_fraction_change = 0., 0.
        for key, group in self.od_groups.items():
            fractions = group['path_fractions']
            previous_fractions = dict(fractions)
            aon = aon_paths[key]
            fractions.setdefault(aon, 0.)
            while len(fractions) > MAX_PATHS_PER_OD:
                removable = min((path for path in fractions if path != aon), key=fractions.get)
                fractions[aon] += fractions.pop(removable)
            for path in list(fractions):
                fractions[path] *= (1. - eta)
            fractions[aon] += eta
            total = sum(fractions.values())
            for path in fractions:
                fractions[path] /= total
            change = sum(abs(fractions.get(path, 0.) - previous_fractions.get(path, 0.))
                         for path in fractions.keys() | previous_fractions.keys())
            weighted_fraction_change += change * len(group['vehicles'])
            max_fraction_change = max(max_fraction_change, change)
        changes = self._rebuild_assignments()
        weighted_gap, total_vehicles, active_paths = 0., 0, 0
        for group in self.od_groups.values():
            fractions = group['path_fractions']
            costs = {path: self._path_cost(path, self.edge_cost) for path in fractions}
            best = min(costs.values())
            average = sum(fractions[path] * costs[path] for path in fractions)
            weighted_gap += (average - best) * len(group['vehicles'])
            total_vehicles += len(group['vehicles'])
            active_paths += sum(fraction > 1e-9 for fraction in fractions.values())
        # Fresh demand invalidates identity-based early stopping. Keep running
        # the requested budget and assess fractions, gap and held-out outcomes.
        if not self.poisson_mode:
            self.stable_count = self.stable_count + 1 if changes == 0 else 0
            if self.stable_count >= CONVERGENCE_PATIENCE:
                self.converged = True
        return dict(dta_iteration=self.iteration, assignment_changes=int(changes),
            mc_gap=float(weighted_gap / max(total_vehicles, 1)),
            path_fraction_change=float(weighted_fraction_change / max(total_vehicles, 1)),
            path_fraction_change_max=float(max_fraction_change),
            mean_estimated_edge_seconds=float(self.realized_seconds.mean()),
            active_paths=int(active_paths))

    def state_dict(self):
        return dict(iteration=self.iteration, stable_count=self.stable_count, converged=self.converged,
            poisson_mode=self.poisson_mode, episode_seed=self.episode_seed,
            episode_groups=[dict(origin=origin, destination=destination,
                vehicles=[dict(vehicle_id=v.vehicle_id, origin_index=v.origin_index,
                    destination_index=v.destination_index, departure_second=v.departure_second)
                    for v in vehicles])
                for (origin, destination), vehicles in sorted(self.episode_groups.items())],
            edge_cost=self.edge_cost.tolist(), realized_seconds=self.realized_seconds.tolist(),
            volume_ema=self.volume_ema.tolist(),
            path_sets=[dict(origin=origin, destination=destination,
                            paths=[list(path) for path in sorted(group['path_fractions'])],
                            fractions=[group['path_fractions'][path] for path in sorted(group['path_fractions'])])
                       for (origin, destination), group in sorted(self.od_groups.items())],
            assignment={vehicle_id: list(path) for vehicle_id, path in self.assignment.items()},
            cost_to_go={str(destination): [float(value) for value in table]
                        for destination, table in self._cost_to_go.items()},
            rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.iteration = int(state['iteration'])
        self.stable_count = int(state['stable_count'])
        self.converged = bool(state['converged'])
        self.poisson_mode = bool(state['poisson_mode'])
        self.episode_seed = state['episode_seed']
        self.episode_groups = {
            (int(entry['origin']), int(entry['destination'])):
                tuple(VehicleDemand(**vehicle) for vehicle in entry['vehicles'])
            for entry in state['episode_groups']}
        self.edge_cost = np.asarray(state['edge_cost'], dtype=np.float64)
        self.realized_seconds = np.asarray(state['realized_seconds'], dtype=np.float64)
        self.volume_ema = np.asarray(state['volume_ema'], dtype=np.float64)
        for entry in state['path_sets']:
            key = (int(entry['origin']), int(entry['destination']))
            self.od_groups[key]['path_fractions'] = {
                tuple(map(int, path)): float(fraction)
                for path, fraction in zip(entry['paths'], entry['fractions'])}
        self.assignment = {vehicle_id: tuple(map(int, path))
                           for vehicle_id, path in state['assignment'].items()}
        self._cost_to_go = {int(destination): np.asarray(table, dtype=np.float64)
                            for destination, table in state['cost_to_go'].items()}
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state['rng']


METHODS['dta'] = DtaMethod
