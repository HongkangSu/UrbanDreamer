"""Routing from static topology and currently observed edge costs only.

The GA adapter executes the author's source unchanged. Synthetic edge-specific
junctions expose legal turns rather than all geometrically adjacent roads.
"""
from __future__ import annotations

import heapq
import importlib.util
import math
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np


class RealTimeShortestRouter:
    def __init__(self, network):
        self.network = network
        self.successors = tuple(tuple(int(j) for j in row if network.passenger_allowed[j])
                                if network.passenger_allowed[i] else ()
                                for i, row in enumerate(network.adjacency_list))
        predecessors = [[] for _ in network.edge_ids]
        for i, row in enumerate(self.successors):
            for j in row:
                predecessors[j].append(i)
        self.predecessors = predecessors
        self.reset()

    def reset(self):
        self.costs = np.array(self.network.free_flow_travel_seconds, dtype=np.float64, copy=True)
        self.timestamp = None
        self._distances = {}
        self.stats = dict(decision_count=0, cache_hits=0, optimize_calls=0,
                          optimize_seconds=0.0, failed_searches=0, fallback_events=0)

    def update_costs(self, costs, timestamp):
        costs = np.asarray(costs, dtype=np.float64)
        if costs.shape != (len(self.network.edge_ids),) or not np.all(np.isfinite(costs)) or np.any(costs < 0):
            raise ValueError('Observed costs must be finite, nonnegative, and one per edge')
        if not math.isfinite(timestamp):
            raise ValueError('Observation timestamp must be finite')
        if timestamp != self.timestamp or not np.array_equal(costs, self.costs):
            self._distances.clear()
            self.costs = costs.copy()
        self.timestamp = timestamp

    def _legal(self, current, destination, legal):
        if not 0 <= destination < len(self.successors):
            raise ValueError('Unknown destination')
        legal = tuple(int(i) for i in legal)
        if not legal or any(i not in self.successors[current] for i in legal):
            raise ValueError('Candidates must be nonempty legal passenger successors')
        return legal

    def _distance(self, destination):
        if destination in self._distances:
            self.stats['cache_hits'] += 1
            return self._distances[destination]
        start = perf_counter()
        costs = self.costs.tolist()
        distance = [math.inf] * len(self.successors)
        distance[destination] = costs[destination]
        queue = [(distance[destination], destination)]
        while queue:
            value, target = heapq.heappop(queue)
            if value != distance[target]:
                continue
            for source in self.predecessors[target]:
                candidate = value + costs[source]
                if candidate < distance[source]:
                    distance[source] = candidate
                    heapq.heappush(queue, (candidate, source))
        distance = np.asarray(distance, dtype=np.float64)
        self._distances[destination] = distance
        self.stats['optimize_calls'] += 1
        self.stats['optimize_seconds'] += perf_counter() - start
        return distance

    def next_edge(self, current, destination, legal):
        legal = self._legal(current, destination, legal)
        self.stats['decision_count'] += 1
        distance = self._distance(destination)
        result = min(legal, key=lambda edge: (distance[edge], edge))
        if not math.isfinite(distance[result]):
            self.stats['failed_searches'] += 1
            raise ValueError('Destination unreachable through supplied candidates')
        return result


class AuthorGeneticRouter(RealTimeShortestRouter):
    """Unmodified GA operators and original first-candidate failure fallback.

    Cache age uses elapsed simulation seconds, not number of vehicle decisions.
    Only the pending vehicle is presented to the original get_action method.
    """
    def __init__(self, network, module_path, *, population_size=100, generations=50,
                 crossover_rate=0.8, mutation_rate=0.1, tournament_size=5, cache_ttl=300):
        super().__init__(network)
        module_path = Path(module_path)
        name = '_urban_dreamer_author_ga_' + str(abs(hash(str(module_path.resolve()))))
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.env = SimpleNamespace(
            idx_to_edge=dict(enumerate(network.edge_ids)),
            edges={edge: dict(length=float(network.lengths_m[i]),
                              speed_limit=float(network.speed_limits_mps[i]), to_node=f'turn_{i}')
                   for i, edge in enumerate(network.edge_ids)},
            nodes={f'turn_{i}': dict(outgoing=[network.edge_ids[j] for j in row])
                   for i, row in enumerate(self.successors)},
            edge_states={}, agents={})
        self.env.get_valid_actions = lambda vehicle_id: self._pending_legal_ids
        owner = self

        class InstrumentedRouter(module.GeneticAlgorithmRouter):
            def _optimize_route(self, origin, destination):
                start = perf_counter()
                owner.stats['optimize_calls'] += 1
                try:
                    route = super()._optimize_route(origin, destination)
                finally:
                    owner.stats['optimize_seconds'] += perf_counter() - start
                if len(route) <= 1 and origin != destination:
                    owner.stats['failed_searches'] += 1
                else:
                    index = network.edge_to_index
                    if not route or route[0] != origin or route[-1] != destination or any(
                            index[b] not in owner.successors[index[a]] for a, b in zip(route, route[1:])):
                        raise ValueError('Author GA produced invalid route')
                return route

        self.author = InstrumentedRouter(self.env, population_size=population_size,
            generations=generations, crossover_rate=crossover_rate, mutation_rate=mutation_rate,
            tournament_size=tournament_size, cache_ttl=cache_ttl)
        self.reset()

    def reset(self):
        super().reset()
        if hasattr(self, 'author'):
            self.author.reset()
            self.env.agents.clear()
            self._pending_legal_ids = []
            self.env.edge_states = {edge: SimpleNamespace(travel_time=float(self.costs[i]))
                                    for i, edge in enumerate(self.network.edge_ids)}

    def update_costs(self, costs, timestamp):
        super().update_costs(costs, timestamp)
        self.author._current_step = int(timestamp)
        for i, edge in enumerate(self.network.edge_ids):
            self.env.edge_states[edge].travel_time = float(self.costs[i])

    def next_edge(self, current, destination, legal):
        legal = self._legal(current, destination, legal)
        self.stats['decision_count'] += 1
        self._pending_legal_ids = [self.network.edge_ids[i] for i in legal]
        self.env.agents = {'pending': SimpleNamespace(current_edge_idx=current,
                                                     destination_edge_idx=destination)}
        key = (self.network.edge_ids[current], self.network.edge_ids[destination])
        if (key in self.author._route_cache and self.author.cache_ttl > 0 and
                self.author._current_step - self.author._cache_timestamps.get(key, 0) < self.author.cache_ttl):
            self.stats['cache_hits'] += 1
        action = self.author.get_action('pending', self.env)
        route = self.author._route_cache.get(key, [])
        if len(route) < 2 or route[1] not in self._pending_legal_ids:
            self.stats['fallback_events'] += 1
        self.env.agents.clear()
        return legal[action]

    def complete_route(self, current, destination, candidates):
        """Return the GA's full path for installation until the next decision.

        The author's [origin] failure keeps its first-legal-action fallback;
        only that fallback uses the candidates' free-flow completion suffix.
        Submission removes closed loops so SUMO cannot select a later occurrence
        of the current edge. The original GA route and its cache stay untouched.
        """
        for name in ('full_route_fallbacks', 'loop_erased_routes', 'loop_edges_removed',
                     'fallback_loop_erased_routes'):
            self.stats.setdefault(name, 0)
        selected = self.next_edge(current, destination, candidates.candidates(current, destination))
        key = (self.network.edge_ids[current], self.network.edge_ids[destination])
        cached = self.author._route_cache.get(key, [])
        fallback = list(cached) == [key[0]]
        if fallback:
            route = tuple(candidates.full_route(current, selected, destination))
            self.stats['full_route_fallbacks'] += 1
        else:
            try:
                route = tuple(self.network.edge_to_index[edge] for edge in cached)
            except KeyError as exc:
                raise ValueError('Author GA produced invalid route') from exc
        if (len(route) < 2 or route[0] != current or route[-1] != destination
                or route[1] != selected
                or any(edge < 0 or edge >= len(self.successors) for edge in route)
                or any(b not in self.successors[a] for a, b in zip(route, route[1:]))):
            raise ValueError('Author GA produced invalid route')
        simple = []
        positions = {}
        for edge in route:
            if edge in positions:
                while len(simple) > positions[edge] + 1:
                    del positions[simple.pop()]
            else:
                positions[edge] = len(simple)
                simple.append(edge)
        removed = len(route) - len(simple)
        if removed:
            self.stats['loop_erased_routes'] += 1
            self.stats['loop_edges_removed'] += removed
            self.stats['fallback_loop_erased_routes'] += int(fallback)
        return tuple(simple)
