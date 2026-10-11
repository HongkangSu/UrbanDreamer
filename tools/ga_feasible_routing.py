"""Explicit GA variant with feasible, static-potential route proposals.

Only the author's random route generator is replaced. Uniformly sampled legal
turns must decrease static free-flow distance to the destination. This makes
initial populations and mutation suffixes reachable and cycle-free, but excludes
routes that temporarily increase that potential. Fitness still uses observed
travel times; proposals never use future demand or current traffic costs.
"""
from __future__ import annotations

import heapq
import random
from types import MethodType

import numpy as np

if __package__:
    from .causal_routing_policies import AuthorGeneticRouter
else:
    from causal_routing_policies import AuthorGeneticRouter


class FeasibleGeneticRouter(AuthorGeneticRouter):
    """Author GA with uniform proposals on a static destination-directed DAG."""

    def __init__(self, network, module_path, **kwargs):
        costs = np.asarray(network.free_flow_travel_seconds, dtype=np.float64)
        if costs.shape != (len(network.edge_ids),) or not np.all(np.isfinite(costs)) or np.any(costs <= 0):
            raise ValueError('Static free-flow costs must be finite and strictly positive')
        self._static_costs = costs.copy()
        super().__init__(network, module_path, **kwargs)
        owner = self

        def generate_random_route(author, origin, destination, max_hops=100):
            # Retain the author method signature, but no arbitrary hop cutoff:
            # strictly decreasing potential bounds the route by the graph size.
            return owner._construct_route(origin, destination)

        self.author._generate_random_route = MethodType(generate_random_route, self.author)
        original_selection = self.author._tournament_selection
        calls_per_generation = 2 * (self.author.population_size // 2)

        def tournament_selection(author, population):
            owner.stats['tournament_calls'] += 1
            # Count reproduction generations, not the final fitness-only
            # convergence check. This instrumentation draws no random numbers.
            if calls_per_generation and owner.stats['tournament_calls'] % calls_per_generation == 0:
                owner.stats['evolved_generations'] += 1
            return original_selection(population)

        self.author._tournament_selection = MethodType(tournament_selection, self.author)

    def reset(self):
        super().reset()
        self._static_potentials = {}
        self._descent_successors = {}
        self.stats.update(constructive_routes=0, unreachable_attempts=0,
                          tournament_calls=0, evolved_generations=0)

    def _proposal_graph(self, destination):
        if destination in self._descent_successors:
            return self._static_potentials[destination], self._descent_successors[destination]
        potential = np.full(len(self.successors), np.inf)
        potential[destination] = self._static_costs[destination]
        queue = [(float(potential[destination]), destination)]
        while queue:
            value, target = heapq.heappop(queue)
            if value != potential[target]:
                continue
            for source in self.predecessors[target]:
                candidate = value + self._static_costs[source]
                if candidate < potential[source]:
                    potential[source] = candidate
                    heapq.heappush(queue, (candidate, source))
        descent = tuple(tuple(target for target in row if potential[target] < potential[source])
                        for source, row in enumerate(self.successors))
        self._static_potentials[destination] = potential
        self._descent_successors[destination] = descent
        return potential, descent

    def _construct_route(self, origin, destination):
        index = self.network.edge_to_index
        current, target = index[origin], index[destination]
        potential, descent = self._proposal_graph(target)
        if not np.isfinite(potential[current]):
            self.stats['unreachable_attempts'] += 1
            return None
        route = [origin]
        while current != target:
            if not descent[current]:
                raise ValueError('Reachable edge has no strictly decreasing static-potential successor')
            current = random.choice(descent[current])
            route.append(self.network.edge_ids[current])
        self.stats['constructive_routes'] += 1
        return route
