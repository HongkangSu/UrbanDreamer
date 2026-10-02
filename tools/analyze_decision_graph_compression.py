#!/usr/bin/env python3
"""Export lossless road-edge decision graphs and verify city OD routing equivalence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.decision_graph_compression import compress_graph
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
from urban_dreamer.workflow.config import load_config

CITIES = ('ingolstadt_custom', 'gretz_armainvilliers', 'fontenay_en_parisis')


def reverse_distances(destination, predecessors):
    distances = {node: float('inf') for node in predecessors}
    distances[destination] = 0.
    queue = [(0., destination)]
    while queue:
        distance, target = heapq.heappop(queue)
        if distance != distances[target]:
            continue
        for source, cost in predecessors[target]:
            candidate = distance + cost
            if candidate < distances[source]:
                distances[source] = candidate
                heapq.heappush(queue, (candidate, source))
    return distances


def analyze(city, directory):
    config = load_config(ROOT / 'configs' / f'{city}.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    network, candidates = topology.network, NextHopCandidates(topology.network)
    od_edges = {network.edge_to_index[name] for name in (*topology.origins, *topology.destinations)}
    incident_edges = {network.edge_to_index[name.strip()] for name in config.simulation.incident_edge.split(',')
                      if name.strip()}
    indegree = np.zeros(network.edge_count, dtype=np.int64)
    for targets in network.adjacency_list:
        for target in targets:
            indegree[target] += 1
    # Keep source feeders represented even when they are not current OD origins.
    sources = set(map(int, np.flatnonzero(network.passenger_allowed & (indegree == 0))))
    protected = {i for i in od_edges | incident_edges | sources if network.passenger_allowed[i]}
    graph = compress_graph(network.adjacency_list, protected, allowed=network.passenger_allowed)
    retained = set(graph['nodes'])
    outgoing = {node: [] for node in retained}
    predecessors = {node: [] for node in retained}
    covered = set(retained)
    for arc_id, arc in enumerate(graph['arcs']):
        path = arc['path']
        assert path[0] == arc['source'] and path[-1] == arc['target']
        assert not (set(path[1:-1]) & retained)
        assert all(v in network.adjacency_list[u] for u, v in zip(path, path[1:]))
        cost = sum(float(network.free_flow_travel_seconds[v])
                   + candidates.junction(u, v)[0] / candidates.junction(u, v)[1]
                   for u, v in zip(path, path[1:]))
        length = sum(float(network.lengths_m[v]) + candidates.junction(u, v)[0]
                     for u, v in zip(path, path[1:]))
        arc.update(id=arc_id, free_flow_seconds=cost, distance_m=length,
                   edge_ids=[network.edge_ids[i] for i in path])
        outgoing[arc['source']].append(arc)
        predecessors[arc['target']].append((arc['source'], cost))
        covered.update(path)
    assert covered == set(map(int, np.flatnonzero(network.passenger_allowed)))
    original_arc_count = sum(len(network.adjacency_list[i]) for i in covered)
    for node in retained:
        assert len(outgoing[node]) == len(network.adjacency_list[node])
    ods = Counter((network.edge_to_index[topology.origins[v.origin_index]],
                   network.edge_to_index[topology.destinations[v.destination_index]]) for v in topology.vehicles)
    destinations = sorted({d for _, d in ods})
    statistics = []
    maximum_error = 0.
    checked_states = 0
    for destination in destinations:
        original_distance = candidates.costs(destination)
        compressed_distance = reverse_distances(destination, predecessors)
        indices = np.asarray(graph['nodes'], dtype=np.int64)
        actual = np.asarray([compressed_distance[i] for i in indices])
        expected = original_distance[indices]
        np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
        finite = np.isfinite(expected)
        np.testing.assert_allclose(actual[finite], expected[finite], rtol=1e-10, atol=1e-7)
        maximum_error = max(maximum_error, float(np.max(np.abs(actual[finite] - expected[finite]), initial=0.)))
        checked_states += len(indices)
        for (origin, dest), weight in ods.items():
            if dest != destination:
                continue
            node = origin
            path = [node]
            original_decisions = 0
            while node != destination:
                choices = candidates.candidates(node, destination)
                original_decisions += len(choices) > 1
                node = min(choices, key=lambda v: (
                    candidates.junction(node, v)[0] / candidates.junction(node, v)[1]
                    + float(network.free_flow_travel_seconds[v]) + original_distance[v], v))
                path.append(node)
                if len(path) > network.edge_count + 1:
                    raise AssertionError('positive-cost shortest path cycled')
            # Collapse this same shortest path, preserving parallel alternatives.
            position = 0
            macro_hops = compressed_decisions = 0
            while position < len(path) - 1:
                source = path[position]
                arcs = outgoing[source]
                legal = [a for a in arcs if np.isfinite(compressed_distance[a['target']])]
                assert len(legal) == len(candidates.candidates(source, destination))
                compressed_decisions += len(legal) > 1
                matching = [a for a in arcs if a['path'][1] == path[position + 1]]
                assert len(matching) == 1
                segment = matching[0]['path']
                assert list(segment) == path[position:position + len(segment)]
                position += len(segment) - 1
                macro_hops += 1
            assert original_decisions == compressed_decisions
            statistics.extend([(len(path) - 1, macro_hops, original_decisions, compressed_decisions)] * weight)
    values = np.asarray(statistics, dtype=np.float64)
    arcs_by_endpoints = Counter((a['source'], a['target']) for a in graph['arcs'])
    report = dict(network=city, registry_edges=network.edge_count, passenger_nodes=len(covered),
        passenger_arcs=original_arc_count, compressed_nodes=len(retained), compressed_arcs=len(graph['arcs']),
        removed_nodes=len(covered) - len(retained), node_reduction_fraction=1 - len(retained) / len(covered),
        arc_reduction_fraction=1 - len(graph['arcs']) / original_arc_count,
        protected_od_nodes=len(od_edges & covered), protected_incident_nodes=len(incident_edges & covered),
        protected_source_nodes=len(sources),
        cycle_anchors=len(graph['cycle_anchors']),
        parallel_endpoint_pairs=sum(count > 1 for count in arcs_by_endpoints.values()),
        maximum_original_candidates=candidates.max_candidates,
        maximum_compressed_candidates=max(map(len, outgoing.values()), default=0),
        od_pairs_checked=len(ods), destinations_checked=len(destinations),
        retained_state_destination_pairs_checked=checked_states, maximum_cost_error_seconds=maximum_error,
        expanded_arcs_all_legal=True, passenger_node_coverage_complete=True,
        mean_original_path_transitions=float(values[:, 0].mean()),
        mean_compressed_path_transitions=float(values[:, 1].mean()),
        mean_original_policy_decisions=float(values[:, 2].mean()),
        mean_compressed_policy_decisions=float(values[:, 3].mean()),
        policy_decision_reduction_fraction=0., input_hashes=prepared.input_hashes)
    payload = dict(network=city, compression='lossless deterministic-successor chain contraction',
        cost_convention='Sum successor-road travel time plus junction time; exclude source road, include target road.',
        protected_node_ids=sorted(protected), nodes=[dict(index=i, edge_id=network.edge_ids[i]) for i in graph['nodes']],
        cycle_anchors=graph['cycle_anchors'], arcs=graph['arcs'], statistics=report)
    (directory / f'{city}.decision_graph.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    return report


def plot_reports(reports, directory):
    import os
    os.environ.setdefault('MPLCONFIGDIR', str(directory / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(reports))
    labels = ['Ingolstadt', 'Gretz', 'Fontenay']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, before, after, title in zip(axes,
        ['passenger_nodes', 'mean_original_path_transitions', 'mean_original_policy_decisions'],
        ['compressed_nodes', 'mean_compressed_path_transitions', 'mean_compressed_policy_decisions'],
        ['Passenger decision-graph nodes', 'Shortest-route graph transitions', 'Actual choice states on shortest routes']):
        left = ax.bar(x - .2, [r[before] for r in reports], width=.4, label='Before')
        right = ax.bar(x + .2, [r[after] for r in reports], width=.4, label='Lossless compression')
        ax.bar_label(left, fmt='%.1f', fontsize=8)
        ax.bar_label(right, fmt='%.1f', fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=10)
        ax.margins(y=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Lossless compression preserves every branching choice')
    fig.tight_layout()
    fig.savefig(directory / 'compression_comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('refusing to overwrite an existing compression analysis')
    directory.mkdir(parents=True, exist_ok=True)
    reports = [analyze(city, directory) for city in CITIES]
    summary = dict(policy='Preserve every legal passenger transition, all permitted configured OD edges, source feeders and incident edges. No turn bans or path pruning.',
        measurement='Path averages weighted by agents.csv OD counts. Policy decisions count destination-conditioned candidate sets of size >1. No running experiment or network XML is modified.',
        reports=reports,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ['tools/decision_graph_compression.py', 'tools/analyze_decision_graph_compression.py']})
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot_reports(reports, directory)


if __name__ == '__main__':
    main()
