"""Lossless suppression must preserve choices, protected nodes and route costs."""

import heapq

import pytest

from tools.decision_graph_compression import compress_graph, expand_compressed_path


def test_chain_and_protected_interior():
    adjacency = [[1], [2], [3], []]
    graph = compress_graph(adjacency, {0, 2})
    assert graph["nodes"] == [0, 2, 3]
    assert graph["arcs"] == [
        {"source": 0, "target": 2, "path": (0, 1, 2)},
        {"source": 2, "target": 3, "path": (2, 3)},
    ]
    assert expand_compressed_path(graph["arcs"]) == (0, 1, 2, 3)


def test_merges_can_be_suppressed_and_parallel_choices_survive():
    graph = compress_graph([[1, 2], [3], [3], [4], []], {0, 4})
    assert graph["nodes"] == [0, 4]
    assert [arc["path"] for arc in graph["arcs"]] == [
        (0, 1, 3, 4), (0, 2, 3, 4)
    ]


@pytest.mark.parametrize(
    "adjacency,protected,nodes,anchors,paths",
    [
        ([[1], [2], [0]], set(), [0], [0], [(0, 1, 2, 0)]),
        ([[1], [2], [0]], {1}, [1], [], [(1, 2, 0, 1)]),
        ([[0]], set(), [0], [0], [(0, 0)]),
        ([[1, 3], [2], [1], []], set(), [0, 1, 3], [1],
         [(0, 1), (0, 3), (1, 2, 1)]),
        ([[], []], set(), [0, 1], [], []),
    ],
)
def test_cycles_deadends_and_isolated_nodes(adjacency, protected, nodes, anchors, paths):
    graph = compress_graph(adjacency, protected)
    assert graph["nodes"] == nodes
    assert graph["cycle_anchors"] == anchors
    assert [arc["path"] for arc in graph["arcs"]] == paths


def test_passenger_filtering_changes_degree_without_illegal_arcs():
    graph = compress_graph([[1, 2], [2], []], {0}, [True, False, True])
    assert graph["nodes"] == [0, 2]
    assert graph["arcs"] == [{"source": 0, "target": 2, "path": (0, 2)}]


@pytest.mark.parametrize(
    "adjacency,protected,allowed",
    [([[2], []], set(), None), ([[-1]], set(), None),
     ([[]], {1}, None), ([[]], {0}, [False]), ([[]], set(), [])],
)
def test_invalid_inputs_are_rejected(adjacency, protected, allowed):
    with pytest.raises(ValueError):
        compress_graph(adjacency, protected, allowed)


def test_expansion_rejects_disconnected_or_malformed_arcs():
    assert expand_compressed_path([]) == ()
    with pytest.raises(ValueError):
        expand_compressed_path([{"source": 0, "target": 2, "path": (0, 1)}])
    with pytest.raises(ValueError):
        expand_compressed_path([
            {"source": 0, "target": 1, "path": (0, 1)},
            {"source": 2, "target": 3, "path": (2, 3)},
        ])


def _distances(edges, source):
    distance = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for target, weight in edges.get(node, []):
            candidate = cost + weight
            if candidate < distance.get(target, float("inf")):
                distance[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return distance


def test_reachability_weighted_shortest_costs_and_decision_points_preserved():
    adjacency = [[1, 2], [3], [3], [4, 5], [0], [], [7], [6], []]
    protected = {0, 2, 4, 5, 6, 8}
    graph = compress_graph(adjacency, protected)
    weights = {(u, v): 1 + u * 0.7 + v * 0.3
               for u, targets in enumerate(adjacency) for v in targets}
    original = {u: [(v, weights[u, v]) for v in targets]
                for u, targets in enumerate(adjacency)}
    compressed = {node: [] for node in graph["nodes"]}
    for arc in graph["arcs"]:
        path = arc["path"]
        assert all(v in adjacency[u] for u, v in zip(path, path[1:]))
        weight = sum(weights[u, v] for u, v in zip(path, path[1:]))
        compressed[arc["source"]].append((arc["target"], weight))
    for source in protected:
        before, after = _distances(original, source), _distances(compressed, source)
        for target in protected:
            assert after.get(target, float("inf")) == pytest.approx(
                before.get(target, float("inf")))
    for node in graph["nodes"]:
        assert len(compressed[node]) == len(adjacency[node])
