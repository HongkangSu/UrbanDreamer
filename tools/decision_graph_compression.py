"""Offline lossless routing compression by suppressing deterministic states.

Protect every origin/destination that must remain addressable. Arcs retain their
full original node sequence, including distinct choices with the same endpoint.
This utility does not modify SUMO maps or the runtime policy action space.
"""

from collections.abc import Sequence


def compress_graph(
    adjacency: Sequence[Sequence[int]],
    protected: set[int],
    allowed: Sequence[bool] | None = None,
) -> dict:
    """Suppress unprotected nodes with exactly one allowed successor.

    Successors outside the allowed subgraph are explicitly filtered. Each closed
    deterministic cycle without a retained node receives its minimum-ID anchor.
    Inputs must use contiguous integer node IDs in ``range(len(adjacency))``.
    """
    size = len(adjacency)
    if allowed is not None and len(allowed) != size:
        raise ValueError("allowed must have one entry per node")
    enabled = [True] * size if allowed is None else list(allowed)
    for successors in adjacency:
        if any(not isinstance(node, int) or not 0 <= node < size for node in successors):
            raise ValueError("adjacency contains an invalid node index")
    if any(not isinstance(node, int) or not 0 <= node < size for node in protected):
        raise ValueError("protected contains an invalid node index")
    if any(not enabled[node] for node in protected):
        raise ValueError("protected nodes must be allowed")

    successors = [tuple(target for target in targets if enabled[target])
                  if enabled[node] else ()
                  for node, targets in enumerate(adjacency)]
    retained = {node for node in range(size)
                if enabled[node] and (len(successors[node]) != 1 or node in protected)}
    explored = set(retained)
    anchors = []
    for start in range(size):
        if not enabled[start] or start in explored:
            continue
        path = []
        positions = {}
        node = start
        while node not in explored and node not in positions:
            positions[node] = len(path)
            path.append(node)
            node = successors[node][0]
        if node in positions:
            anchor = min(path[positions[node]:])
            retained.add(anchor)
            anchors.append(anchor)
        explored.update(path)

    arcs = []
    for source in sorted(retained):
        for first_hop in successors[source]:
            path = [source, first_hop]
            while path[-1] not in retained:
                path.append(successors[path[-1]][0])
            arcs.append({"source": source, "target": path[-1], "path": tuple(path)})
    return {"nodes": sorted(retained), "arcs": arcs, "cycle_anchors": sorted(anchors)}


def expand_compressed_path(arcs: Sequence[dict]) -> tuple[int, ...]:
    """Join expanded arc paths, rejecting mismatched endpoints or disconnected arcs."""
    result = []
    for arc in arcs:
        path = tuple(arc["path"])
        if len(path) < 2 or path[0] != arc["source"] or path[-1] != arc["target"]:
            raise ValueError("arc path does not match its source and target")
        if result and result[-1] != path[0]:
            raise ValueError("compressed route contains disconnected arcs")
        result.extend(path[1:] if result else path)
    return tuple(result)
