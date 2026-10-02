"""Shared frozen-window graphs must preserve GNAC outputs and actor gradients."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from urban_dreamer.learning.baselines.gnac import GraphActorCriticMethod, GraphNextHopActor


def inputs():
    torch.manual_seed(104)
    actor = GraphNextHopActor(np.eye(5, k=1, dtype=np.float32))
    features = torch.randn(6, 3, 10)
    mask = torch.tensor([[True, True, False]] * 6)
    edges = torch.tensor([[1, 3, -1]] * 6)
    current = torch.tensor([0, 1, 2, 3, 0, 1])
    flow = torch.randn(5, 8)
    return actor, features, mask, edges, current, flow


def assert_gradients_match(actor, reference, actual, expected, mask):
    torch.testing.assert_close(actual, expected)
    weights = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) + 1
    (actual[mask] * weights[mask]).mean().backward()
    (expected[mask] * weights[mask]).mean().backward()
    for (name, parameter), (_, other) in zip(actor.named_parameters(), reference.named_parameters()):
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-5, atol=2e-6, msg=name)


def test_shared_flow_encodes_once_with_equivalent_logits_and_gradients():
    actor, features, mask, edges, current, flow = inputs()
    reference = deepcopy(actor)
    expected = reference.forward_logits(features, mask, edges, current, flow.expand(6, -1, -1))
    with patch.object(actor, '_edge_embeddings', wraps=actor._edge_embeddings) as encode:
        actual = actor.forward_logits(features, mask, edges, current, flow)
    assert encode.call_count == 1
    assert encode.call_args.args[0].numel() == flow.numel()
    assert_gradients_match(actor, reference, actual, expected, mask)


def test_interleaved_window_rows_share_graphs_and_preserve_order_and_gradients():
    actor, features, mask, edges, current, flow = inputs()
    reference = deepcopy(actor)
    window_ids = [7, 2, 7, 2, 7, 2]
    flows = [flow if window == 7 else flow + 0.4 for window in window_ids]
    rows = [dict(features=features[i].numpy(), mask=mask[i].numpy(),
                 edge_indices=edges[i].numpy(), current_edge=int(current[i]),
                 flow=flows[i].numpy().copy(), window=window)
            for i, window in enumerate(window_ids)]
    method = GraphActorCriticMethod.__new__(GraphActorCriticMethod)
    method.actor, method.device = actor, torch.device('cpu')
    expected = reference.forward_logits(features, mask, edges, current, torch.stack(flows))
    with patch.object(actor, '_edge_embeddings', wraps=actor._edge_embeddings) as encode:
        actual = method._compute_logits(rows)
    assert encode.call_count == 1
    assert encode.call_args.args[0].numel() == 2 * flow.numel()
    assert_gradients_match(actor, reference, actual, expected, mask)


def test_rows_without_window_keep_distinct_flow_semantics():
    actor, features, mask, edges, current, flow = inputs()
    flows = torch.stack([flow + i for i in range(6)])
    rows = [dict(features=features[i].numpy(), mask=mask[i].numpy(),
                 edge_indices=edges[i].numpy(), current_edge=int(current[i]), flow=flows[i].numpy())
            for i in range(6)]
    method = GraphActorCriticMethod.__new__(GraphActorCriticMethod)
    method.actor, method.device = actor, torch.device('cpu')
    expected = actor.forward_logits(features, mask, edges, current, flows)
    torch.testing.assert_close(method._compute_logits(rows), expected)
