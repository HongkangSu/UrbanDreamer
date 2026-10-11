"""Whole-window Q batching preserves each decision's causal conditions."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.event_critic import build_event_q_inputs
from urban_dreamer.models.world_model import EventActionEncoder


def case():
    torch.manual_seed(731)
    encoder = EventActionEncoder(adjacency=torch.ones(5, 5) - torch.eye(5),
                                 static_features=torch.randn(5, 3))
    model = SimpleNamespace(dynamics=SimpleNamespace(event_encoder=encoder))
    groups = [[ActionEvent(str(i), j * 4 + i, float(t), i, (i + 1) % 5, 4)
               for i in range(4)] for j, t in enumerate((2, 7, 15, 27))]
    kwargs = dict(z=torch.randn(1, 5, 64), observations=torch.randn(4, 3, 10),
                  candidate_edges=torch.tensor([[(i + j) % 5 for j in (1, 2, 3)] for i in range(4)]),
                  mask=torch.ones(4, 3, dtype=torch.bool), current_edges=torch.arange(4),
                  destination_edges=torch.full((4,), 4), context=torch.tensor([.2, .5, 0., 0.]), window_start=0.)
    return model, groups, kwargs


def build(model, records, group, kwargs, **extra):
    return build_event_q_inputs(model, events=pack_action_events([[records]], window_starts=[[kwargs["window_start"]]]),
        event_indices=torch.tensor([e.event_index for e in group]),
        decision_times=torch.tensor([e.decision_time for e in group]), **kwargs, **extra)


def test_future_records_do_not_change_grouped_prefixes():
    model, groups, kwargs = case()
    group = groups[1]
    original = build(model, sum(groups, []), group, kwargs)
    modified = groups[:2] + [[replace(e, next_edge=(e.next_edge + 1) % 5) for e in g]
                             for g in groups[2:]]
    torch.testing.assert_close(build(model, sum(modified, []), group, kwargs), original, atol=0., rtol=0.)



def test_whole_window_batch_matches_decision_time_groups():
    model, groups, kwargs = case()
    records = sum(groups, [])
    whole = {**kwargs, **{key: value.repeat(len(groups), *([1] * (value.ndim - 1)))
                            for key, value in kwargs.items()
                            if key in ('observations', 'candidate_edges', 'mask',
                                       'current_edges', 'destination_edges')}}
    actual = build(model, records, records, whole)
    expected = torch.cat([build(model, records, group, kwargs) for group in groups])
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)


def test_duplicate_stored_event_identity_is_rejected():
    model, groups, kwargs = case()
    with pytest.raises(ValueError, match='exactly one'):
        build(model, groups[0] + [groups[0][0]], groups[0], kwargs)


def test_grouped_leave_one_out_preserves_own_sample_invariance_and_vehicle_history():
    model, groups, kwargs = case()
    history = groups[0] + groups[1]
    original = build(model, history, groups[1], kwargs)
    changed = history.copy()
    changed[4] = replace(changed[4], next_edge=3)
    modified = build(model, changed, groups[1], kwargs)
    torch.testing.assert_close(original[0], modified[0], atol=0., rtol=0.)
    assert not torch.equal(original[1], modified[1])
    without_history = build(model, groups[1], groups[1], kwargs)
    assert not torch.equal(original, without_history)


def test_grouped_inputs_reject_illegal_candidates_and_have_no_world_model_gradient():
    model, groups, kwargs = case()
    actual = build(model, sum(groups, []), groups[1], kwargs)
    assert not actual.requires_grad
    assert all(p.grad is None for p in model.dynamics.event_encoder.parameters())
    bad = {**kwargs, 'candidate_edges': kwargs['candidate_edges'].clone()}
    bad['candidate_edges'][0, 0] = 0
    with pytest.raises(ValueError, match='illegal turn'):
        build(model, sum(groups, []), groups[1], bad)


def test_repeated_episode_ids_are_resolved_without_cross_window_history():
    model, groups, kwargs = case()
    groups = [groups[0], [replace(e, event_index=i) for i, e in enumerate(groups[1])]]
    records = sum(groups, [])
    doubled = {**kwargs, **{key: value.repeat(2, *([1] * (value.ndim - 1)))
                           for key, value in kwargs.items()
                           if key in ('observations', 'candidate_edges', 'mask',
                                      'current_edges', 'destination_edges')}}
    actual = build_event_q_inputs(model, events=pack_action_events([[groups[0]], [groups[1]]]),
        event_indices=torch.tensor([e.event_index for e in records]),
        decision_times=torch.tensor([e.decision_time for e in records]), **doubled)
    expected = torch.cat([build(model, group, group, kwargs) for group in groups])
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=3e-6)
    # An identical identity in two windows remains ambiguous, as before.
    with pytest.raises(ValueError, match='exactly one'):
        build_event_q_inputs(model, events=pack_action_events([[groups[0]], [groups[0]]]),
            event_indices=torch.tensor([e.event_index for e in groups[0]]),
            decision_times=torch.tensor([e.decision_time for e in groups[0]]), **kwargs)


def test_repeated_queries_for_one_event_do_not_remove_other_vehicle_events():
    model, groups, kwargs = case()
    row = {**kwargs, **{key: value[:1] for key, value in kwargs.items()
                       if key in ('observations', 'candidate_edges', 'mask',
                                  'current_edges', 'destination_edges')}}
    duplicated = {**row, **{key: value.repeat(3, *([1] * (value.ndim - 1)))
                            for key, value in row.items()
                            if key in ('observations', 'candidate_edges', 'mask',
                                       'current_edges', 'destination_edges')}}
    original = build(model, sum(groups, []), [groups[1][0]], row)
    repeated = build(model, sum(groups, []), [groups[1][0]] * 3, duplicated)
    torch.testing.assert_close(repeated, original.expand(3, -1, -1), atol=3e-7, rtol=3e-6)


def test_uneven_groups_preserve_causal_inputs():
    model, groups, kwargs = case()
    records = [replace(groups[0][i % 4], event_index=i, decision_time=0.) for i in range(128)]
    records += [replace(groups[0][i % 4], event_index=128+i, decision_time=(i+1)/2.)
                for i in range(33)]
    indices = torch.arange(len(records)) % 4
    all_kwargs = {**kwargs, **{key: value[indices] for key, value in kwargs.items()
                             if key in ('observations', 'candidate_edges', 'mask',
                                        'current_edges', 'destination_edges')}}
    actual = build(model, records, records, all_kwargs)
    for index in (0, 127, 128, 160):
        single = {**all_kwargs, **{key: value[index:index+1] for key, value in all_kwargs.items()
                                 if key in ('observations', 'candidate_edges', 'mask',
                                            'current_edges', 'destination_edges')}}
        expected = build(model, records, [records[index]], single)
        torch.testing.assert_close(actual[index:index+1], expected, atol=3e-7, rtol=3e-6)


def test_large_unique_id_search_matches_small_query_chunks_and_rejects_duplicates():
    model, groups, kwargs = case()
    records = [replace(groups[0][i % 4], event_index=i, decision_time=float(i // 20))
               for i in range(520)]
    indices = torch.arange(len(records)) % 4
    expanded_keys = ('observations', 'candidate_edges', 'mask', 'current_edges', 'destination_edges')
    whole = {**kwargs, **{key: kwargs[key][indices] for key in expanded_keys}}
    actual = build(model, records, records, whole)
    pieces = []
    for start in range(0, len(records), 64):
        piece = {**whole, **{key: whole[key][start:start+64] for key in expanded_keys}}
        pieces.append(build(model, records, records[start:start+64], piece))
    torch.testing.assert_close(actual, torch.cat(pieces), atol=5e-7, rtol=5e-6)
    with pytest.raises(ValueError, match='exactly one'):
        build(model, records + [records[0]], records, whole)
