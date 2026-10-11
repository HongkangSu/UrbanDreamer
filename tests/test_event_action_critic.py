from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.event_critic import (
    EVENT_Q_FEATURE_DIM, EventActionCritic, build_event_q_inputs,
    detached_counterfactual_baseline, executed_action_q_loss,
)
from urban_dreamer.models.world_model import EventActionEncoder


@pytest.fixture
def conditions():
    torch.manual_seed(41)
    adjacency = torch.tensor([[0., 1., 1., 1.], [0., 0., 1., 1.],
                              [0., 0., 0., 1.], [0., 0., 0., 0.]])
    encoder = EventActionEncoder(adjacency=adjacency, static_features=torch.randn(4, 3))
    model = SimpleNamespace(dynamics=SimpleNamespace(event_encoder=encoder))
    records = [ActionEvent('same', 0, 2., 0, 1, 3),
               ActionEvent('same', 1, 12., 1, 2, 3),
               ActionEvent('other', 2, 12., 0, 2, 3),
               ActionEvent('same', 3, 22., 2, 3, 3)]
    kwargs = dict(z=torch.randn(1, 4, 64, requires_grad=True),
                  observations=torch.randn(1, 3, 10, requires_grad=True),
                  candidate_edges=torch.tensor([[2, 3, -1]]),
                  mask=torch.tensor([[True, True, False]]),
                  current_edges=torch.tensor([1]), destination_edges=torch.tensor([3]),
                  decision_times=torch.tensor([12.]), event_indices=torch.tensor([1]),
                  context=torch.tensor([[.2, .5, 0., 0.]], requires_grad=True), window_start=0.)
    return model, records, kwargs


def prepare(model, records, kwargs):
    return build_event_q_inputs(model, events=pack_action_events([[records]]), **kwargs)


def test_future_events_and_current_sample_do_not_change_causal_candidate_inputs(conditions):
    model, records, kwargs = conditions
    original = prepare(model, records, kwargs)
    altered = records.copy()
    altered[1] = replace(altered[1], next_edge=3)
    altered[3] = replace(altered[3], vehicle_id='future', current_edge=0, next_edge=1,
                         destination_edge=2, decision_time=27.)
    torch.testing.assert_close(original, prepare(model, altered, kwargs), atol=0., rtol=0.)
    assert original.shape == (1, 3, EVENT_Q_FEATURE_DIM)
    assert not original.requires_grad
    assert original[0, 2].eq(0).all()


def test_prior_events_of_same_vehicle_and_other_simultaneous_actions_are_retained(conditions):
    model, records, kwargs = conditions
    original = prepare(model, records, kwargs)
    without_prior = prepare(model, records[1:], kwargs)
    without_other = prepare(model, [records[0], records[1], records[3]], kwargs)
    assert not torch.equal(original, without_prior)
    assert not torch.equal(original, without_other)
    torch.testing.assert_close(original, prepare(model, list(reversed(records)), kwargs))


def test_batched_candidate_replacement_matches_full_raw_reaggregation(conditions):
    model, records, kwargs = conditions
    features = prepare(model, records, kwargs)
    encoder = model.dynamics.event_encoder
    for column, edge in enumerate([2, 3]):
        simultaneous = [replace(records[1], next_edge=edge), records[2]]
        raw = encoder.aggregate(pack_action_events([[simultaneous]]))
        expected = encoder.compact_readout(raw)[0, 0]
        torch.testing.assert_close(features[0, column, -64:], expected)


def test_window_start_latent_broadcasts_and_candidate_permutation_is_equivariant(conditions):
    model, records, kwargs = conditions
    keys = ('observations', 'candidate_edges', 'mask', 'current_edges', 'destination_edges',
            'decision_times', 'context')
    doubled = {**kwargs, **{key: kwargs[key].expand(2, *kwargs[key].shape[1:]) for key in keys}}
    doubled['current_edges'] = torch.tensor([1, 0])
    doubled['event_indices'] = torch.tensor([1, 2])
    actual = prepare(model, records, doubled)
    assert actual.shape == (2, 3, EVENT_Q_FEATURE_DIM)
    critic = EventActionCritic()
    permutation = torch.tensor([2, 0, 1])
    torch.testing.assert_close(critic(actual[:, permutation], doubled['mask'][:, permutation]),
                               critic(actual, doubled['mask'])[:, permutation])


def test_baseline_excludes_illegal_candidates_and_is_fixed_after_q_update():
    actor_logits = torch.tensor([[1., 0., 80.]], requires_grad=True)
    probabilities = actor_logits.softmax(-1)
    values = torch.tensor([[2., 6., 1e10]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    baseline = detached_counterfactual_baseline(probabilities, values, mask)
    expected = (torch.tensor([1., 0.]).softmax(-1) * torch.tensor([2., 6.])).sum()
    torch.testing.assert_close(baseline[0], expected)
    assert not baseline.requires_grad
    saved = baseline.clone()
    with torch.no_grad():
        values.add_(1000.)
        actor_logits.add_(20.)
    torch.testing.assert_close(baseline, saved, atol=0., rtol=0.)
    assert actor_logits.grad is None


def test_executed_q_loss_has_no_world_model_actor_or_return_gradient(conditions):
    model, records, kwargs = conditions
    critic = EventActionCritic()
    features = prepare(model, records, kwargs)
    values = critic(features, kwargs['mask'])
    returns = torch.tensor([-3.], requires_grad=True)
    loss = executed_action_q_loss(values, torch.tensor([0]), returns, kwargs['mask'])
    loss.backward()
    assert any(parameter.grad is not None for parameter in critic.parameters())
    assert all(parameter.grad is None for parameter in model.dynamics.event_encoder.parameters())
    assert kwargs['z'].grad is None
    assert kwargs['observations'].grad is None
    assert kwargs['context'].grad is None
    assert returns.grad is None


def test_only_executed_legal_q_receives_supervision():
    values = torch.tensor([[2., 6., 100.]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    executed_action_q_loss(values, torch.tensor([1]), torch.tensor([-3.]), mask).backward()
    torch.testing.assert_close(values.grad, torch.tensor([[0., 18., 0.]]))
    with pytest.raises(ValueError, match='legal'):
        executed_action_q_loss(values, torch.tensor([2]), torch.tensor([-3.]), mask)
    with pytest.raises(ValueError, match='legal candidate'):
        EventActionCritic()(torch.zeros(1, 2, EVENT_Q_FEATURE_DIM), torch.zeros(1, 2, dtype=torch.bool))


def test_missing_event_or_boundary_event_fails_explicitly(conditions):
    model, records, kwargs = conditions
    with pytest.raises(ValueError, match='exactly one'):
        prepare(model, [records[0], records[2]], kwargs)
    kwargs['decision_times'] = torch.tensor([30.])
    with pytest.raises(ValueError, match='half-open'):
        prepare(model, records, kwargs)


def test_empty_causal_background_is_valid_and_candidates_do_not_run_graph_or_dynamics(conditions):
    model, records, kwargs = conditions

    def forbidden(*args, **kwargs):
        pytest.fail('preparing candidate Q conditions must not invoke a full graph or dynamics pass')

    model.dynamics.transition = forbidden
    model.dynamics.action_encoder = forbidden
    features = prepare(model, [records[1]], kwargs)
    assert torch.isfinite(features).all()
    # Fixed prefix and simultaneous-background segments are both empty.
    assert features[0, :2, 147:275].eq(0).all()
    assert not features[0, :2, -64:].eq(0).all()


def test_marking_a_non_successor_legal_fails_explicitly(conditions):
    model, records, kwargs = conditions
    kwargs['candidate_edges'] = torch.tensor([[0, 3, -1]])
    with pytest.raises(ValueError, match='illegal turn'):
        prepare(model, records, kwargs)
