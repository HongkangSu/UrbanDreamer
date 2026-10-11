from dataclasses import replace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.learning.replay import MacroTransition, ReplayBuffer
from urban_dreamer.models.world_model import Dynamics, EventActionEncoder, recursive_fm_loss


def encoder():
    adjacency = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    return EventActionEncoder(adjacency=adjacency, static_features=torch.arange(12.).reshape(3, 4))


def events():
    return (ActionEvent('v', 0, 2., 0, 1, 2), ActionEvent('v', 1, 12., 1, 2, 2),
            ActionEvent('w', 2, 12., 0, 2, 1))


def test_sparse_windows_retain_empty_windows_and_all_same_vehicle_events():
    rows = events()
    packed = pack_action_events([[rows, ()], [(), ()]])
    assert len(packed) == 3
    assert packed.vehicle_ids == ('v', 'v', 'w')
    assert packed.batch_size == packed.time_steps == 2
    assert len(packed.for_step(1)) == 0
    assert packed.for_step(1).time_steps == 1
    with pytest.raises(ValueError, match='half-open'):
        pack_action_events([[(ActionEvent('v', 0, 30., 0, 1, 2),)]])


def test_joint_encoding_is_invariant_to_row_order_and_additive_before_readout():
    model = encoder()
    rows = events()
    packed = pack_action_events([[rows]])
    permuted = pack_action_events([[rows[::-1]]])
    raw = model.aggregate(packed)
    raw_permuted = model.aggregate(permuted)
    assert torch.equal(raw.count, raw_permuted.count)
    assert torch.allclose(raw.feature_sum, raw_permuted.feature_sum)
    assert torch.allclose(model(packed), model(permuted))
    doubled = model.aggregate(pack_action_events([[rows + rows]]))
    assert torch.equal(doubled.count, raw.count * 2)
    assert torch.allclose(doubled.feature_sum, raw.feature_sum * 2)
    empty = model.aggregate(pack_action_events([[()]]))
    assert torch.equal(model.readout(empty), torch.zeros(1, 1, 3, 64))
    assert torch.equal(model.compact_readout(empty), torch.zeros(1, 1, 64))


def test_encoder_distinguishes_entry_destination_and_time_bucket():
    model = encoder()
    first = ActionEvent('v', 0, 2., 0, 1, 2)
    base = model(pack_action_events([[(first,)]]))
    for changed in (replace(first, current_edge=2), replace(first, destination_edge=0),
                    replace(first, decision_time=12.)):
        assert not torch.allclose(base, model(pack_action_events([[(changed,)]])))


def test_single_event_replacement_matches_reaggregation_and_keeps_same_vehicle_history():
    model = encoder()
    rows = events()
    packed = pack_action_events([[rows]])
    raw = model.aggregate(packed)
    replaced = model.replace_event(raw, packed, 1, 0)
    complete = model.aggregate(pack_action_events([[(rows[0], replace(rows[1], next_edge=0), rows[2])]]))
    assert torch.equal(replaced.count, complete.count)
    assert torch.allclose(replaced.feature_sum, complete.feature_sum, atol=1e-6)
    prefix, simultaneous = model.background(packed, 1)
    assert prefix.count.sum() == simultaneous.count.sum() == 1
    expected = model.aggregate(pack_action_events([[(rows[0],)]]))
    assert torch.allclose(prefix.feature_sum, expected.feature_sum)
    changed_rows = (rows[0], replace(rows[1], next_edge=0), rows[2], ActionEvent('v', 3, 22., 2, 0, 2))
    changed_prefix, changed_simultaneous = model.background(pack_action_events([[changed_rows]]), 1)
    assert torch.equal(changed_prefix.count, prefix.count)
    assert torch.allclose(changed_prefix.feature_sum, prefix.feature_sum)
    assert torch.allclose(changed_simultaneous.feature_sum, simultaneous.feature_sum)


def test_event_protocol_requires_explicit_windows_and_fm_reaches_event_parameters():
    torch.set_num_threads(1)
    adjacency = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    dynamics = Dynamics(adjacency=adjacency, static_features=torch.ones(3, 4), event_protocol=True)
    action = torch.zeros(1, 2, 3)
    z = torch.randn(1, 3, 3, 64)
    with pytest.raises(ValueError, match='explicit action_events'):
        recursive_fm_loss(dynamics, z_sequence=z, action=action)
    # Zero velocity initialization deliberately blocks condition gradients on the first update.
    # Verify the full FM training path once that initialized head has begun learning.
    torch.nn.init.normal_(dynamics.flow_head.weight, std=0.01)
    torch.nn.init.normal_(dynamics.graph_prior.output.weight, std=0.01)
    packed = pack_action_events([[events(), (ActionEvent('v', 3, 32., 2, 0, 2),)]])
    loss, _, _ = recursive_fm_loss(dynamics, z_sequence=z, action=action, action_events=packed, steps=1)
    loss.backward()
    for parameter in (dynamics.event_encoder.edge_embedding.weight,
                      dynamics.event_encoder.event_mlp[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_condition_is_encoded_once_per_solve_and_disagreement(monkeypatch):
    torch.set_num_threads(1)
    graph = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
    dynamics = Dynamics(adjacency=graph, static_features=torch.ones(3, 4), event_protocol=True)
    original = dynamics.event_encoder.forward
    calls = []
    def counted(events):
        calls.append(1)
        return original(events)
    monkeypatch.setattr(dynamics.event_encoder, 'forward', counted)
    packed = pack_action_events([[events()]])
    z, action = torch.randn(1, 3, 64), torch.zeros(1, 3)
    with torch.no_grad():
        dynamics.transition(z, action, action_events=packed, solver='picard', steps=2)
        assert len(calls) == 1
        dynamics.latent_disagreement(z, action, action_events=packed, n_samples=3, steps=2)
        assert len(calls) == 2


def test_replay_roundtrip_packs_multi_decisions_and_preserves_missing_distinction():
    flow = torch.zeros(3, 8)
    row = MacroTransition(flow, torch.tensor([0., 1., 2.]), torch.zeros(3), torch.zeros(2), flow,
                          0., terminal=True, action_events=events())
    replay = ReplayBuffer(1)
    replay.append_episode([row], total_travel_minutes=0., total_vehicle_count=2)
    assert replay.state_dict()['episodes'][0]['transitions'][0]['action_events'][0] == {
        'vehicle_id': 'v', 'event_index': 0, 'decision_time': 2.,
        'current_edge': 0, 'next_edge': 1, 'destination_edge': 2}
    restored = ReplayBuffer(1)
    restored.load_state_dict(replay.state_dict())
    batch = restored.sample_batch(1, np.random.default_rng(0))
    assert batch.action_events.vehicle_ids == ('v', 'v', 'w')
    assert restored.episodes[0][0].action_events == events()
    assert replace(row, action_events=None).action_events is None
    assert replace(row, action_events=()).action_events == ()


def test_illegal_turn_and_duplicate_episode_identity_are_rejected():
    model = encoder()
    with pytest.raises(ValueError, match='illegal turn'):
        model.aggregate(pack_action_events([[(ActionEvent('v', 0, 1., 0, 0, 2),)]]))
    flow = torch.zeros(3, 8)
    first = MacroTransition(flow, torch.zeros(3), torch.zeros(3), torch.zeros(2), flow, 0.,
                            action_events=(ActionEvent('v', 0, 1., 0, 1, 2),))
    second = replace(first, time=30., step_index=1,
                     action_events=(ActionEvent('v', 0, 31., 0, 1, 2),))
    with pytest.raises(ValueError, match='strictly increase'):
        ReplayBuffer(1).append_episode([first, second], total_travel_minutes=0., total_vehicle_count=1)
