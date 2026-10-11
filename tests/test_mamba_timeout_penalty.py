"""Individual timeout costs belong to MAMBA's actor, never its team models."""
from copy import deepcopy

import pytest
import torch

from test_event_imagination_protocol import seed
from test_routing_mamba_baseline import episode, fixed_flow, make_method
from urban_dreamer.learning.baselines import mamba
from urban_dreamer.workflow.config import NONARRIVAL_PENALTY


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def mixed_rollout(monkeypatch, *, deadline=90., global_only=False):
    method = make_method()
    method.args.global_reward_only = global_only
    method.runner.deadline = deadline
    rows, windows = episode(method)
    windows[0]['vehicles'] = [seed(v, current_edge=1, pending_decision=True,
        committed_next_edge=3) for v in ('a', 'b')]
    data = method._pack_episode(rows, windows, 1)
    fixed_flow(method, monkeypatch)
    monkeypatch.setattr(method.global_model.reward_model, 'forward',
        lambda x: x.new_full((*x.shape[:-1], 1), -1.))
    monkeypatch.setattr(method.global_model.continuation_model, 'forward',
        lambda x: x.new_full((*x.shape[:-1], 1), 100.))
    monkeypatch.setattr(method.critic, 'forward',
        lambda x: x.new_full((*x.shape[:-1], 1), -10.))

    def motion(vehicles, *, decide, frozen_flow, start_time, **kwargs):
        # Both identities act in the same chunk. Only a physically arrives;
        # b stays active, independently of either learned continuation head.
        pending = [v for v in vehicles if not v.get('arrived', False)]
        decide(pending, frozen_flow, start_time)
        for vehicle in vehicles:
            if vehicle['vehicle_id'] == 'a':
                vehicle['arrived'] = True

    monkeypatch.setattr(mamba, 'advance_imagined_window', motion)
    with torch.no_grad():
        _, states, _, _ = method._global_sequence(data, 0, train=False, cohort={'a', 'b'})
    records, features, returns = method._imagined_rollout(data, states[0], 0, {'a', 'b'})
    return method, data, records, features, returns


def test_real_deadline_penalizes_only_failed_identity_and_keeps_global_returns(monkeypatch):
    method, data, records, features, returns = mixed_rollout(monkeypatch)
    assert records[0]['vehicle_ids'] == ('a', 'b')
    torch.testing.assert_close(records[0]['timeout_returns'], torch.tensor([0., NONARRIVAL_PENALTY]))
    assert all(record['timeout_returns'].eq(NONARRIVAL_PENALTY).all() for record in records[1:])
    assert len(features) == 3
    torch.testing.assert_close(returns, torch.tensor([-3., -2., -1.]))
    torch.testing.assert_close(method._tensor(method._global_returns(data)),
        torch.tensor([-1.125, -.625, -.25]))


def test_horizon_bootstrap_does_not_fail_active_vehicle(monkeypatch):
    _, _, records, features, returns = mixed_rollout(monkeypatch, deadline=300.)
    assert len(features) == mamba.IMAGINATION_HORIZON == 8
    assert max(record['time'] for record in records) < 300.
    assert all(record['timeout_returns'].eq(0.).all() for record in records)
    # The global bootstrap also remains in place at a horizon truncation.
    torch.testing.assert_close(returns[-1], torch.tensor(-11.))


def test_ppo_uses_per_vehicle_penalty_without_rescaling_arrived_peer(monkeypatch):
    method, _, records, _, _ = mixed_rollout(monkeypatch)
    monkeypatch.setattr(method, '_step', lambda *args: None)
    global_advantages = torch.tensor([1., -2., 4.])
    no_failures = deepcopy(records)
    for record in no_failures:
        record['timeout_returns'].zero_()
    method._ppo_update(no_failures, global_advantages)
    method._ppo_update(records, global_advantages)
    assert records[0]['actor_advantage'].shape == (2,)
    assert records[0]['actor_advantage'][1] == -2.5
    assert records[0]['actor_advantage'][0] > 0.
    torch.testing.assert_close(records[0]['actor_advantage'][0],
        no_failures[0]['actor_advantage'][0])
    assert all(record['actor_advantage'].eq(-2.5).all() for record in records[1:])


def test_pure_mode_ignores_timeout_metadata_and_preserves_protocol(monkeypatch):
    method, _, records, _, _ = mixed_rollout(monkeypatch, global_only=True)
    assert all('timeout_returns' not in record for record in records)
    for record in records:
        record['timeout_returns'] = torch.full_like(record['action'], NONARRIVAL_PENALTY,
            dtype=torch.float32)
    monkeypatch.setattr(method, '_step', lambda *args: None)
    method._ppo_update(records, torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(records[0]['actor_advantage'], torch.ones(2))
    assert 'reward_mode' not in method.learning_protocol()


def test_checkpoint_rejects_reward_mode_change_in_both_directions():
    pure = make_method()
    penalty = make_method()
    penalty.args.global_reward_only = False
    pure_state = pure.state_dict()
    penalty_state = penalty.state_dict()
    assert penalty_state['protocol']['reward_mode'] == 'global_with_individual_timeout'
    with pytest.raises(ValueError, match='fresh run'):
        penalty.load_state_dict(pure_state)
    with pytest.raises(ValueError, match='fresh run'):
        pure.load_state_dict(penalty_state)
    restored = make_method()
    restored.args.global_reward_only = False
    restored.load_state_dict(penalty_state)
    assert restored.learning_protocol() == penalty.learning_protocol()


def test_actual_physical_stepper_assigns_timeout_at_partial_deadline(monkeypatch):
    method = make_method()
    method.args.global_reward_only = False
    method.runner.deadline = 5.
    rows, windows = episode(method)
    windows = [dict(windows[0], end_time=5., terminal=True,
        vehicles=[seed('a', current_edge=1, pending_decision=True, committed_next_edge=3)])]
    method._events = method._events[:2]
    data = method._pack_episode(rows[:2], windows, 1)
    fixed_flow(method, monkeypatch)
    with torch.no_grad():
        _, states, _, _ = method._global_sequence(data, 0, train=False, cohort={'a'})
    records, features, _ = method._imagined_rollout(data, states[0], 0, {'a'})
    assert len(features) == 1 and records
    # The destination lane alone takes 100 seconds at full speed. The real
    # five-second physical deadline is necessarily a nonarrival here.
    assert all(record['timeout_returns'].eq(NONARRIVAL_PENALTY).all() for record in records)


def test_penalty_mode_full_learning_update_keeps_team_labels_and_finite_gradients(monkeypatch):
    method = make_method()
    method.args.global_reward_only = False
    rows, windows = episode(method)
    for row in rows:
        row['timeout_return'] = NONARRIVAL_PENALTY
    fixed_flow(method, monkeypatch)
    before = [parameter.detach().clone() for parameter in method.actor.parameters()]
    metrics = method.on_episode(rows, windows, episode=1)
    assert metrics['model_warmup_complete'] and metrics['imagined_events'] > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, method.actor.parameters()))
    torch.testing.assert_close(method._tensor(method._global_returns(method.replay[0])),
        torch.tensor([-1.125, -.625, -.25]))
