"""IQL learns raw Q values from asynchronous, whole-team reward transitions."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines.iql import (
    IndependentQMethod, RoutingQActor, vehicle_decision_transitions)
from urban_dreamer.models.policy import set_shortest_path_prior


def features():
    result = np.zeros((3, 10), dtype=np.float32)
    result[:, 4] = np.log1p([30., 90., 1.]) / math.log1p(3600.)
    return result


def row(vehicle, timestamp, *, action=0, timeout_return=0.):
    return dict(vehicle_id=vehicle, time=float(timestamp), features=features(),
                mask=np.array([True, True, False]), action=action,
                timeout_return=timeout_return)


def windows():
    return [dict(time=0., reward=-3., terminal=False),
            dict(time=30., reward=-6., terminal=False),
            dict(time=60., reward=-9., terminal=True)]


def method(**overrides):
    settings = dict(seed=42, actor_lr=2e-4, iql_batch_size=8,
                    iql_replay_capacity=32, iql_sample_to_insert_ratio=1.)
    settings.update(overrides)
    args = SimpleNamespace(**settings)
    return IndependentQMethod(args, SimpleNamespace(device=torch.device('cpu'), deadline=90.))


def test_raw_q_unchanged_by_behavior_prior_and_illegal_mask():
    torch.set_num_threads(1)
    actor = RoutingQActor()
    data = torch.tensor(features()).unsqueeze(0)
    mask = torch.tensor([[True, True, False]])
    before = actor.forward_q(data, mask).detach().clone()
    set_shortest_path_prior(actor, 2., 30.)
    after = actor.forward_q(data, mask)
    assert torch.equal(before, after)
    behavior = actor.forward_logits(data, mask)
    assert float(behavior[0, 1].detach()) == pytest.approx(float(before[0, 1]) - 4.)
    assert torch.isneginf(after[0, 2]) and torch.isneginf(behavior[0, 2])
    sampled = actor.sample_action(data.expand(100, -1, -1), mask.expand(100, -1))
    assert bool((sampled.candidate_index < 2).all())
    with pytest.raises(ValueError, match='at least one legal'):
        actor.forward_q(data, torch.zeros_like(mask))


def test_dynamic_vehicle_linking_and_last_action_keeps_whole_team_tail():
    # A arrives just after its final decision at 45 s; the team's last window
    # still contributes -9, despite A having no subsequent decisions.
    rows = [row('ep1_A', 15.), row('ep1_B', 20.), row('ep1_A', 45.), row('ep1_B', 75.)]
    transitions = vehicle_decision_transitions(rows, windows(), deadline=90.)
    assert [entry['reward'] for entry in transitions] == pytest.approx([-4.5, -12., -11.5, -4.5])
    assert [entry['terminal'] for entry in transitions] == [False, True, False, True]
    assert transitions[0]['next_features'] is not None
    assert transitions[1]['next_features'] is None
    # A's linked rewards sum to the TEAM return from t=15 onward.
    assert sum(entry['reward'] for entry in transitions[:2]) == pytest.approx(-16.5)


def test_recorded_partial_terminal_window_is_respected():
    recorded = [dict(obs=np.array([0., 0.]), next_obs=np.array([1 / 3, 0.]), reward=-3.),
                dict(obs=np.array([1 / 3, 0.]), next_obs=np.array([.5, 0.]), reward=-6.)]
    transitions = vehicle_decision_transitions([row('new_demand', 35.)], recorded, deadline=90.)
    assert transitions[0]['reward'] == pytest.approx(-4.)
    assert transitions[0]['terminal']


def test_exact_physical_bounds_override_misleading_observations():
    recorded = [dict(start_time=0., end_time=20., obs=np.array([.3, 0.]),
                     next_obs=np.array([.8, 0.]), reward=-4.),
                dict(start_time=20., end_time=35., obs=np.array([.8, 0.]),
                     next_obs=np.array([1., 0.]), reward=-3.)]
    transitions = vehicle_decision_transitions(
        [row('v1', 10.), row('v1', 25.)], recorded, deadline=90.)
    assert [entry['reward'] for entry in transitions] == pytest.approx([-3., -2.])
    assert transitions[-1]['terminal']


def test_timeout_annotations_cannot_change_transitions_or_updates():
    clean = [row('v1', 10.), row('v1', 40.), row('v2', 50.)]
    poisoned = copy.deepcopy(clean)
    for entry in poisoned:
        entry.update(timeout_return=-1e12, timeout_penalty=-1e12, local_cost=-1e12)
    left = vehicle_decision_transitions(clean, windows(), deadline=90.)
    right = vehicle_decision_transitions(poisoned, windows(), deadline=90.)
    assert [entry['reward'] for entry in left] == [entry['reward'] for entry in right]
    original = method()
    restored = method()
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    original.on_episode(clean, windows(), episode=1)
    restored.on_episode(poisoned, windows(), episode=1)
    for left_param, right_param in zip(original.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(left_param, right_param)


def test_bellman_targets_use_unbiased_legal_raw_q():
    learner = method()
    batch = vehicle_decision_transitions([row('v1', 0.), row('v1', 30.)], windows(), deadline=90.)
    with torch.no_grad():
        for parameter in learner.target.parameters():
            parameter.zero_()
        learner.target.policy_head.bias.fill_(2.)
    set_shortest_path_prior(learner.target, 1e6, 1.)
    # Nonterminal gets raw max(Q)=2; terminal gets the team's -15 tail.
    assert learner._batch_targets(batch).tolist() == pytest.approx([-1., -15.])


def test_update_bounded_replay_and_exact_checkpoint_continuation():
    learner = method()
    rows = [row('v1', 0.), row('v1', 30.), row('v2', 45.)]
    before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
    result = learner.on_episode(rows, windows(), episode=1)
    assert result['q_updates'] == 1 and result['q_loss'] > 0
    assert any(not torch.equal(old, new) for old, new in zip(before, learner.actor.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(before, learner.target.parameters()))
    set_shortest_path_prior(learner.actor, .6, 30.)
    checkpoint = copy.deepcopy(learner.state_dict())
    restored = method()
    restored.load_state_dict(checkpoint)
    assert restored.actor.prior_weight == .6
    assert len(restored.replay) == 3 and restored.updates == 1
    assert restored.optimizer.state
    learner.on_episode(rows, windows(), episode=2)
    restored.on_episode(rows, windows(), episode=2)
    for left, right in zip(learner.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(learner.target.parameters(), restored.target.parameters()):
        assert torch.equal(left, right)
    # New episode IDs do not link across different calls, and replay stays bounded.
    for episode in range(3, 16):
        learner.on_episode(rows, windows(), episode=episode)
    assert len(learner.replay) == 32


def test_empty_episode_and_bad_window_or_action():
    learner = method()
    assert learner.on_episode([], [], episode=1)['q_updates'] == 0
    with pytest.raises(ValueError, match='legal'):
        vehicle_decision_transitions([row('v1', 0., action=2)], windows(), deadline=90.)
    with pytest.raises(ValueError, match='contiguous'):
        vehicle_decision_transitions([row('v1', 0.)],
            [dict(time=0., reward=-1.), dict(time=40., reward=-1.)], deadline=90.)


def test_transition_budget_scales_and_empty_episode_does_not_retrain_replay():
    learner = method(iql_batch_size=2)
    rows = [row(f'v{i}', 10.) for i in range(7)]
    result = learner.on_episode(rows, windows(), episode=1)
    assert result['q_sampled_transitions'] == 7
    assert result['q_inserted_transitions'] == 7
    assert result['q_updates'] == 4
    assert result['q_terminal_samples'] == 7
    assert result['q_nonterminal_samples'] == 0
    before = copy.deepcopy(learner.state_dict())
    result = learner.on_episode([], [], episode=2)
    assert result['q_sampled_transitions'] == 0
    assert result['q_updates'] == 4
    for name, value in before['actor'].items():
        assert torch.equal(value, learner.state_dict()['actor'][name])


def test_fractional_draw_budget_exact_across_episodes_and_checkpoint():
    learner = method(iql_sample_to_insert_ratio=.1)
    rows = [row('v', 10.)]
    for episode in range(1, 6):
        result = learner.on_episode(rows, windows(), episode=episode)
        assert result['q_sampled_transitions'] == 0
    restored = method(iql_sample_to_insert_ratio=.1)
    restored.load_state_dict(copy.deepcopy(learner.state_dict()))
    for episode in range(6, 11):
        left = learner.on_episode(rows, windows(), episode=episode)
        right = restored.on_episode(rows, windows(), episode=episode)
        assert left['q_sampled_transitions'] == right['q_sampled_transitions'] == (episode == 10)
    assert restored.total_draws == 1 and restored.total_inserted == 10
    for left, right in zip(learner.actor.parameters(), restored.actor.parameters()):
        assert torch.equal(left, right)


def test_budget_counts_all_new_rows_when_replay_is_smaller():
    learner = method(iql_replay_capacity=2, iql_batch_size=8, iql_sample_to_insert_ratio=1.5)
    rows = [row(f'v{i}', 10.) for i in range(7)]
    result = learner.on_episode(rows, windows(), episode=1)
    assert len(learner.replay) == 2
    assert result['q_sampled_transitions'] == 10
    assert result['q_new_transitions_evicted'] == 5
    assert result['q_updates'] == 5


def test_partial_batch_target_update_uses_sample_units():
    learner = method(iql_batch_size=8, iql_target_tau=.2)
    original = [parameter.detach().clone() for parameter in learner.target.parameters()]
    learner.on_episode([row('v', 10.)], windows(), episode=1)
    coefficient = 1 - .8 ** (1 / 8)
    for old, target, online in zip(original, learner.target.parameters(), learner.actor.parameters()):
        assert torch.allclose(target, torch.lerp(old, online.detach(), coefficient), atol=1e-7)


def test_checkpoint_rejects_legacy_and_changed_settings_before_loading():
    learner = method()
    saved = learner.state_dict()
    legacy = copy.deepcopy(saved)
    legacy.pop('schedule_version')
    with pytest.raises(ValueError, match='schedule'):
        learner.load_state_dict(legacy)
    for settings in ({'iql_sample_to_insert_ratio':2.}, {'actor_lr':1e-3},
                     {'iql_batch_size':4}, {'iql_target_tau':.2}, {'iql_replay_capacity':16}):
        with pytest.raises(ValueError, match='setting'):
            method(**settings).load_state_dict(saved)
    with pytest.raises(ValueError, match='iql_updates'):
        method(iql_updates=4)


def test_terminal_and_nonterminal_td_metrics_preserve_global_team_tail():
    learner = method()
    with torch.no_grad():
        for network in (learner.actor, learner.target):
            for parameter in network.parameters():
                parameter.zero_()
    result = learner.on_episode([row('v', 0.), row('v', 30.)], windows(), episode=1)
    assert result['q_terminal_samples'] == result['q_nonterminal_samples'] == 1
    assert result['q_terminal_td_mean'] == pytest.approx(-15.)
    assert result['q_terminal_td_abs_mean'] == pytest.approx(15.)
    assert result['q_nonterminal_td_mean'] == pytest.approx(-3.)
    assert result['q_nonterminal_td_abs_mean'] == pytest.approx(3.)
    assert result['q_loss'] == pytest.approx(8.5)


@pytest.mark.parametrize('ratio', [0., -1., float('nan'), float('inf')])
def test_reject_invalid_sample_ratio(ratio):
    with pytest.raises(ValueError, match='invalid IQL'):
        method(iql_sample_to_insert_ratio=ratio)


def test_reject_checkpoint_with_inconsistent_fractional_budget():
    learner = method(iql_sample_to_insert_ratio=.1)
    learner.on_episode([row('v', 10.)], windows(), episode=1)
    saved = copy.deepcopy(learner.state_dict())
    saved['total_draws'] = 1
    with pytest.raises(ValueError, match='draw budget'):
        learner.load_state_dict(saved)
