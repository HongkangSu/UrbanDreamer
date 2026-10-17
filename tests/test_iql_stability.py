"""Stable independent Q-learning: legal Double DQN and train/eval alignment."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_routing_iql_baseline import features, method, row, windows
from urban_dreamer.learning.baselines.collection_policy import CollectionPolicy
from urban_dreamer.learning.baselines.iql import RoutingQActor, vehicle_decision_transitions
from urban_dreamer.models.policy import set_shortest_path_prior


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def stable(**kwargs):
    return method(iql_profile='double_dueling_v2', **kwargs)


def test_double_dqn_selects_online_and_evaluates_target_without_behavior_prior(monkeypatch):
    learner = stable()
    batch = vehicle_decision_transitions([row('v', 0.), row('v', 30.)], windows(), deadline=90.)
    monkeypatch.setattr(learner.actor, 'forward_q', lambda *args: torch.tensor([[8., 1., -torch.inf]]))
    monkeypatch.setattr(learner.target, 'forward_q', lambda *args: torch.tensor([[2., 9., -torch.inf]]))
    set_shortest_path_prior(learner.actor, 1e6, 1.)
    # The old target-max implementation would give -3+9=6.
    assert learner._batch_targets(batch).tolist() == pytest.approx([-1., -15.])


def test_dueling_separates_common_value_and_masks_padding_before_pooling():
    actor = RoutingQActor(dueling=True)
    data = torch.tensor(features())[None]
    mask = torch.tensor([[True, True, False]])
    q = actor.forward_q(data, mask)
    _, value, _ = actor._components(data, mask)
    torch.testing.assert_close(q[:, :2].mean(-1), value.squeeze(-1), atol=1e-6, rtol=1e-6)
    logits = actor.forward_logits(data, mask).detach().clone()
    with torch.no_grad():
        actor.value_head.bias.fill_(1e9)
    # Common values may be large, but action ranking must retain precision.
    torch.testing.assert_close(actor.forward_logits(data, mask), logits, rtol=0, atol=0)
    data[:, 2] = 1e6
    torch.testing.assert_close(actor.forward_logits(data, mask), logits, rtol=0, atol=0)
    permutation = [1, 0, 2]
    torch.testing.assert_close(actor.forward_logits(data[:, permutation], mask[:, permutation]), logits[:, permutation])


def test_branch_compression_preserves_team_tail_and_failed_vehicle_penalty():
    rows = [row('v', 0.), row('v', 20.), row('v', 40.), row('v', 60.)]
    for entry in (rows[1], rows[3]):
        entry['mask'] = np.array([True, False, False])
    rows[-1]['timeout_penalty'] = -50.
    transitions = vehicle_decision_transitions(rows, windows(), deadline=90., branch_only=True,
                                              include_individual_timeout=True)
    assert len(transitions) == 2
    assert transitions[0]['reward'] == pytest.approx(-5.)
    assert transitions[1]['reward'] == pytest.approx(-63.)
    assert sum(r['reward'] for r in transitions) == pytest.approx(-18. - 50.)
    assert transitions[-1]['terminal']
    assert transitions[0]['next_mask'].sum() == 2


def test_all_forced_events_do_not_consume_replay_update_budget():
    learner = stable()
    forced = row('v', 0.)
    forced['mask'] = np.array([True, False, False])
    metrics = learner.on_episode([forced], windows(), episode=1)
    assert metrics['q_updates'] == metrics['q_inserted_transitions'] == 0
    assert metrics['q_forced_events_skipped'] == 1
    forced['action'] = 1
    with pytest.raises(ValueError, match='legal'):
        learner.on_episode([forced], windows(), episode=2)


def test_epsilon_greedy_training_distribution_is_legal_and_independent_of_q_scale(monkeypatch):
    learner = stable(iql_epsilon_initial=.2, iql_epsilon_final=.2)
    obs = SimpleNamespace(features=np.zeros((10000, 3, 10), np.float32),
                          mask=np.tile([True, True, False], (10000, 1)))
    for scale in (.0001, 1e6):
        logits = torch.tensor([[scale, 0., -torch.inf]]).expand(10000, -1)
        monkeypatch.setattr(learner.actor, 'forward_logits', lambda *args: logits)
        torch.manual_seed(14)
        actions, aux = learner.select([], obs, None, 0.)
        assert np.mean(np.array(actions) == 1) == pytest.approx(.1, abs=.01)
        assert not np.any(np.array(actions) == 2)
        assert aux[0]['entropy'] == pytest.approx(-.9*np.log(.9)-.1*np.log(.1))
        greedy, _ = learner.select([], obs, None, 0., explore=False)
        assert set(greedy) == {0}


def test_episode_epsilon_schedule_reaches_floor_and_survives_resume_and_cpu_mirror():
    learner = stable(iql_epsilon_decay_episodes=10)
    # A CUDA learner's actor can be mirrored without allocating CUDA tensors.
    learner.device = torch.device('cuda:0')
    collector = CollectionPolicy(learner, 'cpu')
    learner.device = torch.device('cpu')
    assert collector.proxy is not learner
    learner.on_episode([], [], episode=5)
    collector.begin_episode(None, seed=1)
    assert float(collector.proxy.actor.exploration_epsilon) == pytest.approx(.525)
    assert learner.actor is not collector.proxy.actor
    assert any(p.requires_grad for p in learner.actor.parameters())
    restored = stable(iql_epsilon_decay_episodes=10)
    restored.load_state_dict(deepcopy(learner.state_dict()))
    assert restored.completed_episodes == 5
    assert float(restored.actor.exploration_epsilon) == pytest.approx(.525)
    for obj in (learner, restored):
        obj.on_episode([], [], episode=20)
        assert float(obj.actor.exploration_epsilon) == pytest.approx(.05)


def test_stable_profile_rejects_legacy_weights_and_changed_exploration_schedule():
    with pytest.raises(ValueError, match='profile mismatch'):
        stable().load_state_dict(method().state_dict())
    saved = stable().state_dict()
    with pytest.raises(ValueError, match='profile mismatch'):
        method().load_state_dict(saved)
    with pytest.raises(ValueError, match='setting mismatch'):
        stable(iql_epsilon_final=.1).load_state_dict(saved)


def test_stable_resume_reproduces_next_update_and_collection_draws():
    torch.manual_seed(99)
    learner = stable()
    rows = [row('v', 0.), row('v', 30.)]
    learner.on_episode(rows, windows(), episode=1)
    restored = stable()
    restored.load_state_dict(deepcopy(learner.state_dict()))
    learner.on_episode(rows, windows(), episode=2)
    restored.on_episode(rows, windows(), episode=2)
    for key, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, restored.actor.state_dict()[key], rtol=0, atol=0)
    obs = SimpleNamespace(features=np.stack([features()]*50), mask=np.tile([True,True,False], (50,1)))
    torch.manual_seed(80);left = learner.select([], obs, None, 0.)
    torch.manual_seed(80);right = restored.select([], obs, None, 0.)
    assert left == right


def test_terminal_two_action_problem_learns_greedy_ranking():
    torch.manual_seed(44)
    learner = stable(actor_lr=.003, iql_batch_size=16, iql_replay_capacity=128)
    # Same local state: action 0 yields zero cost, action 1 yields one minute.
    for episode in range(1, 161):
        action = episode % 2
        learner.on_episode([row('v', 0., action=action)],
            [dict(time=0., reward=-float(action), terminal=True)], episode=episode)
    q = learner.actor.forward_q(torch.tensor(features())[None], torch.tensor([[True,True,False]]))
    assert float(q[0, 0].detach()) == pytest.approx(0., abs=.15)
    assert float(q[0, 1].detach()) == pytest.approx(-1., abs=.15)
    assert q.argmax(-1).item() == 0
