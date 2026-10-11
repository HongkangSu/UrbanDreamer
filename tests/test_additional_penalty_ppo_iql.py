"""Individual deadline costs retain team critics and cannot cross replay modes."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines.iql import IndependentQMethod, vehicle_decision_transitions
from urban_dreamer.learning.baselines.ppo import IppoMethod, MappoMethod
from urban_dreamer.models.policy import NONARRIVAL_PENALTY, attach_timeout_penalties


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def method(cls, *, pure=False):
    torch.manual_seed(89)
    args = SimpleNamespace(seed=42, actor_lr=3e-4, critic_lr=3e-4,
        global_reward_only=pure, iql_batch_size=8, iql_replay_capacity=32)
    runner = SimpleNamespace(device=torch.device('cpu'), deadline=60.,
        network=SimpleNamespace(edge_count=2))
    return cls(args, runner)


def episode(model=None):
    rng = np.random.default_rng(39)
    features = rng.random((4, 3, 10), dtype=np.float32)
    mask = np.array([[True, True, False]] * 4)
    actions, auxiliary = ([0] * 4, [{}] * 4)
    if model is not None:
        actions, auxiliary = model.select([{}] * 4,
            SimpleNamespace(features=features, mask=mask), None, 0.)
    rows = [dict(vehicle_id=vehicle, time=time, window=index // 2,
        features=features[index], mask=mask[index], action=actions[index], **auxiliary[index])
        for index, (vehicle, time) in enumerate([
            ('failed', 0.), ('arrived', 10.), ('failed', 30.), ('arrived', 40.)])]
    attach_timeout_penalties(rows, {'failed'})
    windows = [dict(start_time=index * 30., end_time=(index + 1) * 30.,
        obs=rng.random(10, dtype=np.float32), next_obs=rng.random(10, dtype=np.float32),
        reward=-30., terminal=index == 1) for index in range(2)]
    return rows, windows


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod])
def test_ppo_penalizes_only_failed_vehicle_without_changing_team_targets(cls, monkeypatch):
    learner = method(cls)
    rows, windows = episode(learner)
    clean = deepcopy(rows)
    attach_timeout_penalties(clean, set())
    targets, global_advantages, _ = learner._targets(rows, windows)
    clean_targets, clean_global_advantages, _ = learner._targets(clean, windows)
    torch.testing.assert_close(targets, clean_targets, rtol=0, atol=0)
    torch.testing.assert_close(global_advantages, clean_global_advantages, rtol=0, atol=0)
    # Use a positive shared advantage to expose penalty centering or double scaling.
    advantages = torch.tensor([1., 2., 3., 4.])
    penalized = learner._actor_advantages(advantages, rows)
    unpenalized = learner._actor_advantages(advantages, clean)
    torch.testing.assert_close(penalized[[1, 3]], unpenalized[[1, 3]], rtol=0, atol=0)
    torch.testing.assert_close(penalized[[0, 2]], torch.tensor([-2.5, -2.5]))
    assert bool((unpenalized > 0.).all())
    # Confirm the optimizer path consumes that helper, including every failed action.
    captured = []
    original = learner._actor_advantages

    def capture(global_values, collected_rows):
        result = original(global_values, collected_rows)
        captured.append(result.clone())
        return result

    monkeypatch.setattr(learner, '_actor_advantages', capture)
    learner.on_episode(rows, windows, episode=1)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0][[0, 2]], torch.tensor([-2.5, -2.5]))


def test_iql_adds_failed_cost_once_only_at_own_terminal_transition():
    rows, windows = episode()
    pure = vehicle_decision_transitions(rows, windows, deadline=60.)
    penalty = vehicle_decision_transitions(rows, windows, deadline=60.,
        include_individual_timeout=True)
    assert [row['reward'] for row in pure] == pytest.approx([-30., -30., -30., -20.])
    deltas = [after['reward'] - before['reward'] for after, before in zip(penalty, pure)]
    assert deltas == [0., NONARRIVAL_PENALTY, 0., 0.]
    assert [row['terminal'] for row in penalty] == [False, True, False, True]
    learner = method(IndependentQMethod)
    learner.on_episode(rows, windows, episode=1)
    assert [row['reward'] for row in learner.replay] == [row['reward'] for row in penalty]


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod, IndependentQMethod])
@pytest.mark.parametrize('source_pure', [False, True])
def test_reward_mode_mismatch_rejected_before_loading_weights(cls, source_pure):
    source = method(cls, pure=source_pure)
    destination = method(cls, pure=not source_pure)
    before = deepcopy(destination.actor.state_dict())
    with pytest.raises(ValueError, match='protocol|reward mode'):
        destination.load_state_dict(deepcopy(source.state_dict()))
    for key, value in destination.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_legacy_iql_replay_without_reward_mode_is_only_accepted_by_pure_mode():
    legacy = method(IndependentQMethod, pure=True).state_dict()
    del legacy['reward_mode']
    method(IndependentQMethod, pure=True).load_state_dict(deepcopy(legacy))
    with pytest.raises(ValueError, match='reward mode'):
        method(IndependentQMethod).load_state_dict(deepcopy(legacy))


@pytest.mark.parametrize('cls', [MappoMethod, IppoMethod, IndependentQMethod])
def test_penalty_mode_checkpoint_continuation_is_exact(cls):
    source = method(cls)
    rows, windows = episode(source)
    source.on_episode(rows, windows, episode=1)
    restored = method(cls)
    restored.load_state_dict(deepcopy(source.state_dict()))
    rows, windows = episode(source)
    torch.manual_seed(682)
    source.on_episode(rows, windows, episode=2)
    torch.manual_seed(682)
    restored.on_episode(rows, windows, episode=2)
    for original, loaded in zip(source.actor.parameters(), restored.actor.parameters()):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
    if cls is IndependentQMethod:
        assert [row['reward'] for row in source.replay] == [row['reward'] for row in restored.replay]
