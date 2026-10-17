"""Static audit regressions for model replay and physical imagination boundaries."""
import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_route_model_based_baselines import collect_toy_episode, make_method
from urban_dreamer.learning.global_reward import causal_context
from urban_dreamer.learning.replay import MacroTransition


def seed_rollout(method, name, window):
    if name == 'mbpo':
        return method._rollout_from_seed(window, 0)
    method.args.dreamer_imagined_horizon = 1
    method.seed_buffer.append_episode([MacroTransition(
        flow=window['flow'], next_flow=window['next_flow'], action=window['action'],
        departure=window['departure'], context=window['context'], vehicles=window['vehicles'],
        active_vehicle_fraction=1., global_reward=window['reward'], time=window['time'],
        terminal=False, episode=0, step_index=0)],
        total_travel_minutes=1., total_vehicle_count=window['total_vehicle_count'])
    return method._imagine_rollout(0)


@pytest.mark.parametrize('name', ['mbpo', 'dreamer'])
def test_imagination_respects_partial_final_window(monkeypatch, name):
    method = make_method(name)
    _, windows = collect_toy_episode(method)
    window = dict(windows[0], time=55.)
    implementation = importlib.import_module(f'urban_dreamer.learning.baselines.{name}')
    durations = []

    def advance(states, **kwargs):
        durations.append(kwargs.get('interval_seconds', 30.))
        return SimpleNamespace(action=np.zeros(method.edge_count), departure=np.zeros(method.edge_count),
            active_vehicle_seconds=kwargs.get('interval_seconds', 30.))

    monkeypatch.setattr(implementation, 'advance_imagined_window', advance)
    seed_rollout(method, name, window)
    assert durations == [5.], 'imagined arrivals/rewards must not extend beyond the real deadline'


@pytest.mark.parametrize('name', ['mbpo', 'dreamer'])
def test_imagined_actor_observes_event_time_and_live_active_count(monkeypatch, name):
    method = make_method(name)
    _, windows = collect_toy_episode(method)
    window = copy.deepcopy(windows[0])
    window['vehicles'].append(dict(window['vehicles'][0], vehicle_id='second'))
    window.update(total_vehicle_count=2, started_vehicle_count=2,
        context=causal_context(0., method.deadline, 2, 2, 0.))
    implementation = importlib.import_module(f'urban_dreamer.learning.baselines.{name}')
    contexts = []
    original = implementation.build_actor_observations

    def observations(**kwargs):
        contexts.append((kwargs['episode_progress'], kwargs['active_vehicle_fraction']))
        return original(**kwargs)

    def advance(states, **kwargs):
        states[1]['arrived'] = True
        kwargs['decide']([states[0]], kwargs['frozen_flow'], kwargs['start_time'] + 5.)
        states[0]['arrived'] = True
        return SimpleNamespace(action=np.zeros(method.edge_count), departure=np.zeros(method.edge_count),
            active_vehicle_seconds=5.)

    monkeypatch.setattr(implementation, 'build_actor_observations', observations)
    monkeypatch.setattr(implementation, 'advance_imagined_window', advance)
    seed_rollout(method, name, window)
    assert contexts == [pytest.approx((5. / method.deadline, .5))]


def test_mbpo_checkpoint_preserves_replay_and_exact_next_model_update():
    original = make_method('mbpo')
    _, windows = collect_toy_episode(original)
    original.window_buffer.append([original._model_sample(window) for window in windows])
    original._train_model()
    state = copy.deepcopy(original.state_dict())
    restored = make_method('mbpo')
    restored.load_state_dict(state)
    assert len(restored.window_buffer) == len(original.window_buffer) == 1
    assert restored.window_buffer.maxlen == original.window_buffer.maxlen
    expected = original._train_model()
    actual = restored._train_model()
    assert actual == pytest.approx(expected)
    for before, after in zip(original.ensemble.parameters(), restored.ensemble.parameters()):
        torch.testing.assert_close(before, after, rtol=0., atol=0.)
    restored.window_buffer[0][0]['x'][0] += 10.
    assert restored.window_buffer[0][0]['x'][0] != state['window_replay'][0][0]['x'][0]


@pytest.mark.parametrize('name', ['mbpo', 'dreamer'])
def test_round_one_learning_protocol_cannot_silently_resume(name):
    method = make_method(name)
    state = copy.deepcopy(method.state_dict())
    state['learning_protocol']['version'] = 2
    with pytest.raises(ValueError, match='learning protocol changed'):
        method.load_state_dict(state)


@pytest.mark.parametrize('name', ['mbpo', 'dreamer'])
def test_model_based_policy_ignores_large_common_head_bias(name):
    method = make_method(name)
    features = torch.rand(3, 4, 10)
    masks = torch.ones(3, 4, dtype=torch.bool)
    before = method.actor.forward_logits(features, masks)
    with torch.no_grad():
        method.actor.policy_head.bias.fill_(2. ** 24)
    after = method.actor.forward_logits(features, masks)
    torch.testing.assert_close(after, before, atol=0., rtol=0.)
    assert method.learning_protocol()['score_gauge'] == 'pre_head_legal_reference_v1'


def test_dreamer_imagination_mirror_preserves_stable_score_gauge():
    from urban_dreamer.learning.baselines.collection_policy import ActorMirror
    method = make_method('dreamer')
    # This creates the separate CPU inference copy without requiring a CUDA GPU.
    method.imagination_mirror = ActorMirror('cuda:0', 'cpu')
    mirrored = method._imagination_actor()
    assert mirrored is not method.actor
    assert mirrored.score_gauge == method.actor.score_gauge == 'pre_head_legal_reference_v1'
    with torch.no_grad():
        method.actor.policy_head.weight.mul_(2.)
    method._imagination_actor()
    features = torch.rand(2, 4, 10)
    masks = torch.ones(2, 4, dtype=torch.bool)
    torch.testing.assert_close(mirrored.forward_logits(features, masks),
        method.actor.forward_logits(features, masks), atol=0., rtol=0.)


@pytest.mark.parametrize('name', ['mbpo', 'dreamer'])
@pytest.mark.parametrize('global_only', [False, True])
def test_model_based_actor_uses_bounded_batches_and_explicit_normalization(monkeypatch, name, global_only):
    method = make_method(name)
    method.args.global_reward_only = global_only
    rows, windows = collect_toy_episode(method)
    implementation = importlib.import_module(f'urban_dreamer.learning.baselines.{name}')
    calls = []
    monkeypatch.setattr(implementation, 'policy_gradient_step',
        lambda **kwargs: calls.append(kwargs) or dict(actor_loss=0.))
    if name == 'mbpo':
        method._policy_updates(rows, torch.tensor([-10., -12.]))
        assert len(calls) == 1
    else:
        method._train_real_policy(rows, windows)
        h, z = method._initial_latent(windows[0]['flow'])
        latent = method._critic_latent(h, z, windows[0]['context'])
        trajectory = ([latent], [latent], [-1.], [True])
        monkeypatch.setattr(method, '_imagine_rollout',
            lambda rollout, actor: ([dict(rows[0], window=0)], trajectory))
        method._imagination_pass()
        assert len(calls) == 2
    assert all(call['minibatch_size'] == 512 for call in calls)
    assert all(call['normalize_global_advantages'] is global_only for call in calls)
    protocol = method.learning_protocol()
    assert protocol['actor_minibatch_size'] == 512
    assert protocol['global_actor_advantage'] == (
        'centered_std_v1' if global_only else 'legacy_uncentered_v1')
