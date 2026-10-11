"""MAMBA uses physical events, fixed-cohort macro actions and fresh posteriors."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_event_imagination_protocol import network, seed
from urban_dreamer.learning.baselines.mamba import MambaMethod, SCHEMA, clipped_policy_loss
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.models.policy import set_shortest_path_prior
from urban_dreamer.traffic.topology import NextHopCandidates


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_method(**options):
    torch.manual_seed(17)
    args = SimpleNamespace(reward_objective='legacy_exposure', seed=42, actor_lr=3e-4, critic_lr=3e-4, entropy_coef=.01,
        step_seconds=1., mamba_model_warmup_updates=options.get('warmup', 0),
        mamba_min_model_frames=options.get('frames', 0), mamba_burn_in_windows=2)
    topology = network()
    runner = SimpleNamespace(device=torch.device('cpu'), candidates=NextHopCandidates(topology, traffic_model='meso'),
        network=topology, deadline=90., config=SimpleNamespace(simulation=SimpleNamespace(
            traffic_model='meso', vehicle_max_speed_mps=10.)))
    return MambaMethod(args, runner)


def episode(method):
    method.begin_episode(SimpleNamespace(vehicles=[SimpleNamespace(vehicle_id=v) for v in ('a', 'b')]), seed=42)
    flow = np.zeros((4, 8), dtype=np.float32)
    flow[:, 1] = 1.
    rows = []
    for now, vehicle_id in ((1., 'a'), (2., 'a'), (31., 'b')):
        vehicle = seed(vehicle_id, current_edge=1, pending_decision=True, committed_next_edge=3)
        obs = build_actor_observations(policy_edge_features=flow[:, [2, 1]], vehicle_states=[vehicle],
            candidates=method.runner.candidates, episode_progress=now / 90., active_vehicle_fraction=.5)
        indices, aux = method.select([vehicle], obs, flow, now)
        rows.append(dict(vehicle_id=vehicle_id, window=int(now // 30), current_edge=1,
            edge_indices=obs.edge_indices[0], action=indices[0], **aux[0]))
    windows = [dict(time=float(i * 30), end_time=float((i + 1) * 30), flow=flow.copy(),
        vehicles=[seed('a')] if i == 0 else [seed('b')] if i == 1 else [],
        total_vehicle_count=2, active_vehicle_seconds=float(60 - i * 15),
        reward=-(60 - i * 15) / 120., terminal=i == 2) for i in range(3)]
    return rows, windows


def fixed_flow(method, monkeypatch):
    def decode(features):
        out = torch.zeros((*features.shape[:-1], method.global_dim), device=features.device)
        out[..., :32].reshape(*features.shape[:-1], 4, 8)[..., 1] = 1.
        return out
    monkeypatch.setattr(method.global_model.observation_decoder, 'forward', decode)


def test_fixed_cohort_actions_and_spatial_identity_include_repeated_events():
    method = make_method()
    rows, windows = episode(method)
    data = method._pack_episode(rows, windows, 1)
    seen = []
    handle = method.global_model.transition.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone()))
    method._global_sequence(data, 0, train=False, cohort=frozenset({'a'}))
    handle.remove()
    assert seen[1][..., :4].sum() == 1.
    assert seen[1][..., 4:8].sum() == 1.
    assert seen[3][..., :8].sum() == 0.  # b is statistical background
    assert seen[1][..., -1].item() == 1.
    a = method._global_observation(windows[0], 2, {'a'})
    changed = deepcopy(windows[0])
    changed['flow'][0, 2] = 1.
    b = method._global_observation(changed, 2, {'a'})
    assert not np.array_equal(a, b)
    assert a.shape == (42,)
    assert 'seeds' not in data and not hasattr(method, '_seeds')


def test_burnin_keeps_same_cohort_and_reconstructs_with_current_model(monkeypatch):
    method = make_method()
    rows, windows = episode(method)
    windows[1]['vehicles'] = [seed('a')]
    data = method._pack_episode(rows, windows, 1)
    seen = []
    original = method._global_observation
    monkeypatch.setattr(method, '_global_observation',
        lambda window, population, cohort: (seen.append((window['time'], cohort)) or original(window, population, cohort)))
    method._global_sequence(data, 1, train=False, cohort=frozenset({'a'}))
    assert seen[0] == (0., frozenset({'a'}))
    assert all(cohort == frozenset({'a'}) for _, cohort in seen)
    _, before, _ = method._reconstruct_local(data, 1, {'a'})
    with torch.no_grad():
        for parameter in method.world_model.parameters():
            parameter.add_(.02)
    _, after, _ = method._reconstruct_local(data, 1, {'a'})
    assert not torch.equal(before.deter, after.deter)
    assert 'latent' not in data['events'][0]


def test_physical_rollout_uses_real_event_times_legal_actions_and_tail(monkeypatch):
    method = make_method()
    rows, windows = episode(method)
    method.runner.network.lengths_m[3] = 10.
    data = method._pack_episode(rows, windows, 1)
    fixed_flow(method, monkeypatch)
    monkeypatch.setattr(method.global_model.reward_model, 'forward', lambda x: x.new_full((*x.shape[:-1], 1), -1.))
    monkeypatch.setattr(method.global_model.continuation_model, 'forward', lambda x: x.new_full((*x.shape[:-1], 1), 100.))
    monkeypatch.setattr(method.world_model.continuation_model, 'forward', lambda x: (_ for _ in ()).throw(AssertionError('unused local arrival head')))
    with torch.no_grad():
        _, states, _, _ = method._global_sequence(data, 0, train=False, cohort={'a'})
    records, features, returns = method._imagined_rollout(data, states[0], 0, {'a'})
    assert records and records[0]['time'] == 1.
    assert all(r['vehicle_ids'] == ('a',) for r in records)
    assert all(r['mask'].gather(1, r['action'][:, None]).all() for r in records)
    assert all(r['step'] == 0 for r in records)
    assert len(features) == 3
    torch.testing.assert_close(returns, torch.tensor([-3., -2., -1.]))


def test_warmup_blocks_actor_but_updates_models_and_real_critic():
    method = make_method(warmup=100, frames=100)
    rows, windows = episode(method)
    before = [p.detach().clone() for p in method.actor.parameters()]
    metrics = method.on_episode(rows, windows, episode=1)
    assert metrics['model_updates'] == 4 and metrics['model_frames'] == 3
    assert not metrics['model_warmup_complete'] and metrics['imagined_events'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before, method.actor.parameters()))
    assert method.model_optimizer.state and method.critic_optimizer.state


def test_multiple_physical_events_and_partial_window_action_reward_alignment(monkeypatch):
    method = make_method()
    rows, windows = episode(method)
    method.runner.deadline = 5.
    windows = [dict(windows[0], end_time=5., vehicles=[seed('a', current_edge=1,
        pending_decision=True, committed_next_edge=3)], terminal=True)]
    # Pack event metadata consistently for the shortened real reference.
    method._events = method._events[:2]
    data = method._pack_episode(rows[:2], windows, 1)
    fixed_flow(method, monkeypatch)
    def detour(latent, candidates, mask, **kwargs):
        logits = torch.zeros_like(mask, dtype=torch.float32)
        logits[:, 0] = 100.
        return torch.distributions.Categorical(logits=logits.masked_fill(~mask, -torch.inf))
    monkeypatch.setattr(method.actor, 'routing_distribution', detour)
    monkeypatch.setattr(method.global_model.reward_model, 'forward', lambda x: x.new_full((*x.shape[:-1], 1), -.2))
    monkeypatch.setattr(method.critic, 'forward', lambda x: x.new_full((*x.shape[:-1], 1), -1000.))
    with torch.no_grad():
        _, states, _, _ = method._global_sequence(data, 0, train=False, cohort={'a'})
    actions = []
    hook = method.global_model.transition.register_forward_pre_hook(
        lambda _, args: actions.append(args[0].detach().clone()))
    records, _, returns = method._imagined_rollout(data, states[0], 0, {'a'})
    hook.remove()
    assert [r['time'] for r in records] == [0., 2., 4.]
    assert actions[-1][..., :4].sum() == 1.5
    assert actions[-1][..., 4:8].sum() == 1.5
    assert actions[-1][..., -1].item() == pytest.approx(5. / 30.)
    torch.testing.assert_close(returns, torch.tensor([-.2]))  # no second duration scaling / deadline bootstrap


def test_actor_update_after_warmup_real_rewards_ignore_local_poison(monkeypatch):
    method = make_method()
    rows, windows = episode(method)
    for row in rows:
        row['timeout_return'] = -9999.
    fixed_flow(method, monkeypatch)
    before = [p.detach().clone() for p in method.actor.parameters()]
    metrics = method.on_episode(rows, windows, episode=1)
    assert metrics['model_warmup_complete'] and metrics['imagined_events'] > 0
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, method.actor.parameters()))
    np.testing.assert_allclose(method._global_returns(method.replay[0]), [-1.125, -.625, -.25])


def test_checkpoint_versions_protocol_counters_rng_and_empty_population():
    method = make_method(warmup=100)
    rows, windows = episode(method)
    method.on_episode(rows, windows, episode=1)
    set_shortest_path_prior(method.actor, .7, 30.)
    saved = deepcopy(method.state_dict())
    restored = make_method(warmup=100)
    restored.load_state_dict(saved)
    assert restored.model_updates == 4 and restored.model_frames == 3
    assert restored.actor.prior_weight == .7 and len(restored.replay) == 1
    np.testing.assert_array_equal(method.rng.integers(999, size=4), restored.rng.integers(999, size=4))
    with pytest.raises(ValueError, match='fresh run'):
        restored.load_state_dict(dict(saved, schema='routing_mamba_global_v1'))
    with pytest.raises(ValueError, match='fresh run'):
        make_method(warmup=0).load_state_dict(saved)
    changed_lr = make_method(warmup=100)
    changed_lr.args.actor_lr *= 2
    with pytest.raises(ValueError, match='fresh run'):
        changed_lr.load_state_dict(saved)
    assert saved['schema'] == SCHEMA
    method.begin_episode(SimpleNamespace(vehicles=[]), seed=42)
    empty = dict(windows[0], vehicles=[], total_vehicle_count=0, active_vehicle_seconds=0., reward=0., terminal=True)
    metrics = method.on_episode([], [empty], episode=2)
    assert np.isfinite(metrics['world_model_loss']) and metrics['imagined_events'] == 0
    assert metrics['routing_frames'] == 2 and metrics['model_frames'] == 4


def test_empty_episodes_do_not_satisfy_routing_warmup():
    method = make_method(warmup=4, frames=1)
    _, windows = episode(method)
    method.begin_episode(SimpleNamespace(vehicles=[]), seed=42)
    empty = dict(windows[0], vehicles=[], total_vehicle_count=0, active_vehicle_seconds=0., reward=0., terminal=True)
    metrics = method.on_episode([], [empty], episode=1)
    assert metrics['model_updates'] == 4 and metrics['model_frames'] == 1
    assert metrics['routing_frames'] == 0 and not metrics['model_warmup_complete']
    assert metrics['local_model_updates'] == 0
    # Historical empty-model optimization cannot warm the routing model.
    method.model_updates = 1000
    method.routing_frames = 1000
    method.warmup_updates = 5
    rows, windows = episode(method)
    metrics = method.on_episode(rows, windows, episode=2)
    assert metrics['local_model_updates'] <= 4
    assert not metrics['model_warmup_complete']


def test_real_prior_and_clipped_ppo():
    method = make_method()
    for parameter in method.actor.parameters():
        parameter.data.zero_()
    set_shortest_path_prior(method.actor, 1., 30.)
    rows, _ = episode(method)
    assert all(row['sp_prior_weight'] == 1. and np.isfinite(row['old_log_prob']) for row in rows)
    logits = torch.tensor([math.log(2.), math.log(.5)], requires_grad=True)
    clipped_policy_loss(logits, torch.zeros(2), torch.tensor([1., -1.])).backward()
    torch.testing.assert_close(logits.grad, torch.zeros(2))
