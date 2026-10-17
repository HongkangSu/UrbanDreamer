import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from micro_config_fixture import write_micro_config

PATH = Path(__file__).resolve().parents[1] / 'tools/train_toy_actor_critic.py'


def module():
    spec = importlib.util.spec_from_file_location('toy_model_free', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def args(tmp_path):
    return SimpleNamespace(reward_objective='legacy_exposure', global_reward_only=False, config=write_micro_config(tmp_path / 'legacy_micro'),
        output_dir=tmp_path, episodes=1, eval_every=1, eval_seeds=1, seed=42, device='cpu',
        threads=1, actor_lr=2e-4, global_critic_lr=3e-4, vehicle_limit=8,
        session='microstep', step_seconds=1., backend='traci', resume=False)


def windows(runner, rewards):
    width = runner.network.edge_count * 4 + 2
    return [dict(obs=np.zeros(width), next_obs=np.zeros(width), reward=r, terminal=i == len(rewards)-1)
            for i, r in enumerate(rewards)]


def rows():
    features = np.zeros((3, 10), dtype=np.float32)
    features[0, 0] = 1.
    features[1, 1] = 1.
    return [dict(vehicle_id=i, time=30.*i, features=features.copy(),
                 mask=np.array([True, True, False]), action=i, window=i) for i in range(2)]


def test_runner_has_only_global_critic_and_no_local_interface(tmp_path):
    import ast
    runner_module = module()
    runner = runner_module.RealActorCritic(args(tmp_path))
    assert runner.global_critic is not None
    assert not hasattr(runner, 'critic') and not hasattr(runner, 'local_weight')
    assert not hasattr(runner_module, 'close_event')
    imports = [node.module or '' for node in ast.walk(ast.parse(PATH.read_text())) if isinstance(node, ast.ImportFrom)]
    assert not any('world_model' in name or 'learning.training' in name for name in imports)


def test_macro_returns_include_empty_windows_and_terminal_masks(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    for p in runner.global_critic.parameters(): p.data.zero_()
    advantage, targets, values = runner.global_targets(windows(runner, [-.1, -.2, -.3]))
    torch.testing.assert_close(targets, torch.tensor([-.1-.95*(.2+.95*.3), -.2-.95*.3, -.3]))
    torch.testing.assert_close(advantage, targets)
    assert values.shape == (3,)


def test_global_reward_changes_actor_and_critic(tmp_path):
    first = module().RealActorCritic(args(tmp_path))
    second = module().RealActorCritic(args(tmp_path))
    first.update(rows(), windows(first, [-.5, -2.]))
    second.update(rows(), windows(second, [.5, 2.]))
    for model in ('actor', 'global_critic'):
        assert any(not torch.equal(a, b) for a, b in zip(getattr(first, model).parameters(), getattr(second, model).parameters()))


def test_shared_advantage_is_not_centered_away(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    for p in runner.global_critic.parameters(): p.data.zero_()
    events = rows()
    for event in events: event.update(action=0, window=0)
    with torch.no_grad():
        dist = torch.distributions.Categorical(logits=runner.actor.forward_logits(
            runner.tensor(np.stack([r['features'] for r in events])),
            torch.as_tensor(np.stack([r['mask'] for r in events]))))
        expected = -(dist.log_prob(torch.zeros(2, dtype=torch.long))*(-1.)+.002*dist.entropy()).mean()
    metrics = runner.update(events, windows(runner, [-1.]))
    assert metrics['actor_loss'] == pytest.approx(float(expected), abs=1e-6)
    assert metrics['actor_gradient_norm'] > 0


def test_no_decisions_still_updates_global_critic(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    before_actor = [p.detach().clone() for p in runner.actor.parameters()]
    before_critic = [p.detach().clone() for p in runner.global_critic.parameters()]
    metrics = runner.update([], windows(runner, [-.5, -2.]))
    assert metrics['actor_gradient_norm'] == 0
    assert all(torch.equal(a, b) for a, b in zip(before_actor, runner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_critic, runner.global_critic.parameters()))


def test_timeout_without_decisions_does_not_pollute_global_reward(tmp_path, monkeypatch):
    runner_module = module()
    runner = runner_module.RealActorCritic(args(tmp_path))
    runner.deadline = 30
    monkeypatch.setattr(runner_module.PoissonDemand, 'sample', lambda self, **kwargs: runner.topology.vehicles[:2])
    flow = runner.network.extract_flow({})
    vehicles = [dict(vehicle_id=v.vehicle_id, departed=True, arrived=False) for v in runner.topology.vehicles]
    class Session:
        done = False
        travel_times = {}
        microstep_seconds = .5
        metrics = {'microsteps': 60}
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def state(self, time): return dict(time=time, flow=flow, active_vehicle_count=runner.n, vehicles=vehicles)
        def reset(self): return self.state(0.)
        def advance_window(self, callback):
            return SimpleNamespace(next_state=self.state(30.), active_vehicle_seconds=runner.n*30.)
    monkeypatch.setattr(runner_module, 'SumoSession', Session)
    events, metrics = runner.run_episode(policy='stochastic', seed=42, collect=True)
    assert events == [] and runner.collected_windows[-1]['terminal']
    assert runner.n == 2 and runner.demand.expected_vehicle_count == 8
    assert metrics['global_reward_sum'] == -.5
    assert metrics['vehicle_timeout_penalty_sum'] == -5000*runner.n
    assert metrics['selection_score'] == -.5-5000*runner.n
    assert runner.collected_windows[0]['reward'] == -.5
    assert metrics['mean_exposure_seconds'] == 30.
    assert runner.collected_windows[0]['obs'][-1] == 1.
    assert 'reward_per_vehicle' not in metrics


def test_checkpoint_schema_optimizer_and_rng_resume(tmp_path):
    import random
    runner_module = module()
    runner = runner_module.RealActorCritic(args(tmp_path))
    runner.update(rows(), windows(runner, [-.5, -2.]))
    checkpoint = tmp_path/'latest.pt'
    runner.save(checkpoint, 17, 1.25)
    expected_torch, expected_numpy, expected_python = torch.rand(4), np.random.random(4), random.random()
    state = torch.load(checkpoint, weights_only=False)
    assert state['format'] == 'toy_global_individual_timeout_poisson_ac_v1'
    assert 'critic' not in state and 'critic_optimizer' not in state
    restored = runner_module.RealActorCritic(args(tmp_path))
    assert restored.load(checkpoint) == (18, 1.25)
    torch.testing.assert_close(torch.rand(4), expected_torch)
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    assert random.random() == expected_python
    # Direct update calls must use the same restored episode clock as the CLI.
    runner_module.set_episode_entropy(runner, 18)
    runner.update(rows(), windows(runner, [-.5, -2.]))
    restored.update(rows(), windows(restored, [-.5, -2.]))
    for model in ('actor', 'global_critic'):
        for key, value in getattr(runner, model).state_dict().items():
            torch.testing.assert_close(value, getattr(restored, model).state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize('end_time,arrived', [(30., (0,)), (30., (0, 1)), (15., (0,))])
def test_only_failed_vehicle_actions_receive_deadline_penalty(tmp_path, monkeypatch, end_time, arrived):
    runner_module = module()
    runner = runner_module.RealActorCritic(args(tmp_path))
    runner.deadline = 30
    demands = runner.topology.vehicles[:2]
    monkeypatch.setattr(runner_module.PoissonDemand, 'sample', lambda self, **kw: demands)
    flow = runner.network.extract_flow({})
    observations = SimpleNamespace(features=np.zeros((2, 3, 10), dtype=np.float32),
        mask=np.array([[True, True, False]]*2), edge_indices=np.array([[0, 1, 0]]*2))
    monkeypatch.setattr(runner_module, 'build_actor_observations', lambda **kw: observations)
    vehicles = [dict(vehicle_id=v.vehicle_id, current_edge=v.origin_index,
        destination_edge=v.destination_index, departed=True, arrived=False) for v in demands]
    class Session:
        done = False
        microstep_seconds = .5
        metrics = {'microsteps': 60}
        def __init__(self, *a, **kw): self.travel_times = {}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def state(self, time): return dict(time=time, flow=flow, active_vehicle_count=2, vehicles=vehicles)
        def reset(self): return self.state(0.)
        def advance_window(self, callback):
            callback(vehicles, flow, 0.)
            callback(vehicles, flow, 10.)
            self.travel_times = {demands[i].vehicle_id: end_time for i in arrived}
            self.done = True
            return SimpleNamespace(next_state=self.state(end_time), active_vehicle_seconds=60.)
    monkeypatch.setattr(runner_module, 'SumoSession', Session)
    events, metrics = runner.run_episode(policy='stochastic', seed=42, collect=True)
    failed = end_time >= runner.deadline and 1 not in arrived
    assert [r['timeout_penalty'] for r in events] == [0., 0., 0., -5000. if failed else 0.]
    assert [r['timeout_return'] for r in events] == [0., -5000. if failed else 0., 0., -5000. if failed else 0.]
    assert metrics['global_reward_sum'] == -.5
    assert metrics['vehicle_timeout_penalty_sum'] == (-5000. if failed else 0.)
    assert metrics['selection_score'] == -.5 + metrics['vehicle_timeout_penalty_sum']
    for parameter in runner.global_critic.parameters(): parameter.data.zero_()
    _, target, _ = runner.global_targets(runner.collected_windows)
    torch.testing.assert_close(target, torch.tensor([-.5]))


def test_timeout_changes_failed_actor_signal_but_not_critic_targets(tmp_path):
    first = module().RealActorCritic(args(tmp_path))
    second = module().RealActorCritic(args(tmp_path))
    plain, failed = rows(), rows()
    for row in plain: row.update(timeout_penalty=0., timeout_return=0.)
    for row in failed: row.update(timeout_penalty=-5000. if row['vehicle_id'] == 1 else 0.,
        timeout_return=-5000. if row['vehicle_id'] == 1 else 0.)
    first.update(plain, windows(first, [-.5, -.5]))
    second.update(failed, windows(second, [-.5, -.5]))
    for a, b in zip(first.global_critic.parameters(), second.global_critic.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert any(not torch.equal(a, b) for a, b in zip(first.actor.parameters(), second.actor.parameters()))


def test_team_timeout_poisson_checkpoint_rejected(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    path = tmp_path/'old_team_timeout.pt'
    torch.save({'format': 'toy_global_poisson_actor_critic_v1'}, path)
    with pytest.raises(ValueError, match='format'): runner.load(path)


def test_checkpoint_ranking_and_protocol_keep_timeout_scope_explicit(tmp_path, monkeypatch):
    import json
    runner_module = module()
    monkeypatch.setattr(runner_module.RealActorCritic, 'run_episode',
        lambda self, **kw: ([], dict(global_reward_sum=-1., vehicle_timeout_penalty_sum=0., selection_score=-1.)))
    monkeypatch.setattr(runner_module.RealActorCritic, 'update', lambda self, *a, **kw: {})
    def evaluate(runner, output, *, phase, **kw):
        return [dict(policy='deterministic', global_reward_sum=-.5 if phase == 'baseline' else -1.,
            vehicle_timeout_penalty_sum=-5000. if phase == 'baseline' else 0.,
            selection_score=-5000.5 if phase == 'baseline' else -1.)]
    monkeypatch.setattr(runner_module, 'evaluate', evaluate)
    assert runner_module.main(['--output-dir', str(tmp_path), '--episodes', '1', '--vehicle-limit', '2',
                              '--no-global-reward-only', '--reward-objective', 'legacy_exposure']) == 0
    checkpoint = torch.load(tmp_path/'best.pt', weights_only=False)
    assert checkpoint['episode'] == 1 and checkpoint['best'] == 1.
    protocol = json.loads((tmp_path/'protocol.json').read_text())
    assert protocol['reward_mode'] == 'global_with_individual_timeout'
    assert protocol['global_reward'] == '-active_vehicle_seconds/(N*60)'
    assert 'excluded from global critic' in protocol['timeout_penalty']
    assert 'diagnostic only' in protocol['reward_note']
    assert protocol['advantage_normalization'] == 'global_scale_only_std_floor_1_then_individual_timeout_clip_2_5'


def test_old_checkpoint_rejected(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    path = tmp_path/'old.pt'
    torch.save({'format': 'toy_real_actor_critic_v2'}, path)
    with pytest.raises(ValueError, match='format'): runner.load(path)


def test_fixed_demand_global_checkpoint_rejected(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    path = tmp_path/'old_global.pt'
    torch.save({'format': 'toy_global_actor_critic_v1'}, path)
    with pytest.raises(ValueError, match='format'): runner.load(path)


def test_episode_demand_is_reproducible_and_reference_topology_immutable(tmp_path, monkeypatch):
    runner_module = module()
    runner = runner_module.RealActorCritic(args(tmp_path))
    reference = runner.topology
    seen = []
    class EmptySession:
        done = True
        travel_times = {}
        microstep_seconds = 1.
        metrics = {'microsteps': 0}
        def __init__(self, config, prepared, topology):
            seen.append(topology)
            self.topology = topology
        def __enter__(self): return self
        def __exit__(self, *unused): pass
        def reset(self):
            return dict(time=0., flow=runner.network.extract_flow({}), active_vehicle_count=0,
                        vehicles=[dict(vehicle_id=v.vehicle_id, departed=False, arrived=False,
                                       destination_edge=runner.network.edge_to_index[
                                           runner.topology.destinations[v.destination_index]])
                                  for v in self.topology.vehicles])
    monkeypatch.setattr(runner_module, 'SumoSession', EmptySession)
    _, first = runner.run_episode(policy='deterministic', seed=10042)
    _, second = runner.run_episode(policy='random', seed=10042)
    assert seen[0].vehicles == seen[1].vehicles
    assert first['demand_realization_sha256'] == second['demand_realization_sha256']
    assert first['vehicle_count'] == len(seen[0].vehicles)
    assert runner.topology is reference and len(reference.vehicles) == 8
    assert runner.demand.expected_vehicle_count == 8
    populations = {len(runner.demand.sample(seed=42, episode_index=i)) for i in range(10)}
    assert len(populations) > 1
    assert runner.ids == {v.vehicle_id: i for i, v in enumerate(seen[-1].vehicles)}


def test_empty_poisson_episode_has_one_terminal_window_and_critic_update(tmp_path, monkeypatch):
    runner_module = module()
    runner = runner_module.RealActorCritic(args(tmp_path))
    monkeypatch.setattr(runner_module.PoissonDemand, 'sample', lambda self, **kwargs: ())
    class EmptySession:
        done = True
        travel_times = {}
        microstep_seconds = 1.
        metrics = {'microsteps': 0}
        def __init__(self, config, prepared, topology): assert topology.vehicles == ()
        def __enter__(self): return self
        def __exit__(self, *unused): pass
        def reset(self):
            return dict(time=0., flow=runner.network.extract_flow({}), active_vehicle_count=0, vehicles=[])
        def advance_window(self, callback): pytest.fail('empty episode should not advance SUMO')
    monkeypatch.setattr(runner_module, 'SumoSession', EmptySession)
    events, metrics = runner.run_episode(policy='stochastic', seed=42, collect=True)
    assert runner.n == 0 and runner.ids == {}
    assert events == [] and metrics['vehicle_count'] == 0
    assert metrics['completion_rate'] == 1. and metrics['global_reward_sum'] == 0.
    assert metrics['window_count'] == 1
    assert runner.collected_windows[0]['terminal']
    before_actor = [p.detach().clone() for p in runner.actor.parameters()]
    before_critic = [p.detach().clone() for p in runner.global_critic.parameters()]
    update = runner.update(events)
    assert update['actor_gradient_norm'] == 0
    assert np.isfinite(update['global_critic_loss'])
    assert all(torch.equal(a, b) for a, b in zip(before_actor, runner.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_critic, runner.global_critic.parameters()))


@pytest.mark.parametrize('option', ['--local-reward-weight', '--global-reward-weight', '--training-policy'])
def test_obsolete_setting_overrides_rejected(tmp_path, option):
    with pytest.raises(SystemExit): module().main(['--output-dir', str(tmp_path), option, '0'])


def test_evaluate_uses_argmax_and_forbids_stochastic(tmp_path):
    runner_module = module()
    calls = []
    runner = SimpleNamespace(run_episode=lambda **kwargs: (calls.append(kwargs) or [], {}))
    runner_module.evaluate(runner, tmp_path, episode=1, seeds=[42], phase='evaluation', policies=('deterministic',))
    assert calls[0]['policy'] == 'deterministic'
    with pytest.raises(ValueError, match='argmax'):
        runner_module.evaluate(runner, tmp_path, episode=1, seeds=[42], phase='holdout_final', policies=('stochastic',))


def test_collect_requires_stochastic_policy(tmp_path):
    runner = module().RealActorCritic(args(tmp_path))
    with pytest.raises(ValueError, match='stochastic'): runner.run_episode(policy='deterministic', seed=42, collect=True)


def test_live_training_sampling_and_argmax_evaluation(tmp_path, monkeypatch):
    import shutil
    import sys
    if shutil.which('sumo') is None and not (Path(sys.executable).parent/'sumo').exists(): pytest.skip('SUMO unavailable')
    runner = module().RealActorCritic(args(tmp_path))
    calls = []
    sample = runner.actor.sample_action
    def record(*args, **kwargs):
        calls.append(kwargs['deterministic'])
        return sample(*args, **kwargs)
    monkeypatch.setattr(runner.actor, 'sample_action', record)
    events, metrics = runner.run_episode(policy='stochastic', seed=42, collect=True)
    assert calls and not any(calls)
    assert metrics['completion_rate'] == 1.
    assert metrics['global_reward_sum'] == pytest.approx(-metrics['exposure_seconds']/(runner.n*60.))
    assert 'reward_per_vehicle' not in metrics
    for event in events:
        assert set(event) == {'vehicle_id', 'time', 'features', 'mask', 'action', 'window', 'timeout_penalty', 'timeout_return'}
        assert event['window']*30 <= event['time'] < (event['window']+1)*30
    assert np.isfinite(runner.update(events, runner.collected_windows)['global_critic_loss'])
    checkpoint = tmp_path/'live.pt'
    runner.save(checkpoint, 1, 1.)
    restored = module().RealActorCritic(args(tmp_path))
    assert restored.load(checkpoint) == (2, 1.)
    _, resumed_metrics = restored.run_episode(policy='deterministic', seed=42)
    calls.clear()
    _, evaluated_metrics = runner.run_episode(policy='deterministic', seed=42)
    assert calls and all(calls)
    assert evaluated_metrics['demand_realization_sha256'] == metrics['demand_realization_sha256']
    assert resumed_metrics['global_reward_sum'] == evaluated_metrics['global_reward_sum']
