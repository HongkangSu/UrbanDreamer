import argparse
import copy
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines import METHODS, build_method
from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.gnac import GraphActorCriticMethod, GraphNextHopActor
from urban_dreamer.models.policy import NextHopActor
from micro_config_fixture import write_micro_config

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/train_toy_baseline.py'


def module():
    spec = importlib.util.spec_from_file_location('toy_baseline_tool', TOOL)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_args(tmp_path, **overrides):
    values = dict(reward_objective='legacy_exposure', global_reward_only=False, method='gnac', config=write_micro_config(tmp_path), session='microstep',
        output_dir=tmp_path, episodes=1, eval_every=1, eval_seeds=1, seed=42, device='cpu',
        threads=1, actor_lr=2e-4, critic_lr=3e-4, critic_updates=1, vehicle_limit=8, resume=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def sumo_available():
    return shutil.which('sumo') is not None or (Path(sys.executable).parent / 'sumo').exists()


def test_method_registry_and_placeholders():
    assert set(METHODS) == {'gnac', 'dreamer', 'mbpo', 'dta', 'ippo', 'mappo', 'iql', 'mamba'}
    assert METHODS['gnac'] is GraphActorCriticMethod
    for name in ('dreamer', 'mbpo', 'dta'):
        with pytest.raises(NotImplementedError, match='not implemented yet'):
            build_method(name, None, None)
    with pytest.raises(ValueError, match='unknown baseline method'):
        build_method('unknown', None, None)


def graph_actor_inputs(batch=3):
    torch.manual_seed(0)
    adjacency = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float32)
    actor = GraphNextHopActor(adjacency)
    features = torch.randn(batch, 2, 10)
    mask = torch.tensor([[True, True], [True, False], [True, True]])[:batch]
    edge_indices = torch.tensor([[1, 2], [1, 0], [2, 3]])[:batch]
    current_edges = torch.tensor([0, 0, 1])[:batch]
    flow = torch.rand(4, 8)
    return actor, features, mask, edge_indices, current_edges, flow


def test_graph_actor_logits_mask_and_gradients():
    actor, features, mask, edge_indices, current_edges, flow = graph_actor_inputs()
    logits = actor.forward_logits(features, mask, edge_indices, current_edges, flow)
    assert logits.shape == (3, 2)
    assert torch.isfinite(logits[mask]).all()
    assert logits[1, 1] == -torch.inf
    selection = actor.sample_action(features, mask, edge_indices, current_edges, flow)
    assert selection.candidate_index.shape == (3,)
    assert selection.log_probability.shape == (3,)
    assert selection.entropy.shape == (3,)
    assert selection.candidate_index[1] == 0
    deterministic = actor.sample_action(features, mask, edge_indices, current_edges, flow, deterministic=True)
    assert torch.equal(deterministic.candidate_index, logits.argmax(dim=-1))
    loss = -selection.log_probability.mean()
    loss.backward()
    gradients = [p.grad for p in actor.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)


def test_graph_actor_batched_flow_matches_shared_flow():
    actor, features, mask, edge_indices, current_edges, flow = graph_actor_inputs(batch=2)
    shared = actor.forward_logits(features, mask, edge_indices, current_edges, flow)
    batched_flow = torch.stack([flow, flow + 1.])
    batched = actor.forward_logits(features, mask, edge_indices, current_edges, batched_flow)
    assert batched.shape == (2, 2)
    torch.testing.assert_close(batched[0], shared[0])
    assert not torch.allclose(batched[1], shared[1])


def test_gnac_global_update_changes_actor_and_global_critic(tmp_path):
    args = make_args(tmp_path)
    runner = common.EpisodeRunner(args)
    method = GraphActorCriticMethod(args, runner)
    assert isinstance(method.global_critic, common.GlobalTrafficCritic)
    assert not hasattr(method, 'critic')
    width = runner.network.edge_count * 4 + 2
    windows = [dict(obs=np.zeros(width, dtype=np.float32), next_obs=np.zeros(width, dtype=np.float32),
                    reward=reward, terminal=index == 2)
               for index, reward in enumerate([-.1, -.2, -.3])]
    flow = runner.network.extract_flow({})
    rng = np.random.default_rng(0)
    rows = [dict(vehicle_id=i, time=30. * i, action=i % 2, window=i % 3,
                 features=rng.normal(size=(2, 10)).astype(np.float32),
                 mask=np.array([True, True]), edge_indices=np.array([1, 2]),
                 current_edge=0, flow=flow.copy()) for i in range(6)]
    before_actor = [p.detach().clone() for p in method.actor.parameters()]
    before_global = [p.detach().clone() for p in method.global_critic.parameters()]
    metrics = method.on_episode(rows, windows, episode=1)
    for key in ('global_critic_loss', 'global_explained_variance', 'global_advantage_std',
                'actor_loss', 'actor_gradient_norm'):
        assert math.isfinite(metrics[key]), key
    assert metrics['global_critic_loss'] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, method.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_global, method.global_critic.parameters()))


def test_gnac_trains_global_critic_in_a_decision_free_window(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path))
    method = GraphActorCriticMethod(runner.args, runner)
    width = runner.network.edge_count * 4 + 2
    windows = [dict(obs=np.zeros(width, dtype=np.float32), next_obs=np.zeros(width, dtype=np.float32),
                    reward=-.5, terminal=True)]
    actor_before = copy.deepcopy(method.actor.state_dict())
    critic_before = copy.deepcopy(method.global_critic.state_dict())
    metrics = method.on_episode([], windows, episode=1)
    assert metrics['global_critic_loss'] > 0.
    assert metrics['actor_loss'] == 0.
    for key, value in actor_before.items():
        torch.testing.assert_close(method.actor.state_dict()[key], value, rtol=0, atol=0)
    assert any(not torch.equal(value, method.global_critic.state_dict()[key])
               for key, value in critic_before.items())


def test_no_vehicle_reward_update_interfaces():
    assert not hasattr(common, 'actor_critic_update')
    assert not hasattr(common, 'close_event')


def test_standard_baseline_evaluation_rejects_stochastic_actor(tmp_path):
    with pytest.raises(ValueError, match='argmax'):
        module().evaluate(None, None, tmp_path, episode=0, seeds=[42], phase='evaluation',
                          policies=('stochastic',))


def test_episode_runner_window_rewards_and_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner = common.EpisodeRunner(make_args(tmp_path))
    monkeypatch.setattr(runner.demand.__class__, 'sample', lambda self, **kwargs: runner.topology.vehicles)
    runner.deadline = 30
    flow = runner.network.extract_flow({})
    vehicles = [dict(vehicle_id=v.vehicle_id, departed=True, arrived=False) for v in runner.topology.vehicles]

    class Session:
        done = False
        travel_times = {}
        microstep_seconds = .5
        metrics = {'microsteps': 60}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def state(self, time):
            return dict(time=time, flow=flow, active_vehicle_count=runner.n, vehicles=vehicles)

        def reset(self):
            return self.state(0.)

        def advance_window(self, callback):
            return SimpleNamespace(next_state=self.state(30.), active_vehicle_seconds=runner.n * 30.)

    monkeypatch.setattr(common, 'SumoSession', Session)
    rows, windows, metrics = runner.run_episode(runner.make_shortest_select(), seed=42, collect=True, label='shortest')
    assert rows == []
    assert metrics['unfinished'] == runner.n
    assert metrics['policy'] == 'shortest'
    assert windows[-1]['terminal']
    expected = -.5
    assert windows[-1]['reward'] == pytest.approx(expected)
    assert 'reward_per_vehicle' not in metrics
    assert metrics['global_reward_sum'] == pytest.approx(expected)
    assert metrics['vehicle_timeout_penalty_sum'] == -5000. * runner.n
    assert metrics['selection_score'] == expected - 5000. * runner.n



def test_real_mixed_cohort_timeout_cost_is_individual(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner = common.EpisodeRunner(make_args(tmp_path, vehicle_limit=3))
    monkeypatch.setattr(runner.demand.__class__, 'sample', lambda self, **kwargs: runner.topology.vehicles)
    runner.deadline = 60
    flow = runner.network.extract_flow({})
    vehicles = [dict(vehicle_id=v.vehicle_id,
                     current_edge=runner.network.edge_to_index[runner.topology.origins[v.origin_index]],
                     destination_edge=runner.network.edge_to_index[runner.topology.destinations[v.destination_index]],
                     departed=True, arrived=False, remaining_distance=1., pending_decision=True)
                for v in runner.topology.vehicles]

    class Session:
        done = False
        microstep_seconds = .5
        metrics = {'microsteps': 120}

        def __init__(self, *args):
            self.now = 0.
            self.travel_times = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def state(self):
            return dict(time=self.now, flow=flow, active_vehicle_count=2, vehicles=vehicles)

        def reset(self):
            return self.state()

        def advance_window(self, callback):
            # Third vehicle is unfinished but never made a decision.
            callback(vehicles[:2], flow, self.now)
            self.now += 30.
            if self.now == 60.:
                self.travel_times[vehicles[0]['vehicle_id']] = 60.
                vehicles[0]['arrived'] = True
            return SimpleNamespace(next_state=self.state(), active_vehicle_seconds=60.)

    monkeypatch.setattr(common, 'SumoSession', Session)
    rows, windows, metrics = runner.run_episode('shortest', seed=42, collect=True)
    arrived = [row for row in rows if row['vehicle_id'] == vehicles[0]['vehicle_id']]
    failed = [row for row in rows if row['vehicle_id'] == vehicles[1]['vehicle_id']]
    assert len(rows) == 4
    assert all(row['timeout_return'] == row['timeout_penalty'] == 0. for row in arrived)
    assert [row['timeout_return'] for row in failed] == [-5000., -5000.]
    assert [row['timeout_penalty'] for row in failed] == [0., -5000.]
    assert [window['reward'] for window in windows] == pytest.approx([-1/3, -1/3])
    assert metrics['vehicle_timeout_penalty_sum'] == -10000.
    assert metrics['global_reward_sum'] == pytest.approx(-2/3)
    assert metrics['selection_score'] == pytest.approx(-10000. - 2/3)


def test_policy_step_keeps_arrived_gradient_independent_of_other_timeout():
    def run(timeout):
        logits = torch.nn.Parameter(torch.zeros(2, 2))
        optimizer = torch.optim.SGD([logits], lr=0.)
        rows = [dict(action=0, timeout_return=0.), dict(action=0, timeout_return=timeout)]
        common.policy_gradient_step(compute_logits=lambda rows: logits,
            actor_parameters=[logits], actor_optimizer=optimizer, rows=rows,
            advantages=[-1., -2.], entropy_coef=0., grad_clip=100.)
        return logits.grad.clone()
    baseline, failed = run(0.), run(-5000.)
    torch.testing.assert_close(baseline[0], failed[0])
    assert not torch.equal(baseline[1], failed[1])

def test_gnac_checkpoint_roundtrip(tmp_path):
    args = make_args(tmp_path)
    runner = common.EpisodeRunner(args)
    method = GraphActorCriticMethod(args, runner)
    original = copy.deepcopy(method.state_dict())
    path = tmp_path / 'ckpt.pt'
    common.save_checkpoint(path, method_name='gnac', episode=3, best=1.5,
        method_state=method.state_dict(), args=args, prepared=runner.prepared, config=runner.config)
    with torch.no_grad():
        for parameter in method.actor.parameters():
            parameter.add_(1.)
    state = common.load_checkpoint(path, method_name='gnac',
        prepared=runner.prepared, config=runner.config, device=runner.device)
    assert state['episode'] == 3 and state['best'] == 1.5
    method.load_state_dict(state['method_state'])
    for key, value in original['actor'].items():
        torch.testing.assert_close(method.actor.state_dict()[key], value, rtol=0, atol=0)
    for key, value in original['global_critic'].items():
        torch.testing.assert_close(method.global_critic.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match='mismatch'):
        common.load_checkpoint(path, method_name='dreamer',
            prepared=runner.prepared, config=runner.config, device=runner.device)


def test_gnac_cli_smoke(tmp_path):
    if not sumo_available():
        pytest.skip('SUMO unavailable')
    tool = module()
    output = tmp_path / 'run'
    status = tool.main(['--method', 'gnac', '--output-dir', str(output), '--episodes', '2',
                        '--eval-every', '1', '--eval-seeds', '1', '--vehicle-limit', '8'])
    assert status == 0
    records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
    train = [r for r in records if r['phase'] == 'train']
    assert len(train) == 2
    for record in train:
        assert math.isfinite(record['global_reward_sum'])
        assert math.isfinite(record['actor_loss'])
        assert math.isfinite(record['global_critic_loss'])
        assert math.isfinite(record['global_explained_variance'])
    assert {r['policy'] for r in records if r['phase'] == 'baseline'} == {'random', 'shortest', 'deterministic'}
    assert any(r['phase'] == 'evaluation' for r in records)
    assert any(r['phase'] == 'holdout_final' for r in records)
    assert any(r['phase'] == 'holdout_best' for r in records)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert protocol['method'] == 'gnac'
    assert protocol['reward_mode'] == 'global_only'
    assert protocol['training_policy'] == 'stochastic'
    assert protocol['evaluation_policy'] == 'argmax'
    assert 'local_reward_weight' not in protocol and 'global_reward_weight' not in protocol
    assert all(r['policy'] == 'stochastic' for r in train)
    assert {r['policy'] for r in records if r['phase'] in
            {'evaluation', 'holdout_best'}} == {'deterministic'}
    assert 'stochastic' not in {r['policy'] for r in records if r['phase'] == 'holdout_final'}
    assert protocol['demand']['expected_vehicle_count'] == pytest.approx(8)
    assert protocol['demand']['law'] == 'independent_poisson_per_od'
    for name in ('best.pt', 'initial.pt', 'latest.pt'):
        assert (output / name).is_file()


@pytest.mark.parametrize('population', [0, 3])
def test_runner_realized_poisson_demand_and_zero_terminal_window(tmp_path, monkeypatch, population):
    from dataclasses import replace
    from types import SimpleNamespace
    runner = common.EpisodeRunner(make_args(tmp_path))
    vehicles = tuple(replace(v, vehicle_id=f'poisson_{index}')
                     for index, v in enumerate(runner.topology.vehicles[:population]))
    calls = []
    monkeypatch.setattr(runner.demand.__class__, 'sample',
                        lambda self, **kwargs: calls.append(kwargs) or vehicles)
    flow = runner.network.extract_flow({})

    class Session:
        microstep_seconds = 1.
        metrics = {'microsteps': 30}

        def __init__(self, config, prepared, topology):
            assert topology.vehicles == vehicles
            self.done = population == 0
            self.travel_times = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def state(self, time):
            return dict(time=time, flow=flow, active_vehicle_count=population if time == 0 else 0,
                        vehicles=[dict(vehicle_id=v.vehicle_id, departed=True, arrived=time > 0) for v in vehicles])

        def reset(self):
            return self.state(0.)

        def advance_window(self, callback):
            self.done = True
            self.travel_times = {v.vehicle_id: 30. for v in vehicles}
            return SimpleNamespace(next_state=self.state(30.), active_vehicle_seconds=population * 30.,
                action_counts=np.zeros(runner.network.edge_count), departure_counts=np.zeros(runner.network.edge_count))

    monkeypatch.setattr(common, 'SumoSession', Session)
    prepared_demands = []
    rows, windows, metrics = runner.run_episode('shortest', seed=91, record_windows=True,
        on_demand=lambda topology, *, seed: prepared_demands.append((topology, seed)))
    assert calls == [{'seed': 42, 'episode_index': 91}]
    assert prepared_demands == [(runner.episode_topology, 91)]
    assert runner.n == population and len(runner.ids) == population
    assert len(windows) == 1 and windows[0]['terminal']
    assert windows[0]['total_vehicle_count'] == population
    assert windows[0]['reward'] == pytest.approx(-.5 if population else 0.)
    assert metrics['vehicle_count'] == population and metrics['completion_rate'] == 1.
    assert len(metrics['demand_sha256']) == 64
    assert np.isfinite(windows[0]['obs']).all()


def test_baseline_poisson_rng_is_reproducible_and_episode_specific(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path, vehicle_limit=0))
    first = runner.demand.sample(seed=runner.args.seed, episode_index=43)
    np.random.seed(7)
    np.random.random(100)
    assert runner.demand.sample(seed=runner.args.seed, episode_index=43) == first
    assert runner.demand.sample(seed=runner.args.seed, episode_index=44) != first
    assert len(runner.topology.vehicles) == 320
