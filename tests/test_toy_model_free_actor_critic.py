import importlib.util
from pathlib import Path

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1] / 'tools/train_toy_actor_critic.py'


def module():
    spec = importlib.util.spec_from_file_location('toy_model_free', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_model_free_reward_deadline_and_next_decision():
    runner = module()
    row = {'time': 10.}
    runner.close_event(row, 40., np.zeros(14), terminal=False)
    assert row['reward'] == -1.
    runner.close_event(row, 70., np.zeros(14), terminal=True, timeout=True)
    assert row['reward'] == -5002.
    assert row['terminal']


def test_terminal_returns_use_real_event_rewards_per_vehicle():
    runner = module()
    rewards = torch.tensor([-1., -2., -3.])
    values = torch.zeros(3)
    ids = torch.tensor([0, 1, 0])
    terminal = torch.tensor([False, True, True])
    returns = runner.lambda_returns_by_vehicle_ids(rewards, values, torch.ones(3, dtype=torch.bool), ids, terminal=terminal)
    torch.testing.assert_close(returns, torch.tensor([-3.85, -2., -3.]))


def test_runner_does_not_import_world_model_or_mainline_trainer():
    import ast
    tree = ast.parse(PATH.read_text())
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any('world_model' in name or 'learning.training' in name for name in imports)


def args(tmp_path, weight):
    import argparse
    return argparse.Namespace(config=PATH.parents[1] / 'configs/toy_cross_bottleneck.toml',
        output_dir=tmp_path, episodes=1, eval_every=1, eval_seeds=1, seed=42, device='cpu',
        threads=1, actor_lr=2e-4, critic_lr=3e-4, critic_updates=1, vehicle_limit=8,
        global_reward_weight=weight, global_critic_lr=3e-4, resume=False)


def test_global_mode_keeps_actor_and_local_critic_initialization_identical(tmp_path):
    runner = module()
    local = runner.RealActorCritic(args(tmp_path, 0.))
    global_runner = runner.RealActorCritic(args(tmp_path, 1.))
    assert global_runner.global_critic is not None
    for name in ('actor', 'critic'):
        for key, value in getattr(local, name).state_dict().items():
            torch.testing.assert_close(value, getattr(global_runner, name).state_dict()[key], rtol=0, atol=0)


def test_macro_returns_include_no_decision_windows_and_terminal_masks(tmp_path):
    runner = module().RealActorCritic(args(tmp_path, 1.))
    width = runner.network.edge_count * 4 + 2
    for parameter in runner.global_critic.parameters():
        parameter.data.zero_()
    windows = [dict(obs=np.zeros(width), next_obs=np.zeros(width), reward=r, terminal=i == 2)
               for i, r in enumerate([-.1, -.2, -.3])]
    advantage, targets, values = runner.global_targets(windows)
    torch.testing.assert_close(targets, torch.tensor([-.1 - .95 * (.2 + .95 * .3), -.2 - .95 * .3, -.3]))
    torch.testing.assert_close(advantage, targets)
    assert values.shape == (3,)


def test_window_advantage_changes_actor_gradient(tmp_path):
    runner_module = module()
    local = runner_module.RealActorCritic(args(tmp_path, 0.))
    global_runner = runner_module.RealActorCritic(args(tmp_path, 1.))
    for parameter in global_runner.global_critic.parameters():
        parameter.data.zero_()
    width = global_runner.network.edge_count * 4 + 2
    windows = [dict(obs=np.zeros(width), next_obs=np.zeros(width), reward=r, terminal=i == 1)
               for i, r in enumerate([-.5, -2.])]
    features = np.zeros((3, 10), dtype=np.float32)
    features[0, 0] = 1.
    features[1, 1] = 1.
    rows = [dict(vehicle_id=i, time=30.*i, obs=np.zeros(14), next_obs=np.zeros(14),
                 reward=-1., terminal=True, features=features.copy(),
                 mask=np.array([True, True, False]), action=i, window=i) for i in range(2)]
    local_metrics = local.update(rows, windows)
    metrics = global_runner.update(rows, windows)
    assert metrics['global_advantage_std'] > 0
    assert metrics['global_critic_loss'] > 0
    assert any(not torch.equal(a,b) for a,b in zip(local.actor.parameters(), global_runner.actor.parameters()))
    assert all(torch.equal(a,b) for a,b in zip(local.critic.parameters(), global_runner.critic.parameters()))


def test_live_macro_reward_sums_to_actual_exposure_and_event_windows(tmp_path):
    import pytest
    import shutil
    import sys
    if shutil.which('sumo') is None and not (Path(sys.executable).parent/'sumo').exists():
        pytest.skip('SUMO unavailable')
    runner = module().RealActorCritic(args(tmp_path, 1.))
    rows, metrics = runner.run_episode(policy='stochastic', seed=42, collect=True)
    windows = runner.collected_windows
    assert metrics['completion_rate'] == 1.
    assert sum(w['reward'] for w in windows) == pytest.approx(-metrics['exposure_seconds'] / (runner.n*60.))
    assert windows[-1]['terminal']
    for row in rows:
        assert row['window']*30 <= row['time'] < (row['window']+1)*30
    before = [p.detach().clone() for p in runner.global_critic.parameters()]
    result = runner.update(rows, windows)
    assert np.isfinite(result['global_critic_loss'])
    assert any(not torch.equal(a,b) for a,b in zip(before, runner.global_critic.parameters()))


def test_global_only_ignores_local_rewards_and_has_no_vehicle_critic(tmp_path):
    runner_module = module()
    options = args(tmp_path, 1.)
    options.local_reward_weight = 0.
    first = runner_module.RealActorCritic(options)
    second = runner_module.RealActorCritic(options)
    assert first.critic is None and first.critic_optimizer is None
    width = first.network.edge_count * 4 + 2
    windows = [dict(obs=np.zeros(width), next_obs=np.zeros(width), reward=r, terminal=i == 1)
               for i,r in enumerate([-.5, -2.])]
    features = np.zeros((3, 10), dtype=np.float32)
    features[0,0] = 1.
    features[1,1] = 1.
    rows = [dict(vehicle_id=i, time=30.*i, obs=np.zeros(14), next_obs=np.zeros(14),
                 reward=-1., terminal=True, features=features.copy(), mask=np.array([True,True,False]),
                 action=i, window=i) for i in range(2)]
    import copy
    changed = copy.deepcopy(rows)
    for row in changed:
        row['reward'] = 123456.
        row['obs'][:] = np.nan
        row['next_obs'][:] = np.nan
    a = first.update(rows, windows)
    b = second.update(changed, windows)
    assert 'critic_loss' not in a and 'explained_variance' not in a
    assert a['global_critic_loss'] > 0
    assert a == b
    for model in ('actor', 'global_critic'):
        for key,value in getattr(first, model).state_dict().items():
            torch.testing.assert_close(value, getattr(second, model).state_dict()[key], rtol=0, atol=0)
    first.save(tmp_path/'global_only.pt',1,1.)
    state=torch.load(tmp_path/'global_only.pt',weights_only=False)
    assert state['critic'] is None and state['critic_optimizer'] is None


def test_global_only_shared_advantage_is_not_centered_away(tmp_path):
    options=args(tmp_path,1.); options.local_reward_weight=0.
    runner=module().RealActorCritic(options)
    for p in runner.global_critic.parameters():
        p.data.zero_()
    features=np.zeros((3,10),dtype=np.float32); features[0,0]=1.; features[1,1]=1.
    mask=np.array([True,True,False])
    rows=[dict(features=features.copy(),mask=mask.copy(),action=0,window=0) for _ in range(2)]
    width=runner.network.edge_count*4+2
    windows=[dict(obs=np.zeros(width),next_obs=np.zeros(width),reward=-1.,terminal=True)]
    with torch.no_grad():
        dist=torch.distributions.Categorical(logits=runner.actor.forward_logits(runner.tensor(np.stack([features]*2)),torch.as_tensor(np.stack([mask]*2))))
        expected=-(dist.log_prob(torch.zeros(2,dtype=torch.long))*(-1.)+.002*dist.entropy()).mean()
    metrics=runner.update(rows,windows)
    assert abs(metrics['actor_loss']-float(expected))<1e-6
    assert metrics['actor_gradient_norm']>0


def test_global_only_keeps_nonarrival_penalty_in_terminal_team_reward(tmp_path, monkeypatch):
    from types import SimpleNamespace
    runner_module=module()
    options=args(tmp_path,1.);options.local_reward_weight=0.
    runner=runner_module.RealActorCritic(options)
    runner.deadline=30
    flow=runner.network.extract_flow({})
    vehicles=[dict(vehicle_id=v.vehicle_id,departed=True,arrived=False) for v in runner.topology.vehicles]
    class Session:
        done=False
        travel_times={}
        microstep_seconds=.5
        metrics={'microsteps':60}
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def state(self,time):return dict(time=time,flow=flow,active_vehicle_count=runner.n,vehicles=vehicles)
        def reset(self):return self.state(0.)
        def advance_window(self,callback):
            return SimpleNamespace(next_state=self.state(30.),active_vehicle_seconds=runner.n*30.)
    monkeypatch.setattr(runner_module,'SumoSession',Session)
    _,metrics=runner.run_episode(policy='stochastic',seed=42,collect=True)
    assert runner.collected_windows[-1]['terminal']
    assert runner.collected_windows[-1]['reward']==-.5-5000*runner.n
    assert metrics['unfinished']==runner.n
