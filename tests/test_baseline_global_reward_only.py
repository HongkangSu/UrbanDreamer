"""Pure team reward must exclude timeout shaping from learning and ranking."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from tools import run_city_baseline_dataset as launcher
from tools import train_toy_baseline as trainer
from urban_dreamer.learning.baselines import common
from urban_dreamer.learning.baselines.gnac import GraphActorCriticMethod
from test_toy_baselines import make_args


def test_exact_window_bounds_and_same_window_reward_integral():
    windows = [dict(start_time=0., end_time=30., reward=-30., obs=np.ones(3), next_obs=np.ones(3)),
               dict(start_time=30., end_time=60., reward=-60.)]
    integral, end = common.global_reward_integral(windows, deadline=60.)
    assert end == 60.
    assert integral(25.) - integral(15.) == -10.
    assert integral(end) - integral(25.) == -65.


def test_global_only_runner_has_no_timeout_cost_in_rows_or_ranking(tmp_path, monkeypatch):
    runner = common.EpisodeRunner(make_args(tmp_path, global_reward_only=True))
    monkeypatch.setattr(runner.demand.__class__, 'sample', lambda self, **kwargs: runner.topology.vehicles)
    runner.deadline = 30.
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

        def reset(self):
            return dict(time=0., flow=flow, active_vehicle_count=runner.n, vehicles=vehicles)

        def advance_window(self, callback):
            return SimpleNamespace(next_state=dict(time=30., flow=flow,
                active_vehicle_count=runner.n, vehicles=vehicles), active_vehicle_seconds=runner.n * 30.)

    monkeypatch.setattr(common, 'SumoSession', Session)
    rows, windows, metrics = runner.run_episode('shortest', seed=42, collect=True)
    assert metrics['unfinished'] == runner.n
    assert metrics['vehicle_timeout_penalty_sum'] == 0.
    assert metrics['selection_score'] == metrics['global_reward_sum'] == -.5
    assert windows[0]['start_time'] == 0. and windows[0]['end_time'] == 30.
    assert all(row.get('timeout_return', 0.) == 0. for row in rows)


def test_global_gnac_update_ignores_poisoned_individual_timeout(tmp_path):
    runner = common.EpisodeRunner(make_args(tmp_path, global_reward_only=True))
    method = GraphActorCriticMethod(runner.args, runner)
    width = runner.network.edge_count * 4 + 2
    windows = [dict(obs=np.zeros(width, dtype=np.float32), next_obs=np.zeros(width, dtype=np.float32),
                    reward=-.2, terminal=True)]
    flow = runner.network.extract_flow({})
    rows = [dict(features=np.ones((2, 10), dtype=np.float32), mask=np.ones(2, dtype=bool),
                 action=i % 2, current_edge=0, edge_indices=np.array([1, 2]), flow=flow,
                 window=0, timeout_return=float('nan')) for i in range(4)]
    metrics = method.on_episode(rows, windows, episode=1)
    assert np.isfinite(metrics['actor_loss'])


def test_reward_mode_is_a_strict_resume_boundary():
    with pytest.raises(ValueError, match='global_reward_only'):
        trainer.validate_prior_resume({}, SimpleNamespace(global_reward_only=True, reward_objective='legacy_exposure'))
    trainer.validate_prior_resume({'global_reward_only': True}, SimpleNamespace(global_reward_only=True, reward_objective='legacy_exposure'))


def test_additional_methods_launch_same_physics_prior_and_global_only(tmp_path):
    args = SimpleNamespace(cities=launcher.CITIES, methods=launcher.EXTRA_METHODS,
        devices=['cuda:0'], episodes=5000, seed=42, checkpoint_every=1, eval_every=100,
        eval_seeds=3, forbid_revisit=False, global_reward_only=True, sp_prior_weight=1.,
        sp_prior_temperature=30., sp_prior_hold_episodes=200, sp_prior_off_episode=1000)
    runs = launcher.build_runs(tmp_path, tmp_path / 'source', args, 'global')
    assert len(runs) == 12
    for run in runs:
        command = run['command']
        assert run['device'] == 'cuda:0' and '--global-reward-only' in command
        assert '--forbid-revisit' not in command
        assert command[command.index('--sp-prior-weight') + 1] == '1.0'
        assert command[command.index('--checkpoint-every') + 1] == '1'
        assert run['simulation_settings']['traffic_model'] == 'meso'


def test_worker_visibility_prevents_rng_checkpoints_touching_other_gpu(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    archives = output / 'episodes' / 'train'
    archives.mkdir(parents=True)
    (archives / 'episode_000001.npz').write_bytes(b'fixture')
    entry = dict(id='run', method='iql', output_dir=str(output), command=['fake'],
                 log=str(tmp_path / 'log.txt'), state=str(tmp_path / 'state.json'))
    manifest = dict(runs=[entry], max_restarts=0, min_free_gb=.001,
                    source_snapshot=str(tmp_path), episode_budget=1, cuda_visible_devices='0')
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    captured = {}

    class Process:
        pid = 123
        returncode = 0

        def __init__(self, *args, **kwargs):
            captured.update(kwargs['env'])

        def poll(self):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, 'Popen', Process)
    assert launcher.supervise(SimpleNamespace(output_root=tmp_path, run_id='run')) == 0
    assert captured['CUDA_VISIBLE_DEVICES'] == '0'
