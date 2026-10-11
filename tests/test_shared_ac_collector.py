"""Legacy AC archive contracts that must survive collection consolidation."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

from test_toy_model_free_actor_critic import args, module
from urban_dreamer.learning.baselines.common import EpisodeRunner


def test_empty_ac_archive_keeps_physical_end_and_padded_transition(tmp_path, monkeypatch):
    ac = module()
    runner = ac.RealActorCritic(args(tmp_path))
    monkeypatch.setattr(ac.PoissonDemand, 'sample', lambda self, **kwargs: ())

    class EmptySession:
        done = True
        travel_times = {}
        microstep_seconds = 1.
        metrics = {'microsteps': 0}

        def __init__(self, *unused): pass
        def __enter__(self): return self
        def __exit__(self, *unused): pass
        def reset(self):
            return dict(time=0., flow=runner.network.extract_flow({}),
                        active_vehicle_count=0, vehicles=[])
        def advance_window(self, callback):
            raise AssertionError('zero demand must not advance physical time')

    monkeypatch.setattr(ac, 'SumoSession', EmptySession)
    archive = tmp_path / 'empty.npz'
    rows, metrics = runner.run_episode(policy='stochastic', seed=43, collect=True,
        archive_path=archive, archive_metadata={'episode': 1, 'phase': 'train'})
    assert not rows and metrics['simulation_end_seconds'] == 0
    assert metrics['window_count'] == 1 and metrics['selection_score'] == 0
    with np.load(archive, allow_pickle=False) as saved:
        metadata = json.loads(saved['summary_json'].tobytes())['metadata']
        assert metadata['episode'] == 1 and metadata['method'] == 'online_ac'
        assert metadata['demand_episode_index'] == 43
        assert metadata['vehicle_count'] == metadata['total_vehicle_count'] == 0
        assert metadata['demand_realization_sha256'] == metrics['demand_realization_sha256']
        assert metadata['padded_empty_episode'] is True
        assert metadata['physical_simulation_end_seconds'] == 0
        states = json.loads(saved['states_json'].tobytes())
        assert [state['time'] for state in states] == [0., 30.]


def test_real_ac_and_shared_runner_match_actions_rewards_and_resume(tmp_path):
    if shutil.which('sumo') is None and not (Path(sys.executable).parent / 'sumo').exists():
        pytest.skip('SUMO unavailable')
    ac = module()
    settings = args(tmp_path)
    runner = ac.RealActorCritic(settings)
    shared = EpisodeRunner(settings)
    actor = runner.actor

    def select(vehicles, observations, flow, now):
        with torch.no_grad():
            selection = actor.sample_action(torch.as_tensor(observations.features),
                torch.as_tensor(observations.mask), deterministic=False)
        indices, entropy = selection.indices_and_entropy()
        return indices, [dict(entropy=value) for value in entropy]

    ac_archive, shared_archive = tmp_path / 'ac.npz', tmp_path / 'shared.npz'
    ac_rows, ac_metrics = runner.run_episode(policy='stochastic', seed=43, collect=True,
        archive_path=ac_archive)
    shared_rows, shared_windows, shared_metrics = shared.run_episode(select, seed=43,
        collect=True, row_flow=False, label='stochastic', archive_path=shared_archive,
        archive_metadata={'episode': 43, 'phase': 'train'})
    assert ac_metrics['demand_realization_sha256'] == shared_metrics['demand_sha256']
    for key in ('vehicle_count', 'global_reward_sum', 'selection_score', 'exposure_seconds',
                'simulation_end_seconds', 'event_count', 'choices', 'branch_entropy'):
        assert ac_metrics[key] == shared_metrics[key]
    for left, right in zip(ac_rows, shared_rows, strict=True):
        assert left['vehicle_id'] == shared.ids[right['vehicle_id']]
        for key in set(left) - {'vehicle_id'}:
            np.testing.assert_equal(left[key], right[key])
    for left, right in zip(runner.collected_windows, shared_windows, strict=True):
        for key in left:
            np.testing.assert_equal(left[key], right[key])
    with np.load(ac_archive) as left, np.load(shared_archive) as right:
        for key in set(left.files) - {'summary_json'}:
            np.testing.assert_array_equal(left[key], right[key])

    checkpoint = tmp_path / 'latest.pt'
    runner.save(checkpoint, 1, ac_metrics['selection_score'])
    restored = ac.RealActorCritic(settings)
    assert restored.load(checkpoint) == (2, ac_metrics['selection_score'])
    _, expected = runner.run_episode(policy='deterministic', seed=44)
    _, actual = restored.run_episode(policy='deterministic', seed=44)
    assert actual['choices'] == expected['choices']
    assert actual['global_reward_sum'] == expected['global_reward_sum']
