import sys
from types import SimpleNamespace

import numpy as np
import pytest

from test_toy_model_free_actor_critic import args, module
from urban_dreamer.learning.episode_archive import EpisodeArchive, read_episode_archive
from urban_dreamer.traffic.environment import WindowResult


@pytest.mark.parametrize('population,completed', [(0, True), (1, True), (1, False)])
def test_episode_archive_observes_real_windows_independently_of_training_collection(
        tmp_path, monkeypatch, population, completed):
    implementation = module()
    runner = implementation.RealActorCritic(args(tmp_path))
    runner.deadline = 30
    vehicles = runner.topology.vehicles[:population]
    monkeypatch.setattr(implementation.PoissonDemand, 'sample', lambda self, **kw: vehicles)
    observed = {}

    class Recorder(EpisodeArchive):
        def __init__(self, metadata):
            super().__init__(metadata)
            observed['metadata'] = metadata
            observed['windows'] = []

        def start(self, state):
            super().start(state)
            observed['initial_state'] = state

        def append(self, result):
            super().append(result)
            observed['windows'].append(result)

        def save(self, path, **kwargs):
            super().save(path, **kwargs)
            observed['path'] = path
            observed['save'] = kwargs

    monkeypatch.setitem(sys.modules, 'urban_dreamer.learning.episode_archive',
                        SimpleNamespace(EpisodeArchive=Recorder))

    class Session:
        microstep_seconds = 1.
        metrics = {'microsteps': 30 if population else 0}

        def __init__(self, *unused):
            self.done = population == 0
            self.travel_times = {}

        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def state(self, now):
            return dict(time=now, done=self.done, flow=runner.network.extract_flow({}),
                        active_vehicle_count=population if not self.done else 0,
                        vehicles=[dict(vehicle_id=v.vehicle_id, departed=True,
                                       arrived=self.done) for v in vehicles])

        def reset(self):
            return self.state(0.)

        def advance_window(self, callback):
            assert population
            self.done = completed
            if completed:
                self.travel_times = {vehicles[0].vehicle_id: 30.}
            return WindowResult((), np.zeros(runner.network.edge_count),
                                np.zeros(runner.network.edge_count), population * 30.,
                                self.state(30.), 0., 30.)

    monkeypatch.setattr(implementation, 'SumoSession', Session)
    path = tmp_path / 'episode.npz'
    rows, archived_metrics = runner.run_episode(policy='random', seed=43, archive_path=path,
                                               archive_metadata=dict(phase='train', episode=1))
    assert rows == [] and runner.collected_windows == []
    assert observed['initial_state']['time'] == 0
    assert len(observed['windows']) == 1
    assert observed['windows'][0].end_time == 30.
    assert observed['path'] == path
    assert observed['save']['completed'] is completed
    assert observed['save']['deadline_seconds'] == 30
    assert observed['metadata']['total_vehicle_count'] == population
    assert len(observed['metadata']['demand_realization']) == population
    assert observed['metadata']['phase'] == 'train'
    assert observed['metadata']['session_protocol'] == 'microstep'
    assert observed['metadata']['edge_ids'] == list(runner.network.edge_ids)
    serialized = read_episode_archive(path, expected_session_protocol='microstep')
    assert serialized['completed'] is completed
    assert serialized['truncated'] is not completed
    assert serialized['terminated'].tolist() == [completed]
    assert serialized['metadata']['total_vehicle_count'] == population
    assert serialized['time'].tolist() == [0., 30.]
    if not population:
        assert serialized['metadata']['padded_empty_episode'] is True
        assert serialized['metadata']['physical_simulation_end_seconds'] == 0.
        assert serialized['metrics']['simulation_end_seconds'] == 0.
    # Recording is opt-in and does not alter collection, RNG, metrics, or physical behavior.
    observed.clear()
    plain_rows, plain_metrics = runner.run_episode(policy='random', seed=43)
    assert not observed and plain_rows == rows
    for value in (archived_metrics, plain_metrics):
        value.pop('wall_seconds')
    assert archived_metrics == plain_metrics


def test_archive_protocol_change_on_resume_rejected(tmp_path):
    implementation = module()
    runner = implementation.RealActorCritic(args(tmp_path))
    runner.args.save_episodes = True
    checkpoint = tmp_path / 'checkpoint.pt'
    runner.save(checkpoint, 1, 0.)
    assert not (tmp_path / 'checkpoint.pt.tmp').exists()
    restored = implementation.RealActorCritic(args(tmp_path))
    with pytest.raises(ValueError, match='archival protocol'):
        restored.load(checkpoint)
    restored.args.save_episodes = True
    restored.args.checkpoint_every = 25
    assert restored.load(checkpoint) == (2, 0.)


def test_cli_archives_only_training_and_respects_checkpoint_cadence(tmp_path, monkeypatch):
    implementation = module()
    calls, saves = [], []

    def run_episode(self, **kwargs):
        calls.append(kwargs)
        return [], dict(selection_score=-1., policy=kwargs['policy'])

    original_save = implementation.RealActorCritic.save

    def save(self, path, episode, best):
        saves.append((path.name, episode))
        original_save(self, path, episode, best)

    monkeypatch.setattr(implementation.RealActorCritic, 'run_episode', run_episode)
    monkeypatch.setattr(implementation.RealActorCritic, 'update', lambda *a, **kw: {})
    monkeypatch.setattr(implementation.RealActorCritic, 'save', save)
    implementation.main(['--output-dir', str(tmp_path), '--episodes', '5', '--eval-every', '3',
                         '--eval-seeds', '1', '--checkpoint-every', '2', '--save-episodes',
                         '--skip-reference-baselines', '--vehicle-limit', '1'])
    training = [item for item in calls if item.get('collect')]
    assert len(training) == 5
    for episode, item in enumerate(training, 1):
        assert item['archive_path'] == tmp_path / 'episodes/train' / f'episode_{episode:06d}.npz'
        assert item['archive_metadata'] == dict(phase='train', episode=episode)
    assert all('archive_path' not in item for item in calls if not item.get('collect'))
    assert [episode for name, episode in saves if name == 'latest.pt'] == [0, 2, 3, 4, 5]


def test_nonpositive_checkpoint_cadence_rejected(tmp_path):
    with pytest.raises(SystemExit):
        module().main(['--output-dir', str(tmp_path), '--checkpoint-every', '0'])


def test_skip_initial_evaluation_preserves_initial_checkpoints_and_resume_protocol(tmp_path, monkeypatch):
    import json
    import torch
    implementation = module()
    monkeypatch.setattr(implementation, 'evaluate',
                        lambda *a, **kw: pytest.fail('zero-budget run should skip evaluation'))
    implementation.main(['--output-dir', str(tmp_path), '--episodes', '0',
                         '--skip-initial-evaluation', '--vehicle-limit', '8'])
    for name in ('initial.pt', 'best.pt', 'latest.pt'):
        state = torch.load(tmp_path / name, weights_only=False)
        assert state['episode'] == 0 and state['best'] == float('inf')
    assert json.loads((tmp_path / 'protocol.json').read_text())['skip_initial_evaluation'] is True
    runner = implementation.RealActorCritic(args(tmp_path))
    with pytest.raises(ValueError, match='initial evaluation protocol'):
        runner.load(tmp_path / 'latest.pt')
    runner.args.skip_initial_evaluation = True
    assert runner.load(tmp_path / 'latest.pt') == (1, float('inf'))
