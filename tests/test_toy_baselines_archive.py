"""Real-episode archival stays independent of baseline learning representations."""
import argparse
import importlib.util
import json
from pathlib import Path

from urban_dreamer.workflow.config import MethodConfig
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines import common
from urban_dreamer.traffic.environment import NextHopDecision, WindowResult


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    spec = importlib.util.spec_from_file_location('archive_baseline_tool', ROOT / 'tools/train_toy_baseline.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


@pytest.mark.parametrize('population,completed', [(0, True), (2, True), (2, False)])
@pytest.mark.parametrize('record_windows', [False, True])
def test_runner_archives_all_real_windows_and_provenance(tmp_path, monkeypatch, population, completed, record_windows):
    args = argparse.Namespace(config=ROOT / 'configs/toy_cross_bottleneck.toml', threads=1,
        device='cpu', seed=42, vehicle_limit=2, method='gnac', session='event', backend='libsumo')
    runner = common.EpisodeRunner(args)
    runner.deadline = 30
    population_vehicles = runner.topology.vehicles[:population]
    monkeypatch.setattr(type(runner.demand), 'sample', lambda self, **kw: population_vehicles)
    flow = runner.network.extract_flow({})
    calls = []

    class Archive:
        def __init__(self, metadata):
            self.metadata = metadata
            calls.append(self)

        def start(self, state):
            self.start_state = state

        def append(self, result):
            self.result = result

        def save(self, path, **kwargs):
            self.path, self.saved = path, kwargs

    monkeypatch.setitem(sys.modules, 'urban_dreamer.learning.episode_archive',
                        SimpleNamespace(EpisodeArchive=Archive))

    class Session:
        microstep_seconds = 1.
        metrics = {'microsteps': 30}

        def __init__(self, *args):
            self.done = population == 0
            self.travel_times = {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def state(self, time):
            return dict(time=time, flow=flow.copy(), active_vehicle_count=population,
                vehicles=[dict(vehicle_id=v.vehicle_id, departed=True, arrived=completed and time > 0)
                          for v in population_vehicles])

        def reset(self):
            return self.state(0.)

        def advance_window(self, callback):
            self.done = completed
            if completed:
                self.travel_times = {v.vehicle_id: 30. for v in population_vehicles}
            decisions = ((NextHopDecision(population_vehicles[0].vehicle_id, 0, 1, 2.),)
                         if population else ())
            counts = np.zeros(runner.network.edge_count)
            counts[1] = len(decisions)
            return WindowResult(decisions, counts, np.zeros_like(counts), population * 30.,
                                self.state(30.), 0., 30.)

    runner.session_factory = Session
    path = tmp_path / 'episode_000001.npz'
    _, windows, metrics = runner.run_episode('shortest', seed=43, record_windows=record_windows,
        archive_path=path, archive_metadata=dict(episode=1, phase='train'))
    assert len(calls) == len(windows) == 1
    archive = calls[0]
    assert archive.start_state['time'] == 0.
    assert archive.result.end_time == 30.
    assert len(archive.result.decisions) == bool(population)
    assert ('flow' in windows[0]) == record_windows
    assert archive.path == path
    assert archive.saved['completed'] == completed
    assert archive.saved['deadline_seconds'] == 30
    assert archive.saved['metrics'] == metrics
    assert archive.metadata['total_vehicle_count'] == population
    assert len(archive.metadata['realized_demand']) == population
    assert archive.metadata['edge_ids'] == list(runner.network.edge_ids)
    assert archive.metadata['session_protocol'] == 'event'
    assert archive.metadata['backend'] == 'libsumo'
    assert archive.metadata['phase'] == 'train'
    assert archive.metadata['demand']['law'] == 'independent_poisson_per_od'
    assert archive.metadata['input_hashes'] == runner.prepared.input_hashes


@pytest.mark.parametrize('skip_initial', [False, True])
def test_cli_archive_only_training_cadence_final_and_resume(tmp_path, monkeypatch, skip_initial):
    tool = load_tool()
    calls, checkpoints, best_values = [], [], []

    class Runner:
        def __init__(self, args, **kw):
            self.args = args
            self.device = 'cpu'
            self.config = SimpleNamespace(resolved_snapshot=lambda: {}, semantic_fingerprint='test',
                method=MethodConfig())
            self.prepared = SimpleNamespace(input_hashes={})
            self.deadline = 30
            self.demand = SimpleNamespace(duration_seconds=30, expected_vehicle_count=1, od_rates=[])

        def run_episode(self, select, **kw):
            calls.append(kw)
            return [], [], dict(selection_score=-1., policy=kw['label'])

    method = SimpleNamespace(select=lambda *a, **kw: None, state_dict=lambda: {},
        load_state_dict=lambda state: None, on_episode=lambda *a, **kw: {})
    monkeypatch.setattr(tool.common, 'EpisodeRunner', Runner)
    monkeypatch.setattr(tool, 'build_method', lambda *a: method)
    def save(runner, method, args, path, episode, best):
        checkpoints.append((path.name, episode))
        best_values.append(best)

    monkeypatch.setattr(tool, 'save', save)
    monkeypatch.setattr(tool.common, 'load_checkpoint', lambda *a, **kw: dict(method_state={}, episode=0))
    argv = ['--method', 'gnac', '--output-dir', str(tmp_path), '--episodes', '5',
            '--eval-every', '4', '--eval-seeds', '1', '--checkpoint-every', '3',
            '--save-episodes', '--skip-reference-baselines']
    if skip_initial:
        argv.append('--skip-initial-evaluation')
    assert tool.main(argv) == 0
    assert calls[0]['label'] == ('stochastic' if skip_initial else 'deterministic')
    assert checkpoints[:3] == [('best.pt', 0), ('initial.pt', 0), ('latest.pt', 0)]
    assert best_values[:3] == [float('inf') if skip_initial else 1.] * 3
    assert [episode for name, episode in checkpoints if name == 'latest.pt'] == [0, 3, 5]
    training = [call for call in calls if call.get('archive_path')]
    assert len(training) == 5
    assert [call['archive_path'].name for call in training] == [f'episode_{i:06d}.npz' for i in range(1, 6)]
    assert all(call['archive_metadata']['phase'] == 'train' for call in training)
    assert all('archive_path' not in call for call in calls if call['label'] != 'stochastic')
    protocol = json.loads((tmp_path / 'protocol.json').read_text())
    assert protocol['episode_archive']['phases'] == ['train']
    assert protocol['skip_initial_evaluation'] == skip_initial
    saved_args = protocol['args']
    monkeypatch.setattr(tool.common, 'load_checkpoint', lambda *a, **kw:
        dict(method_state={}, episode=3, best=1., args=saved_args,
             torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state()))
    calls.clear()
    assert tool.main(argv + ['--resume']) == 0
    training = [call for call in calls if call.get('archive_path')]
    assert [call['archive_metadata']['episode'] for call in training] == [4, 5]
    changed = [arg for arg in argv if arg != '--skip-initial-evaluation']
    if not skip_initial:
        changed.append('--skip-initial-evaluation')
    with pytest.raises(ValueError, match='initial evaluation setting'):
        tool.main(changed + ['--resume'])


def test_failed_checkpoint_write_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / 'latest.pt'
    path.write_bytes(b'previous checkpoint')

    def fail(payload, output):
        output.write_bytes(b'partial replacement')
        raise OSError('disk write interrupted')

    monkeypatch.setattr(common.torch, 'save', fail)
    with pytest.raises(OSError, match='interrupted'):
        common.save_checkpoint(path, method_name='gnac', episode=1, best=1., method_state={},
            args=argparse.Namespace(), prepared=SimpleNamespace(input_hashes={}),
            config=SimpleNamespace(semantic_fingerprint='test'))
    assert path.read_bytes() == b'previous checkpoint'
    assert list(tmp_path.iterdir()) == [path]


def test_event_rows_share_only_their_own_window_flow_snapshot(tmp_path, monkeypatch):
    runner = common.EpisodeRunner(argparse.Namespace(config=ROOT / 'configs/toy_cross_bottleneck.toml',
        threads=1, device='cpu', seed=42, vehicle_limit=1))
    runner.deadline = 60
    monkeypatch.setattr(type(runner.demand), 'sample', lambda self, **kw: runner.topology.vehicles)
    v = runner.topology.vehicles[0]
    vehicle = dict(vehicle_id=v.vehicle_id, departed=True, arrived=False,
        current_edge=runner.network.edge_to_index[runner.topology.origins[v.origin_index]],
        destination_edge=runner.network.edge_to_index[runner.topology.destinations[v.destination_index]],
        remaining_distance=1., pending_decision=True)
    flow = runner.network.extract_flow({})

    class Session:
        done = False
        microstep_seconds = 1.
        metrics = {'microsteps': 60}
        travel_times = {}

        def __init__(self, *args):
            self.now = 0.

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def state(self):
            return dict(time=self.now, flow=flow, active_vehicle_count=1, vehicles=[vehicle])

        def reset(self):
            return self.state()

        def advance_window(self, callback):
            callback([vehicle], flow, self.now + 1.)
            callback([vehicle], flow, self.now + 2.)
            flow[0, 0] += 1.
            self.now += 30.
            return SimpleNamespace(next_state=self.state(), active_vehicle_seconds=30.)

    runner.session_factory = Session
    initial = flow.copy()
    rows, _, _ = runner.run_episode('shortest', seed=43, collect=True)
    assert len(rows) == 4
    assert rows[0]['flow'] is rows[1]['flow']
    assert rows[2]['flow'] is rows[3]['flow']
    assert rows[0]['flow'] is not rows[2]['flow']
    np.testing.assert_array_equal(rows[0]['flow'], initial)
    assert rows[2]['flow'][0, 0] == initial[0, 0] + 1.
