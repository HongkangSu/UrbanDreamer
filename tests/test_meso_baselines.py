"""All baseline entry points must collect and imagine the configured traffic model."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from urban_dreamer.learning.baselines import build_method, common
from urban_dreamer.learning.baselines.dreamer import DreamerMethod
from urban_dreamer.learning.baselines.mbpo import MbpoMethod
from urban_dreamer.learning.episode_archive import read_episode_archive
from urban_dreamer.learning.replay import MacroTransition
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.simulation_settings import physics_identity

ROOT = Path(__file__).resolve().parents[1]


def tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def meso_config(tmp_path):
    text = (ROOT / 'configs/toy_cross_bottleneck.toml').read_text()
    text = text.replace('network_dir = "../networks/toy_cross_bottleneck"',
                        f'network_dir = "{ROOT / "networks/toy_cross_bottleneck"}"')
    # Make the fixture independent of whether shipped configs have been migrated yet.
    simulation_lines = text.split('[simulation]\n', 1)
    body = simulation_lines[1]
    for key in ('traffic_model', 'meso_junction_control', 'meso_lane_queue', 'meso_multi_queue',
                'meso_edge_length_m', 'meso_interpolate_positions'):
        body = '\n'.join(line for line in body.split('\n') if not line.startswith(key + ' ='))
    body = body.replace('backend = "traci"', 'backend = "libsumo"')
    text = simulation_lines[0] + '[simulation]\ntraffic_model = "meso"\n' + body
    path = tmp_path / 'meso.toml'
    path.write_text(text)
    return path


def arguments(config, tmp_path, **overrides):
    values = dict(config=config, output_dir=tmp_path, device='cpu', seed=42, threads=1,
        vehicle_limit=2, method='gnac', actor_lr=2e-4, critic_lr=3e-4,
        global_critic_lr=3e-4, session=None, backend=None, step_seconds=None)
    values.update(overrides)
    return argparse.Namespace(**values)


def test_default_runner_resolves_event_libsumo_and_meso_candidates(meso_config, tmp_path):
    runner = common.EpisodeRunner(arguments(meso_config, tmp_path))
    assert (runner.args.session, runner.args.backend, runner.args.step_seconds) == ('event', 'libsumo', 1.)
    assert runner.session_factory.func is SumoEventSession
    assert runner.candidates.traffic_model == 'meso'
    assert runner.simulation_settings['meso_edge_length_m'] == 98.
    for source, target in runner.network.connection_lengths_m:
        assert runner.candidates.junction(source, target)[0] == 0.
    method = build_method('dta', runner.args, runner)
    for source, target in runner.network.connection_lengths_m:
        assert method._junction_seconds(source, target) == 0.


@pytest.mark.parametrize('override', [dict(session='microstep'), dict(backend='traci'), dict(step_seconds=.5)])
def test_meso_rejects_session_overrides(meso_config, tmp_path, override):
    with pytest.raises(ValueError, match='meso traffic requires'):
        common.EpisodeRunner(arguments(meso_config, tmp_path, **override))


@pytest.mark.parametrize('name', ['train_toy_baseline', 'train_toy_actor_critic', 'collect_random_episodes'])
def test_cli_defaults_use_config_and_reject_microstep(meso_config, tmp_path, monkeypatch, capsys, name):
    module = tool(name)
    seen = {}

    class CollectedSettings(Exception):
        pass

    def capture(args, **kwargs):
        seen.update(vars(args))
        raise CollectedSettings

    if name == 'train_toy_baseline':
        monkeypatch.setattr(module.common, 'EpisodeRunner', capture)
    else:
        monkeypatch.setattr(module, 'RealActorCritic' if name == 'train_toy_actor_critic' else 'EpisodeRunner', capture)
    argv = ['--config', str(meso_config), '--output-dir', str(tmp_path / name)]
    if name == 'train_toy_baseline':
        argv += ['--method', 'gnac']
    with pytest.raises(CollectedSettings):
        module.main(argv)
    assert (seen['session'], seen['backend'], seen['step_seconds']) == ('event', 'libsumo', 1.)
    with pytest.raises(SystemExit) as error:
        module.main(argv + ['--session', 'microstep'])
    assert error.value.code == 2
    assert 'meso traffic requires session' in capsys.readouterr().err


@pytest.mark.parametrize('name', ['dreamer', 'mbpo'])
@pytest.mark.parametrize('traffic_model', ['micro', 'meso'])
def test_model_based_rollouts_receive_configured_physics(meso_config, tmp_path, monkeypatch, name, traffic_model):
    config_path = meso_config
    if traffic_model == 'micro':
        config_path = tmp_path / 'micro.toml'
        config_path.write_text(meso_config.read_text().replace('traffic_model = "meso"', 'traffic_model = "micro"'))
    runner = common.EpisodeRunner(arguments(config_path, tmp_path, method=name))
    method = build_method(name, runner.args, runner)
    assert method.traffic_model == traffic_model
    if traffic_model == 'meso':
        assert method.microstep_seconds == 1.
    demand = runner.topology.vehicles[0]
    current = runner.network.edge_to_index[runner.topology.origins[demand.origin_index]]
    destination = runner.network.edge_to_index[runner.topology.destinations[demand.destination_index]]
    state = dict(vehicle_id=demand.vehicle_id, departed=True, arrived=False, current_edge=current,
        destination_edge=destination, remaining_distance=float(runner.network.lengths_m[current]),
        departure_time=0., pending_decision=True, committed_next_edge=None,
        in_junction=False, junction_remaining_distance=0., junction_speed=1.)
    flow = runner.network.extract_flow({})
    timestamp = runner.deadline - 30.
    captured = []

    def advance(states, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(action=np.zeros(runner.network.edge_count),
            departure=np.zeros(runner.network.edge_count))

    implementation = __import__(f'urban_dreamer.learning.baselines.{name}', fromlist=['advance_imagined_window'])
    monkeypatch.setattr(implementation, 'advance_imagined_window', advance)
    if name == 'dreamer':
        transition = MacroTransition(flow=flow, next_flow=flow, action=np.zeros(runner.network.edge_count),
            departure=np.zeros(runner.network.edge_count), context=np.array([timestamp / runner.deadline, 1.]),
            active_vehicle_fraction=1., vehicles=(state,), time=timestamp,
            terminal=False, episode=0, step_index=0)
        method.seed_buffer.append_episode([transition], total_vehicle_count=1, total_travel_minutes=1.)
        monkeypatch.setattr(method, '_decode_flow', lambda h, z: flow.copy())
        method._imagine_rollout(0)
    else:
        method._rollout_from_seed(dict(flow=flow, vehicles=(state,), time=timestamp, total_vehicle_count=1), 0)
    assert captured
    assert captured[0]['traffic_model'] == traffic_model
    assert captured[0]['session_protocol'] == ('event' if traffic_model == 'meso' else 'legacy')
    assert captured[0]['observation_step_seconds'] == method.microstep_seconds
    assert captured[0]['candidates'].traffic_model == traffic_model


def test_checkpoint_rejects_changed_physics_even_with_same_fingerprint(meso_config, tmp_path):
    runner = common.EpisodeRunner(arguments(meso_config, tmp_path))
    checkpoint = tmp_path / 'latest.pt'
    common.save_checkpoint(checkpoint, method_name='gnac', episode=1, best=1., method_state={},
        args=runner.args, prepared=runner.prepared, config=runner.config)
    state = torch.load(checkpoint, weights_only=False)
    assert state['physics'] == physics_identity(runner.config.simulation)
    state['physics']['meso_edge_length_m'] = 99.
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        common.load_checkpoint(checkpoint, method_name='gnac', prepared=runner.prepared, config=runner.config)


@pytest.mark.parametrize('name', ['train_toy_baseline', 'train_toy_actor_critic'])
def test_baseline_cli_records_config_defaults_and_checks_resume_physics(meso_config, tmp_path, name):
    module = tool(name)
    output = tmp_path / name
    argv = ['--config', str(meso_config), '--output-dir', str(output), '--episodes', '0',
        '--vehicle-limit', '2', '--skip-initial-evaluation', '--skip-reference-baselines']
    if name == 'train_toy_baseline':
        argv += ['--method', 'gnac']
    assert module.main(argv) == 0
    protocol = json.loads((output / 'protocol.json').read_text())
    assert (protocol['session'], protocol['backend'], protocol['step_seconds']) == ('event', 'libsumo', 1.)
    assert protocol['traffic_model'] == 'meso' and protocol['meso_junction_control'] is True
    assert module.main(argv + ['--resume']) == 0
    checkpoint = output / 'latest.pt'
    state = torch.load(checkpoint, weights_only=False)
    state['physics']['meso_edge_length_m'] = 99.
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='checkpoint.*mismatch'):
        module.main(argv + ['--resume'])


def test_random_collector_archives_real_meso_and_resumes(meso_config, tmp_path):
    pytest.importorskip('libsumo')
    module = tool('collect_random_episodes')
    output = tmp_path / 'collection'
    argv = ['--config', str(meso_config), '--output-dir', str(output), '--episodes', '1', '--vehicle-limit', '2']
    module.main(argv)
    archive = read_episode_archive(output / 'episodes/train/episode_000001.npz')
    protocol = json.loads((output / 'protocol.json').read_text())
    assert archive['metadata']['traffic_model'] == protocol['traffic_model'] == 'meso'
    assert archive['metadata']['session_protocol'] == 'event'
    assert archive['metadata']['backend'] == 'libsumo'
    assert archive['metadata']['meso_interpolate_positions'] is True
    assert archive['metadata']['junction_state'] == 'zero_internal_distance'
    module.main(argv + ['--resume'])
    protocol['meso_edge_length_m'] = 99.
    (output / 'protocol.json').write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match='resume protocol mismatch'):
        module.main(argv + ['--resume'])


def test_explicit_legacy_micro_session_still_uses_micro_model(meso_config, tmp_path):
    path = tmp_path / 'micro.toml'
    path.write_text(meso_config.read_text().replace('traffic_model = "meso"', 'traffic_model = "micro"'))
    runner = common.EpisodeRunner(arguments(path, tmp_path, session='microstep', backend='traci', step_seconds=.5))
    assert runner.session_factory is None
    assert runner.candidates.traffic_model == 'micro'
    assert runner.simulation_settings['traffic_model'] == 'micro'


@pytest.mark.parametrize('method_type', [DreamerMethod, MbpoMethod])
@pytest.mark.parametrize('model,minimum', [('micro', .5), ('meso', 0.)])
def test_model_based_motion_preserves_stops_and_vehicle_speed_cap(method_type, model, minimum):
    method = method_type.__new__(method_type)
    method.traffic_model = model
    method.speed_limits = np.full(4, 10.)
    method.vehicle_max_speed = 12.
    flow = np.zeros((4, 8), dtype=np.float32)
    flow[:, 1] = [-.1, 0., .5, 2.]
    np.testing.assert_allclose(method._speeds(flow), [minimum, minimum, 5., 12.])


@pytest.mark.parametrize('model,minimum', [('micro', .05), ('meso', 0.)])
def test_mbpo_predicted_road_speeds_keep_zero_only_for_meso(model, minimum, monkeypatch):
    method = MbpoMethod.__new__(MbpoMethod)
    method.traffic_model = model
    method.edge_count = 4
    method.device = torch.device('cpu')
    method.rng = np.random.default_rng(0)

    def zero_delta(inputs):
        delta = inputs.new_zeros(len(inputs), 4 * 8)
        return delta, delta, inputs.new_tensor([-.25])

    method.ensemble = [zero_delta]
    monkeypatch.setattr(torch, 'randn_like', torch.zeros_like)
    flow = np.zeros((4, 8), dtype=np.float32)
    flow[:, 1] = [-.1, 0., .5, 2.]
    predicted, reward = method._predict_transition(flow, np.zeros(4), np.zeros(4), np.zeros(2))
    np.testing.assert_allclose(predicted[:, 1], [minimum, minimum, .5, 1.5])
    assert reward == -.25
