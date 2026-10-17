"""Route control reuses completed pretraining without changing physical/model state."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_urban_dreamer_config import _write_config
from tools import run_online_pilot as pilot
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.world_model import UrbanDreamerWorldModel
from urban_dreamer.workflow.checkpoint import CheckpointPhase, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def configured(tmp_path):
    return load_config(_write_config(tmp_path / 'config.toml', routing_mode='next_hop'))


def route_config(source):
    target = replace(source, simulation=replace(source.simulation, routing_mode='current_to_dest'))
    return replace(target, semantic_fingerprint=_semantic_fingerprint(**{
        name: getattr(target, name) for name in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def test_routing_only_change_preserves_source_and_physics(tmp_path):
    source = configured(tmp_path)
    before = deepcopy(source.resolved_snapshot())
    target = route_config(source)
    target = replace(target, run=replace(target.run, output_dir=tmp_path / 'route', device='cuda:0'))
    pilot.validate_routing_only_change(source, target)
    assert source.resolved_snapshot() == before
    assert target.semantic_fingerprint != source.semantic_fingerprint


@pytest.mark.parametrize('section,field,value', [
    ('simulation', 'vehicle_max_speed_mps', 39.),
    ('simulation', 'traffic_model', 'meso'),
    ('simulation', 'meso_edge_length_m', 120.),
    ('simulation', 'backend', 'libsumo'),
    ('inputs', 'network_name', 'another_network'),
    ('method', 'fm_steps', 99),
    ('resources', 'online_world_model_batch_size', 99),
    ('run', 'seed', 123),
])
def test_routing_transfer_rejects_other_changes(tmp_path, section, field, value):
    source = configured(tmp_path)
    target = route_config(source)
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only simulation.routing_mode'):
        pilot.validate_routing_only_change(source, target)


def test_routing_transfer_rejects_inverse_or_no_change(tmp_path):
    source = configured(tmp_path)
    for before, after in ((source, source), (route_config(source), source)):
        with pytest.raises(ValueError, match='next_hop to current_to_dest'):
            pilot.validate_routing_only_change(before, after)


@pytest.mark.parametrize('source_mode', ['next_hop', 'current_to_dest', None])
def test_entrypoint_without_transfer_keeps_frozen_source_mode(tmp_path, monkeypatch, source_mode):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    config = configured(tmp_path)
    if source_mode == 'current_to_dest':
        config = route_config(config)
    path = source_dir / 'run_config.toml'
    export_run_config(config, path)
    if source_mode is None:
        # Reproduce a historical export written before routing_mode existed.
        path.write_text(path.read_text().replace('routing_mode = "next_hop"\n', ''))
    state = dict(reward_contract=pilot.training.REWARD_CONTRACT,
        models={}, optimizers={}, rng={}, replay={}, next_environment_episode=0,
        metrics={'world_model_initialization_epochs_completed': config.run.world_model_initialization_epochs,
                 'online_episodes_completed': 0})
    save_checkpoint(output_dir=source_dir, config_fingerprint=config.semantic_fingerprint,
        input_hash='network', phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    captured = {}

    class ConfigCaptured(Exception):
        pass

    def capture(runtime):
        captured['config'] = runtime
        raise ConfigCaptured

    monkeypatch.setattr(pilot, 'validate_prepared_inputs', capture)
    with pytest.raises(ConfigCaptured):
        pilot.main(['--source-dir', str(source_dir), '--output-dir', str(tmp_path / 'online'),
                    '--device', 'cpu'])
    assert captured['config'].simulation.routing_mode == (source_mode or 'next_hop')
    assert captured['config'].semantic_fingerprint == config.semantic_fingerprint


@pytest.mark.parametrize('change', ['phase', 'cursor', 'pretrain', 'online', 'replay', 'environment'])
def test_route_initialization_requires_completed_unused_pretraining(tmp_path, change):
    config = configured(tmp_path)
    state = dict(metrics={'world_model_initialization_epochs_completed': config.run.world_model_initialization_epochs,
                          'online_episodes_completed': 0}, next_environment_episode=0,
                 replay={'online': {'episodes': []}})
    checkpoint = SimpleNamespace(phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    if change == 'phase':
        checkpoint.phase = CheckpointPhase.PRETRAIN
    elif change == 'cursor':
        checkpoint.next_unit = 1
    elif change == 'pretrain':
        state['metrics']['world_model_initialization_epochs_completed'] -= 1
    elif change == 'online':
        state['metrics']['online_episodes_completed'] = 1
    elif change == 'replay':
        state['replay']['online']['episodes'].append({'episode': 0})
    else:
        state['next_environment_episode'] = 1
    with pytest.raises(ValueError):
        pilot.validate_pretrained_source(checkpoint, config)


def test_world_model_strict_load_preserves_all_tensors_and_event_transition():
    torch.set_num_threads(1)
    adjacency = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    kwargs = dict(adjacency=adjacency, static_features=torch.zeros(3, 4),
                  low_frequency_rank=1, event_protocol=True)
    source = UrbanDreamerWorldModel(**kwargs).eval()
    target = UrbanDreamerWorldModel(**kwargs).eval()
    target.load_state_dict(source.state_dict(), strict=True)
    checkpoint = SimpleNamespace(training_state={'models': {'world_model': source.state_dict()}})
    proof = pilot.routing_world_model_proof(SimpleNamespace(world_model=target), checkpoint)
    assert proof['world_model_tensors_unchanged']
    assert len(proof['world_model_sha256']) == 64
    events = pack_action_events([[tuple([ActionEvent('vehicle', 0, 3., 0, 1, 2)])]])
    latent, noise = torch.randn(1, 1, 3, 64), torch.randn(1, 1, 1, 64)
    action = torch.tensor([[[0., 1., 0.]]])
    with torch.no_grad():
        expected = source.transition(latent, action, action_events=events, noise=noise, steps=1)
        actual = target.transition(latent, action, action_events=events, noise=noise, steps=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        next(target.parameters()).add_(1.)
    with pytest.raises(ValueError, match='changed world-model tensor'):
        pilot.routing_world_model_proof(SimpleNamespace(world_model=target), checkpoint)


@pytest.mark.parametrize('use_run_config', [False, True])
def test_entrypoint_reuses_pretrained_state_for_routing_only_change(tmp_path, monkeypatch, use_run_config):
    source_dir, output_dir = tmp_path / 'source', tmp_path / 'route'
    source_dir.mkdir()
    config = configured(tmp_path)
    config = replace(config, run=replace(config.run, output_dir=source_dir))
    export_run_config(config, source_dir / 'run_config.toml')
    world_model = torch.nn.Module()
    world_model.flow_projector = torch.nn.Linear(2, 2)
    state = dict(reward_contract=pilot.training.REWARD_CONTRACT,
        models={'world_model': deepcopy(world_model.state_dict())}, optimizers={}, rng={},
        metrics={'world_model_initialization_epochs_completed': config.run.world_model_initialization_epochs,
                 'online_episodes_completed': 0}, next_environment_episode=0,
        replay={'online': {'episodes': []}})
    save_checkpoint(output_dir=source_dir, config_fingerprint=config.semantic_fingerprint,
        input_hash='network', phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    source_hash = pilot.sha256_file(source_dir / 'checkpoints/latest.pt')
    captured = {}

    class Trainer:
        def __init__(self, **kwargs):
            self.config = kwargs['config']
            self.world_model = world_model
            captured['config'] = self.config

        def _tensor(self, value):
            return torch.tensor(value, dtype=torch.float32)

        def load_state_dict(self, restored):
            captured['restored'] = restored
            self.world_model.load_state_dict(restored['models']['world_model'], strict=True)

        def validate_codec_ready(self):
            captured['codec_validated'] = True

        def close(self):
            captured['closed'] = True

    class Runner:
        def __init__(self, trainer, **kwargs):
            captured['protocol'] = kwargs['protocol']

        def run(self, **kwargs):
            return {'status': 'verified_fixture'}

    monkeypatch.setattr(pilot, 'PilotTrainer', Trainer)
    monkeypatch.setattr(pilot, 'OnlinePilot', Runner)
    monkeypatch.setattr(pilot, 'validate_prepared_inputs', lambda config: SimpleNamespace())
    monkeypatch.setattr(pilot, 'load_traffic_topology', lambda prepared:
        SimpleNamespace(network=SimpleNamespace(adjacency=np.eye(2))))
    monkeypatch.setattr(pilot, 'combined_input_hash', lambda prepared: 'network')
    argv = ['--source-dir', str(source_dir), '--output-dir', str(output_dir), '--device', 'cpu',
            '--routing-mode', 'current_to_dest']
    if use_run_config:
        runtime = tmp_path / 'route.toml'
        export_run_config(route_config(config), runtime)
        argv.extend(['--run-config', str(runtime)])
    pilot.main(argv)
    assert captured['config'].simulation.routing_mode == 'current_to_dest'
    assert captured['restored']['metrics'] == state['metrics']
    assert captured['restored']['replay'] == state['replay']
    assert captured['restored']['next_environment_episode'] == 0
    assert captured['codec_validated'] and captured['closed']
    protocol = captured['protocol']
    assert protocol['source']['config_fingerprint'] == config.semantic_fingerprint
    assert protocol['runtime_config_fingerprint'] == captured['config'].semantic_fingerprint
    assert protocol['routing_initialization']['source_checkpoint_sha256'] == source_hash
    assert protocol['routing_initialization']['world_model_tensors_unchanged']
    assert not protocol['config_semantics_unchanged']
    assert pilot.sha256_file(source_dir / 'checkpoints/latest.pt') == source_hash
