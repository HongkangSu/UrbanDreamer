import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import migrate_pretrained_meso as migration
from tools import run_online_pilot as pilot
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, save_checkpoint
from test_online_pilot import FakeTrainer


def configs():
    root = Path(__file__).parents[1]
    target = load_config(root / 'configs/ingolstadt_custom.toml')
    source = replace(target, simulation=replace(target.simulation, traffic_model='micro', backend='traci'))
    return source, target


def test_transfer_accepts_explicit_physics_and_archive_change_but_not_learning_changes(tmp_path):
    source, target = configs()
    migration.validate_transfer_config(source, target)
    for field, changed in (
        ('method', replace(target.method, fm_steps=target.method.fm_steps + 1)),
        ('codec', replace(target.codec, vehicle_mae_max=1.)),
        ('inputs', replace(target.inputs, network_name='another_city')),
        ('simulation', replace(target.simulation, demand_duration_seconds=99.)),
        ('data', replace(target.data, pretrain_cache_manifest=tmp_path / 'micro-cache.json')),
        ('resources', replace(target.resources, world_model_initialization_sequence_length=99)),
    ):
        with pytest.raises(ValueError):
            migration.validate_transfer_config(source, replace(target, **{field: changed}))


def test_weight_transfer_copies_all_models_and_leaves_fresh_optimizers_rng_and_metrics():
    class Trainer:
        def __init__(self):
            for name in migration.MODELS:
                setattr(self, name, torch.nn.Linear(2, 1))
            self.optimizers = {name: torch.optim.Adam(getattr(self, name).parameters()) for name in migration.MODELS}
            self.metrics = {'world_model_initialization_epochs_completed': 0, 'online_episodes_completed': 0}
        def state_dict(self):
            return {'models': {name: getattr(self, name).state_dict() for name in migration.MODELS}}
    source, target = Trainer(), Trainer()
    state = copy.deepcopy(source.state_dict())
    before_rng = torch.get_rng_state().clone()
    migration.transfer_models(target, SimpleNamespace(training_state=state))
    assert migration.model_hashes(target.state_dict()['models']) == migration.model_hashes(state['models'])
    assert target.metrics == {'world_model_initialization_epochs_completed': 0, 'online_episodes_completed': 0}
    assert all(not optimizer.state for optimizer in target.optimizers.values())
    torch.testing.assert_close(torch.get_rng_state(), before_rng)
    state['models']['event_critic']['weight'] = torch.ones(9, 9)
    with pytest.raises(RuntimeError, match='size mismatch'):
        migration.transfer_models(target, SimpleNamespace(training_state=state))


def test_model_hash_covers_scalar_buffers_dtype_and_tensor_contents():
    original = {'world_model': {'counter': torch.tensor(0), 'weights': torch.tensor([1., 2.])}}
    copied = copy.deepcopy(original)
    assert migration.model_hashes(original) == migration.model_hashes(copied)
    copied['world_model']['counter'] += 1
    assert migration.model_hashes(original) != migration.model_hashes(copied)


def test_bounded_decoder_calibration_preserves_encoder_dynamics_policies_and_fresh_optimizer_rng(tmp_path):
    from urban_dreamer.models.world_model import FlowProjector
    class Trainer:
        def __init__(self):
            self.config = configs()[1]
            self.world_model = torch.nn.Module()
            self.world_model.flow_projector = FlowProjector(adjacency=torch.eye(3))
            self.world_model.dynamics = torch.nn.Linear(2, 2)
            self.world_model.active_vehicle_fraction_readout = torch.nn.Linear(2, 1)
            for name in migration.MODELS[1:]:
                setattr(self, name, torch.nn.Linear(2, 1))
            self.optimizers = {'online': torch.optim.Adam(self.world_model.dynamics.parameters())}
            self.metrics = {'codec_ready': 0, 'codec_updates': 0}
            self.offline_data = object()
            self.network = SimpleNamespace(capacities=[10., 20., 30.], speed_limits_mps=[10., 10., 10.])
        def state_dict(self):
            return dict(models={name: getattr(self, name).state_dict() for name in migration.MODELS},
                        rng={'torch_cpu': torch.get_rng_state()}, metrics=self.metrics)
        def load_state_dict(self, state):
            torch.set_rng_state(state['rng']['torch_cpu'])
        def codec_ready(self):
            return bool(self.metrics['codec_ready'])
        def _validate_codec(self):
            self.metrics['codec_ready'] = 1
            return {'codec_ready': True}
        def _codec_batches(self, source, *, shuffle):
            flow = torch.ones(2, 3, 8) * .2
            yield flow, flow
        def _transition_codec_batches(self, source, *, shuffle):
            return iter(())
    torch.set_num_threads(2)
    trainer = Trainer()
    original = copy.deepcopy(trainer.state_dict()['models'])
    rng = torch.get_rng_state().clone()
    result = migration.calibrate_decoder(trainer, max_updates=1, output=tmp_path, source_models=original)
    assert result['updates'] == 1 and trainer.metrics['codec_updates'] == 0
    assert migration.model_hashes(migration.nondecoder_models(original)) == migration.model_hashes(
        migration.nondecoder_models(trainer.state_dict()['models']))
    assert migration.model_hashes(original)['world_model'] != migration.model_hashes(trainer.state_dict()['models'])['world_model']
    assert not trainer.optimizers['online'].state
    torch.testing.assert_close(torch.get_rng_state(), rng)
    with pytest.raises(ValueError, match='bounded'):
        migration.calibrate_decoder(trainer, max_updates=1001, output=tmp_path, source_models=original)


def test_manifest_rejects_source_mutation_and_initialization_checkpoint_mutation(tmp_path):
    source_config, target_config = configs()
    path = tmp_path / 'source.pt'
    path.write_bytes(b'original-source')
    frozen = tmp_path / 'initialization_checkpoint.pt'
    frozen.write_bytes(b'target-initialization')
    source = SimpleNamespace(path=path, training_state={'models': {'world_model': {'x': torch.ones(1)}}})
    manifest = dict(source_checkpoint_sha256=migration.sha256_file(path),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=target_config.semantic_fingerprint,
        input_hash='inputs', target_output=str(tmp_path.resolve()),
        model_hashes=migration.model_hashes(source.training_state['models']),
        target_initialization_sha256=migration.sha256_file(frozen), codec_validation={'codec_ready': True})
    import libsumo
    manifest['target_engine_version'] = libsumo.__version__
    migration.validate_initialization_manifest(manifest, source, source_config, target_config, 'inputs', tmp_path)
    path.write_bytes(b'mutated-source')
    with pytest.raises(ValueError, match='identity differs'):
        migration.validate_initialization_manifest(manifest, source, source_config, target_config, 'inputs', tmp_path)
    path.write_bytes(b'original-source')
    frozen.write_bytes(b'mutated-target')
    with pytest.raises(ValueError, match='checkpoint hash differs'):
        migration.validate_initialization_manifest(manifest, source, source_config, target_config, 'inputs', tmp_path)


def test_target_validation_rejects_nondecoder_edits_reused_optimizers_and_false_meso_epochs():
    models = {'world_model': {'flow_projector.decoder_volume.weight': torch.ones(1), 'dynamics.weight': torch.ones(1)},
              **{name: {'weight': torch.ones(1)} for name in migration.MODELS[1:]}}
    source = SimpleNamespace(training_state={'models': copy.deepcopy(models)})
    state = dict(models=models, optimizers={'online': {'state': {}}}, replay={'online': {'episodes': []}},
                 metrics={'world_model_initialization_epochs_completed': 0}, next_environment_episode=0)
    manifest = dict(model_hashes=migration.model_hashes(models))
    migration.validate_target_initialization(state, source, manifest)
    state['metrics']['world_model_initialization_epochs_completed'] = 5
    with pytest.raises(ValueError, match='fresh target'):
        migration.validate_target_initialization(state, source, manifest)
    state['metrics']['world_model_initialization_epochs_completed'] = 0
    state['optimizers']['online']['state'] = {'old-momentum': 1}
    with pytest.raises(ValueError, match='optimizers must be fresh'):
        migration.validate_target_initialization(state, source, manifest)
    state['optimizers']['online']['state'] = {}
    state['models']['world_model']['dynamics.weight'] += .1
    manifest['model_hashes'] = migration.model_hashes(state['models'])
    with pytest.raises(ValueError, match='frozen nondecoder'):
        migration.validate_target_initialization(state, source, manifest)


def test_prepared_meso_start_creates_fresh_engine_matched_evaluation_progress(tmp_path):
    trainer = FakeTrainer(tmp_path)
    trainer.validate_codec_ready = lambda: None
    checkpoint = save_checkpoint(output_dir=trainer.config.run.output_dir,
        config_fingerprint=trainer.config.semantic_fingerprint, input_hash='network',
        phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=trainer.state_dict())
    manifest = dict(target_initialization_sha256=pilot.sha256_file(checkpoint), target_engine_version='1.27.1')
    runner = pilot.OnlinePilot(trainer, output=trainer.config.run.output_dir, input_hash='network',
        protocol={'source': {'sha256': 'frozen'}, 'meso_initialization': manifest},
        online_episodes=2000, evaluation_every=20)
    runner.resume()
    assert runner.progress['before'] == runner.progress['after'] == runner.progress['online'] == []
    assert runner.progress['evaluation_engine_version'] == '1.27.1'
    assert trainer._next_environment_episode == 0
    assert runner.protocol['online_episode_budget'] == 2000
    assert runner.protocol['evaluation_every'] == 20
