"""Decoder calibration must not change the learned latent coordinate system."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from urban_dreamer.learning import codec_training, training
from urban_dreamer.models.world_model import FlowProjector
from urban_dreamer.workflow import config as cfg
from test_codec_training import ConstantCodec, network
from test_urban_dreamer_training import make_trainer, write_empty_offline_archive, attach_offline_data
from test_urban_dreamer_config import _write_config


def joint_config():
    return cfg.CodecConfig(decoder_pretrain_enabled=True, occupied_speed_weight=1., slow_speed_weight=1.)


def test_decoder_supervision_freezes_encoder_and_detaches_all_labels():
    model = FlowProjector(adjacency=torch.eye(2), positive_speed=True)
    flow = torch.zeros(2, 2, 8)
    flow[..., 1] = .05
    flow[:, 0, 2] = .6
    flow.requires_grad_()
    following = flow.detach().clone().requires_grad_()
    before = copy.deepcopy(model.state_dict())
    optimizer = codec_training.make_decoder_optimizer(model)
    loss, metrics = codec_training.codec_objective(model, flow, following, network(),
        codec=joint_config(), decoder_only=True)
    loss.backward()
    decoder_ids = {id(p) for p in codec_training.decoder_parameters(model)}
    assert all(p.grad is None for p in model.parameters() if id(p) not in decoder_ids)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec_training.decoder_parameters(model))
    assert flow.grad is None and following.grad is None
    optimizer.step()
    for name, value in model.state_dict().items():
        if not name.startswith('decoder_'):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items() if name.startswith('decoder_'))
    assert metrics['codec_slow_roads'] == 2


def test_speed_groups_are_normalized_and_empty_masks_are_finite():
    model = ConstantCodec([0., .7, .1, 0.])
    flow = torch.zeros(1, 2, 8)
    flow[..., 1] = .05
    flow[:, 0, 2] = .6
    _, base = codec_training.codec_objective(model, flow, flow, network())
    _, weighted = codec_training.codec_objective(model, flow, flow, network(), codec=joint_config())
    assert weighted['codec_speed_loss'] == 3 * base['codec_speed_loss']
    flow[..., 2] = 0.
    loss, result = codec_training.codec_objective(model, flow, flow, network(), codec=joint_config())
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.value.grad).all()
    assert result['codec_speed_loss_occupied'] == result['codec_speed_loss_slow'] == 0.


def test_legacy_fingerprint_and_optimizer_schema_are_preserved(tmp_path, monkeypatch):
    config = cfg.load_config(_write_config(tmp_path / 'legacy.toml'))
    original = cfg.asdict
    added = set(cfg.CodecConfig.__dataclass_fields__) - {'learning_rate', 'batch_size', 'max_updates',
        'validation_every', 'occupied_vehicle_mae_max', 'vehicle_mae_max', 'empty_vehicle_mae_max',
        'total_relative_mae_max', 'speed_mae_mps_max', 'speed_positive'}
    def old_asdict(value):
        result = original(value)
        if isinstance(value, cfg.CodecConfig):
            for name in added:
                result.pop(name)
        return result
    monkeypatch.setattr(cfg, 'asdict', old_asdict)
    old = cfg._semantic_fingerprint(**{name: getattr(config, name) for name in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert old == config.semantic_fingerprint
    monkeypatch.setattr(cfg, 'asdict', original)
    trainer = make_trainer(tmp_path / 'off')
    assert 'decoder_pretrain' not in trainer.optimizers


def prepared_trainer(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, codec=joint_config())
    trainer.optimizers['decoder_pretrain'] = codec_training.make_decoder_optimizer(trainer.world_model)
    trainer.metrics['decoder_pretrain_updates'] = 0.
    flow = torch.tensor(trainer.network.extract_flow({})).unsqueeze(0)
    flow[..., 1] = .01
    flow[0, 0, 2] = 6 / float(trainer.network.capacities[0])
    perfect = codec_training.codec_decoded_metrics(flow[..., :4], flow, flow, trainer.network, trainer.config.codec)
    validation = SimpleNamespace(has_samples=True, manifest={'identity': 'independent-test-demand'}, release_cache=lambda: None)
    validation.state_dict = lambda: {'manifest_identity': validation.manifest['identity']}
    validation.load_state_dict = lambda state: None
    trainer.codec_validation_data = validation
    monkeypatch.setattr(trainer, '_codec_batches', lambda *a, **kw: iter([(flow, flow)]))
    monkeypatch.setattr(trainer, '_transition_codec_batches', lambda *a, **kw: iter([(flow, flow)]))
    monkeypatch.setattr(training, 'codec_metrics', lambda *a, **kw: perfect)
    trainer._validate_codec()
    trainer.world_model.flow_projector.requires_grad_(False)
    from urban_dreamer.learning.replay import ReplayBuffer, MacroTransition
    trainer.offline_data = ReplayBuffer(sequence_length=1, active_only=False)
    vehicles = tuple(dict(vehicle_id=str(i), current_edge=0, departed=True, arrived=False,
                          in_junction=False) for i in range(6))
    rows = [MacroTransition(flow=flow[0], next_flow=flow[0],
        action=torch.zeros(trainer.edge_count), departure=torch.zeros(trainer.edge_count),
        context=torch.tensor([0., 1.]), active_vehicle_fraction=1.,
        time=30. * i, step_index=i, terminal=i == 2, action_events=(), vehicles=vehicles,
        next_vehicles=vehicles) for i in range(3)]
    trainer.offline_data.append_episode(rows, total_travel_minutes=1., total_vehicle_count=6)
    return trainer


def test_actual_pretrain_steps_update_decoder_independently_and_invalidate_proof(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    trainer = prepared_trainer(tmp_path, monkeypatch)
    before = {name: value.detach().clone() for name, value in trainer.world_model.flow_projector.named_parameters()}
    dynamics_before = copy.deepcopy(trainer.world_model.dynamics.state_dict())
    result = trainer._update_world(trainer.offline_data, initial=True)
    assert trainer.metrics['decoder_pretrain_updates'] == 1
    assert result['decoder_pretrain_codec_loss'] > 0
    assert not trainer.codec_ready()
    with pytest.raises(ValueError, match='quality gate'):
        trainer.validate_codec_ready()
    after = dict(trainer.world_model.flow_projector.named_parameters())
    assert any(not torch.equal(after[name], value) for name, value in before.items() if name.startswith('decoder_'))
    for name, value in before.items():
        if not name.startswith('decoder_'):
            torch.testing.assert_close(after[name], value, rtol=0, atol=0)
            assert after[name].grad is None
    assert any(not torch.equal(value, dynamics_before[name]) for name, value in trainer.world_model.dynamics.state_dict().items())
    trainer._update_world(trainer.offline_data, initial=True)
    assert trainer.metrics['decoder_pretrain_updates'] == 2
    state = copy.deepcopy(trainer.state_dict())
    assert state['optimizers']['decoder_pretrain']['state']
    trainer.load_state_dict(copy.deepcopy(state))
    assert not trainer.codec_ready()
    decoder_ids = {id(p) for p in codec_training.decoder_parameters(trainer.world_model)}
    assert all(p.requires_grad == (id(p) in decoder_ids) for p in trainer.world_model.flow_projector.parameters())
    trainer._validate_codec()
    assert trainer.codec_ready()
    before = copy.deepcopy(trainer.world_model.flow_projector.state_dict())
    trainer._update_world(trainer.offline_data, initial=False)
    assert trainer.metrics['decoder_pretrain_updates'] == 2
    for name, value in trainer.world_model.flow_projector.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_new_optimizer_schema_refuses_silent_old_resume(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    state = copy.deepcopy(trainer.state_dict())
    del state['optimizers']['decoder_pretrain']
    with pytest.raises(ValueError, match='optimizer schema'):
        trainer.load_state_dict(copy.deepcopy(state))


def test_foundation_proof_cannot_authorize_changed_source_or_codec(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    trainer.validate_codec_pretraining_ready()
    trainer.metrics['decoder_pretrain_updates'] += 1
    trainer.validate_codec_pretraining_ready()
    trainer.codec_validation_data.manifest['identity'] = 'new-demand'
    with pytest.raises(ValueError, match='initial held-out'):
        trainer.validate_codec_pretraining_ready()


def test_decoder_hybrid_uses_only_available_following_frames(tmp_path, monkeypatch):
    from test_cohort_targets import transition_batch
    trainer = SimpleNamespace(device=torch.device('cpu'), edge_count=3,
        network=SimpleNamespace(capacities=[10., 20., 30.]), world_model=object(),
        config=SimpleNamespace(method=cfg.MethodConfig(), codec=joint_config()))
    batch = transition_batch()
    captured = []
    def objective(model, flow, following, network, method, **kwargs):
        captured.append((flow, following, kwargs))
        return torch.tensor(0.), {}
    monkeypatch.setattr(training, 'codec_objective', objective)
    training.MainlineTrainer._decoder_pretrain_loss(trainer, batch)
    raw_rows = batch.flow.shape[0] * batch.action.shape[1]
    assert len(captured[0][0]) == raw_rows + batch.flow.shape[0] * (batch.action.shape[1] - 1)
    batch = replace(batch, decoder_following_flow=torch.full_like(batch.flow[:, 1:], .77),
                    decoder_following_valid=torch.ones(batch.flow.shape[0], batch.action.shape[1], dtype=torch.bool))
    training.MainlineTrainer._decoder_pretrain_loss(trainer, batch)
    assert len(captured[1][0]) == 2 * raw_rows + batch.flow.shape[0]
    assert torch.all(captured[1][1][-batch.flow.shape[0]:] == .77)
    assert captured[1][2]['decoder_only']


def test_fm_readout_decoder_graphs_have_disjoint_parameter_gradients(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    trainer = prepared_trainer(tmp_path, monkeypatch)
    for parameter in codec_training.decoder_parameters(trainer.world_model):
        parameter.requires_grad_(True)
    batch = trainer.offline_data.sample_batch(batch_size=1, rng=trainer.rng)
    fm_loss, readout_loss = trainer._world_loss(batch)
    fm_loss.backward()
    assert any(p.grad is not None for p in trainer.world_model.dynamics.parameters())
    assert all(p.grad is None for p in codec_training.decoder_parameters(trainer.world_model))
    assert all(p.grad is None for p in trainer.world_model.active_vehicle_fraction_readout.parameters())
    trainer.optimizers['world_model_initial'].zero_grad(set_to_none=True)
    readout_loss.backward()
    assert all(p.grad is None for p in trainer.world_model.dynamics.parameters())
    assert all(p.grad is None for p in codec_training.decoder_parameters(trainer.world_model))
    assert any(p.grad is not None for p in trainer.world_model.active_vehicle_fraction_readout.parameters())
    trainer.optimizers['active_fraction'].zero_grad(set_to_none=True)
    decoder_loss, _ = trainer._decoder_pretrain_loss(batch)
    decoder_loss.backward()
    assert all(p.grad is None for p in trainer.world_model.dynamics.parameters())
    assert all(p.grad is None for p in trainer.world_model.active_vehicle_fraction_readout.parameters())
    decoder_ids = {id(p) for p in codec_training.decoder_parameters(trainer.world_model)}
    assert all(p.grad is None for p in trainer.world_model.flow_projector.parameters() if id(p) not in decoder_ids)
    assert any(p.grad is not None for p in codec_training.decoder_parameters(trainer.world_model))


def test_slow_gate_uses_mean_label_and_ignores_future_birth_counts():
    flow = torch.zeros(1, 2, 8)
    flow[0, 0, 2] = .5
    flow[0, 0, 1] = .1306
    future = flow.clone()
    future[0, 0, 1] = .2093
    occupied, slow = codec_training.codec_speed_masks(flow, future, network(), joint_config())
    assert occupied[0, 0] and slow[0, 0]
    future[..., 2] = 1000.
    _, changed = codec_training.codec_speed_masks(flow, future, network(), joint_config())
    assert torch.equal(slow, changed)


def test_missing_slow_coverage_is_reported_and_strict_gate_rejects(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    original = training.codec_metrics(trainer.world_model, None, None, None)
    missing = dict(original, codec_slow_roads=torch.tensor(0), codec_speed_mae_slow_mps=torch.tensor(0.))
    monkeypatch.setattr(training, 'codec_metrics', lambda *a, **kw: missing)
    result = trainer._validate_codec()
    assert result['codec_speed_mae_slow_mps'] is None
    assert result['codec_slow_roads'] == 0
    assert trainer.codec_ready()  # Toy allows an explicitly unavailable slow group.
    trainer.config = replace(trainer.config, codec=replace(trainer.config.codec, slow_validation_min_roads=1))
    assert not trainer.codec_ready()
    assert not trainer._validate_codec()['codec_ready']


def test_presets_explicitly_enable_supervision_and_stricter_slow_gate():
    from pathlib import Path
    for path in (Path(__file__).resolve().parents[1] / 'configs').glob('*.toml'):
        config = cfg.load_config(path)
        assert config.codec.decoder_pretrain_enabled
        assert config.codec.decoder_online_enabled
        assert config.codec.decoder_online_learning_rate == 1e-5
        assert config.codec.occupied_speed_weight == config.codec.slow_speed_weight == 1.
        assert config.codec.slow_speed_mae_mps_max == .5
        assert config.codec.slow_validation_min_roads == (0 if path.name.startswith('toy_') else 1)


def test_decoder_failure_cannot_commit_other_optimizers(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    parameter = codec_training.decoder_parameters(trainer.world_model)[0]
    monkeypatch.setattr(trainer, '_decoder_pretrain_loss', lambda batch: (parameter.sum() * float('nan'), {}))
    before = copy.deepcopy(trainer.world_model.state_dict())
    with pytest.raises(FloatingPointError, match='nonfinite decoder'):
        trainer._update_world(trainer.offline_data, initial=True)
    assert trainer.metrics['world_model_updates'] == trainer.metrics['decoder_pretrain_updates'] == 0
    for name, value in trainer.world_model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not trainer.optimizers['world_model_initial'].state
    assert not trainer.optimizers['active_fraction'].state


def test_epoch_boundary_validates_current_decoder_revision(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    trainer.offline_data.release_cache = lambda: None
    result = trainer.initialize_world_model_epoch(0)
    assert result['samples'] == 3
    assert trainer.metrics['decoder_pretrain_updates'] == 3
    assert trainer.metrics['codec_validation_decoder_updates'] == 3
    assert trainer.codec_ready()


def test_online_decoder_updates_preserve_encoder_and_resume_optimizer(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    trainer = prepared_trainer(tmp_path, monkeypatch)
    trainer.config = replace(trainer.config, codec=replace(trainer.config.codec,
        decoder_online_enabled=True, decoder_allow_unready=True))
    trainer.optimizers['decoder_online'] = codec_training.make_decoder_optimizer(trainer.world_model, lr=1e-5)
    trainer.metrics['decoder_online_updates'] = 0.
    before = copy.deepcopy(trainer.world_model.flow_projector.state_dict())
    trainer.metrics['codec_ready'] = 0.
    for _ in range(2):
        result = trainer._update_world(trainer.offline_data, initial=False)
        assert result['decoder_online_codec_loss'] > 0
        assert result['decoder_online_gradient_norm'] > 0
        assert result['decoder_online_updated'] == 1
        assert not trainer.codec_ready()
        trainer.validate_codec_ready()
    assert trainer.metrics['decoder_online_updates'] == 2
    assert trainer.metrics['decoder_pretrain_updates'] == 0
    after = trainer.world_model.flow_projector.state_dict()
    assert any(not torch.equal(value, before[name]) for name, value in after.items() if name.startswith('decoder_'))
    for name, value in trainer.world_model.flow_projector.named_parameters():
        if name.startswith('decoder_'):
            assert value.requires_grad
        else:
            assert not value.requires_grad and value.grad is None
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    state = copy.deepcopy(trainer.state_dict())
    assert state['optimizers']['decoder_online']['state']
    trainer.load_state_dict(copy.deepcopy(state))
    assert not trainer.codec_ready()
    assert all(p.requires_grad for p in codec_training.decoder_parameters(trainer.world_model))
    # Exact continuation must include the decoder's Adam moments and RNG.
    trainer._update_world(trainer.offline_data, initial=False)
    expected = copy.deepcopy(trainer.world_model.flow_projector.state_dict())
    trainer.load_state_dict(copy.deepcopy(state))
    trainer._update_world(trainer.offline_data, initial=False)
    for name, value in trainer.world_model.flow_projector.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)


def test_unready_override_requires_validation_source_and_finite_codec(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    trainer.config = replace(trainer.config, codec=replace(trainer.config.codec, decoder_allow_unready=True))
    trainer.metrics['codec_ready'] = 0.
    trainer.validate_codec_ready()
    assert not trainer.codec_ready()
    source = trainer.codec_validation_data
    trainer.codec_validation_data = None
    with pytest.raises(ValueError, match='independent held-out'):
        trainer.validate_codec_ready()
    trainer.codec_validation_data = source
    with torch.no_grad():
        codec_training.decoder_parameters(trainer.world_model)[0].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite codec'):
        trainer.validate_codec_ready()


def test_passed_foundation_admits_continuous_online_without_override(tmp_path, monkeypatch):
    trainer = prepared_trainer(tmp_path, monkeypatch)
    trainer.config = replace(trainer.config, codec=replace(trainer.config.codec, decoder_online_enabled=True))
    trainer.optimizers['decoder_online'] = codec_training.make_decoder_optimizer(trainer.world_model, lr=1e-5)
    assert trainer.codec_ready()
    for _ in range(2):
        trainer._update_world(trainer.offline_data, initial=False)
        assert not trainer.codec_ready()
        trainer.validate_codec_ready()
    trainer.load_state_dict(copy.deepcopy(trainer.state_dict()))
    trainer.validate_codec_ready()
    assert all(p.requires_grad for p in codec_training.decoder_parameters(trainer.world_model))
    trainer.codec_validation_data.manifest['identity'] = 'different-demand'
    with pytest.raises(ValueError, match='quality gate'):
        trainer.validate_codec_ready()


def test_episode_metrics_preserve_cumulative_decoder_revision(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    trainer = prepared_trainer(tmp_path, monkeypatch)
    trainer.config = replace(trainer.config, codec=replace(trainer.config.codec, decoder_online_enabled=True))
    trainer.optimizers['decoder_online'] = codec_training.make_decoder_optimizer(trainer.world_model, lr=1e-5)
    collected = SimpleNamespace(transitions=trainer.offline_data.episodes[0],
        total_travel_minutes=1., total_vehicle_count=6, arrival_times={}, metrics={})
    monkeypatch.setattr(trainer, '_run_episode', lambda **kwargs: collected)
    monkeypatch.setattr(trainer, '_train_episode_imagination', lambda: {})
    for episode in range(2):
        result = trainer.train_online_episode(episode)
        assert result['decoder_online_updated'] == 1.
        assert 'decoder_online_updates' not in result
        assert trainer.metrics['decoder_online_updates'] == 4 * (episode + 1)
        assert trainer._decoder_revision() == 4 * (episode + 1)
        assert not trainer.codec_ready()
        trainer.validate_codec_ready()
