from dataclasses import replace
import copy
import json

import numpy as np
import pytest
import torch

from test_urban_dreamer_training import (make_trainer, _complete_demand_session, _in_transit_seed,
    fixed_demand, mock_oracle_codec_gate_for_isolated_fm)
from urban_dreamer.learning.action_events import pack_action_events
from test_urban_dreamer_config import _write_config
from urban_dreamer.workflow.config import ConfigError, load_config
from urban_dreamer.traffic.demand import demand_identity


def test_method_defaults_and_overrides_are_fingerprinted(tmp_path):
    path = _write_config(tmp_path / 'run.toml')
    first = load_config(path)
    path.write_text(path.read_text() + '\n[method]\nfm_solver="picard"\nfm_steps=3\n')
    second = load_config(path)
    assert second.method.fm_solver == 'picard' and second.method.fm_steps == 3
    assert first.semantic_fingerprint != second.semantic_fingerprint


@pytest.mark.parametrize('setting', ['fm_steps=0', 'fm_solver="bad"',
    'disagreement_samples=1', 'prior_noise_scale=nan', 'evaluation_demand_seed=-1'])
def test_invalid_method_settings_rejected(tmp_path, setting):
    path = _write_config(tmp_path / 'run.toml', extra='\n[method]\n' + setting)
    with pytest.raises(ConfigError):
        load_config(path)


def test_evaluation_demand_and_sumo_seed_do_not_follow_training_counter(tmp_path):
    trainer = make_trainer(tmp_path)
    captured, session_seeds = [], []
    factory = _complete_demand_session(trainer, captured)
    def session(**kwargs):
        session_seeds.append((kwargs['config'].run.seed, kwargs['episode_index']))
        return factory(**kwargs)
    trainer._session_factory = session
    trainer._next_environment_episode = 2
    a = trainer._run_episode(mode='evaluation', episode_index=0)
    trainer._next_environment_episode = 999
    b = trainer._run_episode(mode='evaluation', episode_index=0)
    assert captured[0].vehicles == captured[1].vehicles
    assert a.total_vehicle_count == b.total_vehicle_count
    assert session_seeds == [(trainer.config.method.evaluation_demand_seed, 0)] * 2
    manifest = json.loads((tmp_path / 'evaluation_demands/episode_0000.json').read_text())
    assert manifest['demand_hash'] == demand_identity(captured[0].vehicles)


def test_evaluation_reports_episode_and_vehicle_weighted_means_with_zero_population(tmp_path):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    template = trainer.topology.vehicles[0]
    trainer.demand = fixed_demand(trainer, lambda seed, episode_index:
        tuple(replace(template, vehicle_id=str(i)) for i in range((3, 0, 1)[episode_index])))
    trainer._session_factory = _complete_demand_session(trainer, [])
    for i in range(3):
        trainer.evaluate_episode(i)
    assert trainer.metrics['evaluation_t_all_mean'] == pytest.approx(1. / 3.)
    assert trainer.metrics['evaluation_t_all_vehicle_weighted'] == pytest.approx(.5)
    assert trainer.metrics['evaluation_vehicle_count_sum'] == 4


def test_fm_sample_rng_and_curriculum_restore(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.metrics['world_model_updates'] = 123
    state = copy.deepcopy(trainer.state_dict())
    latent = trainer.world_model.encode(torch.tensor(trainer.network.extract_flow({})).unsqueeze(0))
    action = np.zeros(trainer.edge_count)
    events = pack_action_events([[()]])
    with torch.no_grad():
        expected, risk = trainer._predict_with_disagreement(latent, action, events)
    trainer.load_state_dict(state)
    with torch.no_grad():
        actual, actual_risk = trainer._predict_with_disagreement(latent, action, events)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_risk, risk, rtol=0, atol=0)
    assert trainer.metrics['world_model_updates'] == 123
    assert not trainer.online_replay.active_only
    assert not hasattr(trainer, "warmup_replay")


def test_world_loss_trains_without_reading_replay_context(tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    batch = _in_transit_seed(trainer)
    # Replay still serves policy/value context; the world-model training path
    # must work even when that policy-only field is unavailable.
    world_batch = replace(batch, context=None)
    loss, readout = trainer._world_loss(world_batch)
    assert torch.isfinite(loss) and torch.isfinite(readout)
    (loss + readout).backward()
    assert trainer.world_model.dynamics.flow_head.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in trainer.world_model.flow_projector.parameters())


def test_fm_update_freezes_codec_and_preserves_actor_critics_and_codec_optimizer(tmp_path, monkeypatch):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    batch = _in_transit_seed(trainer)
    codec_before = copy.deepcopy(trainer.world_model.flow_projector.state_dict())
    dynamics_before = copy.deepcopy(trainer.world_model.dynamics.state_dict())
    policy_before = {name: copy.deepcopy(getattr(trainer, name).state_dict())
                     for name in ('actor', 'latent_critic', 'event_critic')}
    assert not trainer.optimizers['codec'].state
    codec_ids = {id(parameter) for parameter in trainer.world_model.flow_projector.parameters()}
    dynamics_ids = {id(parameter) for parameter in trainer.world_model.dynamics.parameters()}
    for key in ('world_model_initial', 'world_model_online'):
        owned = {id(parameter) for group in trainer.optimizers[key].param_groups for parameter in group['params']}
        assert owned == dynamics_ids and not owned.intersection(codec_ids)
    for initial in (True, False):
        metrics = trainer._update_world(SimpleNamespace(has_samples=True), initial=initial, batches=[batch])
        assert all(np.isfinite(value) for value in metrics.values())
    assert not trainer.optimizers['codec'].state
    assert trainer.metrics['world_model_updates'] == 2
    assert any(not torch.equal(value, trainer.world_model.dynamics.state_dict()[name])
               for name, value in dynamics_before.items())
    for name, value in codec_before.items():
        torch.testing.assert_close(trainer.world_model.flow_projector.state_dict()[name], value, rtol=0., atol=0.)
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in trainer.world_model.flow_projector.parameters())
    for model, state in policy_before.items():
        assert all(torch.equal(value, getattr(trainer, model).state_dict()[name]) for name, value in state.items())


def test_unvalidated_codec_blocks_fm_and_online_before_collecting(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = _in_transit_seed(trainer)
    assert not trainer.codec_ready()
    with pytest.raises(ValueError, match='codec quality gate'):
        trainer._world_loss(batch)
    trainer._session_factory = lambda **kwargs: pytest.fail('unvalidated codec must not start a SUMO session')
    with pytest.raises(ValueError, match='codec quality gate'):
        trainer.train_online_episode(0)
    assert trainer._next_environment_episode == 0
    assert not (tmp_path / 'episodes').exists()


def test_initial_world_update_uses_configured_execution_profile_and_online_restores_checkpointing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    batch = _in_transit_seed(trainer)
    settings = dict(vars(trainer.config.resources))
    settings.update(world_model_pretrain_space_chunk=32, world_model_pretrain_time_chunk=8192,
        world_model_pretrain_mlp_chunk=4096, world_model_pretrain_activation_checkpointing=False)
    trainer.config = replace(trainer.config, resources=SimpleNamespace(**settings))
    trainer._update_world(SimpleNamespace(has_samples=True), initial=True, batches=[batch])
    dynamics = trainer.world_model.dynamics
    assert not dynamics.transformer.activation_checkpointing
    for layer in dynamics.transformer.layers:
        assert layer.space.attention_batch_chunk_size == 32
        assert layer.time.attention_batch_chunk_size == 8192
        assert layer.mlp.token_chunk_size == 4096
    trainer._update_world(SimpleNamespace(has_samples=True), initial=False, batches=[batch])
    assert dynamics.transformer.activation_checkpointing


def test_fm_uses_full_observed_inputs_and_all_actions_but_excludes_only_current_window_births(tmp_path, monkeypatch):
    import urban_dreamer.learning.training as module
    from urban_dreamer.learning.action_events import ActionEvent
    trainer = make_trainer(tmp_path)
    mock_oracle_codec_gate_for_isolated_fm(trainer, monkeypatch)
    trainer.config = replace(trainer.config, method=replace(trainer.config.method,
        supervision_horizon=2, fm_training_mode='legacy_independent'))
    trainer.metrics['world_model_updates'] = trainer.config.method.supervision_increment_updates
    batch = _in_transit_seed(trainer)
    old = batch.vehicles[0][0]
    current, nxt, destination = old['current_edge'], old['committed_next_edge'], old['destination_edge']
    newborn = dict(old, vehicle_id='newborn', current_edge=nxt, departure_time=15.)
    later = dict(newborn, vehicle_id='later', departure_time=45.)
    flow = batch.flow[:, :1].expand(-1, 3, -1, -1).clone()
    flow[..., 2] = 0.
    flow[..., current, 2] = 1. / float(trainer.network.capacities[current])
    flow[0, 1, nxt, 2] = 1. / float(trainer.network.capacities[nxt])
    flow[0, 2, nxt, 2] = 2. / float(trainer.network.capacities[nxt])
    windows = [(ActionEvent(old['vehicle_id'], 0, 5., current, nxt, destination),
                ActionEvent('newborn', 1, 15., current, nxt, destination)),
               (ActionEvent('newborn', 2, 35., current, nxt, destination),
                ActionEvent('later', 3, 45., current, nxt, destination))]
    action = batch.action.expand(-1, 2, -1).clone()
    action[..., nxt] = 1.
    batch = replace(batch, flow=flow, action=action,
        boundary_vehicles=(((old,), (old, newborn), (old, newborn, later)),), arrival_times=({},),
        action_events=pack_action_events([windows]), active_vehicle_fraction=torch.tensor([[.75, .875]]))
    seen = {'encoded': []}
    encode = trainer.world_model.encode
    condition = trainer.world_model.dynamics.encode_condition
    mse = module.F.mse_loss
    def capture_encode(value):
        seen['encoded'].append(value.detach().clone())
        return encode(value)
    def capture_condition(value, **kwargs):
        seen['action'] = value.detach().clone()
        seen['events'] = kwargs['action_events']
        return condition(value, **kwargs)
    def capture_readout(prediction, target, **kwargs):
        seen['fraction_target'] = target.detach().clone()
        return mse(prediction, target, **kwargs)
    monkeypatch.setattr(trainer.world_model, 'encode', capture_encode)
    monkeypatch.setattr(trainer.world_model.dynamics, 'encode_condition', capture_condition)
    monkeypatch.setattr(module.F, 'mse_loss', capture_readout)
    loss, readout = trainer._world_loss(batch)
    assert torch.isfinite(loss) and torch.isfinite(readout)
    assert len(seen['encoded']) == 2
    observed, targets = (value.reshape(1, 2, trainer.edge_count, 8) for value in seen['encoded'])
    torch.testing.assert_close(observed, flow[:, :2])
    assert observed[0, 1, nxt, 2] > 0.  # Prior-window newborn is now part of the full input.
    assert targets[0, 0, nxt, 2] == 0.
    assert targets[0, 1, nxt, 2] == pytest.approx(1. / float(trainer.network.capacities[nxt]))
    torch.testing.assert_close(targets[..., [0, 1, 3, 4, 5, 6, 7]], flow[:, 1:, :, [0, 1, 3, 4, 5, 6, 7]])
    assert seen['events'].vehicle_ids == tuple(event.vehicle_id for window in windows for event in window)
    torch.testing.assert_close(seen['action'], action.reshape(2, 1, trainer.edge_count))
    torch.testing.assert_close(seen['fraction_target'], batch.active_vehicle_fraction)
    assert flow[0, 2, nxt, 2] > targets[0, 1, nxt, 2]


def test_old_global_context_checkpoint_fingerprint_is_rejected(tmp_path, monkeypatch):
    import urban_dreamer.workflow.config as config_module
    from test_urban_dreamer_checkpoint import _training_state
    from urban_dreamer.workflow.checkpoint import (
        CheckpointError, CheckpointPhase, load_checkpoint, save_checkpoint)

    path = _write_config(tmp_path / 'config.toml')
    current = load_config(path)
    sha256 = config_module.hashlib.sha256

    def old_conditioning_hash(encoded):
        payload = json.loads(encoded)
        payload['world_model_conditions'] = 'action_context_only_no_future_demand_oracle_v2'
        return sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode())

    # Recreate the previous architecture fingerprint for the same TOML settings.
    with monkeypatch.context() as patch:
        patch.setattr(config_module.hashlib, 'sha256', old_conditioning_hash)
        legacy = load_config(path)
    assert current.semantic_fingerprint != legacy.semantic_fingerprint
    save_checkpoint(output_dir=tmp_path / 'old_run',
        config_fingerprint=legacy.semantic_fingerprint, input_hash='same-road-network',
        phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0,
        training_state=_training_state())
    with pytest.raises(CheckpointError, match='config fingerprint does not match'):
        load_checkpoint(output_dir=tmp_path / 'old_run',
            expected_config_fingerprint=current.semantic_fingerprint,
            expected_input_hash='same-road-network')
