from dataclasses import replace
import copy
import json

import numpy as np
import pytest
import torch

from test_urban_dreamer_training import make_trainer, _complete_demand_session, _in_transit_seed
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
    trainer.demand = SimpleNamespace(sample=lambda seed, episode_index:
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
    with torch.no_grad():
        expected, risk = trainer._predict_with_disagreement(latent, action)
    trainer.load_state_dict(state)
    with torch.no_grad():
        actual, actual_risk = trainer._predict_with_disagreement(latent, action)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_risk, risk, rtol=0, atol=0)
    assert trainer.metrics['world_model_updates'] == 123
    assert not trainer.warmup_replay.active_only and not trainer.online_replay.active_only


def test_world_loss_trains_without_reading_replay_context(tmp_path):
    from dataclasses import fields
    from types import SimpleNamespace
    trainer = make_trainer(tmp_path)
    batch = _in_transit_seed(trainer)
    # Replay still serves policy/value context; the world-model training path
    # must work even when that policy-only field is unavailable.
    world_batch = SimpleNamespace(**{field.name: getattr(batch, field.name)
        for field in fields(batch) if field.name != 'context'})
    loss, readout = trainer._world_loss(world_batch)
    assert torch.isfinite(loss) and torch.isfinite(readout)
    (loss + readout).backward()
    assert trainer.world_model.dynamics.flow_head.weight.grad.abs().sum() > 0


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
