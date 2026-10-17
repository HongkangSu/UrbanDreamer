"""Recursive presets retain explicit single-step and historical compatibility."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_urban_dreamer_config import _write_config
from test_cohort_targets import transition_batch, cohort_replay
from urban_dreamer.workflow import config as cfg
from urban_dreamer.learning import training
from urban_dreamer.learning.cohort_targets import prepare_transition_targets, prepare_cohort_batch


def configured(tmp_path, mode=None, sequence=3):
    extra = '' if mode is None else '\n[method]\nfm_training_mode = ' + mode + '\n'
    path = _write_config(tmp_path / 'run.toml', extra=extra)
    import re
    path.write_text(re.sub(r'world_model_initialization_sequence_length = \d+',
                          f'world_model_initialization_sequence_length = {sequence}', path.read_text()))
    return cfg.load_config(path)


def test_omitted_mode_retains_exact_old_fingerprint(tmp_path, monkeypatch):
    config = configured(tmp_path)
    original = cfg.asdict
    def old_asdict(value):
        result = original(value)
        if isinstance(value, cfg.MethodConfig):
            result.pop('fm_training_mode')
        return result
    monkeypatch.setattr(cfg, 'asdict', old_asdict)
    old_hash = cfg._semantic_fingerprint(**{name: getattr(config, name) for name in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert config.semantic_fingerprint == old_hash


def test_modes_bind_different_semantics(tmp_path):
    values = [configured(tmp_path, '"' + mode + '"').semantic_fingerprint for mode in
              ('legacy_independent', 'single_step', 'recursive_3step')]
    assert len(set(values)) == 3


@pytest.mark.parametrize('mode', ['true', '1', '"three_steps"'])
def test_invalid_mode_rejected(tmp_path, mode):
    with pytest.raises(cfg.ConfigError, match='fm_training_mode'):
        configured(tmp_path, mode)


def test_recursive_requires_three_frame_intervals(tmp_path):
    with pytest.raises(cfg.ConfigError, match='sequence_length.*>= 3'):
        configured(tmp_path, '"recursive_3step"', sequence=1)


def test_presets_enable_recursive_supervision_and_latent_risk():
    paths = list((Path(__file__).resolve().parents[1] / 'configs').glob('*.toml'))
    assert paths
    for path in paths:
        config = cfg.load_config(path)
        assert config.method.fm_training_mode == 'recursive_3step', path
        assert config.resources.world_model_initialization_sequence_length == 3, path
        assert config.method.supervision_horizon == 3, path
        assert config.method.disagreement_samples >= 2, path
        assert config.method.disagreement_weight == .1, path


def test_method_defaults_enable_stability_without_reinterpreting_omitted_legacy_fields(tmp_path):
    current = cfg.MethodConfig()
    assert current.fm_training_mode == 'recursive_3step'
    assert current.disagreement_weight == .1
    historical = configured(tmp_path)
    explicit = configured(tmp_path, '"legacy_independent"')
    assert historical.method.fm_training_mode == 'legacy_independent'
    assert historical.method.disagreement_weight == 0.
    assert historical.semantic_fingerprint == explicit.semantic_fingerprint


@pytest.mark.parametrize('mode', ['legacy_independent', 'single_step'])
def test_stability_profile_changes_checkpoint_identity(tmp_path, mode):
    legacy = configured(tmp_path, '"' + mode + '"')
    recursive = replace(legacy, method=replace(legacy.method,
        fm_training_mode='recursive_3step', disagreement_weight=.1))
    values = {name: getattr(recursive, name) for name in
              ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}
    assert cfg._semantic_fingerprint(**values) != legacy.semantic_fingerprint


def test_default_latent_penalty_reaches_shared_imagined_returns():
    from test_urban_dreamer_event_returns import make_case, seed
    results = []
    for weight in (0., cfg.MethodConfig().disagreement_weight):
        trainer, batch, _ = make_case(states=[seed('a'), seed('b')], population=4)
        trainer.config.method.disagreement_weight = weight
        rows, _ = trainer._imagine_from_seed(batch, 0, rollout_index=0)
        results.append(rows)
    assert len(results[0]) == len(results[1]) == 4
    for unpenalized, penalized in zip(*results):
        # The fixture predicts variance 2 at the sole complete endpoint.
        assert penalized['global_return'].item() == pytest.approx(
            unpenalized['global_return'].item() - .2)
        torch.testing.assert_close(penalized['global_reward'], unpenalized['global_reward'])


def fake_trainer(mode, monkeypatch, *, sampled=False):
    encoded = []
    readouts = []
    encoder_weight = torch.nn.Parameter(torch.ones(()))
    def encode(value):
        encoded.append(value.detach().clone())
        return value[..., 2:3].expand(*value.shape[:-1], 64) * encoder_weight
    def readout(value, **kwargs):
        readouts.append((value, kwargs))
        return value[..., 0].mean(-1)
    model = SimpleNamespace(encode=encode, decode=lambda z: z[..., :8], dynamics=object(),
                            active_vehicle_fraction_readout=readout)
    trainer = SimpleNamespace(validate_codec_ready=lambda: None,
        network=SimpleNamespace(capacities=[10., 20., 30.]),
        config=SimpleNamespace(method=cfg.MethodConfig(fm_training_mode=mode,
            exposure_readout_supervision='sampled_endpoint' if sampled else 'observed_endpoint'),
            resources=SimpleNamespace(world_model_pretrain_precision='fp32')),
        device=torch.device('cpu'), edge_count=3, world_model=model,
        metrics={'world_model_updates': 999})
    return trainer, encoded, readouts, encoder_weight


@pytest.mark.parametrize('mode,depth', [('single_step', 1), ('legacy_independent', 3)])
def test_single_preserves_full_input_birth_exclusion_and_avoids_solves(monkeypatch, mode, depth):
    trainer, encoded, readouts, encoder_weight = fake_trainer(mode, monkeypatch)
    batch = transition_batch()
    expected = prepare_transition_targets(batch, capacities=[10., 20., 30.])
    calls = []
    def independent(dynamics, **kw):
        calls.append(kw)
        assert not kw['predict_endpoints']
        assert not kw['z_inputs'].requires_grad and not kw['z_targets'].requires_grad
        return torch.tensor(1.), {}, kw['z_targets']
    monkeypatch.setattr(training, 'independent_transition_fm_loss', independent)
    monkeypatch.setattr(training, 'recursive_fm_loss', lambda *a, **k: pytest.fail('unexpected recursive solve'))
    training.MainlineTrainer._world_loss(trainer, batch)
    assert len(calls) == 1
    torch.testing.assert_close(encoded[0], batch.flow[:, :depth].reshape(-1, 3, 8))
    torch.testing.assert_close(encoded[1], expected.transition_target_flow[:, :depth].reshape(-1, 3, 8))
    torch.testing.assert_close(calls[0]['action'], batch.action[:, :depth])
    assert trainer.metrics['world_model_supervision_depth_last'] == depth
    assert encoder_weight.grad is None


@pytest.mark.parametrize('sampled', [False, True])
def test_recursive_uses_fixed_cohort_controls_and_correct_readout(monkeypatch, sampled):
    trainer, encoded, readouts, _ = fake_trainer('recursive_3step', monkeypatch, sampled=sampled)
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    expected = prepare_cohort_batch(batch, capacities=[10., 20., 30.])
    calls = []
    def recursive(dynamics, **kw):
        calls.append(kw)
        return torch.tensor(2., requires_grad=True), {}, (kw['z_sequence'][:, 1:] + 123.).requires_grad_()
    monkeypatch.setattr(training, 'recursive_fm_loss', recursive)
    monkeypatch.setattr(training, 'independent_transition_fm_loss', lambda *a, **k: pytest.fail('teacher forcing'))
    loss, readout_loss = training.MainlineTrainer._world_loss(trainer, batch)
    assert loss == 2.
    assert len(calls) == 1
    call = calls[0]
    assert call['predict_endpoints'] is sampled
    assert call['checkpoint_interval'] == 10
    assert len(encoded) == 1 and encoded[0].shape == (4, 3, 8)
    torch.testing.assert_close(encoded[0], expected.transition_flow.reshape(4, 3, 8))
    assert not call['z_sequence'].requires_grad
    assert call['z_sequence'].shape == (1, 4, 3, 64)
    torch.testing.assert_close(call['action'], expected.transition_action)
    assert call['action_events'].vehicle_ids == expected.transition_action_events.vehicle_ids
    assert 'newborn' not in call['action_events'].vehicle_ids
    true_latent = call['z_sequence'][:, 1:]
    torch.testing.assert_close(readouts[0][0], true_latent + (123. if sampled else 0.))
    assert not readouts[0][0].requires_grad
    previous = (torch.cat([call['z_sequence'][:, :1], readouts[0][0][:, :-1]], dim=1)
                if sampled else call['z_sequence'][:, :-1])
    torch.testing.assert_close(readouts[0][1]['z_prev'], previous)
    assert not readouts[0][1]['z_prev'].requires_grad
    predicted_fraction = readouts[0][0][..., 0].mean(-1)
    torch.testing.assert_close(readout_loss, torch.nn.functional.mse_loss(
        predicted_fraction, expected.transition_active_vehicle_fraction))


def test_recursive_rejects_missing_membership_metadata(monkeypatch):
    trainer, *_ = fake_trainer('recursive_3step', monkeypatch)
    with pytest.raises(ValueError, match='boundary snapshots and observed arrival metadata'):
        training.MainlineTrainer._world_loss(trainer, replace(transition_batch(), boundary_vehicles=None))


def test_recursive_condition_only_training_backpropagates_through_generated_states(monkeypatch):
    from urban_dreamer.models.world_model import Dynamics
    torch.manual_seed(131)
    trainer, _, _, encoder_weight = fake_trainer('recursive_3step', monkeypatch)
    dynamics = Dynamics(adjacency=torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]),
        static_features=torch.zeros(3, 5), low_frequency_rank=2, event_protocol=True,
        prior_filter='soft', prior_filter_strength=.1, prior_action_conditioned=False,
        prior_usage='condition_only')
    with torch.no_grad():
        dynamics.flow_head.weight.normal_(std=.01)
    trainer.world_model.dynamics = dynamics
    generated = []
    solve = dynamics.solve
    def capture(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.retain_grad()
        generated.append(result)
        return result
    monkeypatch.setattr(dynamics, 'solve', capture)
    batch = cohort_replay()._batch_from_picks([(0, 0)])
    loss, readout = training.MainlineTrainer._world_loss(trainer, batch)
    (loss + readout).backward()
    assert trainer.metrics['world_model_supervision_depth_last'] == 3
    assert len(generated) == 2
    assert torch.isfinite(loss) and torch.isfinite(readout)
    assert generated[0].grad is not None and generated[0].grad.abs().sum() > 0
    assert generated[1].grad is not None and generated[1].grad.abs().sum() > 0
    assert dynamics.condition_residual.output.weight.grad.abs().sum() > 0
    assert dynamics.graph_prior.output.weight.grad.abs().sum() > 0
    assert encoder_weight.grad is None


def test_single_actual_fm_has_no_solve_and_future_windows_do_not_change_loss(monkeypatch):
    from test_urban_dreamer_world_model import _small_dynamics
    trainer, _, _, encoder_weight = fake_trainer('single_step', monkeypatch)
    dynamics = _small_dynamics(noise_scale=0.)
    dynamics.transformer = torch.nn.Identity()
    trainer.world_model.dynamics = dynamics
    monkeypatch.setattr(dynamics, 'solve', lambda *a, **k: pytest.fail('single observed FM invoked ODE solve'))
    batch = transition_batch()
    future_changed = batch.flow.clone()
    future_changed[:, 2:, :, 0] += 500.
    torch.manual_seed(817)
    first, _ = training.MainlineTrainer._world_loss(trainer, batch)
    torch.manual_seed(817)
    second, _ = training.MainlineTrainer._world_loss(trainer, replace(batch, flow=future_changed))
    torch.testing.assert_close(first, second)
    first.backward()
    assert encoder_weight.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dynamics.parameters())
