import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tools.recover_pretrain_decoder import candidate_training_state, verify_initial_report, validate_recovery_codec
from tools.probe_slow_codec import freeze_encoder_for_decoder_probe
from tools.migrate_mainline_prior import _assert_equal


def states():
    source = dict(models=dict(world_model={
        'flow_projector.input_proj.weight': torch.ones(2),
        'flow_projector.decoder_volume.weight': torch.ones(2),
        'dynamics.weight': torch.ones(2)}, actor={'weight': torch.ones(1)}),
        optimizers=dict(decoder_pretrain={'state': {1: dict(step=torch.tensor(420.), exp_avg=torch.ones(2))}},
                        world_model_initial={'state': {2: dict(step=torch.tensor(420.))}}),
        metrics=dict(decoder_pretrain_updates=420., decoder_online_updates=0., codec_updates=4500.,
                     world_model_updates=420., world_model_initialization_updates=420.,
                     world_model_initialization_samples_seen=0.,
                     corpus_migration_codec_revalidation_required=True),
        rng={'numpy': [123, 456]}, replay={'offline': {'episode_cursor': 0}},
        reward_contract={'semantic': 'unchanged'}, next_environment_episode=0)
    calibrated = copy.deepcopy(source)
    calibrated['models']['world_model']['flow_projector.decoder_volume.weight'].add_(.1)
    calibrated['optimizers']['decoder_pretrain']['state'][1]['step'].add_(3)
    calibrated['optimizers']['decoder_pretrain']['state'][1]['exp_avg'].mul_(.9)
    calibrated['metrics'].update(codec_ready=1., decoder_pretrain_updates=423.,
        codec_validation_manifest_identity='new-heldout', codec_validation_updates=4500.,
        codec_validation_decoder_updates=423., decoder_pretrain_authorized_manifest_identity='new-heldout',
        decoder_pretrain_authorized_codec_updates=4500.)
    return source, calibrated


def test_candidate_preserves_source_rng_replay_coverage_and_non_decoder_moments():
    source, calibrated = states()
    original = copy.deepcopy(source)
    result = candidate_training_state(source, calibrated, updates=3, accepted=True)
    _assert_equal(source, original)
    for key in ('rng', 'replay', 'reward_contract', 'next_environment_episode'):
        _assert_equal(result[key], original[key])
    _assert_equal(result['optimizers']['world_model_initial'], source['optimizers']['world_model_initial'])
    assert result['metrics']['world_model_updates'] == 420
    assert result['metrics']['world_model_initialization_samples_seen'] == 0
    assert result['metrics']['decoder_pretrain_updates'] == 423
    assert not result['metrics']['corpus_migration_codec_revalidation_required']
    assert result['metrics']['decoder_pretrain_authorized_manifest_identity'] == 'new-heldout'
    assert result['optimizers']['decoder_pretrain']['state'][1]['step'] == 423
    assert torch.equal(result['models']['world_model']['flow_projector.decoder_volume.weight'], torch.full((2,), 1.1))


@pytest.mark.parametrize('accepted,ready', [(False, 1.), (True, 0.)])
def test_unpassed_full_gate_cannot_create_a_candidate(accepted, ready):
    source, calibrated = states()
    calibrated['metrics']['codec_ready'] = ready
    with pytest.raises(ValueError, match='full heldout quality gate'):
        candidate_training_state(source, calibrated, updates=3, accepted=accepted)


@pytest.mark.parametrize('field', ['encoder', 'dynamics', 'actor', 'optimizer', 'counter', 'schema'])
def test_recovery_rejects_mutation_outside_decoder(field):
    source, calibrated = states()
    if field == 'encoder':
        calibrated['models']['world_model']['flow_projector.input_proj.weight'].add_(1)
    elif field == 'dynamics':
        calibrated['models']['world_model']['dynamics.weight'].add_(1)
    elif field == 'actor':
        calibrated['models']['actor']['weight'].add_(1)
    elif field == 'optimizer':
        calibrated['optimizers']['world_model_initial']['state'][2]['step'].add_(1)
    elif field == 'counter':
        calibrated['metrics']['world_model_updates'] += 1
    else:
        calibrated['models']['world_model']['extra.weight'] = torch.ones(1)
    with pytest.raises((ValueError, AssertionError)):
        candidate_training_state(source, calibrated, updates=3, accepted=True)


def test_diagnostic_source_hashes_and_revision_are_verified():
    hashes = {'latest.pt': 'abc', 'config': 'def', 'cache': 'ghi'}
    report = dict(protected_sha256_before=hashes, protected_sha256_after=hashes,
                  optimizer_steps=0, models_optimizers_unchanged=True, decoder_revision=420)
    verify_initial_report(report, hashes, 420)
    for change in ({'optimizer_steps': 1}, {'decoder_revision': 421},
                   {'protected_sha256_after': {}}, {'models_optimizers_unchanged': False}):
        with pytest.raises(ValueError, match='does not match'):
            verify_initial_report(dict(report, **change), hashes, 420)


@pytest.mark.parametrize('fail', [False, True])
def test_validation_batch_size_changes_no_quality_limits_and_always_restores_config(fail):
    @dataclass(frozen=True)
    class Codec:
        batch_size: int = 4
        total_relative_mae_max: float = .1
    @dataclass(frozen=True)
    class Config:
        codec: Codec
    config = Config(Codec())
    trainer = SimpleNamespace(config=config)
    def validate():
        assert trainer.config.codec.batch_size == 32
        assert trainer.config.codec.total_relative_mae_max == .1
        if fail:
            raise ValueError('gate failed')
        return {'codec_ready': True}
    trainer._validate_codec = validate
    if fail:
        with pytest.raises(ValueError, match='gate failed'):
            validate_recovery_codec(trainer, 32)
    else:
        assert validate_recovery_codec(trainer, 32) == {'codec_ready': True}
    assert trainer.config is config


def test_actual_decoder_optimizer_keeps_warm_moments_and_encoder_fixed():
    class Codec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder_shared = torch.nn.Linear(2, 2)
            self.decoder_volume = torch.nn.Linear(2, 1)
            self.decoder_speed = torch.nn.Linear(2, 1)
            self.decoder_other = torch.nn.Linear(2, 2)
    model = Codec()
    freeze_encoder_for_decoder_probe(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.0001)
    encoder = copy.deepcopy(model.encoder.state_dict())
    inputs = torch.ones(4, 2)
    def step():
        with torch.no_grad():
            latent = model.encoder(inputs)
        optimizer.zero_grad(set_to_none=True)
        model.decoder_volume(model.decoder_shared(latent)).square().mean().backward()
        optimizer.step()
    step()
    warm = copy.deepcopy(optimizer.state_dict())
    resumed = torch.optim.Adam(parameters, lr=.0001)
    resumed.load_state_dict(warm)
    optimizer = resumed
    step()
    assert all(state['step'] == 2 for state in optimizer.state.values())
    _assert_equal(model.encoder.state_dict(), encoder)
