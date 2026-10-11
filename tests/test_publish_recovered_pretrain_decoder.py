import copy

import pytest
import torch

from tools.publish_recovered_pretrain_decoder import verify_candidate
from tools.recover_pretrain_decoder import candidate_training_state


def candidate_fixture():
    initial = dict(models=dict(world_model={
        'flow_projector.input_proj.weight': torch.ones(2),
        'flow_projector.decoder_volume.weight': torch.ones(2)},
        actor={'weight': torch.ones(1)}),
        optimizers=dict(decoder_pretrain={'state': {1: dict(step=torch.tensor(420.))}},
                        world_model_initial={'state': {2: dict(step=torch.tensor(420.))}}),
        metrics=dict(decoder_pretrain_updates=420., decoder_online_updates=0., codec_updates=4500.,
            world_model_updates=420., world_model_initialization_updates=420.),
        rng={'rng': [1, 2, 3]}, replay={'offline': {'episode_cursor': 0}}, reward_contract={'version': 1})
    calibrated = copy.deepcopy(initial)
    calibrated['models']['world_model']['flow_projector.decoder_volume.weight'].add_(.1)
    calibrated['optimizers']['decoder_pretrain']['state'][1]['step'].add_(3)
    calibrated['metrics'].update(codec_ready=1., decoder_pretrain_updates=423.,
        codec_validation_codec_total_relative_error=.02,
        codec_hybrid_validation_codec_total_relative_error=.09)
    source = dict(config_fingerprint='config', phase='pretrain', input_hash='inputs', next_unit=0,
                  training_state=initial)
    candidate = dict(source, training_state=candidate_training_state(initial, calibrated, updates=3, accepted=True))
    hashes = {'latest.pt': 'source', 'config': 'config', 'cache': 'cache'}
    proof = dict(status='candidate_ready', updates=3, protected_sha256_before=hashes,
        protected_sha256_after=hashes, candidate_sha256='candidate',
        proof=dict(codec_ready=True, codec_total_relative_error=.02,
                   hybrid_codec_total_relative_error=.09))
    return source, candidate, proof, hashes


def test_verified_decoder_candidate_is_accepted_without_source_mutation():
    source, candidate, proof, hashes = candidate_fixture()
    old_rng = copy.deepcopy(source['training_state']['rng'])
    verify_candidate(source, candidate, proof, protected_hashes=hashes, candidate_sha256='candidate')
    assert source['training_state']['rng'] == old_rng


@pytest.mark.parametrize('field', ['source_hash', 'candidate_hash', 'unpassed', 'rng', 'replay',
                                 'encoder', 'actor', 'optimizer', 'coverage', 'metric', 'metadata'])
def test_publisher_rejects_unverified_candidate_or_non_decoder_changes(field):
    source, candidate, proof, hashes = candidate_fixture()
    if field == 'source_hash':
        hashes = dict(hashes, **{'latest.pt': 'changed'})
    elif field == 'candidate_hash':
        proof['candidate_sha256'] = 'changed'
    elif field == 'unpassed':
        proof['proof']['codec_ready'] = False
    elif field == 'rng':
        candidate['training_state']['rng']['rng'][0] += 1
    elif field == 'replay':
        candidate['training_state']['replay']['offline']['episode_cursor'] = 1
    elif field == 'encoder':
        candidate['training_state']['models']['world_model']['flow_projector.input_proj.weight'].add_(1)
    elif field == 'actor':
        candidate['training_state']['models']['actor']['weight'].add_(1)
    elif field == 'optimizer':
        candidate['training_state']['optimizers']['world_model_initial']['state'][2]['step'].add_(1)
    elif field == 'coverage':
        candidate['training_state']['metrics']['world_model_initialization_updates'] += 1
    elif field == 'metric':
        proof['proof']['hybrid_codec_total_relative_error'] = .18
    else:
        candidate['input_hash'] = 'different'
    with pytest.raises((ValueError, AssertionError)):
        verify_candidate(source, candidate, proof, protected_hashes=hashes, candidate_sha256='candidate')
