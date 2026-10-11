import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from tools import migrate_imagination_budget as migration
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def fixture(tmp_path, *, passes=16, seconds=0.):
    source = tmp_path / 'source'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config, run=replace(config.run, output_dir=source, online_training_episodes=2000),
        resources=replace(config.resources, imagination_passes_per_episode=passes,
                          imagination_time_budget_seconds=seconds, world_model_imagination_precision="fp32"),
        method=replace(config.method, disagreement_samples=3),
        codec=replace(config.codec, decoder_pretrain_enabled=False))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, source / 'run_config.toml')
    protocol = dict(source={'sha256': 'pretrained', 'path': '/original/online_seed/latest.pt',
                           'input_hash': 'input', 'config_fingerprint': 'pretrain-fingerprint'},
                    online_episode_budget=2000, evaluation_every=20, device=config.run.device,
                    threads=4, gpu_memory_fraction=.25, config_semantics_unchanged=True,
                    actor_entropy={'mode': 'population'}, paired_evaluation_indices=[0, 1, 2],
                    budgets=dict(world_model_updates_per_episode=4, imagination_passes_per_episode=passes,
                                 rollouts_per_pass=8, macro_windows_per_rollout=8,
                                 imagination_time_budget_seconds=seconds))
    migration.atomic_json(source / 'pilot_protocol.json', protocol)
    progress = dict(online=[{'episode': i} for i in range(81)], before=[{'score': 3.}], after=[],
                    milestones=[{'online_episodes': 80, 'phase': 'completed'}], phase='online_training')
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
        models={'actor': {'w': torch.arange(6.).reshape(2, 3)}, 'world_model': {'w': torch.randn(6)},
                'latent_critic': {'w': torch.ones(3)}, 'event_critic': {'w': torch.arange(3.)}},
        optimizers={name: {'state': {0: {'exp_avg': torch.ones(2, 3),
                                      'exp_avg_sq': torch.ones(2, 3)*2, 'step': torch.tensor(80)}}}
                    for name in ('actor', 'latent_critic', 'event_critic', 'world_model_online')},
        rng={'torch': torch.get_rng_state()}, replay={'online': {'episodes': [{'events': [1, 2, 3]}]}},
        metrics={'online_pilot_progress': progress, 'online_episodes_completed': 81,
                 'world_model_updates': 100001, 'codec_ready': True}, next_environment_episode=81)
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=81, training_state=state)
    migration.atomic_json(source / 'pilot_progress.json', {'online': []})  # stale JSON is not authoritative
    (source / 'pilot_metrics.jsonl').write_text('{"old_record":1}\n')
    for directory in ('before_evaluation', 'milestones', 'episodes'):
        (source / directory).mkdir()
        (source / directory / 'history.json').write_text('{"old":true}')
    return source, config, state, protocol


def test_warm_continuation_preserves_all_training_state_and_history(tmp_path):
    source, old_config, state, protocol = fixture(tmp_path)
    source_hash = migration.sha256_file(source / 'checkpoints/latest.pt')
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(state, restored.training_state)
    assert restored.next_unit == manifest['global_completed_episodes'] == 81
    assert config.semantic_fingerprint != old_config.semantic_fingerprint
    assert config.resources.imagination_passes_per_episode == 2
    assert config.resources.imagination_time_budget_seconds == 60.
    assert not config.codec.decoder_pretrain_enabled
    assert source_hash == migration.sha256_file(source / 'checkpoints/latest.pt')
    assert len(json.loads((output / 'pilot_progress.json').read_text())['online']) == 81
    rewritten = json.loads((output / 'pilot_protocol.json').read_text())
    assert rewritten['source'] == protocol['source']
    assert rewritten['actor_entropy'] == protocol['actor_entropy']
    assert rewritten['budgets'] == manifest['new_budgets']
    assert rewritten['imagination_budget_migration'] == manifest
    for directory in ('before_evaluation', 'milestones', 'episodes'):
        assert (output / directory / 'history.json').read_text() == '{"old":true}'
        assert (output / 'source_history' / directory / 'history.json').exists()
    assert migration.validate_budget_continuation(output, config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('section,field,value', [
    ('method', 'fm_steps', 99), ('simulation', 'incident_speed_mps', 2.), ('run', 'seed', 400),
    ('resources', 'online_world_model_batch_size', 99), ('codec', 'decoder_pretrain_enabled', True)])
def test_other_changes_rejected(tmp_path, section, field, value):
    _, config, _, _ = fixture(tmp_path)
    target = replace(config, resources=replace(config.resources, imagination_passes_per_episode=2,
                                               imagination_time_budget_seconds=60.))
    target = replace(target, **{section: replace(getattr(target, section), **{field: value})})
    with pytest.raises(ValueError, match='only imagination budget'):
        migration.validate_budget_transfer_config(config, target)


@pytest.mark.parametrize('passes,seconds', [(0, 60.), (16, 60.), (True, 60.), (2, 0.),
                                         (2, -1.), (2, float('nan')), (2, float('inf'))])
def test_invalid_reduced_budgets_rejected_without_output(tmp_path, passes, seconds):
    source, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='positive reduced'):
        migration.migrate(source, tmp_path / 'target', passes=passes, time_budget_seconds=seconds)
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('phase', [CheckpointPhase.PRETRAIN, CheckpointPhase.COMPLETED])
def test_non_online_source_rejected(tmp_path, phase):
    source, config, state, _ = fixture(tmp_path)
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=phase, next_unit=81, training_state=state)
    with pytest.raises(ValueError, match='active ONLINE_TRAINING'):
        migration.migrate(source, tmp_path / 'target')
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('counter', ['online_episodes_completed', 'next_environment_episode', 'durable'])
def test_inconsistent_source_counters_rejected(tmp_path, counter):
    source, config, state, _ = fixture(tmp_path)
    if counter == 'durable':
        state['metrics']['online_pilot_progress']['online'].pop()
    elif counter == 'next_environment_episode':
        state['next_environment_episode'] = 82
    else:
        state['metrics'][counter] = 82
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=81, training_state=state)
    with pytest.raises(ValueError, match='differs|differ'):
        migration.migrate(source, tmp_path / 'target')


@pytest.mark.parametrize('filename', ['source_checkpoint.pt', 'source_protocol.json',
                                     'source_history/before_evaluation/history.json'])
def test_frozen_ancestry_tampering_rejected(tmp_path, filename):
    source, _, _, _ = fixture(tmp_path)
    output = tmp_path / 'target'
    migration.migrate(source, output)
    with (output / filename).open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='hash differs'):
        migration.validate_budget_continuation(output, load_config(output / 'run_config.toml'), 'input', 'pretrained')


def test_updated_checkpoint_can_advance_without_changing_initial_proof(tmp_path):
    source, _, _, _ = fixture(tmp_path)
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    config = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    state = copy.deepcopy(restored.training_state)
    state['models']['actor']['w'] += 1
    state['metrics']['online_pilot_progress']['online'].append({'episode': 81})
    state['metrics']['online_episodes_completed'] = state['next_environment_episode'] = 82
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=82, training_state=state)
    with (output / 'pilot_metrics.jsonl').open('a') as stream:
        stream.write('{"new_record":2}\n')
    assert migration.validate_budget_continuation(output, config, 'input', 'pretrained') == manifest
    with pytest.raises(ValueError, match='pretrained ancestry'):
        migration.validate_budget_continuation(output, config, 'input', 'different')


def test_source_protocol_budget_disagreement_rejected(tmp_path):
    source, _, _, protocol = fixture(tmp_path)
    protocol['budgets']['rollouts_per_pass'] = 4
    migration.atomic_json(source / 'pilot_protocol.json', protocol)
    with pytest.raises(ValueError, match='protocol budgets differ'):
        migration.migrate(source, tmp_path / 'target')


def test_target_initialization_state_tampering_rejected_even_with_updated_hash(tmp_path):
    source, _, _, _ = fixture(tmp_path)
    output = tmp_path / 'target'
    migration.migrate(source, output)
    path = output / 'continuation_checkpoint.pt'
    raw = torch.load(path, map_location='cpu', weights_only=True)
    raw['training_state']['models']['actor']['w'] += 1
    torch.save(raw, path)
    manifest_path = output / 'imagination_budget_migration.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['frozen_artifacts']['continuation_checkpoint.pt'] = migration.sha256_file(path)
    migration.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='changed a checkpoint tensor'):
        migration.validate_budget_continuation(output, load_config(output / 'run_config.toml'), 'input', 'pretrained')


def test_uncommitted_online_archives_stay_frozen_but_do_not_block_recollection(tmp_path):
    source, _, _, _ = fixture(tmp_path)
    for mode in ('online', 'warmup', 'evaluation'):
        (source / 'episodes' / mode).mkdir()
    for index in (80, 81, 82):
        (source / 'episodes' / 'online' / f'episode_{index:06d}.npz').write_bytes(b'old archived episode')
    (source / 'episodes' / 'online' / 'other_000082.npz').write_bytes(b'unrelated')
    (source / 'episodes' / 'warmup' / 'episode_000081.npz').write_bytes(b'warmup')
    (source / 'episodes' / 'evaluation' / 'episode_000082.npz').write_bytes(b'evaluation')
    output = tmp_path / 'target'
    manifest = migration.migrate(source, output)
    assert manifest['global_completed_episodes'] == 81
    assert (output / 'episodes/online/episode_000080.npz').exists()
    for index in (81, 82):
        relative = f'episodes/online/episode_{index:06d}.npz'
        frozen = output / 'source_history' / relative
        assert frozen.read_bytes() == b'old archived episode'
        assert 'source_history/' + relative in manifest['frozen_artifacts']
        assert not (output / relative).exists()
        assert (source / relative).exists()
    assert (output / 'episodes/online/other_000082.npz').exists()
    assert (output / 'episodes/warmup/episode_000081.npz').exists()
    assert (output / 'episodes/evaluation/episode_000082.npz').exists()
    # First resumed collection can create the next episode without overwriting
    # an archive generated using the previous sixteen-pass provenance.
    with (output / 'episodes/online/episode_000081.npz').open('xb') as stream:
        stream.write(b'new budget collection')
    config = load_config(output / 'run_config.toml')
    assert migration.validate_budget_continuation(output, config, 'input', 'pretrained') == manifest


def test_noise_sample_continuation_keeps_budgets_and_exact_state(tmp_path):
    source, before, state, protocol = fixture(tmp_path, passes=2, seconds=60.)
    output = tmp_path / 'noise_two'
    manifest = migration.migrate(source, output, kind=migration.SAMPLES_KIND, samples=2)
    after = load_config(output / 'run_config.toml')
    restored = load_checkpoint(output_dir=output, expected_config_fingerprint=after.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(state, restored.training_state)
    assert restored.next_unit == manifest['global_completed_episodes'] == 81
    assert manifest['kind'] == migration.SAMPLES_KIND
    assert manifest['old_disagreement_samples'] == 3
    assert manifest['new_disagreement_samples'] == after.method.disagreement_samples == 2
    assert before.semantic_fingerprint != after.semantic_fingerprint
    assert manifest['old_budgets'] == manifest['new_budgets'] == migration.imagination_budget(before)
    assert json.loads((output / 'pilot_protocol.json').read_text())['source'] == protocol['source']
    assert (output / 'source_history/milestones/history.json').exists()
    assert migration.validate_budget_continuation(output, after, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('section,field,value', [
    ('method', 'fm_steps', 99), ('run', 'seed', 400),
    ('resources', 'imagination_passes_per_episode', 3),
    ('resources', 'imagination_time_budget_seconds', 0.)])
def test_sample_transfer_cannot_also_change_budget_or_model(tmp_path, section, field, value):
    _, before, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    after = replace(before, method=replace(before.method, disagreement_samples=2))
    after = replace(after, **{section: replace(getattr(after, section), **{field: value})})
    with pytest.raises(ValueError, match='only disagreement samples'):
        migration.validate_samples_transfer_config(before, after)


@pytest.mark.parametrize('samples', [1, 3, 4, True, 2.])
def test_sample_transfer_requires_exact_three_to_two(tmp_path, samples):
    source, _, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    with pytest.raises(ValueError, match='exactly three'):
        migration.migrate(source, tmp_path / 'target', kind=migration.SAMPLES_KIND, samples=samples)
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('field,value', [('old_disagreement_samples', 4),
                                        ('new_disagreement_samples', 3), ('kind', 'unknown')])
def test_sample_manifest_tampering_rejected(tmp_path, field, value):
    source, _, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    output = tmp_path / 'target'
    migration.migrate(source, output, kind=migration.SAMPLES_KIND)
    path = output / 'imagination_budget_migration.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    migration.atomic_json(path, manifest)
    with pytest.raises(ValueError, match='recorded disagreement|unsupported imagination'):
        migration.validate_budget_continuation(output, load_config(output / 'run_config.toml'),
                                               'input', 'pretrained')


def test_budget_transfer_still_cannot_change_sample_count(tmp_path):
    source, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='cannot change disagreement samples'):
        migration.migrate(source, tmp_path / 'target', samples=2)
    before = load_config(source / 'run_config.toml')
    before = replace(before, resources=replace(before.resources, imagination_passes_per_episode=2,
                                               imagination_time_budget_seconds=60.))
    with pytest.raises(ValueError, match='original sixteen-pass'):
        migration.validate_budget_transfer_config(before, before)


def test_sample_continuation_from_existing_budget_migration(tmp_path):
    source, _, state, protocol = fixture(tmp_path)
    reduced = tmp_path / 'reduced'
    migration.migrate(source, reduced)
    target = tmp_path / 'two_samples'
    manifest = migration.migrate(reduced, target, kind=migration.SAMPLES_KIND)
    config = load_config(target / 'run_config.toml')
    restored = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(state, restored.training_state)
    assert json.loads((target / 'pilot_protocol.json').read_text())['source'] == protocol['source']
    assert manifest['old_budgets'] == manifest['new_budgets']
    assert migration.validate_budget_continuation(target, config, 'input', 'pretrained') == manifest



def test_precision_continuation_from_two_sample_run_preserves_exact_state(tmp_path):
    source, _, state, protocol = fixture(tmp_path, passes=2, seconds=60.)
    samples = tmp_path / 'samples'
    migration.migrate(source, samples, kind=migration.SAMPLES_KIND)
    target = tmp_path / 'bf16'
    manifest = migration.migrate(samples, target, kind=migration.PRECISION_KIND, precision='bf16')
    config = load_config(target / 'run_config.toml')
    before = load_config(samples / 'run_config.toml')
    restored = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(state, restored.training_state)
    assert restored.next_unit == manifest['global_completed_episodes'] == 81
    assert config.method.disagreement_samples == before.method.disagreement_samples == 2
    assert config.resources.world_model_imagination_precision == 'bf16'
    assert before.semantic_fingerprint != config.semantic_fingerprint
    assert manifest['old_budgets'] == manifest['new_budgets']
    assert manifest['old_imagination_precision'] == 'fp32'
    assert manifest['new_imagination_precision'] == 'bf16'
    assert json.loads((target / 'pilot_protocol.json').read_text())['source'] == protocol['source']
    assert (target / 'source_history/milestones/history.json').exists()
    assert migration.validate_budget_continuation(target, config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('section,field,value', [
    ('method', 'fm_steps', 99), ('method', 'disagreement_samples', 2), ('run', 'seed', 400),
    ('resources', 'imagination_passes_per_episode', 3),
    ('resources', 'imagination_time_budget_seconds', 0.),
    ('resources', 'world_model_pretrain_precision', 'bf16')])
def test_precision_transfer_rejects_other_changes(tmp_path, section, field, value):
    _, before, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    after = replace(before, resources=replace(before.resources, world_model_imagination_precision='bf16'))
    after = replace(after, **{section: replace(getattr(after, section), **{field: value})})
    with pytest.raises(ValueError, match='only imagination precision|permits unchanged budget'):
        migration.validate_precision_transfer_config(before, after)


@pytest.mark.parametrize('precision', ['fp32', 'tf32', 'fp16', True])
def test_precision_transfer_only_allows_bf16_target(tmp_path, precision):
    source, _, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    with pytest.raises(ValueError, match='exactly fp32 source and bf16 target'):
        migration.migrate(source, tmp_path / 'target', kind=migration.PRECISION_KIND, precision=precision)
    assert not (tmp_path / 'target').exists()


@pytest.mark.parametrize('field,value', [('old_imagination_precision', 'tf32'),
                                        ('new_imagination_precision', 'fp32')])
def test_precision_manifest_tampering_rejected(tmp_path, field, value):
    source, _, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    target = tmp_path / 'target'
    migration.migrate(source, target, kind=migration.PRECISION_KIND)
    path = target / 'imagination_budget_migration.json'
    manifest = json.loads(path.read_text())
    manifest[field] = value
    migration.atomic_json(path, manifest)
    with pytest.raises(ValueError, match='recorded imagination precision differs'):
        migration.validate_budget_continuation(target, load_config(target / 'run_config.toml'),
                                               'input', 'pretrained')


@pytest.mark.parametrize('kind', [migration.BUDGET_KIND, migration.SAMPLES_KIND])
def test_existing_kinds_reject_precision_override(tmp_path, kind):
    source, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='only precision continuation accepts'):
        migration.migrate(source, tmp_path / 'target', kind=kind, precision='bf16')



def test_bf16_can_explicitly_restore_eight_complete_actor_updates(tmp_path):
    source, _, state, _ = fixture(tmp_path, passes=2, seconds=60.)
    samples = tmp_path / 'two_samples'
    migration.migrate(source, samples, kind=migration.SAMPLES_KIND)
    target = tmp_path / 'bf16_eight'
    manifest = migration.migrate(samples, target, kind=migration.PRECISION_KIND,
                                 precision='bf16', passes=8, time_budget_seconds=0.)
    config = load_config(target / 'run_config.toml')
    assert config.resources.imagination_passes_per_episode == 8
    assert config.resources.imagination_time_budget_seconds == 0.
    assert manifest['old_budgets']['imagination_passes_per_episode'] == 2
    assert manifest['new_budgets']['imagination_passes_per_episode'] == 8
    assert manifest['old_budgets']['rollouts_per_pass'] == manifest['new_budgets']['rollouts_per_pass'] == 8
    restored = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                               expected_input_hash='input')
    migration._assert_equal(state, restored.training_state)
    assert migration.validate_budget_continuation(target, config, 'input', 'pretrained') == manifest


@pytest.mark.parametrize('passes,seconds', [(4, 0.), (8, 60.), (2, 0.), (8, None), (None, 0.)])
def test_bf16_budget_change_requires_exact_explicit_eight_unlimited(tmp_path, passes, seconds):
    source, _, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    samples = tmp_path / 'two_samples'
    migration.migrate(source, samples, kind=migration.SAMPLES_KIND)
    with pytest.raises(ValueError, match='permits unchanged budget'):
        migration.migrate(samples, tmp_path / 'target', kind=migration.PRECISION_KIND,
                          precision='bf16', passes=passes, time_budget_seconds=seconds)
    assert not (tmp_path / 'target').exists()


def test_bf16_eight_update_override_requires_existing_two_noise_samples(tmp_path):
    source, _, _, _ = fixture(tmp_path, passes=2, seconds=60.)
    with pytest.raises(ValueError, match='permits unchanged budget'):
        migration.migrate(source, tmp_path / 'target', kind=migration.PRECISION_KIND,
                          precision='bf16', passes=8, time_budget_seconds=0.)
