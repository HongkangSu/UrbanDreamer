import copy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools import migrate_pretrain_corpus as migration
from tools import run_complete_pretrain as runner
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CACHE_SCHEMA_VERSION
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config
from urban_dreamer.workflow.pipeline import MainlinePipeline


def cache(entries):
    result = dict(schema_version=CACHE_SCHEMA_VERSION, status='complete',
        contract={'city': 'test', 'physical': {'horizon': 90}}, entries=entries,
        sources=copy.deepcopy(entries), duplicate_files=0)
    result['identity'] = _identity(result)
    return result


def entry(name, split):
    return dict(path=name, trajectory_hash=name, content_hash=name+'-content',
                demand_group=name+'-demand', split=split, windows=5)


class Dataset:
    def __init__(self, source, split):
        self.has_samples = True
        self.manifest = dict(schema_version=1,
            contract=dict(source_manifest=source['identity'], split=split,
                          sequence_length=1, target_deadline_seconds=90,
                          transition_target_scope='previous_state_excluding_current_births_v1'),
            entries=[e for e in source['entries'] if e['split'] == split],
            files=source['sources'], discovered_files=len(source['sources']),
            duplicate_files=source['duplicate_files'])
        self.manifest['identity'] = _identity(self.manifest)

    def state_dict(self):
        return dict(schema_version=1, manifest_identity=self.manifest['identity'],
                    epoch_order=[], start_order=[], episode_cursor=0,
                    start_cursor=0, epochs_completed=0)


def fixture(tmp_path, *, complete=False):
    source_dir = tmp_path / 'source'
    source_dir.mkdir()
    old = cache([entry('old-train', 'train'), entry('old-val', 'validation')])
    new = cache(copy.deepcopy(old['entries']) + [entry('new-train', 'train'), entry('new-val', 'validation')])
    old_path, new_path = tmp_path / 'old.json', tmp_path / 'new.json'
    old_path.write_text(json.dumps(old))
    new_path.write_text(json.dumps(new))
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config,
        run=replace(config.run, output_dir=source_dir, world_model_initialization_epochs=1),
        resources=replace(config.resources, world_model_initialization_sequence_length=1),
        data=replace(config.data, pretrain_cache_manifest=old_path, validation_cache_manifest=old_path))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        k: getattr(config, k) for k in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    training, validation = Dataset(old, 'train'), Dataset(old, 'validation')
    completed = int(complete)
    state = dict(reward_contract='causal_cumulative_mean_global_v1',
        models={'world_model': {'weight': parameter.detach().clone()}},
        optimizers={'world_model_initial': optimizer.state_dict()},
        rng={'torch': torch.get_rng_state()}, next_environment_episode=0,
        replay=dict(online={'episodes': []}, offline=training.state_dict(),
                    codec_validation=validation.state_dict()),
        metrics=dict(codec_ready=1., codec_updates=4500, decoder_pretrain_updates=420,
            codec_validation_updates=4500, codec_validation_manifest_identity=validation.manifest['identity'],
            decoder_pretrain_authorized_manifest_identity=validation.manifest['identity'],
            decoder_pretrain_authorized_codec_updates=4500,
            decoder_online_authorized_manifest_identity=validation.manifest['identity'],
            world_model_initialization_epochs_completed=float(completed),
            world_model_initialization_samples_seen=24002.,
            world_model_initialization_batches_completed=420.,
            world_model_initialization_updates=420., world_model_updates=420.,
            world_model_epoch_samples=2., world_model_epoch_updates=1.,
            world_model_epoch_loss_sum=5., online_episodes_completed=0.))
    state['replay']['offline'].update(epochs_completed=completed)
    if not complete:
        state['replay']['offline'].update(epoch_order=[0], start_order=[3, 2, 0, 1, 4], start_cursor=2)
    export_run_config(config, source_dir / 'run_config.toml')
    save_checkpoint(output_dir=source_dir, config_fingerprint=config.semantic_fingerprint, input_hash='input',
        phase=CheckpointPhase.ONLINE_TRAINING if complete else CheckpointPhase.PRETRAIN,
        next_unit=0, training_state=state)
    trainer = SimpleNamespace(offline_data=Dataset(new, 'train'), codec_validation_data=Dataset(new, 'validation'),
        optimizers={'world_model_initial': optimizer}, load_state_dict=lambda state: None, close=lambda: None)
    state = load_checkpoint(output_dir=source_dir, expected_config_fingerprint=config.semantic_fingerprint,
                            expected_input_hash='input').training_state
    return source_dir, config, state, trainer, new_path, old, new


@pytest.mark.parametrize('complete', [False, True])
def test_supplement_preserves_all_learned_state_and_resets_only_corpus_coverage(tmp_path, monkeypatch, complete):
    source, _, original, trainer, new_path, _, _ = fixture(tmp_path, complete=complete)
    monkeypatch.setattr(migration, '_make_trainer', lambda config: (trainer, 'input'))
    original_hash = migration.sha256_file(source / 'checkpoints/latest.pt')
    target = tmp_path / 'target'
    proof = migration.migrate(source, target, new_path)
    config = load_config(target / 'run_config.toml')
    loaded = load_checkpoint(output_dir=target, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash='input')
    state, metrics = loaded.training_state, loaded.training_state['metrics']
    for key in ('models', 'optimizers', 'rng'):
        migration._assert_equal(state[key], original[key])
    assert loaded.phase is CheckpointPhase.PRETRAIN and loaded.next_unit == 0
    assert metrics['world_model_updates'] == 420
    assert metrics['world_model_initialization_updates'] == 420
    assert metrics['decoder_pretrain_updates'] == 420
    assert metrics['world_model_initialization_samples_seen'] == 0
    assert metrics['world_model_epoch_samples'] == metrics['world_model_epoch_loss_sum'] == 0
    assert state['replay']['offline'] == trainer.offline_data.state_dict()
    assert state['replay']['codec_validation'] == trainer.codec_validation_data.state_dict()
    assert not metrics['codec_ready']
    assert not any(k.startswith('decoder_pretrain_authorized_') for k in metrics)
    assert not any(k.startswith('decoder_online_authorized_') for k in metrics)
    assert 'codec_validation_manifest_identity' not in metrics
    assert metrics['corpus_migration_codec_revalidation_required']
    assert proof['source_completed_epochs'] == int(complete)
    assert proof['corpus']['added_trajectories'] == 2
    assert migration.sha256_file(source / 'checkpoints/latest.pt') == original_hash
    assert migration.validate_corpus_continuation(target, config, 'input') == proof
    new = json.loads(new_path.read_text())
    new['changed'] = True
    new['identity'] = _identity({k:v for k,v in new.items() if k != 'identity'})
    new_path.write_text(json.dumps(new))
    with pytest.raises(ValueError, match='live cache'):
        migration.validate_corpus_continuation(target, config, 'input')


@pytest.mark.parametrize('damage', ['partition', 'leak', 'removed', 'content', 'physics', 'empty', 'nonew'])
def test_reject_incompatible_or_leaky_supplement(tmp_path, damage):
    _, _, _, _, _, old, new = fixture(tmp_path)
    if damage == 'partition':
        new['entries'][0]['split'] = 'validation'
    elif damage == 'leak':
        new['entries'][2]['demand_group'] = old['entries'][1]['demand_group']
    elif damage == 'removed':
        new['entries'].pop(0)
    elif damage == 'content':
        new['entries'][0]['content_hash'] = 'changed'
    elif damage == 'physics':
        new['contract']['physical']['horizon'] = 999
    elif damage == 'empty':
        new['entries'] = [e for e in new['entries'] if e['split'] == 'train']
    else:
        new = old
    with pytest.raises(ValueError):
        migration.validate_corpus_extension(old, new)


def test_reject_online_source_and_unrelated_config_changes(tmp_path, monkeypatch):
    source, config, state, trainer, path, _, _ = fixture(tmp_path)
    target = migration.target_config(config, tmp_path / 'target', path)
    with pytest.raises(ValueError, match='only corpus'):
        migration.validate_transfer_config(config, replace(target, method=replace(target.method, fm_steps=3)))
    state['replay']['online']['episodes'] = [{}]
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='input',
                    phase=CheckpointPhase.PRETRAIN, next_unit=0, training_state=state)
    monkeypatch.setattr(migration, '_make_trainer', lambda config: (trainer, 'input'))
    with pytest.raises(ValueError, match='no online activity'):
        migration.migrate(source, tmp_path / 'target', path)


@pytest.mark.parametrize('passes', [False, True])
def test_new_heldout_revalidation_precedes_authorization_and_restores_rng(passes):
    calls = []
    metrics = {'corpus_migration_codec_revalidation_required': True}
    trainer = SimpleNamespace(metrics=metrics, rng=np.random.default_rng(92))
    def validate():
        calls.append('full_heldout')
        random.random()
        np.random.random()
        trainer.rng.random()
        torch.rand(3)
        metrics['codec_ready'] = float(passes)
        return {'codec_ready': passes}
    def authorize():
        calls.append('authorize')
        if not passes:
            raise ValueError('quality failed')
    trainer._validate_codec = validate
    trainer.validate_codec_pretraining_ready = authorize
    trainer._set_dynamics_codec_gradients = lambda: calls.append('gradients')
    old = (random.getstate(), np.random.get_state(), copy.deepcopy(trainer.rng.bit_generator.state), torch.get_rng_state())
    if passes:
        assert migration.revalidate_corpus_codec(trainer) == {'codec_ready': True}
        assert calls == ['full_heldout', 'authorize', 'gradients']
        assert not metrics['corpus_migration_codec_revalidation_required']
        assert migration.revalidate_corpus_codec(trainer) is None
    else:
        with pytest.raises(ValueError, match='quality failed'):
            migration.revalidate_corpus_codec(trainer)
        assert calls == ['full_heldout', 'authorize']
        assert metrics['corpus_migration_codec_revalidation_required']
    assert random.getstate() == old[0]
    np.testing.assert_equal(np.random.get_state(), old[1])
    assert trainer.rng.bit_generator.state == old[2]
    assert torch.equal(torch.get_rng_state(), old[3])


@pytest.mark.parametrize('passes', [False, True])
def test_actual_runner_hook_blocks_or_persists_revalidation_before_fm_and_skips_on_resume(
        tmp_path, monkeypatch, passes):
    source, _, _, construction, cache_path, _, _ = fixture(tmp_path)
    monkeypatch.setattr(migration, '_make_trainer', lambda config: (construction, 'input'))
    target = tmp_path / 'target'
    migration.migrate(source, target, cache_path)
    config = load_config(target / 'run_config.toml')
    original_checkpoint_hash = migration.sha256_file(target / 'checkpoints/latest.pt')
    calls = []
    saved = runner.save_checkpoint

    def save_revalidation(**kwargs):
        result = saved(**kwargs)
        calls.append('save_revalidation')
        return result

    def base_restore(trainer, state):
        trainer._state = copy.deepcopy(state)
        trainer.metrics = trainer._state['metrics']
        calls.append('restore')

    monkeypatch.setattr(runner.MainlineTrainer, 'load_state_dict', base_restore)
    monkeypatch.setattr(runner, 'save_checkpoint', save_revalidation)
    monkeypatch.setattr(runner, 'combined_input_hash', lambda prepared: 'input')
    monkeypatch.setattr(runner, 'status', lambda *args, **kwargs: None)

    def factory():
        trainer = object.__new__(runner.MonitoredTrainer)
        trainer.config = config
        trainer.prepared_inputs = SimpleNamespace()
        trainer.monitor_path = target / 'monitor.json'
        trainer._runtime_profile = {}
        trainer.codec_validation_data = SimpleNamespace(summary={'heldout': 'new'})
        trainer.rng = np.random.default_rng(71)
        trainer.close = lambda: None
        trainer.codec_ready = lambda: bool(trainer.metrics.get('codec_ready'))
        trainer.state_dict = lambda: copy.deepcopy(trainer._state)
        trainer.summary = lambda: dict(updates=trainer.metrics['world_model_updates'])
        trainer._set_dynamics_codec_gradients = lambda: calls.append('gradients')

        def full_heldout():
            calls.append('full_heldout')
            trainer.metrics.update(codec_ready=float(passes),
                codec_validation_manifest_identity='new-heldout-proof')
            return {'codec_ready': passes}

        def authorize():
            calls.append('authorize')
            if not trainer.codec_ready():
                raise ValueError('new heldout quality gate failed')

        def fm_epoch(epoch):
            persisted = load_checkpoint(output_dir=target,
                expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash='input')
            assert persisted.phase is CheckpointPhase.PRETRAIN and persisted.next_unit == epoch
            assert not persisted.training_state['metrics']['corpus_migration_codec_revalidation_required']
            assert persisted.training_state['metrics']['codec_validation_manifest_identity'] == 'new-heldout-proof'
            calls.append('fm')
            trainer.metrics['world_model_updates'] += 1
            trainer.metrics['world_model_initialization_epochs_completed'] = epoch + 1

        trainer._validate_codec = full_heldout
        trainer.validate_codec_pretraining_ready = authorize
        trainer.validate_codec_ready = authorize
        trainer.initialize_world_model_epoch = fm_epoch
        trainer.initialize_codec_epoch = lambda epoch: pytest.fail('codec training must not restart')
        trainer.train_online_episode = lambda episode: pytest.fail('offline continuation must not train online')
        trainer.evaluate_episode = lambda episode: pytest.fail('offline continuation must not evaluate online')
        return trainer

    pipeline = MainlinePipeline(config=config, input_hash='input', trainer_factory=factory, pretrain_only=True)
    if not passes:
        with pytest.raises(ValueError, match='new heldout quality gate failed'):
            pipeline.run()
        assert calls == ['restore', 'full_heldout', 'authorize']
        assert migration.sha256_file(target / 'checkpoints/latest.pt') == original_checkpoint_hash
    else:
        result = pipeline.run()
        assert result['status'] == 'pretrained'
        assert calls.index('full_heldout') < calls.index('gradients') < calls.index('save_revalidation') < calls.index('fm')
        before = list(calls)
        result = pipeline.run()
        assert result['status'] == 'pretrained'
        assert calls.count('full_heldout') == before.count('full_heldout') == 1
        assert calls.count('save_revalidation') == before.count('save_revalidation') == 1
        assert calls.count('fm') == before.count('fm') == 1
