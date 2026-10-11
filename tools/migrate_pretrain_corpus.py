"""Continue learned offline models on an explicitly supplemented compact corpus."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.migrate_pretrain_single_step import _make_trainer, validate_optimizer_state
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.learning.pretrain_cache import CACHE_SCHEMA_VERSION
from urban_dreamer.workflow.checkpoint import (
    CheckpointPhase, _validate_loaded_payload, latest_checkpoint_path, load_checkpoint,
)
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def target_config(source, output_dir, cache_manifest, *, device=None):
    config = replace(source,
        run=replace(source.run, output_dir=Path(output_dir).resolve(),
                    device=device or source.run.device, world_model_initialization_epochs=1),
        data=replace(source.data, archive_roots=(), validation_archive_roots=(),
                     pretrain_cache_manifest=Path(cache_manifest).resolve(),
                     validation_cache_manifest=Path(cache_manifest).resolve()))
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def validate_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for field in ('output_dir', 'device', 'world_model_initialization_epochs'):
            snapshot['run'].pop(field)
        for field in ('archive_roots', 'validation_archive_roots',
                      'pretrain_cache_manifest', 'validation_cache_manifest'):
            snapshot['data'].pop(field)
    if before != after:
        raise ValueError('corpus migration may change only corpus bindings, location/device and epoch budget')
    if (target.run.world_model_initialization_epochs != 1 or target.data.archive_roots
            or target.data.validation_archive_roots or target.data.pretrain_cache_manifest is None
            or target.data.validation_cache_manifest != target.data.pretrain_cache_manifest):
        raise ValueError('corpus migration requires one complete pass over one partitioned compact corpus')


def read_cache(path):
    cache = json.loads(Path(path).read_text())
    if (cache.get('schema_version') != CACHE_SCHEMA_VERSION or cache.get('status') != 'complete'
            or cache.get('identity') != _identity({k: v for k, v in cache.items() if k != 'identity'})):
        raise ValueError('corpus migration requires a complete cache with a valid identity')
    return cache


def validate_corpus_extension(source, target):
    """Forbid changing physical labels, removing old examples or crossing splits."""
    if source['contract'] != target['contract']:
        raise ValueError('supplemented corpus changed input/physical/cache contract')
    def index(cache):
        entries = {}
        groups = {'train': set(), 'validation': set()}
        for entry in cache['entries']:
            split = entry['split']
            if split not in groups:
                raise ValueError('unsupported corpus partition')
            groups[split].add(entry['demand_group'])
            identity = entry['trajectory_hash']
            binding = (split, entry['demand_group'], entry['content_hash'])
            if identity in entries and entries[identity] != binding:
                raise ValueError('ambiguous trajectory identity in corpus')
            entries[identity] = binding
        if not all(groups.values()) or groups['train'] & groups['validation']:
            raise ValueError('corpus requires independent nonempty train/held-out demand partitions')
        return entries, groups
    old_entries, old_groups = index(source)
    new_entries, new_groups = index(target)
    if (old_groups['train'] & new_groups['validation']
            or old_groups['validation'] & new_groups['train']):
        raise ValueError('supplemented corpus changed an anchored demand partition')
    for identity, binding in old_entries.items():
        if new_entries.get(identity) != binding:
            raise ValueError('supplemented corpus removed or changed an old trajectory')
    if not new_entries.keys() - old_entries.keys():
        raise ValueError('supplemented corpus contains no new unique trajectories')
    return dict(source_trajectories=len(old_entries), target_trajectories=len(new_entries),
                added_trajectories=len(new_entries.keys() - old_entries.keys()),
                train_groups=len(new_groups['train']), heldout_groups=len(new_groups['validation']))


def validate_source(checkpoint, config):
    state, metrics = checkpoint.training_state, checkpoint.training_state['metrics']
    completed = int(metrics['world_model_initialization_epochs_completed'])
    if (checkpoint.phase is CheckpointPhase.PRETRAIN and checkpoint.next_unit == completed
            and 0 <= completed < config.run.world_model_initialization_epochs):
        pass
    elif (checkpoint.phase is CheckpointPhase.ONLINE_TRAINING and checkpoint.next_unit == 0
          and completed == config.run.world_model_initialization_epochs):
        pass
    else:
        raise ValueError('corpus migration requires partial or completed offline pretraining')
    if state['reward_contract'] != 'causal_cumulative_mean_global_v1':
        raise ValueError('corpus migration requires the current physical reward contract')
    if (state['next_environment_episode'] != 0 or metrics.get('online_episodes_completed', 0)
            or metrics.get('evaluation_episodes_completed', 0) or state['replay']['online']['episodes']):
        raise ValueError('corpus migration requires no online activity')
    for name in ('actor', 'latent_critic', 'event_critic', 'world_model_online', 'decoder_online'):
        if state['optimizers'].get(name, {}).get('state'):
            raise ValueError('corpus migration refuses trained online optimizer states')
    if (state['replay']['offline']['epochs_completed'] != completed
            or state['replay'].get('codec_validation') is None):
        raise ValueError('source pretraining sampler/validation state is inconsistent')


def migrate_training_state(state, training, validation):
    if validation is None or not training.has_samples or not validation.has_samples:
        raise ValueError('corpus migration requires nonempty training and held-out data')
    migrated = copy.deepcopy(state)
    migrated['replay']['offline'] = training.state_dict()
    migrated['replay']['codec_validation'] = validation.state_dict()
    metrics = migrated['metrics']
    metrics['corpus_migration_source_counters'] = {
        key: copy.deepcopy(value) for key, value in metrics.items()
        if key.startswith(('world_model_initialization_', 'world_model_epoch_', 'pretrain_progress_'))}
    # Lifetime optimizer/update counters stay cumulative. Coverage belongs to
    # the new corpus, and partial epoch losses cannot be mixed with its pass.
    for key in list(metrics):
        if key.startswith(('world_model_epoch_', 'pretrain_progress_')) or key in (
                'world_model_initialization_epochs_completed',
                'world_model_initialization_samples_seen',
                'world_model_initialization_batches_completed'):
            metrics[key] = 0.
        if key.startswith(('codec_validation_', 'codec_hybrid_validation_',
                           'decoder_pretrain_authorized_', 'decoder_online_authorized_')):
            del metrics[key]
    metrics['codec_ready'] = 0.
    metrics['corpus_migration_codec_revalidation_required'] = True
    return migrated


def revalidate_corpus_codec(trainer):
    """Called after checkpoint restore, before the first dynamics update.

    The runner saves this changed readiness proof before continuing. Validation
    is deterministic and must not consume the restored training RNG stream.
    """
    if not trainer.metrics.get('corpus_migration_codec_revalidation_required'):
        return None
    import random
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generator_state = copy.deepcopy(trainer.rng.bit_generator.state)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        proof = trainer._validate_codec()
        trainer.validate_codec_pretraining_ready()
        trainer._set_dynamics_codec_gradients()
        trainer.metrics['corpus_migration_codec_revalidation_required'] = False
        return proof
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        trainer.rng.bit_generator.state = generator_state
        torch.set_rng_state(cpu_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def validate_corpus_continuation(output_dir, config, input_hash):
    output = Path(output_dir).resolve()
    proof = json.loads((output / 'corpus_migration.json').read_text())
    if (proof['target_output'] != str(output) or proof['input_hash'] != input_hash
            or proof['target_config_fingerprint'] != config.semantic_fingerprint):
        raise ValueError('corpus continuation target identity differs')
    for filename, expected in proof['frozen_artifacts'].items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'corpus continuation frozen {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_transfer_config(source_config, config)
    if source_config.semantic_fingerprint != proof['source_config_fingerprint']:
        raise ValueError('corpus continuation source configuration differs')
    source_cache = read_cache(output / 'source_cache_manifest.json')
    target_cache = read_cache(output / 'target_cache_manifest.json')
    validate_corpus_extension(source_cache, target_cache)
    if read_cache(config.data.pretrain_cache_manifest)['identity'] != target_cache['identity']:
        raise ValueError('corpus continuation live cache differs from frozen corpus')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    validate_source(source, source_config)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    if initial.phase is not CheckpointPhase.PRETRAIN or initial.next_unit != 0:
        raise ValueError('corpus continuation must begin a new coverage pass at epoch zero')
    for key in ('models', 'optimizers', 'rng', 'reward_contract', 'next_environment_episode'):
        _assert_equal(source.training_state[key], initial.training_state[key])
    _assert_equal(source.training_state['replay']['online'], initial.training_state['replay']['online'])
    metrics = initial.training_state['metrics']
    if (metrics['codec_ready'] or not metrics['corpus_migration_codec_revalidation_required']
            or metrics['world_model_initialization_epochs_completed'] != 0
            or metrics['world_model_initialization_samples_seen'] != 0
            or metrics['world_model_initialization_batches_completed'] != 0):
        raise ValueError('corpus continuation coverage/readiness was not reset')
    for key in ('world_model_updates', 'world_model_initialization_updates', 'codec_updates'):
        if metrics[key] != source.training_state['metrics'][key]:
            raise ValueError('corpus continuation changed lifetime updates')
    for split, key in (('train', 'offline'), ('validation', 'codec_validation')):
        dataset = initial.training_state['replay'][key]
        _assert_equal(dataset, proof['target_samplers'][key])
        if (dataset['epochs_completed'] or dataset['episode_cursor'] or dataset['start_cursor']
                or dataset['epoch_order'] or dataset['start_order']):
            raise ValueError(f'corpus continuation {split} sampler was not reset')
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    if latest.training_state['metrics']['world_model_updates'] < metrics['world_model_updates']:
        raise ValueError('corpus continuation latest updates moved backwards')
    return proof


def migrate(source_dir, output_dir, cache_manifest, *, device=None):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError('corpus migration requires a new absent output directory')
    old_config = load_config(source_dir / 'run_config.toml')
    if (old_config.data.pretrain_cache_manifest is None
            or old_config.data.validation_cache_manifest not in
            (None, old_config.data.pretrain_cache_manifest)):
        raise ValueError('source requires one verified partitioned compact corpus')
    config = target_config(old_config, output, cache_manifest, device=device)
    validate_transfer_config(old_config, config)
    old_cache, new_cache = read_cache(old_config.data.pretrain_cache_manifest), read_cache(cache_manifest)
    extension = validate_corpus_extension(old_cache, new_cache)
    raw = torch.load(latest_checkpoint_path(source_dir), map_location='cpu', weights_only=True)
    source = _validate_loaded_payload(payload=raw, path=latest_checkpoint_path(source_dir),
        expected_config_fingerprint=old_config.semantic_fingerprint, expected_input_hash=raw['input_hash'])
    validate_source(source, old_config)
    trainer, input_hash = _make_trainer(config)
    try:
        if input_hash != source.input_hash:
            raise ValueError('corpus migration prepared input hash differs')
        for key, data in (('offline', trainer.offline_data), ('codec_validation', trainer.codec_validation_data)):
            old_manifest = copy.deepcopy(data.manifest)
            split = old_manifest['contract']['split']
            old_manifest.update(entries=[entry for entry in old_cache['entries'] if entry['split'] == split],
                files=old_cache['sources'], discovered_files=len(old_cache['sources']),
                duplicate_files=old_cache['duplicate_files'])
            old_manifest['contract']['source_manifest'] = old_cache['identity']
            old_manifest.pop('identity')
            if source.training_state['replay'][key]['manifest_identity'] != _identity(old_manifest):
                raise ValueError('source checkpoint sampler is not bound to its frozen corpus')
        state = migrate_training_state(source.training_state, trainer.offline_data, trainer.codec_validation_data)
        validate_optimizer_state(trainer, state)
        trainer.load_state_dict(state)
        output.mkdir(parents=True)
        shutil.copy2(source_dir / 'run_config.toml', output / 'source_config.toml')
        # Serialize the already loaded source so a concurrently advancing latest
        # pointer cannot mismatch the state against its recorded frozen ancestry.
        torch.save(raw, output / 'source_checkpoint.pt')
        shutil.copy2(old_config.data.pretrain_cache_manifest, output / 'source_cache_manifest.json')
        shutil.copy2(cache_manifest, output / 'target_cache_manifest.json')
        export_run_config(config, output / 'run_config.toml')
        initial = dict(raw, config_fingerprint=config.semantic_fingerprint,
            phase=CheckpointPhase.PRETRAIN.value, next_unit=0, training_state=state)
        torch.save(initial, output / 'continuation_checkpoint.pt')
        latest_checkpoint_path(output).parent.mkdir(parents=True)
        shutil.copy2(output / 'continuation_checkpoint.pt', latest_checkpoint_path(output))
        filenames = ['source_config.toml', 'source_checkpoint.pt', 'run_config.toml',
                     'continuation_checkpoint.pt', 'source_cache_manifest.json', 'target_cache_manifest.json']
        proof = dict(schema_version=1, kind='supplemented_corpus_warm_pretrain_continuation',
            source_output=str(source_dir), target_output=str(output), input_hash=input_hash,
            source_config_fingerprint=old_config.semantic_fingerprint,
            target_config_fingerprint=config.semantic_fingerprint,
            frozen_artifacts={name: sha256_file(output / name) for name in filenames},
            source_phase=source.phase.value,
            source_completed_epochs=source.training_state['metrics']['world_model_initialization_epochs_completed'],
            source_progress=state['metrics']['corpus_migration_source_counters'],
            source_cache_identity=old_cache['identity'], target_cache_identity=new_cache['identity'],
            source_sampler=source.training_state['replay']['offline'],
            target_samplers={key: state['replay'][key] for key in ('offline', 'codec_validation')},
            target_coverage_epochs=1, codec_revalidation_required=True, corpus=extension,
            preserved='all models, optimizer tensors, RNG, lifetime updates, empty online replay',
            reset='corpus-specific coverage, sampler orders/cursors, epoch sums, held-out codec proof/admission')
        atomic_json(output / 'corpus_migration.json', proof)
        validate_corpus_continuation(output, config, input_hash)
        return proof
    finally:
        trainer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-manifest', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    print(json.dumps(migrate(args.source_dir, args.output_dir, args.cache_manifest,
                             device=args.device), indent=2))


if __name__ == '__main__':
    main()
