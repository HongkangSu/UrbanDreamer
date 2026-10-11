"""Explicit, isolated checkpoint-boundary migration to the baseline SUMO backend."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def backend_config(config, output, *, backend='libsumo', device=None):
    target = replace(config, simulation=replace(config.simulation, backend=backend),
        run=replace(config.run, output_dir=Path(output).resolve(), device=device or config.run.device))
    return replace(target, semantic_fingerprint=_semantic_fingerprint(**{
        name: getattr(target, name) for name in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def validate_backend_only_change(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for values in (before, after):
        values['run'].pop('output_dir')
        values['run'].pop('device')
        values['simulation'].pop('backend')
    if before != after:
        raise ValueError('backend migration permits only simulation.backend, output_dir and device changes')
    if source.simulation.backend != 'traci' or target.simulation.backend != 'libsumo':
        raise ValueError('this migration supports only explicit traci to libsumo migration')


def migrate(source_dir, output_dir, *, source_engine_version, target_engine_version, device=None):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir == source_dir or output_dir.exists():
        raise ValueError('migration requires a new, absent output directory')
    original = load_config(source_dir / 'run_config.toml')
    config = backend_config(original, output_dir, device=device)
    validate_backend_only_change(original, config)
    protocol = json.loads((source_dir / 'pilot_protocol.json').read_text())
    original_path = latest_checkpoint_path(source_dir)
    source_hash = sha256_file(original_path)
    staging = output_dir.with_name(output_dir.name + '.migration-staging')
    if staging.exists():
        raise ValueError(f'incomplete migration staging already exists: {staging}')
    staging.mkdir(parents=True)
    frozen = latest_checkpoint_path(staging)
    frozen.parent.mkdir(parents=True)
    shutil.copy2(original_path, frozen)
    if sha256_file(frozen) != source_hash or sha256_file(original_path) != source_hash:
        raise RuntimeError('source checkpoint changed while freezing migration snapshot')
    checkpoint = load_checkpoint(output_dir=staging,
        expected_config_fingerprint=original.semantic_fingerprint,
        expected_input_hash=protocol['source']['input_hash'])
    state = checkpoint.training_state
    progress = copy.deepcopy(state['metrics'].get('online_pilot_progress'))
    if not progress or checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('migration requires an online checkpoint with durable pilot progress')
    count = len(progress['online'])
    if (checkpoint.next_unit != count or state['next_environment_episode'] != count
            or state['metrics']['online_episodes_completed'] != count):
        raise ValueError('checkpoint is not a consistent completed episode boundary')
    history = dict(source_output=str(source_dir), backend=original.simulation.backend,
        engine_version=source_engine_version, completed_online_episodes=count,
        before=copy.deepcopy(progress['before']), after=copy.deepcopy(progress['after']),
        milestones=copy.deepcopy(progress.get('milestones', [])),
        note='Historical evaluations use a different engine; do not compare against new backend evaluations.')
    progress.setdefault('backend_history', []).append(history)
    progress.update(before=[], after=[], milestones=[], phase='before_evaluation',
        evaluation_reference_online_episodes=count, evaluation_backend=config.simulation.backend,
        evaluation_engine_version=target_engine_version)
    state['metrics']['online_pilot_progress'] = progress
    migration = dict(schema_version=1, source_output=str(source_dir), target_output=str(output_dir),
        source_checkpoint_sha256=source_hash, source_config_fingerprint=original.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint, completed_online_episodes=count,
        source_backend=original.simulation.backend, target_backend=config.simulation.backend,
        source_engine_version=source_engine_version, target_engine_version=target_engine_version,
        preserved_training_state='models, optimizers, replay, RNG, counters; only evaluation progress re-referenced',
        historical_archive_root=str(source_dir / 'episodes'),
        online_episode_budget=protocol['online_episode_budget'])
    protocol.update(backend=config.simulation.backend, device=config.run.device,
        config_semantics_unchanged=False, runtime_config_fingerprint=config.semantic_fingerprint,
        backend_migration=migration)
    export_run_config(config, staging / 'run_config.toml')
    atomic_json(staging / 'pilot_protocol.json', protocol)
    atomic_json(staging / 'pilot_progress.json', progress)
    atomic_json(staging / 'backend_migration.json', migration)
    atomic_json(staging / 'historical_evaluations.json', history)
    frozen.rename(staging / 'source_checkpoint.pt')
    save_checkpoint(output_dir=staging, config_fingerprint=config.semantic_fingerprint,
        input_hash=checkpoint.input_hash, phase=checkpoint.phase, next_unit=count, training_state=state)
    if sha256_file(original_path) != source_hash:
        raise RuntimeError('source checkpoint changed before migration publication')
    staging.rename(output_dir)
    return migration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-engine-version', required=True)
    parser.add_argument('--target-engine-version', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:1'))
    args = parser.parse_args(argv)
    print(json.dumps(migrate(args.source_dir, args.output_dir,
        source_engine_version=args.source_engine_version, target_engine_version=args.target_engine_version,
        device=args.device), indent=2))


if __name__ == '__main__':
    main()
