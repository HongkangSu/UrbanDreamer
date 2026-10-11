"""Preserve online state across an explicit imagination configuration continuation."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config

BUDGET_KIND = 'reduce_imagination_budget_continuation'
SAMPLES_KIND = 'disagreement_samples_continuation'
PRECISION_KIND = 'imagination_precision_continuation'

BUDGET_FIELDS = ('imagination_passes_per_episode', 'imagination_time_budget_seconds')
FIXED_BUDGETS = dict(world_model_updates_per_episode=4, rollouts_per_pass=8,
                    macro_windows_per_rollout=8)


def imagination_budget(config):
    return dict(FIXED_BUDGETS,
        rollouts_per_pass=config.resources.imagination_rollouts_per_pass,
        imagination_passes_per_episode=config.resources.imagination_passes_per_episode,
        imagination_time_budget_seconds=config.resources.imagination_time_budget_seconds)


def validate_budget_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for field in BUDGET_FIELDS:
            snapshot['resources'].pop(field)
        for field in ('output_dir', 'device'):
            snapshot['run'].pop(field)
    if before != after:
        raise ValueError('budget continuation may change only imagination budget, output directory and device')
    if (source.resources.imagination_passes_per_episode != 16 or
            source.resources.imagination_time_budget_seconds != 0):
        raise ValueError('budget continuation requires the original sixteen-pass unlimited source')
    passes, seconds = (getattr(target.resources, name) for name in BUDGET_FIELDS)
    if (isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes < 16 or
            isinstance(seconds, bool) or not isinstance(seconds, (float, int)) or
            not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('budget continuation needs a positive reduced pass count and finite positive time budget')



def validate_samples_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        snapshot['method'].pop('disagreement_samples')
        for field in ('output_dir', 'device'):
            snapshot['run'].pop(field)
    if before != after:
        raise ValueError('sample continuation may change only disagreement samples, output directory and device')
    if (type(source.method.disagreement_samples) is not int or
            type(target.method.disagreement_samples) is not int or
            source.method.disagreement_samples != 3 or target.method.disagreement_samples != 2):
        raise ValueError('sample continuation requires exactly three source samples and two target samples')


def validate_precision_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    old_budget = tuple(getattr(source.resources, field) for field in BUDGET_FIELDS)
    new_budget = tuple(getattr(target.resources, field) for field in BUDGET_FIELDS)
    if old_budget != new_budget and not (
            old_budget == (2, 60.) and new_budget == (8, 0.) and
            type(target.resources.imagination_passes_per_episode) is int and
            source.method.disagreement_samples == target.method.disagreement_samples == 2):
        raise ValueError('precision continuation permits unchanged budget or explicit two/60 to eight/unlimited with two samples')
    for snapshot in (before, after):
        snapshot['resources'].pop('world_model_imagination_precision')
        for field in BUDGET_FIELDS:
            snapshot['resources'].pop(field)
        for field in ('output_dir', 'device'):
            snapshot['run'].pop(field)
    if before != after:
        raise ValueError('precision continuation may change only imagination precision, output directory and device')
    if (source.resources.world_model_imagination_precision != 'fp32' or
            target.resources.world_model_imagination_precision != 'bf16'):
        raise ValueError('precision continuation requires exactly fp32 source and bf16 target')


def _validate_transfer(source, target, kind):
    if kind == BUDGET_KIND:
        validate_budget_transfer_config(source, target)
    elif kind == SAMPLES_KIND:
        validate_samples_transfer_config(source, target)
    elif kind == PRECISION_KIND:
        validate_precision_transfer_config(source, target)
    else:
        raise ValueError(f'unsupported imagination continuation kind: {kind!r}')


def _validate_online(checkpoint, config, *, minimum_cursor=0):
    state = checkpoint.training_state
    progress = state['metrics'].get('online_pilot_progress')
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('budget continuation requires an active ONLINE_TRAINING checkpoint')
    if (not isinstance(progress, dict) or not isinstance(progress.get('online'), list) or
            len(progress['online']) != checkpoint.next_unit):
        raise ValueError('budget continuation durable progress differs from its episode cursor')
    if not minimum_cursor <= checkpoint.next_unit < config.run.online_training_episodes:
        raise ValueError('budget continuation requires remaining episodes and a nondecreasing global cursor')
    if (state['metrics'].get('online_episodes_completed') != checkpoint.next_unit or
            state['next_environment_episode'] != checkpoint.next_unit):
        raise ValueError('budget continuation online counters differ from its durable cursor')
    return progress


def rewrite_protocol(source_protocol, config, manifest):
    """Clone original pretrained ancestry and all execution/evaluation settings."""
    protocol = copy.deepcopy(source_protocol)
    protocol.update(budgets=imagination_budget(config), device=config.run.device,
                    config_semantics_unchanged=False,
                    runtime_config_fingerprint=config.semantic_fingerprint,
                    imagination_budget_migration=copy.deepcopy(manifest))
    return protocol


def validate_budget_continuation(output_dir, config, input_hash, pretrained_source_hash):
    """Check frozen ancestry and exact initial state; allow subsequent online updates."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'imagination_budget_migration.json').read_text())
    if (manifest['target_output'] != str(output) or
            manifest['target_config_fingerprint'] != config.semantic_fingerprint or
            manifest['input_hash'] != input_hash):
        raise ValueError('budget continuation target identity differs')
    for filename, expected in manifest['frozen_artifacts'].items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'budget continuation frozen {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    _validate_transfer(source_config, config, manifest['kind'])
    if manifest['kind'] == SAMPLES_KIND and (
            manifest.get('old_disagreement_samples') != source_config.method.disagreement_samples or
            manifest.get('new_disagreement_samples') != config.method.disagreement_samples):
        raise ValueError('sample continuation recorded disagreement samples differ')
    if manifest['kind'] == PRECISION_KIND and (
            manifest.get('old_imagination_precision') != source_config.resources.world_model_imagination_precision or
            manifest.get('new_imagination_precision') != config.resources.world_model_imagination_precision):
        raise ValueError('precision continuation recorded imagination precision differs')
    if (source_config.semantic_fingerprint != manifest['source_config_fingerprint'] or
            imagination_budget(source_config) != manifest['old_budgets'] or
            imagination_budget(config) != manifest['new_budgets']):
        raise ValueError('budget continuation source configuration or recorded budgets differ')
    protocol = json.loads((output / 'source_protocol.json').read_text())
    if (protocol['source']['sha256'] != pretrained_source_hash or
            manifest['pretrained_source_sha256'] != pretrained_source_hash):
        raise ValueError('budget continuation pretrained ancestry differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    _validate_online(source, source_config)
    progress = _validate_online(initial, config)
    if source.next_unit != manifest['global_completed_episodes'] or initial.next_unit != source.next_unit:
        raise ValueError('budget continuation changed the global episode cursor')
    _assert_equal(source.training_state, initial.training_state)
    if manifest['source_counters'] != dict(online_episodes_completed=source.next_unit,
            next_environment_episode=source.training_state['next_environment_episode']):
        raise ValueError('budget continuation recorded counters differ')
    frozen_progress = json.loads((output / 'continuation_pilot_progress.json').read_text())
    if frozen_progress != progress:
        raise ValueError('budget continuation initial progress differs from checkpoint')
    expected_protocol = rewrite_protocol(protocol, config, manifest)
    frozen_protocol = json.loads((output / 'continuation_pilot_protocol.json').read_text())
    if frozen_protocol != expected_protocol:
        raise ValueError('budget continuation rewritten protocol differs')
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    if latest.phase is CheckpointPhase.COMPLETED:
        latest_progress = latest.training_state['metrics'].get('online_pilot_progress', {})
        if (latest.next_unit != 0 or
                len(latest_progress.get('online', [])) != config.run.online_training_episodes or
                latest.training_state['metrics'].get('online_episodes_completed') != config.run.online_training_episodes or
                latest.training_state['next_environment_episode'] != config.run.online_training_episodes):
            raise ValueError('budget continuation completed checkpoint has inconsistent counters')
    else:
        _validate_online(latest, config, minimum_cursor=source.next_unit)
    return manifest


def _copy_history(source_dir, temporary, *, online_cursor):
    """Freeze all history, but resume only checkpoint-committed online archives.

    SUMO writes its archive before the actor–critic update commits the episode;
    archives at or after the checkpoint cursor must be collected again using
    the new budget. Their original bytes remain in the immutable history.
    """
    copied = {}
    for name in ('before_evaluation', 'after_evaluation', 'milestones', 'episodes'):
        source = source_dir / name
        if not source.is_dir():
            continue
        frozen = temporary / 'source_history' / name
        shutil.copytree(source, frozen)
        def skip_uncommitted_online(directory, names):
            if name != 'episodes' or Path(directory) != frozen / 'online':
                return []
            skipped = []
            for filename in names:
                match = re.fullmatch(r'episode_([0-9]+)\.npz', filename)
                if (match and int(match.group(1)) >= online_cursor and
                        (Path(directory) / filename).is_file()):
                    skipped.append(filename)
            return skipped
        shutil.copytree(frozen, temporary / name, ignore=skip_uncommitted_online)
        for path in sorted(frozen.rglob('*')):
            if path.is_file():
                copied[str(path.relative_to(temporary))] = sha256_file(path)
    for name in ('pilot_metrics.jsonl', 'pilot_progress.json', 'effective_run_budget.json',
                 'offline_manifest.json', 'codec_validation_manifest.json'):
        if (source_dir / name).is_file():
            frozen = temporary / 'source_history' / name
            frozen.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / name, frozen)
            copied[str(frozen.relative_to(temporary))] = sha256_file(frozen)
            if name != 'pilot_progress.json':
                shutil.copy2(frozen, temporary / name)
    return copied


def migrate(source_dir, output_dir, *, device=None, passes=None, time_budget_seconds=None,
            kind=BUDGET_KIND, samples=None, precision=None):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == source_dir:
        raise ValueError('budget continuation requires a new absent output directory')
    source_config = load_config(source_dir / 'run_config.toml')
    if precision is not None and kind != PRECISION_KIND:
        raise ValueError('only precision continuation accepts an imagination precision override')
    if kind == BUDGET_KIND:
        if samples is not None:
            raise ValueError('budget continuation cannot change disagreement samples')
        passes = 2 if passes is None else passes
        time_budget_seconds = 60. if time_budget_seconds is None else time_budget_seconds
    elif kind == SAMPLES_KIND:
        passes = source_config.resources.imagination_passes_per_episode if passes is None else passes
        time_budget_seconds = (source_config.resources.imagination_time_budget_seconds
                               if time_budget_seconds is None else time_budget_seconds)
        samples = 2 if samples is None else samples
    elif kind == PRECISION_KIND:
        passes = source_config.resources.imagination_passes_per_episode if passes is None else passes
        time_budget_seconds = (source_config.resources.imagination_time_budget_seconds
                               if time_budget_seconds is None else time_budget_seconds)
        precision = 'bf16' if precision is None else precision
    else:
        raise ValueError(f'unsupported imagination continuation kind: {kind!r}')
    config = replace(source_config,
        run=replace(source_config.run, output_dir=output, device=device or source_config.run.device),
        method=replace(source_config.method, disagreement_samples=(
            source_config.method.disagreement_samples if samples is None else samples)),
        resources=replace(source_config.resources, imagination_passes_per_episode=passes,
                          imagination_time_budget_seconds=time_budget_seconds,
                          world_model_imagination_precision=(
                              source_config.resources.world_model_imagination_precision
                              if precision is None else precision)))
    _validate_transfer(source_config, config, kind)
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        # A single atomic checkpoint determines all durable counters and progress.
        shutil.copy2(latest_checkpoint_path(source_dir), temporary / 'source_checkpoint.pt')
        shutil.copy2(source_dir / 'run_config.toml', temporary / 'source_config.toml')
        shutil.copy2(source_dir / 'pilot_protocol.json', temporary / 'source_protocol.json')
        raw = torch.load(temporary / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
        source = _read_frozen(temporary / 'source_checkpoint.pt', source_config, raw['input_hash'])
        progress = _validate_online(source, source_config)
        protocol = json.loads((temporary / 'source_protocol.json').read_text())
        if (protocol.get('online_episode_budget') != config.run.online_training_episodes or
                protocol['source'].get('input_hash', source.input_hash) != source.input_hash):
            raise ValueError('budget continuation source protocol episode budget or input identity differs')
        old_budgets = dict(protocol.get('budgets', {}))
        old_budgets.setdefault('imagination_time_budget_seconds', 0.)
        if old_budgets != imagination_budget(source_config):
            raise ValueError('budget continuation source protocol budgets differ from source configuration')
        if (sha256_file(source_dir / 'run_config.toml') != sha256_file(temporary / 'source_config.toml') or
                load_config(temporary / 'source_config.toml').semantic_fingerprint != source_config.semantic_fingerprint):
            raise ValueError('budget continuation source configuration changed during freezing')
        export_run_config(config, temporary / 'run_config.toml')
        raw['config_fingerprint'] = config.semantic_fingerprint
        torch.save(raw, temporary / 'continuation_checkpoint.pt')
        latest_checkpoint_path(temporary).parent.mkdir(parents=True)
        shutil.copy2(temporary / 'continuation_checkpoint.pt', latest_checkpoint_path(temporary))
        copied = _copy_history(source_dir, temporary, online_cursor=source.next_unit)
        atomic_json(temporary / 'continuation_pilot_progress.json', progress)
        files = ('source_checkpoint.pt', 'source_config.toml', 'source_protocol.json',
                 'continuation_checkpoint.pt', 'continuation_pilot_progress.json')
        hashes = {filename: sha256_file(temporary / filename) for filename in files}
        hashes.update(copied)
        manifest = dict(schema_version=1, kind=kind,
            source_output=str(source_dir), target_output=str(output),
            source_checkpoint_sha256=hashes['source_checkpoint.pt'],
            source_config_sha256=hashes['source_config.toml'],
            source_protocol_sha256=hashes['source_protocol.json'],
            target_initial_checkpoint_sha256=hashes['continuation_checkpoint.pt'],
            source_config_fingerprint=source_config.semantic_fingerprint,
            target_config_fingerprint=config.semantic_fingerprint, input_hash=source.input_hash,
            pretrained_source_sha256=protocol['source']['sha256'], global_completed_episodes=source.next_unit,
            source_counters=dict(online_episodes_completed=source.next_unit,
                                 next_environment_episode=source.training_state['next_environment_episode']),
            old_budgets=imagination_budget(source_config), new_budgets=imagination_budget(config),
            frozen_artifacts=hashes, preserved='models, optimizers, replay, RNG, metrics, global cursor and evaluation history')
        if kind == SAMPLES_KIND:
            manifest.update(old_disagreement_samples=source_config.method.disagreement_samples,
                            new_disagreement_samples=config.method.disagreement_samples)
        elif kind == PRECISION_KIND:
            manifest.update(old_imagination_precision=source_config.resources.world_model_imagination_precision,
                            new_imagination_precision=config.resources.world_model_imagination_precision)
        atomic_json(temporary / 'imagination_budget_migration.json', manifest)
        new_protocol = rewrite_protocol(protocol, config, manifest)
        atomic_json(temporary / 'continuation_pilot_protocol.json', new_protocol)
        atomic_json(temporary / 'pilot_protocol.json', new_protocol)
        atomic_json(temporary / 'pilot_progress.json', copy.deepcopy(progress))
        restored = _read_frozen(temporary / 'continuation_checkpoint.pt', config, source.input_hash)
        _assert_equal(source.training_state, restored.training_state)
        temporary.rename(output)
        validate_budget_continuation(output, config, source.input_hash, protocol['source']['sha256'])
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device')
    parser.add_argument('--passes', type=int)
    parser.add_argument('--kind', choices=(BUDGET_KIND, SAMPLES_KIND, PRECISION_KIND), default=BUDGET_KIND)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--precision', choices=('bf16',))
    parser.add_argument('--time-budget-seconds', type=float)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    print(json.dumps(migrate(args.source_dir, args.output_dir, device=args.device, passes=args.passes,
                             time_budget_seconds=args.time_budget_seconds, kind=args.kind, samples=args.samples,
                             precision=args.precision),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
