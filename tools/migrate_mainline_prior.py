"""Explicit, lossless checkpoint continuation when enabling the policy routing prior."""
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

from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.workflow.checkpoint import (
    CheckpointPhase, _validate_loaded_payload, latest_checkpoint_path, load_checkpoint,
)
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config

PRIOR_FIELDS = ('sp_prior_weight', 'sp_prior_temperature', 'sp_prior_hold_episodes', 'sp_prior_off_episode')


def validate_prior_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for key in PRIOR_FIELDS:
            snapshot['method'].pop(key)
        for key in ('output_dir', 'device'):
            snapshot['run'].pop(key)
    if before != after:
        raise ValueError('prior continuation may change only policy prior, output directory and device')
    if source.method.sp_prior_weight != 0 or target.method.sp_prior_weight <= 0:
        raise ValueError('prior continuation requires a previously disabled and now enabled prior')


def prior_protocol(config):
    method = config.method
    return dict(initial_weight=method.sp_prior_weight,
                temperature_seconds=method.sp_prior_temperature,
                hold_episodes=method.sp_prior_hold_episodes,
                off_episode=method.sp_prior_off_episode,
                schedule='cosine_global_one_based_episode', evaluation_policy='zero_prior_argmax')


def _read_frozen(path, config, input_hash):
    raw = torch.load(path, map_location='cpu', weights_only=True)
    return _validate_loaded_payload(payload=raw, path=path,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=input_hash)


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError('prior continuation changed a checkpoint tensor')
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError('prior continuation changed checkpoint mapping keys')
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('prior continuation changed a checkpoint sequence')
        for a, b in zip(left, right):
            _assert_equal(a, b)
    elif left != right:
        raise ValueError('prior continuation changed checkpoint state')


def validate_prior_continuation(output_dir, config, input_hash, pretrained_source_hash):
    """Validate immutable ancestry and the initial continuation; latest may advance."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'policy_prior_migration.json').read_text())
    if (manifest['target_output'] != str(output)
            or manifest['target_config_fingerprint'] != config.semantic_fingerprint
            or manifest['input_hash'] != input_hash):
        raise ValueError('prior continuation target identity differs')
    for filename, key in (
        ('source_checkpoint.pt', 'source_checkpoint_sha256'),
        ('source_config.toml', 'source_config_sha256'),
        ('source_protocol.json', 'source_protocol_sha256'),
        ('continuation_checkpoint.pt', 'target_initial_checkpoint_sha256'),
    ):
        if sha256_file(output / filename) != manifest[key]:
            raise ValueError(f'prior continuation frozen {filename} hash differs')
    for filename, expected in manifest.get('copied_artifacts', {}).items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'prior continuation ancestry {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_prior_transfer_config(source_config, config)
    if source_config.semantic_fingerprint != manifest['source_config_fingerprint']:
        raise ValueError('prior continuation source configuration differs')
    protocol = json.loads((output / 'source_protocol.json').read_text())
    if protocol['source']['sha256'] != pretrained_source_hash:
        raise ValueError('prior continuation pretrained ancestry differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    if (source.phase is not CheckpointPhase.ONLINE_TRAINING or initial.phase != source.phase
            or source.next_unit != manifest['global_completed_episodes'] or initial.next_unit != source.next_unit):
        raise ValueError('prior continuation initial global cursor differs')
    _assert_equal(source.training_state, initial.training_state)
    progress = initial.training_state['metrics']['online_pilot_progress']
    if len(progress['online']) != initial.next_unit:
        raise ValueError('prior continuation durable progress differs from global cursor')
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    latest_progress = latest.training_state['metrics']['online_pilot_progress']
    if len(latest_progress['online']) < source.next_unit:
        raise ValueError('prior continuation latest checkpoint moved backwards')
    if latest.phase is not CheckpointPhase.COMPLETED and latest.next_unit != len(latest_progress['online']):
        raise ValueError('prior continuation latest cursor differs from durable progress')
    return manifest


def migrate(source_dir, output_dir, *, device=None, weight=1., temperature=30., hold_episodes=200, off_episode=1000):
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == source_dir:
        raise ValueError('prior continuation requires a new absent output directory')
    source_config = load_config(source_dir / 'run_config.toml')
    config = replace(source_config,
        run=replace(source_config.run, output_dir=output, device=device or source_config.run.device),
        method=replace(source_config.method, sp_prior_weight=weight, sp_prior_temperature=temperature,
                       sp_prior_hold_episodes=hold_episodes, sp_prior_off_episode=off_episode))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    validate_prior_transfer_config(source_config, config)
    output.mkdir(parents=True)
    shutil.copy2(source_dir / 'run_config.toml', output / 'source_config.toml')
    shutil.copy2(source_dir / 'pilot_protocol.json', output / 'source_protocol.json')
    # Copy an atomic checkpoint file once; all cursors come from this frozen file,
    # never from a progress JSON which might belong to a different checkpoint.
    shutil.copy2(latest_checkpoint_path(source_dir), output / 'source_checkpoint.pt')
    raw = torch.load(output / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, raw['input_hash'])
    if source.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('prior continuation requires an active online checkpoint')
    progress = source.training_state['metrics'].get('online_pilot_progress')
    if not progress or len(progress['online']) != source.next_unit:
        raise ValueError('prior continuation requires checkpoint-authoritative pilot progress')
    if source.next_unit >= config.run.online_training_episodes:
        raise ValueError('prior continuation requires remaining online episodes')
    export_run_config(config, output / 'run_config.toml')
    raw['config_fingerprint'] = config.semantic_fingerprint
    torch.save(raw, output / 'continuation_checkpoint.pt')
    latest_checkpoint_path(output).parent.mkdir(parents=True)
    shutil.copy2(output / 'continuation_checkpoint.pt', latest_checkpoint_path(output))
    copied = {}
    for name in ('offline_manifest.json', 'codec_validation_manifest.json',
                 'meso_initialization.json', 'initialization_checkpoint.pt'):
        if (source_dir / name).is_file():
            shutil.copy2(source_dir / name, output / name)
            copied[name] = sha256_file(output / name)
    manifest = dict(schema_version=1, kind='enable_shortest_path_prior_continuation',
        source_output=str(source_dir), target_output=str(output),
        source_checkpoint_sha256=sha256_file(output / 'source_checkpoint.pt'),
        source_config_sha256=sha256_file(output / 'source_config.toml'),
        source_protocol_sha256=sha256_file(output / 'source_protocol.json'),
        target_initial_checkpoint_sha256=sha256_file(output / 'continuation_checkpoint.pt'),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint, input_hash=source.input_hash,
        global_completed_episodes=source.next_unit, copied_artifacts=copied,
        shortest_path_prior=prior_protocol(config),
        preserved='all training tensors, optimizers, replay, RNG, counters, progress and evaluations')
    atomic_json(output / 'policy_prior_migration.json', manifest)
    protocol = json.loads((output / 'source_protocol.json').read_text())
    protocol.update(runtime_config_fingerprint=config.semantic_fingerprint, device=config.run.device,
                    config_semantics_unchanged=False, shortest_path_prior=prior_protocol(config),
                    policy_prior_migration=manifest)
    atomic_json(output / 'pilot_protocol.json', protocol)
    atomic_json(output / 'pilot_progress.json', copy.deepcopy(progress))
    validate_prior_continuation(output, config, source.input_hash, protocol['source']['sha256'])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(migrate(args.source_dir, args.output_dir, device=args.device), indent=2))


if __name__ == '__main__':
    main()
