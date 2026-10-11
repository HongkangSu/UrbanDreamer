"""Continue online training with entropy in the population-normalized reward units."""
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
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def entropy_protocol():
    return dict(mode='population', base_coefficient=.002,
                coefficient='base/(episode_population*max(global_advantage_std,1))')


def validate_entropy_transfer_config(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        snapshot['method'].pop('actor_entropy_mode')
        for key in ('output_dir', 'device'):
            snapshot['run'].pop(key)
    if before != after:
        raise ValueError('entropy continuation may change only actor entropy mode, output directory and device')
    if source.method.actor_entropy_mode != 'legacy' or target.method.actor_entropy_mode != 'population':
        raise ValueError('entropy continuation requires legacy to population mode')


def _validate_cursor(checkpoint):
    state = checkpoint.training_state
    progress = state['metrics'].get('online_pilot_progress')
    if not isinstance(progress, dict) or not isinstance(progress.get('online'), list):
        raise ValueError('entropy continuation requires checkpoint-authoritative pilot progress')
    count = len(progress['online'])
    if (state['metrics'].get('online_episodes_completed') != count
            or state['next_environment_episode'] != count
            or (checkpoint.phase is not CheckpointPhase.COMPLETED and checkpoint.next_unit != count)):
        raise ValueError('entropy continuation online counters differ from durable progress')
    return progress


def validate_entropy_continuation(output_dir, config, input_hash, pretrained_source_hash):
    """Validate immutable ancestry, exact continuation state and an advancing latest file."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'entropy_migration.json').read_text())
    if (manifest.get('schema_version') != 1
            or manifest.get('kind') != 'population_entropy_continuation'
            or manifest['target_output'] != str(output)
            or manifest['target_config_fingerprint'] != config.semantic_fingerprint
            or manifest['input_hash'] != input_hash
            or manifest['actor_entropy'] != entropy_protocol()):
        raise ValueError('entropy continuation target identity differs')
    for filename, key in (
        ('source_checkpoint.pt', 'source_checkpoint_sha256'),
        ('source_config.toml', 'source_config_sha256'),
        ('source_protocol.json', 'source_protocol_sha256'),
        ('continuation_checkpoint.pt', 'target_initial_checkpoint_sha256'),
    ):
        if sha256_file(output / filename) != manifest[key]:
            raise ValueError(f'entropy continuation frozen {filename} hash differs')
    for filename, expected in manifest['copied_artifacts'].items():
        if sha256_file(output / filename) != expected:
            raise ValueError(f'entropy continuation ancestry {filename} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_entropy_transfer_config(source_config, config)
    if source_config.semantic_fingerprint != manifest['source_config_fingerprint']:
        raise ValueError('entropy continuation source configuration differs')
    protocol = json.loads((output / 'source_protocol.json').read_text())
    if protocol['source']['sha256'] != pretrained_source_hash:
        raise ValueError('entropy continuation pretrained ancestry differs')
    if (protocol['runtime_config_fingerprint'] != source_config.semantic_fingerprint
            or protocol['online_episode_budget'] != config.run.online_training_episodes):
        raise ValueError('entropy continuation source protocol differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    if (source.phase is not CheckpointPhase.ONLINE_TRAINING or initial.phase != source.phase
            or source.next_unit != manifest['global_completed_episodes']
            or initial.next_unit != source.next_unit
            or source.next_unit >= config.run.online_training_episodes):
        raise ValueError('entropy continuation initial global cursor differs')
    _validate_cursor(source)
    _validate_cursor(initial)
    _assert_equal(source.training_state, initial.training_state)
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    latest_progress = _validate_cursor(latest)
    if len(latest_progress['online']) < source.next_unit:
        raise ValueError('entropy continuation latest checkpoint moved backwards')
    return manifest


def migrate(source_dir, output_dir, *, device=None):
    """Create a fresh output; change only entropy semantics in the checkpoint header."""
    source_dir, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == source_dir:
        raise ValueError('entropy continuation requires a new absent output directory')
    source_config = load_config(source_dir / 'run_config.toml')
    config = replace(source_config,
        run=replace(source_config.run, output_dir=output, device=device or source_config.run.device),
        method=replace(source_config.method, actor_entropy_mode='population'))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    validate_entropy_transfer_config(source_config, config)
    output.mkdir(parents=True)
    shutil.copy2(source_dir / 'run_config.toml', output / 'source_config.toml')
    shutil.copy2(source_dir / 'pilot_protocol.json', output / 'source_protocol.json')
    # Checkpoints are published atomically. Freeze one file and derive all progress
    # from its content, rather than the independently written progress JSON.
    shutil.copy2(latest_checkpoint_path(source_dir), output / 'source_checkpoint.pt')
    raw = torch.load(output / 'source_checkpoint.pt', map_location='cpu', weights_only=True)
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, raw['input_hash'])
    if source.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('entropy continuation requires an active online checkpoint')
    progress = _validate_cursor(source)
    if source.next_unit >= config.run.online_training_episodes:
        raise ValueError('entropy continuation requires remaining online episodes')
    export_run_config(config, output / 'run_config.toml')
    raw['config_fingerprint'] = config.semantic_fingerprint
    torch.save(raw, output / 'continuation_checkpoint.pt')
    latest_checkpoint_path(output).parent.mkdir(parents=True)
    shutil.copy2(output / 'continuation_checkpoint.pt', latest_checkpoint_path(output))
    copied = {}
    for name in ('offline_manifest.json', 'codec_validation_manifest.json',
                 'calibrated_noincident_initialization.json', 'meso_initialization.json',
                 'initialization_checkpoint.pt'):
        if (source_dir / name).is_file():
            shutil.copy2(source_dir / name, output / name)
            copied[name] = sha256_file(output / name)
    manifest = dict(schema_version=1, kind='population_entropy_continuation',
        source_output=str(source_dir), target_output=str(output),
        source_checkpoint_sha256=sha256_file(output / 'source_checkpoint.pt'),
        source_config_sha256=sha256_file(output / 'source_config.toml'),
        source_protocol_sha256=sha256_file(output / 'source_protocol.json'),
        target_initial_checkpoint_sha256=sha256_file(output / 'continuation_checkpoint.pt'),
        source_config_fingerprint=source_config.semantic_fingerprint,
        target_config_fingerprint=config.semantic_fingerprint, input_hash=source.input_hash,
        global_completed_episodes=source.next_unit, copied_artifacts=copied,
        actor_entropy=entropy_protocol(),
        preserved='all training tensors, optimizers, replay, RNG, counters, progress and evaluations')
    atomic_json(output / 'entropy_migration.json', manifest)
    protocol = json.loads((output / 'source_protocol.json').read_text())
    protocol.update(runtime_config_fingerprint=config.semantic_fingerprint, device=config.run.device,
                    config_semantics_unchanged=False, actor_entropy=entropy_protocol(), entropy_migration=manifest)
    atomic_json(output / 'pilot_protocol.json', protocol)
    atomic_json(output / 'pilot_progress.json', copy.deepcopy(progress))
    validate_entropy_continuation(output, config, source.input_hash, protocol['source']['sha256'])
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
