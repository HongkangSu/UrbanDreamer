"""Change only an online run's evaluation protocol at a committed boundary."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.change_online_training_profile import archive_prior_evaluations
from tools.migrate_mainline_prior import _assert_equal
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.training import evaluation_protocol
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import load_config


def revised_progress(progress, boundary, policy):
    if len(progress['online']) != boundary:
        raise ValueError('checkpoint and training history differ')
    result = copy.deepcopy(progress)
    result.update(before=[], after=[], milestones=[], phase='before_evaluation',
                  evaluation_reference_online_episodes=boundary,
                  evaluation_policy=policy['archive_policy'])
    return result


def change(output, source, *, temperature=None):
    output, source = Path(output).resolve(), Path(source).resolve()
    status = json.loads((output / 'runtime_status.json').read_text())
    if status['phase'] != 'stopped_at_checkpoint_boundary':
        raise ValueError('evaluation change requires a clean checkpoint boundary stop')
    proc = Path('/proc') / str(status['pid']) / 'stat'
    if proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
        raise ValueError('refusing to change a live training run')
    if not (source / 'tools/run_bounded_online_continuation.py').is_file():
        raise ValueError('missing frozen execution source')
    config = load_config(output / 'run_config.toml')
    raw = torch.load(output / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
    checkpoint = load_checkpoint(output_dir=output,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=raw['input_hash'])
    del raw
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('active online checkpoint required')
    state = checkpoint.training_state
    old_progress = state['metrics']['online_pilot_progress']
    policy = evaluation_protocol(temperature, prior_weight=config.method.sp_prior_evaluation_weight,
                                 demand_seed=config.method.evaluation_demand_seed)
    protocol = json.loads((output / 'pilot_protocol.json').read_text())
    old_policy = copy.deepcopy(protocol.get('evaluation'))
    protocol.update(evaluation=policy, evaluation_policy=policy['archive_policy'], source_root=str(source))
    progress = revised_progress(old_progress, checkpoint.next_unit, policy)
    backup = output / 'evaluation_changes' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup.mkdir(parents=True, exist_ok=False)
    os.link(output / 'checkpoints/latest.pt', backup / 'checkpoint.pt')
    for name in ('run_config.toml', 'pilot_protocol.json', 'pilot_progress.json',
                 'runtime_status.json', 'launch.json', 'pilot_metrics.jsonl'):
        path = output / name
        if path.exists():
            shutil.copy2(path, backup / name)
    moved = archive_prior_evaluations(output, backup)
    state['metrics']['online_pilot_progress'] = progress
    path = save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint,
        input_hash=checkpoint.input_hash, phase=checkpoint.phase, next_unit=checkpoint.next_unit,
        training_state=state)
    restored = load_checkpoint(output_dir=output,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=checkpoint.input_hash)
    for key in ('models', 'optimizers', 'replay', 'rng', 'reward_contract', 'next_environment_episode'):
        _assert_equal(state[key], restored.training_state[key])
    _assert_equal(old_progress['online'], restored.training_state['metrics']['online_pilot_progress']['online'])
    atomic_json(output / 'pilot_protocol.json', protocol)
    atomic_json(output / 'pilot_progress.json', progress)
    proof = dict(kind='evaluation_policy_only_change', boundary=checkpoint.next_unit,
        before=old_policy, after=policy, training_config_fingerprint=config.semantic_fingerprint,
        checkpoint_before_sha256=sha256_file(backup / 'checkpoint.pt'),
        checkpoint_after_sha256=sha256_file(path), backup=str(backup),
        archived_evaluations=moved, source_root=str(source),
        preserved=['models', 'optimizers', 'replay', 'rng', 'online_history', 'online_episode_clock'],
        evaluation_reference_episode=checkpoint.next_unit)
    atomic_json(backup / 'change.json', proof)
    atomic_json(output / 'latest_evaluation_change.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--policy', choices=('argmax', 'sampled'), default='argmax')
    parser.add_argument('--temperature', type=float, default=.2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(change(args.run_dir, args.source_dir,
                           temperature=None if args.policy == 'argmax' else args.temperature), indent=2))


if __name__ == '__main__':
    main()
