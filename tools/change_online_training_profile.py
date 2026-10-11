"""Record a checkpoint-preserving online rollout/prior configuration change."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, learning_budgets, sha256_file
from urban_dreamer.workflow.config import load_config, _semantic_fingerprint
from urban_dreamer.workflow.checkpoint import load_checkpoint, save_checkpoint, CheckpointPhase


def changed_config(config, *, rollouts, evaluation_weight, hold, off):
    resources = replace(config.resources, imagination_rollouts_per_pass=rollouts)
    method = replace(config.method, sp_prior_evaluation_weight=evaluation_weight,
                     sp_prior_hold_episodes=hold, sp_prior_off_episode=off)
    target = replace(config, resources=resources, method=method)
    return replace(target, semantic_fingerprint=_semantic_fingerprint(**{
        key:getattr(target,key) for key in ('inputs','run','simulation','resources','method','data','codec')}))


def revised_protocol(protocol, config):
    result = copy.deepcopy(protocol)
    result['config_fingerprint'] = config.semantic_fingerprint
    result['budgets'] = learning_budgets(config)
    result['prior'].update(evaluation_weight=config.method.sp_prior_evaluation_weight,
        hold_episodes=config.method.sp_prior_hold_episodes, off_episode=config.method.sp_prior_off_episode)
    return result


def rebase_evaluation(progress, *, boundary):
    result = copy.deepcopy(progress)
    result.update(before=[], after=[], milestones=[], phase='before_evaluation',
                  evaluation_reference_online_episodes=boundary,
                  evaluation_policy='zero_prior_argmax')
    return result


def archive_prior_evaluations(output, backup):
    """Do not reuse filenames whose evaluation-policy provenance has changed."""
    moved = {}
    for name in ('before_evaluation','after_evaluation','milestones'):
        source = output/name
        if source.exists():
            destination = backup/name
            if destination.exists():
                raise FileExistsError(destination)
            source.rename(destination)
            moved[str(source)] = str(destination)
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--evaluation-prior-weight', type=float, default=0.)
    parser.add_argument('--prior-hold', type=int, default=200)
    parser.add_argument('--prior-off', type=int, default=1000)
    args = parser.parse_args()
    output = args.run_dir.resolve()
    status = json.loads((output/'runtime_status.json').read_text())
    if status['phase'] != 'stopped_at_checkpoint_boundary':
        raise ValueError('profile change requires a clean checkpoint-boundary stop')
    pid = status['pid']
    if Path(f'/proc/{pid}').exists():
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0] != 'Z':
            raise ValueError('refusing to change a live run')
    torch.set_num_threads(2)
    old = load_config(output/'run_config.toml')
    raw = torch.load(output/'checkpoints/latest.pt', map_location='cpu', weights_only=True)
    checkpoint = load_checkpoint(output_dir=output, expected_config_fingerprint=old.semantic_fingerprint,
                                 expected_input_hash=raw['input_hash'])
    del raw
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('active online checkpoint required')
    state = checkpoint.training_state
    progress = state['metrics']['online_pilot_progress']
    if checkpoint.next_unit != len(progress['online']):
        raise ValueError('checkpoint/history cursor mismatch')
    target = changed_config(old, rollouts=args.rollouts, evaluation_weight=args.evaluation_prior_weight,
                            hold=args.prior_hold, off=args.prior_off)
    if target.semantic_fingerprint == old.semantic_fingerprint:
        raise ValueError('requested settings are already active')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = output/'profile_changes'/stamp
    backup.mkdir(parents=True, exist_ok=False)
    os.link(output/'checkpoints/latest.pt', backup/'checkpoint.pt')
    for name in ['run_config.toml','pilot_protocol.json','pilot_progress.json','runtime_status.json','launch.json']:
        shutil.copyfile(output/name, backup/name)
    export_run_config(target, backup/'target_config.toml')
    target = load_config(backup/'target_config.toml')
    protocol = revised_protocol(json.loads((backup/'pilot_protocol.json').read_text()), target)
    evaluation_changed = old.method.sp_prior_evaluation_weight != target.method.sp_prior_evaluation_weight
    if evaluation_changed:
        moved_evaluations = archive_prior_evaluations(output, backup)
        progress = rebase_evaluation(progress, boundary=checkpoint.next_unit)
        state['metrics']['online_pilot_progress'] = progress
    else:
        moved_evaluations = {}
    path = save_checkpoint(output_dir=output, config_fingerprint=target.semantic_fingerprint,
        input_hash=checkpoint.input_hash, phase=checkpoint.phase, next_unit=checkpoint.next_unit,
        training_state=state)
    shutil.copyfile(backup/'target_config.toml', output/'run_config.toml')
    atomic_json(output/'pilot_protocol.json', protocol)
    atomic_json(output/'pilot_progress.json', progress)
    record = dict(kind='rollout_and_policy_prior_profile_change', boundary=checkpoint.next_unit,
        before=dict(rollouts=old.resources.imagination_rollouts_per_pass,
            evaluation_prior=old.method.sp_prior_evaluation_weight, hold=old.method.sp_prior_hold_episodes,
            off=old.method.sp_prior_off_episode),
        after=dict(rollouts=target.resources.imagination_rollouts_per_pass,
            evaluation_prior=target.method.sp_prior_evaluation_weight, hold=target.method.sp_prior_hold_episodes,
            off=target.method.sp_prior_off_episode),
        old_checkpoint_sha256=sha256_file(backup/'checkpoint.pt'),
        new_checkpoint_sha256=sha256_file(path), old_fingerprint=old.semantic_fingerprint,
        new_fingerprint=target.semantic_fingerprint, evaluation_rebased=evaluation_changed,
        evaluation_archive_locations=moved_evaluations,
        model_optimizer_replay_rng_preserved=True, backup=str(backup))
    atomic_json(backup/'change.json', record)
    atomic_json(output/'latest_profile_change.json', record)
    print(json.dumps(record,indent=2),flush=True)


if __name__ == '__main__':
    main()
