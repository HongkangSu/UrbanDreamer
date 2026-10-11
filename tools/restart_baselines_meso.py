#!/usr/bin/env python3
"""Prepare and launch fresh mesoscopic baselines after an archived dataset stop."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_city_baseline_dataset import (
    CITIES, METHODS, atomic_json, build_runs, now, snapshot,
)
from tools.stop_city_dataset import command


def owned_processes(root):
    encoded = str(root).encode()
    return [int(path.name) for path in Path('/proc').iterdir()
            if path.name.isdecimal() and int(path.name) != os.getpid() and any(
                arg == encoded or arg.startswith(encoded + b'/')
                for arg in command(int(path.name)))]


def prepare(args):
    previous, root = args.previous_root.resolve(), args.output_root.resolve()
    if not (previous / 'STOPPED_BY_USER').is_file() or owned_processes(previous):
        raise ValueError('previous dataset must be stopped without owned processes')
    preserved = json.loads((previous / 'preservation_and_stop.json').read_text())
    if preserved['inventory_errors'] or preserved['remaining_owned_processes']:
        raise ValueError('previous preservation inventory has unresolved errors')
    if root.exists() and any(root.iterdir()):
        raise ValueError('refusing to overwrite an existing mesoscopic dataset')
    if shutil.disk_usage(root.parent).free < 10 * 1024 ** 3:
        raise ValueError('insufficient free disk for dataset collection')
    prior_manifest = json.loads((previous / 'manifest.json').read_text())
    device_map = dict(item.split('=', 1) for item in args.device)
    expected = {f'{city}_{method}' for city in CITIES for method in METHODS}
    if set(device_map) != expected:
        raise ValueError('explicit device assignment required for every city/method')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'processes'):
        (root / name).mkdir()
    source = snapshot(root)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    settings = argparse.Namespace(cities=list(CITIES), methods=list(METHODS),
        devices=[], episodes=5000, seed=42, checkpoint_every=1, eval_every=100,
        eval_seeds=3, forbid_revisit=args.forbid_revisit,
        global_reward_only=bool(getattr(args, 'global_reward_only', False)), sp_prior_weight=args.sp_prior_weight,
        sp_prior_temperature=args.sp_prior_temperature,
        sp_prior_hold_episodes=args.sp_prior_hold_episodes,
        sp_prior_off_episode=args.sp_prior_off_episode)
    runs = build_runs(root, source, settings, stamp)
    for entry in runs:
        physics = entry['simulation_settings']
        if not (physics['traffic_model'] == 'meso'
                and physics['session_protocol'] == 'event'
                and physics['backend'] == 'libsumo'
                and physics['observation_step_seconds'] == 1.):
            raise ValueError(f'incompatible mesoscopic profile: {entry["id"]}')
        entry['device'] = device_map[entry['id']]
        entry['command'][entry['command'].index('--device') + 1] = entry['device']
        entry['initialization'] = 'fresh model, optimizer and replay; no previous checkpoint reuse'
    checkpoint_inventory = []
    for entry in prior_manifest['runs']:
        output = Path(entry['output_dir'])
        for path in sorted(output.glob('*.pt')):
            with zipfile.ZipFile(path) as archive:
                members = [dict(name=item.filename, bytes=item.file_size, crc32=item.CRC)
                           for item in archive.infolist()]
            checkpoint_inventory.append(dict(run=entry['id'], path=str(path),
                bytes=path.stat().st_size, members=members))
    atomic_json(previous / 'preserved_checkpoint_inventory.json', checkpoint_inventory)
    migration = dict(previous_root=str(previous), previous_preservation=str(
        previous / 'preservation_and_stop.json'),
        previous_archive_inventory=str(previous / 'preserved_episode_inventory.jsonl'),
        previous_checkpoint_inventory=str(previous / 'preserved_checkpoint_inventory.json'),
        archived_previous_episodes=preserved['total_archived'],
        initialization='fresh model, optimizer and replay; no previous checkpoint reuse',
        training_budget='5000 additional mesoscopic episodes per independent run',
        physics_changed_from=prior_manifest.get('traffic_model', 'micro'),
        physics_changed_to='meso', prepared_at=now(),
        previous_forbid_revisit=prior_manifest.get('forbid_revisit', False),
        forbid_revisit=args.forbid_revisit)
    policy_prior = dict(prior_manifest['policy_prior'])
    policy_prior.update(enabled=args.sp_prior_weight > 0, initial_weight=args.sp_prior_weight,
        temperature_seconds=args.sp_prior_temperature, hold_episodes=args.sp_prior_hold_episodes,
        off_episode=args.sp_prior_off_episode,
        best_selection='zero_prior' if args.sp_prior_weight > 0 else 'deterministic',
        cost='model-specific junction_time + candidate_edge_free_flow_time + downstream_remaining_time; meso junction_time=0')
    revisit_policy = dict(prior_manifest['revisit_policy'], enabled=args.forbid_revisit)
    manifest = {key: value for key, value in prior_manifest.items()
                if key not in {'runs', 'source_revisions', 'comparison_reference'}}
    manifest.update(created_at=now(), launch_status='prepared', source_snapshot=str(source),
        simulation_settings_by_city={entry['city']: entry['simulation_settings'] for entry in runs},
        episode_budget=5000, total_training_episodes=75000, seed=42,
        checkpoint_every=1, eval_every=100, eval_seeds=3,
        min_free_gb=10., max_restarts=3, traffic_model='meso', macro_step_seconds=30.,
        migration=migration, runs=runs, forbid_revisit=args.forbid_revisit,
        revisit_policy=revisit_policy, policy_prior=policy_prior,
        reward_mode=('global_only' if settings.global_reward_only else 'global_with_individual_timeout'),
        monitor_session=f'uddata_monitor_{stamp}', compactor_session=f'uddata_compact_{stamp}',
        notes=['15 independent fresh mesoscopic runs, 5000 episodes each',
               'Previous archives and checkpoints preserved in place',
               'Actual SUMO event collection, libsumo, 1-second observations, 30-second world-model intervals'])
    atomic_json(root / 'manifest.json', manifest)
    atomic_json(root / 'migration.json', migration)
    prior_manifest['superseded_by'] = dict(root=str(root), migration_record=str(root / 'migration.json'),
                                          reason='user-authorized fresh mesoscopic protocol replacement')
    atomic_json(previous / 'manifest.json', prior_manifest)
    print(json.dumps(dict(prepared=str(root), runs=len(runs), devices=device_map)), flush=True)


def launch(args):
    root = args.output_root.resolve()
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['launch_status'] != 'prepared':
        raise ValueError('launch already attempted; inspect manifest and processes before recovery')
    if owned_processes(Path(manifest['migration']['previous_root'])):
        raise ValueError('previous dataset has running owned processes')
    source = Path(manifest['source_snapshot'])
    hashes = json.loads((root / 'source_hashes.json').read_text())
    for relative, expected in hashes.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed: {relative}')
    manifest.update(launch_status='launching', launched_at=now())
    atomic_json(path, manifest)
    for entry in manifest['runs']:
        output = Path(entry['output_dir'])
        if output.exists() and any(output.iterdir()):
            raise ValueError(f'fresh output directory is not empty: {output}')
        argv = [sys.executable, '-u', str(source / 'tools/run_city_baseline_dataset.py'),
                'supervise', '--output-root', str(root), '--run-id', entry['id']]
        subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', entry['tmux_session'],
                        '-c', str(source), shlex.join(argv)], check=True)
        entry['launched'] = True
        atomic_json(path, manifest)
        print(json.dumps(dict(launched=entry['id'], device=entry['device'])), flush=True)
    controllers = (
        (manifest['monitor_session'], [sys.executable, '-u',
            str(source / 'tools/run_city_baseline_dataset.py'), 'watch', '--output-root', str(root)]),
        (manifest['compactor_session'], [sys.executable, '-u',
            str(source / 'tools/compact_episode_archives.py'), '--root', str(root), '--workers', '4']),
    )
    for session, argv in controllers:
        shell = shlex.join(argv)
        if session == manifest['compactor_session']:
            shell += ' > ' + shlex.quote(str(root / 'compaction.log')) + ' 2>&1'
        subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', session,
                        '-c', str(source), shell], check=True)
    manifest['launch_status'] = 'launched'
    atomic_json(path, manifest)
    print(json.dumps(dict(root=str(root), launch_status='launched', runs=len(manifest['runs']))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    parser.add_argument('--previous-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', action='append', default=[], help='Explicit run_id=cpu/cuda:N assignment')
    parser.add_argument('--forbid-revisit', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sp-prior-weight', type=float, default=0.,
                        help='Initial shortest-path prior weight (default: 0; pass 1 to enable)')
    parser.add_argument('--sp-prior-temperature', type=float, default=30.)
    parser.add_argument('--sp-prior-hold-episodes', type=int, default=200)
    parser.add_argument('--sp-prior-off-episode', type=int, default=1000)
    parser.add_argument('--global-reward-only', action='store_true')
    args = parser.parse_args()
    if args.forbid_revisit and args.sp_prior_weight != 0.:
        parser.error('forbid-revisit requires zero shortest-path prior')
    if (not math.isfinite(args.sp_prior_weight) or args.sp_prior_weight < 0
            or not math.isfinite(args.sp_prior_temperature) or args.sp_prior_temperature <= 0
            or args.sp_prior_hold_episodes < 0 or args.sp_prior_off_episode <= args.sp_prior_hold_episodes):
        parser.error('invalid shortest-path prior schedule')
    if args.mode == 'prepare':
        if args.previous_root is None:
            parser.error('prepare requires --previous-root')
        prepare(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
