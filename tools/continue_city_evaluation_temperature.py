"""Continue unfinished city baselines under explicit evaluation, without rewriting checkpoints."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue
from tools.continue_city_entropy_annealing import (
    controller_identity, controller_stopped, copy_once, input_hashes, owned_workers,
    read, require, sha, source_hashes, verify_sources,
)
from urban_dreamer.workflow.config import load_config

OVERLAY_FILES = (
    'tools/train_toy_actor_critic.py', 'tools/train_toy_baseline.py',
    'urban_dreamer/models/policy.py', 'urban_dreamer/learning/baselines/common.py',
    *('urban_dreamer/learning/baselines/' + name + '.py'
      for name in ('gnac', 'ppo', 'dreamer', 'mbpo', 'iql', 'mamba')),
)


def install_actor_temperature(path):
    """Change only sampling in the old actor; retain its original critic and prior."""
    text = Path(path).read_text()
    actor = next(node for node in ast.parse(text).body
                 if isinstance(node, ast.ClassDef) and node.name == 'NextHopActor')
    method = next(node for node in actor.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'sample_action')
    replacement = '''    def sample_action(self, features: torch.Tensor, mask: torch.Tensor, *, deterministic: bool = False,
                      temperature: float = 1.) -> ActionSelection:
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("action temperature must be finite and positive")
        logits = self.forward_logits(features, mask)
        distribution = Categorical(logits=logits if temperature == 1. else logits / temperature)
        selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return ActionSelection(selected, distribution.log_prob(selected), distribution.entropy())
'''
    lines = text.splitlines(keepends=True)
    Path(path).write_text(''.join(lines[:method.lineno - 1]) + replacement + ''.join(lines[method.end_lineno:]))


def evaluation_label(temperature):
    if temperature is None:
        return 'deterministic'
    require(not isinstance(temperature, bool) and math.isfinite(temperature) and temperature > 0,
            'evaluation temperature must be finite and positive')
    return f'sampled_temperature_{temperature:g}'


def continuation_entry(entry, source, root, temperature=None):
    label = evaluation_label(temperature)
    result = deepcopy(entry)
    old_source = Path(entry['command'][2]).resolve().parents[1]
    command = []
    skip = False
    for value in entry['command']:
        if skip:
            skip = False
            continue
        if value == '--evaluation-temperature':
            skip = True
            continue
        if value in ('--evaluation-argmax', '--evaluation-sampling', '--evaluation-sampled'):
            continue
        try:
            relative = Path(value).relative_to(old_source)
        except ValueError:
            command.append(value)
        else:
            command.append(str(source / relative))
    command += (['--evaluation-argmax'] if temperature is None else
                ['--evaluation-sampled', '--evaluation-temperature', str(float(temperature))])
    result.update(command=command, log=str(root / 'logs' / f'{entry["id"]}.log'),
                  state=str(root / 'processes' / f'{entry["id"]}.json'),
                  continuation_from_source=str(old_source),
                  continuation_reason=f'Evaluation policy {label}; training state and episode clock retained',
                  evaluation_policy=label, evaluation_temperature=temperature)
    return result


def prepare(args):
    temperature = None if getattr(args, 'argmax', False) else args.temperature
    label = evaluation_label(temperature)
    old, ledger = read(args.old_root / 'manifest.json'), read(args.old_root / 'queue_state.json')
    controller = controller_identity(args.old_root, ledger)
    require(not args.new_root.exists() or not any(args.new_root.iterdir()), 'new root must be empty')
    args.new_root.mkdir(parents=True, exist_ok=True)
    for name in ('logs', 'processes', 'attempts', 'handoff_original'):
        (args.new_root / name).mkdir()
    sources, revisions, shared = {}, {}, {}
    for city in queue.CITIES:
        original = Path(old.get('source_snapshots_by_city', {}).get(city, old['source_snapshot']))
        if original not in shared:
            destination = (args.new_root / 'source' if original == Path(old['source_snapshot'])
                           else args.new_root / 'sources' / city)
            shutil.copytree(original, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            before = source_hashes(destination)
            overlays = (('tools/train_toy_actor_critic.py', 'tools/train_toy_baseline.py',
                         'urban_dreamer/learning/baselines/common.py')
                        if temperature is None else OVERLAY_FILES)
            for relative in overlays:
                if relative == 'urban_dreamer/models/policy.py':
                    install_actor_temperature(destination / relative)
                else:
                    shutil.copy2(args.overlay_root / relative, destination / relative)
            revisions[str(destination)] = dict(original_source=str(original), original_hashes=before,
                new_hashes=source_hashes(destination), allowed_overlay_files=list(overlays),
                policy_overlay=('Argmax defaults in evaluation entrypoints; all model and training code retained'
                                if temperature is None else
                                'Only NextHopActor.sample_action transplanted; old critic and static prior retained'))
            shared[original] = destination
        destination = shared[original]
        sources[city] = str(destination)
        require(input_hashes(destination, city) == old['input_hashes_by_city'][city], 'city inputs changed')
        original_config = load_config(original / 'configs' / f'{city}.toml')
        new_config = load_config(destination / 'configs' / f'{city}.toml')
        require(original_config.semantic_fingerprint == new_config.semantic_fingerprint,
                'evaluation continuation changed the training configuration')
    manifest = deepcopy(old)
    manifest.update(source_snapshot=str(args.new_root / 'source'), source_snapshots_by_city=sources,
        created_at=queue.now(), launch_status='prepared', continuation_from=str(args.old_root),
        controller_session=f'ud_{label}_{time.strftime("%Y%m%d_%H%M%S", time.gmtime())}'.replace('.', 'p'),
        evaluation_policy=label, evaluation_temperature=temperature,
        evaluation_seeds=[old['seed'] + 10000 + index for index in range(old['eval_seeds'])])
    manifest['runs'] = [deepcopy(entry) if ledger['jobs'][entry['id']]['status'] in queue.TERMINAL
                       else continuation_entry(entry, Path(sources[entry['city']]), args.new_root, temperature)
                       for entry in old['runs']]
    queue.write_json(args.new_root / 'manifest.json', manifest)
    queue.write_json(args.new_root / 'source_hashes.json', source_hashes(args.new_root / 'source'))
    queue.write_json(args.new_root / 'handoff_protocol.json', dict(phase='prepared', prepared_at=queue.now(),
        old_root=str(args.old_root), new_root=str(args.new_root), controller=controller,
        source_revisions=revisions, evaluation_policy=label, evaluation_temperature=temperature,
        checkpoint_policy='Checkpoint bytes unchanged; runner resets old best score after restoring training state',
        policy='Failed/completed jobs stay terminal; retain waiting jobs and durable training cursors'))


def archive_checkpoint(path, destination):
    if not path.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
    digest = sha(path)
    # Identical inodes are already the same bytes; avoid rereading large replay
    # files. Copies on other filesystems still need the second content hash.
    require(os.path.samefile(path, destination) or sha(destination) == digest,
            'checkpoint archive differs from durable checkpoint')
    return dict(path=str(path), backup=str(destination), sha256=digest, checkpoint_rewritten=False)


def continued_job(state, boundary):
    if state['status'] in queue.TERMINAL:
        return deepcopy(state)
    return dict(state, status='queued_recovery' if boundary else 'queued', pid=None,
                process_starttime=None, next_eligible=0., continuation_episode=boundary)


def update_protocol(output, boundary, archive, seeds, temperature=None):
    label = evaluation_label(temperature)
    path = output / 'protocol.json'
    if not path.exists():
        return
    protocol = read(path)
    protocol.update(evaluation_policy='argmax' if temperature is None else 'temperature_sampling',
                    evaluation_temperature=temperature,
                    evaluation_policies=[label], evaluation_seeds=seeds, evaluation_rng_isolated=True,
                    evaluation_continuation=dict(from_episode=boundary, old_protocol_backup=str(archive / 'protocol.json'),
                        historical_metrics=f'Existing evaluation labels retained; new rows use {label}',
                        best_score_reset='Runner resets in memory during resume; old best checkpoint archived'))
    protocol.setdefault('args', {}).update(evaluation_temperature=temperature,
                                            evaluation_argmax=temperature is None,
                                            evaluation_sampling=temperature is not None)
    prior = protocol.get('shortest_path_prior', {})
    for key in ('checkpoint_selection_policy', 'best_selection'):
        if key in prior:
            prior[key] = label
    queue.write_json(path, protocol)


def archive_previous_best(output, backup, previous_label):
    """Keep actual historical identity when the old best has not been replaced."""
    # A protocol can have changed before its first new evaluation. The runner
    # leaves best.pt pointing at the original historical checkpoint until then.
    for historical in output.glob('best_*.pt'):
        if os.path.samefile(backup, historical):
            return historical
    historical = output / f'best_{previous_label}.pt'
    if historical.exists() and sha(historical) != sha(backup):
        historical = output / f'best_{previous_label}_{sha(backup)}.pt'
    if not historical.exists():
        try:
            os.link(backup, historical)
        except OSError:
            shutil.copy2(backup, historical)
    return historical


def apply(args):
    protocol = read(args.new_root / 'handoff_protocol.json')
    permitted = {'controller_frozen', 'migrating'} if args.resume_handoff else {'prepared'}
    require(protocol['phase'] in permitted, 'handoff phase is not eligible')
    require(protocol['old_root'] == str(args.old_root), 'old root changed')
    for source, revision in protocol['source_revisions'].items():
        verify_sources(Path(source), revision['new_hashes'])
        verify_sources(Path(revision['original_source']), revision['original_hashes'])
    old, ledger = read(args.old_root / 'manifest.json'), read(args.old_root / 'queue_state.json')
    require(controller_identity(args.old_root, ledger) == protocol['controller'], 'controller changed')
    manifest = read(args.new_root / 'manifest.json')
    temperature = manifest['evaluation_temperature']
    requested_temperature = None if getattr(args, 'argmax', False) else args.temperature
    require(requested_temperature == temperature, 'prepared evaluation temperature differs from apply')
    previous_label = old.get('evaluation_policy', 'deterministic')
    require(subprocess.run(['/usr/bin/tmux', 'has-session', '-t', manifest['controller_session']],
                           capture_output=True).returncode != 0, 'new controller already exists')
    pid = protocol['controller']['pid']
    if args.resume_handoff:
        require(controller_stopped(pid), 'resume requires the old controller to remain stopped')
    else:
        os.kill(pid, signal.SIGSTOP)
        for _ in range(50):
            if controller_stopped(pid):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('controller did not freeze')
        protocol.update(phase='controller_frozen', frozen_at=queue.now(), stopped_pids=[], continuation_episodes={})
        queue.write_json(args.new_root / 'handoff_protocol.json', protocol)
    ledger = read(args.old_root / 'queue_state.json')
    for name in ('manifest.json', 'queue_state.json', 'status.json', 'admission_overrides.json'):
        copy_once(args.old_root / name, args.new_root / 'handoff_original' / name)
    victims = owned_workers(old['runs'], ledger['jobs'])
    for worker, (state, entry) in victims.items():
        if queue.owned_alive(state, entry):
            os.kill(worker, signal.SIGTERM)
            protocol['stopped_pids'].append(worker)
            queue.write_json(args.new_root / 'handoff_protocol.json', protocol)
    deadline = time.monotonic() + 45
    while any(queue.owned_alive(state, entry) for state, entry in victims.values()):
        require(time.monotonic() < deadline, 'worker remains alive; controller stays frozen')
        time.sleep(.2)
    ledger = read(args.new_root / 'handoff_original/queue_state.json')
    protocol.update(phase='migrating')
    protocol.setdefault('checkpoint_proofs', {})
    queue.write_json(args.new_root / 'handoff_protocol.json', protocol)
    old_entries = {entry['id']: entry for entry in old['runs']}
    fresh_jobs = {}
    for index, entry in enumerate(manifest['runs']):
        state = ledger['jobs'][entry['id']]
        boundary = queue.durable_episode(old_entries[entry['id']])
        if state['status'] in queue.TERMINAL:
            manifest['runs'][index] = old_entries[entry['id']]
            fresh_jobs[entry['id']] = deepcopy(state)
            continue
        require(boundary < entry['episode_budget'], f'{entry["id"]}: final evaluation pending; do not restart training')
        output = Path(entry['output_dir'])
        archive = args.new_root / 'handoff_original' / entry['id']
        archive.mkdir(exist_ok=True)
        for name in ('protocol.json', 'metrics.jsonl'):
            copy_once(output / name, archive / name)
        queue.recover_output(args.new_root, old_entries[entry['id']], state)
        proofs = {name: archive_checkpoint(output / name, archive / name)
                  for name in ('latest.pt', 'best.pt') if (output / name).exists()}
        # Both runners save checkpoints by atomic replacement. Keep the old
        # selection checkpoint without copying several GiB of Dreamer replay.
        if 'best.pt' in proofs:
            historical_best = archive_previous_best(output, archive / 'best.pt', previous_label)
            proofs['best.pt']['historical_evaluation_checkpoint'] = str(historical_best)
        protocol['checkpoint_proofs'][entry['id']] = proofs
        update_protocol(output, boundary, archive, manifest['evaluation_seeds'], temperature)
        protocol['continuation_episodes'][entry['id']] = boundary
        fresh_jobs[entry['id']] = continued_job(state, boundary)
        entry['required_gpu_mib'] = state['required_gpu_mib']
        queue.write_json(args.new_root / 'handoff_protocol.json', protocol)
    fresh = {key: value for key, value in ledger.items() if key not in {'jobs', 'controller_pid'}}
    fresh.update(jobs=fresh_jobs, transferred_at=queue.now())
    queue.write_json(args.new_root / 'queue_state.json', fresh)
    copy_once(args.old_root / 'admission_overrides.json', args.new_root / 'admission_overrides.json')
    manifest.update(launch_status='launching', launched_at=queue.now())
    queue.write_json(args.new_root / 'manifest.json', manifest)
    command = [sys.executable, '-u', str(args.new_root / 'source/tools/queued_city_baselines.py'),
               'run', '--output-root', str(args.new_root)]
    shell = shlex.join(command) + ' >> ' + shlex.quote(str(args.new_root / 'controller.log')) + ' 2>&1'
    subprocess.run(['/usr/bin/tmux', 'new-session', '-d', '-s', manifest['controller_session'],
                    '-c', str(args.new_root / 'source'), shell], check=True)
    manifest['launch_status'] = 'launched'
    queue.write_json(args.new_root / 'manifest.json', manifest)
    protocol.update(phase='launched', completed_at=queue.now(),
                    keeper_note='Never SIGCONT the old controller after handoff')
    queue.write_json(args.new_root / 'handoff_protocol.json', protocol)
    queue.write_json(args.old_root / 'evaluation_handoff.json', protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'apply'))
    parser.add_argument('--old-root', required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--new-root', required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--overlay-root', type=lambda value: Path(value).resolve(), default=ROOT)
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument('--argmax', action='store_true',
                            help='Deterministic evaluation with current argmax entrypoint defaults')
    evaluation.add_argument('--temperature', type=float, default=None,
                        help='Explicitly enable sampled evaluation; omitted temperature defaults to argmax')
    parser.add_argument('--resume-handoff', action='store_true')
    args = parser.parse_args()
    if args.temperature is None:
        args.argmax = True
    try:
        globals()[args.mode](args)
    except Exception as error:
        record = args.new_root / 'handoff_protocol.json'
        if record.exists():
            queue.write_json(record, {**read(record), 'last_error': repr(error), 'error_at': queue.now()})
        raise
    print(json.dumps(dict(mode=args.mode, new_root=str(args.new_root),
                         phase=read(args.new_root / 'handoff_protocol.json')['phase'])))


if __name__ == '__main__':
    main()
