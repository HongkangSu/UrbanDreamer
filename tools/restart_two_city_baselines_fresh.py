#!/usr/bin/env python3
"""Prepare, then explicitly apply a two-city fresh restart without touching Ingo learners."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queued_city_baselines as queue

CITIES = ('coulommiers', 'montereau_fault_yonne')
EPISODES = 2500
SELECTED = {f'{city}_{method}' for city in CITIES for method in queue.LEARNED}


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flag_value(command, flag):
    if command.count(flag) != 1:
        raise ValueError(f'expected exactly one {flag}')
    return command[command.index(flag) + 1]


def selected(entry):
    return entry['id'] in SELECTED


def verify_manifest(manifest):
    entries = manifest['runs']
    if len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('duplicate run IDs')
    if {entry['id'] for entry in entries if selected(entry)} != SELECTED:
        raise ValueError('all sixteen learning runs must be present')
    for entry in entries:
        if not selected(entry):
            continue
        command = entry['command']
        if (entry['city'] not in CITIES or entry['method'] not in queue.LEARNED
                or entry['id'] != f"{entry['city']}_{entry['method']}"
                or '--global-reward-only' not in command
                or flag_value(command, '--reward-objective') != 'observed_mean'
                or float(flag_value(command, '--sp-prior-weight')) != 0):
            raise ValueError(f"incompatible training semantics: {entry['id']}")


def verify_source(source, hashes):
    if source.is_symlink():
        raise ValueError('source root must not be linked')
    for relative, expected in hashes.items():
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or digest(path) != expected:
            raise ValueError(f'frozen source hash mismatch: {relative}')
    if any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('frozen source must contain independent files, not symlinks')


def source_tree_hashes(source):
    return {str(path.relative_to(source)): digest(path) for path in source.rglob('*')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}


def ensure_empty_outputs(root, manifest):
    for entry in manifest['runs']:
        if not selected(entry):
            continue
        output = Path(entry['output_dir'])
        if not output.is_relative_to(root):
            raise ValueError('fresh output escapes destination')
        for path in (output, *output.parents):
            if path.is_symlink():
                raise ValueError('fresh output must not be linked')
            if path == root:
                break
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError(f'fresh output is not empty: {output}')


def build_manifest(old, ledger, root):
    """Pure transformation; nonselected entries and states are copied verbatim."""
    verify_manifest(old)
    manifest, state = copy.deepcopy(old), copy.deepcopy(ledger)
    source = root / 'source'
    archived = []
    for entry in manifest['runs']:
        if not selected(entry):
            continue
        run_id = entry['id']
        prior = ledger['jobs'][run_id]
        archived.append(dict(id=run_id, output_dir=entry['output_dir'],
                             reason='user_requested_restart_from_scratch', previous_state=copy.deepcopy(prior)))
        output = root / entry['city'] / entry['method']
        command = []
        tokens = iter(entry['command'])
        for value in tokens:
            if value == '--evaluation-temperature':
                next(tokens)
            elif value not in ('--resume', '--skip-initial-evaluation',
                    '--evaluation-argmax', '--evaluation-sampled', '--evaluation-sampling'):
                command.append(value)
        command.append('--evaluation-argmax')
        script_name = ('train_toy_actor_critic.py' if entry['method'] == 'online_ac'
                       else 'train_toy_baseline.py')
        config_name = f"{entry['city']}.toml"
        if Path(command[2]).name != script_name or Path(flag_value(command, '--config')).name != config_name:
            raise ValueError('selected script/config does not match its method/city')
        command[2] = str(source / 'tools' / script_name)
        command[command.index('--config') + 1] = str(source / 'configs' / config_name)
        command[command.index('--output-dir') + 1] = str(output)
        command[command.index('--episodes') + 1] = str(EPISODES)
        for key in tuple(entry):
            if key.startswith('continuation_'):
                entry.pop(key)
        entry.update(command=command, output_dir=str(output), episode_budget=EPISODES,
                     log=str(root / 'logs' / f'{run_id}.log'),
                     state=str(root / 'processes' / f'{run_id}.json'), launched=False,
                     required_gpu_mib=prior['required_gpu_mib'],
                     restart_reason='user_requested_restart_from_scratch')
        state['jobs'][run_id] = dict(status='queued', attempt=0,
            required_gpu_mib=prior['required_gpu_mib'], observed_peak_mib=0.,
            oom_retries=0, pid=None, next_eligible=0.)
    manifest.update(source_snapshot=str(source), created_at=queue.now(),
        episode_budget=None, episode_budget_by_city={city: sorted({e['episode_budget']
            for e in manifest['runs'] if e['city'] == city and e['method'] in queue.LEARNED})
            for city in queue.CITIES},
        total_training_episodes=sum(e['episode_budget'] for e in manifest['runs']
                                    if e['method'] in queue.LEARNED),
        launch_status='stopped', restart_prepared=True,
        controller_session=f'ud_two_city_fresh2500_{root.name}',
        compactor_session=f'ud_two_city_fresh2500_compact_{root.name}')
    sources_by_city = dict(manifest.get('source_snapshots_by_city', {}))
    sources_by_city.update({city: str(source) for city in CITIES})
    manifest['source_snapshots_by_city'] = sources_by_city
    manifest['archive_policy'] = ('Fresh selected training outputs live under the new controller root; '
        'old training outputs are retained in place without restart-tool modification. '
        'Existing lossless archive compactors continue independently.')
    state.pop('controller_pid', None)
    state['transferred_at'] = queue.now()
    return manifest, state, archived


def controller_matches(pid, starttime, old_root):
    argv = queue.process_argv(pid)
    return (queue.process_identity(pid) == starttime and starttime is not None
            and any(value.endswith('/queued_city_baselines.py') for value in argv)
            and 'run' in argv and '--output-root' in argv
            and flag_value(argv, '--output-root') == str(old_root))


def controllers_for(root):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        pid = int(path.name)
        if controller_matches(pid, queue.process_identity(pid), root):
            result.append(pid)
    return result


def prepare(old_root, root, controller_pid):
    old_root = Path(old_root).resolve()
    original_root = Path(root).absolute()
    if any(path.is_symlink() for path in (original_root, *original_root.parents)):
        raise ValueError('destination must not use symlinks')
    root = original_root.resolve()
    if root == old_root or root.is_relative_to(old_root) or old_root.is_relative_to(root):
        raise ValueError('destination must be independent of old controller root')
    old = read_json(old_root / 'manifest.json')
    ledger = read_json(old_root / 'queue_state.json')
    verify_manifest(old)
    starttime = queue.process_identity(controller_pid)
    if not controller_matches(controller_pid, starttime, old_root):
        raise ValueError('old controller identity does not match')
    if controllers_for(old_root) != [controller_pid]:
        raise ValueError('expected exactly one old controller')
    if root.exists() and any(root.iterdir()):
        raise ValueError('destination requires an empty directory')
    source = Path(old['source_snapshot'])
    hashes = read_json(old_root / 'source_hashes.json')
    verify_source(source, hashes)
    full_hashes = source_tree_hashes(source)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, root / 'source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    verify_source(root / 'source', hashes)
    if source_tree_hashes(root / 'source') != full_hashes:
        raise ValueError('source copy is not byte-exact')
    queue.write_json(root / 'source_copy_hashes.json', full_hashes)
    for name in ('logs', 'processes', 'attempts'):
        (root / name).mkdir()
    manifest, state, archived = build_manifest(old, ledger, root)
    ensure_empty_outputs(root, manifest)
    for entry in manifest['runs']:
        if selected(entry):
            Path(entry['output_dir']).mkdir(parents=True)
    for name in ('source_hashes.json', 'admission_overrides.json'):
        if (old_root / name).exists():
            shutil.copy2(old_root / name, root / name)
    queue.write_json(root / 'original_manifest.json', old)
    queue.write_json(root / 'original_queue_state.json', ledger)
    queue.write_json(root / 'manifest.json', manifest)
    queue.write_json(root / 'queue_state.json', state)
    queue.write_json(root / 'archived_runs.json', archived)
    queue.write_json(root / 'restart_plan.json', dict(old_root=str(old_root), root=str(root),
        old_controller_pid=controller_pid, old_controller_starttime=starttime,
        selected_ids=sorted(SELECTED), episodes=EPISODES, status='prepared',
        prepared_manifest_sha256=digest(root / 'manifest.json'),
        prepared_ledger_sha256=digest(root / 'queue_state.json')))
    return root


def checked_signal(pid, sig, predicate):
    """Recheck PID/starttime and exact command ownership immediately before each signal."""
    if not predicate():
        raise RuntimeError(f'process identity changed before signal: {pid}')
    os.kill(pid, sig)


def stopped(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] in {'T', 't'}
    except (OSError, IndexError):
        return False


def verify_worker_inventory(entries, jobs):
    outputs = {entry['output_dir']: entry for entry in entries if selected(entry)}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        pid = int(path.name)
        argv = queue.process_argv(pid)
        if '--output-dir' not in argv:
            continue
        output = flag_value(argv, '--output-dir')
        if output not in outputs:
            continue
        entry = outputs[output]
        state = jobs[entry['id']]
        if state['status'] != 'running' or state.get('pid') != pid or not queue.owned_alive(state, entry):
            raise RuntimeError(f'untracked selected worker {pid}; old controller remains paused')


def stop_selected(entries, jobs, grace_seconds=10):
    running = [(entry, jobs[entry['id']]) for entry in entries
               if selected(entry) and jobs[entry['id']]['status'] == 'running']
    # Validate the complete set before signalling the first learner.
    for entry, state in running:
        if not queue.owned_alive(state, entry):
            if queue.process_identity(state.get('pid')) is not None:
                raise RuntimeError(f"selected worker identity mismatch: {entry['id']}")
    for entry, state in running:
        if queue.owned_alive(state, entry):
            checked_signal(state['pid'], signal.SIGTERM, lambda: queue.owned_alive(state, entry))
            if queue.owned_alive(state, entry):
                checked_signal(state['pid'], signal.SIGCONT, lambda: queue.owned_alive(state, entry))
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and any(queue.owned_alive(s, e) for e, s in running):
        time.sleep(.1)
    for entry, state in running:
        if queue.owned_alive(state, entry):
            checked_signal(state['pid'], signal.SIGKILL, lambda: queue.owned_alive(state, entry))
    deadline = time.monotonic() + 5
    while any(queue.owned_alive(s, e) for e, s in running):
        if time.monotonic() >= deadline:
            raise RuntimeError('selected workers have not exited')
        time.sleep(.1)


def apply(root):
    root = Path(root).resolve()
    lock = (root / 'restart_apply.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        return _apply_locked(root)
    finally:
        lock.close()


def _apply_locked(root):
    plan = read_json(root / 'restart_plan.json')
    if plan['status'] != 'prepared':
        raise ValueError('restart already applied or requires manual recovery')
    if (digest(root / 'manifest.json') != plan['prepared_manifest_sha256']
            or digest(root / 'queue_state.json') != plan['prepared_ledger_sha256']):
        raise ValueError('prepared restart artifacts changed')
    manifest = read_json(root / 'manifest.json')
    ensure_empty_outputs(root, manifest)
    hashes = read_json(root / 'source_hashes.json')
    verify_source(root / 'source', hashes)
    full_hashes = read_json(root / 'source_copy_hashes.json')
    if source_tree_hashes(root / 'source') != full_hashes:
        raise ValueError('prepared source tree changed')
    old_root = Path(plan['old_root'])
    old = read_json(old_root / 'manifest.json')
    original = read_json(root / 'original_manifest.json')
    # Admission resource estimates may evolve; learner commands and budgets may not.
    if ([(e['id'], e['command'], e['output_dir'], e['episode_budget']) for e in old['runs']]
            != [(e['id'], e['command'], e['output_dir'], e['episode_budget']) for e in original['runs']]
            or old['input_hashes_by_city'] != original['input_hashes_by_city']):
        raise ValueError('old learner manifest changed after preparation')
    verify_source(Path(old['source_snapshot']), hashes)
    if source_tree_hashes(Path(old['source_snapshot'])) != full_hashes:
        raise ValueError('old frozen source tree changed')
    pid, identity = plan['old_controller_pid'], plan['old_controller_starttime']
    owns_controller = lambda: controller_matches(pid, identity, old_root)
    if controllers_for(old_root) != [pid] or controllers_for(root):
        raise RuntimeError('unexpected or duplicate controller')
    checked_signal(pid, signal.SIGSTOP, owns_controller)
    # From this boundary, failures deliberately leave the old controller paused.
    plan.update(status='applying', applying_at=queue.now())
    queue.write_json(root / 'restart_plan.json', plan)
    deadline = time.monotonic() + 5
    while not stopped(pid):
        if not owns_controller() or time.monotonic() >= deadline:
            raise RuntimeError('old controller did not stop')
        time.sleep(.05)
    old = read_json(old_root / 'manifest.json')
    ledger = read_json(old_root / 'queue_state.json')
    if ledger.get('controller_pid') != pid:
        raise RuntimeError('old queue controller identity changed')
    for entry in old['runs']:
        state = ledger['jobs'][entry['id']]
        if not selected(entry) and state['status'] == 'running' and not queue.owned_alive(state, entry):
            raise RuntimeError(f"retained worker cannot be adopted: {entry['id']}")
    verify_worker_inventory(old['runs'], ledger['jobs'])
    manifest, state, archived = build_manifest(old, ledger, root)
    ensure_empty_outputs(root, manifest)
    queue.write_json(root / 'original_manifest.json', old)
    queue.write_json(root / 'original_queue_state.json', ledger)
    stop_selected(old['runs'], ledger['jobs'])
    queue.write_json(root / 'archived_runs.json', archived)
    if (old_root / 'admission_overrides.json').exists():
        shutil.copy2(old_root / 'admission_overrides.json', root / 'admission_overrides.json')
    manifest.update(launch_status='launched', restart_prepared=False, launched_at=queue.now())
    queue.write_json(root / 'manifest.json', manifest)
    queue.write_json(root / 'queue_state.json', state)
    command = [sys.executable, '-u', str(root / 'source/tools/queued_city_baselines.py'),
               'run', '--output-root', str(root)]
    with (root / 'controller.log').open('a') as log:
        child = subprocess.Popen(command, cwd=root / 'source', stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    plan.update(new_controller_pid=child.pid, new_controller_starttime=queue.process_identity(child.pid))
    queue.write_json(root / 'restart_plan.json', plan)
    deadline = time.monotonic() + 30
    while read_json(root / 'queue_state.json').get('controller_pid') != child.pid:
        if child.poll() is not None or time.monotonic() >= deadline:
            raise RuntimeError('new controller failed to acknowledge adoption; old controller remains paused')
        time.sleep(.2)
    if not owns_controller() or not stopped(pid) or controllers_for(root) != [child.pid]:
        raise RuntimeError('controller handoff invariant failed')
    plan.update(status='applied', completed_at=queue.now(), old_controller_disposition='left_SIGSTOP_never_resume')
    queue.write_json(root / 'restart_plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'apply'))
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--old-root', type=Path)
    parser.add_argument('--controller-pid', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare':
        if args.old_root is None or args.controller_pid is None:
            parser.error('prepare requires --old-root and --controller-pid')
        result = dict(prepared=str(prepare(args.old_root, args.output_root, args.controller_pid)),
                      episodes=EPISODES, fresh_jobs=len(SELECTED))
    else:
        result = apply(args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
