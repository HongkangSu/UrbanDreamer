"""Prepare and supervise a causal-global-reward online handoff after pretraining.

Preparation writes a separate source revision and plan. It never signals any
process or modifies the original pretraining source, configuration or checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.continue_mainline_entropy_annealing import process_record, verify_process
from tools.queue_pretrained_online import verify_execution_source
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.global_reward import REWARD_CONTRACT


def stage_source(source_root, revision_root, overlay_files):
    source, revision = Path(source_root).resolve(), Path(revision_root).resolve()
    if revision.exists() or source == revision:
        raise ValueError('paper source revision requires a new absent directory')
    verify_execution_source(source)
    names = sorted(set(overlay_files))
    required = {'tools/queue_pretrained_online.py', 'tools/continue_mainline_paper_reward.py',
                'tools/run_online_pilot.py', 'urban_dreamer/learning/replay.py',
                'urban_dreamer/learning/sumo_sampling.py', 'urban_dreamer/learning/observations.py',
                'urban_dreamer/learning/imagination.py', 'urban_dreamer/traffic/event_session.py',
                'urban_dreamer/workflow/checkpoint.py', 'urban_dreamer/workflow/config.py',
                'urban_dreamer/learning/global_reward.py', 'urban_dreamer/learning/training.py',
                'urban_dreamer/models/policy.py', 'urban_dreamer/models/event_critic.py'}
    if not required.issubset(names):
        raise ValueError('paper overlay omits a required reward or migration implementation')
    for name in names:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or Path(name).is_absolute():
            raise ValueError(f'invalid source overlay: {name}')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    before = {str(path.relative_to(source)): sha256_file(path) for path in source.rglob('*')
              if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    shutil.copytree(source, revision, symlinks=False,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in names:
        target = revision / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    after = {str(path.relative_to(revision)): sha256_file(path) for path in revision.rglob('*')
             if path.is_file()}
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if not set(changed).issubset(names):
        raise ValueError('paper revision changed files outside its explicit overlay')
    files = {name: digest for name, digest in after.items()
             if name not in {'source_online_manifest.json', 'paper_reward_overlay.json'}}
    manifest = dict(schema_version=1, kind='paper_reward_source_overlay', source_root=str(source),
        revision_root=str(revision), reward_contract=REWARD_CONTRACT, overlay_files=names,
        changed_files=changed, frozen_source_hashes=before, files=files,
        preservation='files outside explicit overlay retained; original revision untouched')
    atomic_json(revision / 'source_online_manifest.json', manifest)
    atomic_json(revision / 'paper_reward_overlay.json', manifest)
    verify_execution_source(revision)
    return manifest


def prepare_future(entropy_plan_path, revision_root, plan_path, *, waiter_pid, overlay_files):
    old_path, revision, path = (Path(value).resolve()
                                for value in (entropy_plan_path, revision_root, plan_path))
    if path.exists():
        raise ValueError('paper future plan requires a new absent path')
    old = json.loads(old_path.read_text())
    run = Path(old['run_dir'])
    if old.get('kind') != 'future_online_entropy_handoff':
        raise ValueError('expected the already-applied entropy future handoff')
    if verify_execution_source(old['revision_root']) != old['execution_manifest_sha256']:
        raise ValueError('existing entropy source differs from its applied handoff plan')
    verify_process(old['controller'], stopped=True)
    verify_process(old['pretrainer'], stopped=False)
    waiter = process_record(waiter_pid)
    expected = [old['online_command'][0], '-u',
                str(Path(old['revision_root']) / 'tools/continue_mainline_entropy_annealing.py'),
                'run-future', '--plan-path', str(old_path)]
    if not waiter or waiter['command'] != expected:
        raise ValueError('entropy waiter exact process command differs')
    verify_process(waiter, stopped=False)
    for directory in ('online_seed', 'online_actor_critic_2000', 'online_supervisor'):
        if (run / directory).exists():
            raise ValueError('paper handoff requires no online seed, supervisor or training activity')
    for filename, field in (('training_plan.json', 'original_training_plan_sha256'),
                            ('pretrain_config.toml', 'original_pretrain_config_sha256')):
        if sha256_file(run / filename) != old[field]:
            raise ValueError(f'original pretraining artifact changed: {filename}')
    training_plan = json.loads((run / 'training_plan.json').read_text())
    for name, digest in training_plan['frozen_inputs'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'frozen pretraining input changed: {name}')
    stage_source(old['revision_root'], revision, overlay_files)
    online = old['online_command'][:]
    online[online.index(str(Path(old['revision_root']) / 'tools/queue_pretrained_online.py'))] = str(revision / 'tools/queue_pretrained_online.py')
    online[online.index('--execution-source') + 1] = str(revision)
    plan = copy.deepcopy(old)
    plan.update(kind='future_online_paper_reward_handoff', revision_root=str(revision),
        plan_path=str(path), reward_contract=REWARD_CONTRACT, previous_waiter=waiter,
        previous_plan_path=str(old_path), previous_plan_sha256=sha256_file(old_path),
        execution_manifest_sha256=verify_execution_source(revision), online_command=online,
        online_disagreement_weight=0.,
        preservation='same original pretrainer PID, source/config/inputs/checkpoints untouched; '
                     'online seed has explicit context/reward migration and preserved Actor/world-model weights')
    atomic_json(path, plan)
    return plan


def validate_future_plan(plan):
    run = Path(plan['run_dir'])
    if (plan.get('schema_version') != 1 or plan.get('kind') != 'future_online_paper_reward_handoff'
            or plan.get('reward_contract') != REWARD_CONTRACT
            or plan.get('online_disagreement_weight') != 0.
            or verify_execution_source(plan['revision_root']) != plan['execution_manifest_sha256']
            or sha256_file(plan['previous_plan_path']) != plan['previous_plan_sha256']
            or sha256_file(run / 'training_plan.json') != plan['original_training_plan_sha256']
            or sha256_file(run / 'pretrain_config.toml') != plan['original_pretrain_config_sha256']):
        raise ValueError('paper future handoff immutable plan or source changed')
    original = json.loads(Path(plan['previous_plan_path']).read_text())
    expected_command = original['online_command'][:]
    expected_command[expected_command.index(str(Path(original['revision_root']) / 'tools/queue_pretrained_online.py'))] = str(Path(plan['revision_root']) / 'tools/queue_pretrained_online.py')
    expected_command[expected_command.index('--execution-source') + 1] = plan['revision_root']
    if (plan['online_command'] != expected_command or plan['run_dir'] != original['run_dir']
            or plan['controller'] != original['controller'] or plan['pretrainer'] != original['pretrainer']):
        raise ValueError('paper handoff command or original process ownership changed')


def run_future(plan_path):
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    validate_future_plan(plan)
    prior = process_record(plan['previous_waiter']['pid'])
    if prior is not None and prior['state'] != 'Z':
        raise ValueError('previous waiter must exit before paper waiter takes ownership')
    run = Path(plan['run_dir'])
    for directory in ('online_seed', 'online_actor_critic_2000', 'online_supervisor'):
        if (run / directory).exists():
            raise ValueError('prior waiter already created online activity; refusing ambiguous takeover')
    stopped = False
    def request_stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def status(phase, **extra):
        atomic_json(run / 'status.json', dict(phase=phase, controller_pid=os.getpid(),
            updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            paper_reward_handoff=str(path), reward_contract=REWARD_CONTRACT,
            original_controller_stopped=True, **extra))
    with (run / 'entropy_annealing_controller.lock').open('a') as lock:
        # Same ownership lock as the replaced waiter prevents a double launch.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not stopped:
                verify_process(plan['controller'], stopped=True)
                worker = process_record(plan['pretrainer']['pid'])
                if worker is not None and worker['identity'] != plan['pretrainer']['identity']:
                    raise ValueError('pretrainer PID identity changed')
                if worker is None or worker['state'] == 'Z':
                    break
                verify_process(plan['pretrainer'], stopped=False)
                status('pretraining', child_pid=worker['pid'], command=worker['command'],
                       paper_reward_ready_for_online=True)
                time.sleep(5.)
            if stopped:
                status('paper_bridge_stopped', pretrainer_not_signalled=True)
                return
            summary = json.loads((run / 'world_model/pretrain_summary.json').read_text())
            if summary.get('status') != 'pretrained':
                raise ValueError('original pretrainer exited without successful completion')
            validate_future_plan(plan)
            command = plan['online_command']
            environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2')
            environment.pop('URB_ACTOR_ENTROPY_MODE', None)
            with (run / 'online_queue.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, cwd=plan['revision_root'], env=environment,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                status('online', child_pid=child.pid, command=command)
                while child.poll() is None and not stopped:
                    time.sleep(5.)
                if stopped and child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30.)
                    status('paper_bridge_stopped', pretrainer_not_signalled=True)
                    return
                if child.returncode:
                    raise RuntimeError(f'paper online queue exited {child.returncode}')
            queue = json.loads((run / 'online_supervisor/status.json').read_text())
            if queue.get('phase') != 'completed':
                raise RuntimeError('paper online queue exited without completed original budget')
            status('completed')
        except BaseException as error:
            status('failed', error=repr(error), pretrainer_not_signalled=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-future')
    prepare.add_argument('--entropy-plan-path', type=Path, required=True)
    prepare.add_argument('--revision-root', type=Path, required=True)
    prepare.add_argument('--plan-path', type=Path, required=True)
    prepare.add_argument('--waiter-pid', type=int, required=True)
    prepare.add_argument('--overlay-file', action='append', required=True)
    run = commands.add_parser('run-future')
    run.add_argument('--plan-path', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-future':
        result = prepare_future(args.entropy_plan_path, args.revision_root, args.plan_path,
                                waiter_pid=args.waiter_pid, overlay_files=args.overlay_file)
    else:
        result = run_future(args.plan_path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
