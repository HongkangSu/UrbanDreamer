"""Queue one checked corpus continuation without interrupting active pretraining."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.run_city_pretrain import (
    check_exit, cuda_oom_since, gpu_admission, read_json, write_status,
)


def process_snapshot(pid, *, proc_root=Path('/proc')):
    """Read Linux process identity and argv together; treat zombies as exited."""
    try:
        root = proc_root / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        try:
            argv = (root / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        except PermissionError:
            # Keep its parent relation, so an unreadable descendant still
            # blocks replacing the old supervisor. Other users may hide argv.
            argv = []
        return dict(pid=int(pid), state=fields[0], ppid=int(fields[1]),
                    started_ticks=int(fields[19]), argv=argv)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def process_table():
    table = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            row = process_snapshot(int(path.name))
            if row is not None:
                table[row['pid']] = row
    return table


def argument(argv, flag):
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def is_previous_supervisor(row, args):
    return (row is not None and row['pid'] == args.previous_supervisor_pid
            and any(Path(word).name == 'run_city_pretrain.py' for word in row['argv'])
            and argument(row['argv'], '--city-dir') == str(args.previous_city_dir))


def previous_activity(args, table):
    """Find all descendants plus detached old-city training processes on the host."""
    descendants, frontier = set(), {args.previous_supervisor_pid}
    while frontier:
        children = {pid for pid, row in table.items() if row['ppid'] in frontier}
        frontier = children - descendants
        descendants.update(children)
    for pid, row in table.items():
        argv = row['argv']
        if not any(Path(word).name in {'run_complete_pretrain.py', 'train_codec_ready.py'}
                   for word in argv):
            continue
        output = argument(argv, '--output-dir')
        if output and (Path(output) == args.previous_world_model_dir
                       or args.previous_city_dir in Path(output).parents):
            descendants.add(pid)
    return sorted(descendants)


def replace_empty_waiter(args, expected, status, *, timeout=15.):
    """Freeze a verified watcher before checking children, eliminating launch races.

    SIGSTOP applies only to the exact old watcher. If it acquired a child or
    advanced phases, resume it and let its active training complete normally.
    A tmux pane's direct child may be automatically continued by tmux after
    SIGSTOP. That case may replace only a rechecked empty watcher whose live
    GPU admission is still closed. An active child always blocks replacement.
    """
    if not args.replace_waiting_supervisor or not is_previous_supervisor(expected, args):
        return False
    if (status.get('pid') != expected['pid'] or status.get('phase') != 'waiting_for_world_model_gpu'
            or status.get('admission', {}).get('ready') is not False):
        return False
    if status.get('child_pids'):
        return False
    frozen = False
    stopped = False
    admission = None
    deadline = time.monotonic() + timeout
    try:
        current = process_snapshot(expected['pid'])
        if current is None or current['started_ticks'] != expected['started_ticks']:
            raise RuntimeError('previous supervisor process identity changed before handoff')
        os.kill(expected['pid'], signal.SIGSTOP)
        frozen = True
        while True:
            current = process_snapshot(expected['pid'])
            if current is None or current['started_ticks'] != expected['started_ticks']:
                raise RuntimeError('previous supervisor disappeared during handoff')
            if current['state'] in {'T', 't'}:
                stopped = True
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
        latest = read_json(args.previous_city_dir / 'supervisor_status.json')
        if (latest.get('pid') != expected['pid']
                or latest.get('phase') != 'waiting_for_world_model_gpu'
                or latest.get('admission', {}).get('ready') is not False
                or latest.get('child_pids') or previous_activity(args, process_table())):
            return False
        if not stopped:
            # tmux continues stopped pane leaders. Replacing its empty wait
            # loop is safe only while the actual admission remains closed;
            # never infer this from the old status file alone.
            admission = gpu_admission(args.device, .25, margin_mib=2048)
            if admission['ready']:
                return False
            current = process_snapshot(expected['pid'])
            latest = read_json(args.previous_city_dir / 'supervisor_status.json')
            if (not is_previous_supervisor(current, args)
                    or current['started_ticks'] != expected['started_ticks']
                    or latest.get('pid') != expected['pid']
                    or latest.get('phase') != 'waiting_for_world_model_gpu'
                    or latest.get('admission', {}).get('ready') is not False
                    or latest.get('child_pids') or previous_activity(args, process_table())):
                return False
        os.kill(expected['pid'], signal.SIGTERM)
        os.kill(expected['pid'], signal.SIGCONT)
        frozen = False
        deadline = time.monotonic() + timeout
        while True:
            current = process_snapshot(expected['pid'])
            if current is None:
                break
            if current['started_ticks'] != expected['started_ticks']:
                raise RuntimeError('previous supervisor PID was reused during handoff')
            if time.monotonic() >= deadline:
                raise RuntimeError('previous waiting supervisor did not exit after handoff')
            time.sleep(.1)
        if previous_activity(args, process_table()):
            raise RuntimeError('old-city training appeared during the empty-waiter handoff')
        write_status(args.city_dir / 'previous_handoff.json', phase='empty_waiter_replaced',
                     previous_supervisor_pid=expected['pid'],
                     previous_started_ticks=expected['started_ticks'],
                     previous_city_dir=str(args.previous_city_dir),
                     previous_world_model_dir=str(args.previous_world_model_dir),
                     handoff_method='stopped_empty_waiter' if stopped else 'live_gpu_blocked_empty_waiter',
                     live_admission=admission)
        return True
    finally:
        if frozen:
            current = process_snapshot(expected['pid'])
            if current is not None and current['started_ticks'] == expected['started_ticks']:
                os.kill(expected['pid'], signal.SIGCONT)


def previous_ready(args):
    """Active jobs finish first; only an explicitly replaced empty watcher is partial."""
    table = process_table()
    current = table.get(args.previous_supervisor_pid)
    if current is not None and not is_previous_supervisor(current, args):
        raise RuntimeError('previous supervisor PID belongs to a different live process')
    active = previous_activity(args, table)
    status_path = args.previous_city_dir / 'supervisor_status.json'
    status = read_json(status_path) if status_path.is_file() else {}
    if current is not None:
        if active:
            return dict(ready=False, reason='previous_training_running', active_pids=active)
        if replace_empty_waiter(args, current, status):
            return dict(ready=True, reason='empty_waiter_replaced', source_completed=False)
        return dict(ready=False, reason='previous_supervisor_running', active_pids=[])
    if active:
        return dict(ready=False, reason='previous_training_running', active_pids=active)
    summary_path = args.previous_world_model_dir / 'pretrain_summary.json'
    if summary_path.is_file() and read_json(summary_path).get('status') == 'pretrained':
        return dict(ready=True, reason='previous_pretraining_completed', source_completed=True)
    handoff = args.city_dir / 'previous_handoff.json'
    if args.replace_waiting_supervisor and handoff.is_file():
        record = read_json(handoff)
        if (record.get('phase') == 'empty_waiter_replaced'
                and record.get('previous_city_dir') == str(args.previous_city_dir)
                and record.get('previous_world_model_dir') == str(args.previous_world_model_dir)
                and record.get('previous_supervisor_pid') == args.previous_supervisor_pid):
            return dict(ready=True, reason='empty_waiter_replaced', source_completed=False)
    raise RuntimeError('previous pretraining exited without completion or a recorded empty-waiter handoff')


def commands(args, cache):
    counts = {split: sum(entry['split'] == split for entry in cache['entries'])
              for split in ('train', 'validation')}
    if cache.get('status') != 'complete' or not all(counts.values()):
        raise ValueError('supplement requires a complete cache with independent nonempty splits')
    output, tools = args.city_dir / 'world_model', args.source_dir / 'tools'
    migrate = [sys.executable, '-u', str(tools / 'migrate_pretrain_corpus.py'),
               '--source-dir', str(args.previous_world_model_dir), '--output-dir', str(output),
               '--cache-manifest', str(args.city_dir / 'cache/cache_manifest.json'),
               '--device', args.device]
    fm = [sys.executable, '-u', str(tools / 'run_complete_pretrain.py'),
          '--config', str(output / 'run_config.toml'),
          '--codec-dir', str(getattr(args, 'codec_dir', None) or args.previous_city_dir / 'codec'),
          '--cache-manifest', str(args.city_dir / 'cache/cache_manifest.json'),
          '--output-dir', str(output), '--device', args.device,
          '--threads', '4', '--batch-size', '64', '--accumulation-steps', '1',
          '--space-chunk', '64', '--time-chunk', '65536', '--mlp-chunk', '65536',
          '--activation-checkpointing', 'on', '--compile-blocks', 'on', '--precision', 'bf16',
          '--gpu-memory-fraction', str(args.gpu_memory_fraction), '--sequence-length', '1',
          '--fm-training-mode', 'single_step', '--readout-supervision', 'observed_endpoint',
          '--expected-train-episodes', str(counts['train']),
          '--expected-validation-episodes', str(counts['validation']),
          '--codec-validation-batch-size', '32']
    if args.activation_offload:
        fm.append('--activation-offload')
    if args.tail_execution_batch is not None:
        fm.extend(['--tail-execution-batch', str(args.tail_execution_batch)])
    return dict(migrate=migrate, world_model=fm)


def validate_existing_world_model(args):
    """Validate retained corpus ancestry before skipping the original handoff."""
    from tools.migrate_pretrain_corpus import validate_corpus_continuation
    from urban_dreamer.workflow.config import load_config

    output = args.city_dir / 'world_model'
    proof = read_json(output / 'corpus_migration.json')
    if Path(proof['source_output']).resolve() != args.previous_world_model_dir.resolve():
        raise ValueError('existing corpus migration source differs from previous world model')
    for row in process_table().values():
        if (any(Path(word).name == 'run_complete_pretrain.py' for word in row['argv'])
                and argument(row['argv'], '--output-dir') == str(output)):
            raise RuntimeError('existing world model still has a live training process')
    config = load_config(output / 'run_config.toml')
    validate_corpus_continuation(output, config, proof['input_hash'])
    return dict(ready=True, reason='verified_existing_corpus_continuation')


def training_admission(args, extra_mib=0):
    """Startup free memory is independent of the allocator's upper limit."""
    result = gpu_admission(args.device, 0.,
                           margin_mib=args.gpu_admission_min_free_mib + extra_mib)
    result['allocator_fraction'] = args.gpu_memory_fraction
    return result


def checked_cache(args):
    """Fail if an explicitly tracked producer exits before publishing its cache."""
    path = args.city_dir / 'cache/cache_manifest.json'
    if path.is_file():
        cache = read_json(path)
        if cache.get('status') == 'complete':
            return cache
    progress = args.city_dir / 'cache/cache_progress.json'
    if progress.is_file() and read_json(progress).get('status') == 'failed':
        raise RuntimeError('supplement cache producer failed')
    pid = getattr(args, 'cache_builder_pid', None)
    if pid is not None:
        row = process_snapshot(pid)
        argv = row['argv'] if row is not None else []
        scripts = {Path(word).name for word in argv}
        valid = (('build_pretrain_cache.py' in scripts
                  and argument(argv, '--output-dir') == str(args.city_dir / 'cache'))
                 or ('build_city_supplement_cache.py' in scripts
                     and argument(argv, '--city-dir') == str(args.city_dir)))
        if not valid:
            # Completion may have been published between the first lookup and
            # checking the producer's exit, so check the atomic manifest again.
            if path.is_file() and read_json(path).get('status') == 'complete':
                return read_json(path)
            raise RuntimeError('supplement cache builder exited or changed identity before completing')
    return None


def supervise(args):
    output = args.city_dir / 'world_model'
    monitor = output.with_suffix('.status.json')
    status_path = args.city_dir / 'supervisor_status.json'
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    children, state = {}, dict(source_dir=str(args.source_dir), device=args.device,
                              previous_city_dir=str(args.previous_city_dir), stages={})

    def update(phase, **values):
        state.update(phase=phase, **values)
        state['child_pids'] = {name: child.pid for name, child in children.items()
                               if child.poll() is None}
        write_status(status_path, **state)

    def launch(stage, command):
        path = args.city_dir / f'{stage}.log'
        offset = path.stat().st_size if path.exists() else 0
        with path.open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=args.source_dir, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[stage] = child
        state['stages'][stage] = dict(pid=child.pid, command=command, log=str(path),
                                     log_start_bytes=offset, launched_unix_seconds=time.time())
        update(stage + '_running')
        return child

    def wait_child(stage, child):
        while child.poll() is None:
            update(stage + '_running', world_model_status=read_json(monitor) if monitor.is_file() else None)
            time.sleep(args.poll_seconds)

    try:
        cache = checked_cache(args)
        while cache is None:
            update('waiting_for_supplement_cache')
            time.sleep(args.poll_seconds)
            cache = checked_cache(args)
        plan = commands(args, cache)
        if args.resume_existing_world_model:
            prior = validate_existing_world_model(args)
            update('resuming_existing_world_model', previous=prior)
        else:
            while True:
                prior = previous_ready(args)
                update('waiting_for_previous_pretraining', previous=prior)
                if prior['ready']:
                    break
                time.sleep(args.poll_seconds)
        if not (output / 'corpus_migration.json').is_file():
            child = launch('migrate', plan['migrate'])
            wait_child('migrate', child)
            check_exit('migrate', child)
        summary_path = output / 'pretrain_summary.json'
        if summary_path.is_file() and read_json(summary_path).get('status') == 'pretrained':
            update('complete', summary=read_json(summary_path))
            return
        margin = 0
        while True:
            admission = training_admission(args, margin)
            update('waiting_for_world_model_gpu', admission=admission)
            if not admission['ready']:
                time.sleep(args.poll_seconds)
                continue
            child = launch('world_model', plan['world_model'])
            wait_child('world_model', child)
            if child.returncode == 0:
                break
            stage = state['stages']['world_model']
            if cuda_oom_since(stage['log'], stage['log_start_bytes']):
                margin += 1024
                state['oom_retries'] = state.get('oom_retries', 0) + 1
                update('waiting_for_world_model_gpu', last_retry_reason='CUDA out of memory',
                       retained_checkpoint=str(output / 'checkpoints/latest.pt'))
                time.sleep(args.poll_seconds)
                continue
            check_exit('world_model', child)
        summary = read_json(summary_path)
        if summary.get('status') != 'pretrained':
            raise RuntimeError('supplement world model exited without completed pretraining')
        update('complete', summary=summary)
    except BaseException as error:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        update('failed', error=f'{type(error).__name__}: {error}',
               world_model_status=read_json(monitor) if monitor.is_file() else None)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    path = lambda value: Path(value).resolve()
    parser.add_argument('--city-dir', type=path, required=True)
    parser.add_argument('--previous-city-dir', type=path, required=True)
    parser.add_argument('--source-dir', type=path, default=ROOT)
    parser.add_argument('--device', choices=('cuda:0', 'cuda:1'), required=True)
    parser.add_argument('--previous-supervisor-pid', type=int, required=True)
    parser.add_argument('--previous-world-model-dir', type=path)
    parser.add_argument('--codec-dir', type=path,
                        help='original accepted codec artifacts when continuing a prior supplement')
    parser.add_argument('--cache-builder-pid', type=int,
                        help='track an existing producer and fail if it exits before cache completion')
    parser.add_argument('--replace-waiting-supervisor', action='store_true')
    parser.add_argument('--resume-existing-world-model', action='store_true',
                        help='verify and resume the retained corpus checkpoint without repeating old-job handoff')
    parser.add_argument('--gpu-memory-fraction', type=float, default=1.0)
    parser.add_argument('--gpu-admission-min-free-mib', type=float, default=8192.,
                        help='independent startup free-memory threshold; not an allocator limit')
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--tail-execution-batch', type=int)
    parser.add_argument('--poll-seconds', type=float, default=15.)
    args = parser.parse_args(argv)
    if args.previous_supervisor_pid < 1 or not 0 < args.poll_seconds <= 60:
        parser.error('previous PID must be positive and polling interval in (0, 60]')
    if not 0 < args.gpu_memory_fraction <= 1 or args.gpu_admission_min_free_mib <= 0:
        parser.error('GPU memory fraction must be in (0, 1] and admission memory positive')
    if args.tail_execution_batch is not None and not 1 <= args.tail_execution_batch <= 64:
        parser.error('tail execution batch must be in [1, 64]')
    if args.cache_builder_pid is not None and args.cache_builder_pid < 1:
        parser.error('cache builder PID must be positive')
    if args.city_dir == args.previous_city_dir or args.previous_city_dir in args.city_dir.parents:
        parser.error('supplement output must be isolated from the previous city directory')
    if args.previous_world_model_dir is None:
        args.previous_world_model_dir = args.previous_city_dir / 'world_model_ingo_profile'
    for name in ('migrate_pretrain_corpus.py', 'run_complete_pretrain.py'):
        if not (args.source_dir / 'tools' / name).is_file():
            parser.error(f'frozen execution source lacks tools/{name}')
    args.city_dir.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise InterruptedError(f'supplement supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    with (args.city_dir / 'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another supervisor already owns this supplement')
        supervise(args)


if __name__ == '__main__':
    main()
