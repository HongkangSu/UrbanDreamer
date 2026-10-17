"""Resource-admitted pretraining queue for verified compact-only city corpora.

A detached runner durably records exit status so restarting the queue cannot
launch a second copy of a live training command. Explicitly held lower-priority
controllers are resumed after every queued run reaches a terminal state.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

FLAGS = ('all_entries_content_hash_verified', 'all_entries_member_signature_verified',
         'complete_summary_json_preserved', 'real_batch_verified', 'raw_unavailable_verified')
TERMINAL = {'completed', 'failed', 'stopped'}


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(record):
    return bool(record.get('process_identity')) and process_identity(record['pid']) == record['process_identity']


def validate_plan(plan):
    if plan.get('schema_version') != 1 or not plan.get('runs'):
        raise ValueError('invalid queue schema or empty runs')
    cities = [run['city'] for run in plan['runs']]
    if len(cities) != len(set(cities)) or any('/' in city or city in {'.', '..'} for city in cities):
        raise ValueError('cities must be unique safe names')
    for run in plan['runs']:
        adoption_keys = ('adopted_result_path', 'adopted_plan_identity')
        if any(key in run for key in adoption_keys):
            if not all(isinstance(run.get(key), str) and run[key] for key in adoption_keys):
                raise ValueError('adopted result path and plan identity must be provided together')
            if not Path(run['adopted_result_path']).is_absolute():
                raise ValueError('adopted_result_path must be absolute')
        if not isinstance(run['command'], list) or not run['command'] or not all(isinstance(x, str) for x in run['command']):
            raise ValueError('command must be nonempty argv list')
        for key in ('cwd', 'cache_manifest', 'cache_verification', 'completion_status_path'):
            if not Path(run[key]).is_absolute():
                raise ValueError(f'{key} must be absolute')
        for key in ('expected_train_episodes', 'expected_validation_episodes'):
            if not isinstance(run[key], int) or run[key] <= 0:
                raise ValueError(f'{key} must be positive')
    for run in plan['runs']:
        gpu = gpu_settings(plan, run)
        if gpu['required_mib'] <= 0 or int(gpu['index']) < 0:
            raise ValueError('invalid GPU thresholds')
        if 'allocator_fraction' in gpu and not 0 < gpu['allocator_fraction'] <= 1:
            raise ValueError('invalid allocator fraction')
    if plan['disk']['min_free_gb'] < 0:
        raise ValueError('invalid disk threshold')
    for record in plan.get('priority_held_controllers', []):
        if not isinstance(record.get('pid'), int) or record['pid'] <= 0 or not record.get('process_identity'):
            raise ValueError('held controller requires PID and process identity')


def validate_cache(run):
    path = Path(run['cache_manifest'])
    payload = path.read_bytes()
    manifest, report = json.loads(payload), read(run['cache_verification'])
    if manifest.get('status') != 'complete' or manifest.get('source_storage') != 'compact_only_v1':
        raise ValueError('cache is not complete compact_only_v1')
    actual_identity = identity({k: v for k, v in manifest.items() if k != 'identity'})
    if manifest.get('identity') != actual_identity:
        raise ValueError('cache manifest identity mismatch')
    if (report.get('status') != 'passed' or report.get('city') != run['city']
            or manifest.get('contract', {}).get('city') != run['city']
            or report.get('manifest_identity') != actual_identity
            or report.get('manifest_sha256') != hashlib.sha256(payload).hexdigest()
            or not all(report.get(flag) is True for flag in FLAGS)):
        raise ValueError('cache verification proof mismatch')
    for split in ('train', 'validation'):
        if manifest.get(f'{split}_episodes') != run[f'expected_{split}_episodes']:
            raise ValueError(f'{split} count mismatch')
        if sum(row.get('split') == split for row in manifest['entries']) != run[f'expected_{split}_episodes']:
            raise ValueError(f'{split} entries mismatch')
    if len(manifest['entries']) != run['expected_train_episodes'] + run['expected_validation_episodes']:
        raise ValueError('unexpected cache entries')
    if run.get('expected_plan_identity') and manifest.get('plan_identity') != run['expected_plan_identity']:
        raise ValueError('conversion plan identity mismatch')
    return actual_identity


def gpu_settings(plan, run):
    return dict(plan['gpu'], **run.get('gpu', {}))


def resume_held_controllers(plan):
    results = []
    for held in plan.get('priority_held_controllers', []):
        result = dict(held, action='not_matching_or_not_stopped')
        try:
            fields = Path(f"/proc/{held['pid']}/stat").read_text().rsplit(') ', 1)[1].split()
            if fields[0] in {'T', 't'} and process_alive(held):
                os.kill(held['pid'], signal.SIGCONT)
                result['action'] = 'resumed'
        except ProcessLookupError:
            result['action'] = 'already_exited'
        except OSError as exc:
            result['action'] = f'error:{exc}'
        results.append(result)
    return results


def reserved_memory(plan, runs, gpu_index):
    eligible = [run for run in plan['runs']
                if str(gpu_settings(plan, run)['index']) == str(gpu_index)
                and (runs[run['city']].get('status') == 'running'
                     or runs[run['city']].get('launch_pending'))]
    if not eligible:
        return 0
    observed = {}
    try:
        output = subprocess.check_output(['nvidia-smi', '-i', str(gpu_index),
            '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
            text=True, timeout=15)
        for line in output.splitlines():
            pid, mib = line.split(',')
            observed[int(pid)] = int(mib.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass  # Missing accounting retains the full reservation.
    total = 0
    for run in eligible:
        row = runs[run['city']]
        used = 0
        if row.get('child_pid') and row.get('child_process_identity'):
            if process_identity(row['child_pid']) == row['child_process_identity']:
                used = observed.get(row['child_pid'], 0)
        total += max(0, gpu_settings(plan, run)['required_mib'] - used)
    return total


def prerequisites(plan, run=None, reserved_mib=0):
    """Return a waiting reason; unreadable/malformed state never grants admission."""
    try:
        if read(plan['cleanup_status_path']).get('stage') != 'eligible_cleanup_complete':
            return 'waiting_for_cleanup'
        for path in plan.get('blocking_queue_status_paths', []):
            state = read(path)
            runs = state.get('runs')
            if isinstance(runs, dict):
                runs = list(runs.values())
            if not isinstance(runs, list) or not runs or any(r.get('status') not in TERMINAL for r in runs):
                return f'waiting_for_queue:{path}'
        free = shutil.disk_usage(plan['disk']['path']).free
        if free < plan['disk']['min_free_gb'] * 1024 ** 3:
            return 'waiting_for_disk'
        settings = gpu_settings(plan, run or {})
        gpu = str(settings['index'])
        command = ['nvidia-smi', '-i', gpu]
        if not plan.get('allow_gpu_sharing', False):
            processes = subprocess.check_output(command + ['--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            if processes:
                return 'waiting_for_gpu_processes'
        free_mib = int(subprocess.check_output(command + ['--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True, timeout=15).strip())
        if free_mib - reserved_mib < settings['required_mib']:
            return 'waiting_for_gpu_memory'
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        return f'waiting_for_readable_prerequisites:{type(exc).__name__}:{exc}'
    return None


def execute_child(spec_path):
    spec = read(spec_path)
    result_path = Path(spec['result_path'])
    run = spec['run']
    record = dict(status='running', pid=os.getpid(), process_identity=process_identity(os.getpid()),
                  started_at=time.time(), plan_identity=spec['plan_identity'])
    atomic_json(result_path, record)
    try:
        environment = dict(os.environ, **run.get('env', {}))
        with Path(spec['log_path']).open('ab', buffering=0) as log:
            child = subprocess.Popen(run['command'], cwd=run['cwd'], env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
            record.update(child_pid=child.pid, child_process_identity=process_identity(child.pid))
            atomic_json(result_path, record)
            exit_code = child.wait()
        record['exit_code'] = exit_code
        complete = read(run['completion_status_path']) if Path(run['completion_status_path']).exists() else {}
        record['status'] = 'completed' if exit_code == 0 and complete.get('phase') == 'pretraining_completed' else 'failed'
        if record['status'] == 'failed':
            record['error'] = 'child exit or pretraining completion proof failed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    record['finished_at'] = time.time()
    atomic_json(result_path, record)
    return 0 if record['status'] == 'completed' else 1


def run_queue(plan_path, output):
    plan = read(plan_path)
    validate_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = output / 'plan.json'
        if frozen.exists() and read(frozen) != plan:
            raise ValueError('queue plan changed; use a new queue directory')
        atomic_json(frozen, plan)
        try:
            plan_id = identity(plan)
            runs = {run['city']: {'status': 'queued'} for run in plan['runs']}
            def heartbeat(stage):
                atomic_json(output / 'status.json', dict(status=stage, heartbeat_at=time.time(),
                    pid=os.getpid(), process_identity=process_identity(os.getpid()), plan_identity=plan_id, runs=runs))
            poll = max(1., min(30., float(plan.get('poll_seconds', 30))))
            verified = {}
            children = []
            while True:
                for child in children:
                    child.poll()
                # Refresh all receipts before calculating reservations or admitting
                # work. A missing cache in one city never blocks another ready city.
                for run in plan['runs']:
                    city = run['city']
                    adopted = 'adopted_result_path' in run
                    result_path = Path(run['adopted_result_path']) if adopted else output / f'{city}.result.json'
                    expected_identity = run['adopted_plan_identity'] if adopted else plan_id
                    if result_path.exists():
                        record = read(result_path)
                        if record.get('plan_identity') != expected_identity:
                            raise ValueError('child result bound to another plan')
                        runs[city] = record
                        if record['status'] == 'completed':
                            if record.get('exit_code') != 0 or read(run['completion_status_path']).get('phase') != 'pretraining_completed':
                                runs[city].update(status='failed', error='completed child proof no longer valid')
                        elif record['status'] not in TERMINAL and not process_alive(record):
                            child_live = record.get('child_pid') and record.get('child_process_identity') and (
                                process_identity(record['child_pid']) == record['child_process_identity'])
                            if child_live:
                                runs[city].update(status='running', error='runner disappeared; child still alive; retaining reservation')
                            else:
                                runs[city].update(status='failed', error='runner disappeared; refusing duplicate launch')
                    elif adopted:
                        runs[city] = dict(status='waiting', launch_pending=True,
                                         reason='adopted_receipt_missing; no duplicate launch')
                    elif (output / f'{city}.launch.json').exists():
                        runs[city] = dict(status='waiting', launch_pending=True,
                                         reason='launch_receipt_pending_or_unknown; no duplicate launch')
                if all(record['status'] in TERMINAL for record in runs.values()):
                    failed = any(record['status'] != 'completed' for record in runs.values())
                    atomic_json(output / 'priority_release.json', dict(
                        controllers=resume_held_controllers(plan), released_at=time.time()))
                    heartbeat('failed' if failed else 'completed')
                    return int(failed)
                for run in plan['runs']:
                    city = run['city']
                    if 'adopted_result_path' in run:
                        continue  # Adopted runners are monitored only, never launched here.
                    if runs[city]['status'] in TERMINAL or runs[city]['status'] == 'running' or runs[city].get('launch_pending'):
                        continue
                    if not plan.get('allow_gpu_sharing', False) and any(
                            row['status'] == 'running' or row.get('launch_pending') for row in runs.values()):
                        runs[city] = dict(status='waiting', reason='waiting_for_serial_slot')
                        continue
                    gpu = gpu_settings(plan, run)
                    reservation = reserved_memory(plan, runs, gpu['index'])
                    reason = prerequisites(plan, run, reservation)
                    if reason is None:
                        try:
                            if city not in verified:
                                verified[city] = validate_cache(run)
                        except FileNotFoundError:
                            reason = 'waiting_for_verified_city_cache'
                        except (ValueError, KeyError, TypeError) as exc:
                            runs[city] = dict(status='failed', error=f'cache verification: {exc}')
                            continue
                    if reason is None:
                        # Proof loading may be expensive; refresh runtime gates.
                        reason = prerequisites(plan, run, reservation)
                    if reason:
                        runs[city] = dict(status='waiting', reason=reason)
                        continue
                    if Path(run['completion_status_path']).exists():
                        runs[city] = dict(status='failed', error='completion status already exists; require fresh run output')
                        continue
                    result_path = output / f'{city}.result.json'
                    spec_path = output / f'{city}.launch.json'
                    atomic_json(spec_path, dict(run=run, plan_identity=plan_id,
                        result_path=str(result_path.resolve()), log_path=str((output / f'{city}.log').resolve()),
                        manifest_identity=verified[city], gpu=gpu))
                    try:
                        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                            '--execute-child', str(spec_path.resolve())], start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except OSError as exc:
                        record = dict(status='failed', error=f'runner launch failed: {exc}', plan_identity=plan_id)
                        atomic_json(result_path, record)
                        runs[city] = record
                        continue
                    children.append(child)
                    runs[city] = dict(status='running', runner_pid=child.pid, gpu=gpu)
                heartbeat('running' if any(row['status'] == 'running' for row in runs.values()) else 'waiting')
                if not all(row['status'] in TERMINAL for row in runs.values()):
                    time.sleep(poll)
        except Exception:
            atomic_json(output / 'priority_release.json', dict(
                controllers=resume_held_controllers(plan), released_at=time.time(), reason='fatal_queue_error'))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute-child', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute_child:
        return execute_child(args.execute_child)
    if args.plan is None or args.output_dir is None:
        parser.error('--plan and --output-dir are required')
    try:
        return run_queue(args.plan, args.output_dir)
    except Exception as exc:
        atomic_json(args.output_dir / 'fatal.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', time=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
