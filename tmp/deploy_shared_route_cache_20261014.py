"""Checkpoint-preserving cache-only handoff for the three UrbanDreamer runs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
CITIES = ('ingolstadt_custom', 'coulommiers', 'montereau_fault_yonne')


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, payload):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def process(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(identity=fields[19], state=fields[0],
                    argv=[s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s])
    except FileNotFoundError:
        return None


def supervisor(city, record_dir):
    record = json.loads((record_dir / city / 'handoff.json').read_text())
    source = Path(record['source'])
    manifest = json.loads((source / 'source_manifest.json').read_text())
    assert all(digest(source / name) == expected for name, expected in manifest['files'].items())
    output = Path(record['output'])
    assert digest(output / 'checkpoints/latest.pt') == record['checkpoint_sha256']
    assert digest(output / 'run_config.toml') == record['config_sha256']
    environment = dict(os.environ, URB_IMAGINATION_WORKERS='8',
        URB_ROUTE_CACHE_DIR=record['cache_directory'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(BATCH / city / 'inductor_cache'))
    environment.pop('URB_ACTOR_ENTROPY_MODE', None)
    # Other experiments may still be releasing startup allocations. Wait for
    # the existing runner's CUDA preflight rather than changing its device.
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '--id=1',
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 3072:
            break
        write(BATCH / city / 'status.json', dict(city=city, pid=None, supervisor_pid=os.getpid(),
            status='waiting_for_gpu', source=str(source), cache_directory=record['cache_directory'],
            shared_route_cache_capacity=200000, restored_episode=record['completed_episodes'],
            completed_episodes=record['completed_episodes'], updated_utc=utc(), free_gpu_mib=free))
        time.sleep(5)
    with (BATCH / city / 'online_shared_cache_20261014.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(record['command'], cwd=source, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status = dict(city=city, pid=child.pid, identity=process(child.pid)['identity'],
        supervisor_pid=os.getpid(), source=str(source), output=str(output),
        command=record['command'], device='cuda:1', cache_directory=record['cache_directory'],
        shared_route_cache_capacity=200000, restored_episode=record['completed_episodes'],
        status='starting', started_utc=utc(), handoff=str(record_dir / city / 'handoff.json'))
    while True:
        code = child.poll()
        progress = json.loads((output / 'pilot_progress.json').read_text())
        status.update(status='running' if code is None else 'completed' if code == 0 else 'failed',
                      phase=progress['phase'], completed_episodes=len(progress['online']),
                      updated_utc=utc(), exit_code=code)
        write(BATCH / city / 'status.json', status)
        if code is not None:
            return
        time.sleep(5)


def deploy():
    import torch
    torch.set_num_threads(1)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    record_dir = BATCH / 'cache_handoffs' / ('shared_200000_' + stamp)
    record_dir.mkdir(parents=True)
    manifest = json.loads((BATCH / 'manifest.json').read_text())
    original_source = Path(manifest['source'])
    original_manifest = json.loads((original_source / 'source_manifest.json').read_text())
    assert all(digest(original_source / p) == h for p, h in original_manifest['files'].items())
    shutil.copy2(BATCH / 'manifest.json', record_dir / 'original_manifest.json')
    source = BATCH / ('source_shared_cache_' + stamp)
    shutil.copytree(original_source, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    changed = ['urban_dreamer/traffic/route_candidates.py', 'urban_dreamer/traffic/shared_route_cache.py']
    for name in changed:
        shutil.copy2(ROOT / name, source / name)
    files = {str(p.relative_to(source)): digest(p) for p in source.rglob('*')
             if p.is_file() and p.name != 'source_manifest.json'}
    assert all(files[name] == value for name, value in original_manifest['files'].items() if name not in changed)
    write(source / 'source_manifest.json', dict(created_utc=utc(), files=files,
        parent_source=str(original_source), changed_files=changed))
    controller = record_dir / 'controller.py'
    shutil.copy2(__file__, controller)
    pending = []
    for city in CITIES:
        folder = BATCH / city
        output = folder / 'online'
        status = json.loads((folder / 'status.json').read_text())
        child = process(status['pid'])
        assert child and str(output) in child['argv'] and os.getpgid(status['pid']) == status['pid']
        owner = process(status['supervisor_pid'])
        if owner:
            expected_controller = (Path(status['handoff']).parent.parent / 'controller.py'
                                   if status.get('handoff') else BATCH / 'controller.py')
            assert str(expected_controller) in owner['argv']
            os.kill(status['supervisor_pid'], signal.SIGTERM)
        backup = record_dir / city
        backup.mkdir()
        write(backup / 'old_status.json', status)
        command = list(child['argv'])
        command[command.index(str(original_source / 'tools/run_online_pilot.py'))] = str(source / 'tools/run_online_pilot.py')
        initial = len(json.loads((output / 'pilot_progress.json').read_text())['online'])
        previous_handoff = (json.loads(Path(status['handoff']).read_text())
                            if status.get('handoff') else None)
        # A runner still doing its read-only initialization can keep the exact
        # previously preserved checkpoint; no in-flight episode is discarded.
        startup_log = folder / 'online_shared_cache_20261014.log'
        initializing = (previous_handoff is not None and initial == status.get('restored_episode')
                        and startup_log.exists() and startup_log.stat().st_size == 0)
        pending.append(dict(city=city, pid=status['pid'], identity=child['identity'],
            output=str(output), source=str(source), command=command,
            cache_directory=str(BATCH / city / 'shared_route_cache_200000'), initial_completed=initial,
            initializing=initializing, previous_handoff=previous_handoff))
    write(record_dir / 'stop_requests.json', pending)
    write(BATCH / 'shared_cache_handoff_latest.json', dict(record_dir=str(record_dir),
        source=str(source), phase='waiting_for_episode_boundaries', requested_utc=utc()))
    print(json.dumps(dict(phase='waiting_for_episode_boundaries', record_dir=str(record_dir), runs=pending)), flush=True)
    deadline = time.monotonic() + 5400
    while pending:
        for row in pending[:]:
            city, pid = row['city'], row['pid']
            output, backup = Path(row['output']), record_dir / city
            current = process(pid)
            assert current and current['identity'] == row['identity']
            progress = json.loads((output / 'pilot_progress.json').read_text())
            if not row['initializing'] and len(progress['online']) <= row['initial_completed']:
                continue
            os.killpg(pid, signal.SIGSTOP)
            time.sleep(.2)
            assert process(pid)['state'] in ('T', 't')
            checkpoint = output / 'checkpoints/latest.pt'
            os.link(checkpoint, backup / 'checkpoint.pt')
            checkpoint_hash = digest(backup / 'checkpoint.pt')
            previous = row.pop('previous_handoff')
            if previous and checkpoint_hash == previous['checkpoint_sha256']:
                completed = previous['completed_episodes']
                fingerprint, input_hash = previous['config_fingerprint'], previous['input_hash']
                assert completed == len(progress['online'])
            else:
                raw = torch.load(backup / 'checkpoint.pt', map_location='cpu', weights_only=True)
                completed = raw['next_unit']
                assert completed == len(raw['training_state']['metrics']['online_pilot_progress']['online'])
                assert raw['phase'] == 'online_training'
                fingerprint, input_hash = raw['config_fingerprint'], raw['input_hash']
                del raw
            row.update(completed_episodes=completed, config_fingerprint=fingerprint,
                       input_hash=input_hash, checkpoint_sha256=checkpoint_hash,
                       config_sha256=digest(output / 'run_config.toml'), stopped_utc=utc())
            for name in ('run_config.toml', 'pilot_protocol.json', 'pilot_progress.json'):
                shutil.copy2(output / name, backup / name)
            os.killpg(pid, signal.SIGTERM)
            os.killpg(pid, signal.SIGCONT)
            end = time.monotonic() + 15
            while process(pid) and time.monotonic() < end:
                time.sleep(.2)
            if process(pid):
                os.killpg(pid, signal.SIGKILL)
            assert not process(pid)
            assert digest(checkpoint) == row['checkpoint_sha256']
            write(backup / 'handoff.json', row)
            with (backup / 'supervisor.log').open('ab', buffering=0) as log:
                resumed = subprocess.Popen([PYTHON, str(controller), 'supervise', city, str(record_dir)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row.update(new_supervisor_pid=resumed.pid, phase='resuming_with_shared_cache')
            write(backup / 'handoff.json', row)
            print(json.dumps(row), flush=True)
            pending.remove(row)
        if time.monotonic() > deadline:
            raise TimeoutError(f'No committed episode boundary: {pending}')
        time.sleep(.5)
    manifest.update(source=str(source), previous_source=str(original_source),
        source_manifest_sha256=digest(source / 'source_manifest.json'),
        shared_route_cache_capacity=200000, shared_route_cache_handoff=str(record_dir),
        cache_only_runtime_change=True, cache_updated_utc=utc())
    for entry in manifest['entries']:
        row = json.loads((record_dir / entry['city'] / 'handoff.json').read_text())
        entry.update(command=row['command'], shared_route_cache_directory=row['cache_directory'])
    write(BATCH / 'manifest.json', manifest)
    write(BATCH / 'shared_cache_handoff_latest.json', dict(record_dir=str(record_dir),
        source=str(source), phase='all_three_resumed', updated_utc=utc()))
    print(json.dumps(dict(phase='all_three_resumed', record_dir=str(record_dir))), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'supervise':
        supervisor(sys.argv[2], Path(sys.argv[3]))
    else:
        deploy()
