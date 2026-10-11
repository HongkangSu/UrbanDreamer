"""Supervise one frozen city's strict codec/cache/FM pretraining, without online training."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_status(path, **values):
    values.update(pid=os.getpid(), heartbeat_unix_seconds=time.time(),
                  utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_admission(device, fraction, *, margin_mib=512, run=subprocess.run):
    result = run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total',
                  '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    selected = int(device.split(':')[1])
    for line in result.stdout.splitlines():
        index, free, total = [float(part.strip()) for part in line.split(',')]
        if int(index) == selected:
            required = total * fraction + margin_mib
            return dict(ready=free >= required, device=device, free_mib=free,
                        total_mib=total, required_mib=required, allocator_fraction=fraction)
    raise RuntimeError(f'{device} missing from nvidia-smi')


def commands(args, *, cache=None):
    city = args.city_dir
    split = read_json(city / 'pilot/split.json')
    if not split['train'] or not split['validation']:
        raise ValueError('pilot requires nonempty train and held-out validation')
    python = sys.executable
    tools = args.source_dir / 'tools'
    codec = [python, '-u', str(tools / 'train_codec_ready.py'),
        '--pilot-dir', str(city / 'pilot'), '--output-dir', str(city / 'codec'),
        '--device', 'cpu' if args.codec_cpu else args.device,
        '--max-updates', '20000', '--evaluate-every', '500', '--batch-size', '4',
        '--expected-train-episodes', str(len(split['train'])),
        '--expected-validation-episodes', str(len(split['validation'])),
        '--gpu-memory-fraction', str(args.codec_memory_fraction), '--graph-multiply', 'sparse',
        '--configured-codec']
    if (city / 'codec/codec_latest.pt').is_file():
        codec.append('--resume')
    build = [python, '-u', str(tools / 'build_pretrain_cache.py'),
        '--config', str(city / 'base_config.toml'), '--census-dir', str(city / 'census'),
        '--pilot-dir', str(city / 'pilot'), '--output-dir', str(city / 'cache'), '--workers', '2']
    fm = None
    if cache is not None:
        if cache.get('status') != 'complete':
            raise ValueError('FM requires a completed cache')
        counts = {split: sum(row['split'] == split for row in cache['entries'])
                  for split in ('train', 'validation')}
        if not all(counts.values()):
            raise ValueError('full cache requires train and held-out validation')
        output = getattr(args, 'world_model_dir', None) or city / 'world_model'
        config_path = output / 'run_config.toml' if (output / 'run_config.toml').is_file() else city / 'base_config.toml'
        ingo = getattr(args, 'pretrain_profile', 'conservative') == 'ingo'
        fm = [python, '-u', str(tools / 'run_complete_pretrain.py'),
            '--config', str(config_path), '--codec-dir', str(city / 'codec'),
            '--cache-manifest', str(city / 'cache/cache_manifest.json'),
            '--output-dir', str(output), '--device', args.device,
            '--threads', '4' if ingo else '2', '--batch-size', '64' if ingo else '1',
            '--accumulation-steps', '1' if ingo else '32',
            '--space-chunk', '64' if ingo else '16', '--time-chunk', '65536' if ingo else '32768',
            '--mlp-chunk', '65536' if ingo else '16384',
            '--activation-checkpointing', 'on', '--compile-blocks', 'on' if ingo else 'off', '--precision', 'bf16',
            '--gpu-memory-fraction', str(args.wm_memory_fraction), '--sequence-length', '1',
            '--fm-training-mode', 'single_step', '--readout-supervision', 'observed_endpoint',
            '--expected-train-episodes', str(counts['train']),
            '--expected-validation-episodes', str(counts['validation'])]
        if getattr(args, 'activation_offload', False):
            fm.append('--activation-offload')
        if getattr(args, 'tail_execution_batch', None):
            fm.extend(['--tail-execution-batch', str(args.tail_execution_batch)])
    return dict(codec=codec, cache=build, world_model=fm)


def check_exit(stage, child):
    code = child.poll()
    if code is not None and code != 0:
        raise RuntimeError(f'{stage} process {child.pid} exited {code}; inspect {stage}.log')
    return code == 0


def cuda_oom_since(path, offset):
    """Classify only the current attempt, without accepting other failures."""
    with Path(path).open('rb') as stream:
        stream.seek(max(offset, Path(path).stat().st_size - 2 * 1024**2))
        output = stream.read().decode('utf-8', errors='replace')
    return 'CUDA out of memory' in output or 'torch.OutOfMemoryError' in output


def codec_ready(city):
    path = city / 'codec/result.json'
    if not path.is_file():
        return False
    result = read_json(path)
    if result.get('status') in {'budget_exhausted', 'hybrid_gate_failed'}:
        raise RuntimeError('codec exhausted its budget without passing raw/hybrid gates; FM remains blocked')
    return (result.get('status') == 'codec_ready'
            and result.get('gate', {}).get('passed') is True
            and result.get('hybrid_gate', {}).get('passed') is True
            and result.get('configured_gates_required') is True
            and result.get('configured_gate', {}).get('passed') is True
            and result.get('configured_hybrid_gate', {}).get('passed') is True)


def existing_cache_complete(city, pid):
    manifest = city / 'cache/cache_manifest.json'
    if manifest.is_file() and read_json(manifest).get('status') == 'complete':
        return True
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True, timeout=10)
    if result.returncode != 0 or 'build_pretrain_cache.py' not in result.stdout or str(city / 'cache') not in result.stdout:
        raise RuntimeError('existing cache builder ended before publishing a completed manifest')
    return False


def supervise(args):
    city = args.city_dir
    children = {}
    status_path = city / 'supervisor_status.json'
    threads = '4' if getattr(args, 'pretrain_profile', 'conservative') == 'ingo' else '2'
    output = getattr(args, 'world_model_dir', None) or city / 'world_model'
    monitor = output.with_suffix('.status.json')
    environment = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                       OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='1')
    state = dict(source_dir=str(args.source_dir), device=args.device, stages={},
                 pretrain_profile=getattr(args, 'pretrain_profile', 'conservative'), world_model_dir=str(output))

    def update(phase, **values):
        state.update(phase=phase, **values)
        state['child_pids'] = {name: child.pid for name, child in children.items()
                               if child.poll() is None}
        if getattr(args, 'existing_cache_pid', None) and not (city / 'cache/cache_manifest.json').is_file():
            state['child_pids']['existing_cache'] = args.existing_cache_pid
        write_status(status_path, **state)

    def launch(stage, command):
        log_path = city / f'{stage}.log'
        log_start = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=args.source_dir, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children[stage] = child
        state['stages'][stage] = dict(pid=child.pid, command=command,
                                      launched_unix_seconds=time.time(), log=str(log_path), log_start_bytes=log_start)
        update(stage + '_running')

    try:
        update('preparing')
        plan = commands(args)
        external_cache = getattr(args, 'existing_cache_pid', None)
        if external_cache:
            state['stages']['cache'] = dict(pid=external_cache, adopted=True,
                                           log=str(city / 'cache.log'))
        else:
            launch('cache', plan['cache'])
        while True:
            cache_done = existing_cache_complete(city, external_cache) if external_cache else check_exit('cache', children['cache'])
            if 'codec' in children:
                codec_done = check_exit('codec', children['codec'])
                ready = codec_ready(city) if codec_done else False
                if codec_done and not ready:
                    raise RuntimeError('codec exited successfully without a raw/hybrid ready result; FM remains blocked')
            else:
                ready = codec_ready(city)
                if not ready:
                    admission = dict(ready=True, device='cpu') if args.codec_cpu else gpu_admission(
                        args.device, args.codec_memory_fraction)
                    update('waiting_for_codec_gpu', admission=admission)
                    if admission['ready']:
                        launch('codec', plan['codec'])
            if cache_done and ready:
                break
            update('codec_and_cache_running' if 'codec' in children else 'waiting_for_codec_gpu')
            time.sleep(15)
        cache = read_json(city / 'cache/cache_manifest.json')
        plan = commands(args, cache=cache)
        margin = getattr(args, 'gpu_admission_margin_mib', 512)
        while True:
            while True:
                admission = (gpu_admission(args.device, args.wm_memory_fraction)
                             if margin == 512 else gpu_admission(args.device, args.wm_memory_fraction, margin_mib=margin))
                update('waiting_for_world_model_gpu', admission=admission)
                if admission['ready']:
                    break
                time.sleep(15)
            # A resumed attempt uses its published configuration and committed
            # model/optimizer/sampler checkpoint, retaining the requested batch.
            plan = commands(args, cache=cache)
            launch('world_model', plan['world_model'])
            child = children['world_model']
            while child.poll() is None:
                update('world_model_running', world_model_status=read_json(monitor) if monitor.is_file() else None)
                time.sleep(15)
            if child.returncode == 0:
                break
            stage = state['stages']['world_model']
            if (getattr(args, 'retry_cuda_oom', False)
                    and cuda_oom_since(stage['log'], stage['log_start_bytes'])):
                margin = max(margin, 2048)
                state['oom_retries'] = state.get('oom_retries', 0) + 1
                update('waiting_for_world_model_gpu', last_retry_reason='CUDA out of memory',
                       retained_checkpoint=str(output / 'checkpoints/latest.pt'), admission_margin_mib=margin)
                time.sleep(15)
                continue
            check_exit('world_model', child)
        summary = read_json(output / 'pretrain_summary.json')
        if summary.get('status') != 'pretrained':
            raise RuntimeError('world model ended without a completed pretraining summary')
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
    parser.add_argument('--city-dir', type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument('--device', choices=('cuda:0', 'cuda:1'), required=True)
    parser.add_argument('--source-dir', type=lambda p: Path(p).resolve(), default=ROOT)
    parser.add_argument('--codec-memory-fraction', type=float, default=.03)
    parser.add_argument('--wm-memory-fraction', type=float, default=.06)
    parser.add_argument('--codec-cpu', action='store_true', help='explicitly use CPU for codec only')
    parser.add_argument('--pretrain-profile', choices=('conservative', 'ingo'), default='conservative')
    parser.add_argument('--world-model-dir', type=lambda p: Path(p).resolve(),
                        help='target directory for a verified runtime continuation')
    parser.add_argument('--existing-cache-pid', type=int,
                        help='wait for the already running cache builder without starting a duplicate')
    parser.add_argument('--gpu-admission-margin-mib', type=int, default=512)
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--tail-execution-batch', type=int)
    parser.add_argument('--retry-cuda-oom', action='store_true',
                        help='resume the same requested profile from checkpoint after CUDA OOM when memory becomes available')
    args = parser.parse_args(argv)
    if args.gpu_admission_margin_mib < 0:
        parser.error('GPU admission margin must be non-negative')
    for fraction in (args.codec_memory_fraction, args.wm_memory_fraction):
        if not math.isfinite(fraction) or not 0 < fraction <= 1:
            parser.error('allocator fractions must be finite and in (0,1]')
    for tool in ('train_codec_ready.py', 'build_pretrain_cache.py', 'run_complete_pretrain.py'):
        if not (args.source_dir / 'tools' / tool).is_file():
            parser.error(f'execution source lacks tools/{tool}')
    sys.path.insert(0, str(args.source_dir))
    from urban_dreamer.workflow.config import load_config
    if load_config(args.city_dir / 'base_config.toml').codec.decoder_allow_unready:
        parser.error('city pretraining requires strict codec acceptance, decoder_allow_unready=False')
    def interrupted(signum, frame):
        raise InterruptedError(f'supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with (args.city_dir / 'supervisor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('another supervisor already owns this city')
        supervise(args)


if __name__ == '__main__':
    main()
