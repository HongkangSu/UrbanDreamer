"""Own a frozen full-data pretrain followed by a fresh 2000-episode online run."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def completed_input(path):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('status', value.get('phase')) in {'failed', 'error'}:
        raise ValueError(f'prerequisite failed: {path}: {value.get("error", value)}')
    return value if value.get('status') == 'complete' else None


def preparation_verified(cache_manifest):
    """A completed cache still waits for the data monitor's split/provenance proof."""
    monitor = Path(cache_manifest).parent / 'status.json'
    return not monitor.exists() or completed_input(monitor) is not None


def tail_seconds(profile, size):
    """Interpolate measured eager tails; never substitute compiled full timings."""
    points = sorted((int(key), float(value)) for key, value in profile['tail_update_seconds'].items())
    if not points or any(x <= 0 or not math.isfinite(y) or y <= 0 for x, y in points):
        raise ValueError('benchmark lacks finite positive eager tail timings')
    if size <= points[0][0]:
        return points[0][1]  # conservative below the smallest measurement
    for (left, low), (right, high) in zip(points, points[1:]):
        if size <= right:
            return low + (high - low) * (size - left) / (right - left)
    # Conservative extrapolation includes the measured eager cost per sample.
    return points[-1][1] * size / points[-1][0]


def estimate_epoch(entries, profile):
    batch = int(profile['batch_size'])
    sequence_length = int(profile.get('sequence_length', 1))
    full_time = float(profile['full_update_seconds_median'])
    loading = profile.get('sample_loading_seconds', 0)
    if not isinstance(loading, dict):
        loading = float(loading)
    if batch < 1 or sequence_length < 1 or not math.isfinite(full_time) or full_time <= 0 or (not isinstance(loading, dict) and loading < 0):
        raise ValueError('invalid full-batch benchmark')
    full = tails = windows = 0
    seconds = 0.
    for entry in entries:
        count = int(entry['windows'])
        if count < 1:
            raise ValueError('episode has no legal sequence starts')
        count = max(1, count - sequence_length + 1)
        batches, remainder = divmod(count, batch)
        full += batches
        tails += bool(remainder)
        windows += count
        seconds += batches * full_time
        if remainder:
            seconds += tail_seconds(profile, remainder)
    if isinstance(loading, dict):
        measured = sorted((int(key), value) for key, value in loading.items())
        if not measured:
            raise ValueError('sample loading benchmark is empty')
        def loading_cost(size, field):
            selected = next((row for count, row in measured if count >= size), measured[-1][1])
            value = float(selected[field])
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid sample loading time')
            return value
        # Every episode incurs its first cold load; remaining batches reuse its cache.
        for entry in entries:
            count = max(1, int(entry['windows']) - sequence_length + 1)
            batches, remainder = divmod(count, batch)
            seconds += loading_cost(min(count, batch), 'cold')
            seconds += max(0, batches - 1) * loading_cost(batch, 'warm_median')
            if remainder and batches:
                seconds += loading_cost(remainder, 'warm_median')
    else:
        seconds += windows * loading
    return dict(batch_size=batch, train_windows=windows, full_batches=full, tail_batches=tails,
                updates_per_epoch=full + tails, training_seconds_per_epoch=seconds,
                estimated_seconds_per_epoch=seconds * 1.15,
                peak_allocated_mib=profile.get('peak_allocated_mib'))


def validate_benchmark_profile(config, benchmark):
    """Recursive execution needs timings measured with the same unroll depth."""
    if config.method.fm_training_mode != 'recursive_3step':
        return
    sequence_length = config.resources.world_model_initialization_sequence_length
    for profile in benchmark['profiles']:
        if profile.get('status', 'complete') != 'complete':
            continue
        if (profile.get('fm_training_mode') != 'recursive_3step'
                or profile.get('sequence_length') != sequence_length):
            raise ValueError('recursive_3step requires matching benchmark fm_training_mode '
                             f'and sequence_length = {sequence_length}')


def make_plan(cache, benchmark, *, exposure_target=5263655, target_hours=6, max_hours=8,
              available_memory_mib=None, memory_margin_mib=768):
    if cache.get('status') != 'complete' or benchmark.get('status') != 'complete':
        raise ValueError('planning requires completed inputs')
    train = [entry for entry in cache['entries'] if entry['split'] == 'train']
    validation = [entry for entry in cache['entries'] if entry['split'] == 'validation']
    if not train or not validation or not cache.get('identity'):
        raise ValueError('full data must include train, heldout validation and identity')
    profiles = [row for row in benchmark['profiles'] if row.get('status', 'complete') == 'complete']
    measured_profiles = profiles
    memory_excluded = []
    if available_memory_mib is not None:
        feasible = []
        for profile in profiles:
            footprint = profile.get('peak_reserved_mib', profile.get('peak_allocated_mib', math.inf))
            if float(footprint) + memory_margin_mib <= available_memory_mib:
                feasible.append(profile)
            else:
                memory_excluded.append(dict(batch_size=profile['batch_size'],
                    estimated_reserved_mib=footprint, required_mib=float(footprint) + memory_margin_mib))
        profiles = feasible
        if not profiles:
            raise ValueError('no measured pretraining batch fits current GPU memory headroom')
    # Eager tail kernels are shape-specific, independent of the compiled full batch.
    # Pool measurements across profiles instead of scaling one small tail blindly.
    tail_measurements = {}
    for profile in measured_profiles:
        for size, seconds in profile['tail_update_seconds'].items():
            tail_measurements.setdefault(size, []).append(float(seconds))
    shared_tails = {size: statistics.median(values) for size, values in tail_measurements.items()}
    estimates = [estimate_epoch(train, {**profile, 'tail_update_seconds': shared_tails})
                 for profile in profiles]
    if not estimates:
        raise ValueError('no successful benchmark profile')
    best = min(estimates, key=lambda row: row['estimated_seconds_per_epoch'])
    epoch_seconds = best['estimated_seconds_per_epoch']
    fixed_seconds = 600
    maximum_epochs = math.floor((max_hours * 3600 - fixed_seconds) / epoch_seconds)
    if maximum_epochs < 1:
        raise ValueError('one complete epoch exceeds the eight-hour budget; refusing a data subset')
    wanted = math.ceil(exposure_target / best['train_windows'])
    # When full reference exposure is too expensive, retain the greatest
    # complete exposure that fits the explicit eight-hour limit.
    epochs = min(wanted, maximum_epochs)
    return dict(schema_version=1, data_identity=cache['identity'], **best,
                train_episodes=len(train), validation_episodes=len(validation),
                validation_windows=sum(int(row['windows']) for row in validation),
                method_train_episodes=dict(Counter(str(row.get('method', 'unspecified')) for row in train)),
                epochs=epochs, exposure_target=exposure_target,
                exposure_target_epochs=wanted, actual_sample_exposures=epochs * best['train_windows'],
                total_optimizer_updates=epochs * best['updates_per_epoch'],
                estimated_total_seconds=epochs * epoch_seconds + fixed_seconds,
                validation_and_overhead_fraction=.15, fixed_startup_allowance_seconds=fixed_seconds,
                candidate_profiles=estimates, eager_tail_seconds=shared_tails,
                target_hours=target_hours, maximum_hours=max_hours,
                available_memory_mib=available_memory_mib, memory_margin_mib=memory_margin_mib,
                memory_excluded_profiles=memory_excluded,
                rollout_count=16, actor_updates_per_episode=4,
                uncertainty='Measured update timings; 15% validation/checkpoint allowance plus 600s startup. Shared GPU load may vary. Full epochs only, every legal start once per epoch.')


def immutable_plan(path, plan):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('frozen training plan or input identity changed on resume')
    else:
        atomic_json(path, plan)


def write_config(args, plan):
    from urban_dreamer.workflow.config import DataConfig, _semantic_fingerprint, load_config
    from tools.run_complete_pretrain import export_run_config
    config = load_config(args.config)
    config = replace(config,
        run=replace(config.run, output_dir=args.run_dir / 'world_model', device=args.device,
                    world_model_initialization_epochs=plan['epochs'], online_training_episodes=2000),
        resources=replace(config.resources, world_model_initialization_batch_size=plan['batch_size'],
            world_model_gradient_accumulation_steps=1,
            world_model_pretrain_precision='bf16', world_model_pretrain_compile_blocks=True,
            world_model_pretrain_space_chunk=64, world_model_pretrain_time_chunk=65536,
            world_model_pretrain_mlp_chunk=65536, world_model_pretrain_activation_checkpointing=True,
            world_model_imagination_precision='bf16', world_model_imagination_compile=True,
            imagination_rollouts_per_pass=16, imagination_passes_per_episode=4,
            imagination_time_budget_seconds=0.),
        method=replace(config.method, fm_steps=100,
                       fm_solver='euler', disagreement_samples=2),
        codec=replace(config.codec, decoder_pretrain_enabled=True, decoder_online_enabled=True,
            decoder_allow_unready=True, occupied_speed_weight=1., slow_speed_weight=1.,
            decoder_pretrain_learning_rate=1e-4, decoder_online_learning_rate=1e-5),
        data=DataConfig(archive_roots=(), cache_episodes=2,
                       pretrain_cache_manifest=args.cache_manifest,
                       validation_cache_manifest=args.cache_manifest))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    destination = args.run_dir / 'pretrain_config.toml'
    if destination.exists():
        if load_config(destination).resolved_snapshot() != config.resolved_snapshot():
            raise ValueError('frozen execution configuration changed')
    else:
        export_run_config(config, destination)
    return destination


def commands(args, plan, config):
    from urban_dreamer.workflow.config import load_config
    frozen = load_config(config)
    pretrain = [sys.executable, '-u', str(args.source_root / 'tools/run_complete_pretrain.py'),
        '--config', str(config), '--calibrated-codec', str(args.calibrated_codec),
        '--cache-manifest', str(args.cache_manifest), '--output-dir', str(args.run_dir / 'world_model'),
        '--device', args.device, '--threads', '4', '--batch-size', str(plan['batch_size']),
        '--accumulation-steps', '1', '--space-chunk', '64', '--time-chunk', '65536',
        '--mlp-chunk', '65536', '--activation-checkpointing', 'on', '--compile-blocks', 'on',
        '--gpu-memory-fraction', str(args.gpu_memory_fraction), '--precision', 'bf16',
        '--sequence-length', str(frozen.resources.world_model_initialization_sequence_length),
        '--fm-training-mode', frozen.method.fm_training_mode,
        '--readout-supervision', frozen.method.exposure_readout_supervision,
        '--expected-train-episodes', str(plan['train_episodes']),
        '--expected-validation-episodes', str(plan['validation_episodes'])]
    if getattr(args, 'calibration_binding', None) is not None:
        pretrain.extend(['--calibration-binding', str(args.calibration_binding)])
    online = [sys.executable, '-u', str(args.source_root / 'tools/queue_pretrained_online.py'),
        '--pretrain-dir', str(args.run_dir / 'world_model'), '--execution-source', str(args.source_root),
        '--queue-dir', str(args.run_dir / 'online_supervisor'), '--seed-dir', str(args.run_dir / 'online_seed'),
        '--output-dir', str(args.run_dir / 'online_actor_critic_2000'), '--device', args.device]
    return pretrain, online


def run_child(command, log_path, *, cwd, on_launch=lambda pid: None, popen=subprocess.Popen):
    environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2')
    environment.pop('URB_ACTOR_ENTROPY_MODE', None)
    with Path(log_path).open('ab', buffering=0) as log:
        child = popen(command, cwd=cwd, env=environment, stdout=log,
                      stderr=subprocess.STDOUT, start_new_session=True)
        on_launch(child.pid)
        try:
            code = child.wait()
        except BaseException:
            # Only the process group created by this controller is signalled.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait()
            raise
    if code:
        raise RuntimeError(f'child exited {code}: {command[2]}')


def stop_owned_online(args):
    """The queue deliberately leaves its child running; stop only our bound child."""
    path = args.run_dir / 'online_supervisor/status.json'
    if not path.exists():
        return
    state = json.loads(path.read_text())
    pid = state.get('child_pid')
    if not pid:
        return
    try:
        process = Path(f'/proc/{pid}')
        raw = (process / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        command = (process / 'cmdline').read_bytes().decode().split('\0')
        if (fields[19] == state.get('child_identity') and
                str(args.source_root / 'tools/run_online_pilot.py') in command and
                str(args.run_dir / 'online_actor_critic_2000') in command):
            os.killpg(pid, signal.SIGTERM)
    except (FileNotFoundError, ProcessLookupError):
        pass


def run(args):
    from urban_dreamer.workflow.config import load_config
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.run_dir / 'status.json'
    def status(phase, **extra):
        atomic_json(status_path, dict(phase=phase, controller_pid=os.getpid(),
                    updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), **extra))
    status('waiting_data')
    try:
        while True:
            cache, benchmark = completed_input(args.cache_manifest), completed_input(args.benchmark)
            for failure_path in (args.cache_manifest.parent / 'status.json', args.benchmark.parent / 'status.json'):
                if failure_path.exists():
                    completed_input(failure_path)
            if cache is not None and benchmark is not None and preparation_verified(args.cache_manifest):
                break
            preparation = args.cache_manifest.parent / 'status.json'
            state = json.loads(preparation.read_text()) if preparation.exists() else {}
            status('waiting_data', data_stage=state.get('stage', state.get('status')),
                   data_progress=state.get('progress', {}),
                   benchmark_ready=benchmark is not None)
            time.sleep(args.poll_seconds)
        saved_plan = args.run_dir / 'training_plan.json'
        if saved_plan.exists():
            recorded = json.loads(saved_plan.read_text())
            memory = recorded.get('available_memory_mib')
        elif args.device.startswith('cuda:'):
            device_index = args.device.split(':')[1]
            # Query without initializing a CUDA context in this waiting controller.
            memory = float(subprocess.check_output(['nvidia-smi', '--id', device_index,
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        else:
            memory = None
        validate_benchmark_profile(load_config(args.config), benchmark)
        plan = make_plan(cache, benchmark, available_memory_mib=memory)
        frozen_paths = [args.config, args.calibrated_codec, args.cache_manifest, args.benchmark,
                        args.source_root / 'manifest.json']
        if getattr(args, 'calibration_binding', None) is not None:
            frozen_paths.append(args.calibration_binding)
        plan['frozen_inputs'] = {str(path): digest(path) for path in frozen_paths}
        plan['device'] = args.device
        plan['gpu_memory_fraction'] = args.gpu_memory_fraction
        immutable_plan(args.run_dir / 'training_plan.json', plan)
        config = write_config(args, plan)
        pretrain, online = commands(args, plan, config)
        atomic_json(args.run_dir / 'commands.json', dict(pretrain=pretrain, online=online))
        status('planned', plan=str(args.run_dir / 'training_plan.json'))
        for attempt in range(1, 4):
            try:
                status('pretraining', attempt=attempt, command=pretrain)
                run_child(pretrain, args.run_dir / 'pretrain.log', cwd=args.source_root,
                    on_launch=lambda pid: status('pretraining', attempt=attempt, child_pid=pid, command=pretrain))
                break
            except RuntimeError:
                if attempt == 3:
                    raise
        summary = json.loads((args.run_dir / 'world_model/pretrain_summary.json').read_text())
        if summary.get('status') != 'pretrained':
            raise RuntimeError('pretraining exited without a successful completion summary')
        status('online', command=online)
        run_child(online, args.run_dir / 'online_queue.log', cwd=args.source_root,
                  on_launch=lambda pid: status('online', child_pid=pid, command=online))
        queue = json.loads((args.run_dir / 'online_supervisor/status.json').read_text())
        if queue.get('phase') != 'completed':
            raise RuntimeError(f'online queue did not complete: {queue.get("phase")}')
        status('completed')
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            stop_owned_online(args)
        status('failed', error=repr(error))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'config', 'calibrated-codec', 'source-root', 'cache-manifest', 'benchmark'):
        parser.add_argument('--' + name, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--calibration-binding', type=lambda value: Path(value).resolve())
    parser.add_argument('--gpu-memory-fraction', type=float, default=.25)
    parser.add_argument('--poll-seconds', type=float, default=10.)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('controller terminated')))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()
