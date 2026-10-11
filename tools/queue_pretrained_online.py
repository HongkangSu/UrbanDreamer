"""Queue the full online actor–critic run after a completed offline pretrain.

This process owns only its queue directory, immutable seed and new online output.
It never signals the pretrainer or changes its checkpoints.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import BUDGETS, atomic_json, sha256_file, learning_budgets
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config
from urban_dreamer.learning.global_reward import REWARD_CONTRACT
from urban_dreamer.learning.replay import REPLAY_SCHEMA_VERSION

COMPILE_FIELD = 'world_model_imagination_compile'
LEGACY_REWARD_CONTRACT = 'event_q_counterfactual_global_individual_timeout_v2'


def checkpoint_source_config(config, raw):
    """Select only the exact historical fingerprint declared by a legacy source."""
    contract = raw['training_state']['reward_contract']
    if contract == LEGACY_REWARD_CONTRACT:
        from urban_dreamer.workflow.config import legacy_semantic_fingerprint
        config = replace(config, semantic_fingerprint=legacy_semantic_fingerprint(config))
    elif contract != REWARD_CONTRACT:
        raise ValueError('unsupported source checkpoint reward contract')
    if raw['config_fingerprint'] != config.semantic_fingerprint:
        raise ValueError('source checkpoint reward contract and semantic fingerprint disagree')
    return config


def _expand_context_columns(tensor, *, old_width, insertion):
    if tensor.ndim != 2 or tensor.shape[1] != old_width:
        raise ValueError('legacy critic context weight or optimizer moment has unexpected shape')
    return torch.cat((tensor[:, :insertion], tensor.new_zeros((tensor.shape[0], 2)),
                      tensor[:, insertion:]), dim=1)


def migrate_pretrained_reward(state):
    """Retain pretraining knowledge, explicitly migrate critic context and replay.

    This is only the zero-online-episode handoff. Old online rewards cannot be
    reconstructed from the archived two-dimensional context.
    """
    if state['reward_contract'] == REWARD_CONTRACT:
        return copy.deepcopy(state), dict(kind='already_causal_global', changed_fields=[],
                                          training_state_preserved_exactly=True)
    if state['reward_contract'] != LEGACY_REWARD_CONTRACT:
        raise ValueError('unsupported pretraining reward contract')
    if (state['next_environment_episode'] != 0 or state['metrics'].get('online_episodes_completed', 0)
            or state['replay']['online']['episodes']):
        raise ValueError('paper reward seed migration requires zero prior online activity')
    target = copy.deepcopy(state)
    changes = ['reward_contract', 'replay.online.schema_version']
    for name, key, width, insertion in (('latent_critic', 'trunk.0.weight', 66, 66),
                                       ('event_critic', 'scorer.0.weight', 413, 147)):
        model = target['models'][name]
        model[key] = _expand_context_columns(model[key], old_width=width, insertion=insertion)
        changes.append(f'models.{name}.{key}')
        optimizer = target['optimizers'].get(name)
        if optimizer is not None:
            # Both critics register their first Linear weight as parameter zero.
            parameters = [parameter for group in optimizer['param_groups'] for parameter in group['params']]
            first_state = optimizer['state'].get(parameters[0], {}) if parameters else {}
            for moment in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if moment in first_state:
                    first_state[moment] = _expand_context_columns(
                        first_state[moment], old_width=width, insertion=insertion)
                    changes.append(f'optimizers.{name}.state.first_weight.{moment}')
    online = target['replay']['online']
    online.update(schema_version=REPLAY_SCHEMA_VERSION, episodes=[], sample_order=[], sample_cursor=0)
    target['reward_contract'] = REWARD_CONTRACT
    for name in ('world_model', 'actor'):
        _assert_equal(state['models'][name], target['models'][name])
    for name in ('rng', 'metrics', 'next_environment_episode'):
        _assert_equal(state[name], target[name])
    for name in ('offline', 'codec_validation'):
        _assert_equal(state['replay'][name], target['replay'][name])
    retained_optimizers = sorted(set(state['optimizers']) - {'latent_critic', 'event_critic'})
    for name in retained_optimizers:
        _assert_equal(state['optimizers'][name], target['optimizers'][name])
    for name, first_weight in (('latent_critic', 'trunk.0.weight'),
                                ('event_critic', 'scorer.0.weight')):
        for key in state['models'][name]:
            if key != first_weight:
                _assert_equal(state['models'][name][key], target['models'][name][key])
    return target, dict(kind='pretrain_to_causal_global', changed_fields=changes,
        training_state_preserved_exactly=False,
        preserved_exactly=['models.world_model', 'models.actor', 'rng', 'metrics',
                          'next_environment_episode', 'replay.offline', 'replay.codec_validation',
                          *['optimizers.' + name for name in retained_optimizers]],
        critic_new_context_columns='zero initialized; event critic inserts at column 147',
        online_replay='empty source retained, schema upgraded; no old reward targets reused')


def process_identity(pid):
    """Linux process start time prevents PID reuse from identifying another job."""
    if not pid:
        return None
    try:
        content = Path(f'/proc/{pid}/stat').read_text()
        fields = content[content.rfind(')') + 2:].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_is_same(pid, identity):
    return identity is not None and process_identity(pid) == identity


def validate_completed_source(checkpoint, config):
    state, metrics = checkpoint.training_state, checkpoint.training_state['metrics']
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING or checkpoint.next_unit != 0:
        raise ValueError('source is not a completed-pretrain online-start checkpoint')
    if metrics.get('world_model_initialization_epochs_completed') != config.run.world_model_initialization_epochs:
        raise ValueError('source has not completed all configured initialization epochs')
    if (metrics.get('online_episodes_completed', 0) != 0 or
            state['next_environment_episode'] != 0 or state['replay']['online']['episodes']):
        raise ValueError('source already contains online activity')
    for name in ('actor', 'latent_critic', 'event_critic', 'world_model_online', 'decoder_online'):
        if state['optimizers'].get(name, {}).get('state'):
            raise ValueError('source contains trained online optimizers')
    validation = state['replay'].get('codec_validation') or {}
    identity = validation.get('manifest_identity', validation.get('identity'))
    override = (config.codec.decoder_allow_unready and config.codec.decoder_pretrain_enabled
                and config.codec.decoder_online_enabled)
    if (not metrics.get('codec_ready') and not override) or not identity:
        raise ValueError('source lacks a ready codec validation proof')
    if (metrics.get('codec_validation_manifest_identity') != identity or
            metrics.get('codec_validation_updates') != metrics.get('codec_updates')):
        raise ValueError('source codec proof identity or update differs')
    if config.codec.decoder_pretrain_enabled or config.codec.decoder_online_enabled:
        revision = metrics.get('decoder_pretrain_updates', 0) + metrics.get('decoder_online_updates', 0)
        if metrics.get('codec_validation_decoder_updates', 0) != revision:
            raise ValueError('source codec validation decoder revision differs')
    for name, model in state.get('models', {}).items():
        if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all())
               for value in model.values()):
            raise ValueError(f'source contains nonfinite model weights: {name}')
    limits = {'codec_count_mae_occupied': config.codec.occupied_vehicle_mae_max,
              'codec_count_mae_all': config.codec.vehicle_mae_max,
              'codec_count_mae_empty': config.codec.empty_vehicle_mae_max,
              'codec_total_relative_error': config.codec.total_relative_mae_max,
              'codec_speed_mae_mps': config.codec.speed_mae_mps_max}
    if config.codec.decoder_pretrain_enabled or config.codec.decoder_online_enabled:
        limits.update(codec_speed_mae_occupied_mps=config.codec.occupied_speed_mae_mps_max,
                      codec_speed_mae_slow_mps=config.codec.slow_speed_mae_mps_max)
    for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
        for key, limit in limits.items():
            if key == 'codec_speed_mae_slow_mps' and metrics.get(prefix + 'codec_slow_roads', 0) == 0:
                if not override and config.codec.slow_validation_min_roads > 0:
                    raise ValueError('source codec validation lacks required slow roads')
                continue
            value = metrics.get(prefix + key, math.nan)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not override and value > limit):
                raise ValueError(f'source codec quality gate failed: {prefix + key}')
    method, simulation, resources = config.method, config.simulation, config.resources
    if (method.fm_steps != 100 or method.fm_solver != 'euler' or
            method.fm_training_mode != 'single_step' or method.sp_prior_weight != 0 or
            method.actor_entropy_mode not in {'population', 'annealed'} or simulation.traffic_model != 'meso' or
            simulation.session_protocol != 'event' or simulation.observation_step_seconds != 1 or
            simulation.incident_start_seconds < 1000000):
        raise ValueError('source does not match the approved Euler/no-incident/meso/prior/entropy settings')
    if (resources.world_model_imagination_precision not in {'fp32', 'bf16', 'tf32'} or
            resources.world_model_imagination_compile_mode != 'default' or
            (resources.world_model_imagination_space_chunk,
             resources.world_model_imagination_time_chunk,
             resources.world_model_imagination_mlp_chunk) != (16, 32768, 16384)):
        raise ValueError('source imagination precision or compile mode differs')


def inspect_source(source_dir, *, pretrain_pid=None, pretrain_identity=None):
    source = Path(source_dir)
    summary_path = source / 'pretrain_summary.json'
    if not summary_path.exists():
        return dict(ready=False, reason='waiting_for_completed_pretrain_summary')
    summary = json.loads(summary_path.read_text())
    if summary.get('status') != 'pretrained':
        raise ValueError('pretrain summary does not report successful completion')
    if process_is_same(pretrain_pid, pretrain_identity):
        return dict(ready=False, reason='waiting_for_pretrainer_exit')
    config = load_config(source / 'run_config.toml')
    raw = torch.load(latest_checkpoint_path(source), map_location='cpu', weights_only=True)
    config = checkpoint_source_config(config, raw)
    checkpoint = _read_frozen(latest_checkpoint_path(source), config, raw['input_hash'])
    validate_completed_source(checkpoint, config)
    return dict(ready=True, reason='completed_pretrain_verified',
                source_fingerprint=config.semantic_fingerprint,
                source_checkpoint_sha256=sha256_file(checkpoint.path),
                epochs=config.run.world_model_initialization_epochs, codec_ready=bool(checkpoint.training_state['metrics'].get('codec_ready')), updates=checkpoint.training_state['metrics'].get('world_model_updates'))


def target_config(source, seed_dir):
    from tools.continue_mainline_entropy_annealing import annealed_config
    config = annealed_config(source, seed_dir)
    config = replace(config, method=replace(config.method, disagreement_weight=0.))
    config = replace(config, resources=replace(source.resources, **{COMPILE_FIELD: True}))
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def validate_profile_transfer(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        snapshot['resources'].pop(COMPILE_FIELD)
        snapshot['run'].pop('output_dir')
        for name in ('actor_entropy_mode', 'actor_entropy_initial_coefficient',
                     'actor_entropy_final_coefficient', 'actor_entropy_hold_fraction',
                     'actor_entropy_end_fraction'):
            snapshot['method'].pop(name, None)
        snapshot['method'].pop('disagreement_weight')
    from tools.continue_mainline_entropy_annealing import annealed_config
    expected_method = replace(annealed_config(source, target.run.output_dir).method, disagreement_weight=0.)
    if (before != after or not target.resources.world_model_imagination_compile
            or target.method != expected_method):
        raise ValueError('seed may change only entropy schedule, global reward risk setting, imagination compile flag and output location')


def stage_seed(source_dir, seed_dir):
    """Atomic immutable initialization with an explicit reward-contract migration."""
    source, seed = Path(source_dir).resolve(), Path(seed_dir).resolve()
    source_config = load_config(source / 'run_config.toml')
    source_path = latest_checkpoint_path(source)
    source_hash, config_hash = sha256_file(source_path), sha256_file(source / 'run_config.toml')
    if seed.exists():
        manifest = json.loads((seed / 'handoff_manifest.json').read_text())
        if (manifest['source_checkpoint_sha256'] != source_hash or
                manifest['source_config_sha256'] != config_hash):
            raise ValueError('completed pretrain source changed after seed was frozen')
        if manifest.get('reward_contract') != REWARD_CONTRACT:
            raise ValueError('existing seed uses another reward contract; use a fresh seed directory')
        for filename, digest in manifest['frozen_artifacts'].items():
            if sha256_file(seed / filename) != digest:
                raise ValueError(f'immutable seed artifact changed: {filename}')
        return manifest
    raw = torch.load(source_path, map_location='cpu', weights_only=True)
    source_config = checkpoint_source_config(source_config, raw)
    checkpoint = _read_frozen(source_path, source_config, raw['input_hash'])
    validate_completed_source(checkpoint, source_config)
    config = target_config(source_config, seed)
    validate_profile_transfer(source_config, config)
    seed.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + seed.name + '-', dir=seed.parent))
    try:
        shutil.copy2(source_path, temporary / 'source_checkpoint.pt')
        shutil.copy2(source / 'run_config.toml', temporary / 'source_config.toml')
        shutil.copy2(source / 'pretrain_summary.json', temporary / 'source_pretrain_summary.json')
        if (sha256_file(source_path) != source_hash or
                sha256_file(temporary / 'source_checkpoint.pt') != source_hash or
                sha256_file(temporary / 'source_config.toml') != config_hash):
            raise ValueError('source changed during immutable seed staging')
        export_run_config(config, temporary / 'run_config.toml')
        training_state, reward_migration = migrate_pretrained_reward(checkpoint.training_state)
        saved = save_checkpoint(output_dir=temporary, config_fingerprint=config.semantic_fingerprint,
            input_hash=checkpoint.input_hash, phase=checkpoint.phase, next_unit=checkpoint.next_unit,
            training_state=training_state)
        transferred = _read_frozen(saved, config, checkpoint.input_hash)
        _assert_equal(training_state, transferred.training_state)
        files = ('source_checkpoint.pt', 'source_config.toml', 'source_pretrain_summary.json',
                 'run_config.toml', 'checkpoints/latest.pt')
        manifest = dict(schema_version=1, source_output=str(source), seed_output=str(seed),
            source_checkpoint_sha256=source_hash, source_config_sha256=config_hash,
            source_config_fingerprint=source_config.semantic_fingerprint,
            target_config_fingerprint=config.semantic_fingerprint, input_hash=checkpoint.input_hash,
            changed_fields=[f'resources.{COMPILE_FIELD}', 'run.output_dir', 'method.actor_entropy_mode',
                'method.actor_entropy_initial_coefficient', 'method.actor_entropy_final_coefficient',
                'method.actor_entropy_hold_fraction', 'method.actor_entropy_end_fraction',
                'method.disagreement_weight'], reward_contract=REWARD_CONTRACT,
            reward_migration=reward_migration,
            training_state_preserved_exactly=reward_migration['training_state_preserved_exactly'],
            source_counters=checkpoint.training_state['metrics'],
            online_budgets=learning_budgets(config), frozen_artifacts={name: sha256_file(temporary / name) for name in files})
        atomic_json(temporary / 'handoff_manifest.json', manifest)
        temporary.rename(seed)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_execution_source(source):
    source = Path(source).resolve()
    manifest = json.loads((source / 'source_online_manifest.json').read_text())
    entries = manifest.get('files', manifest.get('frozen_artifacts'))
    if not isinstance(entries, dict) or not entries:
        raise ValueError('execution source requires nonempty frozen file hashes')
    for filename, digest in entries.items():
        path = (source / filename).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'frozen execution source changed: {filename}')
    if 'tools/run_online_pilot.py' not in entries:
        raise ValueError('execution manifest does not freeze the online runner')
    return sha256_file(source / 'source_online_manifest.json')


def choose_device(requested, minimum_free_gib=3.):
    if requested == 'cpu':
        return 'cpu'
    if not torch.cuda.is_available():
        return None
    candidates = range(min(torch.cuda.device_count(), 2)) if requested == 'auto' else [int(requested.split(':')[1])]
    available = []
    for index in candidates:
        if index < torch.cuda.device_count():
            free, _ = torch.cuda.mem_get_info(index)
            if free >= minimum_free_gib * 1024 ** 3:
                available.append((free, index))
    if not available:
        return None
    # Select maximal headroom; choose GPU1 on exact ties.
    return f'cuda:{max(available)[1]}'


def launch_command(args, device):
    return [str(args.python), str(args.execution_source / 'tools/run_online_pilot.py'),
            '--source-dir', str(args.seed_dir), '--output-dir', str(args.output_dir),
            '--device', device, '--online-episodes', '2000', '--evaluation-every', '20',
            '--threads', '4', '--gpu-memory-fraction', '.25']


def recover_unrecorded_child(args):
    """Close the crash window between Popen and persisting the child PID."""
    expected_script = str(args.execution_source / 'tools/run_online_pilot.py')
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if (len(command) < 2 or command[1] != expected_script or '--output-dir' not in command):
                continue
            index = command.index('--output-dir')
            if index + 1 < len(command) and command[index + 1] == str(args.output_dir):
                identity = process_identity(int(directory.name))
                if identity:
                    matches.append((int(directory.name), identity))
        except (FileNotFoundError, ProcessLookupError, PermissionError, UnicodeDecodeError):
            continue
    if len(matches) > 1:
        raise ValueError('multiple live online runners target this queue output')
    return matches[0] if matches else None


@contextmanager
def queue_lock(queue_dir):
    directory = Path(queue_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'watcher.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another watcher owns this queue') from error
        yield


def run_queue(args):
    status_path = args.queue_dir / 'status.json'
    stop = False
    def stop_waiting(_signal, _frame):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, stop_waiting)
    signal.signal(signal.SIGINT, stop_waiting)
    with queue_lock(args.queue_dir):
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        binding = dict(pretrain_dir=str(args.pretrain_dir), seed_dir=str(args.seed_dir),
                       output_dir=str(args.output_dir), execution_source=str(args.execution_source))
        if state.get('binding', binding) != binding:
            raise ValueError('queue restart configuration differs from its saved binding')
        state.update(binding=binding, watcher_pid=os.getpid())
        execution_hash = verify_execution_source(args.execution_source)
        config_hash = sha256_file(args.pretrain_dir / 'run_config.toml')
        for key, digest in (('execution_manifest_sha256', execution_hash),
                            ('pretrain_config_sha256', config_hash)):
            if state.get(key, digest) != digest:
                raise ValueError(f'queue frozen identity changed: {key}')
            state[key] = digest
        previous_identity = state.get('pretrain_identity', process_identity(args.pretrain_pid))
        state['pretrain_identity'] = previous_identity
        owned = None
        while not stop:
            if state.get('phase') == 'launching' and not state.get('child_pid'):
                recovered = recover_unrecorded_child(args)
                if recovered:
                    state.update(child_pid=recovered[0], child_identity=recovered[1])
            child_alive = process_is_same(state.get('child_pid'), state.get('child_identity'))
            if child_alive:
                state['phase'] = 'online_running'
            elif state.get('child_pid'):
                returncode = owned.poll() if owned is not None else None
                progress = args.output_dir / 'pilot_progress.json'
                completed = json.loads(progress.read_text()) if progress.exists() else {}
                if completed.get('phase') == 'completed' and len(completed.get('online', [])) == 2000:
                    state.update(phase='completed', returncode=returncode)
                    atomic_json(status_path, state)
                    return state
                state.update(phase='online_retry_pending', previous_returncode=returncode,
                             child_pid=None, child_identity=None)
                owned = None
            if not child_alive:
                readiness = inspect_source(args.pretrain_dir, pretrain_pid=args.pretrain_pid,
                                           pretrain_identity=previous_identity)
                state['readiness'] = readiness
                if not readiness['ready']:
                    state['phase'] = 'waiting_for_pretrain'
                else:
                    execution_hash = verify_execution_source(args.execution_source)
                    if state.get('execution_manifest_sha256', execution_hash) != execution_hash:
                        raise ValueError('queue frozen execution manifest changed')
                    state['execution_manifest_sha256'] = execution_hash
                    manifest = stage_seed(args.pretrain_dir, args.seed_dir)
                    state['seed_manifest'] = str(args.seed_dir / 'handoff_manifest.json')
                    state['source_checkpoint_sha256'] = manifest['source_checkpoint_sha256']
                    attempts = state.get('launch_attempts', 0)
                    if attempts >= args.max_launch_attempts:
                        state['phase'] = 'failed_retry_limit'
                        atomic_json(status_path, state)
                        return state
                    device = choose_device(state.get('device') or args.device, args.minimum_free_gib)
                    if device is None:
                        state['phase'] = 'waiting_for_gpu'
                    else:
                        if (attempts == 0 and args.output_dir.exists() and
                                any(args.output_dir.iterdir())):
                            raise ValueError('first online launch requires a fresh output directory')
                        command = launch_command(args, device)
                        environment = dict(os.environ, URB_IMAGINATION_WORKERS='8',
                            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                            TORCHINDUCTOR_COMPILE_THREADS='2')
                        environment.pop('URB_ACTOR_ENTROPY_MODE', None)
                        environment.setdefault('TORCHINDUCTOR_CACHE_DIR',
                                               str(args.output_dir.parent / 'inductor_cache'))
                        state.update(phase='launching', command=command, device=device,
                                     launch_attempts=attempts + 1)
                        atomic_json(status_path, state)
                        try:
                            with (args.queue_dir / 'online.log').open('ab', buffering=0) as log:
                                owned = subprocess.Popen(command, cwd=args.execution_source, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        except OSError as error:
                            state.update(phase='online_retry_pending', launch_error=repr(error))
                        else:
                            state.update(phase='online_running', child_pid=owned.pid,
                                         child_identity=process_identity(owned.pid))
            state['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            atomic_json(status_path, state)
            if args.once_until_ready:
                return state
            # Signal wakes the loop within one second, without touching the child.
            for _ in range(args.poll_seconds):
                if stop:
                    break
                time.sleep(1)
        state.update(watcher_stopped=True, child_left_running=True)
        atomic_json(status_path, state)
        return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrain-dir', type=Path, required=True)
    parser.add_argument('--execution-source', type=Path, required=True)
    parser.add_argument('--queue-dir', type=Path, required=True)
    parser.add_argument('--seed-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pretrain-pid', type=int)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--device', choices=('auto', 'cuda:0', 'cuda:1', 'cpu'), default='auto')
    parser.add_argument('--minimum-free-gib', type=float, default=3.)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--max-launch-attempts', type=int, default=3)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--once-until-ready', action='store_true',
                        help='perform one poll (may launch when ready), then leave child running')
    args = parser.parse_args(argv)
    if (not 1 <= args.poll_seconds <= 60 or args.max_launch_attempts < 1 or
            not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib <= 0):
        parser.error('poll must be 1..60 seconds, retry count and GPU headroom positive')
    for key in ('pretrain_dir', 'execution_source', 'queue_dir', 'seed_dir', 'output_dir', 'python'):
        setattr(args, key, getattr(args, key).resolve())
    torch.set_num_threads(1)
    if args.inspect_only:
        result = inspect_source(args.pretrain_dir, pretrain_pid=args.pretrain_pid,
                                pretrain_identity=process_identity(args.pretrain_pid))
    else:
        try:
            result = run_queue(args)
        except Exception as error:
            # Do not replace another watcher's status when lock acquisition fails.
            if not isinstance(error, RuntimeError) or 'owns this queue' not in str(error):
                atomic_json(args.queue_dir / 'failure.json', dict(error=repr(error)))
            raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
