"""Stage entropy-only source revisions and lossless online checkpoint continuations.

The frozen pretrainer source is never edited. Completed online runs are refused;
existing weights, Adam state, replay, RNG and global episode counters survive.
Process ownership and restart remain with the training controller.
"""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import replace
import fcntl
import json
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

from tools.migrate_mainline_entropy import _validate_cursor
from tools.migrate_mainline_prior import _assert_equal, _read_frozen
from tools.run_complete_pretrain import export_run_config
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.learning.entropy_schedule import SCHEDULE_FIELDS, entropy_schedule_profile
from urban_dreamer.workflow.checkpoint import (CheckpointPhase, latest_checkpoint_path,
                                              load_checkpoint, _validate_loaded_payload)
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def entropy_protocol(method):
    return dict(mode='annealed', initial_coefficient=method.actor_entropy_initial_coefficient,
                final_coefficient=method.actor_entropy_final_coefficient,
                hold_fraction=method.actor_entropy_hold_fraction,
                end_fraction=method.actor_entropy_end_fraction,
                schedule='cosine_global_completed_episode', coefficient='fixed_without_population_scaling')


def annealed_config(source, output, *, device=None):
    target = replace(source,
        run=replace(source.run, output_dir=Path(output).resolve(), device=device or source.run.device),
        method=replace(source.method, actor_entropy_mode='annealed',
            actor_entropy_initial_coefficient=.002, actor_entropy_final_coefficient=.0001,
            actor_entropy_hold_fraction=.1, actor_entropy_end_fraction=.8))
    return replace(target, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(target, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def validate_transfer(source, target):
    before, after = source.resolved_snapshot(), target.resolved_snapshot()
    for snapshot in (before, after):
        for key in ('actor_entropy_mode', *SCHEDULE_FIELDS):
            snapshot['method'].pop(key, None)
        for key in ('output_dir', 'device'):
            snapshot['run'].pop(key)
    if before != after or target.method.actor_entropy_mode != 'annealed':
        raise ValueError('annealing continuation may change only entropy settings, output location and device')
    expected = annealed_config(source, target.run.output_dir, device=target.run.device)
    if target.method != expected.method:
        raise ValueError('annealing continuation must use approved .002 to .0001 cosine schedule')


def validate_continuation(output_dir, config, input_hash, pretrained_source_hash):
    output = Path(output_dir).resolve()
    manifest = json.loads((output / 'entropy_migration.json').read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != 'fixed_entropy_annealing_continuation'
            or manifest['target_output'] != str(output)
            or manifest['target_config_fingerprint'] != config.semantic_fingerprint
            or manifest['input_hash'] != input_hash
            or manifest['actor_entropy'] != entropy_protocol(config.method)):
        raise ValueError('annealing continuation target identity differs')
    for name, digest in manifest['frozen_artifacts'].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f'annealing continuation frozen {name} hash differs')
    source_config = load_config(output / 'source_config.toml')
    validate_transfer(source_config, config)
    protocol = json.loads((output / 'source_protocol.json').read_text())
    if (source_config.semantic_fingerprint != manifest['source_config_fingerprint']
            or protocol['runtime_config_fingerprint'] != source_config.semantic_fingerprint
            or protocol['source']['sha256'] != pretrained_source_hash
            or protocol['online_episode_budget'] != config.run.online_training_episodes):
        raise ValueError('annealing continuation source ancestry or budget differs')
    source = _read_frozen(output / 'source_checkpoint.pt', source_config, input_hash)
    initial = _read_frozen(output / 'continuation_checkpoint.pt', config, input_hash)
    _validate_cursor(source)
    _validate_cursor(initial)
    if (source.phase is not CheckpointPhase.ONLINE_TRAINING or initial.phase != source.phase
            or source.next_unit != initial.next_unit or source.next_unit != manifest['global_completed_episodes']
            or source.next_unit >= config.run.online_training_episodes):
        raise ValueError('annealing continuation requires remaining episodes at unchanged cursor')
    _assert_equal(source.training_state, initial.training_state)
    latest = load_checkpoint(output_dir=output, expected_config_fingerprint=config.semantic_fingerprint,
                             expected_input_hash=input_hash)
    progress = _validate_cursor(latest)
    if not source.next_unit <= len(progress['online']) <= config.run.online_training_episodes:
        raise ValueError('annealing latest checkpoint moved backwards or exceeded original budget')
    return manifest


def migrate(source_dir, output_dir, *, device=None):
    """Prepare a separate continuation; do not signal or launch any process."""
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists() or output == source:
        raise ValueError('annealing continuation requires a new absent output directory')
    original_config = load_config(source / 'run_config.toml')
    config = annealed_config(original_config, output, device=device)
    validate_transfer(original_config, config)
    raw = torch.load(latest_checkpoint_path(source), map_location='cpu', weights_only=True)
    original = _validate_loaded_payload(payload=raw, path=latest_checkpoint_path(source),
        expected_config_fingerprint=original_config.semantic_fingerprint, expected_input_hash=raw['input_hash'])
    progress = _validate_cursor(original)
    if original.phase is not CheckpointPhase.ONLINE_TRAINING or original.next_unit >= config.run.online_training_episodes:
        raise ValueError('completed runs must not restart or extend their episode budget')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        shutil.copy2(source / 'run_config.toml', temporary / 'source_config.toml')
        shutil.copy2(source / 'pilot_protocol.json', temporary / 'source_protocol.json')
        # Save exactly the payload read above, avoiding a concurrent latest-file publication race.
        torch.save(raw, temporary / 'source_checkpoint.pt')
        source_frozen = _read_frozen(temporary / 'source_checkpoint.pt', original_config, original.input_hash)
        _assert_equal(source_frozen.training_state, original.training_state)
        raw['config_fingerprint'] = config.semantic_fingerprint
        torch.save(raw, temporary / 'continuation_checkpoint.pt')
        latest_checkpoint_path(temporary).parent.mkdir(parents=True)
        shutil.copy2(temporary / 'continuation_checkpoint.pt', latest_checkpoint_path(temporary))
        export_run_config(config, temporary / 'run_config.toml')
        names = ['source_config.toml', 'source_protocol.json', 'source_checkpoint.pt', 'continuation_checkpoint.pt']
        for name in ('offline_manifest.json', 'codec_validation_manifest.json',
                     'calibrated_noincident_initialization.json', 'meso_initialization.json',
                     'initialization_checkpoint.pt'):
            if (source / name).is_file():
                shutil.copy2(source / name, temporary / name)
                names.append(name)
        manifest = dict(schema_version=1, kind='fixed_entropy_annealing_continuation',
            source_output=str(source), target_output=str(output),
            source_config_fingerprint=original_config.semantic_fingerprint,
            target_config_fingerprint=config.semantic_fingerprint, input_hash=original.input_hash,
            global_completed_episodes=original.next_unit,
            original_episode_budget=config.run.online_training_episodes,
            actor_entropy=entropy_protocol(config.method),
            next_entropy=entropy_schedule_profile(config.method, completed_episodes=original.next_unit,
                                                 total_episodes=config.run.online_training_episodes),
            frozen_artifacts={name: sha256_file(temporary / name) for name in names},
            training_state_preserved_exactly=True)
        atomic_json(temporary / 'entropy_migration.json', manifest)
        protocol = json.loads((temporary / 'source_protocol.json').read_text())
        protocol.update(runtime_config_fingerprint=config.semantic_fingerprint, device=config.run.device,
                        actor_entropy=manifest['actor_entropy'], entropy_migration=manifest,
                        config_semantics_unchanged=False)
        atomic_json(temporary / 'pilot_protocol.json', protocol)
        atomic_json(temporary / 'pilot_progress.json', copy.deepcopy(progress))
        temporary.rename(output)
        validate_continuation(output, config, original.input_hash, protocol['source']['sha256'])
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('frozen source has an unsupported entropy implementation; refusing broad overlay')
    return text.replace(old, new, 1)


def _function(text, name):
    node = next(node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[node.lineno - 1:node.end_lineno])


def stage_source(source_root, revision_root, *, anneal_pretrained_online=False):
    """Copy the frozen revision, touching only entropy-specific code regions."""
    source, revision = Path(source_root).resolve(), Path(revision_root).resolve()
    if revision.exists() or source == revision:
        raise ValueError('entropy source revision requires a new absent directory')
    shutil.copytree(source, revision, symlinks=False)
    before = {str(path.relative_to(source)): sha256_file(path) for path in source.rglob('*') if path.is_file()}
    config_path = revision / 'urban_dreamer/workflow/config.py'
    text = config_path.read_text()
    defaults = ''.join(f'    {field}: float = {getattr(load_config(ROOT / "configs/ingolstadt_custom.toml").method, field)!r}\n'
                       for field in SCHEDULE_FIELDS)
    text = _replace_once(text, "    actor_entropy_mode: str = 'legacy'\n", "    actor_entropy_mode: str = 'legacy'\n" + defaults)
    text = _replace_once(text, '    method_values = asdict(method)\n',
        '    method_values = asdict(method)\n' + "    if method.actor_entropy_mode != 'annealed':\n"
        + f'        for name in {SCHEDULE_FIELDS!r}:\n            method_values.pop(name, None)\n')
    text = _replace_once(text,
        "    if parsed['actor_entropy_mode'] not in {'population', 'legacy', 'off'}:\n"
        "        raise ConfigError('[method].actor_entropy_mode must be population, legacy, or off')\n",
        "    if parsed['actor_entropy_mode'] not in {'population', 'legacy', 'off', 'annealed'}:\n"
        "        raise ConfigError('[method].actor_entropy_mode must be population, legacy, off, or annealed')\n"
        + f"    for name in {SCHEDULE_FIELDS!r}:\n        parsed[name] = _nonnegative_number(values, name, 'method')\n"
        "    if parsed['actor_entropy_final_coefficient'] > parsed['actor_entropy_initial_coefficient']:\n"
        "        raise ConfigError('entropy floor exceeds initial coefficient')\n"
        "    if not (parsed['actor_entropy_hold_fraction'] < parsed['actor_entropy_end_fraction'] <= 1):\n"
        "        raise ConfigError('invalid entropy schedule fractions')\n")
    config_path.write_text(text)
    policy_path = revision / 'urban_dreamer/models/policy.py'
    text = policy_path.read_text()
    policy_path.write_text(_replace_once(text, _function(text, 'population_scaled_entropy'),
        _function((ROOT / 'urban_dreamer/models/policy.py').read_text(), 'population_scaled_entropy')))
    shutil.copy2(ROOT / 'urban_dreamer/learning/entropy_schedule.py', revision / 'urban_dreamer/learning/entropy_schedule.py')
    training_path = revision / 'urban_dreamer/learning/training.py'
    text = training_path.read_text()
    text = _replace_once(text, 'from ..workflow.config import MainlineConfig, NONARRIVAL_PENALTY\n',
        'from ..workflow.config import MainlineConfig, NONARRIVAL_PENALTY\nfrom .entropy_schedule import entropy_schedule_profile\n')
    text = _replace_once(text,
        "            entropy_mode = os.environ.get('URB_ACTOR_ENTROPY_MODE', self.config.method.actor_entropy_mode)\n",
        "            entropy_mode = os.environ.get('URB_ACTOR_ENTROPY_MODE', self.config.method.actor_entropy_mode)\n"
        "            entropy_profile = entropy_schedule_profile(\n"
        "                replace(self.config.method, actor_entropy_mode=entropy_mode),\n"
        "                completed_episodes=int(self.metrics.get('online_episodes_completed', 0)),\n"
        "                total_episodes=self.config.run.online_training_episodes)\n")
    text = _replace_once(text, '                base_coef=ACTOR_ENTROPY_COEF, mode=entropy_mode)\n',
        "                base_coef=entropy_profile['coefficient'], mode=entropy_mode)\n")
    text = _replace_once(text, "            metrics['policy_entropy_mean'] = float(entropies.detach().mean())\n",
        "            metrics['policy_entropy_mean'] = float(entropies.detach().mean())\n"
        "            metrics['entropy_schedule_progress'] = entropy_profile['progress']\n"
        "            metrics['entropy_schedule_completed_episodes'] = entropy_profile['completed_episodes']\n"
        "            metrics['entropy_schedule_total_episodes'] = entropy_profile['total_episodes']\n")
    training_path.write_text(text)
    shutil.copy2(Path(__file__), revision / 'tools/continue_mainline_entropy_annealing.py')
    runner_path = revision / 'tools/run_online_pilot.py'
    text = runner_path.read_text()
    text = _replace_once(text,
        "        from tools.migrate_mainline_entropy import validate_entropy_continuation\n"
        "        validate_entropy_continuation(args.output_dir, config, input_hash, source_hash)\n",
        "        if entropy_migration.get('kind') == 'fixed_entropy_annealing_continuation':\n"
        "            from tools.continue_mainline_entropy_annealing import validate_continuation\n"
        "            validate_continuation(args.output_dir, config, input_hash, source_hash)\n"
        "        else:\n"
        "            from tools.migrate_mainline_entropy import validate_entropy_continuation\n"
        "            validate_entropy_continuation(args.output_dir, config, input_hash, source_hash)\n")
    text = _replace_once(text,
        "    if entropy_mode != 'legacy':\n",
        "    if entropy_mode == 'annealed':\n"
        "        from tools.continue_mainline_entropy_annealing import entropy_protocol\n"
        "        protocol['actor_entropy'] = entropy_protocol(config.method)\n"
        "        protocol['runtime_config_fingerprint'] = config.semantic_fingerprint\n"
        "    elif entropy_mode != 'legacy':\n")
    runner_path.write_text(text)
    if anneal_pretrained_online:
        queue_path = revision / 'tools/queue_pretrained_online.py'
        text = queue_path.read_text()
        text = _replace_once(text,
            "    config = replace(source, run=replace(source.run, output_dir=Path(seed_dir).resolve()),\n"
            "                     resources=replace(source.resources, **{COMPILE_FIELD: True}))\n",
            "    from tools.continue_mainline_entropy_annealing import annealed_config\n"
            "    config = annealed_config(source, seed_dir)\n"
            "    config = replace(config, resources=replace(source.resources, **{COMPILE_FIELD: True}))\n")
        text = _replace_once(text,
            "        snapshot['resources'].pop(COMPILE_FIELD)\n",
            "        snapshot['resources'].pop(COMPILE_FIELD)\n"
            + f"        for name in ('actor_entropy_mode', *{SCHEDULE_FIELDS!r}):\n"
            "            snapshot['method'].pop(name, None)\n")
        text = _replace_once(text,
            "    if before != after or not target.resources.world_model_imagination_compile:\n",
            "    from tools.continue_mainline_entropy_annealing import annealed_config\n"
            "    if (before != after or not target.resources.world_model_imagination_compile\n"
            "            or target.method != annealed_config(source, target.run.output_dir).method):\n")
        text = _replace_once(text,
            "            changed_fields=[f'resources.{COMPILE_FIELD}', 'run.output_dir'],\n",
            "            changed_fields=[f'resources.{COMPILE_FIELD}', 'run.output_dir', 'method.actor_entropy_mode',\n"
            + f"                            *['method.' + name for name in {SCHEDULE_FIELDS!r}]],\n")
        queue_path.write_text(text)
    after = {str(path.relative_to(revision)): sha256_file(path) for path in revision.rglob('*') if path.is_file()}
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    allowed = {'urban_dreamer/workflow/config.py', 'urban_dreamer/models/policy.py',
               'urban_dreamer/learning/training.py', 'urban_dreamer/learning/entropy_schedule.py',
               'tools/continue_mainline_entropy_annealing.py', 'tools/run_online_pilot.py'}
    if anneal_pretrained_online:
        allowed.add('tools/queue_pretrained_online.py')
    if set(changed) != allowed:
        raise ValueError(f'entropy overlay changed unexpected files: {changed}')
    for name in changed:
        ast.parse((revision / name).read_text(), filename=name)
    manifest = dict(schema_version=1, kind='entropy_only_source_overlay', source_root=str(source),
        revision_root=str(revision), changed_files=changed, frozen_source_hashes=before,
        files={name: digest for name, digest in after.items()
               if name != 'source_online_manifest.json' and '__pycache__' not in Path(name).parts
               and Path(name).suffix != '.pyc'},
        preserved='all other frozen training, model, replay, sampling, decoder and precision code')
    atomic_json(revision / 'source_online_manifest.json', manifest)
    atomic_json(revision / 'entropy_overlay.json', manifest)
    return manifest


def process_record(pid):
    """Bind every signal to the Linux start time and complete argument vector."""
    process = Path('/proc') / str(pid)
    try:
        raw = (process / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        command = (process / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        return dict(pid=int(pid), identity=fields[19], state=fields[0], command=command)
    except (FileNotFoundError, ProcessLookupError):
        return None


def verify_process(expected, *, stopped=None):
    current = process_record(expected['pid'])
    if (current is None or current['identity'] != expected['identity']
            or current['command'] != expected['command'] or current['state'] == 'Z'):
        raise ValueError('owned process identity or exact command changed')
    if stopped is not None and (current['state'] in {'T', 't'}) != stopped:
        raise ValueError('owned controller stop state differs')
    return current


def _argument(command, flag):
    return command[command.index(flag) + 1]


def _verify_revision(revision):
    manifest = json.loads((Path(revision) / 'source_online_manifest.json').read_text())
    for name, digest in manifest['files'].items():
        path = (Path(revision) / name).resolve()
        if not path.is_relative_to(Path(revision).resolve()) or sha256_file(path) != digest:
            raise ValueError(f'entropy execution source changed: {name}')
    return sha256_file(Path(revision) / 'source_online_manifest.json')


def prepare_future(run_dir, revision_root, plan_path):
    """Prepare the future online handoff without interrupting the pretrainer."""
    run, revision, path = (Path(value).resolve() for value in (run_dir, revision_root, plan_path))
    if path.exists():
        raise ValueError('future online plan requires a new absent path')
    status_path = run / 'status.json'
    status = json.loads(status_path.read_text())
    if status.get('phase') != 'pretraining':
        raise ValueError('future online handoff requires the still-running pretraining phase')
    controller = process_record(status['controller_pid'])
    pretrainer = process_record(status['child_pid'])
    if not controller or not pretrainer:
        raise ValueError('pretraining controller and worker must both still exist')
    source = Path(_argument(controller['command'], '--source-root')).resolve()
    if (str(source / 'tools/run_full_retrain.py') not in controller['command']
            or _argument(controller['command'], '--run-dir') != str(run)
            or str(source / 'tools/run_complete_pretrain.py') not in pretrainer['command']
            or _argument(pretrainer['command'], '--output-dir') != str(run / 'world_model')
            or status['command'] != pretrainer['command']):
        raise ValueError('pretraining process ownership does not match the selected full run')
    verify_process(controller, stopped=False)
    verify_process(pretrainer, stopped=False)
    for directory in ('online_seed', 'online_actor_critic_2000', 'online_supervisor'):
        if (run / directory).exists():
            raise ValueError('future online handoff requires untouched online output locations')
    old_plan = json.loads((run / 'training_plan.json').read_text())
    for name, digest in old_plan['frozen_inputs'].items():
        if sha256_file(name) != digest:
            raise ValueError(f'frozen pretraining input changed: {name}')
    stage_source(source, revision, anneal_pretrained_online=True)
    commands = json.loads((run / 'commands.json').read_text())
    online = commands['online'][:]
    online[online.index(str(source / 'tools/queue_pretrained_online.py'))] = str(revision / 'tools/queue_pretrained_online.py')
    online[online.index('--execution-source') + 1] = str(revision)
    online.extend(['--pretrain-pid', str(pretrainer['pid'])])
    plan = dict(schema_version=1, kind='future_online_entropy_handoff', run_dir=str(run),
        revision_root=str(revision), plan_path=str(path), controller=controller, pretrainer=pretrainer,
        prepared_status=status, original_status_sha256=sha256_file(status_path),
        original_training_plan_sha256=sha256_file(run / 'training_plan.json'),
        original_pretrain_config_sha256=sha256_file(run / 'pretrain_config.toml'),
        execution_manifest_sha256=_verify_revision(revision), online_command=online,
        original_online_command=commands['online'],
        preservation='pretrainer same PID; pretrain source, config, inputs, weights and progress unchanged',
        online_entropy=dict(initial_coefficient=.002, final_coefficient=.0001,
                            hold_fraction=.1, end_fraction=.8, budget=2000))
    atomic_json(path, plan)
    return plan


def _validate_future_plan(plan):
    run, revision = Path(plan['run_dir']), Path(plan['revision_root'])
    if (plan['schema_version'] != 1 or plan['kind'] != 'future_online_entropy_handoff'
            or _verify_revision(revision) != plan['execution_manifest_sha256']
            or sha256_file(run / 'training_plan.json') != plan['original_training_plan_sha256']
            or sha256_file(run / 'pretrain_config.toml') != plan['original_pretrain_config_sha256']):
        raise ValueError('future online handoff immutable plan or source changed')


def apply_future(plan_path):
    """Pause only the verified controller; leave its worker running at the same PID."""
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    _validate_future_plan(plan)
    run = Path(plan['run_dir'])
    record_path = run / 'entropy_annealing_handoff.json'
    if record_path.exists():
        raise ValueError('future online handoff was already applied')
    verify_process(plan['controller'], stopped=False)
    verify_process(plan['pretrainer'], stopped=False)
    status = json.loads((run / 'status.json').read_text())
    if (status.get('phase') != 'pretraining' or status.get('controller_pid') != plan['controller']['pid']
            or status.get('child_pid') != plan['pretrainer']['pid']):
        raise ValueError('controller already advanced beyond owned pretraining phase')
    backup = run / 'entropy_annealing_original_status.json'
    if backup.exists():
        raise ValueError('immutable handoff backup already exists')
    atomic_json(backup, status)
    os.kill(plan['controller']['pid'], signal.SIGSTOP)
    bridge = None
    try:
        # SIGSTOP delivery is asynchronous. Wait briefly before checking the phase race.
        deadline = time.monotonic() + 5.
        while time.monotonic() < deadline:
            current = process_record(plan['controller']['pid'])
            if current and current['state'] in {'T', 't'}:
                break
            time.sleep(.02)
        verify_process(plan['controller'], stopped=True)
        status = json.loads((run / 'status.json').read_text())
        if status.get('phase') != 'pretraining' or status.get('child_pid') != plan['pretrainer']['pid']:
            raise ValueError('controller changed phase while handoff was being applied')
        command = [plan['online_command'][0], '-u', str(Path(plan['revision_root']) / 'tools/continue_mainline_entropy_annealing.py'),
                   'run-future', '--plan-path', str(path)]
        with (run / 'entropy_annealing_bridge.log').open('ab', buffering=0) as log:
            bridge = subprocess.Popen(command, cwd=plan['revision_root'], stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(schema_version=1, plan_path=str(path), plan_sha256=sha256_file(path),
                      original_controller=plan['controller'], same_pretrainer=plan['pretrainer'],
                      bridge_pid=bridge.pid, bridge_command=command, old_controller_stopped=True)
        atomic_json(record_path, record)
        return record
    except BaseException:
        # Before a bridge owns the run, the original controller can safely resume.
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()
            bridge.wait(timeout=15.)
        if process_record(plan['controller']['pid']) is not None:
            os.kill(plan['controller']['pid'], signal.SIGCONT)
        raise


def run_future(plan_path):
    """Wait for the existing pretrainer, then run the revised normal online queue."""
    path = Path(plan_path).resolve()
    plan = json.loads(path.read_text())
    _validate_future_plan(plan)
    run = Path(plan['run_dir'])
    stopped = False
    def request_stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def status(phase, **extra):
        atomic_json(run / 'status.json', dict(phase=phase, controller_pid=os.getpid(),
            updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            entropy_handoff=str(path), original_controller_stopped=True, **extra))
    with (run / 'entropy_annealing_controller.lock').open('a') as lock:
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
                       entropy_schedule_ready_for_online=True)
                time.sleep(5.)
            if stopped:
                status('entropy_bridge_stopped', pretrainer_not_signalled=True)
                return
            summary = json.loads((run / 'world_model/pretrain_summary.json').read_text())
            if summary.get('status') != 'pretrained':
                raise ValueError('original pretrainer exited without successful completion')
            # The revised queue verifies completed epochs, codec proof and checkpoint ancestry.
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
                    status('entropy_bridge_stopped', pretrainer_not_signalled=True)
                    return
                if child.returncode:
                    raise RuntimeError(f'annealed online queue exited {child.returncode}')
            queue = json.loads((run / 'online_supervisor/status.json').read_text())
            if queue.get('phase') != 'completed':
                raise RuntimeError('annealed online queue exited without completed original budget')
            status('completed')
        except BaseException as error:
            status('failed', error=repr(error), pretrainer_not_signalled=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage-source')
    stage.add_argument('--source-root', type=Path, required=True)
    stage.add_argument('--revision-root', type=Path, required=True)
    stage.add_argument('--anneal-pretrained-online', action='store_true')
    transfer = commands.add_parser('migrate')
    transfer.add_argument('--source-dir', type=Path, required=True)
    transfer.add_argument('--output-dir', type=Path, required=True)
    transfer.add_argument('--device')
    future = commands.add_parser('prepare-future')
    future.add_argument('--run-dir', type=Path, required=True)
    future.add_argument('--revision-root', type=Path, required=True)
    future.add_argument('--plan-path', type=Path, required=True)
    for name in ('apply-future', 'run-future'):
        command = commands.add_parser(name)
        command.add_argument('--plan-path', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'stage-source':
        result = stage_source(args.source_root, args.revision_root,
                              anneal_pretrained_online=args.anneal_pretrained_online)
    elif args.command == 'migrate':
        result = migrate(args.source_dir, args.output_dir, device=args.device)
    elif args.command == 'prepare-future':
        result = prepare_future(args.run_dir, args.revision_root, args.plan_path)
    elif args.command == 'apply-future':
        result = apply_future(args.plan_path)
    else:
        result = run_future(args.plan_path)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
