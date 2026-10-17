"""Prepare recorded fresh seeds and launch the requested three-city route experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/wangwenrui/shk/UrbanDreamer/urban_dreamer')
PYTHON = '/home/wangwenrui/opt/miniconda3/envs/URB/bin/python'
BATCH = ROOT / 'outputs/urbandreamer_current_to_dest_1000_fp16_w8_r8_p2_20261013'


def utc():
    return time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def prepare():
    from dataclasses import replace
    import torch
    sys.path.insert(0, str(ROOT))
    from tools.migrate_bounded_critics import migrate_state
    from tools.migrate_mainline_prior import _assert_equal
    from tools.migrate_pretrained_meso import model_hashes
    from tools.queue_pretrained_online import checkpoint_source_config, migrate_pretrained_reward
    from tools.run_complete_pretrain import export_run_config
    from tools.run_online_pilot import validate_pretrained_source
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.workflow.checkpoint import _validate_loaded_payload, save_checkpoint, CheckpointPhase
    from urban_dreamer.workflow.config import load_config, _semantic_fingerprint
    from urban_dreamer.workflow.pipeline import combined_input_hash
    torch.set_num_threads(2)
    BATCH.mkdir(exist_ok=False)
    source = BATCH / 'source'
    source.mkdir()
    for name in ('urban_dreamer', 'tools', 'configs'):
        shutil.copytree(ROOT / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    files = {str(p.relative_to(source)): digest(p) for p in source.rglob('*') if p.is_file()}
    write(source / 'source_manifest.json', dict(created_utc=utc(), files=files))
    assert 'def evaluation_protocol(' in (source / 'urban_dreamer/learning/training.py').read_text()
    shutil.copy2(__file__, BATCH / 'controller.py')
    parent = ROOT / 'outputs/two_city_urbandreamer_pretrain_supplement_20261012'
    ingo = ROOT / 'outputs/ingolstadt_full_retrain_20261009'
    specs = [
        ('ingolstadt_custom', ingo / 'online_seed',
         ingo / 'online_bounded_noprior_fresh_t2_2000_20261011/run_config.toml', 'cuda:1'),
        ('coulommiers', parent / 'coulommiers/online_seed',
         parent / 'coulommiers/online_mainline_2000/run_config.toml', 'cuda:1'),
        ('montereau_fault_yonne', parent / 'montereau_fault_yonne/online_seed',
         parent / 'montereau_fault_yonne/online_seed/run_config.toml', 'cuda:1'),
    ]
    entries = []
    for city, original, template_path, device in specs:
        destination = BATCH / city
        destination.mkdir()
        seed = destination / 'seed'
        seed.mkdir()
        output = destination / 'online'
        original_path = original / 'checkpoints/latest.pt'
        original_hash = digest(original_path)
        raw = torch.load(original_path, map_location='cpu', weights_only=True)
        config = checkpoint_source_config(load_config(original / 'run_config.toml'), raw)
        checkpoint = _validate_loaded_payload(payload=raw, path=original_path,
            expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=raw['input_hash'])
        validate_pretrained_source(checkpoint, config)
        template = load_config(template_path)
        assert template.inputs == config.inputs and template.simulation == config.simulation
        assert template.data == config.data and template.codec == config.codec
        target = replace(template, run=replace(template.run, output_dir=seed,
            device=device, online_training_episodes=1000), resources=replace(template.resources,
            world_model_imagination_precision='fp16', imagination_rollouts_per_pass=8,
            imagination_passes_per_episode=2, imagination_time_budget_seconds=0.))
        target = replace(target, semantic_fingerprint=_semantic_fingerprint(**{
            k: getattr(target, k) for k in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
        export_run_config(target, seed / 'run_config.toml')
        target = load_config(seed / 'run_config.toml')
        assert combined_input_hash(validate_prepared_inputs(target)) == checkpoint.input_hash
        state, reward = migrate_pretrained_reward(checkpoint.training_state)
        if target.method.critic_value_mode == 'bounded_cohort_v1':
            state = migrate_state(state, target.method, seed=target.run.seed, fresh_online=True)
        _assert_equal(checkpoint.training_state['models']['world_model'], state['models']['world_model'])
        assert not state['metrics'].get('online_episodes_completed', 0)
        assert not state['replay']['online']['episodes'] and state['next_environment_episode'] == 0
        save_checkpoint(output_dir=seed, config_fingerprint=target.semantic_fingerprint,
            input_hash=checkpoint.input_hash, phase=CheckpointPhase.ONLINE_TRAINING,
            next_unit=0, training_state=state)
        proof = dict(original_seed=str(original), original_checkpoint_sha256=original_hash,
            original_config_fingerprint=config.semantic_fingerprint,
            prepared_config_fingerprint=target.semantic_fingerprint,
            prepared_checkpoint_sha256=digest(seed / 'checkpoints/latest.pt'),
            world_model_sha256=model_hashes({'world_model': state['models']['world_model']})['world_model'],
            pretrained_world_model_tensors_unchanged=True, online_episodes=0,
            reward_migration=reward, critic_mode=target.method.critic_value_mode,
            changes=dict(online_training_episodes=1000, imagination_precision='fp16',
                         workers=8, rollouts=8, imagination_passes=2, imagination_time_budget_seconds=0.),
            prepared_utc=utc())
        assert digest(original_path) == original_hash
        write(seed / 'initialization_proof.json', proof)
        command = [PYTHON, '-u', str(source / 'tools/run_online_pilot.py'),
            '--source-dir', str(seed), '--output-dir', str(output),
            '--routing-mode', 'current_to_dest', '--device', device,
            '--online-episodes', '1000', '--evaluation-every', '20',
            '--evaluation-policy', 'argmax', '--threads', '4', '--gpu-memory-fraction', '.25']
        entries.append(dict(city=city, seed=str(seed), output=str(output), device=device,
                            command=command, status='prepared'))
        print(json.dumps(dict(city=city, phase='prepared', proof=proof)), flush=True)
        del state, checkpoint, raw
    write(BATCH / 'manifest.json', dict(created_utc=utc(), source=str(source),
        source_manifest_sha256=digest(source / 'source_manifest.json'),
        routing_mode='current_to_dest', policy_decision_interval_seconds=30,
        imagination_workers=8, imagination_rollouts_per_pass=8,
        imagination_passes_per_episode=2, world_model_updates_per_episode=4,
        imagination_precision='fp16', online_episodes=1000, entries=entries))


def supervise(city):
    manifest = json.loads((BATCH / 'manifest.json').read_text())
    entry = next(row for row in manifest['entries'] if row['city'] == city)
    directory = BATCH / city
    path = directory / 'status.json'
    if path.exists():
        raise RuntimeError('supervisor already has state; inspect before relaunch')
    source = Path(manifest['source'])
    assert digest(source / 'source_manifest.json') == manifest['source_manifest_sha256']
    for name, expected in json.loads((source / 'source_manifest.json').read_text())['files'].items():
        assert digest(source / name) == expected, name
    environment = dict(os.environ, URB_IMAGINATION_WORKERS='8', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCHINDUCTOR_CACHE_DIR=str(directory / 'inductor_cache'))
    environment.pop('URB_ACTOR_ENTROPY_MODE', None)
    with (directory / 'online.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(entry['command'], cwd=source, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    entry.update(supervisor_pid=os.getpid(), pid=child.pid, status='starting', started_utc=utc())
    while True:
        code = child.poll()
        entry.update(updated_utc=utc(), exit_code=code)
        progress = Path(entry['output']) / 'pilot_progress.json'
        if progress.exists():
            data = json.loads(progress.read_text())
            entry.update(completed_episodes=len(data['online']), phase=data['phase'],
                         completed_before_evaluations=len(data['before']))
        entry['status'] = ('running' if code is None else 'completed' if code == 0 else 'failed')
        write(path, entry)
        if code is not None:
            return
        time.sleep(5)


def launch():
    assert (ROOT / 'outputs/online_route_restart_stop_20261013/completed.json').exists()
    manifest = json.loads((BATCH / 'manifest.json').read_text())
    launched = []
    for entry in manifest['entries']:
        folder = BATCH / entry['city']
        assert not (folder / 'status.json').exists()
        with (folder / 'supervisor.log').open('ab', buffering=0) as log:
            child = subprocess.Popen([PYTHON, str(BATCH / 'controller.py'), 'supervise', '--city', entry['city']],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launched.append(dict(city=entry['city'], supervisor_pid=child.pid, launched_utc=utc()))
    write(BATCH / 'launch.json', launched)
    print(json.dumps(launched, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'supervise'))
    parser.add_argument('--city')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'launch':
        launch()
    else:
        supervise(args.city)
