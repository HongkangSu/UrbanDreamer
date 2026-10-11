from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import queue_pretrained_online as queue
from tools.run_complete_pretrain import export_run_config
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, legacy_semantic_fingerprint, load_config


def source_fixture(tmp_path, *, final=True, entropy_mode='annealed'):
    source = tmp_path / 'pretrain'
    source.mkdir()
    config = load_config(Path(__file__).parents[1] / 'configs/ingolstadt_custom.toml')
    config = replace(config, run=replace(config.run, output_dir=source, world_model_initialization_epochs=5),
                     method=replace(config.method, actor_entropy_mode=entropy_mode),
                     resources=replace(config.resources, world_model_imagination_compile=False))
    config = replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    export_run_config(config, source / 'run_config.toml')
    # The frozen pretrainer writes the same fields under the old reward semantics.
    config = replace(config, semantic_fingerprint=legacy_semantic_fingerprint(config))
    metrics = dict(codec_ready=True, codec_updates=4500, codec_validation_updates=4500,
                   codec_validation_manifest_identity='heldout', online_episodes_completed=0,
                   world_model_initialization_epochs_completed=5 if final else 1,
                   world_model_updates=999, world_model_initialization_samples_seen=101)
    for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
        metrics[prefix + 'codec_slow_roads'] = 20
        for key in ('codec_count_mae_occupied', 'codec_count_mae_all', 'codec_count_mae_empty',
                    'codec_total_relative_error', 'codec_speed_mae_mps', 'codec_speed_mae_occupied_mps',
                    'codec_speed_mae_slow_mps'):
            metrics[prefix + key] = .001
    state = dict(reward_contract='event_q_counterfactual_global_individual_timeout_v2',
                 models={'world_model': {'w': torch.arange(5.)}, 'actor': {'w': torch.ones(3)},
                         'latent_critic': {'trunk.0.weight': torch.arange(66.).reshape(1, 66)},
                         'event_critic': {'scorer.0.weight': torch.arange(413.).reshape(1, 413)}},
                 optimizers={'world_model_initial': {'state': {0: {
                     'step': torch.tensor(999), 'exp_avg': torch.arange(5.), 'exp_avg_sq': torch.ones(5)}}}},
                 rng={'torch': torch.get_rng_state()}, metrics=metrics, next_environment_episode=0,
                 replay={'online': {'schema_version': 7, 'sequence_length': 3, 'max_episodes': 8,
                                   'active_only': False, 'episodes': [], 'next_storage_id': 0,
                                   'sample_order': [], 'sample_cursor': 0}, 'offline': {'epochs_completed': 5},
                         'codec_validation': {'manifest_identity': 'heldout'}})
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='network',
                    phase=CheckpointPhase.ONLINE_TRAINING if final else CheckpointPhase.PRETRAIN,
                    next_unit=0 if final else 1, training_state=state)
    if final:
        queue.atomic_json(source / 'pretrain_summary.json', {'status': 'pretrained'})
    return source, config, state


def execution_fixture(tmp_path):
    directory = tmp_path / 'source_online'
    (directory / 'tools').mkdir(parents=True)
    (directory / 'tools/run_online_pilot.py').write_text('# frozen runner\n')
    queue.atomic_json(directory / 'source_online_manifest.json', {'files': {
        'tools/run_online_pilot.py': queue.sha256_file(directory / 'tools/run_online_pilot.py')}})
    return directory


def args_fixture(tmp_path, source):
    return SimpleNamespace(pretrain_dir=source, execution_source=execution_fixture(tmp_path),
        queue_dir=tmp_path / 'queue', seed_dir=tmp_path / 'seed', output_dir=tmp_path / 'online',
        pretrain_pid=None, python=Path('/env/python'), device='cpu', minimum_free_gib=3.,
        max_launch_attempts=3, poll_seconds=1, once_until_ready=True)


def test_partial_pretrain_waits_without_staging_or_launch(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path, final=False)
    args = args_fixture(tmp_path, source)
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('premature launch'))
    status = queue.run_queue(args)
    assert status['phase'] == 'waiting_for_pretrain'
    assert not args.seed_dir.exists()
    assert not args.output_dir.exists()


def test_final_pretrain_requires_process_exit(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    monkeypatch.setattr(queue, 'process_identity', lambda pid: 'original-start')
    assert queue.inspect_source(source, pretrain_pid=5, pretrain_identity='original-start')['reason'] == 'waiting_for_pretrainer_exit'
    assert queue.inspect_source(source, pretrain_pid=5, pretrain_identity='old-reused-pid')['ready']


@pytest.mark.parametrize('entropy_mode', ['population', 'annealed'])
def test_paper_seed_preserves_pretraining_and_is_idempotent(tmp_path, entropy_mode):
    source, config, state = source_fixture(tmp_path, entropy_mode=entropy_mode)
    source_hash = queue.sha256_file(source / 'checkpoints/latest.pt')
    seed = tmp_path / 'seed'
    first = queue.stage_seed(source, seed)
    target = load_config(seed / 'run_config.toml')
    checkpoint = load_checkpoint(output_dir=seed, expected_config_fingerprint=target.semantic_fingerprint,
                                 expected_input_hash='network')
    expected, migration = queue.migrate_pretrained_reward(state)
    queue._assert_equal(expected, checkpoint.training_state)
    queue.validate_profile_transfer(config, target)
    assert target.resources.world_model_imagination_compile
    assert target.method.actor_entropy_mode == 'annealed'
    assert target.method.actor_entropy_initial_coefficient == .002
    assert target.method.actor_entropy_final_coefficient == .0001
    assert target.run.online_training_episodes == config.run.online_training_episodes
    assert target.method.disagreement_weight == 0
    assert not first['training_state_preserved_exactly']
    assert first['reward_contract'] == queue.REWARD_CONTRACT
    assert first['reward_migration'] == migration
    assert first['online_budgets'] == queue.learning_budgets(config)
    assert queue.stage_seed(source, seed) == first
    assert queue.sha256_file(source / 'checkpoints/latest.pt') == source_hash


@pytest.mark.parametrize('field,value', [('codec_ready', False), ('codec_validation_updates', 4444),
                                       ('codec_validation_manifest_identity', 'wrong'),
                                       ('codec_hybrid_validation_codec_speed_mae_mps', float('nan'))])
def test_invalid_codec_proof_blocks_handoff(tmp_path, field, value):
    source, config, state = source_fixture(tmp_path)
    state['metrics'][field] = value
    save_checkpoint(output_dir=source, config_fingerprint=config.semantic_fingerprint, input_hash='network',
                    phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    with pytest.raises(ValueError, match='codec'):
        queue.inspect_source(source)


def test_summary_cannot_override_partial_checkpoint(tmp_path):
    source, _, _ = source_fixture(tmp_path, final=False)
    queue.atomic_json(source / 'pretrain_summary.json', {'status': 'pretrained'})
    with pytest.raises(ValueError, match='completed-pretrain'):
        queue.inspect_source(source)


def test_source_changes_after_staging_are_rejected(tmp_path):
    source, _, _ = source_fixture(tmp_path)
    seed = tmp_path / 'seed'
    queue.stage_seed(source, seed)
    with (source / 'checkpoints/latest.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='source changed'):
        queue.stage_seed(source, seed)


def test_changed_frozen_seed_is_rejected(tmp_path):
    source, _, _ = source_fixture(tmp_path)
    seed = tmp_path / 'seed'
    queue.stage_seed(source, seed)
    with (seed / 'checkpoints/latest.pt').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='immutable seed'):
        queue.stage_seed(source, seed)


def test_profile_cannot_change_training_or_solver(tmp_path):
    source, config, _ = source_fixture(tmp_path)
    target = queue.target_config(config, tmp_path / 'seed')
    target = replace(target, method=replace(target.method, fm_steps=2))
    with pytest.raises(ValueError, match='only entropy schedule'):
        queue.validate_profile_transfer(config, target)


def test_execution_source_hashes_are_verified(tmp_path):
    execution = execution_fixture(tmp_path)
    assert queue.verify_execution_source(execution)
    (execution / 'tools/run_online_pilot.py').write_text('changed')
    with pytest.raises(ValueError, match='execution source changed'):
        queue.verify_execution_source(execution)


def test_launch_and_restart_do_not_duplicate_child_or_reduce_budget(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    args = args_fixture(tmp_path, source)
    calls = []
    monkeypatch.setenv('URB_ACTOR_ENTROPY_MODE', 'legacy')
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=123456, poll=lambda: None)
    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    monkeypatch.setattr(queue, 'process_identity', lambda pid: 'child-start' if pid == 123456 else None)
    first = queue.run_queue(args)
    second = queue.run_queue(args)
    assert first['phase'] == second['phase'] == 'online_running'
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index('--online-episodes') + 1] == '2000'
    assert command[command.index('--evaluation-every') + 1] == '20'
    assert command[command.index('--gpu-memory-fraction') + 1] == '.25'
    assert '--run-config' not in command
    assert kwargs['env']['URB_IMAGINATION_WORKERS'] == '8'
    assert kwargs['env']['OMP_NUM_THREADS'] == '4'
    assert kwargs['env']['MKL_NUM_THREADS'] == '4'
    assert kwargs['env']['OPENBLAS_NUM_THREADS'] == '1'
    assert kwargs['env']['TORCHINDUCTOR_COMPILE_THREADS'] == '2'
    assert 'URB_ACTOR_ENTROPY_MODE' not in kwargs['env']
    assert kwargs['start_new_session']
    assert not args.output_dir.exists()  # watcher never contaminates fresh output


def test_launch_failure_retries_are_bounded(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    args = args_fixture(tmp_path, source)
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(pid=123456, poll=lambda: 1)
    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    monkeypatch.setattr(queue, 'process_identity', lambda pid: None)
    for _ in range(4):
        status = queue.run_queue(args)
    assert len(calls) == 3
    assert status['phase'] == 'failed_retry_limit'


def test_auto_gpu_selects_maximum_available_headroom(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda index: ((4 if index == 0 else 6) * 1024 ** 3, 24 * 1024 ** 3))
    assert queue.choose_device('auto') == 'cuda:1'
    assert queue.choose_device('cuda:0') == 'cuda:0'
    assert queue.choose_device('auto', minimum_free_gib=7) is None


def test_nonblocking_lock_prevents_second_watcher(tmp_path):
    with queue.queue_lock(tmp_path / 'queue'):
        with pytest.raises(RuntimeError, match='another watcher'):
            with queue.queue_lock(tmp_path / 'queue'):
                pytest.fail('duplicate lock')


def test_recover_child_from_crash_between_popen_and_status(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    args = args_fixture(tmp_path, source)
    args.queue_dir.mkdir()
    binding = dict(pretrain_dir=str(args.pretrain_dir), seed_dir=str(args.seed_dir),
                   output_dir=str(args.output_dir), execution_source=str(args.execution_source))
    queue.atomic_json(args.queue_dir / 'status.json', dict(binding=binding, phase='launching', launch_attempts=1))
    monkeypatch.setattr(queue, 'recover_unrecorded_child', lambda args: (98765, 'birth'))
    monkeypatch.setattr(queue, 'process_identity', lambda pid: 'birth' if pid == 98765 else None)
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('duplicate child'))
    status = queue.run_queue(args)
    assert status['phase'] == 'online_running'
    assert status['child_pid'] == 98765
    assert status['launch_attempts'] == 1


def test_popen_os_errors_obey_finite_retry_budget(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    args = args_fixture(tmp_path, source)
    def launch(*a, **kw):
        raise OSError('temporary startup error')
    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    for _ in range(4):
        status = queue.run_queue(args)
    assert status['phase'] == 'failed_retry_limit'
    assert status['launch_attempts'] == 3


def test_first_launch_rejects_unrelated_existing_output(tmp_path, monkeypatch):
    source, _, _ = source_fixture(tmp_path)
    args = args_fixture(tmp_path, source)
    args.output_dir.mkdir()
    (args.output_dir / 'unrelated').write_text('existing experiment')
    monkeypatch.setattr(queue.subprocess, 'Popen', lambda *a, **kw: pytest.fail('launched into existing output'))
    with pytest.raises(ValueError, match='fresh output'):
        queue.run_queue(args)


def test_inspect_only_does_not_write_any_queue_artifacts(tmp_path):
    source, _, _ = source_fixture(tmp_path, final=False)
    result = queue.main(['--pretrain-dir', str(source), '--execution-source', str(tmp_path / 'future_source'),
                        '--queue-dir', str(tmp_path / 'queue'), '--seed-dir', str(tmp_path / 'seed'),
                        '--output-dir', str(tmp_path / 'online'), '--inspect-only'])
    assert not result['ready']
    assert not (tmp_path / 'queue').exists()
    assert not (tmp_path / 'seed').exists()


def test_execution_manifest_is_bound_while_pretrain_is_still_running(tmp_path):
    source, _, _ = source_fixture(tmp_path, final=False)
    args = args_fixture(tmp_path, source)
    queue.run_queue(args)
    script = args.execution_source / 'tools/run_online_pilot.py'
    script.write_text('# replaced both source and its manifest\n')
    queue.atomic_json(args.execution_source / 'source_online_manifest.json', {'files': {
        'tools/run_online_pilot.py': queue.sha256_file(script)}})
    with pytest.raises(ValueError, match='frozen identity changed'):
        queue.run_queue(args)


def test_dynamic_epochs_bf16_and_configured_budget_handoff(tmp_path):
    _, config, state = source_fixture(tmp_path)
    config = replace(config, run=replace(config.run, world_model_initialization_epochs=2),
        resources=replace(config.resources, world_model_imagination_precision='bf16',
                          imagination_passes_per_episode=4, imagination_rollouts_per_pass=16))
    checkpoint = SimpleNamespace(phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    with pytest.raises(ValueError, match='configured initialization epochs'):
        queue.validate_completed_source(checkpoint, config)
    state['metrics']['world_model_initialization_epochs_completed'] = 2
    queue.validate_completed_source(checkpoint, config)
    assert queue.learning_budgets(config)['rollouts_per_pass'] == 16


def test_authorized_unready_handoff_preserves_failure_but_requires_fresh_proof(tmp_path):
    _, config, state = source_fixture(tmp_path)
    config = replace(config, codec=replace(config.codec, decoder_allow_unready=True,
                                          decoder_pretrain_enabled=True, decoder_online_enabled=True))
    state['metrics'].update(codec_ready=False, decoder_pretrain_updates=12,
                            codec_validation_decoder_updates=12,
                            codec_validation_codec_speed_mae_mps=9.)
    checkpoint = SimpleNamespace(phase=CheckpointPhase.ONLINE_TRAINING, next_unit=0, training_state=state)
    queue.validate_completed_source(checkpoint, config)
    assert state['metrics']['codec_ready'] is False
    state['metrics']['codec_validation_decoder_updates'] = 11
    with pytest.raises(ValueError, match='revision'):
        queue.validate_completed_source(checkpoint, config)
    state['metrics']['codec_validation_decoder_updates'] = 12
    state['models']['world_model']['w'][0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        queue.validate_completed_source(checkpoint, config)
