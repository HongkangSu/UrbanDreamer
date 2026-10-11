import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_city_pretrain import check_exit, codec_ready, commands, cuda_oom_since, existing_cache_complete, gpu_admission, supervise


def options(tmp_path):
    (tmp_path / 'pilot').mkdir()
    (tmp_path / 'pilot/split.json').write_text(json.dumps(dict(train=[{}, {}], validation=[{}])))
    return SimpleNamespace(city_dir=tmp_path, source_dir=Path('/frozen/source'), device='cuda:1',
                           codec_cpu=False, codec_memory_fraction=.03, wm_memory_fraction=.06)


def test_commands_preserve_stage_counts_and_limit_execution_to_pretrain(tmp_path):
    args = options(tmp_path)
    cache = dict(status='complete', entries=[dict(split='train')] * 7 + [dict(split='validation')] * 3)
    plan = commands(args, cache=cache)
    assert plan['codec'][plan['codec'].index('--expected-train-episodes') + 1] == '2'
    assert plan['codec'][plan['codec'].index('--expected-validation-episodes') + 1] == '1'
    fm = plan['world_model']
    assert fm[2] == '/frozen/source/tools/run_complete_pretrain.py'
    assert fm[fm.index('--expected-train-episodes') + 1] == '7'
    assert fm[fm.index('--expected-validation-episodes') + 1] == '3'
    assert fm[fm.index('--precision') + 1] == 'bf16'
    assert not any('online' in word or 'full_retrain' in word for word in fm)
    assert '--calibrated-codec' not in fm
    assert '--configured-codec' in plan['codec']


def test_codec_resume_and_explicit_cpu_leave_fm_on_gpu(tmp_path):
    args = options(tmp_path)
    (tmp_path / 'codec').mkdir()
    (tmp_path / 'codec/codec_latest.pt').touch()
    args.codec_cpu = True
    plan = commands(args, cache=dict(status='complete', entries=[dict(split='train'), dict(split='validation')]))
    assert '--resume' in plan['codec']
    assert plan['codec'][plan['codec'].index('--device') + 1] == 'cpu'
    assert plan['world_model'][plan['world_model'].index('--device') + 1] == 'cuda:1'


def test_ingo_profile_uses_batch64_compiled_and_migrated_config(tmp_path):
    args = options(tmp_path)
    args.pretrain_profile = 'ingo'
    args.wm_memory_fraction = .25
    args.world_model_dir = tmp_path / 'world_model_ingo_profile'
    args.world_model_dir.mkdir()
    (args.world_model_dir / 'run_config.toml').touch()
    fm = commands(args, cache=dict(status='complete', entries=[dict(split='train'), dict(split='validation')]))['world_model']
    expected = {'--batch-size': '64', '--accumulation-steps': '1', '--threads': '4',
                '--space-chunk': '64', '--time-chunk': '65536', '--mlp-chunk': '65536',
                '--compile-blocks': 'on', '--precision': 'bf16', '--gpu-memory-fraction': '0.25',
                '--config': str(args.world_model_dir / 'run_config.toml'),
                '--output-dir': str(args.world_model_dir)}
    for key, value in expected.items():
        assert fm[fm.index(key) + 1] == value


def test_gpu_admission_reserves_cap_plus_margin_and_selects_device():
    run = lambda *a, **k: SimpleNamespace(stdout='0, 2000, 20000\n1, 1712, 20000\n')
    assert gpu_admission('cuda:1', .06, run=run)['ready']
    assert not gpu_admission('cuda:1', .061, run=run)['ready']
    with pytest.raises(RuntimeError, match='missing'):
        gpu_admission('cuda:2', .03, run=run)


def test_adopted_cache_wait_checks_process_identity_and_completion(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.run_city_pretrain.subprocess.run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=f'python build_pretrain_cache.py --output-dir {tmp_path / "cache"}'))
    assert not existing_cache_complete(tmp_path, 123)
    monkeypatch.setattr('tools.run_city_pretrain.subprocess.run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout='unrelated process'))
    with pytest.raises(RuntimeError, match='ended before'):
        existing_cache_complete(tmp_path, 123)
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache/cache_manifest.json').write_text(json.dumps(dict(status='complete')))
    assert existing_cache_complete(tmp_path, 123)


def test_failed_process_does_not_admit_next_stage():
    with pytest.raises(RuntimeError, match='codec process 123 exited 2'):
        check_exit('codec', SimpleNamespace(pid=123, poll=lambda: 2))
    assert not check_exit('cache', SimpleNamespace(pid=1, poll=lambda: None))
    assert check_exit('cache', SimpleNamespace(pid=1, poll=lambda: 0))


def test_oom_retry_classification_ignores_historical_errors_and_quality_failures(tmp_path):
    log = tmp_path / 'world_model.log'
    old = 'torch.OutOfMemoryError: CUDA out of memory\n'
    log.write_text(old + 'ValueError: codec quality gate has not passed\n')
    assert cuda_oom_since(log, 0)
    assert not cuda_oom_since(log, len(old.encode()))
    with log.open('a') as stream:
        stream.write('torch.OutOfMemoryError: CUDA out of memory\n')
    assert cuda_oom_since(log, len(old.encode()))


def test_budget_exhaustion_and_both_codec_gates_are_required(tmp_path):
    (tmp_path / 'codec').mkdir()
    result = tmp_path / 'codec/result.json'
    result.write_text(json.dumps(dict(status='budget_exhausted')))
    with pytest.raises(RuntimeError, match='FM remains blocked'):
        codec_ready(tmp_path)
    result.write_text(json.dumps(dict(status='codec_ready', gate=dict(passed=True), hybrid_gate=dict(passed=False))))
    assert not codec_ready(tmp_path)
    accepted = dict(status='codec_ready', gate=dict(passed=True), hybrid_gate=dict(passed=True),
                    configured_gates_required=True, configured_gate=dict(passed=True),
                    configured_hybrid_gate=dict(passed=True))
    result.write_text(json.dumps(accepted))
    assert codec_ready(tmp_path)
    accepted['configured_hybrid_gate']['passed'] = False
    result.write_text(json.dumps(accepted))
    assert not codec_ready(tmp_path)


def test_fm_rejects_incomplete_or_empty_cache(tmp_path):
    args = options(tmp_path)
    with pytest.raises(ValueError, match='completed'):
        commands(args, cache=dict(status='building'))
    with pytest.raises(ValueError, match='held-out'):
        commands(args, cache=dict(status='complete', entries=[dict(split='train')]))


def test_supervisor_records_full_partition_failure_and_never_starts_online(tmp_path, monkeypatch):
    args = options(tmp_path)
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache/cache_manifest.json').write_text(json.dumps(dict(status='complete',
        entries=[dict(split='train'), dict(split='validation')])))
    (tmp_path / 'world_model.status.json').write_text(json.dumps(dict(
        phase='full_partition_codec_validation', codec_ready=False)))
    launched = []
    def popen(command, **kwargs):
        launched.append(command)
        return SimpleNamespace(pid=len(launched), returncode=0 if len(launched) == 1 else 1,
                               poll=lambda: 0 if len(launched) == 1 else 1)
    monkeypatch.setattr('tools.run_city_pretrain.subprocess.Popen', popen)
    monkeypatch.setattr('tools.run_city_pretrain.codec_ready', lambda city: True)
    monkeypatch.setattr('tools.run_city_pretrain.gpu_admission', lambda *a: dict(ready=True))
    with pytest.raises(RuntimeError, match='world_model process 2 exited 1'):
        supervise(args)
    status = json.loads((tmp_path / 'supervisor_status.json').read_text())
    assert status['phase'] == 'failed'
    assert status['world_model_status']['phase'] == 'full_partition_codec_validation'
    assert status['world_model_status']['codec_ready'] is False
    assert len(launched) == 2
    assert not any('online' in word for command in launched for word in command)


def test_oom_retries_same_batch_with_more_headroom_and_retained_checkpoint(tmp_path, monkeypatch):
    args = options(tmp_path)
    args.pretrain_profile = 'ingo'
    args.retry_cuda_oom = True
    (tmp_path / 'cache').mkdir()
    (tmp_path / 'cache/cache_manifest.json').write_text(json.dumps(dict(status='complete',
        entries=[dict(split='train'), dict(split='validation')])))
    (tmp_path / 'world_model/checkpoints').mkdir(parents=True)
    checkpoint = tmp_path / 'world_model/checkpoints/latest.pt'
    checkpoint.write_bytes(b'committed model optimizer and sampler')
    (tmp_path / 'world_model/pretrain_summary.json').write_text(json.dumps(dict(status='pretrained')))
    launched, margins = [], []
    def popen(command, **kwargs):
        launched.append(command)
        code = 1 if len(launched) == 2 else 0
        if code:
            kwargs['stdout'].write(b'torch.OutOfMemoryError: CUDA out of memory\n')
        return SimpleNamespace(pid=len(launched), returncode=code, poll=lambda: code)
    monkeypatch.setattr('tools.run_city_pretrain.subprocess.Popen', popen)
    monkeypatch.setattr('tools.run_city_pretrain.codec_ready', lambda city: True)
    monkeypatch.setattr('tools.run_city_pretrain.time.sleep', lambda seconds: None)
    def admission(*a, **k):
        margins.append(k.get('margin_mib', 512))
        return dict(ready=True)
    monkeypatch.setattr('tools.run_city_pretrain.gpu_admission', admission)
    supervise(args)
    status = json.loads((tmp_path / 'supervisor_status.json').read_text())
    assert status['phase'] == 'complete'
    assert status['oom_retries'] == 1
    assert margins == [512, 2048]
    assert checkpoint.read_bytes() == b'committed model optimizer and sampler'
    assert len(launched) == 3
    assert all(command[command.index('--batch-size') + 1] == '64' for command in launched[1:])
