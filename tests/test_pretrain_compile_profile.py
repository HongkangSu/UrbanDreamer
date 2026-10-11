"""Compiled pretraining preserves model identity and explicitly binds its profile."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_urban_dreamer_config import _write_config
from tools import run_complete_pretrain as runner
from urban_dreamer.models.world_model import BlockCausalLayer
from urban_dreamer.workflow import config as configuration


def config_with_compile(tmp_path, value=None):
    path = _write_config(tmp_path / 'config.toml')
    if value is not None:
        path.write_text(path.read_text().replace('[resources]',
            '[resources]\nworld_model_pretrain_compile_blocks = ' + value))
    return configuration.load_config(path)


def test_disabled_compile_preserves_legacy_fingerprint_bytes(tmp_path, monkeypatch):
    config = config_with_compile(tmp_path)
    assert config.resources.world_model_pretrain_compile_blocks is False
    original = configuration.asdict

    def legacy_asdict(value):
        result = original(value)
        if isinstance(value, configuration.ResourceConfig):
            result.pop('world_model_pretrain_compile_blocks')
        return result

    monkeypatch.setattr(configuration, 'asdict', legacy_asdict)
    expected = configuration._semantic_fingerprint(**{name: getattr(config, name) for name in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert config.semantic_fingerprint == expected


def test_compile_enabled_changes_fingerprint_and_exports_profile(tmp_path):
    baseline = config_with_compile(tmp_path)
    enabled = config_with_compile(tmp_path, 'true')
    disabled = config_with_compile(tmp_path, 'false')
    assert baseline.semantic_fingerprint == disabled.semantic_fingerprint
    assert baseline.semantic_fingerprint != enabled.semantic_fingerprint
    assert runner.runtime_profile(enabled, gpu_memory_fraction=.5, threads=2)['compile_blocks']
    target = tmp_path / 'export.toml'
    runner.export_run_config(enabled, target)
    assert configuration.load_config(target).semantic_fingerprint == enabled.semantic_fingerprint


@pytest.mark.parametrize('value', ['1', '"on"', '[]', '0.5'])
def test_compile_flag_requires_boolean(tmp_path, value):
    with pytest.raises(configuration.ConfigError, match='compile_blocks.*boolean'):
        config_with_compile(tmp_path, value)


@pytest.mark.parametrize('configured', [False, True])
@pytest.mark.parametrize('override', [None, 'on', 'off'])
def test_cli_override_inherits_only_when_omitted(tmp_path, configured, override):
    config = config_with_compile(tmp_path, str(configured).lower())
    assert runner.resolve_compile_blocks(config, override) == (
        configured if override is None else override == 'on')


def test_compile_hook_keeps_state_and_optimizer_parameter_identity(tmp_path, monkeypatch):
    config = config_with_compile(tmp_path, 'true')
    model = torch.nn.Sequential(BlockCausalLayer(3), BlockCausalLayer(3))
    optimizer = torch.optim.Adam(model.parameters())
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    keys = tuple(model.state_dict())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    calls = []
    def fake_compile(forward, **kwargs):
        calls.append((forward, kwargs))
        return forward

    monkeypatch.setattr(torch, 'compile', fake_compile)
    monkeypatch.delenv('TORCHINDUCTOR_COMPILE_THREADS', raising=False)
    monkeypatch.delenv('TORCHINDUCTOR_CACHE_DIR', raising=False)
    trainer = SimpleNamespace(config=config, world_model=model)
    assert runner.compile_pretrain_blocks(trainer) == 2
    assert all(kwargs == dict(backend='inductor', mode='default', fullgraph=True, dynamic=False)
               for _, kwargs in calls)
    assert ids == [id(p) for p in model.parameters()]
    assert keys == tuple(model.state_dict())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    assert runner.os.environ['TORCHINDUCTOR_COMPILE_THREADS'] == '2'
    assert runner.os.path.isabs(runner.os.environ['TORCHINDUCTOR_CACHE_DIR'])
    monkeypatch.setenv('TORCHINDUCTOR_COMPILE_THREADS', '1')
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', '/tmp/explicit-compile-cache')
    runner.compile_pretrain_blocks(trainer)
    assert runner.os.environ['TORCHINDUCTOR_COMPILE_THREADS'] == '1'
    assert runner.os.environ['TORCHINDUCTOR_CACHE_DIR'] == '/tmp/explicit-compile-cache'
    trainer.config = replace(config,
        resources=replace(config.resources, world_model_pretrain_compile_blocks=False))
    calls.clear()
    assert runner.compile_pretrain_blocks(trainer) == 0
    assert not calls


@pytest.mark.parametrize('batch', [96, 23 * 3, 31 * 3])
@pytest.mark.parametrize('device,dtype,grad', [
    ('cuda', torch.bfloat16, True), ('cuda', torch.bfloat16, False),
    ('cuda', torch.float32, True), ('cpu', torch.bfloat16, True)])
def test_dispatch_only_validated_full_batch(batch, device, dtype, grad, monkeypatch):
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device_type: True)
    monkeypatch.setattr(torch, 'get_autocast_dtype', lambda device_type: torch.bfloat16)
    calls = []
    eager = lambda x: calls.append('eager')
    compiled = lambda x: calls.append('compiled')
    dispatch = runner._dispatch_pretrain_block(eager, compiled, (96, 1, 857, 256))
    x = SimpleNamespace(shape=(batch, 1, 857, 256),
                        device=SimpleNamespace(type=device), dtype=dtype)
    with torch.set_grad_enabled(grad):
        dispatch(x)
    expected = ('compiled' if batch == 96 and device == 'cuda'
                and dtype in (torch.bfloat16, torch.float32) and grad else 'eager')
    assert calls == [expected]


@pytest.mark.parametrize('bad_group', ['world', 'readout'])
def test_compiled_update_rejects_nonfinite_before_any_optimizer_step(tmp_path, monkeypatch, bad_group):
    from test_urban_dreamer_training import make_trainer

    trainer = make_trainer(tmp_path)
    trainer.config = replace(trainer.config, resources=replace(trainer.config.resources,
        world_model_pretrain_compile_blocks=True))
    monkeypatch.setattr(trainer, 'validate_codec_ready', lambda: None)
    world = next(iter(trainer._world_parameters))
    readout = next(trainer.world_model.active_vehicle_fraction_readout.parameters())
    def losses(batch):
        return (world.sum() * (float('nan') if bad_group == 'world' else 1.),
                readout.sum() * (float('nan') if bad_group == 'readout' else 1.))
    monkeypatch.setattr(trainer, '_world_loss', losses)
    steps = []
    monkeypatch.setattr(trainer.optimizers['world_model_initial'], 'step', lambda: steps.append('world'))
    monkeypatch.setattr(trainer.optimizers['active_fraction'], 'step', lambda: steps.append('readout'))
    before = trainer.metrics['world_model_updates']
    calibration = trainer.world_model.active_vehicle_fraction_readout.calibration_updates.clone()
    with pytest.raises(RuntimeError, match='non-finite'):
        trainer._update_world(SimpleNamespace(has_samples=True), initial=True,
                              batches=[SimpleNamespace(flow=torch.zeros(1))])
    assert not steps
    assert trainer.metrics['world_model_updates'] == before
    assert torch.equal(calibration, trainer.world_model.active_vehicle_fraction_readout.calibration_updates)


@pytest.mark.parametrize('enabled,dtype', [(False, torch.bfloat16), (True, torch.float16)])
def test_compile_requires_bf16_autocast(enabled, dtype, monkeypatch):
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device_type: enabled)
    monkeypatch.setattr(torch, 'get_autocast_dtype', lambda device_type: dtype)
    calls = []
    dispatch = runner._dispatch_pretrain_block(lambda x: calls.append('eager'),
        lambda x: calls.append('compiled'), (32, 1, 857, 256))
    dispatch(SimpleNamespace(shape=(32, 1, 857, 256), device=SimpleNamespace(type='cuda'),
                             dtype=torch.float32))
    assert calls == ['eager']


@pytest.mark.parametrize('mode,full_batch', [('legacy_independent', 96), ('single_step', 32), ('recursive_3step', 32)])
def test_compile_shape_tracks_physical_training_mode(tmp_path, monkeypatch, mode, full_batch):
    config = config_with_compile(tmp_path, 'true')
    config = replace(config, method=replace(config.method, fm_training_mode=mode),
        resources=replace(config.resources, world_model_initialization_batch_size=32,
                          world_model_initialization_sequence_length=3))
    calls = []
    monkeypatch.setattr(BlockCausalLayer, 'forward', lambda self, x: calls.append('eager'))
    monkeypatch.setattr(torch, 'compile', lambda forward, **kwargs: lambda x: calls.append('compiled'))
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device_type: True)
    monkeypatch.setattr(torch, 'get_autocast_dtype', lambda device_type: torch.bfloat16)
    block = BlockCausalLayer(3)
    runner.compile_pretrain_blocks(SimpleNamespace(config=config, world_model=block))
    for batch, expected in [(full_batch, 'compiled'), (full_batch-1, 'eager')]:
        block.forward(SimpleNamespace(shape=(batch, 1, 3, 256),
            device=SimpleNamespace(type='cuda'), dtype=torch.float32))
        assert calls[-1] == expected
