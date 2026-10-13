"""Inference acceleration changes only the velocity execution profile."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_urban_dreamer_config import _write_config
from test_urban_dreamer_world_model import _small_dynamics
from urban_dreamer.workflow import config as cfg
from urban_dreamer.learning import fm_inference as inference
from urban_dreamer.models.world_model import DynamicsOutput


def configured(tmp_path, fields=''):
    path = _write_config(tmp_path / 'config.toml')
    path.write_text(path.read_text().replace('[resources]', '[resources]\n' + fields))
    return cfg.load_config(path)


def test_fp16_default_is_explicitly_fingerprinted(tmp_path):
    default = configured(tmp_path)
    explicit = configured(tmp_path, 'world_model_imagination_precision = "fp16"\n')
    legacy = configured(tmp_path, 'world_model_imagination_precision = "fp32"\n')
    assert default.resources.world_model_imagination_precision == 'fp16'
    assert default.semantic_fingerprint == explicit.semantic_fingerprint
    assert default.semantic_fingerprint != legacy.semantic_fingerprint


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for actual FP16 execution')
def test_cuda_fp16_velocity_keeps_prior_conditions_integrator_and_parameters_fp32(tmp_path):
    dynamics = _small_dynamics().cuda().eval()
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.001)
    trainer = SimpleNamespace(config=configured(tmp_path), world_model=SimpleNamespace(dynamics=dynamics))
    assert inference.configure_imagination_inference(trainer)['precision'] == 'fp16'
    action = torch.zeros(2, 1, 3, device='cuda')
    latent = torch.randn(2, 1, 3, 64, device='cuda')
    ids = tuple(map(id, dynamics.parameters()))
    with torch.no_grad():
        condition = dynamics.encode_condition(action)
        start, increment, cached = dynamics.prior_start(latent, condition)
        result = dynamics.solve(start, action, cached, steps=3, solver='euler')
        velocity = dynamics(action, packed_latents=start, flow_time=torch.zeros(2, 1, device='cuda'),
                            condition=cached).latent_velocity
    assert all(tensor.dtype == torch.float32 for tensor in (
        start, increment, condition.edge_tokens, cached.reference_tokens, velocity, result))
    assert torch.isfinite(result).all() and velocity.abs().sum() > 0
    assert ids == tuple(map(id, dynamics.parameters()))
    assert all(parameter.dtype == torch.float32 for parameter in dynamics.parameters())


def test_legacy_resource_defaults_preserve_exact_fingerprint(tmp_path, monkeypatch):
    config = configured(tmp_path, 'world_model_imagination_precision = "fp32"\n')
    old_asdict = cfg.asdict
    def historical(value):
        result = old_asdict(value)
        if isinstance(value, cfg.ResourceConfig):
            result = {key: item for key, item in result.items() if not key.startswith('world_model_imagination_')}
        return result
    monkeypatch.setattr(cfg, 'asdict', historical)
    expected = cfg._semantic_fingerprint(**{key: getattr(config, key) for key in
        ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')})
    assert expected == config.semantic_fingerprint


@pytest.mark.parametrize('field,value', [('space_chunk','0'),('time_chunk','65536'),
    ('mlp_chunk','65536'),('precision','"fp32"'),('precision','"bf16"'),('precision','"tf32"'),('compile','true'),('compile_mode','"reduce-overhead"')])
def test_nondefault_inference_resources_are_fingerprinted(tmp_path, field, value):
    default = configured(tmp_path)
    changed = configured(tmp_path, f'world_model_imagination_{field} = {value}\n')
    assert default.semantic_fingerprint != changed.semantic_fingerprint


@pytest.mark.parametrize('field,value', [('space_chunk','-1'),('time_chunk','true'),
    ('mlp_chunk','0.5'),('precision','"float16"'),('compile','1'),('compile_mode','"max-autotune"'),('compile_mode','true')])
def test_invalid_inference_resources_rejected(tmp_path, field, value):
    with pytest.raises(cfg.ConfigError, match='world_model_imagination_'):
        configured(tmp_path, f'world_model_imagination_{field} = {value}\n')


@pytest.mark.parametrize('precision', ['fp16', 'bf16'])
def test_cpu_fallback_identity_and_idempotent_configuration(tmp_path, monkeypatch, precision):
    config = configured(tmp_path, f'world_model_imagination_precision = "{precision}"\n'
        'world_model_imagination_compile = true\nworld_model_imagination_space_chunk = 64\n')
    dynamics = _small_dynamics()
    optimizer = torch.optim.Adam(dynamics.parameters())
    original_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    state = {k: v.clone() for k,v in dynamics.state_dict().items()}
    monkeypatch.setattr(torch, 'compile', lambda *a, **k: pytest.fail('CPU unexpectedly compiled'))
    trainer = SimpleNamespace(config=config, world_model=SimpleNamespace(dynamics=dynamics))
    effective = inference.configure_imagination_inference(trainer)
    assert effective['precision'] == 'fp32' and not effective['compile']
    wrapper = dynamics.forward
    assert inference.configure_imagination_inference(trainer) == effective
    assert dynamics.forward is wrapper
    assert original_ids == [id(p) for p in dynamics.parameters()]
    assert state.keys() == dynamics.state_dict().keys()
    assert all(torch.equal(v, dynamics.state_dict()[k]) for k,v in state.items())
    assert dynamics.transformer.layers[0].space.attention_batch_chunk_size == 64
    trainer.config = replace(config, resources=replace(config.resources, world_model_imagination_space_chunk=0))
    inference.configure_imagination_inference(trainer)
    assert dynamics.forward.eager is wrapper.eager  # No nested wrappers after profile change.
    assert dynamics.transformer.layers[0].space.attention_batch_chunk_size == 0


def test_two_noise_samples_use_sixteen_sample_compile_shape(tmp_path):
    config = configured(tmp_path)
    config = replace(config, method=replace(config.method, disagreement_samples=2))
    dynamics = _small_dynamics()
    trainer = SimpleNamespace(config=config, world_model=SimpleNamespace(dynamics=dynamics))
    profile = inference.configure_imagination_inference(trainer)
    assert profile['full_batch'] == 16
    assert dynamics.forward.profile.full_batch == 16


@pytest.mark.parametrize('batch,device,grad,condition,expected', [
    (24,'cuda',False,True,'compiled'), (3,'cuda',False,True,'eager'),
    (24,'cpu',False,True,'eager'), (24,'cuda',True,True,'eager'),
    (24,'cuda',False,False,'eager')])
def test_static_compile_dispatch_and_fp32_output(monkeypatch, batch, device, grad, condition, expected):
    calls = []
    def eager(*args, **kwargs):
        calls.append('eager')
        return DynamicsOutput(torch.ones(1))
    def compile_stub(forward, **kwargs):
        assert kwargs == dict(backend='inductor',mode='default',fullgraph=True,dynamic=False)
        def compiled(*a, **k):
            calls.append('compiled')
            return DynamicsOutput(torch.ones(1))
        return compiled
    monkeypatch.setattr(torch, 'compile', compile_stub)
    dynamics = SimpleNamespace(n_edges=3,forward=eager)
    wrapper = inference._VelocityInference(dynamics,
        inference.ImaginationInferenceProfile(16,32768,16384,'fp32',True))
    latent = SimpleNamespace(shape=(batch,1,3,64),device=SimpleNamespace(type=device),dtype=torch.float32)
    with torch.set_grad_enabled(grad):
        result = wrapper(None,packed_latents=latent,flow_time=torch.zeros(batch,1),
                         condition=object() if condition else None)
    assert calls == [expected]
    assert result.latent_velocity.dtype == torch.float32


@pytest.mark.parametrize('precision,dtype', [('fp16', torch.float16), ('bf16', torch.bfloat16)])
def test_autocast_only_encloses_velocity_and_not_training(monkeypatch, precision, dtype):
    contexts = []
    @contextmanager
    def autocast(device, **kwargs):
        assert device == 'cuda' and kwargs['dtype'] == dtype
        contexts.append('enter')
        yield
        contexts.append('exit')
    monkeypatch.setattr(torch, 'autocast', autocast)
    def eager(*a, **kw):
        assert kw['packed_latents'].dtype == torch.float32
        assert kw['flow_time'].dtype == torch.float32
        assert kw['condition'].dtype == torch.float32
        contexts.append('forward')
        return DynamicsOutput(torch.ones(1,dtype=dtype))
    wrapper = inference._VelocityInference(SimpleNamespace(n_edges=3,forward=eager),
        inference.ImaginationInferenceProfile(64,32768,65536,precision,False))
    latent = SimpleNamespace(shape=(3,1,3,64),device=SimpleNamespace(type='cuda'),dtype=torch.float32)
    with torch.no_grad():
        result = wrapper(None,packed_latents=latent,flow_time=torch.zeros(3,1),condition=torch.zeros(1))
    assert contexts == ['enter','forward','exit']
    assert result.latent_velocity.dtype == torch.float32
    contexts.clear()
    wrapper(None,packed_latents=latent,flow_time=torch.zeros(3,1),condition=torch.zeros(1))
    assert contexts == ['forward']


def test_cpu_eager_matches_original_and_preserves_gradients(tmp_path):
    dynamics = _small_dynamics(noise_scale=0.)
    dynamics.transformer = torch.nn.Identity()
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.01)
    # Identity transformer needs no chunk mutation in this small numerical fixture.
    dynamics._set_chunk_profile = lambda **kwargs: None
    action = torch.zeros(2,1,3)
    latent = torch.randn(2,1,3,64,requires_grad=True)
    condition = dynamics.encode_condition(action)
    tau = torch.full((2,1),.5)
    original = dynamics(action,packed_latents=latent,flow_time=tau,condition=condition).latent_velocity
    trainer = SimpleNamespace(config=configured(tmp_path),world_model=SimpleNamespace(dynamics=dynamics))
    inference.configure_imagination_inference(trainer)
    actual = dynamics(action,packed_latents=latent,flow_time=tau,condition=condition).latent_velocity
    torch.testing.assert_close(actual,original,rtol=0,atol=0)
    gradients = (latent, dynamics.spatial_proj.weight, dynamics.flow_head.weight)
    reference_grads = torch.autograd.grad(original.sum(), gradients, retain_graph=True)
    actual_grads = torch.autograd.grad(actual.sum(), gradients)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
    assert actual_grads[0].abs().sum() > 0



@pytest.mark.parametrize('precision', ['fp16', 'bf16'])
def test_no_grad_training_forward_stays_eager_fp32(monkeypatch, precision):
    monkeypatch.setattr(torch, 'autocast', lambda *a, **k: pytest.fail('training autocast altered'))
    monkeypatch.setattr(torch, 'compile', lambda *a, **k: lambda *args, **kw: pytest.fail('training compiled'))
    calls = []
    def eager(*a, **k):
        calls.append('eager')
        return DynamicsOutput(torch.ones(1))
    dynamics = SimpleNamespace(n_edges=3,forward=eager,training=True)
    wrapper = inference._VelocityInference(dynamics,
        inference.ImaginationInferenceProfile(16,32768,16384,precision,True))
    latent = SimpleNamespace(shape=(24,1,3,64),device=SimpleNamespace(type='cuda'),dtype=torch.float32)
    with torch.no_grad():
        wrapper(None,packed_latents=latent,flow_time=torch.zeros(24,1),condition=object())
    assert calls == ['eager']



def test_effective_profile_and_checkpoint_state_are_weights_only_serializable(tmp_path):
    import io
    import json
    dynamics = _small_dynamics()
    trainer = SimpleNamespace(config=configured(tmp_path), world_model=SimpleNamespace(dynamics=dynamics))
    profile = inference.configure_imagination_inference(trainer)
    assert json.loads(json.dumps(profile)) == profile
    stream = io.BytesIO()
    torch.save({'model': dynamics.state_dict(), 'metrics': {'profile': profile}}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    assert restored['metrics']['profile'] == profile
    for key, value in dynamics.state_dict().items():
        torch.testing.assert_close(restored['model'][key], value, rtol=0, atol=0)


def test_deepcopy_rebinds_eager_and_compiled_to_copied_parameters(monkeypatch):
    import copy
    # A compile wrapper closes over its bound method. Copying the function
    # itself would leave the compiled path connected to the original model.
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: lambda *a, **k: fn(*a, **k))
    dynamics = _small_dynamics()
    wrapper = inference._VelocityInference(dynamics,
        inference.ImaginationInferenceProfile(16,32768,16384,'fp32',True))
    dynamics.forward = wrapper
    dynamics._imagination_velocity_inference = wrapper
    copied = copy.deepcopy(dynamics)
    assert copied.forward.eager.__self__ is copied
    assert copied.forward.dynamics is copied
    assert copied.forward.compiled is not wrapper.compiled
    assert copied.forward is copied._imagination_velocity_inference
    assert not set(map(id, copied.parameters())) & set(map(id, dynamics.parameters()))



@pytest.mark.parametrize('initial', [False, True])
@pytest.mark.parametrize('raises', [False, True])
def test_tf32_flag_is_velocity_local_and_restored_on_exception(monkeypatch, initial, raises):
    matmul = SimpleNamespace(allow_tf32=initial)
    monkeypatch.setattr(torch.backends.cuda, 'matmul', matmul)
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: pytest.fail('TF32 must not autocast'))
    def field(*a, **kw):
        assert matmul.allow_tf32
        if raises:
            raise RuntimeError('field failed')
        return DynamicsOutput(torch.ones(1))
    wrapper = inference._VelocityInference(SimpleNamespace(n_edges=3,forward=field,training=False),
        inference.ImaginationInferenceProfile(64,32768,65536,'tf32',False))
    latent = SimpleNamespace(shape=(24,1,3,64),device=SimpleNamespace(type='cuda'),dtype=torch.float32)
    with torch.no_grad():
        if raises:
            with pytest.raises(RuntimeError, match='field failed'):
                wrapper(None,packed_latents=latent,flow_time=torch.zeros(24,1),condition=object())
        else:
            result = wrapper(None,packed_latents=latent,flow_time=torch.zeros(24,1),condition=object())
            assert result.latent_velocity.dtype == torch.float32
    assert matmul.allow_tf32 == initial


@pytest.mark.parametrize('training_mode,grad_enabled', [(True,False),(False,True)])
def test_tf32_profile_leaves_training_and_gradient_policy_unchanged(monkeypatch, training_mode, grad_enabled):
    matmul = SimpleNamespace(allow_tf32=False)
    monkeypatch.setattr(torch.backends.cuda, 'matmul', matmul)
    def field(*a, **kw):
        assert matmul.allow_tf32 is False
        return DynamicsOutput(torch.ones(1))
    wrapper = inference._VelocityInference(SimpleNamespace(n_edges=3,forward=field,training=training_mode),
        inference.ImaginationInferenceProfile(64,32768,65536,'tf32',False))
    latent = SimpleNamespace(shape=(24,1,3,64),device=SimpleNamespace(type='cuda'),dtype=torch.float32)
    with torch.set_grad_enabled(grad_enabled):
        wrapper(None,packed_latents=latent,flow_time=torch.zeros(24,1),condition=object())
    assert matmul.allow_tf32 is False


def test_tf32_requested_on_cpu_records_effective_fp32(tmp_path):
    config = configured(tmp_path, 'world_model_imagination_precision = "tf32"\n')
    trainer = SimpleNamespace(config=config,world_model=SimpleNamespace(dynamics=_small_dynamics()))
    assert inference.configure_imagination_inference(trainer)['precision'] == 'fp32'



@pytest.mark.parametrize('initial', ['none','ieee','tf32'])
@pytest.mark.parametrize('raises', [False,True])
def test_tf32_new_api_restores_policy_without_touching_legacy(monkeypatch, initial, raises):
    class NewMatmul:
        fp32_precision = initial
        @property
        def allow_tf32(self):
            pytest.fail('new precision API must not read legacy flag')
        @allow_tf32.setter
        def allow_tf32(self, value):
            pytest.fail('new precision API must not set legacy flag')
    matmul = NewMatmul()
    monkeypatch.setattr(torch.backends.cuda, 'matmul', matmul)
    if raises:
        with pytest.raises(RuntimeError, match='failed'):
            with inference._velocity_tf32():
                assert matmul.fp32_precision == 'tf32'
                raise RuntimeError('failed')
    else:
        with inference._velocity_tf32():
            assert matmul.fp32_precision == 'tf32'
    assert matmul.fp32_precision == initial



@pytest.mark.parametrize('mode', ['default','reduce-overhead'])
def test_compile_mode_is_applied_and_survives_deepcopy(monkeypatch, mode):
    import copy
    calls = []
    def compile_fn(forward, **kwargs):
        calls.append(kwargs)
        return lambda *a, **k: forward(*a, **k)
    monkeypatch.setattr(torch, 'compile', compile_fn)
    dynamics = _small_dynamics()
    wrapper = inference._VelocityInference(dynamics,
        inference.ImaginationInferenceProfile(16,32768,16384,'fp32',True,compile_mode=mode))
    dynamics.forward = wrapper
    dynamics._imagination_velocity_inference = wrapper
    copied = copy.deepcopy(dynamics)
    assert copied.forward.profile.compile_mode == mode
    assert len(calls) == 2
    assert all(call == dict(backend='inductor',mode=mode,fullgraph=True,dynamic=False) for call in calls)


def test_cpu_retains_requested_compile_mode_in_effective_metadata(tmp_path):
    config = configured(tmp_path, 'world_model_imagination_compile = true\n'
        'world_model_imagination_compile_mode = "reduce-overhead"\n')
    trainer = SimpleNamespace(config=config,world_model=SimpleNamespace(dynamics=_small_dynamics()))
    profile = inference.configure_imagination_inference(trainer)
    assert profile['compile_mode'] == 'reduce-overhead'
    assert profile['compile'] is False
