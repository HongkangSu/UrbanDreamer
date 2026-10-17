import copy
import math

import pytest
import torch

import urban_dreamer.models.world_model as wm


def _reference_positions(tokens):
    _, time_steps, token_count, width = tokens.shape
    position_t = wm.sinusoid_table(time_steps, width, device=tokens.device)
    position_s = wm.sinusoid_table(token_count, width, device=tokens.device)
    position = (position_t[None, :, None, :] + position_s[None, None, :, :]) * (1.0 / math.sqrt(width))
    return tokens + position.to(dtype=tokens.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_position_outputs_and_gradients_are_exact(dtype):
    tokens = torch.randn(2, 3, 7, 32, dtype=dtype, requires_grad=True)
    reference = tokens.detach().clone().requires_grad_()
    expected = _reference_positions(reference)
    actual = wm.add_sinusoidal_positions(tokens)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    assert torch.equal(tokens.grad, reference.grad)


def test_inference_warmup_does_not_poison_training_constants():
    with torch.inference_mode():
        wm.add_sinusoidal_positions(torch.zeros(2, 2, 9, 32))
    tokens = torch.randn(1, 2, 9, 32, requires_grad=True)
    output = wm.add_sinusoidal_positions(tokens)
    output.square().sum().backward()
    assert torch.equal(tokens.grad, 2 * output.detach())


def test_position_reuse_skips_table_construction_and_is_bounded(monkeypatch):
    wm._sinusoidal_positions.cache_clear()
    original = wm.sinusoid_table
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(wm, "sinusoid_table", counted)
    for batch in (1, 2, 3):
        wm.add_sinusoidal_positions(torch.zeros(batch, 2, 8, 32))
    assert len(calls) == 2
    for count in range(9, 15):
        wm.add_sinusoidal_positions(torch.zeros(1, 2, count, 32))
    assert wm._sinusoidal_positions.cache_info().currsize == 4


def test_compile_path_does_not_mutate_eager_cache(monkeypatch):
    wm._sinusoidal_positions.cache_clear()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    tokens = torch.zeros(1, 2, 8, 32)
    assert torch.equal(wm.add_sinusoidal_positions(tokens), _reference_positions(tokens))
    assert wm._sinusoidal_positions.cache_info().currsize == 0


def test_deterministic_checkpoint_matches_reference_gradients_and_rng(monkeypatch):
    monkeypatch.setattr(wm, "TRANSFORMER_DEPTH", 1)
    monkeypatch.setattr(wm, "WORLD_MODEL_DIM", 32)
    model = wm.BlockCausalTransformer(5)
    reference = copy.deepcopy(model)
    reference.activation_checkpointing = False
    tokens = torch.randn(2, 2, 5, 32, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    state_keys = tuple(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    output = model(tokens)
    expected = reference(reference_tokens)
    output.square().mean().backward()
    expected.square().mean().backward()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(output, expected)
    assert torch.equal(tokens.grad, reference_tokens.grad)
    assert tuple(model.state_dict()) == state_keys
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        assert torch.equal(parameter.grad, reference_parameter.grad)
