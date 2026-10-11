from copy import deepcopy

import pytest
import torch

from urban_dreamer.models.world_model import MLP, SpaceSelfAttention, WORLD_MODEL_DIM


@pytest.mark.parametrize("chunk_size", [0, 1, 2, 16])
def test_packed_matches_chunked_outputs_and_gradients(chunk_size):
    torch.manual_seed(41)
    reference = SpaceSelfAttention(token_count=7).double()
    reference.attention_batch_chunk_size = chunk_size
    packed = deepcopy(reference)
    packed.packed_projections = True
    x_reference = torch.randn(3, 2, 7, WORLD_MODEL_DIM, dtype=torch.float64, requires_grad=True)
    x_packed = x_reference.detach().clone().requires_grad_(True)
    expected = reference(x_reference)
    actual = packed(x_packed)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    weights = torch.randn_like(expected)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(x_packed.grad, x_reference.grad, rtol=1e-9, atol=1e-10)
    for parameter, reference_parameter in zip(packed.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-9, atol=1e-10)


def test_packed_preserves_trajectory_independence():
    torch.manual_seed(7)
    layer = SpaceSelfAttention(token_count=5).eval()
    layer.packed_projections = True
    layer.attention_batch_chunk_size = 2
    x = torch.randn(3, 2, 5, WORLD_MODEL_DIM)
    changed = x.clone()
    changed[1] = torch.randn_like(changed[1]) * 10
    with torch.no_grad():
        original_output = layer(x)
        changed_output = layer(changed)
        single_output = layer(x[:1])
    torch.testing.assert_close(original_output[[0, 2]], changed_output[[0, 2]], rtol=0, atol=0)
    torch.testing.assert_close(original_output[:1], single_output)
    assert not torch.allclose(original_output[1], changed_output[1])


def test_only_attention_is_chunked_and_state_dict_is_unchanged():
    layer = SpaceSelfAttention(token_count=5).eval()
    assert layer.packed_projections is False
    saved = deepcopy(layer.state_dict())
    layer.attention_batch_chunk_size = 2
    calls = {"qkv": [], "output": []}
    handles = [
        module.register_forward_pre_hook(
            lambda _module, args, name=name: calls[name].append(tuple(args[0].shape))
        )
        for name, module in [("qkv", layer.attention.qkv), ("output", layer.attention.output)]
    ]
    x = torch.randn(3, 2, 5, WORLD_MODEL_DIM)
    try:
        with torch.no_grad():
            layer(x)
            assert len(calls["qkv"]) == len(calls["output"]) == 3
            calls["qkv"].clear()
            calls["output"].clear()
            layer.packed_projections = True
            layer(x)
        assert calls == {"qkv": [(30, WORLD_MODEL_DIM)], "output": [(30, WORLD_MODEL_DIM)]}
    finally:
        for handle in handles:
            handle.remove()
    assert layer.state_dict().keys() == saved.keys()
    for name, value in saved.items():
        torch.testing.assert_close(layer.state_dict()[name], value, rtol=0, atol=0)
    layer.load_state_dict(saved, strict=True)


def test_unchunked_mlp_matches_token_chunked_mlp():
    torch.manual_seed(11)
    layer = MLP(width=16).double()
    x = torch.randn(3, 2, 7, 16, dtype=torch.float64)
    expected = layer(x)
    layer.token_chunk_size = 11
    torch.testing.assert_close(layer(x), expected, rtol=1e-10, atol=1e-11)
