from types import SimpleNamespace
from contextlib import nullcontext
import os

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tools.pretrain_activation_offload import enable_pretrain_activation_offload, pack_saved_tensor, unpack_saved_tensor


def test_offload_preserves_checkpointed_gradients_and_adam_updates():
    torch.manual_seed(13)
    ordinary = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
    offloaded = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4))
    offloaded.load_state_dict(ordinary.state_dict())
    inputs = torch.randn(3, 4, requires_grad=True)

    def trainer(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        def update(x, *, initial=False):
            optimizer.zero_grad(set_to_none=True)
            loss = checkpoint(model, x, use_reentrant=False).square().mean()
            loss.backward()
            optimizer.step()
            return loss.detach()
        return SimpleNamespace(_update_world=update, optimizer=optimizer)

    left, right = trainer(ordinary), trainer(offloaded)
    enable_pretrain_activation_offload(right)
    original_wrapper = right._update_world
    enable_pretrain_activation_offload(right)
    assert right._update_world is original_wrapper
    for _ in range(3):
        assert torch.equal(left._update_world(inputs, initial=True), right._update_world(inputs, initial=True))
    for a, b in zip(ordinary.parameters(), offloaded.parameters()):
        assert torch.equal(a, b)
        assert torch.equal(a.grad, b.grad)
    for a, b in zip(left.optimizer.state.values(), right.optimizer.state.values()):
        assert torch.equal(a['exp_avg'], b['exp_avg'])
        assert torch.equal(a['exp_avg_sq'], b['exp_avg_sq'])


def test_broadcast_storage_is_not_materialized_and_transpose_layout_is_preserved():
    original = torch.arange(12.).reshape(3, 4).T.unsqueeze(0).expand(64, 4, 3)
    packed = pack_saved_tensor(original)
    assert packed[-1].numel() == 12
    restored = unpack_saved_tensor(packed)
    assert torch.equal(original, restored)
    assert original.stride() == restored.stride()


def test_sparse_and_empty_saved_tensors_round_trip():
    sparse = torch.eye(4).to_sparse()
    assert torch.equal(unpack_saved_tensor(pack_saved_tensor(sparse)).to_dense(), sparse.to_dense())
    empty = torch.empty(0)
    assert unpack_saved_tensor(pack_saved_tensor(empty)).shape == empty.shape


def test_shape_change_clears_only_selected_gpu_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'device', lambda device: (calls.append(str(device)), nullcontext())[1])
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: calls.append('clear'))
    trainer = SimpleNamespace(device=torch.device('cuda:1'), _update_world=lambda *a, **k: 1)
    enable_pretrain_activation_offload(trainer)
    for size in (64, 64, 63, 63, 64):
        trainer._update_world(None, initial=True, batches=[SimpleNamespace(flow=SimpleNamespace(shape=(size, 1, 5, 8)))])
    assert calls == ['cuda:1', 'clear', 'cuda:1', 'clear']


@pytest.mark.skipif(os.environ.get('URB_TEST_CUDA_OFFLOAD') != '1', reason='explicit bounded CUDA verification')
def test_cuda_broadcast_layout_and_checkpointed_gradients_round_trip():
    device = torch.device('cuda:1')
    original = torch.arange(12., device=device).reshape(3, 4).T.unsqueeze(0).expand(64, 4, 3)
    restored = unpack_saved_tensor(pack_saved_tensor(original))
    assert torch.equal(original, restored)
    assert original.stride() == restored.stride()
    torch.manual_seed(19)
    left = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4)).to(device)
    right = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4)).to(device)
    right.load_state_dict(left.state_dict())
    data = torch.randn(3, 4, device=device, requires_grad=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ordinary_loss = checkpoint(left, data, use_reentrant=False).square().mean()
    ordinary_loss.backward()
    with torch.autograd.graph.saved_tensors_hooks(pack_saved_tensor, unpack_saved_tensor):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            offloaded_loss = checkpoint(right, data, use_reentrant=False).square().mean()
        offloaded_loss.backward()
    assert torch.equal(ordinary_loss, offloaded_loss)
    for a, b in zip(left.parameters(), right.parameters()):
        assert torch.equal(a.grad, b.grad)
