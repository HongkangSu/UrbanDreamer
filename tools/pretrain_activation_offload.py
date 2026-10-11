"""Offload autograd saved tensors without changing model or optimizer state."""
from functools import wraps

import torch


def pack_saved_tensor(tensor):
    """Copy unique broadcast storage, preserving the saved tensor's layout."""
    if tensor.layout != torch.strided:
        return tensor.device, tensor.shape, None, tensor.detach().to('cpu')
    shape, strides = tensor.shape, tensor.stride()
    selectors = tuple(slice(0, 1) if stride == 0 and size > 1 else slice(None)
                      for size, stride in zip(shape, strides))
    compact = tensor.detach()[selectors]
    # Avoid materializing a batched broadcast adjacency (64 x M x M), and
    # avoid multi-GiB pinned allocations in a shared training process.
    packed = torch.empty_strided(compact.shape, compact.stride(), dtype=tensor.dtype, device='cpu')
    packed.copy_(compact)
    return tensor.device, shape, strides, packed


def unpack_saved_tensor(record):
    device, shape, strides, packed = record
    if strides is None:
        return packed.to(device)
    if device.type == 'cpu':
        restored = packed
    else:
        restored = torch.empty_strided(packed.shape, packed.stride(), dtype=packed.dtype, device=device)
        restored.copy_(packed)
    return restored.expand(shape)


def enable_pretrain_activation_offload(trainer):
    if getattr(trainer, '_pretrain_activation_offload', False):
        return
    update = trainer._update_world
    last_batch_shapes = None

    @wraps(update)
    def offloaded_update(*args, **kwargs):
        nonlocal last_batch_shapes
        if kwargs.get('initial', False):
            batches = kwargs.get('batches')
            shapes = tuple(tuple(batch.flow.shape) for batch in batches) if batches else None
            device = getattr(trainer, 'device', None)
            if shapes is not None and last_batch_shapes is not None and shapes != last_batch_shapes and device is not None and device.type == 'cuda':
                # Full and remainder shapes otherwise retain incompatible large
                # cached blocks under the bounded shared-GPU allocator.
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            last_batch_shapes = shapes
            with torch.autograd.graph.saved_tensors_hooks(pack_saved_tensor, unpack_saved_tensor):
                return update(*args, **kwargs)
        return update(*args, **kwargs)

    trainer._update_world = offloaded_update
    trainer._pretrain_activation_offload = True
