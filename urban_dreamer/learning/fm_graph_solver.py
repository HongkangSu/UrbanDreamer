"""Replay a complete, unchanged FP32 Euler integration as one CUDA graph.

This is a plain callable, not a module: graph workspaces never enter checkpoints
or optimizers. Conditions and prior noise are prepared by Dynamics before solve.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import copy
import time

import torch

from ..models.world_model import ConditionEncoding


_CONDITION_FIELDS = tuple(field.name for field in fields(ConditionEncoding))


@dataclass
class _CapturedSolve:
    graph: object
    start: torch.Tensor
    action: torch.Tensor
    condition: ConditionEncoding
    output: torch.Tensor
    flow_times: tuple[torch.Tensor, ...] = ()


class EulerGraphSolver:
    """Bounded shape cache with eager fallback for training and unsupported calls.

    Parameter values are read from their existing storage on every replay. An
    optimizer's in-place updates therefore need no recapture. Storage replacement
    and chunk policy changes invalidate cached graphs. No learned values are
    precomputed outside the graph.
    """
    def __init__(self, dynamics, *, max_cached_shapes=2, memory_reserve_bytes=512 * 1024**2,
                 allowed_batches=(3, 6, 12, 24)):
        if max_cached_shapes < 1 or memory_reserve_bytes < 0:
            raise ValueError('cache size must be positive and memory reserve nonnegative')
        self.dynamics = dynamics
        self.eager = dynamics.solve
        self.max_cached_shapes = int(max_cached_shapes)
        self.memory_reserve_bytes = int(memory_reserve_bytes)
        self.allowed_batches = tuple(allowed_batches)
        self.cache = OrderedDict()
        self.failed = set()
        self.signature = None
        self.stats = {'captures': 0, 'replays': 0, 'fallbacks': 0, 'capture_seconds': 0., 'failures': []}

    def __deepcopy__(self, memo):
        copied = type(self).__new__(type(self))
        memo[id(self)] = copied
        # Register before copying Dynamics, which can hold this callable itself.
        copied.dynamics = copy.deepcopy(self.dynamics, memo)
        copied.eager = copy.deepcopy(self.eager, memo)
        copied.max_cached_shapes = self.max_cached_shapes
        copied.memory_reserve_bytes = self.memory_reserve_bytes
        copied.allowed_batches = self.allowed_batches
        copied.cache = OrderedDict()
        copied.failed = set()
        copied.signature = None
        copied.stats = {'captures': 0, 'replays': 0, 'fallbacks': 0, 'capture_seconds': 0., 'failures': []}
        return copied

    def clear(self):
        self.cache.clear()
        self.failed.clear()

    def _signature(self):
        tensors = tuple((id(t), t.data_ptr(), tuple(t.shape), t.dtype, t.device)
                        for t in (*self.dynamics.parameters(), *self.dynamics.buffers()))
        chunks = tuple((layer.space.attention_batch_chunk_size,
                        layer.time.attention_batch_chunk_size, layer.mlp.token_chunk_size)
                       for layer in self.dynamics.transformer.layers)
        backend = torch.backends.cuda.matmul
        precision = backend.fp32_precision if hasattr(backend, 'fp32_precision') else backend.allow_tf32
        wrapper = getattr(self.dynamics, '_imagination_velocity_inference', None)
        return tensors, chunks, precision, getattr(wrapper, 'profile', None)

    def _supported(self, start, action, condition, steps, solver):
        return (solver == 'euler' and isinstance(steps, int) and not isinstance(steps, bool) and steps > 0
                and not torch.is_grad_enabled() and not self.dynamics.training
                and not torch.is_autocast_enabled('cuda')
                and getattr(getattr(getattr(self.dynamics, '_imagination_velocity_inference', None),
                                    'profile', None), 'precision', 'fp32') == 'fp32'
                and start.device.type == 'cuda' and start.dtype == torch.float32
                and start.ndim == 4 and start.shape[0] in self.allowed_batches and start.shape[1] == 1
                and action.device == start.device and action.dtype == torch.float32
                and tuple(action.shape) == tuple(start.shape[:-1])
                and condition.edge_tokens is not None
                and all(value is None or (value.device == start.device and value.dtype == torch.float32)
                        for value in (getattr(condition, name) for name in _CONDITION_FIELDS))
                and all(value is None or value.shape == start.shape
                        for value in (condition.current_state, condition.background)))

    def _capture(self, start, action, condition, steps):
        free, _ = torch.cuda.mem_get_info(start.device)
        if free < self.memory_reserve_bytes:
            raise RuntimeError('insufficient GPU headroom for Euler graph capture')
        static_start = start.clone()
        static_action = action.clone()
        static_condition = ConditionEncoding(**{
            name: None if (value := getattr(condition, name)) is None else value.clone()
            for name in _CONDITION_FIELDS})
        # Bypass the compile dispatcher inside a manually captured CUDA graph.
        wrapper = getattr(self.dynamics, '_imagination_velocity_inference', None)
        forward = wrapper.eager if wrapper is not None else self.dynamics.forward
        times = [torch.full(start.shape[:2], node / steps, device=start.device, dtype=start.dtype)
                 for node in range(steps)]

        def integrate():
            state = static_start
            for flow_time in times:
                velocity = forward(static_action, packed_latents=state, flow_time=flow_time,
                                   condition=static_condition).latent_velocity
                state = state + velocity / steps
            return state

        current = torch.cuda.current_stream(start.device)
        stream = torch.cuda.Stream(device=start.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            # Initialize cuBLAS/SDPA kernels before capture, without changing state.
            for _ in range(3):
                forward(static_action, packed_latents=static_start, flow_time=times[0], condition=static_condition)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = integrate()
        current.wait_stream(stream)
        return _CapturedSolve(graph, static_start, static_action, static_condition, output, tuple(times))

    def __call__(self, start, action, condition, *, steps=2, solver='euler',
                 picard_iterations=None, picard_batch_chunk_size=64):
        options = dict(steps=steps, solver=solver, picard_iterations=picard_iterations,
                       picard_batch_chunk_size=picard_batch_chunk_size)
        if not self._supported(start, action, condition, steps, solver):
            self.stats['fallbacks'] += 1
            return self.eager(start, action, condition, **options)
        signature = self._signature()
        if signature != self.signature:
            self.clear()
            self.signature = signature
        key = (tuple(start.shape), start.dtype, start.device,
               tuple(None if (value := getattr(condition, name)) is None else tuple(value.shape)
                     for name in _CONDITION_FIELDS), steps)
        if key in self.failed:
            self.stats['fallbacks'] += 1
            return self.eager(start, action, condition, **options)
        if key not in self.cache:
            while len(self.cache) >= self.max_cached_shapes:
                self.cache.popitem(last=False)
            before = time.perf_counter()
            try:
                self.cache[key] = self._capture(start, action, condition, steps)
            except (RuntimeError, torch.OutOfMemoryError) as exc:
                self.failed.add(key)
                self.stats['failures'].append(str(exc))
                self.stats['fallbacks'] += 1
                return self.eager(start, action, condition, **options)
            self.stats['capture_seconds'] += time.perf_counter() - before
            self.stats['captures'] += 1
        cached = self.cache[key]
        self.cache.move_to_end(key)
        cached.start.copy_(start)
        cached.action.copy_(action)
        for name in _CONDITION_FIELDS:
            value = getattr(condition, name)
            if value is not None:
                getattr(cached.condition, name).copy_(value)
        cached.graph.replay()
        self.stats['replays'] += 1
        # Each caller owns its result: later replay may overwrite graph outputs.
        return cached.output.clone()
