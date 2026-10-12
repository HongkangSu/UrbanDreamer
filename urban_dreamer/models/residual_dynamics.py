"""Experimental normalized residual FM around one frozen regression predictor.

The codec stays outside this adapter. The center is an approximate deterministic
prediction, not a known conditional mean. Inputs, center and condition features
are detached: repeated inference transitions are supported, but recursive
training/BPTT and parallel policy imagination are not admitted by this module.
Only the small residual network learns; source noise has full latent support.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping

import torch
from torch import nn

from .world_model import ConditionEncoding, EdgeGNNLayer, FLOW_LATENT_DIM, WORLD_MODEL_DIM

RESIDUAL_VARIANT = 'residual_fm_regression_center_v1'
_IDENTITY_HASHES = ('mean_source_sha256', 'codec_sha256', 'manifest_sha256', 'scale_artifact_sha256')


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')


def _state_digest(state):
    """Content hash independent of torch.save container/storage serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b'\0')
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def _identity(identity, scale):
    if not isinstance(identity, Mapping):
        raise ValueError('residual identity must be a mapping')
    result = deepcopy(dict(identity))
    for key in _IDENTITY_HASHES:
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
            raise ValueError(f'residual identity requires lowercase SHA256 {key}')
    if 'scale_values' in result:
        recorded = torch.as_tensor(result['scale_values'], dtype=torch.float32)
        if recorded.shape != (FLOW_LATENT_DIM,) or not torch.equal(recorded, scale.detach().cpu()):
            raise ValueError('scale buffer must equal the declared scale artifact values')
    result['scale_values'] = scale.detach().cpu().tolist()
    # Check that the entire identity can survive JSON and weights-only loading.
    return json.loads(json.dumps(result, sort_keys=True, allow_nan=False))


class ResidualVelocity(nn.Module):
    """Two directed graph layers with a solve-local trainable context cache."""

    def __init__(self, adjacency):
        super().__init__()
        graph = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or graph.shape[0] < 1:
            raise ValueError('residual adjacency must be a nonempty square matrix')
        _finite(graph, 'residual adjacency')
        if (graph < 0).any():
            raise ValueError('residual adjacency must be nonnegative')
        graph = graph / graph.sum(-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer('adjacency', graph)
        self.state_proj = nn.Linear(FLOW_LATENT_DIM, 128)
        self.mean_proj = nn.Linear(FLOW_LATENT_DIM, 128)
        self.condition_proj = nn.Linear(WORLD_MODEL_DIM, 128)
        self.input_proj = nn.Linear(FLOW_LATENT_DIM, 128)
        self.time_proj = nn.Linear(33, 128)
        self.layers = nn.ModuleList([EdgeGNNLayer(128), EdgeGNNLayer(128)])
        self.output = nn.Linear(128, FLOW_LATENT_DIM)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def context(self, current, mean, condition):
        return self.state_proj(current.detach()) + self.mean_proj(mean.detach()) + self.condition_proj(condition.edge_tokens.detach())

    def forward(self, residual, flow_time, context):
        frequencies = torch.arange(1, 17, device=residual.device, dtype=torch.float32)
        angles = math.pi * flow_time.float().unsqueeze(-1) * frequencies
        time_features = torch.cat((flow_time.float().unsqueeze(-1), angles.sin(), angles.cos()), -1)
        hidden = context + self.input_proj(residual) + self.time_proj(time_features).unsqueeze(-2)
        graph = self.adjacency.to(dtype=hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, graph)
        return self.output(hidden)


class ResidualDynamics(nn.Module):
    """Take ownership of a verified regression mean; retain one frozen backbone.

    Fresh construction's source SHA/codec/manifest verification belongs to the
    builder that read those artifacts. This module validates identity and scale,
    embeds the frozen mean including its inference buffers, and validates its
    content digest on checkpoint load. A matching random regression shell can
    therefore restore entirely from the embedded state, without the source file.
    """
    variant = RESIDUAL_VARIANT

    def __init__(self, mean_dynamics, scale, *, identity, initialization_seed):
        super().__init__()
        if getattr(mean_dynamics, 'variant', None) != 'regression':
            raise ValueError('residual v1 requires the no-prior deterministic regression center')
        if isinstance(initialization_seed, bool) or not isinstance(initialization_seed, int) or initialization_seed < 0:
            raise ValueError('initialization_seed must be a nonnegative integer')
        scale = torch.as_tensor(scale, dtype=torch.float32).detach().cpu().clone()
        if scale.shape != (FLOW_LATENT_DIM,) or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError('residual scale must contain 64 finite positive channel values')
        self._identity = _identity(identity, scale)
        self.initialization_seed = initialization_seed
        self.n_edges = mean_dynamics.n_edges
        self.mean_dynamics = mean_dynamics
        if any(p.dtype != torch.float32 for p in mean_dynamics.parameters()):
            raise ValueError('frozen regression center parameters must remain FP32')
        mean_dynamics.requires_grad_(False)
        for parameter in mean_dynamics.parameters():
            parameter.grad = None
        # Inference adjacency/static features/masks must survive a standalone
        # checkpoint; these buffers were nonpersistent in the production shell.
        for module in mean_dynamics.modules():
            module._non_persistent_buffers_set.clear()
        mean_dynamics.eval()
        mean_dynamics.transformer.activation_checkpointing = False
        device = next(mean_dynamics.parameters()).device
        self.register_buffer('scale', scale.to(device))
        with torch.random.fork_rng(devices=[]):
            # Seed only CPU initialization; leave caller CPU and CUDA streams
            # unchanged, including when the supplied center already lives on GPU.
            torch.random.default_generator.manual_seed(initialization_seed)
            self.residual_velocity = ResidualVelocity(mean_dynamics.action_encoder.adjacency.cpu()).to(device)
        self._mean_digest = _state_digest(mean_dynamics.state_dict())
        self.last_execution = None

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def event_encoder(self):
        return self.mean_dynamics.event_encoder

    @property
    def event_protocol(self):
        return self.mean_dynamics.event_protocol

    @property
    def action_encoder(self):
        return self.mean_dynamics.action_encoder

    def train(self, mode=True):
        super().train(mode)
        self.mean_dynamics.eval()
        self.mean_dynamics.transformer.activation_checkpointing = False
        return self

    def set_initial_chunk_profile(self):
        self.mean_dynamics.set_initial_chunk_profile()
        self.mean_dynamics.transformer.activation_checkpointing = False

    def set_online_chunk_profile(self):
        self.mean_dynamics.set_online_chunk_profile()
        self.mean_dynamics.transformer.activation_checkpointing = False

    def set_pretraining_execution_profile(self, *, space_chunk, time_chunk, mlp_chunk,
                                          activation_checkpointing):
        self.mean_dynamics.set_pretraining_execution_profile(space_chunk=space_chunk,
            time_chunk=time_chunk, mlp_chunk=mlp_chunk, activation_checkpointing=activation_checkpointing)
        self.mean_dynamics.transformer.activation_checkpointing = False

    def _architecture(self):
        return dict(n_edges=self.n_edges, latent_width=64, condition_width=256,
                    hidden_width=128, graph_layers=2, time_frequencies=16,
                    mean_transformer_layers=len(self.mean_dynamics.transformer.layers),
                    event_protocol=self.event_protocol)

    def get_extra_state(self):
        return dict(version=1, variant=self.variant, identity=self.identity,
                    initialization_seed=self.initialization_seed, architecture=self._architecture(),
                    mean_state_sha256=self._mean_digest)

    def set_extra_state(self, state):
        if (state.get('version') != 1 or state.get('variant') != self.variant
                or state.get('identity') != self._identity or state.get('architecture') != self._architecture()):
            raise ValueError('residual checkpoint identity or architecture mismatch')
        self.initialization_seed = int(state['initialization_seed'])
        self._mean_digest = state['mean_state_sha256']

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        extra = state_dict.get(prefix + '_extra_state')
        if not isinstance(extra, dict):
            raise ValueError('residual checkpoint requires versioned identity state')
        if (extra.get('identity') != self._identity or extra.get('architecture') != self._architecture()
                or extra.get('version') != 1 or extra.get('variant') != self.variant):
            raise ValueError('residual checkpoint source/codec/scale identity or architecture mismatch')
        scale = state_dict.get(prefix + 'scale')
        if scale is None or scale.dtype != torch.float32 or not torch.equal(
                scale.detach().cpu(), torch.tensor(self._identity['scale_values'], dtype=torch.float32)):
            raise ValueError('checkpoint scale buffer differs from declared artifact')
        mean_prefix = prefix + 'mean_dynamics.'
        embedded_mean = {key[len(mean_prefix):]: value for key, value in state_dict.items() if key.startswith(mean_prefix)}
        if _state_digest(embedded_mean) != extra.get('mean_state_sha256'):
            raise ValueError('checkpoint frozen mean content digest mismatch')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _check_integrity(self):
        expected = torch.tensor(self._identity['scale_values'], dtype=torch.float32, device=self.scale.device)
        if self.scale.dtype != torch.float32 or not torch.equal(self.scale, expected):
            raise ValueError('residual scale is immutable and must equal its artifact')
        if next(self.mean_dynamics.parameters()).dtype != torch.float32:
            raise ValueError('frozen regression center must remain FP32')

    def metadata(self):
        return dict(variant=self.variant, residual_contract_version=1, identity=self.identity,
            initialization_seed=self.initialization_seed, architecture=self._architecture(),
            parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            frozen_mean_parameters=sum(p.numel() for p in self.mean_dynamics.parameters()),
            mean_state_sha256=self._mean_digest,
            noise_space='full normalized latent residual; unit coordinate Gaussian',
            source_variance_per_channel=self.scale.detach().cpu().square().tolist(),
            expected_source_energy=self.n_edges * float(self.scale.detach().double().square().sum()),
            solver_support=['euler', 'midpoint'], precision='FP32 center, source, integrator and outputs',
            state_gradient='detached; no recursive training/BPTT',
            time_features='tau, sin(pi*j*tau) j=1..16, cos(pi*j*tau) j=1..16')

    def _inputs(self, z, action):
        self._check_integrity()
        if z.ndim not in (3, 4) or tuple(z.shape[-2:]) != (self.n_edges, FLOW_LATENT_DIM) or any(size < 1 for size in z.shape[:-2]):
            raise ValueError('z requires nonempty (B,E,64) or (B,T,E,64) shape')
        if action.shape != z.shape[:-1]:
            raise ValueError('action shape must match the latent batch/time/road axes')
        if z.device != self.scale.device or action.device != z.device:
            raise ValueError('latent, action and residual model devices must agree')
        _finite(z, 'z')
        _finite(action, 'action')
        squeezed = z.ndim == 3
        z, action = z.detach().float(), action.detach().float()
        return (z.unsqueeze(1), action.unsqueeze(1), True) if squeezed else (z, action, False)

    def encode_condition(self, action, *, action_events=None):
        self.mean_dynamics.eval()
        with torch.no_grad(), torch.autocast(device_type=action.device.type, enabled=False):
            condition = self.mean_dynamics.encode_condition(action.detach().float(), action_events=action_events)
        _finite(condition.edge_tokens, 'frozen condition')
        return condition

    def mean_and_condition(self, z, action, *, action_events=None):
        """One frozen forward; accepts canonical (B,T,E,64) latent inputs."""
        z, action, squeezed = self._inputs(z, action)
        if squeezed:
            raise ValueError('mean_and_condition requires an explicit time axis')
        self.mean_dynamics.eval()
        with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
            condition = self.encode_condition(action, action_events=action_events)
            cached = ConditionEncoding(condition.edge_tokens, self.mean_dynamics.reference_proj(z))
            delta = self.mean_dynamics.backbone(action, packed_latents=z,
                flow_time=z.new_zeros(z.shape[:2]), condition=cached).latent_velocity
            mean = z + delta
        _finite(mean, 'frozen regression center')
        return mean, condition

    @staticmethod
    def solver_nfe(steps=2, solver='euler', picard_iterations=None, picard_batch_chunk_size=64):
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError('residual solver steps must be a positive integer')
        if solver not in ('euler', 'midpoint') or picard_iterations is not None:
            raise ValueError('residual solver supports only Euler/midpoint; Picard is unsupported')
        if isinstance(picard_batch_chunk_size, bool) or not isinstance(picard_batch_chunk_size, int) or picard_batch_chunk_size < 0:
            raise ValueError('picard_batch_chunk_size must be a nonnegative compatibility argument')
        return steps * (2 if solver == 'midpoint' else 1)

    @staticmethod
    def _noise(noise, shape, device):
        if noise is None:
            return torch.randn(shape, device=device, dtype=torch.float32)
        if tuple(noise.shape) != tuple(shape):
            raise ValueError(f'residual noise must have full latent shape {tuple(shape)}')
        _finite(noise, 'residual noise')
        return noise.detach().to(device=device, dtype=torch.float32)

    def _solve(self, noise, context, *, steps, solver):
        # The center and normalized solver always use FP32, even if called from
        # an enclosing BF16 training context. Only loss-field calls may autocast.
        with torch.autocast(device_type=noise.device.type, enabled=False):
            residual, context = noise.float(), context.float()
            for node in range(steps):
                tau = residual.new_full(residual.shape[:-2], node / steps)
                velocity = self.residual_velocity(residual, tau, context).float()
                if solver == 'midpoint':
                    midpoint = residual + velocity / (2 * steps)
                    velocity = self.residual_velocity(midpoint, tau + .5 / steps, context).float()
                residual = residual + velocity / steps
                _finite(residual, 'integrated residual')
        return residual

    def transition(self, z, action, *, noise=None, steps=2, solver='euler',
                   picard_iterations=None, picard_batch_chunk_size=64, action_events=None):
        nfe = self.solver_nfe(steps, solver, picard_iterations, picard_batch_chunk_size)
        original_shape = z.shape
        z, action, squeezed = self._inputs(z, action)
        noise = self._noise(noise, original_shape, z.device)
        if squeezed:
            noise = noise.unsqueeze(1)
        mean, condition = self.mean_and_condition(z, action, action_events=action_events)
        with torch.autocast(device_type=z.device.type, enabled=False):
            context = self.residual_velocity.context(z, mean, condition)
            residual = self._solve(noise, context, steps=steps, solver=solver)
            result = mean + self.scale * residual
        _finite(result, 'predicted latent')
        self.last_execution = dict(center_forwards=1, residual_nfe_per_sample=nfe,
                                   vectorized_field_calls=nfe, samples=1, solver=solver)
        return result.squeeze(1) if squeezed else result

    def latent_disagreement(self, z, action, *, n_samples=2, noise=None, steps=2,
                            solver='euler', picard_iterations=None, picard_batch_chunk_size=64,
                            action_events=None, sample_chunk_size=None):
        nfe = self.solver_nfe(steps, solver, picard_iterations, picard_batch_chunk_size)
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 2:
            raise ValueError('latent disagreement requires at least two integer samples')
        chunk = n_samples if sample_chunk_size is None else sample_chunk_size
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError('sample_chunk_size must be a positive integer')
        original_shape = z.shape
        z, action, squeezed = self._inputs(z, action)
        noise = self._noise(noise, (n_samples, *original_shape), z.device)
        if squeezed:
            noise = noise.unsqueeze(2)
        mean, condition = self.mean_and_condition(z, action, action_events=action_events)
        with torch.autocast(device_type=z.device.type, enabled=False):
            context = self.residual_velocity.context(z, mean, condition)
            outputs = []
            for begin in range(0, n_samples, chunk):
                count = min(chunk, n_samples - begin)
                repeated = context.unsqueeze(0).expand(count, *context.shape).reshape(count * z.shape[0], *context.shape[1:])
                residual = self._solve(noise[begin:begin + count].reshape(count * z.shape[0], *z.shape[1:]),
                                       repeated, steps=steps, solver=solver)
                outputs.append(mean.unsqueeze(0) + self.scale * residual.reshape(count, *z.shape))
            samples = torch.cat(outputs, dim=0)
        _finite(samples, 'predicted samples')
        if squeezed:
            samples = samples.squeeze(2)
        variance = samples.var(0, unbiased=True).mean(-1)
        self.last_execution = dict(center_forwards=1, residual_nfe_per_sample=nfe,
            vectorized_field_calls=nfe * math.ceil(n_samples / chunk), samples=n_samples, solver=solver)
        return samples, variance.mean(-1), variance


def residual_fm_loss(dynamics: ResidualDynamics, *, z_inputs, z_targets, action,
                     action_events=None, noise=None, flow_time=None, predict_endpoints=True,
                     steps=2, solver='euler'):
    """Single-step normalized velocity regression; no endpoint/prior penalties.

    Inputs, targets and frozen context are detached. Scale calibration and codec
    supervision are external and are never performed by this loss.
    """
    if not isinstance(dynamics, ResidualDynamics):
        raise ValueError('residual loss requires ResidualDynamics')
    if z_inputs.ndim != 4 or z_inputs.shape[1] != 1 or z_targets.shape != z_inputs.shape:
        raise ValueError('residual v1 training requires matching (B,1,E,64) single-step inputs/targets')
    if not isinstance(predict_endpoints, bool):
        raise ValueError('predict_endpoints must be boolean')
    dynamics.solver_nfe(steps, solver)
    current, action, _ = dynamics._inputs(z_inputs, action)
    _finite(z_targets, 'z_targets')
    if z_targets.device != current.device:
        raise ValueError('target and input devices must agree')
    noise = dynamics._noise(noise, current.shape, current.device)
    if flow_time is None:
        flow_time = torch.rand(current.shape[:2], device=current.device, dtype=torch.float32)
    if flow_time.shape != current.shape[:2]:
        raise ValueError('flow_time must match batch and single-step axes')
    _finite(flow_time, 'flow_time')
    if (flow_time < 0).any() or (flow_time > 1).any():
        raise ValueError('flow_time must lie in [0,1]')
    flow_time = flow_time.detach().to(device=current.device, dtype=torch.float32)
    mean, condition = dynamics.mean_and_condition(current, action, action_events=action_events)
    target = z_targets.detach().float()
    residual_target = (target - mean) / dynamics.scale
    tau = flow_time[..., None, None]
    point = (1 - tau) * noise + tau * residual_target
    target_velocity = residual_target - noise
    context = dynamics.residual_velocity.context(current, mean, condition)
    velocity = dynamics.residual_velocity(point, flow_time, context)
    loss = (velocity.float() - target_velocity).square().mean()
    _finite(loss, 'residual FM loss')
    if predict_endpoints:
        with torch.no_grad():
            predicted = mean + dynamics.scale * dynamics._solve(noise, context.detach(), steps=steps, solver=solver)
    else:
        predicted = target
    return loss, dict(flow_mse=loss.detach(), residual_flow_mse=loss.detach(),
                     normalized_target_rms=residual_target.square().mean().sqrt().detach()), predicted.detach()
