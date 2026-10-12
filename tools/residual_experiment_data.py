"""Strict source, training-only normalization and self-contained residual loading."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

from tools.component_experiment_data import load_frozen, sha256
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.models.experimental_dynamics import apply_variant

VARIANT = 'residual_fm_regression_center_v1'
SCHEMA = 2


def json_sha256(value):
    """Hash the exact serialization used by component_experiment_data.atomic_json."""
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True,
        allow_nan=False) + '\n').encode()).hexdigest()


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


@contextmanager
def preserved_torch_rng():
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        yield


def validate_campaign(directory):
    directory = Path(directory)
    parent = json.loads((directory / 'protocol.json').read_text())
    for relative, key in [('snapshot/run_config.toml', 'config_sha256'),
                          ('snapshot/checkpoints/latest.pt', 'checkpoint_sha256')]:
        if sha256(directory / relative) != parent[key]:
            raise ValueError(f'campaign source digest changed: {relative}')
    path = directory / 'development_manifest.json'
    manifest = json.loads(path.read_text())
    if manifest['identity'] != _identity({k: v for k, v in manifest.items() if k != 'identity'}):
        raise ValueError('development manifest identity is invalid')
    if manifest['identity'] != parent['development_manifest_identity']:
        raise ValueError('campaign development manifest changed')
    groups = {split: {row['demand_group'] for row in manifest['entries'] if row['split'] == split}
              for split in ('train', 'validation')}
    if not all(groups.values()) or groups['train'] & groups['validation']:
        raise ValueError('training and validation demand groups must be disjoint and nonempty')
    return parent, sha256(path)


def load_center(directory, checkpoint, expected_sha256, *, device='cpu'):
    parent, manifest_sha = validate_campaign(directory)
    raw = Path(checkpoint).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('mean source checkpoint SHA256 mismatch')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    protocol = payload['protocol']
    if (payload.get('schema_version') != 1 or payload.get('variant') != 'regression'
            or protocol.get('variant', {}).get('variant') != 'regression'
            or protocol.get('parent') != parent or protocol.get('mode') != 'single_step'
            or protocol.get('horizon') != 1 or not protocol.get('frozen_codec')
            or payload['seed'] != protocol['training_seed']
            or payload['update'] != protocol['planned_updates'] or payload['update'] < 1
            or payload['exposures'] != payload['update'] * protocol['batch_size']):
        raise ValueError('mean source must be a completed compatible single-step regression')
    if payload['model'].get('dynamics._extra_state') != dict(
            variant='regression', seed=payload['seed'], version=1):
        raise ValueError('mean serialized variant/seed differs')
    if (not all(name in payload for name in ('data_rng', 'torch_rng', 'cuda_rng', 'optimizer'))
            or protocol.get('sampler') != 'uniform_legal_windows_with_replacement_episode_batches'
            or 'data_seed' not in protocol):
        raise ValueError('mean source requires full RNG and supported sampler continuation contract')
    sampler = np.random.default_rng()
    sampler.bit_generator.state = copy.deepcopy(payload['data_rng'])
    with preserved_torch_rng():
        config, prepared, topology, model, _ = load_frozen(directory,
            fresh_dynamics_seed=payload['seed'], device='cpu')
        codec = state_digest(model.flow_projector.state_dict())
        source_codec = {k.removeprefix('flow_projector.'): v for k, v in payload['model'].items()
                        if k.startswith('flow_projector.')}
        if codec != state_digest(source_codec):
            raise ValueError('mean source codec differs from frozen campaign codec')
        apply_variant(model, 'regression', payload['seed'])
        model.load_state_dict(payload['model'], strict=True)
    model.requires_grad_(False).to(device).eval()
    provenance = dict(mean_source_sha256=expected_sha256, codec_sha256=codec,
        mean_source_path=str(Path(checkpoint).resolve()),
        manifest_sha256=manifest_sha, mean_seed=payload['seed'], mean_updates=payload['update'],
        mean_exposures=payload['exposures'], mean_training_seconds=payload['training_seconds'],
        mean_data_rng=copy.deepcopy(payload['data_rng']),
        mean_data_rng_identity=_identity(payload['data_rng']),
        mean_training_protocol=protocol)
    return config, prepared, topology, model, parent, provenance


def training_lengths(source):
    entries = source.manifest['entries']
    if not entries or any(row.get('split') != 'train' for row in entries):
        raise ValueError('residual calibration/training accepts training partition only')
    if getattr(source, 'sequence_length', 1) != 1:
        raise ValueError('residual v1 requires single-step training data')
    lengths = np.asarray([row['windows'] for row in entries], dtype=np.int64)
    if (lengths < 0).any() or lengths.sum() <= 0:
        raise ValueError('no legal training windows')
    return lengths


def latent_batch(model, batch, capacities, device):
    scoped = prepare_transition_targets(batch, capacities=capacities)
    if batch.action.shape[1] != 1:
        raise ValueError('residual v1 accepts exactly one transition')
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        current = model.encode(batch.flow[:, 0].to(device).float()).unsqueeze(1)
        target = model.encode(scoped.transition_target_flow[:, 0].to(device).float()).unsqueeze(1)
    return dict(z_inputs=current, z_targets=target, action=batch.action.to(device),
                action_events=batch.action_events.to(device))


@torch.no_grad()
def calibrate_scale(model, source, capacities, provenance, *, count=4096, seed=20261012, batch_size=16):
    if min(count, batch_size) < 1:
        raise ValueError('calibration budgets must be positive')
    lengths = training_lengths(source)
    rng = np.random.default_rng(seed)
    before = state_digest(model.state_dict())
    mode = model.training
    selected, total, coordinates = [], torch.zeros(64, dtype=torch.float64), 0
    device = next(model.parameters()).device
    try:
        model.eval()
        with preserved_torch_rng():
            for offset in range(0, count, batch_size):
                size = min(batch_size, count - offset)
                episode = int(rng.choice(len(lengths), p=lengths / lengths.sum()))
                starts = rng.integers(lengths[episode], size=size).tolist()
                row = source.manifest['entries'][episode]
                selected.extend(dict(episode=episode, start=int(start), path=row['path'],
                    demand_group=row['demand_group']) for start in starts)
                batch = source.batch_from_starts(episode, starts, fm_only=True)
                encoded = latent_batch(model, batch, capacities, device)
                mean = model.transition(encoded['z_inputs'], encoded['action'],
                    action_events=encoded['action_events'])
                residual = (encoded['z_targets'] - mean).double().cpu()
                if not torch.isfinite(residual).all():
                    raise FloatingPointError('nonfinite mean residual during scale calibration')
                total += residual.square().sum(dim=(0, 1, 2))
                coordinates += residual.numel() // 64
    finally:
        model.train(mode)
    if state_digest(model.state_dict()) != before:
        raise RuntimeError('scale calibration mutated model weights')
    raw = (total / coordinates).sqrt()
    global_rms = raw.square().mean().sqrt().item()
    floor = max(1e-4, .05 * global_rms)
    scale = raw.clamp_min(floor).float().tolist()
    artifact = dict(schema_version=1, kind='training_residual_channel_rms_v1',
        source=provenance, training_manifest_identity=source.manifest['identity'],
        count=count, batch_size=batch_size, seed=seed, samples=selected,
        sample_identity=_identity(selected), sampler='episode_legal_window_weighted_batches_with_replacement',
        raw_rms=raw.tolist(), global_rms=global_rms, floor=floor, scale=scale,
        floored_channels=int((raw < floor).sum()), latent_coordinates_per_channel=coordinates,
        precision='fp32_mean_codec_float64_accumulation', partition='train')
    artifact['identity'] = _identity(artifact)
    return artifact


def validate_scale(artifact, *, source=None, provenance=None):
    if (artifact.get('schema_version') != 1 or artifact.get('kind') != 'training_residual_channel_rms_v1'
            or artifact.get('partition') != 'train'
            or artifact.get('identity') != _identity({k: v for k, v in artifact.items() if k != 'identity'})
            or artifact.get('sample_identity') != _identity(artifact['samples'])
            or artifact['count'] < 1 or artifact['count'] != len(artifact['samples'])):
        raise ValueError('invalid residual scale artifact identity or sampling record')
    raw, scale = np.asarray(artifact['raw_rms']), np.asarray(artifact['scale'], dtype=np.float32)
    if (raw.shape != (64,) or scale.shape != (64,) or not np.isfinite(raw).all()
            or not np.isfinite(scale).all() or (raw < 0).any() or (scale <= 0).any()):
        raise ValueError('invalid residual channel scales')
    floor = max(1e-4, .05 * float(np.sqrt(np.mean(raw**2))))
    if (not np.isclose(artifact['floor'], floor, rtol=1e-12, atol=1e-15)
            or not np.array_equal(scale, np.maximum(raw, floor).astype(np.float32))):
        raise ValueError('residual scale does not match declared RMS/floor')
    if (not np.isclose(artifact['global_rms'], float(np.sqrt(np.mean(raw**2))), rtol=1e-12)
            or artifact['floored_channels'] != int((raw < floor).sum())
            or artifact['latent_coordinates_per_channel'] < artifact['count']):
        raise ValueError('residual scale statistics differ from declared values')
    if provenance is not None and artifact['source'] != provenance:
        raise ValueError('residual scale center identity differs')
    if source is not None:
        lengths = training_lengths(source)
        if artifact['training_manifest_identity'] != source.manifest['identity']:
            raise ValueError('residual scale training manifest differs')
        for sample in artifact['samples']:
            index = sample['episode']
            if not isinstance(index, int) or not 0 <= index < len(lengths):
                raise ValueError('scale sample episode outside training manifest')
            row = source.manifest['entries'][index]
            if (sample['path'] != row['path'] or sample['demand_group'] != row['demand_group']
                    or not isinstance(sample['start'], int) or not 0 <= sample['start'] < lengths[index]):
                raise ValueError('scale sample identity outside training manifest')
    return torch.tensor(scale)


def install_residual(model, artifact, *, initialization_seed):
    from urban_dreamer.models.residual_dynamics import ResidualDynamics
    scale = validate_scale(artifact)
    identity = {key: artifact['source'][key] for key in
                ('mean_source_sha256', 'codec_sha256', 'manifest_sha256')}
    identity['scale_artifact_sha256'] = json_sha256(artifact)
    if state_digest(model.flow_projector.state_dict()) != identity['codec_sha256']:
        raise ValueError('residual codec identity differs')
    with preserved_torch_rng():
        model.dynamics = ResidualDynamics(model.dynamics, scale,
            identity=identity, initialization_seed=initialization_seed)
    return model


def restore_residual_model(model, payload, parent):
    """Restore embedded mean/codec/scale; no external regression checkpoint access."""
    if (payload.get('schema_version') != SCHEMA or payload.get('variant') != VARIANT
            or payload['protocol']['parent'] != parent):
        raise ValueError('unsupported residual checkpoint or campaign identity')
    protocol, artifact = payload['protocol'], payload['calibration']
    if (protocol.get('kind') != 'frozen_regression_residual_cfm_v1'
            or protocol.get('mode') != 'single_step' or protocol.get('horizon') != 1
            or not protocol.get('frozen_codec') or not protocol.get('frozen_center')):
        raise ValueError('residual checkpoint learning contract differs')
    validate_scale(artifact, provenance=protocol['center'])
    if (json_sha256(artifact) != protocol['scale_artifact_sha256']
            or payload['seed'] != protocol['initialization_seed']
            or protocol['variant']['variant'] != VARIANT
            or not 0 <= payload['update'] <= protocol['planned_updates']
            or payload['exposures'] != payload['update'] * protocol['batch_size']
            or state_digest(payload['model']) != payload['model_sha256']):
        raise ValueError('residual checkpoint model/scale/seed identity differs')
    with preserved_torch_rng():
        apply_variant(model, 'regression', protocol['center']['mean_seed'])
        install_residual(model, artifact, initialization_seed=payload['seed'])
        model.load_state_dict(payload['model'], strict=True)
    if (state_digest(model.flow_projector.state_dict()) != protocol['center']['codec_sha256']
            or state_digest(model.dynamics.mean_dynamics.state_dict()) != protocol['frozen_mean_sha256']
            or not torch.equal(model.dynamics.scale.cpu(), validate_scale(artifact))):
        raise ValueError('restored frozen center, codec or residual scale differs')
    model.flow_projector.requires_grad_(False)
    model.active_vehicle_fraction_readout.requires_grad_(False)
    return model
