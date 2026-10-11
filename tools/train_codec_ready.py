"""Fit a codec on the frozen pilot split; stop only on a held-out reconstruction gate."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.codec_training import codec_decoded_metrics, codec_metrics, codec_train_step, make_codec_optimizer
from urban_dreamer.learning.episode_archive import archive_physics_identity
from urban_dreamer.learning.offline_data import _source_config
from urban_dreamer.learning.pretrain_cache import file_stamp, member_signature
from urban_dreamer.models.world_model import FlowProjector
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


GATES = dict(codec_count_mae_occupied=.5, codec_count_mae_all=.1, codec_count_mae_empty=.05,
             codec_total_relative_error=.1, codec_speed_mae_mps=1.5)


class SparseGraphCodec(FlowProjector):
    """Same codec parameters and graph operation, skipping zero adjacency entries."""
    def __init__(self, *, adjacency, positive_speed=True):
        super().__init__(adjacency=adjacency, positive_speed=positive_speed)
        self.register_buffer('sparse_adjacency', adjacency.float().to_sparse().coalesce(), persistent=False)

    def forward(self, flow):
        if flow.shape[-1] != 8:
            raise ValueError('codec flow must end in8 features')
        hidden = self.input_proj(flow)
        for layer in self.gnn_layers:
            messages = layer.message(hidden)
            # Sparse.mm accepts2D: flatten every batch/time/channel column, retaining the road axis.
            packed = messages.movedim(-2, 0)
            message = torch.sparse.mm(self.sparse_adjacency.to(dtype=messages.dtype),
                                      packed.reshape(self.n_edges, -1))
            message = message.reshape(packed.shape).movedim(0, -2)
            update = F.gelu(layer.self_proj(hidden) + message)
            hidden = layer.norm(hidden + update)
        return hidden


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def member_hash(path, name):
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(name + '.npy') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_sources(split, prepared, config, network):
    """Freeze only already selected paths; no scan of a growing collection root."""
    train_groups = {row['demand_group'] for row in split['train']}
    validation_groups = {row['demand_group'] for row in split['validation']}
    if train_groups & validation_groups:
        raise ValueError('training/validation realized demand leakage')
    train_paths = {row['path'] for row in split['train']}
    if train_paths & {row['path'] for row in split['validation']}:
        raise ValueError('training/validation source path leakage')
    fields = ('vehicle_max_speed_mps', 'incident_edge', 'incident_start_seconds', 'incident_end_seconds',
              'incident_speed_mps', 'sumo_extra_args', 'demand_duration_seconds')
    physical = {key: config.resolved_snapshot()['simulation'][key] for key in fields}
    physics = physics_identity(config.simulation)
    entries = {}
    for partition in ('train', 'validation'):
        entries[partition] = []
        for row in split[partition]:
            path = Path(row['path'])
            if not row.get('metadata_compatible', False) or 'train' not in path.parts:
                raise ValueError(f'{path}: source is not a previously compatible training archive')
            with np.load(path, allow_pickle=False) as archive:
                summary = json.loads(archive['summary_json'].tobytes().decode())
                flow = archive['flow']
            meta = summary['metadata']
            if (meta['city'] != config.inputs.network_name or meta['input_hashes'] != prepared.input_hashes
                    or meta['edge_ids'] != list(network.edge_ids) or meta['session_protocol'] != 'event'):
                raise ValueError(f'{path}: source network or protocol differs')
            if meta.get('requested_step_seconds', meta.get('step_seconds')) != 1.:
                raise ValueError(f'{path}: observation step differs')
            actual = _source_config(path, meta).get('simulation', {})
            if archive_physics_identity(meta, simulation=actual) != physics:
                raise ValueError(f'{path}: traffic physics configuration differs')
            if any(actual.get(key) != value for key, value in physical.items()):
                raise ValueError(f'{path}: physical/demand configuration differs')
            if (flow.ndim != 3 or flow.shape[1:] != (network.edge_count, 8) or len(flow) < 2
                    or not np.isfinite(flow).all() or (flow[..., 2] < 0).any()):
                raise ValueError(f'{path}: invalid raw boundary flows')
            entries[partition].append(dict(path=str(path), frames=len(flow) - 1,
                flow_member_sha256=member_hash(path, 'flow'),
                metadata_member_sha256=member_hash(path, 'summary_json'),
                demand_group=row['demand_group'], method=row['method'], root=row['root'],
                population=meta['total_vehicle_count']))
    contract = dict(entries=entries, physical=physical, input_hashes=prepared.input_hashes,
                    edge_ids=list(network.edge_ids), scope='Exact frozen pilot membership; raw same-frame codec targets')
    if physics['traffic_model'] == 'meso':
        contract['physics_identity'] = physics
    contract['identity'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    return contract


class FlowCache:
    def __init__(self, limit=2):
        self.limit = limit
        self.cache = OrderedDict()

    def get(self, entry):
        path = entry['path']
        stamp = file_stamp(Path(path))
        # ZIP central-directory CRCs also catch same-size rapid writes when the
        # filesystem timestamp granularity is coarser than the write interval.
        with zipfile.ZipFile(path) as archive:
            signature = member_signature(archive.infolist())
        stamp = (stamp, signature)
        cached = self.cache.get(path)
        # Stat every access; decompress and verify members on first access/change.
        if (cached is None or cached['stamp'] != stamp) and (member_hash(path, 'flow') != entry['flow_member_sha256']
                or member_hash(path, 'summary_json') != entry['metadata_member_sha256']):
            raise ValueError(f'frozen codec source content changed: {path}')
        if path not in self.cache:
            while len(self.cache) >= self.limit:
                self.cache.popitem(last=False)
            with np.load(path, allow_pickle=False) as archive:
                self.cache[path] = dict(flow=archive['flow'].copy(), stamp=stamp)
        self.cache.move_to_end(path)
        self.cache[path]['stamp'] = stamp
        return self.cache[path]['flow']


class FrameSampler:
    """One shuffled visit per valid frame per epoch, including partial final batches."""
    def __init__(self, entries, batch_size, seed):
        self.entries, self.batch_size = entries, batch_size
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.order = []
        self.cursor = 0
        self.frame_order = []
        self.frame_cursor = 0

    def next(self):
        if not self.order or self.cursor == len(self.order):
            if self.order:
                self.epoch += 1
            self.order = self.rng.permutation(len(self.entries)).tolist()
            self.cursor = 0
        index = self.order[self.cursor]
        if not self.frame_order:
            self.frame_order = self.rng.permutation(self.entries[index]['frames']).tolist()
            self.frame_cursor = 0
        frames = self.frame_order[self.frame_cursor:self.frame_cursor + self.batch_size]
        self.frame_cursor += len(frames)
        if self.frame_cursor == len(self.frame_order):
            self.cursor += 1
            self.frame_order = []
            self.frame_cursor = 0
        return index, np.asarray(frames, dtype=np.int64)

    def state_dict(self):
        return dict(epoch=self.epoch, order=self.order, cursor=self.cursor,
            frame_order=self.frame_order, frame_cursor=self.frame_cursor, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        for key in ('epoch', 'order', 'cursor', 'frame_order', 'frame_cursor'):
            setattr(self, key, state[key])
        if self.order and sorted(self.order) != list(range(len(self.entries))):
            raise ValueError('codec sampler episode order invalid')
        if not 0 <= self.cursor <= len(self.order):
            raise ValueError('codec sampler cursor invalid')
        if self.frame_order:
            count = self.entries[self.order[self.cursor]]['frames']
            if sorted(self.frame_order) != list(range(count)) or not 0 <= self.frame_cursor < count:
                raise ValueError('codec sampler frame order invalid')
        self.rng.bit_generator.state = state['rng']


class MetricAccumulator:
    """Weight masks and physical frames exactly instead of averaging batch means."""
    def __init__(self):
        self.sums, self.weights = {}, {}

    def add(self, metrics, frames, edges):
        occupied = float(metrics['codec_count_occupied_roads'])
        empty = float(metrics['codec_count_empty_roads'])
        for name, raw in metrics.items():
            value = float(raw)
            if not math.isfinite(value):
                raise FloatingPointError(f'nonfinite validation metric {name}')
            if name in ('codec_count_occupied_roads', 'codec_count_empty_roads', 'codec_empty_predicted_total', 'codec_persistent_slow_road_frames', 'codec_slow_roads'):
                self.sums[name] = self.sums.get(name, 0.) + value
                self.weights[name] = 1.
                continue
            if name.startswith('codec_persistent_slow_'):
                weight = float(metrics['codec_persistent_slow_road_frames'])
                self.sums[name] = self.sums.get(name, 0.) + value * weight
                self.weights[name] = self.weights.get(name, 0.) + weight
                continue
            weight = (float(metrics['codec_slow_roads']) if name == 'codec_speed_mae_slow_mps'
                      else occupied if '_occupied' in name
                      else empty if '_empty' in name or name.startswith('codec_empty_') else frames * edges)
            if name.startswith('codec_total_') or name in ('codec_predicted_count_total', 'codec_target_count_total'):
                weight = frames
            if name == 'codec_count_rmse_all':
                value *= value
            self.sums[name] = self.sums.get(name, 0.) + value * weight
            self.weights[name] = self.weights.get(name, 0.) + weight

    def result(self):
        result = {name: self.sums[name] / weight if weight else None for name, weight in self.weights.items()}
        if result.get('codec_count_rmse_all') is not None:
            result['codec_count_rmse_all'] = math.sqrt(result['codec_count_rmse_all'])
        return result


def gate_result(metrics):
    checks = {key: metrics.get(key) is not None and math.isfinite(metrics[key]) and metrics[key] <= value
              for key, value in GATES.items()}
    ratios = [metrics[key] / limit for key, limit in GATES.items()
              if metrics.get(key) is not None and math.isfinite(metrics[key])]
    score = max(ratios) + .01 * sum(ratios) if len(ratios) == len(GATES) else None
    return dict(passed=all(checks.values()), checks=checks, thresholds=GATES, score=score)


def joint_gate_result(raw, hybrid):
    raw_gate, hybrid_gate = gate_result(raw), gate_result(hybrid)
    scores = (raw_gate['score'], hybrid_gate['score'])
    return dict(passed=raw_gate['passed'] and hybrid_gate['passed'], raw=raw_gate, hybrid=hybrid_gate,
                score=max(scores) if all(value is not None for value in scores) else None)


def configured_speed_gate(metrics, codec):
    """Apply the same occupied/slow admission limits as the mainline trainer."""
    thresholds = dict(codec_speed_mae_occupied_mps=codec.occupied_speed_mae_mps_max,
                      codec_speed_mae_slow_mps=codec.slow_speed_mae_mps_max)
    checks = {key: metrics.get(key) is not None and math.isfinite(metrics[key])
              and 0 <= metrics[key] <= limit for key, limit in thresholds.items()}
    checks['slow_validation_min_roads'] = metrics.get('codec_slow_roads', 0) >= codec.slow_validation_min_roads
    return dict(passed=all(checks.values()), checks=checks, thresholds=thresholds,
                slow_validation_min_roads=codec.slow_validation_min_roads)


def cohort_codec_anchors(windows, sequence_length=16):
    length = min(windows, sequence_length)
    if length < 1:
        raise ValueError('cohort codec source requires at least one complete physical interval')
    starts = np.unique(np.rint(np.linspace(0, windows - length, min(4, windows - length + 1))).astype(int))
    positions = np.unique(np.rint(np.linspace(0, length - 1, min(4, length))).astype(int))
    return starts, positions, length


class HybridFrameCache:
    """Small derived same-frame C0 pools; only two episodes resident at once."""
    def __init__(self, directory, capacities, allowed_paths, *, sequence_length=16, limit=2):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacities, self.sequence_length, self.limit = capacities, sequence_length, limit
        self.allowed_paths = set(allowed_paths)
        self.cache = OrderedDict()

    def get(self, entry):
        from urban_dreamer.learning.pretrain_cache import (
            compact_arrays, compact_cohort_targets, read_hash_source, file_stamp,
        )
        from urban_dreamer.learning.offline_data import archive_content_hash
        path = entry['path']
        if path not in self.allowed_paths:
            raise ValueError('cohort pool source is outside its frozen split')
        if (member_hash(path, 'flow') != entry['flow_member_sha256']
                or member_hash(path, 'summary_json') != entry['metadata_member_sha256']):
            raise ValueError(f'frozen codec source content changed: {path}')
        key = hashlib.sha256(path.encode()).hexdigest()
        pool_path = self.directory / f'{key}.npz'
        identity_path = self.directory / f'{key}.json'
        if not pool_path.is_file():
            data, identity = read_hash_source(Path(path))
            arrays = compact_arrays(data)
            arrays['metadata'] = json.loads(arrays.pop('metadata_json').tobytes().decode())
            starts, positions, length = cohort_codec_anchors(len(arrays['flow']) - 1, self.sequence_length)
            scoped, *_ = compact_cohort_targets(arrays, starts.tolist(), length, self.capacities)
            original = np.stack([arrays['flow'][start:start + length + 1] for start in starts])
            retained = [0, 1, 3, 4, 5, 6, 7]
            if not np.array_equal(scoped[..., retained], original[..., retained]):
                raise AssertionError('cohort augmentation changed full-environment background fields')
            current = scoped[:, positions].reshape(-1, original.shape[-2], 8)
            following = scoped[:, positions + 1].reshape(current.shape)
            temporary = pool_path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, flow=current, next_flow=following)
            temporary.replace(pool_path)
            write_json(identity_path, dict(path=path, source_content_hash=identity['content_hash'],
                source_stamp=identity['source_stamp'], sequence_length=self.sequence_length,
                starts=starts.tolist(), positions=positions.tolist(), length=length,
                source_flow_sha256=entry['flow_member_sha256'],
                source_metadata_sha256=entry['metadata_member_sha256'],
                flow_sha256=member_hash(pool_path, 'flow'), next_flow_sha256=member_hash(pool_path, 'next_flow')))
            del data, arrays, scoped, original
        identity = json.loads(identity_path.read_text())
        if (identity['sequence_length'] != self.sequence_length
                or identity['source_flow_sha256'] != entry['flow_member_sha256']
                or identity['source_metadata_sha256'] != entry['metadata_member_sha256']):
            raise ValueError('derived cohort pool semantic/source identity changed')
        stamp = file_stamp(Path(path))
        known = self.cache.get(path, {}).get('source_stamp', identity['source_stamp'])
        if stamp != known and archive_content_hash(Path(path)) != identity['source_content_hash']:
            raise ValueError(f'frozen cohort membership content changed: {path}')
        if path not in self.cache:
            while len(self.cache) >= self.limit:
                self.cache.popitem(last=False)
            if (member_hash(pool_path, 'flow') != identity['flow_sha256']
                    or member_hash(pool_path, 'next_flow') != identity['next_flow_sha256']):
                raise ValueError('derived cohort pool content changed')
            with np.load(pool_path, allow_pickle=False) as archive:
                self.cache[path] = dict(flow=archive['flow'].copy(), next_flow=archive['next_flow'].copy())
        self.cache.move_to_end(path)
        self.cache[path]['source_stamp'] = stamp
        return self.cache[path]


def validation_metrics(model, flow, future, network, codec=None):
    """Report persistent occupied queues separately without changing acceptance gates.

    A persistent slow queue has >=5 vehicles and <=2 m/s at both observed
    endpoints; this does not assert its unobserved within-window speed.
    """
    decoded = model.decode(model(flow))
    metrics = codec_decoded_metrics(decoded, flow, future, network, codec)
    capacity = torch.as_tensor(network.capacities, device=flow.device)
    speed_limit = torch.as_tensor(network.speed_limits_mps, device=flow.device)
    speed, next_speed = flow[..., 1] * speed_limit, future[..., 1] * speed_limit
    mask = ((flow[..., 2] * capacity >= 5) & (future[..., 2] * capacity >= 5)
            & (speed <= 2) & (next_speed <= 2))
    error = decoded[..., 1] * speed_limit - .5 * (speed + next_speed)
    count = mask.sum()
    metrics.update(codec_persistent_slow_road_frames=count,
        codec_persistent_slow_speed_mae_mps=(error.abs() * mask).sum() / count.clamp_min(1),
        codec_persistent_slow_speed_bias_mps=(error * mask).sum() / count.clamp_min(1))
    return metrics


@torch.no_grad()
def evaluate_hybrid_pool(model, entries, cache, network, device, batch_size, heartbeat=None, codec=None):
    accumulator = MetricAccumulator()
    frames = 0
    model.eval()
    for entry in entries:
        if heartbeat is not None:
            heartbeat()
        pool = cache.get(entry)
        for start in range(0, len(pool['flow']), batch_size):
            stop = min(start + batch_size, len(pool['flow']))
            flow = torch.as_tensor(pool['flow'][start:stop], device=device)
            future = torch.as_tensor(pool['next_flow'][start:stop], device=device)
            accumulator.add(validation_metrics(model, flow, future, network, codec), stop - start, network.edge_count)
            frames += stop - start
    model.train()
    result = accumulator.result()
    result.update(validation_episodes=len(entries), validation_frames=frames, cohort_sequence_length=16,
        scope='Four seed-cohort starts, four valid current-frame anchors; actual following observations; full background retained')
    return result


@torch.no_grad()
def evaluate(model, entries, cache, network, device, batch_size, codec=None):
    accumulator = MetricAccumulator()
    model.eval()
    frames = 0
    for entry in entries:
        raw = cache.get(entry)
        for start in range(0, len(raw) - 1, batch_size):
            end = min(start + batch_size, len(raw) - 1)
            flow = torch.as_tensor(raw[start:end], device=device)
            future = torch.as_tensor(raw[start + 1:end + 1], device=device)
            metrics = validation_metrics(model, flow, future, network, codec)
            accumulator.add(metrics, end - start, network.edge_count)
            frames += end - start
    result = accumulator.result()
    result.update(validation_episodes=len(entries), validation_frames=frames)
    model.train()
    return result


@torch.no_grad()
def evaluate_hybrid(model, entries, network, device):
    from urban_dreamer.learning.episode_archive import read_episode_archive, archive_to_replay_episode
    from urban_dreamer.learning.replay import ReplayBuffer
    from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
    accumulator = MetricAccumulator()
    frames = 0
    for entry in entries:
        data = read_episode_archive(Path(entry['path']), expected_session_protocol='event')
        episode = archive_to_replay_episode(data)
        replay = ReplayBuffer(4, max_episodes=1)
        replay.append_episode(episode['transitions'], total_travel_minutes=episode['total_travel_minutes'],
            total_vehicle_count=episode['total_vehicle_count'], arrival_times=episode['arrival_times'])
        last = max(0, len(episode['transitions']) - 4)
        starts = np.unique(np.rint(np.linspace(0, last, min(4, last + 1))).astype(int))
        for start in starts:
            batch = replay._batch_from_picks([(0, int(start))])
            scoped = prepare_cohort_batch(batch, capacities=network.capacities)
            if not torch.equal(scoped.transition_flow[..., [0, 1, 3, 4, 5, 6, 7]], batch.flow[..., [0, 1, 3, 4, 5, 6, 7]]):
                raise AssertionError('cohort codec check changed full-environment background fields')
            flow = scoped.transition_flow[:, :-1].to(device)
            future = scoped.transition_flow[:, 1:].to(device)
            metrics = codec_metrics(model, flow, future, network)
            count = flow.shape[0] * flow.shape[1]
            accumulator.add(metrics, count, network.edge_count)
            frames += count
        del replay, episode, data
    result = accumulator.result()
    result.update(validation_episodes=len(entries), validation_frames=frames,
                  scope='Four fixed seed-cohort sequences per held-out episode; background fields retained')
    return result


def choose_device(requested, memory_fraction=.05):
    if requested == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu'), dict(requested=requested, selected='cpu')
    if requested not in ('cuda:0', 'cuda:1'):
        raise ValueError('codec runner permits only CPU, GPU0 or GPU1')
    if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
        raise ValueError('gpu memory fraction must be finite and in (0, 1]')
    index = torch.device(requested).index
    free, total = torch.cuda.mem_get_info(index)
    if free < .5 * 1024 ** 3:
        return torch.device('cpu'), dict(requested=requested, selected='cpu', free_bytes=free,
                                        reason=f'{requested} below .5GiB free')
    torch.cuda.set_per_process_memory_fraction(memory_fraction, index)
    return torch.device(requested), dict(requested=requested, selected=requested, free_bytes=free,
                                      memory_fraction=memory_fraction, maximum_allocator_bytes=memory_fraction * total)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, default=ROOT / 'outputs/ingolstadt_pretrain_pilot_20261002')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ingolstadt_codec_ready_20261002')
    parser.add_argument('--device', choices=('cpu', 'cuda:0', 'cuda:1'), default='cuda:1')
    parser.add_argument('--max-updates', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--evaluate-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--expected-train-episodes', type=int, default=80)
    parser.add_argument('--expected-validation-episodes', type=int, default=20)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.05)
    parser.add_argument('--graph-multiply', choices=('dense', 'sparse'), default='sparse')
    parser.add_argument('--cohort-augmentation-fraction', type=float, default=.25)
    parser.add_argument('--configured-codec', action='store_true',
                        help='Use configured speed loss weights and require occupied/slow quality gates')
    args = parser.parse_args(argv)
    if not 1 <= args.max_updates <= 20000 or args.batch_size < 1 or args.evaluate_every < 1:
        parser.error('bounded codec runner needs max-updates1..20000 and positive batch/evaluation sizes')
    if args.expected_train_episodes < 1 or args.expected_validation_episodes < 1:
        parser.error('expected episode counts must be positive')
    if not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('gpu memory fraction must be finite and in (0, 1]')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('codec ready training requires a fresh output directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    if not 0 < args.cohort_augmentation_fraction <= 1:
        raise ValueError('cohort augmentation must be positive and at most the raw frame batch size')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = load_config(args.pilot_dir / 'run_config.toml')
    training_codec = config.codec if args.configured_codec else None
    prepared = validate_prepared_inputs(config)
    network = load_traffic_topology(prepared).network
    split = json.loads((args.pilot_dir / 'split.json').read_text())
    if (len(split['train']), len(split['validation'])) != (args.expected_train_episodes, args.expected_validation_episodes):
        raise ValueError('frozen pilot split does not match the expected train/validation counts')
    manifest = freeze_sources(split, prepared, config, network)
    identity = manifest['identity']
    manifest_path = output / 'codec_manifest.json'
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('codec frozen manifest changed on resume')
    else:
        write_json(manifest_path, manifest)
        write_json(output / 'split.json', split)
        write_json(output / 'resolved_config.json', config.resolved_snapshot())
    device, device_info = choose_device(args.device, args.gpu_memory_fraction)
    codec_type = SparseGraphCodec if args.graph_multiply == 'sparse' else FlowProjector
    model = codec_type(adjacency=torch.as_tensor(network.adjacency), positive_speed=True).to(device)
    optimizer = (make_codec_optimizer(model, lr=config.codec.learning_rate)
                 if args.configured_codec else make_codec_optimizer(model))
    sampler = FrameSampler(manifest['entries']['train'], args.batch_size, args.seed)
    cache = FlowCache(2)
    train_hybrid = HybridFrameCache(output / 'train_cohort_pools_seq16', network.capacities,
        [entry['path'] for entry in manifest['entries']['train']])
    validation_hybrid = HybridFrameCache(output / 'validation_cohort_pools_seq16', network.capacities,
        [entry['path'] for entry in manifest['entries']['validation']])
    started, update, best_score = time.monotonic(), 0, None
    latest = output / 'codec_latest.pt'
    if args.resume:
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        if checkpoint['manifest_identity'] != identity or checkpoint['batch_size'] != args.batch_size:
            raise ValueError('codec checkpoint source or sampling contract differs')
        if checkpoint.get('configured_codec', False) != args.configured_codec:
            raise ValueError('codec configured loss/gate contract differs on resume')
        model.load_state_dict(checkpoint['codec'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        sampler.load_state_dict(checkpoint['sampler'])
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(), device)
        update = checkpoint['update']
        # A raw-only score cannot select the best checkpoint for the new joint gate.
        best_score = checkpoint.get('joint_best_score')
        if checkpoint.get('cohort_sequence_length', 16) != 16:
            raise ValueError('codec cohort sampling semantics changed on resume')
        for name in ('result', 'progress'):
            path = output / f'{name}.json'
            preserved = output / f'{name}_raw_seq4_4500.json'
            if path.exists() and not preserved.exists() and json.loads(path.read_text()).get('status') == 'raw_gate_passed':
                shutil.copy2(path, preserved)
        preserved = output / 'codec_raw_seq4_4500.pt'
        if update == 4500 and not preserved.exists():
            shutil.copy2(latest, preserved)

    def log(record):
        record = dict(record)
        record.update(elapsed_seconds=time.monotonic() - started, heartbeat_unix_seconds=time.time())
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
        return record

    def checkpoint(path):
        payload = dict(schema_version=1, codec=model.state_dict(), optimizer=optimizer.state_dict(),
            sampler=sampler.state_dict(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
            manifest_identity=identity, update=update, best_score=best_score, batch_size=args.batch_size,
            joint_best_score=best_score, cohort_sequence_length=16,
            cohort_augmentation_fraction=args.cohort_augmentation_fraction,
            codec_speed_positive=True, speed_head_initial_free_flow_ratio=1.,
            architecture='FlowProjector unchanged-width64, positive speed, normalized density output',
            graph_multiply=args.graph_multiply,
            configured_codec=args.configured_codec,
            device=device_info, scope='codec only; no actor, value critic or latent dynamics instantiated')
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    validation, gate, hybrid, hybrid_gate = None, None, None, None
    configured_gate = configured_hybrid_gate = None

    def ready():
        return (gate['passed'] and hybrid_gate['passed'] and
                (not args.configured_codec or (configured_gate['passed'] and configured_hybrid_gate['passed'])))

    def heartbeat():
        payload = dict(status='hybrid_training_active', update=update, pid=os.getpid(),
            heartbeat_unix_seconds=time.time(), manifest_identity=identity, cohort_sequence_length=16,
            augmentation_scope='Only frozen training episodes; raw frames retained; no validation labels trained',
            validation=validation, gate=gate, hybrid_validation=hybrid, hybrid_gate=hybrid_gate,
            configured_gates_required=args.configured_codec,
            configured_gate=configured_gate, configured_hybrid_gate=configured_hybrid_gate,
            resume_from_update=4500, maximum_updates=args.max_updates)
        write_json(output / 'result.json', payload)
        write_json(output / 'progress.json', payload)

    def validate():
        nonlocal best_score, configured_gate, configured_hybrid_gate
        metrics = evaluate(model, manifest['entries']['validation'], cache, network, device, args.batch_size, training_codec)
        hybrid_metrics = evaluate_hybrid_pool(model, manifest['entries']['validation'], validation_hybrid,
            network, device, args.batch_size, heartbeat=heartbeat, codec=training_codec)
        if args.configured_codec:
            configured_gate = configured_speed_gate(metrics, config.codec)
            configured_hybrid_gate = configured_speed_gate(hybrid_metrics, config.codec)
        joint = joint_gate_result(metrics, hybrid_metrics)
        record = log(dict(phase='validation', status='hybrid_training_active', update=update,
            epoch=sampler.epoch, **metrics, gate=joint['raw'], hybrid_validation=hybrid_metrics,
            hybrid_gate=joint['hybrid'], both_gates_passed=joint['passed'],
            configured_gates_required=args.configured_codec,
            configured_gate=configured_gate, configured_hybrid_gate=configured_hybrid_gate))
        if joint['score'] is not None and (best_score is None or joint['score'] < best_score):
            best_score = joint['score']
            checkpoint(output / 'codec_best.pt')
        checkpoint(latest)
        record['heartbeat_unix_seconds'] = time.time()
        write_json(output / 'progress.json', record)
        return metrics, joint['raw'], hybrid_metrics, joint['hybrid']

    log(dict(phase='start', update=update, device=device_info, train_episodes=len(split['train']), validation_episodes=len(split['validation']),
        train_frames=sum(entry['frames'] for entry in manifest['entries']['train']),
        validation_frames=sum(entry['frames'] for entry in manifest['entries']['validation']),
        manifest_identity=identity, codec_coefficients=dict(count=1., foreground=.2, mass=.1, speed=1., other=.05),
        maximum_updates=args.max_updates, batch_size=args.batch_size, graph_multiply=args.graph_multiply,
        scope='independent codec only; resumed weights' if args.resume else 'fresh independent codec only'))
    heartbeat()
    validation, gate, hybrid, hybrid_gate = validate()
    heartbeat()
    while update < args.max_updates and not ready():
        episode_index, frame_indices = sampler.next()
        raw = cache.get(manifest['entries']['train'][episode_index])
        flow = torch.as_tensor(raw[frame_indices], device=device)
        future = torch.as_tensor(raw[frame_indices + 1], device=device)
        pool = train_hybrid.get(manifest['entries']['train'][episode_index])
        count = max(1, round(len(frame_indices) * args.cohort_augmentation_fraction))
        selected = np.random.randint(len(pool['flow']), size=count)
        flow = torch.cat([flow, torch.as_tensor(pool['flow'][selected], device=device)])
        future = torch.cat([future, torch.as_tensor(pool['next_flow'][selected], device=device)])
        metrics = codec_train_step(model, flow, future, network, None, optimizer, training_codec)
        update += 1
        if update % 100 == 0:
            log(dict(phase='update', update=update, epoch=sampler.epoch, cache_episodes=len(cache.cache),
                **{name: float(value) for name, value in metrics.items()}))
            heartbeat()
        if update % args.evaluate_every == 0 or update == args.max_updates:
            validation, gate, hybrid, hybrid_gate = validate()
            heartbeat()
    result = dict(status='codec_ready' if ready() else 'budget_exhausted', update=update,
        validation=validation, gate=gate, hybrid_validation=hybrid,
        hybrid_gate=hybrid_gate, both_gates_passed=gate['passed'] and hybrid_gate['passed'],
        configured_gates_required=args.configured_codec,
        configured_gate=configured_gate, configured_hybrid_gate=configured_hybrid_gate,
        heartbeat_unix_seconds=time.time(), pid=os.getpid(), cohort_sequence_length=16,
        manifest_identity=identity, best_score=best_score, device=device_info,
        maximum_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        note='Held-out reconstruction gate; no traffic performance claim and no FM training started.')
    checkpoint(latest)
    write_json(output / 'result.json', result)
    write_json(output / 'progress.json', result)
    log(dict(phase='complete', **result))


if __name__ == '__main__':
    main()
