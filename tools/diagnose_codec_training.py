"""Fit a fixed real traffic batch to separate codec optimization from FM/data size.

All arms clone exactly the same pretrained codec. Two alternatives change only
the volume reconstruction term to vehicle units, with ordinary averaging or
equal empty/occupied group weight. Speed labels, other channels, BPR, AdamW
settings and data are identical.
No dynamics module is instantiated or optimized. This is an in-sample capacity
and optimization diagnostic, not a held-out traffic-performance experiment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.offline_data import archive_content_hash
from urban_dreamer.models.world_model import (
    BPR_SUPERVISION_WEIGHT, DECODER_RECON_WEIGHT, FlowProjector,
    decoder_bpr_supervision_loss, decoder_reconstruction_loss,
)
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.config import load_config


def balanced_group_mean(values: torch.Tensor, occupied: torch.Tensor) -> torch.Tensor:
    """Give existing empty and occupied groups equal weight, avoiding empty means."""
    groups = [values[mask].mean() for mask in (occupied, ~occupied) if bool(mask.any())]
    if not groups:
        raise ValueError('codec batch has no road targets')
    return torch.stack(groups).mean()


def volume_objective(decoded: torch.Tensor, flow: torch.Tensor,
                     capacities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized = (torch.log1p(decoded[..., 2]) - torch.log1p(flow[..., 2].clamp_min(0))).square()
    predicted_count = decoded[..., 2] * capacities
    observed_count = flow[..., 2].clamp_min(0) * capacities
    vehicles = (torch.log1p(predicted_count) - torch.log1p(observed_count)).square()
    # The original all-valid reconstruction has four channels and volume
    # weight2. Preserve its coefficient while changing units/group balance.
    coefficient = DECODER_RECON_WEIGHT * 2 / 4
    return (coefficient * normalized.mean(), coefficient * vehicles.mean(),
            coefficient * balanced_group_mean(vehicles, observed_count > 0))


def codec_loss(projector: FlowProjector, flow: torch.Tensor, next_flow: torch.Tensor,
               *, capacities: torch.Tensor, lengths: torch.Tensor,
               speed_limits: torch.Tensor, arm: str) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    if arm not in {'original', 'vehicle_units_all', 'vehicle_units_balanced'}:
        raise ValueError('codec arm must be original, vehicle_units_all or vehicle_units_balanced')
    decoded = projector.decode(projector(flow))
    target = flow.clone()
    target[..., 1] = (flow[..., 1] + next_flow[..., 1]) / 2
    reconstruction, _ = decoder_reconstruction_loss(decoded_dyn=decoded, flow_targets=target)
    bpr, _ = decoder_bpr_supervision_loss(decoded_dyn=decoded, flow_targets=flow,
        capacities=capacities, lengths=lengths, speed_limits=speed_limits)
    original_volume, vehicle_volume_all, vehicle_volume_balanced = volume_objective(decoded, flow, capacities)
    loss = reconstruction + bpr
    if arm != 'original':
        replacement = vehicle_volume_all if arm == 'vehicle_units_all' else vehicle_volume_balanced
        loss = loss - original_volume + replacement
    return loss, dict(reconstruction=reconstruction.detach(), bpr=bpr.detach(),
                     original_volume=original_volume.detach(), vehicle_volume_all=vehicle_volume_all.detach(),
                     vehicle_volume_balanced=vehicle_volume_balanced.detach()), decoded


def select_frames(flow: np.ndarray, capacities: np.ndarray) -> tuple[list[int], list[str]]:
    if flow.ndim != 3 or flow.shape[-1] != 8 or len(flow) < 4:
        raise ValueError('codec selection needs at least three frames plus forward speed endpoint')
    volume = flow[:-1, :, 2] * capacities
    totals = volume.sum(-1)
    empty = int(np.argmin(totals))
    occupied = int(np.argmax(totals))
    if totals[empty] != 0 or totals[occupied] <= 0:
        raise ValueError('episode must contain both empty and occupied observed traffic')
    jam_score = (volume * (flow[:-1, :, 1] < .3)).sum(-1)
    candidates = [int(index) for index in np.argsort(-jam_score, kind='stable')
                  if int(index) not in {empty, occupied} and jam_score[index] > 0]
    if not candidates:
        raise ValueError('episode has no distinct occupied frame with observed slow/jammed traffic')
    return [empty, occupied, candidates[0]], ['empty', 'occupied', 'jam']


def reconstruction_metrics(projector: FlowProjector, flow: torch.Tensor, next_flow: torch.Tensor,
                           capacities: torch.Tensor, speed_limits: torch.Tensor) -> dict:
    with torch.no_grad():
        z = projector(flow)
        hidden = projector.decoder_shared(z)
        raw_volume = projector.decoder_volume(hidden).squeeze(-1)
        derivative = torch.sigmoid(10 * raw_volume)
        decoded = projector.decode(z)
        truth = flow[..., 2] * capacities
        count = decoded[..., 2] * capacities
        error = (count - truth).abs()
        occupied = truth > 0
        speed_target = (flow[..., 1] + next_flow[..., 1]) / 2
        speed_error = (decoded[..., 1] - speed_target).abs() * speed_limits
        def mean(value, mask=None):
            selected = value if mask is None else value[mask]
            return float(selected.mean()) if selected.numel() else None
        return dict(vehicle_mae=mean(error), occupied_vehicle_mae=mean(error, occupied),
            empty_vehicle_mae=mean(error, ~occupied),
            normalized_volume_mae=mean((decoded[..., 2] - flow[..., 2]).abs()),
            total_volume_bias_vehicles=mean((count - truth).sum(-1)),
            total_volume_mae_vehicles=mean((count - truth).sum(-1).abs()),
            observed_total_vehicles=mean(truth.sum(-1)), predicted_total_vehicles=mean(count.sum(-1)),
            speed_mae_mps=mean(speed_error), occupied_speed_mae_mps=mean(speed_error, occupied),
            negative_raw_speed_fraction=mean((decoded[..., 1] < 0).float()),
            occupied_roads=int(occupied.sum()), empty_roads=int((~occupied).sum()), frames=int(flow.shape[0]),
            volume_softplus_derivative_mean=mean(derivative),
            occupied_volume_softplus_derivative_mean=mean(derivative, occupied),
            occupied_volume_softplus_derivative_below_1e4_fraction=mean((derivative < 1e-4).float(), occupied))


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode() + b'\0')
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_metrics(projector: FlowProjector) -> dict:
    def norm(parameters):
        gradients = [p.grad.detach().double().square().sum() for p in parameters if p.grad is not None]
        return float(torch.stack(gradients).sum().sqrt()) if gradients else 0.
    return dict(encoder_gradient_norm=norm([*projector.input_proj.parameters(), *projector.gnn_layers.parameters()]),
                decoder_gradient_norm=norm([*projector.decoder_shared.parameters(), *projector.decoder_other.parameters(),
                                            *projector.decoder_speed.parameters(), *projector.decoder_volume.parameters()]),
                volume_head_gradient_norm=norm(projector.decoder_volume.parameters()))


def train_arm(initial: dict, adjacency: torch.Tensor, flow: torch.Tensor, next_flow: torch.Tensor,
              *, capacities: torch.Tensor, lengths: torch.Tensor, speed_limits: torch.Tensor,
              arm: str, steps: int, learning_rate: float = 1e-4, log=None) -> tuple[dict, dict]:
    projector = FlowProjector(adjacency=adjacency).to(flow.device)
    projector.load_state_dict(initial, strict=True)
    projector.train()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=1e-2)
    checkpoints = []
    milestone = {0, min(100, steps), steps}
    if flow.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(flow.device)
    started = time.perf_counter()
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, components, _ = codec_loss(projector, flow, next_flow, capacities=capacities,
                                        lengths=lengths, speed_limits=speed_limits, arm=arm)
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(not bool(torch.isfinite(p.grad).all())
                                              for p in projector.parameters() if p.grad is not None):
            raise FloatingPointError(f'nonfinite codec loss/gradient at {arm} step{step}')
        if step in milestone:
            if flow.device.type == 'cuda':
                torch.cuda.synchronize(flow.device)
            elapsed = time.perf_counter() - started
            row = dict(step=step, arm=arm, training_loss=float(loss.detach()),
                       **{key: float(value) for key, value in components.items()},
                       **reconstruction_metrics(projector, flow, next_flow, capacities, speed_limits),
                       **gradient_metrics(projector), elapsed_seconds=elapsed,
                       updates_per_second=step / max(elapsed, 1e-9),
                       frames_per_second=step * flow.shape[0] / max(elapsed, 1e-9))
            checkpoints.append(row)
            if log:
                log(row)
        if step == steps:
            break
        torch.nn.utils.clip_grad_norm_(projector.parameters(), 10.)
        optimizer.step()
    final = {name: tensor.detach().cpu().clone() for name, tensor in projector.state_dict().items()}
    untouched_head = all(torch.equal(initial[name], final[name]) for name in initial if name.startswith('t_all_head.'))
    return dict(arm=arm, steps=steps, initial_codec_hash=state_hash(initial), final_codec_hash=state_hash(final),
                learning_rate=learning_rate, weight_decay=1e-2, gradient_clip=10.,
                checkpoints=checkpoints, unused_total_travel_head_unchanged=untouched_head,
                max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(flow.device) if flow.device.type == 'cuda' else 0,
                max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(flow.device) if flow.device.type == 'cuda' else 0), final


def choose_device(requested: str, fraction: float):
    if requested == 'cpu':
        return torch.device('cpu'), dict(requested=requested, selected='cpu')
    if requested != 'cuda:1':
        raise ValueError('codec diagnosis allows CPU or GPU1 only')
    if not torch.cuda.is_available():
        return torch.device('cpu'), dict(requested=requested, selected='cpu', reason='CUDA unavailable')
    free, total = torch.cuda.mem_get_info(1)
    if free < 2 * 1024 ** 3:
        return torch.device('cpu'), dict(requested=requested, selected='cpu', reason='GPU1 has less than2GiB free',
                                        free_bytes=free, total_bytes=total)
    torch.cuda.set_per_process_memory_fraction(fraction, 1)
    return torch.device('cuda:1'), dict(requested=requested, selected='cuda:1', free_bytes=free,
                                       total_bytes=total, memory_fraction=fraction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, default=ROOT / 'outputs/ingolstadt_pretrain_pilot_20261002')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ingolstadt_codec_diagnosis_20261002')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--device', choices=('cpu', 'cuda:1'), default='cuda:1')
    parser.add_argument('--cuda-memory-fraction', type=float, default=.07)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 500 or not 0 < args.cuda_memory_fraction <= .07:
        parser.error('steps must be1..500 and CUDA memory fraction at most.07')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / 'codec_result.json'
    if result_path.exists():
        raise ValueError(f'refusing to overwrite completed codec diagnosis: {result_path}')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = load_config(args.pilot_dir / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    network = load_traffic_topology(prepared).network
    manifest = json.loads((args.pilot_dir / 'offline_manifest.json').read_text())
    if manifest['contract']['input_hashes'] != prepared.input_hashes:
        raise ValueError('pilot manifest input hashes differ from current prepared network')
    selected, current, future = [], [], []
    for entry in manifest['entries'][:2]:
        path = Path(entry['path'])
        if archive_content_hash(path) != entry['content_hash']:
            raise ValueError(f'pilot training archive changed: {path}')
        with np.load(path, allow_pickle=False) as archive:
            flow = archive['flow']
        indexes, labels = select_frames(flow, network.capacities)
        current.extend(flow[indexes])
        future.extend(flow[np.asarray(indexes) + 1])
        selected.append(dict(path=str(path), content_hash=entry['content_hash'],
                             frames=[dict(index=index, seconds=30 * index, condition=label)
                                     for index, label in zip(indexes, labels)]))
    if len(selected) != 2:
        raise ValueError('codec experiment requires two frozen pilot training episodes')
    raw_flow, raw_future = np.stack(current), np.stack(future)
    device, device_info = choose_device(args.device, args.cuda_memory_fraction)
    flow = torch.tensor(raw_flow, dtype=torch.float32, device=device)
    next_flow = torch.tensor(raw_future, dtype=torch.float32, device=device)
    physical = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in (
        ('capacities', network.capacities), ('lengths', network.lengths_m), ('speed_limits', network.speed_limits_mps))}
    adjacency = torch.tensor(network.adjacency, dtype=torch.float32, device=device)
    checkpoint_path = args.pilot_dir / 'checkpoints/world_model_pretrained.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['optimizer_updates'] != 320:
        raise ValueError('controlled diagnosis expects the pilot320-update starting state')
    prefix = 'flow_projector.'
    initial = {name.removeprefix(prefix): tensor.clone() for name, tensor in checkpoint['world_model'].items()
               if name.startswith(prefix)}
    unchanged = {name: tensor for name, tensor in checkpoint['world_model'].items() if not name.startswith(prefix)}
    unchanged_hash = state_hash(unchanged)
    report = dict(scope='fixed_real_batch_codec_fit_no_FM', starting_checkpoint=str(checkpoint_path),
        starting_updates=checkpoint['optimizer_updates'], fixed_data_hash=hashlib.sha256(raw_flow.tobytes() + raw_future.tobytes()).hexdigest(),
        episodes=selected, device=device_info, threads=1, edge_count=network.edge_count, frames=len(raw_flow),
        speed_labels='forward observed endpoint mean; every selected frame has a following boundary',
        original_objective='exact decoder_reconstruction_loss plus existing decoder_bpr_supervision_loss',
        alternate_objective='replace normalized-volume log1p MSE with log1p vehicle-count MSE: all-road average and equal empty/occupied group average; other terms identical',
        decoder_reconstruction_weight=DECODER_RECON_WEIGHT, bpr_weight=BPR_SUPERVISION_WEIGHT,
        non_codec_state_hash=unchanged_hash, arms=[])
    with (args.output_dir / 'codec_training.log').open('w') as logfile:
        def log(row):
            line = json.dumps(row, sort_keys=True)
            logfile.write(line + '\n'); logfile.flush()
            print(line, flush=True)
        for arm in ('original', 'vehicle_units_all', 'vehicle_units_balanced'):
            result, trained = train_arm(initial, adjacency, flow, next_flow, arm=arm, steps=args.steps,
                                         log=log, **physical)
            full = deepcopy(checkpoint)
            full['world_model'].update({prefix + name: tensor for name, tensor in trained.items()})
            full['scope'] = 'diagnostic fixed-real-batch codec-only fitting; FM and other parameters unchanged'
            full['codec_diagnostic'] = dict(arm=arm, steps=args.steps,
                starting_joint_updates=checkpoint['optimizer_updates'],
                fixed_data_hash=report['fixed_data_hash'], initial_codec_hash=result['initial_codec_hash'])
            result['non_codec_parameters_unchanged'] = state_hash({name: tensor for name, tensor in full['world_model'].items()
                                                                  if not name.startswith(prefix)}) == unchanged_hash
            torch.save(full, args.output_dir / f'codec_checkpoint_{arm}.pt')
            report['arms'].append(result)
            result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if len({arm['initial_codec_hash'] for arm in report['arms']}) != 1:
        raise AssertionError('controlled codec arms did not share their starting weights')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
