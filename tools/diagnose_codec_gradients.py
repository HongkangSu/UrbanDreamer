"""Read-only, one-window decomposition of the actual WM objective and gradients."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.models.world_model import (
    _transform_dynamic_targets, decoder_reconstruction_loss,
    decoder_bpr_supervision_loss, episode_t_all_supervision_loss,
)


def _summary(values):
    values = values.detach().double().flatten()
    if not values.numel():
        return dict(count=0, mean=0., min=0., median=0., max=0.)
    return dict(count=values.numel(), mean=float(values.mean()), min=float(values.min()),
                median=float(values.median()), max=float(values.max()))


def gradient_statistics(grads, parameters):
    """Count unused, exactly-zero, and nonfinite derivatives independently."""
    count = sum(parameter.numel() for parameter in parameters)
    unused = sum(p.numel() for g, p in zip(grads, parameters) if g is None)
    finite_square, zero, nonfinite = 0., 0, 0
    for gradient in grads:
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        finite_square += float(gradient[finite].double().square().sum())
        zero += int((gradient == 0).sum())
        nonfinite += int((~finite).sum())
    return dict(l2_norm=math.sqrt(finite_square), elements=count, unused_elements=unused,
                zero_fraction=zero / max(count, 1), nonfinite_fraction=nonfinite / max(count, 1))


def gradient_cosine(left, right):
    dot = ln = rn = 0.
    for a, b in zip(left, right):
        if a is not None:
            ln += float(a.double().square().sum())
        if b is not None:
            rn += float(b.double().square().sum())
        if a is not None and b is not None:
            dot += float((a.double() * b.double()).sum())
    return dict(cosine=dot / math.sqrt(ln * rn) if ln and rn else 0.,
                defined=bool(ln and rn), left_norm=math.sqrt(ln), right_norm=math.sqrt(rn))


def analyze_batch(model, batch, network, method, *, seed=31415):
    """Decompose production H=1 loss without stepping optimizers or storing .grad."""
    device = next(model.parameters()).device
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.train()
        with torch.random.fork_rng(devices=[]), torch.enable_grad():
            torch.random.default_generator.manual_seed(seed)
            flow = batch.flow.to(device)
            size, frames, edges, _ = flow.shape
            z = model.encode(flow.reshape(-1, edges, 8)).reshape(size, frames, edges, 64)
            raw = {}
            volume_hook = model.flow_projector.decoder_volume.register_forward_hook(
                lambda module, args, output: raw.__setitem__('volume', output))
            speed_hook = model.flow_projector.decoder_speed.register_forward_hook(
                lambda module, args, output: raw.__setitem__('speed', output))
            try:
                decoded = model.decode(z)
            finally:
                volume_hook.remove()
                speed_hook.remove()
            targets = flow.clone()
            targets[:, :-1, :, 1] = (flow[:, :-1, :, 1] + flow[:, 1:, :, 1]) / 2.
            valid = torch.ones_like(decoded)
            valid[:, -1, :, 1] = 0.
            prediction_transformed, target_transformed = _transform_dynamic_targets(decoded, targets[..., :4])
            square = (prediction_transformed - target_transformed).square()
            weights = [1., 4., 2., 1.]
            components = {name: .2 * weight * (square[..., index] * valid[..., index]).sum() / valid.sum()
                for index, (name, weight) in enumerate(zip(('recon_occupancy', 'recon_speed', 'recon_volume', 'recon_halt'), weights))}
            reconstruction, _ = decoder_reconstruction_loss(decoded_dyn=decoded, flow_targets=targets,
                speed_target_mask=valid[..., 1].bool())
            components['reconstruction'] = reconstruction
            bpr, _ = decoder_bpr_supervision_loss(decoded_dyn=decoded, flow_targets=flow,
                lengths=torch.as_tensor(network.lengths_m, device=device),
                speed_limits=torch.as_tensor(network.speed_limits_mps, device=device),
                capacities=torch.as_tensor(network.capacities, device=device))
            components['bpr'] = bpr
            dynamics = model.dynamics
            action = batch.action[:, :1].to(device)
            tau = torch.rand(size, 1, device=device)
            condition = dynamics.encode_condition(action, action_events=batch.action_events.for_step(0).to(device))
            current, target = z[:, :1], z[:, 1:2].detach()
            start, increment, condition = dynamics.prior_start(current, condition)
            interpolation = (1 - tau[..., None, None]) * start + tau[..., None, None] * target
            velocity_target = (target - start).detach()
            velocity = dynamics(action, packed_latents=interpolation, flow_time=tau, condition=condition).latent_velocity
            components['fm_velocity'] = (velocity.float() - velocity_target.float()).square().mean()
            prior_target = dynamics.graph_prior.project(target - current).detach()
            components['fm_prior'] = method.prior_loss_weight * (increment.float() - prior_target.float()).square().mean()
            completed = (batch.total_travel_minutes > 0) & batch.episode_complete
            if bool(completed.any()):
                components['t_all'], _ = episode_t_all_supervision_loss(flow_projector=model.flow_projector,
                    z_tokens=z[completed.to(device)], reward_targets=-batch.total_travel_minutes[completed].to(device))
            else:
                components['t_all'] = z.sum() * 0.
            components['fm_total'] = components['fm_velocity'] + components['fm_prior']
            components['world_total'] = (components['fm_total'] + reconstruction + bpr + components['t_all'])
            named = [(name, parameter) for name, parameter in model.named_parameters()
                     if not name.startswith('active_vehicle_fraction_readout.') and parameter.requires_grad]
            groups = {name: [] for name in ('encoder', 'decoder_shared', 'decoder_volume', 'decoder_speed', 'decoder_other', 't_all_head', 'dynamics')}
            for index, (name, _) in enumerate(named):
                if name.startswith('dynamics.'):
                    group = 'dynamics'
                elif name.startswith(('flow_projector.input_proj.', 'flow_projector.gnn_layers.')):
                    group = 'encoder'
                else:
                    group = name.split('.')[1]
                groups[group].append(index)
            parameters = [parameter for _, parameter in named]
            reports, encoder_grads = {}, {}
            for name, loss in components.items():
                grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
                reports[name] = dict(loss=float(loss.detach()), global_gradient=gradient_statistics(grads, parameters),
                    blocks={group: gradient_statistics([grads[i] for i in indices], [parameters[i] for i in indices])
                            for group, indices in groups.items()})
                encoder_grads[name] = [None if grads[i] is None else grads[i].detach().clone() for i in groups['encoder']]
            raw_volume_gradient, = torch.autograd.grad(components['recon_volume'], raw['volume'], retain_graph=True)
            raw_speed_gradient, = torch.autograd.grad(components['recon_speed'], raw['speed'], retain_graph=True)
            occupied = flow[..., 2] > 0
            raw_volume = raw['volume'][..., 0].detach()
            derivative = torch.sigmoid(10 * raw_volume)
            capacities = torch.as_tensor(network.capacities, device=device).clamp_min(1.)
            physical_count = flow[..., 2] * capacities
            raw_probe = dict(raw_volume=_summary(raw_volume), decoded_volume=_summary(decoded[..., 2]),
                raw_speed=_summary(raw['speed']),
                negative_raw_speed_fraction=float((raw['speed'] < 0).float().mean()),
                raw_speed_0_to_0_01_fraction=float(((raw['speed'] >= 0) & (raw['speed'] < .01)).float().mean()),
                raw_speed_0_01_to_0_05_fraction=float(((raw['speed'] >= .01) & (raw['speed'] < .05)).float().mean()),
                softplus_derivative=_summary(derivative), occupied_softplus_derivative=_summary(derivative[occupied]),
                empty_softplus_derivative=_summary(derivative[~occupied]),
                softplus_derivative_below_1e_3_fraction=float((derivative < 1e-3).float().mean()),
                occupied_softplus_derivative_below_1e_3_fraction=float((derivative[occupied] < 1e-3).float().mean()) if bool(occupied.any()) else 0.,
                raw_volume_negative_fraction=float((raw_volume < 0).float().mean()),
                volume_raw_loss_gradient=_summary(raw_volume_gradient.abs()),
                speed_raw_loss_gradient=_summary(raw_speed_gradient.abs()),
                occupied_volume_raw_loss_gradient=_summary(raw_volume_gradient[..., 0][occupied].abs()),
                empty_volume_raw_loss_gradient=_summary(raw_volume_gradient[..., 0][~occupied].abs()))
            norm = reports['world_total']['global_gradient']['l2_norm']
            denominator = reports['recon_volume']['blocks']['encoder']['l2_norm']
            return dict(seed=seed, depth=1, windows=batch.time.tolist(), frames=frames, edges=edges,
                components=reports, encoder_angles={name: gradient_cosine(encoder_grads[left], encoder_grads[right])
                    for name, left, right in (('fm_vs_reconstruction', 'fm_total', 'reconstruction'),
                                             ('fm_vs_volume', 'fm_total', 'recon_volume'),
                                             ('speed_vs_volume', 'recon_speed', 'recon_volume'))},
                encoder_speed_to_volume_gradient_ratio=reports['recon_speed']['blocks']['encoder']['l2_norm'] / max(denominator, 1e-30),
                hypothetical_clip10_factor=min(1., 10. / max(norm, 1e-30)), raw_decoder_probe=raw_probe,
                labels=dict(occupied_road_fraction=float(occupied.float().mean()),
                    speed_target_below_0_1_fraction=float((targets[:, :-1, :, 1] < .1).float().mean()),
                    normalized_volume=_summary(flow[..., 2]), occupied_normalized_volume=_summary(flow[..., 2][occupied]),
                    capacities=_summary(capacities), physical_vehicle_counts=_summary(physical_count),
                    occupied_physical_vehicle_counts=_summary(physical_count[occupied]),
                    physical_counts_integer_mae=float((physical_count - physical_count.round()).abs().mean()),
                    reconstruction_volume_target_input_max_abs=float((targets[..., 2] - flow[..., 2]).abs().max()),
                    reconstruction_channel_sum_error=float((sum(components[name] for name in
                        ('recon_occupancy', 'recon_speed', 'recon_volume', 'recon_halt')) - reconstruction).detach().abs()),
                    speed_forward_mean_target=True, last_speed_label_excluded=True, t_all_completed_sequences=int(completed.sum())))
    finally:
        for module, mode in modes:
            module.training = mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-dir', type=Path, default=ROOT / 'outputs/ingolstadt_pretrain_pilot_20261002')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ingolstadt_codec_diagnosis_20261002')
    args = parser.parse_args()
    torch.set_num_threads(1)
    from urban_dreamer.learning.training import MainlineTrainer
    from urban_dreamer.learning.episode_archive import read_episode_archive, archive_to_replay_episode
    from urban_dreamer.learning.replay import ReplayBuffer
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.config import load_config, DataConfig
    config = load_config(args.pilot_dir / 'run_config.toml')
    config = replace(config, run=replace(config.run, device='cpu', output_dir=args.output_dir / 'isolated_empty_source'), data=DataConfig())
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    split = json.loads((args.pilot_dir / 'split.json').read_text())
    chosen, archive, starts = None, None, None
    incident = topology.network.edge_to_index[config.simulation.incident_edge]
    for row in split['train']:
        candidate = read_episode_archive(Path(row['path']), expected_session_protocol='event')
        at_incident = np.flatnonzero((candidate['time'][:-1] >= config.simulation.incident_start_seconds)
            & (candidate['time'][:-1] < config.simulation.incident_end_seconds)
            & (candidate['flow'][:-1, incident, 2] > 0))
        if at_incident.size:
            chosen, archive = row, candidate
            starts = dict(occupied_incident=int(at_incident[len(at_incident) // 2]),
                          late=max(0, len(candidate['action']) - 2))
            break
    if chosen is None:
        raise ValueError('no selected training archive has occupied incident observations')
    replay = ReplayBuffer(sequence_length=1, max_episodes=1)
    episode = archive_to_replay_episode(archive, target_deadline_seconds=trainer._episode_end_time())
    replay.append_episode(episode['transitions'], total_travel_minutes=episode['total_travel_minutes'],
                          total_vehicle_count=episode['total_vehicle_count'])
    batches = {label: replay._batch_from_picks([(0, start)]) for label, start in starts.items()}
    label_audit = {}
    capacities = np.maximum(topology.network.capacities, 1.)
    for label, start in starts.items():
        label_audit[label] = []
        for index in (start, start + 1):
            counts = np.zeros(topology.network.edge_count)
            for vehicle in archive['states'][index]['vehicles']:
                if vehicle['departed'] and not vehicle['arrived'] and not vehicle.get('in_junction', False):
                    counts[vehicle['current_edge']] += 1
            archived_counts = archive['flow'][index, :, 2] * capacities
            label_audit[label].append(dict(time=float(archive['time'][index]),
                snapshot_external_vehicle_count=float(counts.sum()), archived_road_vehicle_count=float(archived_counts.sum()),
                maximum_edge_count_difference=float(np.abs(counts - archived_counts).max()),
                mismatched_edges=int(np.count_nonzero(np.abs(counts - archived_counts) > 1e-6))))
    result = dict(source_path=chosen['path'], source_completed=archive['completed'], starts=starts,
        physical_label_audit=label_audit,
        checkpoint=str(args.pilot_dir / 'checkpoints/latest.pt'), scope='read-only initial versus pilot320 gradients on identical real samples; FM depth1',
        weights=dict(reconstruction=.2, inverse_speed_channel=4., log1p_volume_channel=2., bpr=.1, prior=config.method.prior_loss_weight))
    for phase in ('initial', 'pilot320'):
        if phase == 'pilot320':
            payload = torch.load(args.pilot_dir / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
            trainer.world_model.load_state_dict(payload['training_state']['models']['world_model'], strict=True)
            del payload
        result[phase] = {}
        for label, batch in batches.items():
            print(f'Analyzing {phase} {label} at {float(batch.time[0]):.0f}s', flush=True)
            result[phase][label] = analyze_batch(trainer.world_model, batch, trainer.network, config.method)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'gradients.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(args.output_dir / 'gradients.json', flush=True)


if __name__ == '__main__':
    main()
