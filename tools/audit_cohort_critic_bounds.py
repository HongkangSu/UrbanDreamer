"""Read-only critic feasibility check on observed fixed-cohort seed states."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--archive-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--episodes', nargs='+', type=int, default=[0, 1])
    p.add_argument('--starts', nargs='+', type=int, default=[10, 30, 60, 80])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = a.source_dir.resolve(strict=True)
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from urban_dreamer.learning.episode_archive import read_episode_archive, archive_physics_identity
    from urban_dreamer.learning.global_reward import causal_context
    from urban_dreamer.models.policy import CentralizedLatentValueCritic
    from urban_dreamer.models.world_model import UrbanDreamerWorldModel
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash
    from tools.train_codec_ready import SparseGraphCodec

    loaded_sources = {}
    for name, module in list(sys.modules.items()):
        if (name.startswith('urban_dreamer') or name == 'tools.train_codec_ready') and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(source):
                raise ValueError(f'import escaped source snapshot: {name}')
            loaded_sources[name] = dict(path=str(path), sha256=sha256(path))
    spec = importlib.util.spec_from_file_location('cohort_bound_helpers', Path(__file__).with_name('audit_cohort_reward.py'))
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    config = load_config(a.config)
    prepared = validate_prepared_inputs(config)
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if checkpoint['config_fingerprint'] != config.semantic_fingerprint:
        raise ValueError('configuration fingerprint mismatch')
    if checkpoint['input_hash'] != combined_input_hash(prepared):
        raise ValueError('checkpoint inputs mismatch')
    if config.method.disagreement_weight != 0.:
        raise ValueError('physical value bounds require zero disagreement penalty')
    net = load_traffic_topology(prepared).network
    model = UrbanDreamerWorldModel(adjacency=torch.as_tensor(net.adjacency, dtype=torch.float32),
        static_features=torch.as_tensor(net.static_features, dtype=torch.float32),
        low_frequency_rank=config.method.low_frequency_rank,
        prior_filter=getattr(config.method, 'prior_filter', 'hard'),
        prior_filter_strength=getattr(config.method, 'prior_filter_strength', 10.0),
        prior_action_conditioned=getattr(config.method, 'prior_action_conditioned', True),
        prior_filter_input=getattr(config.method, 'prior_filter_input', False),
        prior_usage=getattr(config.method, 'prior_usage', 'source_reference'),
        prior_noise_scale=config.method.prior_noise_scale, event_protocol=True,
        codec_speed_positive=config.codec.speed_positive)
    model.load_state_dict(checkpoint['training_state']['models']['world_model'], strict=True)
    model.flow_projector.__class__ = SparseGraphCodec
    model.flow_projector.register_buffer('sparse_adjacency',
        torch.as_tensor(net.adjacency).to_sparse().coalesce(), persistent=False)
    bounded = getattr(config.method, 'critic_value_mode', 'legacy') == 'bounded_cohort_v1'
    critic = CentralizedLatentValueCritic(bounded=True) if bounded else CentralizedLatentValueCritic()
    critic.load_state_dict(checkpoint['training_state']['models']['latent_critic'], strict=True)
    model.eval().requires_grad_(False)
    critic.eval().requires_grad_(False)
    result = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha256(a.checkpoint),
        checkpoint_episode=checkpoint['next_unit'], source_dir=str(source), loaded_sources=loaded_sources,
        config_sha256=sha256(a.config), units='minutes per seed-started vehicle',
        bound='-remaining_seconds * seed_active_count / (60 * seed_started_count) <= V <= 0',
        limitations=[
            'Bounds assume fixed seed cohort, no new births, zero risk/entropy reward and finite deadline.',
            'Critic is trained under stochastic actor; real comparison uses frozen greedy policy, so residual is not same-policy calibration.',
            'Observed seed latents only; this does not measure imagined latent off-distribution values or establish training-collapse causation.',
            'Real cohort label includes external congestion from future births; physical cost bounds still apply.',
        ], rows=[])
    del checkpoint
    for episode in a.episodes:
        path = a.archive_root / f'episode_{episode:06d}.npz'
        archive = read_episode_archive(path, expected_session_protocol='event')
        archive_physics_identity(archive['metadata'], simulation=config.simulation)
        if archive['metadata']['input_hashes'] != prepared.input_hashes:
            raise ValueError('archive inputs mismatch')
        if tuple(archive['metadata']['edge_ids']) != tuple(net.edge_ids):
            raise ValueError('archive edge ordering mismatch')
        for index in a.starts:
            seed = archive['states'][index]
            deadline = float(archive['deadline_seconds'])
            label = helper.real_cohort_label(archive, index, deadline)
            count, active = label.started_count, len(label.vehicle_ids)
            if count <= 0 or not active:
                raise ValueError('positive started and active cohort required')
            now = float(seed['time'])
            if now > deadline:
                raise ValueError('seed beyond deadline')
            context = causal_context(now, deadline, active, count, float(seed['cumulative_vehicle_seconds']))
            lower = -(deadline - now) * active / (60. * count)
            with torch.no_grad():
                latent = model.encode(torch.as_tensor(archive['flow'][index:index+1]))
                value = float(critic(latent, torch.as_tensor(context)[None],
                    **({'budget': torch.tensor([-lower], dtype=torch.float32)} if bounded else {})).item())
            if not np.isfinite(value):
                raise ValueError('critic produced nonfinite value')
            result['rows'].append(dict(episode=episode, seed_index=index, start_seconds=now,
                deadline_seconds=deadline, active_count=active, started_count=count,
                cumulative_vehicle_seconds=float(seed['cumulative_vehicle_seconds']),
                context=context.tolist(), value=value, lower_bound=lower, upper_bound=0.,
                below_lower_bound=max(0., lower-value), above_upper_bound=max(0., value),
                bound_violated=bool(value < lower-1e-5 or value > 1e-5),
                real_cohort_remaining_reward=label.reward,
                value_minus_real_reward=value-label.reward,
                real_label_end_seconds=label.end_time, archive_sha256=sha256(path)))
    rows = result['rows']
    result['summary'] = dict(cases=len(rows), violated_cases=sum(r['bound_violated'] for r in rows),
        mean_value=float(np.mean([r['value'] for r in rows])),
        mean_lower_bound=float(np.mean([r['lower_bound'] for r in rows])),
        mean_real_cohort_remaining_reward=float(np.mean([r['real_cohort_remaining_reward'] for r in rows])),
        max_below_lower_bound=max(r['below_lower_bound'] for r in rows),
        max_above_upper_bound=max(r['above_upper_bound'] for r in rows))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    temporary.replace(a.output)
    print(json.dumps(result['summary'], indent=2), flush=True)
    for row in rows:
        print(json.dumps({k: row[k] for k in ('episode','seed_index','value','lower_bound',
              'real_cohort_remaining_reward','below_lower_bound')}), flush=True)


if __name__ == '__main__':
    main()
