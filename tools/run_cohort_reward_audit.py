"""Retrospective fixed-route, fixed-cohort reward diagnostic; never resumes training."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--starts', type=int, nargs='+', default=[10, 30, 60, 80])
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--control', choices=('recorded_routes', 'greedy_policy'), default='recorded_routes')
    parser.add_argument('--real-policy-manifest', type=Path)
    args = parser.parse_args()
    if args.horizon < 1 or args.samples < 1 or min(args.starts) < 0:
        parser.error('positive horizon/samples and nonnegative starts required')
    # Import the executed, immutable source snapshot, not the current worktree.
    sys.path.insert(0, str(args.source_dir.resolve()))
    import numpy as np
    import torch
    from urban_dreamer.learning.episode_archive import read_episode_archive, archive_physics_identity
    from urban_dreamer.learning.action_events import pack_action_events
    from urban_dreamer.learning.fm_inference import configure_imagination_inference
    from urban_dreamer.learning.imagination import advance_imagined_window, clip_road_speed_fractions
    import urban_dreamer.learning.imagination as physical_engine
    from urban_dreamer.learning.global_reward import RewardAccounting
    from urban_dreamer.learning.observations import build_actor_observations, prepare_actor_window
    from urban_dreamer.models.policy import NextHopActor
    from urban_dreamer.traffic.environment import NextHopDecision
    from urban_dreamer.models.world_model import UrbanDreamerWorldModel
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology, NextHopCandidates
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash
    from tools.train_codec_ready import SparseGraphCodec

    loaded_sources = {}
    for name in ('urban_dreamer.models.world_model', 'urban_dreamer.learning.imagination',
                 'urban_dreamer.learning.global_reward', 'urban_dreamer.traffic.topology',
                 'tools.train_codec_ready'):
        path = Path(sys.modules[name].__file__).resolve()
        if not path.is_relative_to(args.source_dir.resolve()):
            raise ValueError(f'diagnostic imported outside frozen source: {name}: {path}')
        loaded_sources[name] = dict(path=str(path), sha256=sha256(path))

    helper_path = Path(__file__).with_name('audit_cohort_reward.py')
    spec = importlib.util.spec_from_file_location('cohort_audit_helpers', helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(.10, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'results.json'
    if output.exists():
        raise FileExistsError(f'refusing to replace an audit: {output}')
    config = load_config(args.config)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if payload['config_fingerprint'] != config.semantic_fingerprint:
        raise ValueError('checkpoint configuration fingerprint mismatch')
    if payload['input_hash'] != combined_input_hash(prepared):
        raise ValueError('checkpoint network/input identity mismatch')
    network = topology.network
    actor = NextHopActor().eval().requires_grad_(False)
    actor.load_state_dict(payload['training_state']['models']['actor'], strict=True)
    real_manifest = None
    if args.control == 'greedy_policy':
        if args.real_policy_manifest is None:
            parser.error('greedy_policy requires a paired real collection manifest')
        real_manifest = json.loads(args.real_policy_manifest.read_text())
        if (real_manifest['checkpoint_sha256'] != sha256(args.checkpoint)
                or real_manifest['policy'] != 'zero_prior_argmax'):
            raise ValueError('real collection is not the same frozen zero-prior greedy policy')
    model = UrbanDreamerWorldModel(adjacency=torch.tensor(network.adjacency, dtype=torch.float32),
        static_features=torch.tensor(network.static_features, dtype=torch.float32),
        low_frequency_rank=config.method.low_frequency_rank,
        prior_filter=getattr(config.method, 'prior_filter', 'hard'),
        prior_filter_strength=getattr(config.method, 'prior_filter_strength', 10.0),
        prior_action_conditioned=getattr(config.method, 'prior_action_conditioned', True),
        prior_filter_input=getattr(config.method, 'prior_filter_input', False), prior_noise_scale=config.method.prior_noise_scale,
        event_protocol=True, codec_speed_positive=config.codec.speed_positive).to(device)
    model.load_state_dict(payload['training_state']['models']['world_model'], strict=True)
    model.flow_projector.__class__ = SparseGraphCodec
    model.flow_projector.register_buffer('sparse_adjacency',
        torch.tensor(network.adjacency, device=device).to_sparse().coalesce(), persistent=False)
    model.eval().requires_grad_(False)
    # Eager inference avoids compiling new kernels beside production jobs.
    diagnostic_config = replace(config, resources=replace(config.resources, world_model_imagination_compile=False))
    profile = configure_imagination_inference(SimpleNamespace(config=diagnostic_config, world_model=model))
    result = dict(status='running', checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=sha256(args.checkpoint), checkpoint_episode=payload['next_unit'],
        source_dir=str(args.source_dir.resolve()), inference_profile=profile, cases=[], rows=[], failures=[],
        accounting_checks=[], loaded_sources=loaded_sources, control=args.control,
        limitations=[
            'Retrospective training archives, not untouched held-out episodes or significance tests.',
            'Future recorded route decisions are diagnostic controls only; times are generated by the model.',
            'No new vehicles in imagined trajectories; real cohort can still experience congestion from later arrivals.',
            'Physical rewards only: no critic bootstrap, uncertainty penalty or entropy bonus.',
            'This does not test alternative-action ranking or actor closed-loop improvement.',
            'Observed-speed control uses real future road snapshots and is not a deployable predictor.',
            'Route records that cannot cover faster imagined motion are rejected, not replaced with guessed routes.',
        ])
    if args.control == 'greedy_policy':
        result['limitations'] = [
            'Fresh real episodes from the exact same frozen greedy actor; small diagnostic sample, no significance claim.',
            'Closed-loop comparison: different predicted observations can change subsequent actions; not fixed-action model error.',
            'Imagined cohort excludes later births; their congestion and policy-observation effects remain in SUMO.',
            'Physical rewards only; excludes critic bootstrap, risk and entropy.',
            'Observed-speed control uses future real road snapshots; it is an oracle diagnostic, not a deployable predictor.',
            'No alternative-action ranking or stochastic training-policy claim.',
        ]
        result['real_policy_manifest'] = str(args.real_policy_manifest.resolve())
    del payload
    started = time.monotonic()
    candidates = NextHopCandidates(network, traffic_model=config.simulation.traffic_model)
    for episode in args.episodes:
        path = args.archive_root / f'episode_{episode:06d}.npz'
        archive = read_episode_archive(path, expected_session_protocol='event')
        if real_manifest is not None:
            entry = next(row for row in real_manifest['episodes'] if row['episode'] == episode)
            if Path(entry['archive_path']).resolve() != path.resolve() or entry['archive_sha256'] != sha256(path):
                raise ValueError('paired real archive identity mismatch')
        archive_physics_identity(archive['metadata'], simulation=config.simulation)
        if archive['metadata']['input_hashes'] != prepared.input_hashes:
            raise ValueError('archive inputs differ from checkpoint')
        if tuple(archive['metadata']['edge_ids']) != tuple(network.edge_ids):
            raise ValueError('archive road ordering differs from checkpoint')
        first, last = archive['states'][0], archive['states'][-1]
        ledger = RewardAccounting(first['started_vehicle_count'], first['cumulative_vehicle_seconds'])
        whole_rewards = [ledger.update(float(exposure), state['started_vehicle_count'])
                         for exposure, state in zip(archive['active_vehicle_seconds'], archive['states'][1:])]
        full_reward = -(last['cumulative_vehicle_seconds'] / max(last['started_vehicle_count'], 1)
                       - first['cumulative_vehicle_seconds'] / max(first['started_vehicle_count'], 1)) / 60.
        accounting = dict(episode=episode,
            full_exposure_error_seconds=ledger.cumulative_vehicle_seconds-last['cumulative_vehicle_seconds'],
            full_telescope_error_minutes=sum(whole_rewards)-full_reward)
        if abs(accounting['full_exposure_error_seconds']) > 1e-6 or abs(accounting['full_telescope_error_minutes']) > 1e-8:
            raise ValueError('real archive cumulative accounting failed')
        result['accounting_checks'].append(accounting)
        for seed_index in args.starts:
            if seed_index + args.horizon >= len(archive['states']):
                raise ValueError('audit interval exceeds observed archive')
            seed = archive['states'][seed_index]
            t0 = float(seed['time'])
            labels = [helper.real_cohort_label(archive, seed_index, t0 + 30 * step)
                      for step in range(1, args.horizon + 1)]
            n0 = int(seed['started_vehicle_count'])
            if n0 <= 0 or not labels[0].vehicle_ids:
                raise ValueError('seed requires a nonempty active cohort and positive started count')
            windows = helper.cohort_window_labels(archive, seed_index,
                [t0 + 30 * step for step in range(1, args.horizon + 1)])
            errors = [abs(label.vehicle_seconds-label.count_integral_seconds) for label in labels]
            telescoping_error = abs(sum(label.reward for label in windows)-labels[-1].reward)
            if max(errors) > 1e-6 or telescoping_error > 1e-8:
                raise ValueError('fixed cohort bookkeeping failed')
            case = dict(episode=episode, seed_index=seed_index, start_seconds=t0,
                archive_sha256=sha256(path), started_count=n0, active_count=len(labels[0].vehicle_ids),
                labels=[asdict(label) for label in labels],
                accounting_integral_error_seconds=max(errors), accounting_telescope_error_minutes=telescoping_error)
            case['no_arrival_baseline'] = [dict(horizon_seconds=(step+1)*30,
                predicted_reward=-(step+1)*30*len(label.vehicle_ids)/(60*n0),
                real_reward=label.reward,
                reward_error=-(step+1)*30*len(label.vehicle_ids)/(60*n0)-label.reward)
                for step,label in enumerate(labels)]
            case['full_real_rewards'] = [-(
                archive['states'][seed_index + step]['cumulative_vehicle_seconds'] /
                max(archive['states'][seed_index + step]['started_vehicle_count'], 1)
                - seed['cumulative_vehicle_seconds'] / max(n0, 1)) / 60.
                for step in range(1, args.horizon + 1)]
            result['cases'].append(case)
            for mode in ('observed_speed', 'persistence_speed', 'world_model'):
                for sample in range(args.samples if mode == 'world_model' else 1):
                    states = copy.deepcopy([v for v in seed['vehicles'] if v['departed'] and not v['arrived']])
                    controller = None
                    if args.control == 'recorded_routes':
                        try:
                            controller = helper.RecordedPhysicalRouteReplay(archive, seed_index, candidates)
                        except helper.RouteReplayUnavailable as error:
                            result['failures'].append(dict(episode=episode, seed_index=seed_index,
                                mode=mode, sample=sample, failed_horizon_seconds=0, error=str(error)))
                            atomic_json(output, result)
                            continue
                    rng_seed = 314159 + episode * 1000 + seed_index * 10 + sample
                    torch.manual_seed(rng_seed)
                    cumulative = 0.
                    event_count = 0
                    with torch.no_grad():
                        latent = model.encode(torch.as_tensor(archive['flow'][seed_index:seed_index+1], device=device))
                        try:
                            for step in range(args.horizon):
                                now = t0 + step * 30.
                                if mode == 'world_model':
                                    decoded = model.decode(latent)[0].cpu().numpy()
                                else:
                                    decoded = archive['flow'][seed_index + (step if mode == 'observed_speed' else 0)]
                                speeds = np.minimum(clip_road_speed_fractions(decoded[:, 1],
                                    config.simulation.traffic_model) * network.speed_limits_mps,
                                    config.simulation.vehicle_max_speed_mps)
                                actor_window = prepare_actor_window(decoded[:, [2, 1]], network.edge_count)
                                def greedy(pending, flow, timestamp):
                                    observation = build_actor_observations(policy_edge_features=actor_window,
                                        vehicle_states=pending, candidates=candidates,
                                        episode_progress=min(timestamp/archive['deadline_seconds'], 1.),
                                        active_vehicle_fraction=sum(not v['arrived'] for v in states)/n0)
                                    selected = actor.forward_logits(torch.as_tensor(observation.features),
                                        torch.as_tensor(observation.mask)).argmax(-1).tolist()
                                    return [NextHopDecision(str(v['vehicle_id']), int(v['current_edge']),
                                        int(observation.edge_indices[j, selected[j]]), float(timestamp))
                                        for j,v in enumerate(pending)]
                                if any(not v['arrived'] for v in states):
                                    original_entry = physical_engine._record_physical_entry
                                    def validate_entry(state, edge, choices):
                                        controller.validate_entry(state, edge)
                                        original_entry(state, edge, choices)
                                    # Instrument only this isolated diagnostic process. Restore on failure.
                                    if controller is not None:
                                        physical_engine._record_physical_entry = validate_entry
                                    try:
                                        window = advance_imagined_window(states, candidates=candidates,
                                            edge_speeds_mps=speeds, frozen_flow=decoded, start_time=now,
                                            interval_seconds=30., total_vehicle_count=archive['metadata']['total_vehicle_count'],
                                            decide=controller if controller is not None else greedy,
                                            observation_step_seconds=1., session_protocol='event',
                                            event_index_start=event_count, traffic_model=config.simulation.traffic_model,
                                            vehicle_max_speed_mps=config.simulation.vehicle_max_speed_mps, stop_when_empty=True)
                                    finally:
                                        physical_engine._record_physical_entry = original_entry
                                    cumulative += window.active_vehicle_seconds
                                    event_count += len(window.action_events)
                                    if mode == 'world_model' and step + 1 < args.horizon and any(not v['arrived'] for v in states):
                                        sparse = pack_action_events([[window.action_events]], window_starts=[[now]])
                                        latent = model.transition(latent,
                                            torch.as_tensor(window.action, dtype=torch.float32, device=device)[None],
                                            action_events=sparse.to(device), steps=config.method.fm_steps,
                                            solver=config.method.fm_solver, picard_iterations=config.method.picard_iterations)
                                predicted = -cumulative / (60 * n0)
                                label = labels[step]
                                result['rows'].append(dict(episode=episode, seed_index=seed_index,
                                    horizon_seconds=(step+1)*30, mode=mode, sample=sample, rng_seed=rng_seed,
                                    real_reward=label.reward, predicted_reward=predicted,
                                    reward_error=predicted-label.reward, real_vehicle_seconds=label.vehicle_seconds,
                                    predicted_vehicle_seconds=cumulative,
                                    mean_seconds_error_per_started=(cumulative-label.vehicle_seconds)/n0,
                                    predicted_arrived_count=sum(v['arrived'] for v in states),
                                    real_arrived_count=label.arrivals))
                        except helper.RouteReplayUnavailable as error:
                            result['failures'].append(dict(episode=episode, seed_index=seed_index,
                                mode=mode, sample=sample, failed_horizon_seconds=(step+1)*30, error=str(error)))
                    result['elapsed_seconds'] = time.monotonic()-started
                    atomic_json(output, result)
                    print(json.dumps(dict(episode=episode, start=t0, mode=mode, sample=sample,
                        rows=len(result['rows']), failures=len(result['failures']), seconds=result['elapsed_seconds'])), flush=True)
    result['aggregates'] = []
    result['paired_coverage'] = []
    modes = ('observed_speed', 'persistence_speed', 'world_model')
    common = {}
    for horizon in sorted({30, 90, args.horizon*30}):
        eligible = {}
        for mode in modes:
            rows = [r for r in result['rows'] if r['mode'] == mode and r['horizon_seconds'] == horizon]
            counts = {}
            for row in rows:
                key = (row['episode'], row['seed_index'])
                counts[key] = counts.get(key, 0) + 1
            eligible[mode] = {key for key, count in counts.items()
                              if count == (args.samples if mode == 'world_model' else 1)}
        common[horizon] = set.intersection(*eligible.values())
        result['paired_coverage'].append(dict(horizon_seconds=horizon,
            planned_cases=len(result['cases']), valid_cases={mode:len(keys) for mode,keys in eligible.items()},
            common_cases=[list(key) for key in sorted(common[horizon])]))
    for mode in ('observed_speed', 'persistence_speed', 'world_model'):
        for horizon in sorted(common):
            rows = [r for r in result['rows'] if r['mode'] == mode and r['horizon_seconds'] == horizon
                    and (r['episode'], r['seed_index']) in common[horizon]]
            if not rows:
                continue
            result['aggregates'].append(dict(mode=mode, horizon_seconds=horizon, samples=len(rows),
                cases=len({(r['episode'], r['seed_index']) for r in rows}),
                reward_mae=float(np.mean([abs(r['reward_error']) for r in rows])),
                reward_bias=float(np.mean([r['reward_error'] for r in rows])),
                real_reward_mean=float(np.mean([r['real_reward'] for r in rows])),
                arrival_count_mae=float(np.mean([abs(r['predicted_arrived_count']-r['real_arrived_count']) for r in rows]))))
    result['status'] = 'completed' if any(common.values()) else 'no_valid_paired_rollouts'
    result['elapsed_seconds'] = time.monotonic()-started
    atomic_json(output, result)
    print(json.dumps(result['aggregates'], indent=2))


if __name__ == '__main__':
    main()
