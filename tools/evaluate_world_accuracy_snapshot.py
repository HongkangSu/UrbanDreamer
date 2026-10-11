"""Read-only online world-model snapshot checks, with fixed held-out comparisons."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.migrate_pretrained_meso import install_sparse_codec
from tools.world_model_evaluation import evaluate_world_model, _fixed_evaluation_state, _repeat_events
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch
from urban_dreamer.learning.fm_inference import configure_imagination_inference
from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import load_checkpoint
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def road_scores(prediction, target, speed_target, capacities, limits):
    """Score physical quantities; an empty road mask is unavailable, not zero."""
    count = target[..., 2] * capacities
    speed = speed_target * limits
    occupied = count > 0
    slow = (count >= 5 - 1e-5) & (speed <= 2)
    count_error = (prediction[..., 2] - target[..., 2]) * capacities
    speed_error = (prediction[..., 1] - speed_target) * limits
    def mean(values, mask=None):
        selected = values if mask is None else values[mask]
        if not bool(torch.isfinite(selected).all()):
            raise ValueError('nonfinite diagnostic predictions')
        return float(selected.double().mean()) if selected.numel() else None
    return dict(occupied_roads=int(occupied.sum()), slow_roads=int(slow.sum()),
        speed_mae_mps=mean(speed_error.abs()),
        occupied_speed_mae_mps=mean(speed_error.abs(), occupied),
        slow_speed_mae_mps=mean(speed_error.abs(), slow),
        slow_speed_bias_mps=mean(speed_error, slow),
        occupied_count_mae_vehicles=mean(count_error.abs(), occupied),
        total_count_true=float(count.sum()),
        total_count_prediction=float((prediction[..., 2] * capacities).sum()),
        total_count_bias_vehicles=float(count_error.sum()),
        total_count_absolute_error_vehicles=float(count_error.sum().abs()))


def fixed_cohort_rollout(trainer, batch, *, seed, samples=3, horizon=8):
    """Roll sample zero forward; extra samples diagnose each identical next-state condition.

    Recorded future states are used solely for metric labels and codec oracles.
    Recorded controls are filtered to sequence-start cohort C0; this is a fixed-
    control diagnostic, not an on-policy or real-time actor trajectory.
    """
    model, device, method = trainer.world_model, trainer.device, trainer.config.method
    capacities = torch.as_tensor(trainer.network.capacities, device=device).float().clamp_min(1.)
    limits = torch.as_tensor(trainer.network.speed_limits_mps, device=device).float()
    scoped = prepare_cohort_batch(batch, capacities=capacities)
    target = scoped.transition_flow.to(device)
    if target.shape[1] < horizon + 2:
        raise ValueError('rollout needs one following boundary for the decoder speed label')
    rows = []
    with _fixed_evaluation_state(trainer, seed):
        current = model.encode(target[:, 0])
        for step in range(horizon):
            generated = model.transition(current.repeat(samples, 1, 1),
                scoped.transition_action[:, step].to(device).repeat(samples, 1),
                action_events=_repeat_events(scoped.transition_action_events.for_step(step).to(device), samples),
                steps=method.fm_steps, solver=method.fm_solver, picard_iterations=method.picard_iterations)
            decoded = model.decode(generated).reshape(samples, len(target), target.shape[2], -1)
            current = generated[:len(target)]
            if step + 1 not in (1, 3, 8):
                continue
            truth = target[:, step + 1]
            speed_target = .5 * (truth[..., 1] + target[:, step + 2, :, 1])
            oracle = model.decode(model.encode(truth))
            boundary = scoped.boundary_vehicles[0][step + 1]
            members = {str(v['vehicle_id']) for v in scoped.boundary_vehicles[0][0]
                       if v['departed'] and not v['arrived']}
            active = sum(str(v['vehicle_id']) in members and v['departed'] and not v['arrived'] for v in boundary)
            rows.append(dict(horizon_seconds=30 * (step + 1), true_cohort_active=int(active),
                true_cohort_terminal=active == 0,
                sample0=road_scores(decoded[0], truth, speed_target, capacities, limits),
                mean3=road_scores(decoded.mean(0), truth, speed_target, capacities, limits),
                endpoint_codec=road_scores(oracle, truth, speed_target, capacities, limits),
                persistence=road_scores(target[:, 0], truth, speed_target, capacities, limits),
                latent_variance_mean=float(generated.reshape(samples, len(target), target.shape[2], -1).var(0, unbiased=False).mean())))
    return rows


def atomic_json(path, result):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--online-dir', type=Path, required=True)
    parser.add_argument('--pretrain-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'online_snapshot'
    (snapshot / 'checkpoints').mkdir(parents=True, exist_ok=True)
    checkpoint_path = snapshot / 'checkpoints/latest.pt'
    if not checkpoint_path.exists():
        # Atomic production replacement makes an opened source file stable.
        shutil.copyfile(args.online_dir / 'checkpoints/latest.pt', checkpoint_path)
        shutil.copyfile(args.online_dir / 'run_config.toml', snapshot / 'run_config.toml')
    torch.set_num_threads(2)
    if args.device.startswith('cuda'):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.set_per_process_memory_fraction(.10, torch.device(args.device))
    config = load_config(snapshot / 'run_config.toml')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    checkpoint = load_checkpoint(output_dir=snapshot,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(prepared))
    trainer = MainlineTrainer(config=replace(config, run=replace(config.run, device=args.device)),
        prepared_inputs=prepared, topology=topology)
    started = time.monotonic()
    baseline = json.loads((args.pretrain_dir / 'heldout_prediction_evaluation.json').read_text())
    result = dict(status='running', benchmark_only=True, production_modified=False,
        source_checkpoint=str((args.online_dir / 'checkpoints/latest.pt').resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        online_completed=checkpoint.next_unit, snapshot_metrics=checkpoint.training_state['metrics'],
        seed=31415, stochastic_samples=3, solver=config.method.fm_solver, steps=config.method.fm_steps,
        selected=baseline['selected'], pretrain_checkpoint_sha256=baseline['checkpoint_sha256'],
        pretrain_single_window_metrics=baseline['metrics'], recursive_cases=[])
    atomic_json(output / 'results.json', result)
    try:
        install_sparse_codec(trainer)
        trainer.load_state_dict(checkpoint.training_state)
        trainer.validate_codec_ready()
        trainer.world_model.eval()
        result['inference_profile'] = configure_imagination_inference(trainer)
        dataset = CompactPretrainDataset(config, prepared, topology, 4,
            cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,
            split='validation')
        if dataset.manifest['identity'] != baseline['validation_manifest_identity']:
            raise ValueError('held-out comparison dataset identity changed')
        def batches():
            for row in baseline['selected']:
                entry = dataset.manifest['entries'][row['index']]
                if entry['path'] != row['path']:
                    raise ValueError('held-out selected archive index changed')
                yield dataset.batch_from_starts(row['index'], row['starts'], materialization_horizon=4, fm_only=True)
        result['current_single_window_metrics'] = evaluate_world_model(trainer, batches(), seed=31415, n_samples=3, horizon=3)
        result['single_window_seconds'] = time.monotonic() - started
        atomic_json(output / 'results.json', result)
        print(json.dumps(dict(stage='single_window_complete', seconds=result['single_window_seconds'], online_completed=checkpoint.next_unit)), flush=True)
        rolling = CompactPretrainDataset(config, prepared, topology, 9,
            cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,
            split='validation')
        cases = []
        for method in ('online_ac', 'dta', 'random'):
            selected = next(row for row in baseline['selected'] if row['method'] == method)
            cases.append(dict(method=method, path=selected['path'], start=40))
        benchmark = json.loads((args.pretrain_dir.parent / 'picard_benchmark/results.json').read_text())
        congested = max(benchmark['windows'], key=lambda row: row['slow_queue_vehicles'])
        cases.append(dict(method='congested', path=congested['archive'], start=congested['window']))
        current_weights = {key: value.detach().cpu().clone() for key, value in trainer.world_model.state_dict().items()}
        pretrained = torch.load(args.pretrain_dir / 'checkpoints/latest.pt', map_location='cpu', weights_only=True)
        for case_index, case in enumerate(cases):
            index = next(i for i, entry in enumerate(rolling.manifest['entries']) if entry['path'] == case['path'])
            entry = rolling.manifest['entries'][index]
            start = min(case['start'], entry['windows'] - 9)
            if start < 0:
                raise ValueError('selected rollout archive too short')
            batch = rolling.batch_from_starts(index, [start], materialization_horizon=9, fm_only=False)
            record = dict(**case, actual_start=start, population=int(batch.total_vehicle_count[0]))
            for name, weights in [('current_online', current_weights), ('pretrained', pretrained['training_state']['models']['world_model'])]:
                trainer.world_model.load_state_dict(weights, strict=True)
                record[name] = fixed_cohort_rollout(trainer, batch, seed=31415 + case_index)
            result['recursive_cases'].append(record)
            atomic_json(output / 'results.json', result)
            print(json.dumps(dict(stage='recursive_case_complete', case=case_index, seconds=time.monotonic()-started)), flush=True)
        result.update(status='complete', elapsed_seconds=time.monotonic()-started,
            notes=['Single-window scores average three samples, unlike actual sample-zero trajectory propagation.',
                   'Recursive diagnostic uses fixed initial C0 volume and cohort-filtered recorded actions; it is not an actor rollout.',
                   'Recursive speed retains actual surrounding traffic background; omitted future births affect volume only.',
                   'Ground truth is used for scoring and oracle codec reconstruction, never as a recursive prediction input.',
                   'An empty slow or occupied road mask yields null, not measured zero error.',
                   'Continuing beyond a true empty cohort is a stress test; actual imagination can stop when vehicles finish.'])
        atomic_json(output / 'results.json', result)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
