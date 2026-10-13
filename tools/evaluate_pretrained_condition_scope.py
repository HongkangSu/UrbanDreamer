"""Paired H1 control-scope diagnostic using the completed pretrained checkpoint.

This changes only recorded controls, never targets, starting state, noise, model,
or masks. Full factual controls are an offline dynamics diagnostic and include
within-window events; they are not information available to an online actor.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools.component_experiment_data import atomic_json, dataset, load_frozen, sha256
from tools.component_prediction_metrics import evaluate_prediction_samples
from tools.evaluate_component_world_models import endpoint_target
from tools.evaluate_pretrained_distribution import CHECKPOINT_SHA256, supplementary_scores, trajectory_bank
from tools.evaluate_component_world_models import stable_seed
from tools.summarize_component_predictions import demand_bootstrap
from urban_dreamer.learning.cohort_targets import prepare_cohort_batch

FIELDS = ('count_mae_all', 'count_mae_occupied', 'count_rmse_occupied',
          'speed_mae_occupied', 'speed_rmse_occupied', 'count_crps_u_occupied',
          'speed_crps_u_occupied', 'count_coverage90_occupied',
          'speed_coverage90_occupied', 'count_width90_occupied',
          'speed_width90_occupied', 'energy_score_u', 'total_count_absolute_error')
REFERENCE_ATOL = 2e-4
REFERENCE_RTOL = 0.
DETERMINISTIC_REFERENCE_ATOL = .05
SELF_REPEAT_ATOL = 1e-7


def verify_source(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    for relative, expected in manifest.get('files', manifest).items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise ValueError('frozen runtime changed: ' + relative)
    return sha256(manifest_path)


def reference_drift(actual, expected, *, atol=REFERENCE_ATOL):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError('reference bank shape/finiteness mismatch')
    return dict(max_absolute_error=float(np.max(np.abs(actual-expected))),
                count_max_absolute_error=float(np.max(np.abs(actual[...,0]-expected[...,0]))),
                speed_max_absolute_error=float(np.max(np.abs(actual[...,1]-expected[...,1]))),
                bit_identical=bool(np.array_equal(actual, expected)),
                count_mean_absolute_error=float(np.abs(actual[...,0]-expected[...,0]).mean()),
                speed_mean_absolute_error=float(np.abs(actual[...,1]-expected[...,1]).mean()),
                within_prospective_tolerance=bool(np.all(np.abs(actual-expected) <= atol)),
                atol=atol, rtol=REFERENCE_RTOL)


def assert_reference(actual, expected):
    proof = reference_drift(actual, expected)
    np.testing.assert_allclose(actual, expected, atol=REFERENCE_ATOL, rtol=REFERENCE_RTOL)
    return proof


def paired_summary(rows, *, seed=2026101301):
    """Pair by condition first; bootstrap whole demand groups, never draws."""
    by_condition = defaultdict(dict)
    for row in rows:
        arms = by_condition[row['condition_index']]
        if row['control_scope'] in arms:
            raise ValueError('duplicate condition/arm')
        arms[row['control_scope']] = row
    deltas = []
    for arms in by_condition.values():
        if set(arms) != {'c0', 'full_factual'}:
            raise ValueError('both paired control arms required')
        left, right = arms['full_factual'], arms['c0']
        for key in ('demand_group', 'episode', 'condition_seed'):
            if left[key] != right[key]:
                raise ValueError('paired condition identity differs')
        delta = {key: left[key] for key in ('demand_group', 'episode', 'condition_seed')}
        for field in FIELDS:
            a, b = left.get(field), right.get(field)
            if (a is None) != (b is None):
                raise ValueError('paired metric availability differs')
            delta[field] = None if a is None else a-b
        deltas.append(delta)
    return dict(arms={arm: {field: demand_bootstrap([r for r in rows if r['control_scope']==arm],
                    field, seed=seed) for field in FIELDS} for arm in ('c0', 'full_factual')},
                full_minus_c0={field: demand_bootstrap(deltas, field, seed=seed) for field in FIELDS},
                conditions=len(by_condition), paired_condition_differences=deltas)


def score(draws, target, scales, count, limits, metadata):
    upper = np.stack((np.full_like(limits, float(count)), limits), -1)
    row = evaluate_prediction_samples(draws, target, scales=scales, cohort_count=count,
        state_metadata=[metadata], upper_bounds=upper)['states'][0]
    row.update(supplementary_scores(draws, target, scales))
    return row


def run(args):
    if args.output_dir.exists():
        raise ValueError('preserve previous output; choose a new directory')
    if args.sample_chunk != 4 or args.threads < 1 or not 0 < args.gpu_memory_fraction <= 1:
        raise ValueError('prespecified chunk4 and positive thread/memory limits required')
    checkpoint = args.experiment_dir/'snapshot/checkpoints/latest.pt'
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('only the completed pretrained checkpoint is allowed')
    runtime_hash = verify_source(args.runtime_manifest)
    source_summary = json.loads((args.reference_dir/'summary.json').read_text())
    protocol = json.loads((args.reference_dir/'protocol.json').read_text())
    if (source_summary['status'] != 'complete' or protocol['checkpoint_sha256'] != CHECKPOINT_SHA256
            or protocol['solver_steps'] != 100 or protocol['sample_chunk'] != 4):
        raise ValueError('completed pretrained reference evaluation required')
    conditions = json.loads((args.reference_dir/'conditions.json').read_text())
    manifest = json.loads((args.reference_dir/'bank_manifest.json').read_text())
    selected = []
    for index, (condition, entry) in enumerate(zip(conditions, manifest, strict=True)):
        if condition != entry['condition'] or sha256(entry['file']) != entry['sha256']:
            raise ValueError('reference condition/bank identity changed')
        if condition['traffic_phase'] != 'empty_stress':
            selected.append((index, condition, entry))
    if len(selected) != 24 or len({c['demand_group'] for _, c, _ in selected}) != 8:
        raise ValueError('prespecified 24 nonempty conditions / 8 groups required')
    pilot_selection = None
    if args.deterministic_pilot_audit:
        if not args.deterministic_cuda or not args.reproduction_only:
            raise ValueError('deterministic pilot requires deterministic CUDA and reproduction-only')
        audit = json.loads((args.deterministic_pilot_audit/'reference_reproduction.json').read_text())
        audit_summary = json.loads((args.deterministic_pilot_audit/'summary.json').read_text())
        if audit_summary.get('status') != 'reproduction_audit_complete' or len(audit) != 24:
            raise ValueError('completed 24-condition C0-only drift audit required')
        maximum = max(audit, key=lambda row: row['within_run_c0_repeat_drift']['max_absolute_error'])
        ordinals = {selected[0][0], maximum['condition_index']}
        selected = [entry for entry in selected if entry[0] in ordinals]
        pilot_selection = dict(audit_summary_sha256=sha256(args.deterministic_pilot_audit/'summary.json'),
            audit_proofs_sha256=sha256(args.deterministic_pilot_audit/'reference_reproduction.json'),
            rule='first nonempty condition plus maximum within-run C0 repeat drift; no truth/full-arm scores',
            selected_condition_indices=sorted(ordinals))
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if args.deterministic_cuda:
        if device.type != 'cuda' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise ValueError('deterministic CUDA needs prespecified CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup')
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
    reference_tolerance = DETERMINISTIC_REFERENCE_ATOL if args.deterministic_cuda else REFERENCE_ATOL
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        if free < max(4096*1024**2, args.gpu_memory_fraction*total + 512*1024**2):
            raise RuntimeError('insufficient GPU admission headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    config, prepared, topology, model, _ = load_frozen(args.experiment_dir, device=device)
    model.eval()
    validation = dataset(args.experiment_dir, config, prepared, topology, split='validation', horizon=9)
    capacities = torch.tensor(topology.network.capacities, dtype=torch.float32).clamp_min(1)
    limits = torch.tensor(topology.network.speed_limits_mps, dtype=torch.float32)
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'banks').mkdir()
    atomic_json(args.output_dir/'protocol.json', dict(checkpoint_sha256=CHECKPOINT_SHA256,
        source_manifest_sha256=runtime_hash, code_sha256=sha256(__file__),
        reference_summary_sha256=sha256(args.reference_dir/'summary.json'),
        reference_bank_manifest_sha256=sha256(args.reference_dir/'bank_manifest.json'),
        conditions=len(selected), groups=len({c['demand_group'] for _,c,_ in selected}),
        k=8, horizons=[1], solver='euler', steps=100,
        paired='same observed initial state, encoded latent, per-draw source noise, C0 target, masks and fixed scales',
        controls='C0 recorded events vs full factual recorded events including current-window births',
        purpose='H1 conditioning mismatch diagnosis, not an online causal policy forecast or a new model',
        training_performed=False, final_test_used=False, bootstrap_seed=2026101301,
        comparison='full_factual minus c0; negative errors better; positive coverage higher',
        reproduction_only=args.reproduction_only,
        deterministic_cuda=args.deterministic_cuda, deterministic_pilot_selection=pilot_selection,
        strict_self_repeat_atol=SELF_REPEAT_ATOL,
        numeric_environment=dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
            matmul_precision=torch.get_float32_matmul_precision(),
            matmul_fp32_precision=getattr(torch.backends.cuda.matmul, 'fp32_precision', None),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            environment={name:os.environ.get(name) for name in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG',
                'NVIDIA_TF32_OVERRIDE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}),
        reference_tolerance=dict(atol=reference_tolerance, rtol=REFERENCE_RTOL,
            interpretation='gross semantic discrepancy guard; actual FP32 kernel drift reported separately')))
    rows, proofs, new_banks, paired_inputs = [], [], [], []
    with (args.output_dir/'states.jsonl').open('x') as stream:
        for index, item, entry in selected:
            with np.load(entry['file'], allow_pickle=False) as stored:
                old = {key: stored[key].copy() for key in stored.files}
            if old['initial_cohort_count'] <= 0:
                raise ValueError('nonempty condition has no initial cohort')
            h1 = int(np.flatnonzero(old['horizons']==1)[0])
            expected = old['draws'][h1, :8]
            target, scales = old['targets'][h1], old['scales']
            batch = validation.batch_from_starts(item['episode_index'], [item['start']], fm_only=False)
            cohort = prepare_cohort_batch(batch, capacities=capacities)
            np.testing.assert_allclose(cohort.transition_action.numpy(), old['recorded_actions'], atol=1e-8)
            np.testing.assert_allclose(endpoint_target(cohort.transition_flow[:, 1], batch.flow[:, 2], capacities, limits),
                                       target, atol=1e-5)
            torch.testing.assert_close(batch.flow[:, 0], cohort.transition_flow[:, 0], atol=1e-6, rtol=1e-6)
            with torch.no_grad():
                initial = model.encode(cohort.transition_flow[:, 0].to(device))
            options = dict(capacities=capacities.to(device), limits=limits.to(device), seed=item['condition_seed'],
                           horizons=(1,), samples=8, chunk=4, steps=100)
            c0 = trajectory_bank(model, initial, cohort.transition_action.to(device),
                cohort.transition_action_events.to(device), **options)[1]
            reproduction_path = args.output_dir/'banks'/f'reproduction_{index:03d}.npz'
            coefficient_seeds = [stable_seed(item['condition_seed'],'source_coefficients_v1',ordinal,0)
                                 for ordinal in range(8)]
            coefficients = np.concatenate([torch.randn((1,model.dynamics.graph_prior.rank,initial.shape[-1]),
                generator=torch.Generator(device='cpu').manual_seed(seed)).numpy() for seed in coefficient_seeds])
            stored_reproduction = dict(regenerated_c0_draws=c0, reference_c0_draws=expected,
                initial_latent=initial.detach().cpu().numpy(), coefficient_seeds=np.asarray(coefficient_seeds,dtype=np.uint64),
                source_coefficients=coefficients)
            if args.reproduction_only or args.deterministic_cuda:
                repeat = trajectory_bank(model, initial, cohort.transition_action.to(device),
                    cohort.transition_action_events.to(device), **options)[1]
                stored_reproduction['second_regenerated_c0_draws'] = repeat
            np.savez_compressed(reproduction_path, **stored_reproduction)
            proof = dict(condition_index=index, **reference_drift(c0, expected, atol=reference_tolerance),
                         condition_seed=item['condition_seed'], coefficient_seeds=coefficient_seeds,
                         source_coefficients_sha256=hashlib.sha256(coefficients.tobytes()).hexdigest(),
                         reproduction_file=str(reproduction_path.resolve()), reproduction_sha256=sha256(reproduction_path),
                         c0_events=len(cohort.transition_action_events.for_step(0)),
                         full_events=len(batch.action_events.for_step(0)))
            if args.reproduction_only or args.deterministic_cuda:
                proof['within_run_c0_repeat_drift'] = reference_drift(repeat, c0, atol=SELF_REPEAT_ATOL if args.deterministic_cuda else REFERENCE_ATOL)
            proofs.append(proof)
            paired_inputs.append((index, item, batch, cohort, initial, c0, old, target, scales, options, proof))
            atomic_json(args.output_dir/'reference_reproduction.json', proofs)
            print(json.dumps(dict(phase='c0_reproduction', conditions_done=len(proofs), total=len(selected), **proof)), flush=True)
            if args.deterministic_cuda:
                np.testing.assert_allclose(repeat, c0, atol=SELF_REPEAT_ATOL, rtol=0)
                np.testing.assert_allclose(c0, expected, atol=reference_tolerance, rtol=0)
            elif not args.reproduction_only:
                assert_reference(c0, expected)
        if args.reproduction_only:
            atomic_json(args.output_dir/'summary.json', dict(status='reproduction_audit_complete',
                conditions=len(proofs), full_control_predictions_generated=False, predictive_scores_computed=False,
                checkpoint_sha256=CHECKPOINT_SHA256,
                max_count_drift=max(p['count_max_absolute_error'] for p in proofs),
                max_speed_drift=max(p['speed_max_absolute_error'] for p in proofs),
                max_within_run_count_drift=max(p['within_run_c0_repeat_drift']['count_max_absolute_error'] for p in proofs),
                max_within_run_speed_drift=max(p['within_run_c0_repeat_drift']['speed_max_absolute_error'] for p in proofs),
                tolerance_pass_conditions=sum(p['within_prospective_tolerance'] for p in proofs)))
            return
        # Every original C0 bank must pass before any new full-control prediction.
        for index, item, batch, cohort, initial, c0, old, target, scales, options, proof in paired_inputs:
            full = trajectory_bank(model, initial, batch.action.to(device), batch.action_events.to(device), **options)[1]
            original_events, cohort_events = batch.action_events.for_step(0), cohort.transition_action_events.for_step(0)
            same_events = (original_events.vehicle_ids == cohort_events.vehicle_ids and
                all(torch.equal(getattr(original_events, field), getattr(cohort_events, field)) for field in
                    ('batch_index','time_index','phase','current_edge','next_edge','destination_edge','event_index')))
            proof.update(mean_absolute_prediction_change=float(np.abs(full-c0).mean()),
                         controls_identical=bool(same_events and torch.equal(batch.action[:,0], cohort.transition_action[:,0])))
            if proof['controls_identical']:
                assert_reference(full, c0)
            count = int(old['initial_cohort_count'])
            for arm, draws in [('c0', c0), ('full_factual', full)]:
                row = score(draws, target, scales, count, limits.numpy(),
                    dict(item, condition_index=index, horizon=1, k=8, control_scope=arm))
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            path = args.output_dir/'banks'/f'condition_{index:03d}.npz'
            np.savez_compressed(path, c0_draws=c0, full_factual_draws=full, targets=target,
                                scales=scales, capacities=capacities.numpy(), speed_limits=limits.numpy())
            new_banks.append(dict(condition=item, file=str(path.resolve()), sha256=sha256(path)))
            atomic_json(args.output_dir/'bank_manifest.json', new_banks)
            atomic_json(args.output_dir/'reference_reproduction.json', proofs)
            stream.flush()
            print(json.dumps(dict(phase='full_control_comparison', conditions_done=len(new_banks), total=len(selected), **proof)), flush=True)
    summary = dict(status='complete', checkpoint_sha256=CHECKPOINT_SHA256, **paired_summary(rows))
    atomic_json(args.output_dir/'summary.json', summary)
    with (args.output_dir/'paired_metrics.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['metric','c0','full_factual','delta','ci_low','ci_high'])
        writer.writeheader()
        for field in FIELDS:
            delta = summary['full_minus_c0'][field]
            writer.writerow(dict(metric=field, c0=summary['arms']['c0'][field]['estimate'],
                full_factual=summary['arms']['full_factual'][field]['estimate'],
                delta=delta['estimate'], ci_low=delta['ci_low'], ci_high=delta['ci_high']))
    text = ['# Completed pretrained H1 control-scope diagnostic', '',
        '24 paired nonempty states, eight development demand groups; K8/Euler100. No training.', '',
        'Each C0 bank was reproduced before replacing only the recorded action conditions with full factual controls.',
        'The controls include within-window new vehicles, so this is an offline conditioning diagnostic, not an actor evaluation.', '',
        '| Metric | C0 | Full factual | Full−C0 | Paired 95% CI |', '|---|---:|---:|---:|---|']
    for field in FIELDS:
        left = summary['arms']['c0'][field]['estimate']; right = summary['arms']['full_factual'][field]['estimate']
        delta = summary['full_minus_c0'][field]
        if left is not None:
            text.append(f"| {field} | {left:.6g} | {right:.6g} | {delta['estimate']:.6g} | [{delta['ci_low']:.6g}, {delta['ci_high']:.6g}] |")
    text.extend(['', 'Intervals resample demand groups, are unadjusted, and cover one pretrained checkpoint in one road network.',
        'K8 empirical central90% coverage has finite-sample discretization and should not be compared with nominal90% alone.',
        'This experiment separates a one-step control-scope change only. It does not identify multistep recursion effects or superiority to regression.'])
    (args.output_dir/'report.md').write_text('\n'.join(text)+'\n')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--experiment-dir', type=Path, required=True)
    result.add_argument('--reference-dir', type=Path, required=True)
    result.add_argument('--runtime-manifest', type=Path, required=True)
    result.add_argument('--output-dir', type=Path, required=True)
    result.add_argument('--sample-chunk', type=int, default=4)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--threads', type=int, default=2)
    result.add_argument('--gpu-memory-fraction', type=float, default=.06)
    result.add_argument('--reproduction-only', action='store_true')
    result.add_argument('--deterministic-cuda', action='store_true')
    result.add_argument('--deterministic-pilot-audit', type=Path)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
