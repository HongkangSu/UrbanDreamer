"""Audit and summarize the paired 100-episode, three-seed SUMO screening."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'outputs/baseline_validation_20261015'
METHODS = ('ippo', 'mappo', 'mamba', 'mbpo', 'dreamer')
VARIANTS = ('original', 'repaired')
SEEDS = (42, 43, 44)
EPISODES = 100
EVAL_SEEDS = tuple(range(10042, 10047))
POLICIES = ('argmax', 'sampled_temperature_1')
PAIR_FIELDS = ('demand_sha256', 'reward_contract', 'vehicle_count')
CORE_OUTCOMES = ('global_reward_sum', 'completion_rate', 'branch_entropy')
OUTCOMES = CORE_OUTCOMES + ('never_departed', 'unfinished', 'observed_mean_seconds', 'started_vehicle_count')
TIMINGS = ('total_seconds', 'collection_measured_seconds', 'learning_measured_seconds',
           'collection_seconds', 'wall_seconds', 'actor_update_seconds',
           'critic_update_seconds', 'world_model_seconds', 'imagination_seconds', 'learning_seconds')


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def distribution(values):
    return dict(n=len(values), mean=mean(values),
                seed_std=statistics.stdev(values) if len(values) > 1 else None,
                min=min(values) if values else None, max=max(values) if values else None)


def timing_medians(rows):
    return {field: statistics.median([row[field] for row in rows if field in row])
            for field in TIMINGS if any(field in row for row in rows)}


def update_diagnostics(rows):
    counters = {}
    for field in ('actor_optimizer_steps', 'kl_early_stopped', 'forced_decision_count',
                  'actor_decision_count', 'actual_choice_count', 'model_warmup_complete'):
        values = [row[field] for row in rows if field in row]
        counters[field] = dict(observed_episodes=len(values), missing_episodes=len(rows) - len(values),
            total=sum(values) if values else None,
            positive_episodes=sum(value > 0 for value in values) if values else None,
            max=max(values) if values else None)
    for field in ('policy_kl', 'mamba_policy_kl'):
        values = [row[field] for row in rows if field in row]
        counters[field] = dict(observed_episodes=len(values), mean=mean(values),
                               max=max(values) if values else None)
    return counters


def outcome_means(rows):
    return {field: mean([row[field] for row in rows if field in row]) for field in OUTCOMES}


def metric_key(row):
    if row['phase'] == 'train':
        return ('train', row['episode'])
    return ('evaluation', row['episode'], row['policy'], row['evaluation_seed'])


def json_file(path, errors):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors.append(f'{path}: {type(exc).__name__}: {exc}')
        return None


def read_metrics(path, partial, errors, warnings):
    if not path.exists():
        return []
    rows = []
    lines = path.read_text().splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except ValueError as exc:
            if partial and line_number == len(lines) and not line.endswith('\n'):
                warnings.append(f'{path}: ignored in-progress final line')
                continue
            errors.append(f'{path}:{line_number}: {exc}')
            continue
        if record.get('phase') not in ('train', 'evaluation'):
            errors.append(f'{path}:{line_number}: unknown phase {record.get("phase")!r}')
            continue
        required = ('episode', 'seed', *PAIR_FIELDS, *CORE_OUTCOMES)
        required += (('actor_delta_l2',) if record['phase'] == 'train' else ('policy', 'evaluation_seed'))
        missing = [field for field in required if field not in record]
        if missing:
            errors.append(f'{path}:{line_number}: missing fields {missing}')
            continue
        nonfinite = [field for field, value in record.items()
                     if isinstance(value, (int, float)) and not math.isfinite(value)]
        if nonfinite:
            errors.append(f'{path}:{line_number}: nonfinite metrics {nonfinite}')
            continue
        if not 0. <= record['completion_rate'] <= 1.:
            errors.append(f'{path}:{line_number}: completion_rate outside [0,1]')
        if not record['demand_sha256'] or not record['reward_contract'] or record['vehicle_count'] <= 0:
            errors.append(f'{path}:{line_number}: missing/empty pairing contract')
        rows.append(record)
    return rows


def summarize_job(directory, partial, errors, warnings):
    method = directory.parent.name
    variant, seed_text = directory.name.split('_seed')
    seed = int(seed_text)
    records = read_metrics(directory / 'metrics.jsonl', partial, errors, warnings)
    indexed = {}
    for row in records:
        key = metric_key(row)
        if key in indexed:
            errors.append(f'{directory}: duplicate record {key}')
        indexed[key] = row
        expected_seed = seed + row['episode'] if row['phase'] == 'train' else row['evaluation_seed']
        if row['seed'] != expected_seed:
            errors.append(f'{directory}: demand seed mismatch at {key}')
    trains = sorted((row for row in records if row['phase'] == 'train'), key=lambda row: row['episode'])
    evaluations = [row for row in records if row['phase'] == 'evaluation']
    expected_train = {('train', ep) for ep in range(1, EPISODES + 1)}
    expected_eval = {('evaluation', ep, policy, eval_seed)
                     for ep in (0, EPISODES) for policy in POLICIES for eval_seed in EVAL_SEEDS}
    expected = expected_train | expected_eval
    unexpected = set(indexed) - expected
    if unexpected:
        errors.append(f'{directory}: unexpected keys {sorted(unexpected)}')
    completed_path = directory / 'completed.json'
    completed = json_file(completed_path, errors) if completed_path.exists() else None
    provenance_path = directory / 'provenance.json'
    provenance = json_file(provenance_path, errors) if provenance_path.exists() else None
    if completed is not None:
        if completed.get('episodes') != EPISODES:
            errors.append(f'{directory}: completed.episodes must equal {EPISODES}')
        if completed.get('numerical_parameters_finite') is not True:
            errors.append(f'{directory}: numerical_parameters_finite is not true')
        if set(indexed) != expected or len(records) != 120:
            errors.append(f'{directory}: completed job lacks exactly 100 train + 20 evaluation records')
    if not partial:
        if completed is None:
            errors.append(f'{directory}: missing completed.json')
        if provenance is None:
            errors.append(f'{directory}: missing provenance.json')
        if set(indexed) != expected:
            errors.append(f'{directory}: missing {len(expected - set(indexed))} expected records')
    final_window = [row for row in trains if 81 <= row['episode'] <= 100]
    # Partial reports distinguish the actual 81..100 window from the latest available rows.
    def train_window(rows):
        return dict(episodes=[row['episode'] for row in rows], count=len(rows),
            means=outcome_means(rows),
            actor_delta_l2_mean=mean([row['actor_delta_l2'] for row in rows]),
            actor_delta_l2_sum=sum(row['actor_delta_l2'] for row in rows),
            actor_updated_episodes=sum(row['actor_delta_l2'] > 1e-12 for row in rows),
            entropy_coefficient_mean=mean([row['entropy_coefficient'] for row in rows
                                           if 'entropy_coefficient' in row]),
            timing_medians=timing_medians(rows), update_diagnostics=update_diagnostics(rows))
    summary = dict(method=method, variant=variant, training_seed=seed, path=str(directory),
        completed=completed, train_count=len(trains), evaluation_count=len(evaluations),
        missing_record_count=len(expected - set(indexed)),
        training_episodes_observed=[row['episode'] for row in trains],
        actor_updated_episodes=sum(row['actor_delta_l2'] > 1e-12 for row in trains),
        actor_update_threshold_l2=1e-12, actor_delta_l2_mean=mean([row['actor_delta_l2'] for row in trains]),
        last20_train=train_window(final_window), last20_window_complete=len(final_window) == 20,
        latest_available_train=train_window(trains[-20:]), timing_medians=timing_medians(trains),
        update_diagnostics=update_diagnostics(trains),
        evaluations={stage: {policy: dict(count=len(subset), means=outcome_means(subset),
            evaluation_seeds=sorted(row['evaluation_seed'] for row in subset))
            for policy in POLICIES
            for subset in [[row for row in evaluations if row['episode'] == episode and row['policy'] == policy]]}
            for stage, episode in (('initial', 0), ('final', EPISODES))},
        provenance=provenance)
    return dict(summary=summary, indexed=indexed, trains=trains, evaluations=evaluations)


def audit_provenance(jobs, errors):
    signatures = {}
    for key, job in jobs.items():
        if not job['indexed']:
            continue
        provenance = job['summary']['provenance']
        if provenance is None:
            continue
        config, args = provenance.get('config', {}), provenance.get('args', {})
        required = dict(simulation=config.get('simulation'), inputs=config.get('inputs'),
                        input_hashes=provenance.get('input_hashes'))
        if any(not value for value in required.values()):
            errors.append(f'{key}: provenance lacks resolved simulation/inputs/input_hashes')
        required['runtime'] = {name: args.get(name) for name in (
            'session', 'backend', 'step_seconds', 'reward_objective', 'global_reward_only', 'vehicle_limit')}
        signatures[key] = required
        method, variant, seed = key.split('/')
        if args.get('method') != method or args.get('seed') != int(seed) or args.get('episodes') != EPISODES:
            errors.append(f'{key}: provenance job identity/episodes disagree')
        expected_source = Path(job['summary']['path']).parents[2] / ('source_' + variant)
        if Path(provenance.get('source', '')).resolve() != expected_source.resolve():
            errors.append(f'{key}: unexpected source {provenance.get("source")}')
        for field in ('common_module', 'method_module'):
            if not Path(provenance.get(field, '')).resolve().is_relative_to(expected_source.resolve()):
                errors.append(f'{key}: {field} outside isolated source')
    reference = next(iter(signatures.values()), None)
    for key, signature in signatures.items():
        if signature != reference:
            errors.append(f'{key}: simulator physics/runtime/inputs differ across jobs')
    return dict(jobs_checked=len(signatures), all_equal=all(value == reference for value in signatures.values()),
                reference=reference, per_job_signatures=signatures)


def paired_analysis(jobs, errors, partial):
    methods = {}
    demand_contracts = {}
    checked_pairs = 0
    for method in METHODS:
        per_seed = {}
        for seed in SEEDS:
            original = jobs[f'{method}/original/{seed}']
            repaired = jobs[f'{method}/repaired/{seed}']
            common = original['indexed'].keys() & repaired['indexed'].keys()
            unpaired = original['indexed'].keys() ^ repaired['indexed'].keys()
            if unpaired and not partial:
                errors.append(f'{method}/{seed}: unpaired metric keys')
            differences = {}
            for key in sorted(common):
                left, right = original['indexed'][key], repaired['indexed'][key]
                for field in PAIR_FIELDS:
                    if left[field] != right[field]:
                        errors.append(f'{method}/{seed}/{key}: paired {field} mismatch')
                checked_pairs += 1
                differences[key] = {field: right[field] - left[field]
                    if field in left and field in right else None for field in OUTCOMES}
            for job in (original, repaired):
                for key, row in job['indexed'].items():
                    # Prepared demand cohorts depend on the training seed. Check
                    # repeated evaluation/episode seeds within that cohort only.
                    contract = tuple(row[field] for field in PAIR_FIELDS)
                    demand_key = (seed, row['seed'])
                    if demand_key in demand_contracts and demand_contracts[demand_key] != contract:
                        errors.append(f'{method}/{seed}/{key}: cross-job demand contract mismatch')
                    demand_contracts[demand_key] = contract
            train_differences = [value for key, value in differences.items()
                                 if key[0] == 'train' and 81 <= key[1] <= 100]
            evaluations = {stage: {policy: dict(count=len(subset),
                new_minus_original={field: mean([row[field] for row in subset]) for field in OUTCOMES})
                for policy in POLICIES
                for subset in [[value for key, value in differences.items()
                    if key[0] == 'evaluation' and key[1] == episode and key[2] == policy]]}
                for stage, episode in (('initial', 0), ('final', EPISODES))}
            delta_change = {}
            for policy in POLICIES:
                initial = evaluations['initial'][policy]['new_minus_original']
                final = evaluations['final'][policy]['new_minus_original']
                delta_change[policy] = {field: final[field] - initial[field]
                    if initial[field] is not None and final[field] is not None else None for field in OUTCOMES}
            original_time, repaired_time = original['summary']['timing_medians'], repaired['summary']['timing_medians']
            per_seed[str(seed)] = dict(paired_records=len(common), pending_unpaired_records=len(unpaired),
                last20_train=dict(count=len(train_differences), new_minus_original={field:
                    mean([row[field] for row in train_differences]) for field in OUTCOMES}),
                evaluations=evaluations, evaluation_change_new_minus_original=delta_change,
                median_timing_speedup_original_over_repaired={field: original_time[field] / repaired_time[field]
                    for field in original_time.keys() & repaired_time.keys() if repaired_time[field] > 0})
        aggregate = dict(last20_train={field: distribution([entry['last20_train']['new_minus_original'][field]
            for entry in per_seed.values() if entry['last20_train']['new_minus_original'][field] is not None])
            for field in OUTCOMES}, evaluations={stage: {policy: {field: distribution([
                entry['evaluations'][stage][policy]['new_minus_original'][field] for entry in per_seed.values()
                if entry['evaluations'][stage][policy]['new_minus_original'][field] is not None])
                for field in OUTCOMES} for policy in POLICIES} for stage in ('initial', 'final')},
            evaluation_change_new_minus_original={policy: {field: distribution([
                entry['evaluation_change_new_minus_original'][policy][field] for entry in per_seed.values()
                if entry['evaluation_change_new_minus_original'][policy][field] is not None])
                for field in OUTCOMES} for policy in POLICIES})
        methods[method] = dict(per_seed=per_seed, paired_new_minus_original=aggregate)
    return dict(paired_records_checked=checked_pairs,
                unique_training_cohort_demand_seeds_checked=len(demand_contracts), methods=methods)


def make_figures(jobs, output, partial):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        return dict(available=False, reason=str(exc))
    fig, axes = plt.subplots(2, len(METHODS), figsize=(22, 7), sharex=True, squeeze=False)
    for column, method in enumerate(METHODS):
        for variant, color in (('original', '#c75d39'), ('repaired', '#276eb1')):
            for row_number, (metric, ylabel) in enumerate((('global_reward_sum', 'Team reward (higher is better)'),
                                                          ('completion_rate', 'Completion fraction'))):
                x, means, deviations = [], [], []
                for episode in range(1, EPISODES + 1):
                    values = [job['indexed'][('train', episode)][metric] for seed in SEEDS
                        for job in [jobs[f'{method}/{variant}/{seed}']]
                        if ('train', episode) in job['indexed']]
                    # A partial plot never mixes n=1/2 bands with n=3 estimates.
                    if len(values) == len(SEEDS):
                        x.append(episode)
                        means.append(statistics.fmean(values))
                        deviations.append(statistics.stdev(values))
                ax = axes[row_number, column]
                if x:
                    m, std = np.asarray(means), np.asarray(deviations)
                    ax.plot(x, m, color=color, label=variant, linewidth=1.)
                    ax.fill_between(x, m - std, m + std, color=color, alpha=.18)
                ax.grid(alpha=.2)
                ax.set_xlabel('Training episode')
                if column == 0:
                    ax.set_ylabel(ylabel)
                if row_number == 0:
                    ax.set_title(method.upper())
                if metric == 'completion_rate':
                    ax.set_ylim(bottom=0.)
        if axes[0, column].get_legend_handles_labels()[0]:
            axes[0, column].legend(fontsize=8)
    fig.suptitle(('PARTIAL: ' if partial else '') +
        'Real SUMO training: raw episode means ± sample SD across seeds 42, 43, 44\n'
        'Shading describes seed variability; no smoothing, no convergence claim', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91))
    stem = 'partial_training_curves' if partial else 'training_curves'
    paths = []
    for extension in ('png', 'pdf'):
        path = output / f'{stem}.{extension}'
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    return dict(available=True, paths=paths, smoothing='none',
                line='raw episode mean across exactly three seeds',
                band='sample standard deviation across seeds, ddof=1; not confidence intervals')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--partial', action='store_true', help='Allow unfinished jobs; never issue final conclusions')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    errors, warnings, jobs = [], [], {}
    for method in METHODS:
        for variant in VARIANTS:
            for seed in SEEDS:
                key = f'{method}/{variant}/{seed}'
                jobs[key] = summarize_job(output / 'runs' / method / f'{variant}_seed{seed}',
                                          args.partial, errors, warnings)
    provenance_audit = audit_provenance(jobs, errors)
    paired = paired_analysis(jobs, errors, args.partial)
    completed = sum(job['summary']['completed'] is not None for job in jobs.values())
    complete = completed == 30 and all(job['summary']['train_count'] == 100
        and job['summary']['evaluation_count'] == 20 for job in jobs.values()) and not errors
    effect = None
    if not args.partial and complete:
        effect = {}
        for method, comparison in paired['methods'].items():
            effect[method] = {}
            for policy in POLICIES:
                deltas = [entry['evaluations']['final'][policy]['new_minus_original']['global_reward_sum']
                          for entry in comparison['per_seed'].values()]
                completion_deltas = [entry['evaluations']['final'][policy]['new_minus_original']['completion_rate']
                                     for entry in comparison['per_seed'].values()]
                if all(abs(value) <= 1e-12 for value in deltas):
                    verdict = 'no_observed_reward_difference'
                elif all(value > 1e-12 for value in deltas):
                    verdict = 'all_three_seeds_have_higher_reward'
                elif all(value < -1e-12 for value in deltas):
                    verdict = 'all_three_seeds_have_lower_reward'
                else:
                    verdict = 'mixed_or_unclear_reward_change'
                effect[method][policy] = dict(final_paired_reward_deltas=deltas,
                    final_paired_completion_deltas=completion_deltas, verdict=verdict,
                    seeds_with_positive_delta=sum(value > 1e-12 for value in deltas),
                    seeds_with_negative_delta=sum(value < -1e-12 for value in deltas),
                    description='Descriptive comparison of three training seeds; no significance or convergence claim')
            if method in ('ippo', 'mappo'):
                diagnostic = [jobs[f'{method}/repaired/{seed}']['summary']['update_diagnostics'] for seed in SEEDS]
                kl_episodes = sum(value['kl_early_stopped']['positive_episodes'] or 0 for value in diagnostic)
                forced_events = sum(value['forced_decision_count']['total'] or 0 for value in diagnostic)
                effect[method]['repaired_guard_activation'] = dict(kl_early_stop_episodes=kl_episodes,
                    forced_decisions=forced_events,
                    scenario_exercises_repaired_guards=bool(kl_episodes or forced_events),
                    interpretation=('Observed guard activations; inspect paired outcomes separately.'
                        if kl_episodes or forced_events else
                        'No guard activation observed. This scenario cannot validate mitigation of those instability triggers.'))
    result = dict(schema='paired_sumo_baseline_validation_v1', generated_at=time.time(),
        partial=args.partial, status='partial' if args.partial else 'complete' if complete else 'invalid_or_incomplete',
        expected_jobs=30, completed_jobs=completed, valid_complete=complete,
        expected_per_job=dict(train=100, evaluation=20), errors=errors, warnings=warnings,
        provenance_audit=provenance_audit, jobs={key: value['summary'] for key, value in jobs.items()},
        paired=paired, final_effect_description=effect,
        limits=['Toy screening with three seeds and 100 episodes does not establish city or asymptotic convergence.',
                'Reward is the recorded global_reward_sum; greater values are better. Entropy is branch_entropy.',
                'Actor update counts use measured parameter L2 change > 1e-12, not nonzero actor loss.',
                'Finite parameters establish numerical execution only; effectiveness depends on paired outcome deltas.',
                'Absent KL/forced-action activations mean this SUMO scenario does not exercise those repaired safeguards.',
                'Last20 means episodes 81..100; partial latest_available_train is separately labelled.',
                'Paired checks compare exogenous demand and reward contracts; policy actions may differ.',
                'Timing medians reflect concurrent CPU runs and scheduling/load; speedup is descriptive.',
                'Seed SD describes three independent training seeds; five fixed evaluation demands are averaged within seed.'])
    result['figures'] = (dict(available=False, reason='--no-plots') if args.no_plots
                         else make_figures(jobs, output, args.partial or not complete))
    path = output / ('partial_summary.json' if args.partial else 'summary.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    print(json.dumps(dict(path=str(path), status=result['status'], completed_jobs=completed,
        errors=errors, figures=result['figures']), indent=2))
    return 1 if errors or (not args.partial and not complete) else 0


if __name__ == '__main__':
    raise SystemExit(main())
