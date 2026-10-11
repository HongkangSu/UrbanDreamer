"""Strictly paired development WM reports; never pool incompatible evaluations.

All intervals resample demand groups of a fixed trained checkpoint. Independent
training seeds are kept separate; these intervals do not measure training
variability. Input directories must have completed evaluator summaries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tools.component_prediction_metrics import disagreement_analysis

METRICS = tuple(f'{feature}_{metric}_{scope}' for feature in ('count', 'speed')
                for metric in ('mae', 'crps') for scope in ('all', 'occupied')) + ('energy_score', 'total_count_absolute_error')
CONDITION_KEYS = ('source_path', 'episode', 'demand_group', 'start', 'condition_seed', 'requested_horizon', 'traffic_phase')
STATE_KEYS = ('source_path', 'episode', 'demand_group', 'start', 'protocol', 'horizon')
COMMON_KEYS = ('horizons', 'samples', 'sample_chunk', 'steps', 'solver', 'target_speed', 'scales',
               'condition_sampling', 'phase_definition', 'missing_horizons', 'disagreement', 'empty_stress_aggregated_separately')
CAMPAIGN_KEYS = ('development_manifest_identity', 'checkpoint_sha256', 'config_sha256')
TRAINING_KEYS = ('batch_size', 'data_seed', 'horizon', 'learning_rate', 'mode', 'optimizer', 'precision',
                 'sampler', 'training_manifest_identity', 'recursive_solver_steps', 'ms_samples', 'ms_weight', 'prior_weight')
PREDICTORS = {'predictor', 'persistence', 'endpoint_codec'}
KNOWN_HORIZON_POLICIES = {
    'omit endpoints beyond recorded trajectory, never extrapolate truth',
    'CLI rejects incomplete batches; direct evaluate_condition supports missing-label diagnostics',
}
COMPLETE_HORIZON_POLICY = 'verified complete requested horizons with full count and speed labels'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row, fields):
    try:
        return tuple(row[name] for name in fields)
    except KeyError as error:
        raise ValueError(f'missing pairing field {error.args[0]}') from error


def read_evaluation(directory):
    directory = Path(directory)
    files = {name: directory / name for name in ('protocol.json', 'conditions.json', 'states.jsonl', 'summary.json')}
    if any(not path.is_file() for path in files.values()):
        raise ValueError(f'incomplete evaluation directory: {directory}')
    protocol = json.loads(files['protocol.json'].read_text())
    summary = json.loads(files['summary.json'].read_text())
    if summary.get('status') != 'complete' or protocol.get('role') != 'development_not_final_test':
        raise ValueError('only complete development evaluations may be summarized')
    conditions = json.loads(files['conditions.json'].read_text())
    condition_keys = [key(row, CONDITION_KEYS) for row in conditions]
    if len(set(condition_keys)) != len(condition_keys) or len(conditions) != summary['conditions']:
        raise ValueError('duplicate conditions or condition count mismatch')
    rows = [json.loads(line) for line in files['states.jsonl'].read_text().splitlines() if line.strip()]
    by_state = defaultdict(dict)
    observed_conditions = set()
    for row in rows:
        observed_conditions.add(key(row, CONDITION_KEYS))
        state = by_state[key(row, STATE_KEYS)]
        predictor = row['predictor']
        if predictor not in PREDICTORS or predictor in state:
            raise ValueError('unknown or duplicated state predictor')
        if row['variant'] != protocol['model']['variant']:
            raise ValueError('row variant differs from protocol model')
        expected_draws = protocol['samples'] if predictor == 'predictor' else 1
        if row['draws'] != expected_draws:
            raise ValueError('state draw count differs from declared sample protocol')
        state[predictor] = row
    if observed_conditions != set(condition_keys):
        raise ValueError('states do not cover exactly the declared conditions')
    if any(set(values) != PREDICTORS for values in by_state.values()):
        raise ValueError('each state must include predictor, persistence and endpoint codec')
    training = protocol.get('training_protocol')
    if training is not None and training.get('frozen_codec') is not True:
        raise ValueError('comparison requires a declared frozen codec')
    return dict(directory=str(directory.resolve()), label=directory.name, protocol=protocol,
                conditions=conditions, states=dict(by_state), variant=protocol['model']['variant'],
                training_seed=None if training is None else training['training_seed'],
                hashes={name: file_hash(path) for name, path in files.items()})


def complete_horizon_proof(evaluation):
    """Prove equivalence of known selection policies from the realized records.

    Never infer complete observations from a policy name or from equal missing
    windows in both runs. Every declared condition needs all requested forecasts
    and labels, including the following-boundary speed target at the final step.
    """
    protocol = evaluation['protocol']
    original = protocol.get('missing_horizons')
    if original not in KNOWN_HORIZON_POLICIES:
        raise ValueError(f'unknown missing_horizons policy: {original!r}')
    horizons = protocol.get('horizons')
    if (not isinstance(horizons, list) or not horizons or len(set(horizons)) != len(horizons)
            or any(isinstance(h, bool) or not isinstance(h, int) or h < 1 for h in horizons)):
        raise ValueError('horizons must declare distinct positive integer forecasts')
    declared = {key(row, CONDITION_KEYS) for row in evaluation['conditions']}
    required = {('one_step_previous_state', 1)} | {
        ('recorded_action_recursive_initial_cohort', h) for h in horizons}
    observed = defaultdict(set)
    valid_road_counts = []
    for state in evaluation['states'].values():
        row = state['predictor']
        condition = key(row, CONDITION_KEYS)
        if condition not in declared:
            raise ValueError('states contain undeclared conditions')
        forecast = (row['protocol'], row['horizon'])
        if forecast in observed[condition]:
            raise ValueError('duplicate predictor condition/protocol/horizon')
        observed[condition].add(forecast)
        roads = row.get('roads')
        if (isinstance(roads, bool) or not isinstance(roads, int) or roads <= 0
                or row.get('count_valid_roads_all') != roads
                or row.get('speed_valid_roads_all') != roads):
            raise ValueError('complete-horizon proof requires full positive count and speed road labels')
        valid_road_counts.append(roads)
    if not declared or any(observed[condition] != required for condition in declared):
        raise ValueError('conditions lack complete one-step and requested recursive horizons')
    return dict(original_policy=original, canonical_policy=COMPLETE_HORIZON_POLICY,
                conditions_checked=len(declared), recursive_horizons=sorted(horizons),
                predictor_states_checked=len(valid_road_counts),
                minimum_valid_roads=min(valid_road_counts),
                all_requested_forecasts_present=True, all_count_and_speed_roads_valid=True)


def compatibility_signature(evaluation):
    protocol = evaluation['protocol']
    try:
        result = {name: protocol[name] for name in COMMON_KEYS}
        result['campaign'] = {name: protocol['campaign'][name] for name in CAMPAIGN_KEYS}
    except KeyError as error:
        raise ValueError(f'missing comparison protocol field {error.args[0]}') from error
    result['missing_horizons'] = complete_horizon_proof(evaluation)['canonical_policy']
    training = protocol.get('training_protocol')
    if training is not None:
        parent = training.get('parent', training)
        if any(parent.get(name) != result['campaign'][name] for name in CAMPAIGN_KEYS):
            raise ValueError('training source differs from evaluator frozen-codec campaign')
    result['training_origin'] = 'frozen_pretrain_snapshot' if training is None else 'fresh_dynamics_screen'
    result['training_update'] = protocol.get('training_update')
    result['training_budget'] = None if training is None else {name: training.get(name) for name in TRAINING_KEYS}
    result['conditions'] = sorted(key(row, CONDITION_KEYS) for row in evaluation['conditions'])
    return result


def validate_pairing(evaluations):
    if not evaluations:
        raise ValueError('at least one evaluation is required')
    reference = evaluations[0]
    signature = compatibility_signature(reference)
    identities = set()
    for evaluation in evaluations:
        identity = (evaluation['variant'], evaluation['training_seed'])
        if identity in identities:
            raise ValueError('duplicate variant/training-seed: report separate checkpoints in separate reports')
        identities.add(identity)
        candidate = compatibility_signature(evaluation)
        differences = [name for name in signature if candidate[name] != signature[name]]
        if differences:
            raise ValueError(f"incompatible evaluation {evaluation['label']}: {', '.join(differences)}")
        if evaluation['states'].keys() != reference['states'].keys():
            raise ValueError('state condition/protocol/horizon sets differ; refusing partial pairing')
        for state_key, state in evaluation['states'].items():
            source = reference['states'][state_key]
            for predictor in PREDICTORS:
                row, expected = state[predictor], source[predictor]
                invariants = ['roads', 'occupied_roads', 'energy_score_dimensions', 'true_total_count', 'congestion',
                              'forecast_seconds', 'disagreement_independent_samples']
                invariants += [f'{feature}_valid_roads_{scope}' for feature in ('count', 'speed') for scope in ('all', 'occupied')]
                for name in invariants:
                    if row.get(name) != expected.get(name):
                        raise ValueError(f'target mask or state semantics mismatch: {name}')
                # Equal deterministic references additionally detect changed labels/codec.
                if predictor != 'predictor':
                    for metric in METRICS:
                        if row.get(metric) != expected.get(metric):
                            raise ValueError(f'baseline/codec mismatch: {predictor}.{metric}')
    return signature


def demand_bootstrap(rows, metric, *, repetitions=1000, seed=0):
    """Equal demands, then episodes, then states; resample whole demands only."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = row.get(metric)
        if value is None:
            continue
        if not np.isfinite(value):
            raise ValueError(f'nonfinite metric {metric}')
        grouped[row['demand_group']][row['episode']].append(float(value))
    values = np.array([np.mean([np.mean(windows) for windows in episodes.values()])
                       for _, episodes in sorted(grouped.items())])
    result = dict(estimate=None, ci_low=None, ci_high=None, demand_groups=len(values))
    if not len(values):
        return result
    result['estimate'] = float(values.mean())
    if len(values) >= 2:
        indices = np.random.default_rng(seed).integers(len(values), size=(repetitions, len(values)))
        low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
        result.update(ci_low=float(low), ci_high=float(high))
    return result


def paired_metric(left, right, metric, *, repetitions, seed):
    if left.keys() != right.keys():
        raise ValueError('paired metrics require identical state keys')
    differences = []
    for state_key, row in left.items():
        other = right[state_key]
        a, b = row.get(metric), other.get(metric)
        if (a is None) != (b is None):
            raise ValueError(f'paired metric availability differs: {metric}')
        differences.append(dict(row, paired_delta=None if a is None else a - b))
    return demand_bootstrap(differences, 'paired_delta', repetitions=repetitions, seed=seed)


def partitions(evaluation, predictor):
    groups = defaultdict(dict)
    for state_key, state in evaluation['states'].items():
        row = state[predictor]
        stratum = 'empty_stress' if row['traffic_phase'] == 'empty_stress' else 'traffic_phases'
        groups[(row['protocol'], row['horizon'], stratum)][state_key] = row
    return groups


def create_report(evaluations, *, repetitions=1000, seed=20261011):
    signature = validate_pairing(evaluations)
    aggregates, differences, calibration, risk = [], [], [], []
    by_seed = {item['training_seed']: item for item in evaluations if item['variant'] == 'graph_fm'}
    for evaluation in evaluations:
        common = dict(run=evaluation['label'], variant=evaluation['variant'], training_seed=evaluation['training_seed'])
        for predictor in sorted(PREDICTORS):
            for (protocol, horizon, stratum), selected in partitions(evaluation, predictor).items():
                info = dict(common, predictor=predictor, protocol=protocol, horizon=horizon, stratum=stratum)
                for metric in METRICS:
                    aggregates.append(dict(info, metric=metric,
                        **demand_bootstrap(selected.values(), metric, repetitions=repetitions, seed=seed)))
                if predictor == 'predictor':
                    for feature in ('count', 'speed'):
                        for scope in ('all', 'occupied'):
                            for level in (50, 80, 90, 95):
                                coverage = demand_bootstrap(selected.values(), f'{feature}_coverage{level}_{scope}', repetitions=repetitions, seed=seed)
                                width = demand_bootstrap(selected.values(), f'{feature}_width{level}_{scope}', repetitions=repetitions, seed=seed)
                                calibration.append(dict(info, feature=feature, scope=scope, nominal=level / 100,
                                    coverage=coverage['estimate'], coverage_ci_low=coverage['ci_low'], coverage_ci_high=coverage['ci_high'],
                                    width=width['estimate'], demand_groups=coverage['demand_groups']))
                    references = [('persistence', partitions(evaluation, 'persistence')[(protocol, horizon, stratum)])]
                    graph = by_seed.get(evaluation['training_seed'])
                    if graph is not None and graph is not evaluation:
                        references.append(('graph_fm', partitions(graph, 'predictor')[(protocol, horizon, stratum)]))
                    for reference, reference_rows in references:
                        for metric in METRICS:
                            differences.append(dict(info, reference=reference, metric=metric,
                                **paired_metric(selected, reference_rows, metric, repetitions=repetitions, seed=seed)))
                    if horizon == 1:
                        if not all(row['disagreement_independent_samples'] for row in selected.values()):
                            raise ValueError('risk report requires independent disagreement ensembles at h1')
                        for metric in ('count_mae_occupied', 'speed_mae_occupied'):
                            analysis = disagreement_analysis(list(selected.values()), error_key=metric,
                                congestion_edges=evaluation['protocol']['scales']['congestion_edges'],
                                group_keys=('demand_group',), n_bootstrap=min(repetitions, 300), seed=seed)
                            risk.append(dict(info, metric=metric, analysis=analysis))
    seeds = {item['variant']: sorted({str(e['training_seed']) for e in evaluations if e['variant'] == item['variant']}) for item in evaluations}
    return dict(status='complete', role='development_screen', comparison_complete=False,
        report_scope='single_checkpoint_diagnostic' if len(evaluations) == 1 else 'paired_development_checkpoint_comparison',
        explanation='Component campaign completion is not claimed. Training seeds remain separate; all intervals are conditional on the individual checkpoint.',
        interval='95% percentile bootstrap of demand groups; equal demands, episodes, then states; no training-seed uncertainty interval.',
        difference_sign='candidate minus reference; negative is better for all reported error/score metrics.',
        independent_training_seeds=seeds, repetitions=repetitions, seed=seed, compatibility=signature,
        sources=[{name: item[name] for name in ('directory', 'label', 'variant', 'training_seed', 'hashes')} for item in evaluations],
        missing_horizon_equivalence=[dict(run=item['label'], **complete_horizon_proof(item)) for item in evaluations],
        metrics=aggregates, paired_differences=differences, calibration=calibration, risk=risk)


def write_csv(path, records):
    if not records:
        Path(path).write_text('')
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def render(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(figure, name):
        for extension in ('png', 'pdf'):
            figure.savefig(output / f'{name}.{extension}', dpi=160)
        plt.close(figure)
    for stratum in sorted({row['stratum'] for row in report['metrics']}):
        figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        plotted = ['count_mae_all', 'count_mae_occupied', 'speed_mae_all', 'speed_mae_occupied', 'count_crps_occupied', 'energy_score']
        for axis, metric in zip(axes.flat, plotted):
            rows = [r for r in report['metrics'] if r['stratum'] == stratum and r['metric'] == metric]
            groups = sorted({(r['run'], r['predictor'], r['protocol']) for r in rows})
            for run, predictor, protocol in groups:
                values = sorted([r for r in rows if (r['run'], r['predictor'], r['protocol']) == (run, predictor, protocol)
                                 and r['estimate'] is not None], key=lambda r:r['horizon'])
                if not values:
                    continue
                label = f"{run}:{predictor}:{'recursive' if 'recursive' in protocol else '1step'}"
                line, = axis.plot([r['horizon'] for r in values], [r['estimate'] for r in values], 'o-', label=label, markersize=3)
                if all(r['ci_low'] is not None for r in values):
                    axis.fill_between([r['horizon'] for r in values], [r['ci_low'] for r in values],
                                      [r['ci_high'] for r in values], color=line.get_color(), alpha=.10)
            axis.set(title=metric, xlabel='Forecast horizon (30 s windows)')
            axis.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', fontsize=6, ncol=2)
        figure.suptitle(f'Development screening, fixed checkpoints — {stratum}')
        save(figure, f'prediction_errors_{stratum}')
        for protocol, horizon in sorted({(r['protocol'], r['horizon']) for r in report['calibration'] if r['stratum'] == stratum}):
            figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            for axis, (feature, scope) in zip(axes.flat, [('count','all'),('count','occupied'),('speed','all'),('speed','occupied')]):
                rows = [r for r in report['calibration'] if (r['stratum'],r['protocol'],r['horizon'],r['feature'],r['scope']) == (stratum,protocol,horizon,feature,scope)]
                for run in sorted({r['run'] for r in rows}):
                    values = sorted([r for r in rows if r['run']==run and r['coverage'] is not None], key=lambda r:r['nominal'])
                    line, = axis.plot([r['nominal'] for r in values], [r['coverage'] for r in values], 'o-', label=run)
                    if values and all(r['coverage_ci_low'] is not None for r in values):
                        axis.fill_between([r['nominal'] for r in values], [r['coverage_ci_low'] for r in values],
                                          [r['coverage_ci_high'] for r in values], color=line.get_color(), alpha=.15)
                axis.plot([0,1],[0,1],':',color='gray')
                axis.set(title=f'{feature} {scope}',xlabel='Nominal coverage',ylabel='Observed coverage',xlim=(0,1),ylim=(0,1))
                axis.grid(alpha=.2)
            axes[0,0].legend(fontsize=6)
            figure.suptitle(f'{stratum}, {protocol}, h={horizon}; widths in calibration.csv')
            save(figure, f'calibration_{stratum}_{protocol}_h{horizon}')
    for item in report['risk']:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        all_risk = item['analysis']['all']
        for axis, analysis, title in [(axes[0], all_risk, 'All congestion'),
                                      (axes[1], None, 'Congestion strata')]:
            analyses = [(analysis, 'all')] if analysis is not None else [(r, f"stratum {r['stratum']}") for r in item['analysis']['congestion_strata']]
            for result, label in analyses:
                bins = result['deciles']
                axis.plot([r['disagreement_mean'] for r in bins], [r['error']['estimate'] for r in bins], 'o-', label=label)
            axis.set(title=title,xlabel='Mean disagreement in risk bin',ylabel=item['metric'])
            axis.grid(alpha=.2)
            if analyses:
                axis.legend(fontsize=7)
        figure.suptitle(f"{item['run']} {item['protocol']} {item['stratum']}; descriptive Spearman={all_risk['spearman_descriptive_windows']}", fontsize=8)
        save(figure, f"risk_{item['run']}_{item['protocol']}_{item['stratum']}_{item['metric']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261011)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    evaluations = [read_evaluation(directory) for directory in args.evaluations]
    report = create_report(evaluations, repetitions=args.bootstrap, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'prediction_metrics.csv', report['metrics'])
    write_csv(args.output/'paired_differences.csv', report['paired_differences'])
    write_csv(args.output/'calibration.csv', report['calibration'])
    risk_rows = []
    for item in report['risk']:
        for bucket in item['analysis']['all']['deciles']:
            risk_rows.append({name:item[name] for name in ('run','variant','training_seed','protocol','horizon','stratum','metric')} |
                dict(bin=bucket['bin'],states=bucket['states'],disagreement=bucket['disagreement_mean'],
                     error=bucket['error']['estimate'],ci_low=bucket['error']['ci_low'],ci_high=bucket['error']['ci_high']))
    write_csv(args.output/'risk_bins.csv', risk_rows)
    render(report,args.output)
    # Recheck immutable input identities before declaring the report complete.
    for source in evaluations:
        if any(file_hash(Path(source['directory'])/name) != digest for name,digest in source['hashes'].items()):
            raise ValueError('input evaluation changed while generating report')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',scope=report['report_scope'],evaluations=len(evaluations),output=str(args.output))))


if __name__ == '__main__':
    main()
