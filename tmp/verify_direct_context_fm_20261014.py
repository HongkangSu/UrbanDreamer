"""Independent bank/statistic/selection verifier; never imports model code.

Run only after the study completes. The validation-selection guard is checked
before opening any test results. NumPy implements scores/bootstrap independently;
Torch is used only to inspect the small serialized selected-head checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

NEW = ('zero_direct', 'soft_direct', 'condition_only')
ALL = NEW + ('old_zero', 'old_soft')
CP_SHA = '2f63f0b2a5fe18f3a3bd9be27ffd1e63a2755464e7f2944e1227334f0ffe6af6'
COUNTERS = dict(bank_metric_checks=0, interval_checks=0, files_hashed=0)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    COUNTERS['files_hashed'] += 1
    return h.hexdigest()


def archive_sha(path):
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        names = sorted(z.namelist())
        assert len(names) == len(set(names)) and all(n.endswith('.npy') for n in names)
        for name in names:
            h.update(name.encode() + b'\0')
            with z.open(name) as f:
                while block := f.read(1024 * 1024):
                    h.update(block)
            h.update(b'\0')
    COUNTERS['files_hashed'] += 1
    return h.hexdigest()


def close(actual, expected, label=''):
    if expected is None:
        assert actual is None, label
    else:
        assert actual is not None, label
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9, err_msg=label)


def clusters(rows, metric):
    groups = {}
    for r in rows:
        if r.get(metric) is not None:
            groups.setdefault(r['demand_group'], {}).setdefault(r['episode'], []).append(r[metric])
    return np.array([np.mean([np.mean(v) for v in episodes.values()])
                     for _, episodes in sorted(groups.items())], dtype=np.float64)


def interval(values, report, seed, root=False):
    assert report['demand_groups'] == len(values)
    if not len(values):
        for key in ('estimate', 'ci_low', 'ci_high'):
            assert report[key] is None
        return
    point = values.mean()
    close(report['estimate'], np.sqrt(point) if root else point)
    if len(values) == 1:
        assert report['ci_low'] is None and report['ci_high'] is None
    else:
        ix = np.random.default_rng(seed).integers(len(values), size=(2000, len(values)))
        bounds = np.quantile(values[ix].mean(1), [.025, .975])
        if root:
            bounds = np.sqrt(bounds)
        close([report['ci_low'], report['ci_high']], bounds)
    COUNTERS['interval_checks'] += 1


def bank_scores(samples, truth, scales, count):
    k = len(samples)
    valid = np.isfinite(truth)
    occupied = valid[:, 0] & (truth[:, 0] > 0)
    out = dict(draws=k, roads=len(truth), occupied_roads=int(occupied.sum()))
    for feature, name in enumerate(('count', 'speed')):
        for scope in ('all', 'occupied'):
            mask = valid[:, feature] & (occupied if scope == 'occupied' else True)
            prefix, suffix = name + '_', '_' + scope
            out[prefix + 'valid_roads' + suffix] = int(mask.sum())
            keys = ('mae', 'mse', 'rmse', 'sample_mae', 'crps', 'crps_u', 'sample_variance',
                    'coverage90', 'width90', 'below_lower_fraction')
            if not mask.any():
                out.update({prefix + key + suffix: None for key in keys})
                continue
            x, y = samples[:, mask, feature], truth[mask, feature]
            first = np.abs(x - y).mean(0)
            half = .5 * np.abs(x[:, None] - x[None, :]).mean((0, 1))
            error = x.mean(0) - y
            lo, hi = np.quantile(x, [.05, .95], axis=0)
            values = dict(mae=np.abs(error).mean(), mse=np.square(error).mean(),
                          rmse=np.sqrt(np.square(error).mean()), sample_mae=first.mean(),
                          crps=np.maximum(first - half, 0).mean(),
                          crps_u=(first - half * k / (k - 1)).mean(),
                          sample_variance=x.var(0).mean(), coverage90=((y >= lo) & (y <= hi)).mean(),
                          width90=(hi - lo).mean(), below_lower_fraction=(x < 0).mean())
            out.update({prefix + key + suffix: float(v) for key, v in values.items()})
    x, y = (samples / scales)[:, valid], (truth / scales)[valid]
    first = np.linalg.norm(x - y, axis=1).mean()
    pair_sum = sum(np.linalg.norm(x - draw, axis=1).sum() for draw in x)
    out['energy_score'] = float(max(first - pair_sum / (2 * k * k), 0))
    out['energy_score_u'] = float(first - pair_sum / (2 * k * (k - 1)))
    totals = samples[..., 0].sum(1)
    if valid[:, 0].all():
        true_total = truth[:, 0].sum()
        out.update(true_total_count=float(true_total), predicted_total_count=float(totals.mean()),
                   total_count_bias=float(totals.mean() - true_total),
                   total_count_absolute_error=float(abs(totals.mean() - true_total)),
                   total_count_sample_absolute_error=float(np.abs(totals - true_total).mean()),
                   total_count_sample_variance=float(totals.var()),
                   total_count_over_cohort_fraction=float((totals > count + 1e-6).mean()),
                   total_count_max_cohort_excess=float(max(0, totals.max() - count)))
    return out


def physical_count(item, horizon):
    with np.load(item['cache_path'], allow_pickle=False) as a:
        start = item['start']
        first, last = a['boundary_offsets'][start:start + 2]
        cohort = a['active_vehicle'][first:last]
        left, right = a['boundary_offsets'][start + horizon:start + horizon + 2]
        external = np.isin(a['active_vehicle'][left:right], cohort) & ~a['active_in_junction'][left:right]
        count = np.bincount(a['active_current_edge'][left:right][external], minlength=a['flow'].shape[1])
    return len(cohort), count


def verify_banks(items, rows, banks, arms, samples, steps, horizon=1):
    by_id = {r['condition_index']: r for r in items}
    assert len(banks) == len(items) and len(rows) == len(items) * len(arms)
    for ordinal, entry in enumerate(banks):
        index = entry.get('condition_index', items[ordinal]['condition_index'])
        item = by_id[index]
        assert sha(entry['path']) == entry['sha256']
        count, actual_count = physical_count(item, horizon)
        selected = [r for r in rows if r['condition_index'] == index]
        assert {r['arm'] for r in selected} == set(arms)
        with np.load(entry['path']) as bank:
            truth = bank['target'].astype(np.float64).reshape(-1, 2)
            scales = bank['scales'].astype(np.float64)
            assert np.isfinite(scales).all() and (scales > 0).all()
            np.testing.assert_allclose(truth[:, 0], actual_count, rtol=0, atol=2e-5)
            for row in selected:
                assert row['steps'] == steps and row['horizon'] == horizon and row['k'] == samples
                assert row['demand_group'] == item['demand_group']
                draws = bank[row['arm']].astype(np.float64).reshape(samples, len(truth), 2)
                assert np.isfinite(draws).all()
                expected = bank_scores(draws, truth, scales, count)
                for metric, value in expected.items():
                    if metric in row:
                        close(row[metric], value, f'{index}/{row["arm"]}/{metric}')
                        COUNTERS['bank_metric_checks'] += 1


def verify_summary(rows, report, seed):
    assert {r['arm'] for r in report['arms']} == {r['arm'] for r in rows}
    for arm in report['arms']:
        selected = [r for r in rows if r['arm'] == arm['arm']]
        assert arm['states'] == len(selected)
        for metric, stats in arm['metrics'].items():
            interval(clusters(selected, metric.replace('_rmse_', '_mse_')), stats, seed, '_rmse_' in metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('outputs/direct_context_fm_20261014/v2_cpu4'))
    args = parser.parse_args()
    study = args.study.resolve()
    # No test artifacts, metrics, or banks are opened above this guard.
    assert (study / 'validation_selection.json').exists(), 'Selection not frozen: do not read blind test.'
    selection = read(study / 'validation_selection.json')
    assert selection['status'] == 'frozen_before_new_blindtest'
    summary = read(study / 'summary.json')
    assert summary['status'] == 'complete', 'Run at completion only.'
    protocol, split, launch = [read(study / p) for p in ('protocol.json', 'split.json', 'launch.json')]
    seed, steps = protocol['seed'], protocol['integration_steps']
    assert protocol['device'] == 'cpu' and steps == 4 and protocol['screening'] is True
    argv = launch['argv']
    options = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i].startswith('--')}
    assert options['--device'] == 'cpu' and int(options['--steps']) == steps and int(options['--threads']) == 4
    parent, experiment = Path(options['--parent']), Path(options['--experiment'])
    expected_refs = dict(checkpoint=experiment / 'snapshot/checkpoints/latest.pt', split=Path(options['--split']),
        cache_manifest=Path(split['cache_manifest_path']), parent_features=parent / 'feature_manifest.json',
        parent_draws=parent / 'feature_draw_schedule.json', parent_source=parent / 'source_manifest.json',
        weak_head=Path(options['--weak-head']), basis=Path(options['--basis']),
        known_banks=Path(options['--known-reference']) / 'bank_manifest.json',
        known_conditions=Path(options['--known-reference']) / 'conditions.json', known_cache=Path(options['--known-cache']),
        scales=Path(options['--scale-reference']), old_zero_head=parent / 'checkpoints/best_zero_mean.pt',
        old_soft_head=parent / 'checkpoints/best_weak_soft.pt')
    assert set(expected_refs) == set(protocol['reference_hashes'])
    for name, path in expected_refs.items():
        assert sha(path) == protocol['reference_hashes'][name], name
    assert protocol['reference_hashes']['checkpoint'] == CP_SHA
    assert sha(study / 'source/tools/train_direct_context_fm.py') == launch['runner_sha256']
    for name, digest in read(study / 'source_manifest.json').items():
        assert sha(study / 'source' / name) == digest, name
    for value in read(study / 'import_audit.json').values():
        assert Path(value).resolve().is_relative_to(study / 'source')
    assert all(summary[k] for k in ('original_model_unchanged', 'parent_features_unchanged', 'source_unchanged', 'data_unchanged'))
    assert summary['original_model_digest'] == read(parent / 'summary.json')['original_model_digest']
    parent_split = read(parent / 'split.json')
    assert split['train'] == parent_split['train'] and split['validation'] == parent_split['validation']
    strip = lambda items: [{k: v for k, v in item.items() if k != 'episode_index'} for item in items]
    assert strip(split['validation_conditions']) == strip(parent_split['validation_conditions'])
    groups = [{e['demand_group'] for e in split[k]} for k in ('train', 'validation', 'test')]
    assert tuple(map(len, groups)) == (64, 16, 16)
    assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
    assert not groups[2] & {e['demand_group'] for e in parent_split['test']}
    for stage in ('train', 'validation', 'test'):
        for entry in split[stage]:
            assert archive_sha(entry['cache_path']) == entry['cache_content_hash']
    gates = read(study / 'initial_gates.json')
    assert len(gates) == 6 and {g['arm'] for g in gates} == set(NEW)
    assert all(g['zero_head_exact_old'] and g['steps'] == 4 and g['device'] == 'cpu'
               and not g['archived_bit_identity_checked'] for g in gates)
    compatibility = read(study / 'parent_cache_compatibility.json')
    assert {(g['family'], g['field']) for g in compatibility} == {(a, b) for a in ('zero_mean', 'weak_soft') for b in ('x', 'y')}
    feature_manifest = read(parent / 'feature_manifest.json')
    for g in compatibility:
        assert g['rtol'] == g['atol'] == 1e-4 and g['rows'] == 12 * 853
        assert np.isfinite(g['max_absolute_error'])
        ref = np.load(feature_manifest['train'][g['family']][g['field']], mmap_mode='r')[:g['rows']]
        # A necessary bound from the actual prespecified elementwise gate;
        # the fail-closed CPU forward itself is not repeated by this verifier.
        assert g['max_absolute_error'] <= g['atol'] + g['rtol'] * np.abs(ref).max()
    for arms in feature_manifest.values():
        for entry in arms.values():
            for key in ('x', 'y'):
                assert sha(entry[key]) == entry[key + '_sha256']
    for filename, digest in read(study / 'context_manifest.json').items():
        assert sha(study / 'context' / filename) == digest
    normalizers = read(study / 'normalizers.json')
    assert normalizers['fit_split'] == 'trainonly' and normalizers['centering'] is False
    for key in ('z', 'b', 'relative'):
        close(normalizers['scales'][key], max(np.sqrt(normalizers['sums'][key] / normalizers['counts'][key]), 1e-8))
        close(summary['normalizers'][key], normalizers['scales'][key])

    history = read(study / 'training_history.json')
    n_epochs = len(history)
    assert 20 <= n_epochs <= 120 and [h['epoch'] for h in history] == list(range(1, n_epochs + 1))
    rows_per_epoch = sum(len(e['starts']) for e in split['train']) * 4 * 853
    updates = (rows_per_epoch + 32767) // 32768
    assert all(h['updates_per_arm'] == updates and set(h['train_mse']) == set(NEW)
               and set(h['validation_feature_mse']) == set(NEW) for h in history)
    assert summary['actual_equal_epochs'] == selection['actual_equal_epochs'] == n_epochs
    assert summary['updates_per_arm'] == selection['updates_per_arm'] == n_epochs * updates
    assert summary['only_new_parameters_per_arm'] == 65856
    pilot = read(study / 'epoch_timing_pilot.json')
    assert pilot['full_epoch'] and pilot['continued_exact_state'] and pilot['extra_updates'] == 0 and pilot['epoch'] == 1
    initial = read(study / 'validation/epoch000_summary.json')
    verify_banks(split['validation_conditions'], initial['rows'], initial['banks'], NEW, 4, steps)
    verify_summary(initial['rows'], initial, seed)
    best = {}
    for arm in NEW:
        es = np.mean([r['energy_score'] for r in initial['rows'] if r['arm'] == arm])
        best[arm] = dict(ES=float(es), epoch=0, reset_ES=float(es), reset_epoch=0, patience=0)
    for h in history:
        should_validate = h['epoch'] >= 20 and h['epoch'] % 5 == 0
        assert ('integrated_validation' in h) == should_validate
        if should_validate:
            report = read(study / f'validation/epoch{h["epoch"]:03d}_summary.json')
            verify_banks(split['validation_conditions'], report['rows'], report['banks'], NEW, 4, steps)
            verify_summary(report['rows'], report, seed)
            for arm in NEW:
                es = float(np.mean([r['energy_score'] for r in report['rows'] if r['arm'] == arm]))
                previous = best[arm].copy()
                improved = es < previous['ES'] - 1e-5
                meaningful = es <= previous['reset_ES'] * .998
                if improved:
                    best[arm].update(ES=es, epoch=h['epoch'])
                if meaningful:
                    best[arm].update(reset_ES=best[arm]['ES'], reset_epoch=h['epoch'], patience=0)
                else:
                    best[arm]['patience'] += 1
                logged = h['integrated_validation'][arm]
                close(logged['ES'], es)
                assert logged['best_update'] == improved and logged['meaningful_reset'] == meaningful
                assert logged['selection'] == best[arm]
        if h['epoch'] != n_epochs:
            assert not (h['epoch'] >= 20 and all(b['patience'] >= 3 for b in best.values()))
    assert best == summary['best'] == selection['best']
    expected_reason = 'all_meaningful_ES_patience3' if all(b['patience'] >= 3 for b in best.values()) else 'maximum_epoch_cap'
    assert summary['stop_reason'] == selection['stop_reason'] == expected_reason
    assert summary['budget_limited'] == selection['budget_limited'] == (expected_reason == 'maximum_epoch_cap')
    if expected_reason == 'maximum_epoch_cap':
        assert n_epochs == 120
    import torch  # Serialization only; no model is instantiated or evaluated.
    for arm in NEW:
        checkpoint = torch.load(study / f'checkpoints/best_{arm}.pt', map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == best[arm]['epoch'] and sum(v.numel() for v in checkpoint['state'].values()) == 65856
        close(checkpoint['validation_ES'], best[arm]['ES'])

    # Final-bank access happens only after validation-only selection is verified.
    test_rows = read(study / 'test/states.json')
    verify_banks(split['test_conditions'], test_rows, read(study / 'test/bank_manifest.json'), ALL, 8, steps)
    verify_summary(test_rows, summary['test'], seed)
    pairs = dict(soft_vs_zero=('soft_direct', 'zero_direct'), hint_vs_zero=('condition_only', 'zero_direct'),
                 shift_vs_hint=('soft_direct', 'condition_only'), direct_zero_vs_old=('zero_direct', 'old_zero'),
                 direct_soft_vs_old=('soft_direct', 'old_soft'))
    for name, (left, right) in pairs.items():
        a = {r['condition_index']: r for r in test_rows if r['arm'] == left}
        b = {r['condition_index']: r for r in test_rows if r['arm'] == right}
        assert len(a) == 32 and a.keys() == b.keys()
        for metric, report in summary['test_pairs'][name].items():
            differences = []
            for index, row in a.items():
                other = b[index]
                assert (row['demand_group'], row['episode']) == (other['demand_group'], other['episode'])
                assert (row[metric] is None) == (other[metric] is None)
                differences.append(dict(demand_group=row['demand_group'], episode=row['episode'],
                                        delta=None if row[metric] is None else row[metric] - other[metric]))
            interval(clusters(differences, 'delta'), report, seed)
    eligible = []
    for arm, label in (('soft_direct', 'soft_vs_zero'), ('condition_only', 'hint_vs_zero')):
        p = summary['test_pairs'][label]
        if all(p[m]['ci_high'] is not None and p[m]['ci_high'] < 0 for m in
               ('energy_score', 'count_crps_occupied', 'speed_crps_occupied')) and p['total_count_absolute_error']['estimate'] <= 0:
            eligible.append(arm)
    assert eligible == summary['H4_eligible'] == read(study / 'h4_gate.json')['eligible']
    if eligible:
        report = read(study / 'h4_summary.json')
        verify_banks(split['test_conditions'][:8], report['rows'], report['banks'], ('zero_direct', *eligible), 8, steps, horizon=4)
        verify_summary(report['rows'], report, seed)
    else:
        assert not (study / 'h4_summary.json').exists()
    result = dict(status='pass', independent_numpy_bank_scores=True, independent_cluster_bootstrap=True,
        validation_only_checkpoint_selection=True, equal_actual_training_budgets=True,
        consumed_epochs=n_epochs, consumed_updates_per_arm=n_epochs * updates, best=best,
        device='cpu', integration_steps=steps, screening=True, initial_noop_gates=6,
        parent_cache_compatibility_checked=True, frozen_source_and_checkpoint_verified=True,
        selected_head_parameters=65856, test_states=32, test_groups=16, test_rows=len(test_rows),
        old_head_references='descriptive; different capacity and training budget',
        source_vs_hint_contrast='source AND reference path change; not source-only',
        h4_eligible=eligible, verifier_sha256=sha(__file__), **COUNTERS)
    (study / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
