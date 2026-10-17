"""Independent decoded-bank, grouped-score, budget and selection arithmetic."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('outputs/nonlinear_prior_fm_20261014/v1')
summary = json.loads((root / 'summary.json').read_text())
rows = json.loads((root / 'test/states.json').read_text())
split = json.loads((root / 'split.json').read_text())
assert summary['status'] == 'complete' and len(rows) == 128
assert len(split['test_conditions']) == 32
checks = 0
for entry in json.loads((root / 'test/bank_manifest.json').read_text()):
    path = Path(entry['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    with np.load(path) as bank:
        truth = bank['target'].astype(np.float64).reshape(-1, 2)
        scales = bank['scales'].astype(np.float64)
        selected = [r for r in rows if r['condition_index'] == entry['condition_index']]
        assert len(selected) == 4
        for row in selected:
            samples = bank[row['arm']].astype(np.float64).reshape(8, *truth.shape)
            assert np.isfinite(samples).all()
            for feature, name in enumerate(('count', 'speed')):
                for scope in ('all', 'occupied'):
                    mask = np.isfinite(truth[:, feature])
                    if scope == 'occupied':
                        mask &= np.isfinite(truth[:, 0]) & (truth[:, 0] > 0)
                    if not mask.any():
                        continue
                    values, target = samples[:, mask, feature], truth[mask, feature]
                    error = values.mean(axis=0) - target
                    crps = np.abs(values - target).mean(axis=0) - .5 * np.abs(values[:, None] - values[None, :]).mean(axis=(0, 1))
                    lo, hi = np.quantile(values, [.05, .95], axis=0)
                    expected = dict(mae=np.abs(error).mean(), mse=np.square(error).mean(),
                                    crps=crps.mean(), coverage90=((target >= lo) & (target <= hi)).mean())
                    for metric, value in expected.items():
                        np.testing.assert_allclose(row[f'{name}_{metric}_{scope}'], value, rtol=1e-10, atol=1e-10)
                        checks += 1
            mask = np.isfinite(truth)
            values, target = (samples / scales)[:, mask], (truth / scales)[mask]
            energy = np.linalg.norm(values - target, axis=1).mean() - .5 * np.linalg.norm(values[:, None] - values[None, :], axis=2).mean()
            np.testing.assert_allclose(row['energy_score'], energy, rtol=1e-10, atol=1e-10)
            totals = samples[..., 0].sum(axis=1)
            mass_error = abs(totals.mean() - truth[:, 0].sum())
            np.testing.assert_allclose(row['total_count_absolute_error'], mass_error, rtol=1e-10, atol=1e-10)
            checks += 2

def clustered(records, metric):
    groups = {}
    for row in records:
        if row[metric] is not None:
            groups.setdefault(row['demand_group'], {}).setdefault(row['episode'], []).append(row[metric])
    return np.array([np.mean([np.mean(states) for states in episodes.values()]) for _, episodes in sorted(groups.items())])

def verify_interval(values, report, square_root=False):
    assert len(values) == report['demand_groups'] == 16
    indices = np.random.default_rng(summary['protocol']['seed']).integers(len(values), size=(2000, len(values)))
    lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
    expected = np.array([values.mean(), lo, hi])
    if square_root:
        expected = np.sqrt(expected)
    np.testing.assert_allclose([report['estimate'], report['ci_low'], report['ci_high']], expected, rtol=1e-10, atol=1e-10)

summary_checks = 0
for arm in summary['test']['arms']:
    selected = [r for r in rows if r['arm'] == arm['arm']]
    assert len(selected) == arm['states'] == 32
    for metric, report in arm['metrics'].items():
        source = metric.replace('_rmse_', '_mse_')
        values = clustered(selected, source)
        if not len(values):
            assert report['estimate'] is None
            continue
        verify_interval(values, report, '_rmse_' in metric)
        summary_checks += 3

pairs = {'prior_effect_after_equal_adapter': ('adapter_soft', 'adapter_zero'),
         'prior_effect_before_adapter': ('old_soft', 'old_zero'),
         'soft_adapter_effect': ('adapter_soft', 'old_soft'),
         'zero_adapter_effect': ('adapter_zero', 'old_zero')}
contrast_checks = 0
for name, (left, right) in pairs.items():
    a = {r['condition_index']: r for r in rows if r['arm'] == left}
    b = {r['condition_index']: r for r in rows if r['arm'] == right}
    assert len(a) == 32 and a.keys() == b.keys()
    for metric, report in summary['test_pairs'][name].items():
        differences = []
        for index, row in a.items():
            assert (row['demand_group'], row['episode']) == (b[index]['demand_group'], b[index]['episode'])
            if row[metric] is None:
                assert b[index][metric] is None
                continue
            differences.append(dict(demand_group=row['demand_group'], episode=row['episode'], delta=row[metric] - b[index][metric]))
        values = clustered(differences, 'delta')
        if len(values):
            verify_interval(values, report)
            contrast_checks += 3
        else:
            assert report['estimate'] is None

history = json.loads((root / 'training_history.json').read_text())
assert [r['epoch'] for r in history] == list(range(1, 81))
assert all(r['updates_per_arm'] == 80 for r in history)
assert summary['updates_per_arm'] == sum(r['updates_per_arm'] for r in history) == 6400
initial = json.loads((root / 'validation/epoch000_summary.json').read_text())
expected_best = {r['arm']: dict(ES=r['metrics']['energy_score']['estimate'], epoch=0) for r in initial['arms']}
for row in history:
    for arm, score in row.get('integrated_validation', {}).items():
        if score < expected_best[arm]['ES'] - 1e-5:
            expected_best[arm] = dict(ES=score, epoch=row['epoch'])
for arm, best in expected_best.items():
    assert summary['best'][arm]['epoch'] == best['epoch']
    np.testing.assert_allclose(summary['best'][arm]['ES'], best['ES'], rtol=1e-12)
assert summary['budget_limited'] and not summary['H4_gate_passed']

alignment = json.loads((root / 'alignment_stats.json').read_text())
alignment_summary = {}
for stage in ('train', 'validation'):
    selected = [r for r in alignment if r['stage'] == stage]
    ratio = sum(r['weak_no_noise_sse'] for r in selected) / sum(r['zero_delta_energy'] for r in selected)
    noisy_ratio = sum(r['weak_noisy_sse'] for r in selected) / sum(r['zero_noisy_sse'] for r in selected)
    alignment_summary[stage] = dict(prior_normalized_sse=ratio, skill_vs_zero_delta=1-ratio, paired_noisy_source_sse_ratio=noisy_ratio)
result = dict(status='pass', rows=len(rows), metric_checks=checks, summary_interval_checks=summary_checks,
              paired_interval_checks=contrast_checks, equal_consumed_epochs=80, equal_consumed_updates=6400,
              validation_only_best_selection_verified=True, expected_best=expected_best,
              budget_limited=True, h4_not_run=True, alignment=alignment_summary)
(root / 'independent_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
