"""Independent arithmetic and grouped-score checks for the prior split."""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path('outputs/prior_source_reference_20261014/v1')
rows = [json.loads(line) for line in (root / 'states.jsonl').read_text().splitlines()]
summary = json.loads((root / 'summary.json').read_text())
assert summary['status'] == 'complete' and len(rows) == 128
checks = 0
for entry in json.loads((root / 'bank_manifest.json').read_text()):
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
            values = (samples / scales)[:, mask]
            target = (truth / scales)[mask]
            energy = np.linalg.norm(values - target, axis=1).mean() - .5 * np.linalg.norm(values[:, None] - values[None, :], axis=2).mean()
            np.testing.assert_allclose(row['energy_score'], energy, rtol=1e-10, atol=1e-10)
            checks += 1
summary_checks = 0
for arm in summary['arms']:
    selected = [r for r in rows if r['arm'] == arm['arm']]
    assert len(selected) == 32 and arm['states'] == 32
    for metric, report in arm['metrics'].items():
        source_metric = metric.replace('_rmse_', '_mse_')
        grouped = {}
        for row in selected:
            if row.get(source_metric) is not None:
                grouped.setdefault(row['demand_group'], {}).setdefault(row['episode'], []).append(row[source_metric])
        if not grouped:
            assert report['estimate'] is None
            continue
        values = np.array([np.mean([np.mean(states) for states in episodes.values()]) for _, episodes in sorted(grouped.items())])
        assert len(values) == 16 and report['demand_groups'] == 16
        indices = np.random.default_rng(summary['protocol']['bootstrap_seed']).integers(len(values), size=(2000, len(values)))
        lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
        expected = np.array([values.mean(), lo, hi])
        if '_rmse_' in metric:
            expected = np.sqrt(expected)
        np.testing.assert_allclose([report['estimate'], report['ci_low'], report['ci_high']], expected, rtol=1e-10, atol=1e-10)
        summary_checks += 3
contrast_checks = 0
indexed = {arm: {r['condition_index']: r for r in rows if r['arm'] == arm}
           for arm in ('native', 'source_only', 'reference_only', 'zero_mean')}
for contrast in summary['mechanism_contrasts']:
    weights = ({contrast['arm']: 1., contrast['reference']: -1.} if 'arm' in contrast else
               {'native': 1., 'source_only': -1., 'reference_only': -1., 'zero_mean': 1.})
    assert all(len(indexed[arm]) == 32 for arm in weights)
    for metric, report in contrast['metrics'].items():
        grouped = {}
        for index, row in indexed['native'].items():
            values = [indexed[arm][index][metric] for arm in weights]
            if all(v is None for v in values):
                continue
            assert all(v is not None for v in values)
            value = sum(w * indexed[arm][index][metric] for arm, w in weights.items())
            grouped.setdefault(row['demand_group'], {}).setdefault(row['episode'], []).append(value)
        if not grouped:
            assert report['estimate'] is None
            continue
        values = np.array([np.mean([np.mean(states) for states in episodes.values()]) for _, episodes in sorted(grouped.items())])
        assert len(values) == report['demand_groups'] == 16
        indices = np.random.default_rng(summary['protocol']['bootstrap_seed']).integers(len(values), size=(2000, len(values)))
        lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
        np.testing.assert_allclose([report['estimate'], report['ci_low'], report['ci_high']], [values.mean(), lo, hi], rtol=1e-10, atol=1e-10)
        contrast_checks += 3
result = dict(status='pass', rows=len(rows), independent_metric_checks=checks,
              independent_summary_checks=summary_checks, independent_contrast_checks=contrast_checks)
(root / 'independent_arithmetic_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
