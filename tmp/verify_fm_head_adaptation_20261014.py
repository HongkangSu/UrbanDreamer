"""Independent arithmetic checks on saved decoded forecasts and fitted heads."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('outputs/fm_head_adaptation_20261014/v1')
rows = [json.loads(line) for line in (ROOT / 'states.jsonl').read_text().splitlines()]
checked = 0
for entry in json.loads((ROOT / 'bank_manifest.json').read_text()):
    path = Path(entry['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
    with np.load(path) as bank:
        truth = bank['target'].astype(np.float64)
        if truth.ndim == 3:
            truth = truth[0]
        scales = bank['scales'].astype(np.float64)
        for row in (r for r in rows if r['condition_index'] == entry['condition_index']):
            samples = bank[row['arm']].astype(np.float64)
            if samples.ndim == 4:
                samples = samples[:, 0]
            assert samples.shape == (8, *truth.shape)
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
                        checked += 1
            mask = np.isfinite(truth)
            values = (samples / scales)[:, mask]
            target = (truth / scales)[mask]
            energy = np.linalg.norm(values - target, axis=1).mean() - .5 * np.linalg.norm(values[:, None] - values[None, :], axis=2).mean()
            np.testing.assert_allclose(row['energy_score'], energy, rtol=1e-10, atol=1e-10)
            checked += 1
fits = json.loads((ROOT / 'fit_diagnostics.json').read_text())['fits']
head_checks = {}
with np.load(ROOT / 'heads.npz') as heads, np.load(ROOT / 'training_statistics.npz') as stats:
    anchor = heads['anchor']
    for name, fit in fits.items():
        head = heads['head_' + name]
        gram = stats['rank_' + name + '__gram']
        cross = stats['rank_' + name + '__cross']
        energy = float(stats['rank_' + name + '__energy'])
        ridge = 1e-4 * np.trace(gram) / len(gram)
        np.testing.assert_allclose(fit['ridge'], ridge, rtol=1e-13)
        residual = (gram + np.eye(len(gram)) * ridge) @ head - (cross + ridge * anchor)
        relative = np.linalg.norm(residual) / np.linalg.norm(cross + ridge * anchor)
        assert relative < 1e-12
        loss = lambda w: energy - 2 * np.sum(w * cross) + np.sum(w * (gram @ w))
        old, new = loss(anchor), loss(head)
        np.testing.assert_allclose([fit['original_train_sse'], fit['adapted_train_sse']], [old, new], rtol=1e-10)
        assert new + ridge * np.square(head - anchor).sum() <= old + 1e-6
        head_checks[name] = dict(normal_equation_relative_residual=float(relative), original_train_sse=float(old), adapted_train_sse=float(new))
summary_checks = 0
summary_path = ROOT / 'summary.json'
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'complete' and len(rows) == 16 * 9
    for arm in summary['arms']:
        selected = [r for r in rows if r['arm'] == arm['arm']]
        assert len(selected) == 16 and arm['states'] == 16
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
            assert len(values) == 8 and report['demand_groups'] == 8
            indices = np.random.default_rng(summary['protocol']['bootstrap_seed']).integers(len(values), size=(2000, len(values)))
            lo, hi = np.quantile(values[indices].mean(axis=1), [.025, .975])
            expected = np.array([values.mean(), lo, hi])
            if '_rmse_' in metric:
                expected = np.sqrt(expected)
            np.testing.assert_allclose([report['estimate'], report['ci_low'], report['ci_high']], expected, rtol=1e-10, atol=1e-10)
            summary_checks += 3
result = dict(status='pass',rows=len(rows),independent_metric_checks=checked,independent_summary_checks=summary_checks,head_checks=head_checks)
(ROOT / 'independent_arithmetic_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
