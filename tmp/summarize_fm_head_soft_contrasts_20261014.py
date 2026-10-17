"""Prespecified soft-versus-hard contrasts, using paired saved forecasts."""
import json
from pathlib import Path

import numpy as np

root = Path('outputs/fm_head_adaptation_20261014/v1')
protocol = json.loads((root / 'supplementary_contrasts_protocol.json').read_text())
rows = [json.loads(line) for line in (root / 'states.jsonl').read_text().splitlines()]
report = []
for left, right in protocol['pairs']:
    a = {r['condition_index']: r for r in rows if r['arm'] == left}
    b = {r['condition_index']: r for r in rows if r['arm'] == right}
    assert len(a) == 16 and a.keys() == b.keys()
    metrics = {}
    for metric in protocol['metrics']:
        grouped = {}
        for index, row in a.items():
            other = b[index]
            assert (row['demand_group'], row['episode']) == (other['demand_group'], other['episode'])
            if row[metric] is None:
                assert other[metric] is None
                continue
            grouped.setdefault(row['demand_group'], {}).setdefault(row['episode'], []).append(row[metric] - other[metric])
        values = np.array([np.mean([np.mean(states) for states in episodes.values()]) for _, episodes in sorted(grouped.items())])
        indices = np.random.default_rng(protocol['bootstrap_seed']).integers(len(values), size=(protocol['bootstrap_repetitions'], len(values)))
        low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
        metrics[metric] = dict(estimate=float(values.mean()), ci_low=float(low), ci_high=float(high), demand_groups=len(values))
    report.append(dict(arm=left, reference=right, states=len(a), metrics=metrics))
result = dict(status='complete', protocol=protocol, pairs=report)
(root / 'supplementary_soft_hard_contrasts.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
