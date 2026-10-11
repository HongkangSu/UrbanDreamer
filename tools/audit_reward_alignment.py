"""Read-only reward audit: actual logged trends and exact two-action diagnostics.

The analytic cases isolate objective behavior, not city-scale causal estimates.
No checkpoint, simulator or training process is modified.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from urban_dreamer.models.policy import individual_actor_advantage
from urban_dreamer.learning.training import ACTOR_ENTROPY_COEF


def audit(log: Path):
    buckets = {}
    for line in log.open():
        row = json.loads(line)
        if row.get('phase') != 'actor_critic_pass_finished':
            continue
        bucket = (row['episode_index'] // 50) * 50
        buckets.setdefault(bucket, []).append(row)
    trends = []
    fields = ('counterfactual_baseline_mean', 'global_advantage_mean',
              'global_advantage_std', 'actor_advantage_mean', 'actor_advantage_std',
              'event_critic_loss', 'imagined_events')
    for bucket, rows in sorted(buckets.items()):
        trends.append(dict(episode_index_start=bucket, passes=len(rows),
            zero_actor_advantage_std_passes=sum(r['actor_advantage_std'] < 1e-5 for r in rows),
            mean_global_return=statistics.mean(r['counterfactual_baseline_mean'] +
                r['global_advantage_mean'] for r in rows),
            **{key: statistics.mean(r[key] for r in rows) for key in fields}))
    # Exact expected score-function gradient wrt fast-action logit for independent
    # actions; social mean travel-time objective divides each vehicle benefit by N.
    entropy = []
    p = .9
    for seconds in (10., 30., 60., 120.):
        delta = seconds / (1000. * 60.)
        task_gradient = p * (1 - p) * delta
        entropy_gradient = -ACTOR_ENTROPY_COEF * p * (1 - p) * math.log(p / (1 - p))
        entropy.append(dict(population=1000, saved_vehicle_seconds=seconds,
            fast_action_probability=p, mean_minutes_reward_difference=delta,
            task_gradient=task_gradient, entropy_gradient=entropy_gradient,
            total_gradient=task_gradient + entropy_gradient,
            optimum_fast_action_probability=1 / (1 + math.exp(-delta / ACTOR_ENTROPY_COEF))))
    # Holding scale=1 via this two-event vector exposes the clipping problem.
    # Returns differ by 1 minute. An action-independent baseline should not alter
    # the expected policy-gradient reward contrast before nonlinear clipping.
    clipping = []
    returns = torch.tensor([-1., -2.])
    for baseline in (0., 3., -5.):
        advantages = individual_actor_advantage(returns - baseline, torch.zeros(2))
        clipping.append(dict(baseline=baseline, raw_returns=returns.tolist(),
            advantages=advantages.tolist(), fast_minus_slow=float(advantages[0] - advantages[1])))
    assert clipping[0]['fast_minus_slow'] == 1.
    assert clipping[1]['fast_minus_slow'] == clipping[2]['fast_minus_slow'] == 0.
    assert all(row['total_gradient'] < 0 for row in entropy[:3])
    return dict(source_log=str(log), log_trends=trends, exact_entropy_examples=entropy,
        exact_clipping_examples=clipping,
        caveats=['Analytic entropy examples assume one vehicle travel-time change and no spillovers.',
                 'Logged global return is a sampled truncated bootstrapped objective, not full episode return.',
                 'Seed cohort excludes future births; horizon is 240 s; timeout signal exists only at real deadline.',
                 'Observed late-phase advantage standard deviation refutes blanket clipping saturation as late root cause.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, default=ROOT / 'outputs/ingolstadt_online_libsumo_20261004/pilot_metrics.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/convergence_diagnosis_20261005/reward/audit.json')
    args = parser.parse_args()
    result = audit(args.log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
