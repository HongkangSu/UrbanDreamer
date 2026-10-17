"""Compare imagination budgets on the same tiny physical routing replay."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import torch
from test_route_model_based_baselines import collect_toy_episode, make_method

torch.set_num_threads(1)
results = []
for name in ('dreamer', 'mbpo'):
    for budget in ('previous', 'reduced'):
        samples = []
        for repeat in range(3):
            method = make_method(name)
            method.deadline = method.runner.deadline = 300.
            if name == 'dreamer':
                settings = dict(dreamer_world_model_updates=8 if budget == 'previous' else 4,
                    dreamer_imagination_passes=4 if budget == 'previous' else 1,
                    dreamer_imagination_rollouts=4 if budget == 'previous' else 2,
                    dreamer_imagined_horizon=8 if budget == 'previous' else 3)
            else:
                settings = dict(mbpo_model_train_rounds=4 if budget == 'previous' else 2,
                    mbpo_rollout_seeds=16 if budget == 'previous' else 4,
                    mbpo_rollout_length=2 if budget == 'previous' else 1)
            for key, value in settings.items():
                setattr(method.args, key, value)
            rows, windows = collect_toy_episode(method)
            started = time.monotonic()
            metrics = method.on_episode(rows, windows, episode=1)
            samples.append(dict(elapsed_seconds=time.monotonic() - started,
                imagination_seconds=metrics['imagination_seconds'],
                world_model_seconds=metrics['world_model_seconds'],
                actor_update_seconds=metrics['actor_update_seconds'],
                imagined_events=metrics['imagined_events']))
        results.append(dict(method=name, budget=budget, settings=settings, samples=samples,
            median_seconds=statistics.median(s['elapsed_seconds'] for s in samples)))
report = dict(scope='CPU tiny physical-route replay; current repaired code under both budgets; not city throughput',
    repeats=3, results=results)
path = ROOT / 'outputs/baseline_stability_20261015/budget_benchmark.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
