"""Untrained shortest-path controls on the baseline evaluation and holdout seeds."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_city_baseline_dataset import atomic_json
from urban_dreamer.learning.baselines.common import EpisodeRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--policy', choices=('shortest','dynamic_shortest'), default='shortest')
    args = parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    settings=argparse.Namespace(config=args.config,device='cpu',threads=1,seed=42,
        vehicle_limit=0,method=args.policy,session=None,backend=None,step_seconds=None,
        forbid_revisit=False,sp_prior_weight=0.,global_reward_only=True,reward_objective='observed_mean')
    runner=EpisodeRunner(settings)
    assert runner.config.simulation.incident_start_seconds>runner.deadline
    seeds=[10042,10043,10044,20042,20043,20044]
    path=out/'rows.json';rows=json.loads(path.read_text()) if path.exists() else []
    assert [r['seed'] for r in rows]==seeds[:len(rows)]
    for seed in seeds[len(rows):]:
        _,_,metrics=runner.run_episode(args.policy,seed=seed,label=args.policy)
        rows.append({k:v for k,v in metrics.items() if k not in {'choices','branch_choices'}})
        atomic_json(path,rows)
        print(json.dumps(dict(seed=seed,completion=metrics['completion_rate'])),flush=True)
    atomic_json(out/'results.json',dict(method=args.policy,training_iterations=0,
        reward_mode='global_only',reward_objective='observed_mean',input_hashes=runner.prepared.input_hashes,
        config=runner.config.resolved_snapshot(),rows=rows))


if __name__=='__main__':main()
