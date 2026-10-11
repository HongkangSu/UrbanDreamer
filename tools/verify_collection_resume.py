"""Produce a real Toy checkpoint with a specified frozen legacy source."""
import argparse
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--global-reward-only', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('legacy_baseline_trainer', source/'tools/train_toy_baseline.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    from urban_dreamer.learning.baselines import common, build_method
    import torch
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = argparse.Namespace(method=args.method, config=args.config.resolve(), output_dir=output,
        episodes=1, eval_every=100, eval_seeds=1, seed=42, device='cuda:0', threads=1,
        actor_lr=2e-4, critic_lr=3e-4, vehicle_limit=4, session='event', backend='libsumo',
        step_seconds=1., resume=False, sp_prior_weight=1., sp_prior_temperature=30.,
        sp_prior_hold_episodes=200, sp_prior_off_episode=1000, save_episodes=True,
        checkpoint_every=1, skip_initial_evaluation=True, skip_reference_baselines=True,
        forbid_revisit=False, global_reward_only=args.global_reward_only)
    runner = common.EpisodeRunner(settings)
    method = build_method(args.method, settings, runner)
    prior = trainer.apply_episode_prior(method, settings, 1)
    rows, windows, metrics = runner.run_episode(functools.partial(method.select, explore=True),
        seed=43, collect=True, label='stochastic', record_windows=getattr(method,'needs_window_records',False),
        row_flow=getattr(method,'needs_row_flow',True), on_demand=getattr(method,'begin_episode',None),
        archive_path=output/'episodes/train/episode_000001.npz',
        archive_metadata=dict(method=args.method,phase='train',episode=1,sp_prior_weight=prior))
    for row in rows: row['sp_prior_weight']=prior
    update = method.on_episode(rows, windows, episode=1)
    trainer.save(runner,method,settings,output/'latest.pt',1,float('inf'))
    shutil.copy2(output/'latest.pt',output/'best.pt')
    (output/'metrics.jsonl').write_text(json.dumps(dict(phase='train',episode=1,**metrics,**update))+'\n')
    digest=hashlib.sha256()
    for key,value in method.actor.state_dict().items():
        digest.update(key.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    (output/'legacy_proof.json').write_text(json.dumps(dict(source=str(source),episode=1,
        actual_N=runner.n, actor_sha256=digest.hexdigest(),reward=metrics['global_reward_sum']),indent=2)+'\n')
    print(json.dumps(dict(method=args.method,source=str(source),actual_N=runner.n,checkpoint=str(output/'latest.pt'))))


if __name__=='__main__':
    main()
