"""Evaluate an explicitly fixed DTA iteration without performing any update."""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--city',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--iteration',type=int,default=5)
    args=parser.parse_args()
    source=args.source.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(source))
    import torch
    from tools.compare_incident_dta_sp import aggregate,compact,save
    from tools.evaluate_current_dta import timing_summary
    from urban_dreamer.learning.baselines.common import EpisodeRunner
    from urban_dreamer.learning.baselines.dta import DtaMethod
    settings=argparse.Namespace(config=source/'configs'/f'{args.city}.toml',device='cpu',threads=1,seed=42,
        vehicle_limit=0,method='dta',session=None,backend=None,step_seconds=None,
        forbid_revisit=False,sp_prior_weight=0.,global_reward_only=False)
    runner=EpisodeRunner(settings);method=DtaMethod(settings,runner)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['iteration']==args.iteration
    method.load_state_dict(state)
    reference=json.loads((args.reference_dir/'shortest_test.json').read_text())
    previous=json.loads((args.reference_dir/'protocol.json').read_text())
    assert previous['input_hashes']==runner.prepared.input_hashes
    assert [r['seed'] for r in reference]==list(range(51000,51010))
    # Training after this checkpoint is historical only and is never used here.
    training=[r for r in json.loads((args.reference_dir/'training.json').read_text()) if r['iteration']<=args.iteration]
    assert len(training)==args.iteration
    save(out/'training_first5.json',training)
    save(out/'shortest_test.json',reference)
    save(out/'protocol.json',dict(city=args.city,iteration=args.iteration,evaluation_only=True,
        additional_training_updates=0,checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source=str(source),input_hashes=runner.prepared.input_hashes,prior=0,
        test_seeds=[r['seed'] for r in reference],reference_dir=str(args.reference_dir.resolve()),
        note='Use the fixed fifth update, not a validation-selected or later checkpoint. Full population,1800-second release,3810-second deadline; archives included in collection timing.'))
    rows=[];started=time.monotonic()
    for ref in reference:
        _,_,metrics=runner.run_episode(functools.partial(method.select,explore=False),seed=ref['seed'],label='dta_so_5',
            record_windows=True,on_demand=method.begin_episode,
            archive_path=out/'episodes'/f'episode_{ref["seed"]}.npz',
            archive_metadata=dict(phase='test',episode=ref['seed'],assignment_iteration=args.iteration))
        assert method.iteration==args.iteration
        assert metrics['demand_sha256']==ref['demand_sha256'] and metrics['vehicle_count']==ref['vehicle_count']
        rows.append(compact(metrics));save(out/'dta_test.json',rows)
        progress=dict(city=args.city,iteration=method.iteration,test_demands_finished=len(rows),
            completion=metrics['completion_rate'],collection_seconds=metrics['collection_seconds'],elapsed_seconds=time.monotonic()-started)
        save(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    sp,so=aggregate(reference),aggregate(rows)
    result=dict(city=args.city,evaluated_iteration=args.iteration,additional_training_updates=0,
        paired_test_demands=len(rows),shortest=dict(sp,timing=timing_summary(reference)),
        dta_so=dict(so,timing=timing_summary(rows)),
        completion_gain_pp=100*(so['completion']-sp['completion']),
        observed_exposure_reduction_percent=100*(1-so['observed_minutes']/sp['observed_minutes']),
        full_travel_time_reduction_percent=(100*(1-so['observed_minutes']/sp['observed_minutes'])
            if sp['unfinished']==so['unfinished']==0 else None),
        training_full_completion_timing=timing_summary([r['metrics'] for r in training]),
        elapsed_seconds=time.monotonic()-started,all_demands_paired=True)
    save(out/'results.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
