"""Independent meso incident ablation with validation-selected DTA and paired test demands."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
from dataclasses import replace
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.simulation_settings import physics_identity
from urban_dreamer.workflow.config import load_config


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def compact(metrics):
    return {key:value for key,value in metrics.items() if key not in {'choices','branch_choices'}}


def aggregate(rows):
    n = sum(row['vehicle_count'] for row in rows)
    failed = sum(row['unfinished'] for row in rows)
    exposure = sum(row['exposure_seconds'] for row in rows)
    return dict(episodes=len(rows), vehicles=n, unfinished=failed,
        completion=1-failed/n, observed_minutes=exposure/(60*n),
        total_vehicle_hours=exposure/3600,
        mean_travel_minutes=exposure/(60*n) if failed==0 else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/ingolstadt_custom.toml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario',choices=('off','on'),required=True)
    parser.add_argument('--iterations',type=int,default=50)
    args=parser.parse_args()
    if args.iterations<1:parser.error('positive iterations required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(args.config)
    sim=cfg.simulation
    if args.scenario=='off':
        sim=replace(sim,incident_start_seconds=1_000_000.,incident_end_seconds=1_001_000.)
    cfg=replace(cfg,simulation=sim,run=replace(cfg.run,output_dir=out,device='cpu'),
        data=replace(cfg.data,archive_roots=(),validation_archive_roots=(),
                     pretrain_cache_manifest=None,validation_cache_manifest=None))
    text='\n\n'.join('['+section+']\n'+'\n'.join(k+' = '+json.dumps(v)
        for k,v in values.items() if v is not None) for section,values in cfg.resolved_snapshot().items())
    config_path=out/'config.toml';config_path.write_text(text+'\n')
    cfg=load_config(config_path)
    runner_args=argparse.Namespace(config=config_path,device='cpu',threads=1,seed=42,
        vehicle_limit=0,method='dta_so',session=None,backend=None,step_seconds=None,
        forbid_revisit=False,sp_prior_weight=0.)
    # Check every physical window, not merely the configured interval.
    class AuditedSession(SumoEventSession):
        def advance_window(self, callback):
            result=super().advance_window(callback)
            if args.scenario=='off' and self._incident_active:
                raise AssertionError('incident activated in no-incident scenario')
            return result
    runner=EpisodeRunner(runner_args,session_factory=AuditedSession)
    method=DtaMethod(runner_args,runner)
    validation_seeds=[20000,20001,20002]
    test_seeds=list(range(30000,30010))
    checkpoints=sorted({0,1,10,25,args.iterations}&set(range(args.iterations+1)))
    protocol=dict(scenario=args.scenario,settings=cfg.resolved_snapshot(),
        physics_identity=physics_identity(sim),input_hashes=runner.prepared.input_hashes,
        training_indices=list(range(43,43+args.iterations)),validation_indices=validation_seeds,
        test_indices=test_seeds,demand_seed=42,deadline_seconds=runner.deadline,
        checkpoints=checkpoints,selection='fewest validation nonarrivals, then lowest exposure/person',
        description='Existing approximate BPR marginal-cost/MSA DTA; no known future demand input. Both methods use event fork-only execution, including skipped short forks. Not a certified system optimum.')
    save(out/'protocol.json',protocol)
    frozen_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),ROOT/'urban_dreamer/learning/baselines/dta.py',
            ROOT/'urban_dreamer/learning/baselines/common.py',ROOT/'urban_dreamer/traffic/event_session.py',
            ROOT/'urban_dreamer/traffic/topology.py']}
    save(out/'source_hashes.json',frozen_sources)
    started=time.monotonic();training=[];validations=[];best_key=None;best_state=None;best_iteration=None
    def evaluate(policy,seeds):
        result=[];before=method.iteration
        for seed in seeds:
            selected='shortest' if policy=='shortest' else functools.partial(method.select,explore=False)
            _,_,metrics=runner.run_episode(selected,seed=seed,label=policy,
                on_demand=method.begin_episode if policy!='shortest' else None)
            result.append(compact(metrics))
        assert method.iteration==before
        return result
    validation_sp=evaluate('shortest',validation_seeds)
    save(out/'validation_shortest.json',validation_sp)
    for iteration in range(args.iterations+1):
        if iteration:
            rows,windows,metrics=runner.run_episode(functools.partial(method.select,explore=False),
                seed=42+iteration,label='dta_so',record_windows=True,on_demand=method.begin_episode)
            update=method.on_episode(rows,windows,episode=iteration)
            del rows,windows
            training.append(dict(iteration=iteration,metrics=compact(metrics),update=update))
            save(out/'training.json',training)
        if iteration in checkpoints:
            rows=evaluate('dta_so',validation_seeds)
            for a,b in zip(validation_sp,rows):assert a['demand_sha256']==b['demand_sha256']
            summary=aggregate(rows);key=(summary['unfinished'],summary['observed_minutes'])
            if best_key is None or key<best_key:
                best_key=key;best_state=copy.deepcopy(method.state_dict());best_iteration=iteration
            validations.append(dict(iteration=iteration,summary=summary,rows=rows))
            save(out/'validation.json',validations)
            torch.save(method.state_dict(),out/f'assignment_{iteration:04d}.pt')
            print(json.dumps(dict(stage='validation',iteration=iteration,best_iteration=best_iteration,
                summary=summary,elapsed=time.monotonic()-started)),flush=True)
    method.load_state_dict(best_state)
    torch.save(best_state,out/'selected_assignment.pt')
    sp=evaluate('shortest',test_seeds);dta=evaluate('dta_so',test_seeds)
    for a,b in zip(sp,dta):
        assert a['demand_sha256']==b['demand_sha256'] and a['vehicle_count']==b['vehicle_count']
    paired=[dict(seed=a['seed'],population=a['vehicle_count'],demand_sha256=a['demand_sha256'],
        shortest_minutes=a['mean_exposure_seconds']/60,dta_minutes=b['mean_exposure_seconds']/60,
        saving_seconds=(a['mean_exposure_seconds']-b['mean_exposure_seconds']),
        shortest_unfinished=a['unfinished'],dta_unfinished=b['unfinished']) for a,b in zip(sp,dta)]
    sa,da=aggregate(sp),aggregate(dta)
    rng=np.random.default_rng(917);boot=[]
    for ids in rng.integers(0,len(paired),(10000,len(paired))):
        x=sum(sp[i]['exposure_seconds'] for i in ids);y=sum(dta[i]['exposure_seconds'] for i in ids)
        boot.append(100*(x-y)/x)
    result=dict(scenario=args.scenario,selected_iteration=best_iteration,training_iterations=args.iterations,
        shortest=sa,dta=da,relative_exposure_saving_percent=100*(sa['observed_minutes']-da['observed_minutes'])/sa['observed_minutes'],
        paired_demand_bootstrap95_percentile=np.percentile(boot,[2.5,97.5]).tolist(),
        saving_seconds_per_vehicle=60*(sa['observed_minutes']-da['observed_minutes']),
        paired=paired,shortest_rows=sp,dta_rows=dta,validation=validations,
        seconds=time.monotonic()-started,
        limitations=['Bootstrap measures test-demand variation for one fitted assignment, not optimization-seed uncertainty.',
            'DTA is approximate MSA/BPR with at most8paths/OD and event-executed next hops; failure to beat SP does not prove no routing headroom.',
            'Exposure is full travel time only when every vehicle arrives. Selection used validation only; test was not used for fitting or selection.'])
    save(out/'results.json',result)
    print(json.dumps({k:result[k] for k in ('scenario','selected_iteration','shortest','dta',
        'relative_exposure_saving_percent','paired_demand_bootstrap95_percentile','seconds')}),flush=True)


if __name__=='__main__':main()
