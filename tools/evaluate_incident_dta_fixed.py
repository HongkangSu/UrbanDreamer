"""Evaluate existing iteration-50 assignments; never fit or select on validation."""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.workflow.config import load_config

BASE = ROOT/'outputs/incident_ablation_dta_sp_20261007'

def main():
    out=BASE/'fixed_iteration50';out.mkdir(exist_ok=True)
    for name in ['no_incident','with_incident']:
        source=BASE/name
        old=json.loads((source/'results.json').read_text())
        cfg=load_config(source/'config.toml')
        protocol=json.loads((source/'protocol.json').read_text())
        current=cfg.resolved_snapshot()
        for section, fields in protocol['settings'].items():
            for key, value in fields.items():
                assert current[section][key]==value, (section,key)
        assert current['method'].get('sp_prior_weight',0.)==0.
        args=argparse.Namespace(config=source/'config.toml',device='cpu',threads=1,seed=42,
            vehicle_limit=0,method='dta_so',session=None,backend=None,step_seconds=None,
            forbid_revisit=False,sp_prior_weight=0.)
        runner=EpisodeRunner(args)
        assert runner.prepared.input_hashes==protocol['input_hashes']
        method=DtaMethod(args,runner)
        checkpoint=source/'assignment_0050.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['iteration']==50
        method.load_state_dict(state)
        if old['selected_iteration']==50:
            rows=old['dta_rows']
            origin='Reused existing exact-iteration-50 evaluation'
        else:
            rows=[]
            for seed in protocol['test_indices']:
                _,_,metrics=runner.run_episode(functools.partial(method.select,explore=False),
                    seed=seed,label='dta_so',on_demand=method.begin_episode)
                rows.append({k:v for k,v in metrics.items() if k not in {'choices','branch_choices'}})
                assert method.iteration==50
                print(json.dumps({'scenario':name,'seed':seed,'minutes':metrics['mean_exposure_seconds']/60}),flush=True)
            origin='Fresh evaluation of saved assignment_0050.pt; no fitting'
        sp=old['shortest_rows']
        assert len(rows)==len(sp)==10
        for a,b in zip(sp,rows,strict=True):
            assert a['seed']==b['seed']
            assert a['demand_sha256']==b['demand_sha256']
            assert a['vehicle_count']==b['vehicle_count']
        def aggregate(rs):
            n=sum(r['vehicle_count'] for r in rs);fail=sum(r['unfinished'] for r in rs)
            exp=sum(r['exposure_seconds'] for r in rs)
            return dict(episodes=len(rs),vehicles=n,unfinished=fail,completion=1-fail/n,
                observed_minutes=exp/(60*n),mean_travel_minutes=exp/(60*n) if not fail else None,total_vehicle_hours=exp/3600)
        sa,da=aggregate(sp),aggregate(rows)
        rng=np.random.default_rng(917);boot=[]
        for ids in rng.integers(0,10,(10000,10)):
            x=sum(sp[i]['exposure_seconds'] for i in ids);y=sum(rows[i]['exposure_seconds'] for i in ids)
            boot.append(100*(x-y)/x)
        pairs=[dict(seed=a['seed'],population=a['vehicle_count'],demand_sha256=a['demand_sha256'],
            shortest_minutes=a['mean_exposure_seconds']/60,dta_minutes=b['mean_exposure_seconds']/60,
            saving_seconds=a['mean_exposure_seconds']-b['mean_exposure_seconds'],
            shortest_unfinished=a['unfinished'],dta_unfinished=b['unfinished']) for a,b in zip(sp,rows,strict=True)]
        result=dict(scenario=old['scenario'],fixed_iteration=50,training_iterations=50,additional_training_iterations=0,
            validation_selection=False,shortest=sa,dta=da,paired=pairs,shortest_rows=sp,dta_rows=rows,
            relative_exposure_saving_percent=100*(sa['observed_minutes']-da['observed_minutes'])/sa['observed_minutes'],
            saving_seconds_per_vehicle=60*(sa['observed_minutes']-da['observed_minutes']),
            paired_demand_bootstrap95_percentile=np.percentile(boot,[2.5,97.5]).tolist(),
            checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            evaluation_origin=origin,limitations=old['limitations'][:2])
        dest=out/name;dest.mkdir(exist_ok=True)
        (dest/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ['paired','shortest_rows','dta_rows']}),flush=True)

if __name__=='__main__':main()
