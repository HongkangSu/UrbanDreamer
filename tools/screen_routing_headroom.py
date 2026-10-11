"""Resumable fixed-iteration DTA screen; held-out tests are optional and separate."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_incident_dta_sp import aggregate, compact, save
import numpy as np
import torch
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.traffic.event_session import SumoEventSession


class NoIncidentSession(SumoEventSession):
    def advance_window(self, callback):
        result = super().advance_window(callback)
        assert not self._incident_active
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.iterations <= 50 or (args.test and args.iterations != 50):
        parser.error('iterations must be1..50; independent test requires iteration50')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    runner_args = argparse.Namespace(config=args.config.resolve(),device='cpu',threads=1,seed=42,
        vehicle_limit=0,method='dta_so',session=None,backend=None,step_seconds=None,
        forbid_revisit=False,sp_prior_weight=0.)
    runner = EpisodeRunner(runner_args, session_factory=NoIncidentSession)
    assert runner.config.simulation.incident_start_seconds > runner.deadline
    assert runner.config.simulation.traffic_model == 'meso'
    method = DtaMethod(runner_args,runner)
    protocol = dict(settings=runner.config.resolved_snapshot(),input_hashes=runner.prepared.input_hashes,
        training_indices=list(range(43,93)),screen_indices=[21000,21001,21002],
        test_indices=list(range(41000,41010)),demand_seed=42,deadline_seconds=runner.deadline,
        selection='Exploratory network screen; final test fixed iteration50, no checkpoint selection.',
        metric='In-network exposure from actual departure; excludes insertion delay; censored if unfinished.')
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Protocol changed during continuation'
    else:
        save(protocol_path,protocol)
    frozen_files = [Path(__file__),ROOT/'urban_dreamer/learning/baselines/dta.py',
        ROOT/'urban_dreamer/learning/baselines/common.py',ROOT/'urban_dreamer/traffic/event_session.py']
    source_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_files}
    source_path=out/'source_hashes.json'
    if source_path.exists():
        assert json.loads(source_path.read_text()) == source_hashes, 'Source changed during continuation'
    else:
        save(source_path,source_hashes)
    latest = out/'latest.pt'; training=[]
    if latest.exists():
        payload=torch.load(latest,map_location='cpu',weights_only=False)
        method.load_state_dict(payload['method']);training=payload['training']
        assert len(training)==method.iteration<=args.iterations

    def evaluate(policy,indices,path):
        rows=json.loads(path.read_text()) if path.exists() else []
        before=method.iteration
        assert len(rows)<=len(indices)
        for seed in indices[len(rows):]:
            selector='shortest' if policy=='shortest' else functools.partial(method.select,explore=False)
            _,_,metrics=runner.run_episode(selector,seed=seed,label=policy,
                on_demand=None if policy=='shortest' else method.begin_episode)
            rows.append(compact(metrics));save(path,rows)
            assert method.iteration==before
            print(json.dumps(dict(stage=policy,seed=seed,iteration=before,
                completion=metrics['completion_rate'],seconds=metrics['mean_exposure_seconds'])),flush=True)
        assert [r['seed'] for r in rows] == indices
        return rows

    screen_sp=evaluate('shortest',protocol['screen_indices'],out/'screen_shortest.json')
    for iteration in range(method.iteration+1,args.iterations+1):
        rows,windows,metrics=runner.run_episode(functools.partial(method.select,explore=False),
            seed=42+iteration,label='dta_so',record_windows=True,on_demand=method.begin_episode)
        update=method.on_episode(rows,windows,episode=iteration)
        del rows,windows
        training.append(dict(iteration=iteration,metrics=compact(metrics),update=update))
        temporary=out/'latest.tmp'
        torch.save(dict(method=method.state_dict(),training=training),temporary);temporary.replace(latest)
        save(out/'training.json',training)
        print(json.dumps(dict(stage='training',iteration=iteration,completion=metrics['completion_rate'],
            seconds=metrics['mean_exposure_seconds'])),flush=True)
    torch.save(method.state_dict(),out/f'assignment_{method.iteration:04d}.pt')
    screen_dta=evaluate('dta_so',protocol['screen_indices'],out/f'screen_dta_{method.iteration:04d}.json')

    def compare(sp,dta):
        assert len(sp)==len(dta)
        for a,b in zip(sp,dta,strict=True):
            assert a['seed']==b['seed'] and a['demand_sha256']==b['demand_sha256']
            assert a['vehicle_count']==b['vehicle_count']
        sa,da=aggregate(sp),aggregate(dta)
        result=dict(iteration=method.iteration,shortest=sa,dta=da,shortest_rows=sp,dta_rows=dta,
            saving_seconds=60*(sa['observed_minutes']-da['observed_minutes']),
            saving_percent=100*(sa['observed_minutes']-da['observed_minutes'])/sa['observed_minutes'],
            full_in_network_travel_times=(sa['unfinished']==da['unfinished']==0))
        if len(sp)>=10:
            boot=[]
            for indices in np.random.default_rng(917).integers(0,len(sp),(10000,len(sp))):
                x=sum(sp[i]['exposure_seconds'] for i in indices);y=sum(dta[i]['exposure_seconds'] for i in indices)
                boot.append(100*(x-y)/x)
            result['paired_demand_bootstrap95']=np.percentile(boot,[2.5,97.5]).tolist()
        return result
    screen=compare(screen_sp,screen_dta);save(out/f'screen_result_{method.iteration:04d}.json',screen)
    print(json.dumps({k:v for k,v in screen.items() if not k.endswith('_rows')}),flush=True)
    if args.test:
        sp=evaluate('shortest',protocol['test_indices'],out/'test_shortest.json')
        dta=evaluate('dta_so',protocol['test_indices'],out/'test_dta_0050.json')
        result=compare(sp,dta);save(out/'test_results.json',result)
        print(json.dumps({k:v for k,v in result.items() if not k.endswith('_rows')}),flush=True)


if __name__=='__main__':main()
