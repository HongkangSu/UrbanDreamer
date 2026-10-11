"""Fit current-city DTA and compare validation-selected and final policies on held-out demands."""
from __future__ import annotations
import argparse
import copy
import functools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from tools.compare_incident_dta_sp import compact, save, aggregate
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.traffic.routing import validate_prepared_inputs


def timing_summary(rows):
    complete = [r for r in rows if r['unfinished'] == 0]
    def summarize(values):
        return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                    minimum=min(values), maximum=max(values)) if values else None
    return dict(full_completion_episodes=len(complete), episodes=len(rows),
        collection_including_archive_seconds=summarize([r['collection_seconds'] for r in complete]),
        archive_seconds=summarize([r['archive_seconds'] for r in complete]),
        simulated_seconds=summarize([r['simulation_end_seconds'] for r in complete]),
        all_episode_collection_seconds=summarize([r['collection_seconds'] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--city', choices=('coulommiers','montereau_fault_yonne'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol_path = out/('shortest_protocol.json' if args.reference_only else 'protocol.json')
    if protocol_path.exists():
        raise ValueError('refusing to overwrite an existing DTA study')
    runargs = argparse.Namespace(config=ROOT/'configs'/f'{args.city}.toml', device='cpu', threads=1,
        seed=42, vehicle_limit=0, method='dta', session=None, backend=None, step_seconds=None,
        forbid_revisit=False, sp_prior_weight=0., global_reward_only=True, reward_objective='observed_mean')
    started = time.monotonic(); runner = EpisodeRunner(runargs)
    assert runner.config.simulation.demand_duration_seconds == 1800
    assert runner.config.simulation.incident_start_seconds > runner.deadline
    expected = {'coulommiers':1811, 'montereau_fault_yonne':1500}[args.city]
    assert runner.prepared.vehicle_count == expected
    method = DtaMethod(runargs, runner)
    validation_seeds = [31000,31001,31002]
    test_seeds = list(range(51000,51010))
    checkpoints = sorted({0,1,5,10,20,30,40,args.iterations} & set(range(args.iterations+1)))
    protocol = dict(city=args.city, iterations=args.iterations, prior=0, reward_mode='global_only', reward_objective='observed_mean',
        expected_vehicles=expected, deadline_seconds=runner.deadline, config=runner.config.resolved_snapshot(),
        input_hashes=runner.prepared.input_hashes, training_seeds=list(range(43,43+args.iterations)),
        validation_seeds=validation_seeds, test_seeds=test_seeds, checkpoints=checkpoints,
        selection='fewest validation nonarrivals, then lowest aggregate observed exposure per vehicle',
        archive='Every training and test episode archived; validation is not archived.',
        timing='Single CPU thread. Collection includes assignment preparation, SUMO, selector, window records, and archive writing. Model initialization and DTA update timed separately.',
        travel_time='Actual departure to arrival. If incomplete, exposure is censored and is not full mean travel time.',
        algorithm='Existing approximate BPR marginal-cost/MSA DTA-SO; no claim of certified system optimum.')
    save(protocol_path,protocol)
    training, validations, best_key, best_state, best_iteration = [],[],None,None,None

    def progress(**fields):
        payload = dict(city=args.city, elapsed_seconds=time.monotonic()-started, **fields)
        save(out/('reference_progress.json' if args.reference_only else 'progress.json'),payload)
        print(json.dumps(payload),flush=True)

    def evaluate(policy, seeds, archive=False):
        rows = []; iteration = method.iteration
        for seed in seeds:
            selector = 'shortest' if policy=='shortest' else functools.partial(method.select,explore=False)
            archive_args = dict(archive_path=out/'episodes'/policy/f'episode_{seed:06d}.npz',
                archive_metadata=dict(phase='test',episode=seed,assignment_iteration=iteration)) if archive else {}
            _,_,metrics = runner.run_episode(selector, seed=seed, label=policy,
                record_windows=archive, on_demand=None if policy=='shortest' else method.begin_episode, **archive_args)
            assert method.iteration == iteration
            rows.append(compact(metrics))
            save(out/f'{policy}_current.json',rows)
            progress(stage=policy,iteration=iteration,seed=seed,completion=metrics['completion_rate'],
                     collection_seconds=metrics['collection_seconds'])
        return rows

    if args.reference_only:
        shortest = evaluate('shortest',test_seeds,archive=True)
        save(out/'shortest_test.json',shortest)
        progress(stage='reference_completed')
        return

    for iteration in range(args.iterations+1):
        if iteration:
            rows,windows,metrics = runner.run_episode(functools.partial(method.select,explore=False),
                seed=42+iteration,label='dta_so',record_windows=True,on_demand=method.begin_episode,
                archive_path=out/'episodes'/'train'/f'episode_{iteration:06d}.npz',
                archive_metadata=dict(phase='train',episode=iteration,assignment_iteration=iteration-1))
            tick = time.monotonic(); update = method.on_episode(rows,windows,episode=iteration)
            update['learning_seconds'] = time.monotonic()-tick
            del rows,windows
            training.append(dict(iteration=iteration,metrics=compact(metrics),update=update))
            save(out/'training.json',training)
            temporary=out/'latest.tmp'; torch.save(method.state_dict(),temporary); temporary.replace(out/'latest.pt')
            progress(stage='training',iteration=iteration,completion=metrics['completion_rate'],
                     collection_seconds=metrics['collection_seconds'],update_seconds=update['learning_seconds'])
        if iteration in checkpoints:
            records = evaluate('validation',validation_seeds)
            summary = aggregate(records)
            key = (summary['unfinished'],summary['observed_minutes'])
            if best_key is None or key < best_key:
                best_key=key;best_state=copy.deepcopy(method.state_dict());best_iteration=iteration
                torch.save(best_state,out/'selected_assignment.pt')
            validations.append(dict(iteration=iteration,summary=summary,rows=records,best_iteration=best_iteration))
            save(out/'validation.json',validations)
            torch.save(method.state_dict(),out/f'assignment_{iteration:04d}.pt')
    final_state=copy.deepcopy(method.state_dict())
    if (out/'shortest_protocol.json').exists():
        reference_protocol=json.loads((out/'shortest_protocol.json').read_text())
        assert reference_protocol['input_hashes']==protocol['input_hashes']
        while not (out/'shortest_test.json').exists():
            progress(stage='waiting_for_reference',iteration=args.iterations)
            time.sleep(10)
        shortest=json.loads((out/'shortest_test.json').read_text())
        assert [r['seed'] for r in shortest]==test_seeds
    else:
        shortest = evaluate('shortest',test_seeds,archive=True)
        save(out/'shortest_test.json',shortest)
    method.load_state_dict(best_state)
    selected = evaluate('dta_selected',test_seeds,archive=True); save(out/'dta_selected_test.json',selected)
    if best_iteration == args.iterations:
        final = selected
    else:
        method.load_state_dict(final_state)
        final = evaluate('dta_final',test_seeds,archive=True)
    save(out/'dta_final_test.json',final)
    for group in (selected,final):
        for a,b in zip(shortest,group,strict=True):
            assert a['seed']==b['seed'] and a['vehicle_count']==b['vehicle_count'] and a['demand_sha256']==b['demand_sha256']
    assert validate_prepared_inputs(runner.config).input_hashes == protocol['input_hashes']
    summaries = {name:dict(aggregate(records),timing=timing_summary(records))
                 for name,records in [('shortest',shortest),('dta_selected',selected),('dta_final',final)]}
    for name in ('dta_selected','dta_final'):
        base=summaries['shortest'];value=summaries[name]
        value['completion_gain_pp']=100*(value['completion']-base['completion'])
        value['observed_exposure_reduction_percent']=100*(1-value['observed_minutes']/base['observed_minutes'])
        value['full_travel_time_reduction_percent']=(value['observed_exposure_reduction_percent']
            if base['unfinished']==value['unfinished']==0 else None)
    result=dict(city=args.city,trained_iterations=args.iterations,selected_iteration=best_iteration,
        paired_test_demands=len(test_seeds),summaries=summaries,
        training_full_completion_timing=timing_summary([r['metrics'] for r in training]),
        elapsed_seconds=time.monotonic()-started,all_test_demands_paired=True,input_hashes_unchanged=True)
    save(out/'results.json',result); progress(stage='completed',selected_iteration=best_iteration,result=result)


if __name__=='__main__': main()
