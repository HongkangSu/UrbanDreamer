"""Bounded no-prior AC collection benchmark with phase timings and replay digest."""
from __future__ import annotations
import argparse
import cProfile
import hashlib
import importlib.util
import json
from pathlib import Path
import pstats
import sys
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--city',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--direct-vehicle-probe',action='store_true')
    parser.add_argument('--policy',choices=('stochastic','shortest'),default='stochastic')
    parser.add_argument('--episode-index',type=int,default=43)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--forbid-revisit',action='store_true')
    parser.add_argument('--meso-multi-queue',action='store_true')
    parser.add_argument('--demand-scale',type=float,default=1.)
    args=parser.parse_args()
    source=args.source.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('city_profile_trainer',source/'tools/train_toy_actor_critic.py')
    trainer=importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)
    import numpy as np
    import libsumo
    from urban_dreamer.traffic.event_session import SumoEventSession
    if args.direct_vehicle_probe:
        # Diagnostic only: isolate libsumo subscription wrapper overhead while
        # retaining the same three real fields on every one-second observation.
        import traci.constants as tc
        original_observe=SumoEventSession._observe
        libsumo.vehicle.subscribe=lambda *a,**kw: None
        def observe_direct(self,*a,**kw):
            def results():
                arrived=set(libsumo.simulation.getArrivedIDList())
                return {vid:{tc.VAR_ROAD_ID:libsumo.vehicle.getRoadID(vid),
                             tc.VAR_LANEPOSITION:libsumo.vehicle.getLanePosition(vid),
                             tc.VAR_ROUTE_INDEX:libsumo.vehicle.getRouteIndex(vid)}
                        for vid in sorted(self._active) if vid not in arrived}
            libsumo.vehicle.getAllSubscriptionResults=results
            return original_observe(self,*a,**kw)
        SumoEventSession._observe=observe_direct
    phases={}
    def timed(name, function):
        def call(*a, **kw):
            tick=time.monotonic()
            try:
                return function(*a, **kw)
            finally:
                phases[name]=phases.get(name,0.)+time.monotonic()-tick
        return call
    libsumo.simulationStep=timed('sumo_step_seconds',libsumo.simulationStep)
    for name in ('reset','_observe','_apply_decisions','_flow'):
        setattr(SumoEventSession,name,timed(name+'_seconds',getattr(SumoEventSession,name)))
    settings=SimpleNamespace(config=source/'configs'/f'{args.city}.toml',device='cpu',threads=1,
        seed=42,vehicle_limit=0,session='event',backend='libsumo',step_seconds=1.,
        forbid_revisit=args.forbid_revisit,global_reward_only=False,sp_prior_weight=0.,sp_prior_temperature=30.,
        sp_prior_hold_episodes=200,sp_prior_off_episode=1000,collection_device=None,
        actor_lr=2e-4,global_critic_lr=3e-4)
    started=time.monotonic(); runner=trainer.RealActorCritic(settings); setup=time.monotonic()-started
    if args.meso_multi_queue:
        from dataclasses import replace
        runner.config=replace(runner.config,simulation=replace(runner.config.simulation,meso_multi_queue=True))
        runner.simulation_settings=trainer.resolve_collection_settings(runner.config,settings)
    if args.demand_scale != 1.:
        from urban_dreamer.traffic.demand import PoissonDemand
        runner.demand=PoissonDemand(tuple((o,d,rate*args.demand_scale) for o,d,rate in runner.demand.od_rates),
                                   runner.demand.duration_seconds)
    profiler=cProfile.Profile(); results=[]
    for episode in range(args.episodes):
        phases.clear()
        if args.profile: profiler.enable()
        rows,metrics=runner.run_episode(policy=args.policy,seed=args.episode_index,
            collect=args.policy=='stochastic',archive_path=out/f'episode_{episode+1}.npz',
            archive_metadata=dict(phase='performance_probe',episode=episode+1))
        if args.profile: profiler.disable()
        digest=hashlib.sha256()
        for row in rows:
            for key in ('features','mask'):
                digest.update(np.ascontiguousarray(row[key]).tobytes())
            digest.update(str((row['vehicle_id'],row['time'],row['action'],row['window'],row.get('timeout_return'))).encode())
        digest.update(json.dumps({k:metrics[k] for k in ('choices','vehicle_count','completion_rate','unfinished',
            'exposure_seconds','global_reward_sum','vehicle_timeout_penalty_sum','microsteps','event_count')},sort_keys=True).encode())
        update=runner.update(rows) if args.policy=='stochastic' else {}
        results.append(dict(episode=episode+1,metrics={k:v for k,v in metrics.items() if k not in ('choices','branch_choices')},
                            learning=update,event_rows_sha256=digest.hexdigest(),
                            collection_phases=dict(phases),
                            archive_bytes=(out/f'episode_{episode+1}.npz').stat().st_size))
        print(json.dumps(dict(city=args.city,episode=episode+1,collection_seconds=metrics['collection_seconds'],
                             learning_seconds=update.get('learning_seconds'),digest=digest.hexdigest())),flush=True)
    report=dict(source=str(source),city=args.city,policy=args.policy,prior=0.,threads=1,setup_seconds=setup,
                network_edges=runner.network.edge_count,expected_vehicles=runner.demand.expected_vehicle_count,
                profiled=args.profile,direct_vehicle_probe=args.direct_vehicle_probe,
                forbid_revisit=args.forbid_revisit,meso_multi_queue=args.meso_multi_queue,
                demand_scale=args.demand_scale,results=results)
    if args.profile:
        profiler.dump_stats(out/'collection.prof')
        stats=pstats.Stats(profiler)
        report['profile_top_cumulative']=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],
                self_seconds=v[2],cumulative_seconds=v[3])
            for k,v in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:60]]
        with (out/'profile.txt').open('w') as stream:
            pstats.Stats(profiler,stream=stream).sort_stats('cumulative').print_stats(80)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
