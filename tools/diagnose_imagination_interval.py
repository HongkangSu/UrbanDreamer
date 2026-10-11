"""CPU-only factual oracle refresh audit; never treats a 30s WM as a 1s model."""
from __future__ import annotations
import argparse, copy, gzip, json, pickle, sys, time
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology, NextHopCandidates
from urban_dreamer.traffic.demand import PoissonDemand, demand_identity
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.models.policy import NextHopActor
from urban_dreamer.learning.observations import build_actor_observations
from urban_dreamer.learning.sumo_sampling import SumoEpisodeSampler
from urban_dreamer.learning.imagination import advance_imagined_window

class RecordingSession(SumoEventSession):
    def _observe(self, counts):
        super()._observe(counts)
        self.record()
    def record(self):
        now=int(self._time_ms/1000)
        self.frames[now]=self._flow().astype(np.float32).copy()
        active=[dict(self._vehicles[k]) for k in sorted(self._active)]
        self.positions[now]={v['vehicle_id']:(int(v['current_edge']),float(v['remaining_distance']),bool(v['in_junction'])) for v in active}
        if now%30==0:self.seeds[now]=active
        self.arrivals.update({k:float(v['arrival_time']) for k,v in self._vehicles.items() if v['arrived']})

def destination_only_audit(record, candidates, max_speed):
    """No turns or route fallbacks: isolate within-edge motion error."""
    network = candidates.network
    results = []
    for refresh in (30, 10, 5, 1):
        population = actual = predicted = 0
        errors = []
        for start, seed in record['seeds'].items():
            if start + 30 not in record['positions']:
                continue
            states = copy.deepcopy([v for v in seed
                if v['current_edge'] == v['destination_edge'] and not v['in_junction']])
            population += len(states)
            actual += sum(record['arrivals'].get(v['vehicle_id'], float('inf')) <= start + 30
                          for v in states)
            arrivals = {}
            for now in range(start, start + 30, refresh):
                flow = record['frames'][now]
                speeds = np.minimum(np.clip(flow[:, 1], .05, 1.5)
                                    * network.speed_limits_mps, max_speed)
                result = advance_imagined_window(states, candidates=candidates,
                    edge_speeds_mps=speeds, frozen_flow=flow, start_time=now,
                    total_vehicle_count=record['population'], decide=lambda *args: [],
                    interval_seconds=refresh, observation_step_seconds=1.,
                    session_protocol='event', vehicle_max_speed_mps=max_speed)
                arrivals.update(result.arrivals)
            predicted += len(arrivals)
            errors.extend(timestamp - record['arrivals'][key]
                          for key, timestamp in arrivals.items() if key in record['arrivals'])
        results.append(dict(refresh=refresh, vehicle_seeds=population,
            actual_arrivals=actual, predicted_arrivals=predicted,
            arrival_time_bias=float(np.mean(errors)),
            arrival_time_mae=float(np.mean(np.abs(errors)))))
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=3810);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
    cfg=load_config(a.config);prepared=validate_prepared_inputs(cfg);topology=load_traffic_topology(prepared)
    raw=torch.load(a.checkpoint,map_location='cpu',weights_only=False);actor=NextHopActor().eval();actor.load_state_dict(raw['training_state']['models']['actor']);completed=raw['next_unit'];del raw
    demand=PoissonDemand.from_vehicles(topology.vehicles,duration_seconds=cfg.simulation.demand_duration_seconds)
    topology=replace(topology,vehicles=demand.sample(seed=cfg.method.evaluation_demand_seed,episode_index=0));n=len(topology.vehicles);net=topology.network;candidates=NextHopCandidates(net)
    cfg=replace(cfg,run=replace(cfg.run,seed=cfg.method.evaluation_demand_seed));session=RecordingSession(cfg,prepared,topology,0,backend='libsumo',step_seconds=1.)
    session.frames={};session.positions={};session.seeds={};session.arrivals={};events=[]
    def policy(pending,flow,now):
        obs=build_actor_observations(policy_edge_features=flow[:,[2,1]],vehicle_states=pending,candidates=candidates,episode_progress=now/a.limit,active_vehicle_fraction=session.active_vehicle_count/max(n,1))
        with torch.no_grad():indices=actor.sample_action(torch.from_numpy(obs.features),torch.from_numpy(obs.mask),deterministic=True).candidate_index.tolist()
        return [NextHopDecision(v['vehicle_id'],int(v['current_edge']),int(obs.edge_indices[i,indices[i]]),now) for i,v in enumerate(pending)]
    begin=time.monotonic()
    with SumoEpisodeSampler(session=session,candidates=candidates,deadline_seconds=a.limit,total_vehicle_count=n,context_mode='observation',sort_pending=True,pad_empty=False,advance_empty=True) as sampler:
        session.record()
        for w in sampler.windows(lambda pending,obs,flow,now,index,context:policy(pending,flow,now)):
            events.extend(asdict(e) for e in w.result.decisions)
    record=dict(frames=session.frames,positions=session.positions,seeds=session.seeds,arrivals=session.arrivals,events=events,demand=[asdict(v) for v in topology.vehicles],population=n,checkpoint_completed=completed,checkpoint=str(a.checkpoint),demand_hash=demand_identity(topology.vehicles))
    with gzip.open(a.output/'oracle_record.pkl.gz','wb',compresslevel=1) as f:pickle.dump(record,f,protocol=5)
    print(json.dumps(dict(stage='collection_complete',seconds=time.monotonic()-begin,frames=len(session.frames),seeds=len(session.seeds),events=len(events),arrived=len(session.arrivals))),flush=True)
    rows=[]
    # Same start cohort and factual next-turn sequence in all refresh arms. No births enter.
    for start,seed in session.seeds.items():
        if not seed or start+30 not in session.positions:continue
        factual={}
        for e in events:
            if start<=e['decision_time']<start+120:factual.setdefault(e['vehicle_id'],[]).append(e)
        ids={v['vehicle_id'] for v in seed};real_arrived=sum(session.arrivals.get(k,float('inf'))<=start+30 for k in ids)
        real_cost=sum(sum(k in session.positions[t] for k in ids) for t in range(start,start+30))
        for refresh in (30,10,5,1):
          for floor in (.05,0.):
            states=copy.deepcopy(seed);cursor={};fallback=[0];cost=0.;arrived={}
            def replay(pending,flow,now):
                out=[]
                for v in pending:
                    choices=factual.get(v['vehicle_id'],[]);old=cursor.get(v['vehicle_id'],0);hit=next((i for i in range(old,len(choices)) if choices[i]['current_edge']==v['current_edge']),None)
                    if hit is None:
                        fallback[0]+=1;dest=int(v['destination_edge']);legal=candidates.candidates(int(v['current_edge']),dest);target=min(legal,key=lambda edge:net.free_flow_travel_seconds[edge]+candidates.costs(dest)[edge])
                    else:target=choices[hit]['next_edge'];cursor[v['vehicle_id']]=hit+1
                    out.append(NextHopDecision(v['vehicle_id'],int(v['current_edge']),int(target),now))
                return out
            for now in range(start,start+30,refresh):
                flow=session.frames[now];speeds=np.minimum(np.clip(flow[:,1],floor,1.5)*net.speed_limits_mps,cfg.simulation.vehicle_max_speed_mps)
                result=advance_imagined_window(states,candidates=candidates,edge_speeds_mps=speeds,frozen_flow=flow,start_time=now,total_vehicle_count=n,decide=replay,interval_seconds=refresh,observation_step_seconds=1.,session_protocol='event',vehicle_max_speed_mps=cfg.simulation.vehicle_max_speed_mps)
                cost+=result.active_vehicle_seconds;arrived.update(result.arrivals)
            real=session.positions[start+30];same=0;distance=[];pred_active=0
            for v in states:
                if not v['arrived']:
                    pred_active+=1;truth=real.get(v['vehicle_id'])
                    if truth and truth[0]==v['current_edge'] and truth[2]==v['in_junction']:
                        same+=1;distance.append(abs(truth[1]-v['remaining_distance']))
            rows.append(dict(start=start,refresh=refresh,speed_floor=floor,cohort=len(ids),actual_arrivals=real_arrived,predicted_arrivals=len(arrived),actual_cost=real_cost,predicted_cost=cost,same_edge=same,actual_active=len(ids)-real_arrived,predicted_active=pred_active,same_edge_distance_error_sum=sum(distance),same_edge_distance_count=len(distance),fallback_actions=fallback[0]))
    summary=[]
    for refresh in (30,10,5,1):
      for floor in (.05,0.):
        selected=[r for r in rows if r['refresh']==refresh and r['speed_floor']==floor];total=lambda k:sum(r[k] for r in selected)
        summary.append(dict(refresh_seconds=refresh,speed_floor=floor,windows=len(selected),cohort_total=total('cohort'),actual_arrivals=total('actual_arrivals'),predicted_arrivals=total('predicted_arrivals'),actual_vehicle_seconds=total('actual_cost'),predicted_vehicle_seconds=total('predicted_cost'),cost_relative_bias=(total('predicted_cost')/max(total('actual_cost'),1)-1),same_edge_fraction=total('same_edge')/max(total('actual_active'),1),same_edge_distance_mae=total('same_edge_distance_error_sum')/max(total('same_edge_distance_count'),1),fallback_actions=total('fallback_actions')))
    payload=dict(protocol='Factual future edge-speed oracle; same observed start cohorts, deterministic real actor, factual turn replay with shortest-path fallback. No WM prediction. Future traffic is an optimistic diagnostic oracle, not deployable dynamics.',checkpoint_completed=completed,population=n,arrived=len(session.arrivals),seconds=time.monotonic()-begin,summary=summary,rows=rows)
    (a.output/'results.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
    destination = destination_only_audit(record, candidates, cfg.simulation.vehicle_max_speed_mps)
    (a.output/'destination_only.json').write_text(json.dumps(destination, indent=2) + '\n')
if __name__=='__main__':main()
