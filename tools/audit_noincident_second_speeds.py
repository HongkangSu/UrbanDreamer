"""Replay archived real actions exactly and measure 1s speeds between saved boundaries."""
from pathlib import Path
import argparse,json,sys,time
from dataclasses import replace
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/ingolstadt_online_meso_noincident_noprior_20261007'
# Match the collecting episode's frozen simulator adapter and route rules.
sys.path.insert(0,str(RUN/'source'))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology,VehicleDemand
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.traffic.environment import NextHopDecision
from urban_dreamer.learning.episode_archive import read_episode_archive


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode-index',type=int,required=True);p.add_argument('--end',type=int,default=1380);args=p.parse_args()
 out=ROOT/'outputs/noincident_speed_decomposition_20261008/one_second'/f'episode_{args.episode_index:06d}';out.mkdir(parents=True,exist_ok=False)
 archived=read_episode_archive(RUN/f'episodes/online/episode_{args.episode_index:06d}.npz',expected_session_protocol='event')
 c=load_config(RUN/'run_config.toml');prep=validate_prepared_inputs(c);top=load_traffic_topology(prep);top=replace(top,vehicles=tuple(VehicleDemand(**v) for v in archived['metadata']['realized_demand']))
 assert list(top.network.edge_ids)==archived['metadata']['edge_ids']
 assert c.run.seed==archived['metadata']['seed']
 records={(float(e['decision_time']),str(e['vehicle_id'])):e for e in archived['decisions'] if e['decision_time']<args.end}
 assert len(records)==sum(e['decision_time']<args.end for e in archived['decisions'])
 observed=[];speeds=[];counts=[];boundary=[];used=set()
 class Recording(SumoEventSession):
  def _observe(self,departures):
   super()._observe(departures);self.record()
  def record(self):
   t=int(self._time_ms/1000);f=self._flow();observed.append(t);speeds.append((f[:,1]*self.network.speed_limits_mps).astype(np.float32));counts.append((f[:,2]*self.network.capacities).astype(np.float32))
   if t%30==0:
    expected=archived['flow'][t//30]
    err=float(np.max(np.abs(f-expected)));boundary.append(dict(time=t,max_abs_feature_error=err));assert np.allclose(f,expected,rtol=1e-6,atol=1e-6),(t,err)
 def decide(pending,flow,now):
  result=[]
  for v in pending:
   key=(float(now),str(v['vehicle_id']));e=records[key];assert key not in used;assert int(v['current_edge'])==e['current_edge'];used.add(key)
   result.append(NextHopDecision(key[1],e['current_edge'],e['next_edge'],now))
  return result
 begin=time.monotonic()
 with Recording(c,prep,top,args.episode_index,backend='libsumo',step_seconds=1.) as s:
  s.reset();s.record()
  for i in range(args.end//30):s.advance_window(decide)
 assert used==set(records),(len(used),len(records))
 assert observed==list(range(args.end+1))
 speed=np.stack(speeds);count=np.stack(counts)
 np.savez_compressed(out/'seconds.npz',time=np.asarray(observed),speed_mps=speed,vehicle_count=count,edge_ids=np.asarray(top.network.edge_ids))
 diagnostic=json.load(open(ROOT/'outputs/noincident_decline_diagnosis_20261008/world/results.json'))
 examples=[]
 for row in diagnostic['slow_cells']:
  if row['model']!='online' or row['kind']!='teacher_forced' or row['episode']!=args.episode_index or not row['persistent_slow'] or row['full_count']<5:continue
  t=row['boundary']*30;e=row['edge_index'];v=speed[t+1:t+31,e];n=count[t+1:t+31,e];start=float(speed[t,e]);end=float(speed[t+30,e])
  examples.append(dict(time=t,edge_id=row['edge_id'],vehicles_at_start=float(count[t,e]),start_speed=start,end_speed=end,endpoint_mean=.5*(start+end),actual_30s_mean=float(v.mean()),trapezoidal_1s_mean=float(np.trapezoid(speed[t:t+31,e],dx=1)/30),occupied_seconds=int((n>0).sum()),occupied_time_mean=float(v[n>0].mean()) if (n>0).any() else None,vehicle_time_weighted_speed=float((v*n).sum()/n.sum()) if n.sum() else None,min_speed=float(v.min()),max_speed=float(v.max()),std_speed=float(v.std()),slow_seconds=int((v<=1).sum()),prediction=row['predicted_speed_mps'],seconds=v.tolist(),counts=n.tolist()))
 # Whole replay: definition discrepancy, conditioned on occupancy at both endpoints.
 endpoints=.5*(speed[0:-1:30]+speed[30::30]);true=speed[1:].reshape(-1,30,speed.shape[1]).mean(1);occupied=(count[0:-1:30]>0)&(count[30::30]>0)
 slow=occupied&(speed[0:-1:30]<=1)&(speed[30::30]<=1)
 groups={}
 for name,mask in [('both_endpoints_occupied',occupied),('both_endpoints_slow',slow)]:
  err=np.abs(endpoints-true)[mask];groups[name]=dict(cells=int(mask.sum()),endpoint_vs_actual_mean_mae=float(err.mean()),p95=float(np.quantile(err,.95)),max=float(err.max()))
 report=dict(episode_index=args.episode_index,simulation_end=args.end,demand_hash=archived['metadata']['demand_sha256'],population=len(top.vehicles),matched_action_events=len(used),matched_boundaries=len(boundary),max_boundary_error=max(r['max_abs_feature_error'] for r in boundary),seconds=time.monotonic()-begin,examples=examples,groups=groups,note='Exact historical action replay with original frozen session code and demand. Every30s full flow checked against original archive; zero fallback decisions. 1s speed is SUMO last-step edge mean, not individual vehicle travel speed. Right-endpoint sample mean for30 steps plus trapezoidal estimate.')
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
