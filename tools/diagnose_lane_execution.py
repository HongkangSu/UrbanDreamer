"""Read-only lane execution audit and frozen-policy SUMO intervention."""
from __future__ import annotations
import argparse, collections, json, re, sys, time
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_grounded_credit_diagnostic import make_adapter, collect
from tools.run_entropy_ablation import tree_hash
from tools.run_online_pilot import atomic_json, sha256_file
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.models.policy import NextHopActor

WARNING = re.compile(r"Vehicle '([^']+)' performs emergency stop at the end of lane '([^']+)' because there is no connection to the next edge .*time=([\d.]+)")

def compare_actor_lane_probability(adapter, observations, output):
    """Same real observations and lane constraints, actor weights only differ."""
    checkpoint_paths={
        'after160':ROOT/'outputs/imagination_40s_20261004/checkpoint_before_parallel.pt',
        'after453':ROOT/'outputs/convergence_diagnosis_20261005/frozen/checkpoints/latest.pt'}
    all_ids=np.arange(len(observations['current']))
    rng=np.random.default_rng(981273)
    trouble=np.isin(observations['current'],[adapter.network.edge_to_index[e]
        for e in ('-170018165#1','164051413')])
    ids={name:rng.choice(index,min(5000,len(index)),replace=False) if len(index) else index
         for name,index in [('all',all_ids),('trouble_edges_7_11m_8_93m',np.flatnonzero(trouble)),
             ('short_edges_le20m',np.flatnonzero(adapter.network.lengths_m[observations['current']]<=20))]}
    rows=[]
    for name,path in checkpoint_paths.items():
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        actor=NextHopActor();actor.load_state_dict(checkpoint['training_state']['models']['actor']);actor.eval()
        del checkpoint
        for stratum,index in ids.items():
            if not len(index): continue
            features=torch.from_numpy(observations['features'][index]);mask=torch.from_numpy(observations['mask'][index])
            illegal=torch.from_numpy(observations['mask'][index]&~observations['lane_legal'][index])
            with torch.no_grad():
                probability=torch.cat([actor.forward_logits(f,m).softmax(-1)
                    for f,m in zip(features.split(512),mask.split(512))])
                greedy=probability.argmax(-1)
                rows.append(dict(actor=name,stratum=stratum,states=len(index),actor_hash=tree_hash(actor.state_dict()),
                    mean_incompatible_action_mass=float((probability*illegal).sum(-1).mean()),
                    argmax_incompatible_fraction=float(illegal.gather(1,greedy[:,None]).float().mean()),
                    entropy=float(-(probability*probability.clamp_min(1e-30).log()).sum(-1).mean()),
                    mean_max_probability=float(probability.amax(-1).mean())))
    atomic_json(output/'fixed_real_lane_policy_comparison.json',dict(rows=rows,
        protocol='Same real policy inputs, original edge-level mask and observed current-lane connectivity. At most5000 identical rows perstratum, subset RNG separate from simulation. Lane-incompatible can still be feasible if strategic lanechange ispossible; this is immediate-execution mismatch propensity, notglobally illegal-action probability.'))

def lane_connections(path):
    root = ET.parse(path).getroot()
    lanes, permissions, links = {}, {}, collections.defaultdict(set)
    for edge in root.findall('edge'):
        for lane in edge.findall('lane'):
            lanes[lane.attrib['id']] = (edge.attrib['id'], lane.attrib['index'])
            allow = lane.get('allow', '').split(); disallow = lane.get('disallow', '').split()
            permissions[(edge.attrib['id'], lane.attrib['index'])] = (not allow or 'passenger' in allow or 'all' in allow) and 'passenger' not in disallow and 'all' not in disallow
    for connection in root.findall('connection'):
        source = (connection.get('from'), connection.get('fromLane'))
        target = (connection.get('to'), connection.get('toLane'))
        if permissions.get(source, False) and permissions.get(target, False):
            links[source].add(connection.get('to'))
    return lanes, links

def audit(adapter, output, lanes, links):
    source = ROOT/'outputs/ingolstadt_online_libsumo_20261004'
    log = (ROOT/'outputs/ingolstadt_online_libsumo_20261004_supervisor/run.log').read_text()
    warnings = [(v,l,float(t.rstrip('.'))) for v,l,t in WARNING.findall(log)]
    rows=[]
    for episode in (400,450,452):
        path=source/f'episodes/online/episode_{episode:06d}.npz'
        if not path.exists(): continue
        with np.load(path) as archive:
            events=json.loads(archive['decisions_json'].tobytes())
            summary=json.loads(archive['summary_json'].tobytes())
            states=json.loads(archive['states_json'].tobytes());state=states[-1]
            snapshots=[(float(s['time']),{v['vehicle_id']:v for v in s['vehicles']}) for s in states]
        indexed=collections.defaultdict(list)
        for event in events: indexed[(event['vehicle_id'],event['current_edge'])].append(event)
        matched=[]
        for vehicle,lane,timestamp in warnings:
            if not vehicle.startswith(f'ep{episode}_v'): continue
            edge,lane_index=lanes[lane]; edge_index=adapter.network.edge_to_index.get(edge)
            prior=[event for event in indexed.get((vehicle,edge_index),()) if event['decision_time']<=timestamp]
            event=max(prior,key=lambda x:x['decision_time']) if prior else None
            next_edge=adapter.network.edge_ids[event['next_edge']] if event else None
            stuck_until=timestamp;queued=0
            for snapshot_time,vehicles in snapshots:
                if snapshot_time<timestamp: continue
                row=vehicles[vehicle]
                if row['arrived'] or row['current_edge']!=edge_index or row.get('in_junction',False): break
                stuck_until=snapshot_time
                queued=max(queued,sum(v['departed'] and not v['arrived'] and v['current_edge']==edge_index
                    for v in vehicles.values()))
            matched.append(dict(vehicle=vehicle,lane=lane,time=timestamp,event=event,
                has_direct_passenger_connection=next_edge in links[(edge,lane_index)] if event else None,
                arrived=vehicle in summary['travel_times'],
                observed_same_external_edge_seconds=max(0.,stuck_until-timestamp),
                max_vehicles_on_warning_edge_while_observed_stuck=queued))
        affected={row['vehicle'] for row in matched}; unfinished=[v for v in state['vehicles'] if not v['arrived']]
        occupied=collections.Counter(adapter.network.edge_ids[v['current_edge']] for v in unfinished if v['departed'])
        row=dict(episode=episode,warning_count=len(matched),unique_affected_vehicles=len(affected),
            affected_arrived=sum(v in summary['travel_times'] for v in affected),
            affected_unfinished=sum(v not in summary['travel_times'] for v in affected),
            matched_events=sum(r['event'] is not None for r in matched),
            confirmed_lane_incompatible=sum(r['has_direct_passenger_connection'] is False for r in matched),
            warning_equals_action_timestamp=sum(r['event'] is not None and r['time']==r['event']['decision_time'] for r in matched),
            warnings_followed_by_60s_same_edge=sum(r['observed_same_external_edge_seconds']>=60 for r in matched),
            longest_observed_same_edge_seconds=max((r['observed_same_external_edge_seconds'] for r in matched),default=0),
            population=len(state['vehicles']),completion_rate=summary['metrics']['online_completion_rate'],
            final_unfinished_edges=occupied.most_common(15),warnings=matched)
        rows.append(row)
    atomic_json(output/'audit.json',dict(protocol='Warning lane linked to latest action on same external edge. Final queues descriptive, not causal proof.',episodes=rows))
    print(json.dumps([{k:v for k,v in row.items() if k!='warnings'} for row in rows]),flush=True)
    # Collection JSON terminates one physical episode's warning block. Linking
    # by those boundaries avoids confusing ep0 IDs reused in different arms.
    diagnostic=[]
    for logname in ('grounded_credit.log','real_feedback.log'):
        path=output.parent/logname
        if not path.exists(): continue
        pending=[]
        for line in path.read_text().splitlines():
            match=WARNING.search(line)
            if match: pending.append(match.groups())
            if not line.startswith('{'): continue
            try: item=json.loads(line)
            except json.JSONDecodeError: continue
            if item.get('stage')!='collection': continue
            directory=Path(item['directory']);ep=int(item['index'])
            archive_path=directory/f'episodes/online/episode_{ep:06d}.npz'
            if archive_path.exists():
                with np.load(archive_path) as archive:
                    events=json.loads(archive['decisions_json'].tobytes());summary=json.loads(archive['summary_json'].tobytes())
                by_key=collections.defaultdict(list)
                for event in events: by_key[(event['vehicle_id'],event['current_edge'])].append(event)
                nmatch=ninvalid=nsame=0;cars=set()
                for vid,lane,rawtime in pending:
                    if not vid.startswith(f'ep{ep}_v'): continue
                    timestamp=float(rawtime.rstrip('.'));edge,li=lanes[lane]
                    prior=[e for e in by_key.get((vid,adapter.network.edge_to_index.get(edge)),()) if e['decision_time']<=timestamp]
                    cars.add(vid)
                    if not prior: continue
                    event=max(prior,key=lambda e:e['decision_time']);nmatch+=1
                    ninvalid+=adapter.network.edge_ids[event['next_edge']] not in links[(edge,li)]
                    nsame+=event['decision_time']==timestamp
                diagnostic.append(dict(log=logname,directory=str(directory),episode=ep,
                    warnings=len(pending),unique_affected=len(cars),matched_events=nmatch,
                    confirmed_incompatible=ninvalid,same_timestamp=nsame,
                    affected_unfinished=sum(v not in summary['travel_times'] for v in cars),
                    completion_rate=item['metrics']['online_completion_rate']))
            pending=[]
    atomic_json(output/'diagnostic_log_audit.json',diagnostic)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/convergence_diagnosis_20261005/frozen')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/convergence_diagnosis_20261005/lane')
    parser.add_argument('--arm',choices=['audit','current_lane','near_junction','lane_or_commitment'],default='audit')
    parser.add_argument('--indices',type=int,nargs='+',default=[0])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    adapter=make_adapter(args.source,args.output/args.arm)
    lanes,links=lane_connections(adapter.prepared_inputs.network_xml)
    if args.arm=='audit': audit(adapter,args.output,lanes,links);return
    checkpoint=args.source/'checkpoints/latest.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    adapter.actor.load_state_dict(state['training_state']['models']['actor']);del state
    actor_hash=tree_hash(adapter.actor.state_dict());original_select=adapter._select
    tracker={};counts=collections.Counter();interventions=[];captured=[]
    def factory(**kwargs):
        session=SumoEventSession(**kwargs,step_seconds=1.,backend='libsumo');tracker['session']=session;return session
    adapter._session_factory=factory
    def select(states,flow,timestamp,*,context,mode,actor=None,observations=None):
        session=tracker['session'];api=session._connection.vehicle
        mask=observations.mask.copy();direct=np.zeros_like(mask)
        for index,vehicle in enumerate(states):
            counts['decisions']+=1;vid=vehicle['vehicle_id'];lane=api.getLaneID(vid)
            edge,lane_index=lanes[lane]
            near=float(vehicle['remaining_distance'])<max(20.,api.getSpeed(vid)*3.)
            permitted=links[(edge,lane_index)]
            compatible=np.asarray([j>=0 and adapter.network.edge_ids[j] in permitted for j in observations.edge_indices[index]])
            direct[index]=compatible & observations.mask[index]
            if args.arm=='near_junction' and not near: continue
            if args.arm=='lane_or_commitment' and vehicle.get('committed_next_edge') is not None:
                committed=observations.edge_indices[index]==int(vehicle['committed_next_edge'])
                if (committed & ~compatible & mask[index]).any(): counts['incompatible_existing_commitment_retained']+=1
                compatible |= committed
            proposed=mask[index]&compatible
            if not proposed.any(): counts['empty_compatible_fallback']+=1;continue
            if not np.array_equal(proposed,mask[index]):
                counts['intervention_decisions']+=1;counts['removed_candidates']+=int((mask[index]&~proposed).sum())
                if len(interventions)<100:
                    interventions.append(dict(vehicle=vid,time=timestamp,lane=lane,remaining_distance=vehicle['remaining_distance'],near_junction=near,old_mask=mask[index].tolist(),new_mask=proposed.tolist()))
                mask[index]=proposed
        captured.append(dict(features=observations.features.copy(),mask=observations.mask.copy(),
            lane_legal=direct,current=np.asarray([v['current_edge'] for v in states],np.int64)))
        return original_select(states,flow,timestamp,context=context,mode=mode,actor=actor,
            observations=replace(observations,mask=mask))
    adapter._select=select
    atomic_json(args.output/args.arm/'protocol.json',dict(source_sha256=sha256_file(checkpoint),actor_hash=actor_hash,
        arm=args.arm,indices=args.indices,demand_seed=314159,policy_seed='271828 + index',
        policy='Original actor logits normalized over passenger-permitted current-lane outgoing candidates only; no learning, temperature or shortest-path prior.',
        limitation='Current-lane mask excludes turns attainable by lane changes; near_junction arm masks only below max(20m, 3*current speed). lane_or_commitment also preserves existing next edge, allowing originally planned lane changes but forbidding fresh incompatible immediate reroutes. Empty mask falls back to original action set.'))
    started=time.monotonic();rows=collect(adapter,args.output/args.arm,args.indices)
    if tree_hash(adapter.actor.state_dict())!=actor_hash: raise RuntimeError('Actor changed')
    atomic_json(args.output/args.arm/'summary.json',dict(rows=rows,counts=dict(counts),examples=interventions,
        actor_unchanged=True,actor_hash=actor_hash,wall_seconds=time.monotonic()-started))
    all_observations={key:np.concatenate([batch[key] for batch in captured]) for key in captured[0]}
    np.savez_compressed(args.output/args.arm/'policy_observations.npz',**all_observations)
    if args.arm!='near_junction': compare_actor_lane_probability(adapter,all_observations,args.output/args.arm)
    print(json.dumps(dict(arm=args.arm,counts=dict(counts),rows=rows)),flush=True)

if __name__=='__main__':main()
