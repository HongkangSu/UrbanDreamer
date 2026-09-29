#!/usr/bin/env python3
"""Controlled Toy route assignment probe, not an optimizer or policy change."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from urban_dreamer.workflow.config import load_config
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.traffic.environment import SumoSession, NextHopDecision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    config=load_config(ROOT/'configs/toy_cross_bottleneck.toml')
    prepared=validate_prepared_inputs(config)
    topology=load_traffic_topology(prepared)
    order=[v.vehicle_id for v in topology.vehicles]
    np.random.default_rng(731).shuffle(order)
    results=[]
    target=args.output_dir/'route_externalities.json'
    for seed in (10042,10043,10044):
        baseline=None
        for count in (0,1,4,16,64,160,320):
            short=set(order[:count])
            settings=replace(config,run=replace(config.run,seed=seed))
            with SumoSession(settings,prepared,topology) as session:
                session.reset()
                def decide(vehicles,flow,now):
                    decisions=[]
                    for vehicle in vehicles:
                        current,dest=vehicle['current_edge'],vehicle['destination_edge']
                        choices=session.candidates.candidates(current,dest)
                        def cost(edge):
                            length,speed=session.candidates.junction(current,edge)
                            return length/speed+session.network.free_flow_travel_seconds[edge]+session.candidates.costs(dest)[edge]
                        selected=min(choices,key=cost)
                        if vehicle['vehicle_id'] not in short:
                            for edge in choices:
                                if session.network.edge_ids[edge] in ('A_to_R','B_to_R'):
                                    selected=edge
                        decisions.append(NextHopDecision(vehicle['vehicle_id'],current,selected,now))
                    return decisions
                for _ in range(160):
                    session.advance_window(decide)
                    if session.done:
                        break
                if not session.done:
                    raise RuntimeError('controlled probe did not complete')
                times=dict(session.travel_times)
            if baseline is None:
                baseline=times
            treated=[times[k]-baseline[k] for k in order if k in short]
            others=[times[k]-baseline[k] for k in order if k not in short]
            row=dict(seed=seed,shortest_assigned=count,direct_R_assigned=len(order)-count,
                fleet_mean_seconds=float(np.mean(list(times.values()))),
                fleet_delta_seconds=float(np.mean([times[k]-baseline[k] for k in order])),
                treated_delta_seconds=float(np.mean(treated)) if treated else None,
                others_delta_seconds=float(np.mean(others)) if others else None,
                travel_times=times)
            results.append(row)
            target.write_text(json.dumps({'description':'Nested fixed vehicle subsets follow static shortest; all other vehicles take direct R. Deltas compare the same vehicle IDs against all-R with the same SUMO seed.','subset_order':order,'runs':results},indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!='travel_times'}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
