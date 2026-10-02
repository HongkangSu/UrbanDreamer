#!/usr/bin/env python3
"""Breakdown probe: where does episode wall time go on a real network?

A: vehicles added via TraCI, then only simulationStep() to the horizon.
B: A + per-microstep departed/arrived lists and vehicle subscription pull.
Prints per-mode wall seconds, microstep count and seconds per microstep.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urban_dreamer.traffic.environment import safe_microstep_seconds
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
from urban_dreamer.workflow.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default='traci')
    parser.add_argument('--step', type=float, default=0.,
                        help='override microstep seconds (0 = safe_microstep_seconds)')
    args = parser.parse_args()
    config = load_config(args.config)
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    network = topology.network
    candidates = NextHopCandidates(network)
    microstep = args.step or safe_microstep_seconds(tuple(network.external_lane_lengths_m.values())
                                       or network.lengths_m[network.passenger_allowed],
                                       config.simulation.vehicle_max_speed_mps)
    print(f"microstep_seconds={microstep} edges={network.edge_count} vehicles={len(topology.vehicles)} backend={args.backend}", flush=True)

    import shutil
    import uuid
    if args.backend == 'libsumo':
        import libsumo as backend
    else:
        import traci as backend
    import traci.constants as tc
    binary = shutil.which('sumo') or str(Path(sys.executable).parent / 'sumo')
    label = 'probe-' + uuid.uuid4().hex
    start_args = [binary, '--net-file', str(prepared.network_xml), '--step-length', str(microstep),
                  '--seed', '42', '--no-step-log', 'true', '--duration-log.disable', 'true',
                  '--time-to-teleport', '-1']
    if args.backend == 'libsumo':
        import libsumo
        libsumo.start(start_args)
        conn = libsumo
    else:
        backend.start(start_args, label=label)
        conn = backend.getConnection(label)
    try:
        for index, demand in enumerate(sorted(topology.vehicles, key=lambda v: (v.departure_second, v.vehicle_id))):
            origin = network.edge_to_index[topology.origins[demand.origin_index]]
            destination = network.edge_to_index[topology.destinations[demand.destination_index]]
            choices = candidates.candidates(origin, destination)
            costs = candidates.costs(destination)
            initial = min(choices, key=lambda e: (network.free_flow_travel_seconds[e] + costs[e], e)) if choices else origin
            route = candidates.temporary_route(origin, initial, destination) if choices else (origin,)
            conn.route.add(f'r{index}', [network.edge_ids[e] for e in route])
            conn.vehicle.add(demand.vehicle_id, f'r{index}', depart=str(demand.departure_second),
                             departPos='0', departSpeed='0')
        # ---- mode A: pure stepping
        steps = int(args.seconds / microstep)
        started = time.monotonic()
        for _ in range(steps):
            conn.simulationStep()
        a = time.monotonic() - started
        active_a = conn.vehicle.getIDCount()
        # ---- mode B: + per-step subscription traffic
        conn.vehicletype.copy('DEFAULT_VEHTYPE', 'probe_cav')
        started = time.monotonic()
        for _ in range(steps):
            conn.simulationStep()
            for vid in conn.simulation.getDepartedIDList():
                conn.vehicle.subscribe(vid, (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_LANE_ID))
            conn.simulation.getArrivedIDList()
            conn.simulation.getStartingTeleportIDList()
            conn.simulation.getEndingTeleportIDList()
            subs = conn.vehicle.getAllSubscriptionResults()
            _ = len(subs)
        b = time.monotonic() - started
        print(f"A pure simulationStep : {a:8.1f}s for {steps} steps ({a/steps*1000:6.1f} ms/step), active={active_a}")
        print(f"B + subscription pull : {b:8.1f}s for {steps} steps ({b/steps*1000:6.1f} ms/step)")
        # ---- mode C (reference): full SumoSession advance_window, from AC probe metrics
        horizon = config.simulation.world_model_intervals_per_episode * 30 + config.simulation.training_completion_buffer_seconds
        print(f"episode horizon={horizon}s -> {int(horizon/microstep)} microsteps")
        print(f"estimated episode: A-only={a/steps*horizon/microstep/60:.1f}min  B-rate={(b/steps)*horizon/microstep/60:.1f}min")
    finally:
        conn.close()


if __name__ == '__main__':
    main()
