"""Replay a frozen policy with read-only, one-second mesoscopic stall telemetry."""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_incident_dta_sp import save
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.dta import DtaMethod
from urban_dreamer.traffic.event_session import SumoEventSession


def reference_row(path, seed):
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise ValueError('reference JSON must contain a list of metrics or training entries')
    matches = [r.get('metrics', r) for r in rows if r.get('metrics', r)['seed'] == seed]
    if len(matches) != 1:
        raise ValueError(f'expected exactly one reference for seed {seed}, found {len(matches)}')
    return matches[0]


class StallProbe(SumoEventSession):
    def __init__(self, *args, original_deadline, reference, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_deadline = original_deadline
        self.reference = reference
        self.ids = [v.vehicle_id for v in self.topology.vehicles]
        self.id_index = {v: i for i, v in enumerate(self.ids)}
        self.samples = []
        self.times = []
        self.series = []
        self.last_decisions = {}
        self.baseline = None
        self.final = None
        self.exposure = 0.
        payload = [dict(vehicle_id=v.vehicle_id, origin_index=v.origin_index,
            destination_index=v.destination_index, departure_second=v.departure_second)
            for v in self.topology.vehicles]
        self.demand_hash = hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()

    def _apply_decisions(self, callback, flow):
        decisions = super()._apply_decisions(callback, flow)
        for d in decisions:
            self.last_decisions[d.vehicle_id] = float(d.decision_time)
        return decisions

    def _observe(self, departure_counts):
        super()._observe(departure_counts)
        now = self._time_ms / 1000.
        if now < self.original_deadline - 1200:
            return
        # Columns: active, road index, segment index, raw position, odometer, speed.
        sample = np.full((len(self.ids), 6), np.nan, dtype=np.float64)
        sample[:, 0] = 0
        api = self._connection.vehicle
        for vehicle_id in self._active:
            i = self.id_index[vehicle_id]
            road = api.getRoadID(vehicle_id)
            sample[i] = (1, self.network.edge_to_index[road], api.getSegmentIndex(vehicle_id),
                api.getLanePosition(vehicle_id), api.getDistance(vehicle_id), api.getSpeed(vehicle_id))
        active = sample[:, 0] == 1
        if not np.isfinite(sample[active]).all():
            raise ValueError('active vehicle telemetry contains nonfinite values')
        if (sample[active, 1:] < 0).any():
            raise ValueError('active vehicle telemetry contains invalid negative values')
        self.times.append(now)
        self.samples.append(sample)

    def snapshot(self, state):
        return dict(time=float(state['time']), vehicles=copy.deepcopy(state['vehicles']),
            travel_times=dict(self.travel_times), exposure_seconds=self.exposure,
            vehicle_count=len(self.ids), unfinished=len(self.ids)-len(self.travel_times),
            never_departed=sum(not v['departed'] for v in state['vehicles']),
            demand_sha256=self.demand_hash, last_decisions=dict(self.last_decisions))

    def advance_window(self, callback):
        result = super().advance_window(callback)
        if self._incident_active:
            raise ValueError('diagnostic requires no incident')
        self.exposure += result.active_vehicle_seconds
        state = result.next_state
        self.series.append(dict(time=float(state['time']), arrived=len(self.travel_times),
            departed=sum(v['departed'] for v in state['vehicles']), active=len(self._active)))
        if state['time'] == self.original_deadline:
            self.baseline = self.snapshot(state)
            keys = ('demand_sha256', 'vehicle_count', 'exposure_seconds', 'unfinished', 'never_departed')
            checks = {key: dict(actual=self.baseline[key], expected=self.reference[key],
                               exact_match=self.baseline[key] == self.reference[key]) for key in keys}
            self.baseline['reference_checks'] = checks
            if not all(check['exact_match'] for check in checks.values()):
                raise AssertionError(f'original deadline does not reproduce reference: {checks}')
            api = self._connection.edge
            self.baseline['edge_traffic'] = {
                edge: dict(mean_speed_mps=float(api.getLastStepMeanSpeed(edge)),
                           halting=int(api.getLastStepHaltingNumber(edge)),
                           vehicles=int(api.getLastStepVehicleNumber(edge)),
                           lanes=int(api.getLaneNumber(edge)))
                for edge in {self.network.edge_ids[v['current_edge']] for v in state['vehicles']
                             if v['departed'] and not v['arrived']}}
        self.final = self.snapshot(state)
        return result


def describe(probe):
    if probe.baseline is None:
        raise ValueError('episode completed before original deadline; no censored vehicles to diagnose')
    data = np.asarray(probe.samples)
    times = np.asarray(probe.times)
    deadline = probe.original_deadline
    baseline = probe.baseline
    unfinished = [v for v in baseline['vehicles'] if v['departed'] and not v['arrived']]
    rows = []
    for v in unfinished:
        i = probe.id_index[v['vehicle_id']]
        row = dict(vehicle_id=v['vehicle_id'], edge_id=probe.network.edge_ids[v['current_edge']],
                   last_decision_seconds=baseline['last_decisions'].get(v['vehicle_id']),
                   actual_departure_seconds=v['actual_departure_time'],
                   entered_last600=v['actual_departure_time'] > deadline-600)
        for duration in (300, 600):
            mask = (times >= deadline-duration) & (times <= deadline)
            x = data[mask, i]
            continuous = len(x) == duration+1 and bool((x[:, 0] == 1).all())
            structural = continuous and bool((x[:, 1:3] == x[0, 1:3]).all())
            pos_range = float(np.ptp(x[:, 3])) if continuous else None
            odo_range = float(np.ptp(x[:, 4])) if continuous else None
            stationary = structural and pos_range <= 1. and odo_range <= 1.
            row[f'active_continuously_{duration}'] = continuous
            row[f'stationary_{duration}'] = stationary
            row[f'no_segment_progress_{duration}'] = structural
            row[f'structural_progress_{duration}'] = continuous and not structural
            row[f'interpolated_movement_same_segment_{duration}'] = structural and not stationary
            row[f'no_decision_{duration}'] = baseline['last_decisions'].get(v['vehicle_id'], -1) <= deadline-duration
            row[f'odometer_delta_{duration}_m'] = float(x[-1, 4]-x[0, 4]) if continuous else None
            row[f'position_range_{duration}_m'] = pos_range
            row[f'low_speed_fraction_{duration}'] = float((x[:, 5] < .1).mean()) if continuous else None
        after = data[times >= deadline, i]
        after = after[after[:, 0] == 1]
        row['arrived_during_extension'] = v['vehicle_id'] in probe.final['travel_times']
        row['progress_during_extension'] = row['arrived_during_extension'] or (len(after) > 1 and
            bool(np.ptp(after[:, 4]) > 1. or (after[:, 1:3] != after[0, 1:3]).any()))
        rows.append(row)
    aggregate = {}
    for duration in (300, 600):
        for prefix in ('active_continuously', 'stationary', 'no_segment_progress',
                       'structural_progress', 'interpolated_movement_same_segment', 'no_decision'):
            aggregate[f'{prefix}_{duration}'] = sum(r[f'{prefix}_{duration}'] for r in rows)
    aggregate['entered_last600'] = sum(r['entered_last600'] for r in rows)
    distance = [r['odometer_delta_600_m'] for r in rows if r['odometer_delta_600_m'] is not None]
    aggregate['odometer_delta_600_m_quantiles_0_25_50_75_100'] = np.percentile(distance, [0,25,50,75,100]).tolist() if distance else []
    aggregate['arrived_during_extension'] = sum(r['arrived_during_extension'] for r in rows)
    aggregate['stationary600_progress_during_extension'] = sum(r['stationary_600'] and r['progress_during_extension'] for r in rows)
    aggregate['arrivals_last600_before_deadline'] = sum(v.get('arrival_time', -1) > deadline-600 and v['arrived'] for v in baseline['vehicles'])
    edges = []
    for edge in sorted({r['edge_id'] for r in rows}):
        members = [r for r in rows if r['edge_id'] == edge]
        edges.append(dict(edge_id=edge, unfinished=len(members),
            stationary600=sum(r['stationary_600'] for r in members),
            length_m=float(probe.network.lengths_m[probe.network.edge_to_index[edge]]),
            **baseline['edge_traffic'][edge]))
    edges.sort(key=lambda r: (-r['stationary600'], -r['unfinished'], r['edge_id']))
    summary = dict(reference_reproduced_exactly=True, original_deadline_seconds=deadline,
        final_time_seconds=probe.final['time'], vehicle_count=len(probe.ids),
        original_unfinished=baseline['unfinished'], original_never_departed=baseline['never_departed'],
        original_departed_unfinished=len(rows), final_unfinished=probe.final['unfinished'],
        final_never_departed=probe.final['never_departed'], reference_checks=baseline['reference_checks'],
        definitions=dict(stationary='Continuously active, unchanged edge and meso segment, raw position and odometer range <=1 metre.',
            structural_no_progress='Continuously active on same edge and segment; meso position may interpolate without leaving segment.',
            cutoff='All original unfinished vehicles are deadline-censored, not proven permanently deadlocked.'),
        counts=aggregate, top_edges=edges[:30], series_30s=probe.series)
    return summary, rows, data, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reference-json', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--extend-seconds', type=int, default=600)
    args = parser.parse_args()
    if args.extend_seconds < 0 or args.extend_seconds % 30:
        parser.error('extend-seconds must be a nonnegative multiple of 30')
    args.output.mkdir(parents=True, exist_ok=True)
    reference = reference_row(args.reference_json, args.seed)
    runner_args = argparse.Namespace(config=args.config.resolve(), device='cpu', threads=1, seed=42,
        vehicle_limit=0, method='dta_so', session=None, backend=None, step_seconds=None,
        forbid_revisit=False, sp_prior_weight=0.)
    runner = EpisodeRunner(runner_args)
    original_deadline = runner.deadline
    runner.deadline += args.extend_seconds
    if runner.config.simulation.incident_start_seconds <= runner.deadline:
        raise ValueError('incident must start after entire extended replay')
    if runner.config.simulation.traffic_model != 'meso':
        raise ValueError('segment diagnostic requires meso')
    probes = []
    def factory(*a, **kw):
        probe = StallProbe(*a, original_deadline=original_deadline, reference=reference, **kw)
        probes.append(probe)
        return probe
    method = None
    selector = 'shortest'
    if args.checkpoint:
        method = DtaMethod(runner_args, runner)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        method.load_state_dict(state.get('method', state))
        selector = functools.partial(method.select, explore=False)
    iteration = method.iteration if method else None
    _, _, metrics = runner.run_episode(selector, seed=args.seed, session_factory=factory,
        label='dta_so' if method else 'shortest', on_demand=method.begin_episode if method else None)
    if method and method.iteration != iteration:
        raise AssertionError('frozen DTA iteration changed')
    probe = probes[0]
    summary, rows, data, times = describe(probe)
    summary.update(seed=args.seed, checkpoint=str(args.checkpoint) if args.checkpoint else None,
                   extended_metrics=metrics)
    save(args.output/'summary.json', summary)
    save(args.output/'vehicles.json', rows)
    save(args.output/'baseline_snapshot.json', probe.baseline)
    save(args.output/'final_snapshot.json', probe.final)
    np.savez_compressed(args.output/'telemetry.npz', times=times, vehicle_ids=np.asarray(probe.ids),
        edge_ids=np.asarray(probe.network.edge_ids), active=data[:,:,0].astype(bool),
        road_index=np.nan_to_num(data[:,:,1], nan=-1).astype(np.int32),
        segment_index=np.nan_to_num(data[:,:,2], nan=-1).astype(np.int32),
        position_m=data[:,:,3], odometer_m=data[:,:,4], speed_mps=data[:,:,5].astype(np.float32))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('series_30s','extended_metrics','top_edges')}, indent=2))


if __name__ == '__main__':
    main()
