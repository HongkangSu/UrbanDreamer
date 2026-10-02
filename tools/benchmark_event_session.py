#!/usr/bin/env python3
"""Compare boundary reads with former persistent edge subscriptions using saved DTA.

Runs two full episodes without training or writing experiment checkpoints.
The reference session exists only here to verify traffic and action equivalence.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.baselines import build_method, common
from urban_dreamer.traffic.event_session import SumoEventSession, _EDGE_STATE_VARS


class ObservedSession(SumoEventSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_digest = hashlib.sha256()

    def _apply_decisions(self, callback, flow):
        decisions = super()._apply_decisions(callback, flow)
        for action in decisions:
            self.decision_digest.update(json.dumps(vars(action), sort_keys=True).encode())
        return decisions


class PersistentEdgeSubscriptionsSession(ObservedSession):
    """Reference implementation of the former libsumo edge-read behavior."""

    def _flow(self):
        if self._flow_cache is not None and self._flow_cache[0] == self._time_ms:
            return self._flow_cache[1]
        if not getattr(self, '_edges_subscribed', False):
            for edge in self.network.edge_ids:
                self._connection.edge.subscribe(edge, _EDGE_STATE_VARS)
            self._edges_subscribed = True
        edge_state = {}
        if not self.done:
            tc = self._tc
            results = self._connection.edge.getAllSubscriptionResults()
            for edge in self.network.edge_ids:
                values = results.get(edge)
                if values is not None:
                    edge_state[edge] = {
                        'occupancy': float(values[tc.LAST_STEP_OCCUPANCY]) / 100.,
                        'mean_speed': float(values[tc.LAST_STEP_MEAN_SPEED]),
                        'vehicle_number': float(values[tc.LAST_STEP_VEHICLE_NUMBER]),
                        'halting_number': float(values[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]),
                    }
        flow = self.network.extract_flow(edge_state)
        self._flow_cache = (self._time_ms, flow)
        return flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=10044)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('refusing to overwrite an existing benchmark report')
    protocol = json.loads((args.experiment / 'protocol.json').read_text())
    if protocol['method'] != 'dta' or protocol['backend'] != 'libsumo':
        raise ValueError('benchmark requires a saved libsumo DTA experiment')
    saved_args = SimpleNamespace(**protocol['args'])
    saved_args.config = ROOT / saved_args.config
    runner = common.EpisodeRunner(saved_args)
    method = build_method('dta', saved_args, runner)
    state = common.load_checkpoint(args.experiment / 'latest.pt', method_name='dta',
        prepared=runner.prepared, config=runner.config, device=runner.device)
    method.load_state_dict(state['method_state'])
    records = []
    import libsumo
    for label, session_type in [('boundary_getters', ObservedSession),
                                 ('persistent_subscriptions', PersistentEdgeSubscriptionsSession)]:
        sessions = []
        def factory(*a, **k):
            session = session_type(*a, **k, backend='libsumo', step_seconds=protocol['step_seconds'])
            sessions.append(session)
            return session
        runner.session_factory = factory
        step_seconds = 0.
        original_step = libsumo.simulationStep
        def timed_step(*a, **k):
            nonlocal step_seconds
            started = time.monotonic()
            try:
                return original_step(*a, **k)
            finally:
                step_seconds += time.monotonic() - started
        libsumo.simulationStep = timed_step
        try:
            _, windows, metrics = runner.run_episode(functools.partial(method.select, explore=False),
                seed=args.seed, record_windows=True, row_flow=False, label='timing_probe')
        finally:
            libsumo.simulationStep = original_step
        digest = hashlib.sha256()
        for window in windows:
            for key in ('flow', 'next_flow', 'action', 'departure', 'context'):
                digest.update(np.asarray(window[key]).tobytes())
            digest.update(json.dumps({key: window[key] for key in
                ('time', 'active_vehicle_seconds', 'reward', 'terminal', 'vehicles')}, sort_keys=True).encode())
        physical_metrics = {key: value for key, value in metrics.items() if key != 'wall_seconds'}
        record = dict(mode=label, wall_seconds=metrics['wall_seconds'],
            simulation_step_wall_seconds=step_seconds,
            metrics=physical_metrics, transition_sha256=digest.hexdigest(),
            decisions_sha256=sessions[0].decision_digest.hexdigest(),
            vehicle_travel_sha256=hashlib.sha256(json.dumps(sessions[0].travel_times, sort_keys=True).encode()).hexdigest())
        records.append(record)
        print(json.dumps({key: value for key, value in record.items() if key != 'metrics'}), flush=True)
    equivalent = all(records[0][key] == records[1][key] for key in
        ('metrics', 'transition_sha256', 'decisions_sha256', 'vehicle_travel_sha256'))
    report = dict(experiment=str(args.experiment), checkpoint_episode=state['episode'],
        seed=args.seed, step_seconds=protocol['step_seconds'], backend='libsumo',
        equivalent=equivalent, speedup=records[1]['wall_seconds'] / records[0]['wall_seconds'], runs=records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'runs'}), flush=True)
    if not equivalent:
        raise AssertionError('boundary read optimization changed the episode results')


if __name__ == '__main__':
    main()
