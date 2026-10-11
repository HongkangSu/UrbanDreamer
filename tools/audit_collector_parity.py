"""Bounded diagnostic: identical routes through both collectors, then policy-only replay.

No learning, checkpoint loading or production process changes. Diagnostic
archives explicitly record that a fixed feature-cost policy replaced the actor.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_toy_actor_critic as ac_tool
from tools.run_city_baseline_dataset import atomic_json, now
from urban_dreamer.learning.baselines import common, build_method
from urban_dreamer.models.policy import ActionSelection, NextHopActor, set_shortest_path_prior
from urban_dreamer.traffic.event_session import SumoEventSession


class FixedActor:
    prior_weight = 0.
    prior_temperature_seconds = 30.

    def sample_action(self, features, mask, **unused):
        indices = features[..., 4].masked_fill(~mask, torch.inf).argmin(-1)
        zero = torch.zeros(len(indices), device=features.device)
        return ActionSelection(indices, zero, zero)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--vehicle-limit', type=int, default=64)
    parser.add_argument('--inference-batches', type=int, default=96)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(config=ROOT / 'configs/ingolstadt_custom.toml', output_dir=output,
        device='cpu', seed=42, threads=1, vehicle_limit=args.vehicle_limit,
        actor_lr=2e-4, critic_lr=3e-4, global_critic_lr=3e-4,
        session='event', backend='libsumo', step_seconds=1.,
        forbid_revisit=False, global_reward_only=True, sp_prior_weight=0.,
        sp_prior_temperature=30., sp_prior_hold_episodes=200, sp_prior_off_episode=1000)
    ac = ac_tool.RealActorCritic(SimpleNamespace(**settings))
    generic = common.EpisodeRunner(SimpleNamespace(**settings, method='collector_audit'))
    ac.actor = FixedActor()
    capture, samples = {}, []
    original_observations = common.build_actor_observations

    def observations(**kwargs):
        started = time.perf_counter()
        result = original_observations(**kwargs)
        capture['observation_seconds'] += time.perf_counter() - started
        capture['latest_observations'] = result
        return result

    ac_tool.build_actor_observations = common.build_actor_observations = observations

    class MeasuredSession(SumoEventSession):
        def _observe(self, counts):
            started = time.perf_counter()
            result = super()._observe(counts)
            capture['vehicle_sync_seconds'] += time.perf_counter() - started
            return result

        def advance_window(self, callback):
            def measured(pending, frozen, timestamp):
                started = time.perf_counter()
                decisions = callback(pending, frozen, timestamp)
                capture['callback_seconds'] += time.perf_counter() - started
                capture['callbacks'] += 1
                if capture['save_samples'] and capture['callbacks'] % 10 == 0:
                    samples.append((copy.deepcopy(pending),
                                    copy.deepcopy(capture['latest_observations']), timestamp))
                for decision in decisions:
                    capture['digest'].update(json.dumps([decision.vehicle_id, decision.current_edge,
                        decision.next_edge, decision.decision_time]).encode())
                return decisions
            return super().advance_window(measured)

    ac.session_factory = generic.session_factory = lambda *a: MeasuredSession(*a, step_seconds=1., backend='libsumo')

    def fixed_select(vehicles, observations, flow, timestamp):
        selected = np.where(observations.mask, observations.features[..., 4], np.inf).argmin(-1)
        return selected.tolist(), [dict(entropy=0.) for _ in selected]

    report = dict(created_at=now(),config=str(settings['config']),vehicle_limit=args.vehicle_limit,
        seed=43,collection=[],inference=[],
        limitations='Shared busy machine; fixed-policy diagnostic, no learning. Policy replay uses fresh networks and identical captured batches; outputs do not change traffic.')
    for index, name in enumerate(('ac', 'common', 'common', 'ac')):
        capture.clear()
        capture.update(observation_seconds=0., vehicle_sync_seconds=0., callback_seconds=0.,
            callbacks=0, digest=hashlib.sha256(), save_samples=index == 0)
        archive = output / f'{index}_{name}.npz'
        extra = dict(phase='diagnostic', episode=43, policy_override='fixed_cost_argmin', not_for_training=True)
        if name == 'ac':
            rows, metrics = ac.run_episode(policy='stochastic', seed=43, collect=True,
                archive_path=archive, archive_metadata=extra)
        else:
            rows, _, metrics = generic.run_episode(fixed_select, seed=43, collect=True, row_flow=False,
                archive_path=archive, archive_metadata=extra)
        result = dict(collector=name,index=index,seconds=metrics['collection_seconds'],
            archive_seconds=metrics['archive_seconds'], observation_seconds=capture['observation_seconds'],
            callback_seconds=capture['callback_seconds'],vehicle_sync_seconds=capture['vehicle_sync_seconds'],
            callbacks=capture['callbacks'],events=len(rows),N=metrics['vehicle_count'],
            exposure=metrics['exposure_seconds'],completion=metrics['completion_rate'],
            microsteps=metrics['microsteps'],event_sha256=capture['digest'].hexdigest(),
            demand_sha256=metrics.get('demand_sha256',metrics.get('demand_realization_sha256')))
        report['collection'].append(result)
        atomic_json(output / 'report.json',report)
        print(json.dumps(result),flush=True)
    for key in ('event_sha256','demand_sha256','events','N','exposure','completion','microsteps'):
        assert len({r[key] for r in report['collection']}) == 1, key
    picks = np.linspace(0,len(samples)-1,min(len(samples),args.inference_batches),dtype=int)
    samples = [samples[int(i)] for i in picks]
    assert samples
    report['captured_batch_sizes'] = dict(count=len(samples),
        mean=float(np.mean([len(v) for v, _, _ in samples])),max=max(len(v) for v, _, _ in samples))
    # Restore builder before independent inference measurements.
    ac_tool.build_actor_observations = common.build_actor_observations = original_observations
    for device in ('cpu','cuda:0') if torch.cuda.is_available() else ('cpu',):
        for name in ('ac','mappo','ippo','iql','mamba'):
            torch.manual_seed(42)
            generic.device = torch.device(device)
            method_args = SimpleNamespace(**(settings | dict(device=device,method=name)))
            if name == 'ac':
                actor = NextHopActor().to(device)
                def select(v, obs, flow, timestamp):
                    with torch.no_grad():
                        result = actor.sample_action(torch.as_tensor(obs.features,device=device),
                            torch.as_tensor(obs.mask,device=device))
                    return result.indices_and_entropy()
                method = SimpleNamespace(actor=actor,select=select)
            else:
                method = build_method(name,method_args,generic)
            set_shortest_path_prior(method.actor,1.,30.)
            durations=[]
            for repeat in range(3):
                if hasattr(method,'begin_episode'):method.begin_episode(generic.episode_topology,seed=43)
                if device.startswith('cuda'):torch.cuda.synchronize()
                started=time.perf_counter()
                for vehicles,obs,timestamp in samples:method.select(vehicles,obs,None,timestamp)
                if device.startswith('cuda'):torch.cuda.synchronize()
                if repeat:durations.append(time.perf_counter()-started)
            result=dict(method=name,device=device,batches=len(samples),
                median_seconds=statistics.median(durations),measurements=durations)
            report['inference'].append(result)
            print(json.dumps(result),flush=True)
            atomic_json(output/'report.json',report)
            del method
            if device.startswith('cuda'):torch.cuda.empty_cache()
    report['finished_at']=now()
    atomic_json(output/'report.json',report)


if __name__=='__main__':
    main()
