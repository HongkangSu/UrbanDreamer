"""Bounded Dreamer rollout profile from an immutable city checkpoint.

Runs the original four rollouts per imagination pass, without learning or
writing the source checkpoint. Separate actor, observation, RSSM and physical
simulation timers include the CPU synchronization each operation requires.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--implementation', type=Path)
    parser.add_argument('--compare-implementation', type=Path,
        help='interleave this reference with the selected implementation; requires --no-profile')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--fixture-output', type=Path,
        help='save two real seed episodes in a compact benchmark-only checkpoint')
    parser.add_argument('--imagination-device', default='auto')
    parser.add_argument('--allow-legacy-imagination-migration', action='store_true',
        help='explicitly permit moving an unmarked legacy imagination actor to another device')
    parser.add_argument('--no-profile', action='store_true',
        help='time the whole pass without per-operation CUDA synchronization overhead')
    args = parser.parse_args()
    started_utc = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    initial_load = os.getloadavg()
    if args.repeats < 1 or (args.compare_implementation and not args.no_profile):
        parser.error('repeats must be positive and comparisons require --no-profile')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from urban_dreamer.learning.baselines import common, dreamer
    if args.implementation:
        spec = importlib.util.spec_from_file_location(
            'urban_dreamer.learning.baselines.dreamer_profile', args.implementation)
        dreamer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dreamer)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.fixture_output:
        state['method_state']['seed_replay']['episodes'] = state['method_state']['seed_replay']['episodes'][-2:]
        state['method_state']['sequence_replay']['episodes'] = []
        for replay in ('seed_replay', 'sequence_replay'):
            state['method_state'][replay].update(sample_order=[], sample_cursor=0)
        args.fixture_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, args.fixture_output)
    config = dict(state['args'])
    config.update(device=args.device, threads=1, dreamer_imagination_device=args.imagination_device,
        dreamer_allow_legacy_imagination_migration=args.allow_legacy_imagination_migration)
    runner = common.EpisodeRunner(SimpleNamespace(**config))
    method = dreamer.DreamerMethod(runner.args, runner)
    method.load_state_dict(state['method_state'])
    from tools.train_toy_baseline import apply_episode_prior
    apply_episode_prior(method, runner.args, int(state['episode']))
    method.world_model.eval()
    methods = {'selected': method}
    if args.compare_implementation:
        spec = importlib.util.spec_from_file_location(
            'urban_dreamer.learning.baselines.dreamer_reference', args.compare_implementation)
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        reference = reference_module.DreamerMethod(runner.args, runner)
        reference.load_state_dict(state['method_state'])
        apply_episode_prior(reference, runner.args, int(state['episode']))
        reference.world_model.eval()
        methods['reference'] = reference
    del state
    totals = {}
    def timed(name, original):
        if args.no_profile:
            return original
        def call(*a, **kw):
            start = time.perf_counter()
            value = original(*a, **kw)
            if args.device.startswith('cuda') and name != 'observations':
                torch.cuda.synchronize(args.device)
            entry = totals.setdefault(name, {'seconds': 0., 'calls': 0})
            entry['seconds'] += time.perf_counter() - start
            entry['calls'] += 1
            return value
        return call
    dreamer.advance_imagined_window = timed('physical_with_actor', dreamer.advance_imagined_window)
    dreamer.build_actor_observations = timed('observations', dreamer.build_actor_observations)
    if hasattr(method, '_sample_imagined_actions'):
        method._sample_imagined_actions = timed('actor', method._sample_imagined_actions)
    else:
        method.actor.sample_action = timed('actor', method.actor.sample_action)
    for name in ('_initial_latent', '_decode_flow', '_prior_dynamics_step'):
        setattr(method, name, timed(name, getattr(method, name)))
    method.world_model.predict_reward = timed('reward_head', method.world_model.predict_reward)
    rng_state = method.rng.bit_generator.state
    results = []
    for repeat in range(args.repeats):
        order = list(methods.items())
        if repeat % 2:
            order.reverse()
        for label, method in order:
            method.rng.bit_generator.state = rng_state
            torch.manual_seed(43219)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            totals.clear()
            started = time.perf_counter()
            events = windows = 0
            rollouts = []
            for rollout in range(dreamer.IMAGINATION_ROLLOUTS):
                result = method._imagine_rollout(rollout)
                if result is not None:
                    rows, trajectory = result
                    events += len(rows)
                    windows += len(trajectory[0])
                    rollouts.append({'events': len(rows), 'windows': len(trajectory[0]),
                        'actions': [row['action'] for row in rows],
                        'rewards': trajectory[2], 'terminals': trajectory[3]})
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            results.append(dict(repeat=repeat, implementation=label, seconds=time.perf_counter() - started,
                events=events, windows=windows, timers=dict(totals), rollouts=rollouts))
            print(json.dumps({k: v for k, v in results[-1].items() if k != 'rollouts'}), flush=True)
    selected_results = [entry for entry in results if entry['implementation'] == 'selected']
    report = dict(checkpoint=str(args.checkpoint), implementation=str(args.implementation or dreamer.__file__),
        device=args.device, repeats=results, median_seconds=float(np.median([x['seconds'] for x in selected_results])),
        warm_median_seconds=float(np.median([x['seconds'] for x in selected_results[1:] or selected_results])),
        started_utc=started_utc, host_load_start=initial_load, host_load_end=os.getloadavg(),
        prior_weight=float(method.actor.prior_weight),
        profiled=not args.no_profile,
        legacy_imagination_migration_allowed=args.allow_legacy_imagination_migration,
        warm_medians={label: float(np.median([entry['seconds'] for entry in results
            if entry['implementation'] == label and (entry['repeat'] > 0 or args.repeats == 1)]))
            for label in methods},
        note='Unchanged four-rollout/eight-window budget; isolated inference only, no learner updates.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
