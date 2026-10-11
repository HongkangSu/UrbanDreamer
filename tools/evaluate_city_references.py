#!/usr/bin/env python3
"""One reference evaluation round; DTA performs exactly one training episode/update."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import sys
import time


def evaluate_reference(runner, method, seeds, training_seed):
    """Fit DTA once on an independent demand, then freeze its costs for evaluation."""
    training = None
    if method is not None:
        rows, windows, metrics = runner.run_episode(
            functools.partial(method.select, explore=False), seed=training_seed,
            record_windows=True, label='dta_so', on_demand=method.begin_episode)
        update = method.on_episode(rows, windows, episode=1)
        if method.iteration != 1:
            raise ValueError('reference DTA must have exactly one assignment update')
        training = dict(seed=training_seed, metrics=metrics, update=update)
    evaluations = []
    for seed in seeds:
        if method is None:
            _, _, metrics = runner.run_episode('shortest', seed=seed, label='shortest')
        else:
            _, _, metrics = runner.run_episode(
                functools.partial(method.select, explore=False), seed=seed,
                label='dta_so', on_demand=method.begin_episode)
            if method.iteration != 1:
                raise ValueError('evaluation advanced DTA assignment learning')
        evaluations.append(metrics)
        print(json.dumps(dict(seed=seed, completion_rate=metrics['completion_rate'],
                              global_reward_sum=metrics['global_reward_sum'])), flush=True)
    return training, evaluations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--city', required=True)
    parser.add_argument('--method', choices=('shortest', 'dta_so'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    source = Path(manifest.get('source_revisions', [{}])[-1].get('source', manifest['source_snapshot']))
    sys.path.insert(0, str(source))
    from urban_dreamer.learning.baselines.common import EpisodeRunner
    from urban_dreamer.learning.baselines.dta import DtaMethod
    from urban_dreamer.traffic.event_session import SumoEventSession
    import torch

    entry = next(row for row in manifest['runs'] if row['city'] == args.city)
    config = Path(entry['command'][entry['command'].index('--config') + 1])
    training_seed = int(manifest['seed']) + 1
    seeds = manifest['evaluation_seeds']
    if training_seed in seeds:
        raise ValueError('DTA fitting demand must be independent of evaluation demands')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runner_args = argparse.Namespace(config=config, method=args.method, seed=int(manifest['seed']),
        device='cpu', threads=1, vehicle_limit=0, session=manifest['session_protocol'],
        backend=manifest['backend'], step_seconds=manifest['step_seconds'])
    session = functools.partial(SumoEventSession, step_seconds=manifest['step_seconds'], backend=manifest['backend'])
    started = time.monotonic()
    runner = EpisodeRunner(runner_args, session_factory=session)
    method = DtaMethod(runner_args, runner) if args.method == 'dta_so' else None
    training, evaluations = evaluate_reference(runner, method, seeds, training_seed)
    # Keep bulky branch histograms out of the reference plotting contract.
    clean = lambda row: {key: value for key, value in row.items() if key not in {'choices', 'branch_choices'}}
    if training is not None:
        training['metrics'] = clean(training['metrics'])
        torch.save(dict(method='dta_so', training_seed=training_seed,
                        training_episodes=1, method_state=method.state_dict()), output / 'assignment.pt')
    result = dict(city=args.city, method=args.method, training_episodes=int(method is not None),
        evaluation_rounds=1, evaluation_seeds=seeds, evaluations=[clean(row) for row in evaluations],
        training=training, source_snapshot=str(source), config=str(config),
        input_hashes=runner.prepared.input_hashes, session_protocol=manifest['session_protocol'],
        backend=manifest['backend'], step_seconds=manifest['step_seconds'],
        deadline_seconds=runner.deadline, elapsed_seconds=time.monotonic() - started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        interpretation='Approximate BPR marginal-cost DTA-SO after one MSA update; not converged SO' if method else
                       'Free-flow shortest path including junction cost; no training updates')
    temporary = output / '.result.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, output / 'result.json')
    print(json.dumps(dict(status='complete', output=str(output), seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
