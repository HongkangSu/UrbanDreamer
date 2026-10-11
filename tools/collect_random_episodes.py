#!/usr/bin/env python3
"""Archive real episodes from uniform legal next-hop decisions."""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.baselines.common import resolve_collection_settings
from urban_dreamer.traffic.event_session import SumoEventSession
from urban_dreamer.workflow.config import load_config


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--vehicle-limit', type=int, default=0)
    parser.add_argument('--session', choices=('microstep', 'event'), default=None,
                        help='Defaults to simulation.session_protocol from the config')
    parser.add_argument('--backend', choices=('traci', 'libsumo'), default=None)
    parser.add_argument('--step-seconds', type=float, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--forbid-revisit', action='store_true')
    parser.add_argument('--global-reward-only', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reward-objective', choices=('observed_mean', 'legacy_exposure'), default='observed_mean')
    args = parser.parse_args(argv)
    try:
        simulation_settings = resolve_collection_settings(load_config(args.config), args)
    except ValueError as error:
        parser.error(str(error))
    if args.forbid_revisit and args.session != 'event':
        parser.error('forbid-revisit requires event session')
    if args.episodes < 1 or args.threads < 1:
        parser.error('episodes and threads must be positive')
    output = args.output_dir.resolve()
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite nonempty collection')
    output.mkdir(parents=True, exist_ok=True)
    factory = functools.partial(SumoEventSession, step_seconds=args.step_seconds,
                                backend=args.backend, **({'forbid_revisit': True} if args.forbid_revisit else {})) if args.session == 'event' else None
    runner = EpisodeRunner(args, session_factory=factory)
    protocol = dict(method='random', policy='uniform_over_legal_candidates', seed=args.seed,
        session=args.session, backend=args.backend, step_seconds=args.step_seconds,
        forbid_revisit=args.forbid_revisit,
        global_reward_only=args.global_reward_only,
        reward_objective=args.reward_objective,
        vehicle_limit=args.vehicle_limit, config=runner.config.resolved_snapshot(),
        input_hashes=runner.prepared.input_hashes,
        archive='episodes/train/episode_XXXXXX.npz; real SUMO only, all populations and completion outcomes')
    protocol.update(simulation_settings)
    start = 1
    if args.resume:
        saved_protocol = json.loads((output / 'protocol.json').read_text())
        saved_protocol.setdefault('forbid_revisit', False)
        saved_protocol.setdefault('global_reward_only', False)
        saved_protocol.setdefault('reward_objective', 'legacy_exposure')
        if saved_protocol != protocol:
            raise ValueError('random collection resume protocol mismatch')
        start = json.loads((output / 'progress.json').read_text())['last_episode'] + 1
    else:
        atomic_json(output / 'protocol.json', protocol)
        atomic_json(output / 'progress.json', {'last_episode': 0, 'episode_budget': args.episodes})
    for episode in range(start, args.episodes + 1):
        archive = output / 'episodes' / 'train' / f'episode_{episode:06d}.npz'
        _, _, metrics = runner.run_episode('random', seed=args.seed + episode,
            archive_path=archive, archive_metadata={'method': 'random', 'phase': 'train', 'episode': episode})
        record = dict(phase='train', episode=episode, **metrics)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        atomic_json(output / 'progress.json', {'last_episode': episode, 'episode_budget': args.episodes})
        print(json.dumps({k: v for k, v in record.items() if k not in {'choices', 'branch_choices'}},
                         allow_nan=False), flush=True)
    print(json.dumps({'status': 'completed', 'episodes': args.episodes, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
