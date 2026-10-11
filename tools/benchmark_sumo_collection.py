"""Compare isolated SUMO backends with a frozen actor, demand, and CPU RNG."""
from __future__ import annotations

import argparse
import copy
import cProfile
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import random
import shutil
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--demand-index', type=int, default=42)
    parser.add_argument('--profile-pair', action='store_true')
    parser.add_argument('--backends', nargs='+', choices=('traci', 'libsumo'),
                        default=['traci', 'libsumo'])
    args = parser.parse_args()
    if len(set(args.backends)) != len(args.backends):
        parser.error('each backend may be requested only once')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(args.source_dir.resolve()))
    import numpy as np
    import torch
    import libsumo
    from urban_dreamer.learning.training import MainlineTrainer
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import NextHopCandidates, load_traffic_topology
    from urban_dreamer.workflow.checkpoint import load_checkpoint
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    config = load_config(args.snapshot_dir / 'run_config.toml')
    config = replace(config, run=replace(config.run, output_dir=output, device='cpu'))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    checkpoint = load_checkpoint(output_dir=args.snapshot_dir,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(prepared))
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    trainer.load_state_dict(checkpoint.training_state)
    binary = shutil.which('sumo') or str(Path(sys.executable).parent / 'sumo')
    binary_version = subprocess.check_output([binary, '--version'], text=True).splitlines()[0]
    embedded_version = libsumo.getVersion()
    versions_match = binary_version.split()[-1] == embedded_version[1].split()[-1]
    actor_hash = lambda: hashlib.sha256(b''.join(
        p.detach().cpu().numpy().tobytes() for p in trainer.actor.parameters())).hexdigest()
    identity = dict(checkpoint=str(checkpoint.path), completed_online_episodes=checkpoint.next_unit,
        source_dir=str(args.source_dir.resolve()), device='cpu', threads=2,
        backends=args.backends,
        demand_seed=config.run.seed, demand_index=args.demand_index,
        binary=binary, binary_version=binary_version, libsumo_version=embedded_version,
        sumo_versions_match=versions_match, actor_sha256=actor_hash(),
        duration_seconds=trainer._episode_end_time(), observation_seconds=1,
        note='Backend override is isolated to experimental config; no training checkpoint is saved. '
             'Archive metadata records actual backend/config. Version mismatch forbids backend-only attribution.')
    (output / 'protocol.json').write_text(json.dumps(identity, indent=2) + '\n')
    metrics = copy.deepcopy(trainer.metrics)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           copy.deepcopy(trainer.rng.bit_generator.state))
    runs = []
    comparisons = []

    def compare(first, second):
        checks = {}
        with np.load(first['archive'], allow_pickle=False) as a, np.load(second['archive'], allow_pickle=False) as b:
            for name in a.files:
                if name == 'summary_json':
                    continue
                left, right = a[name], b[name]
                row = dict(shape_equal=left.shape == right.shape, exactly_equal=bool(np.array_equal(left, right)))
                if left.shape == right.shape and np.issubdtype(left.dtype, np.number):
                    row['max_abs_difference'] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.
                checks[name] = row
            sa = json.loads(a['summary_json'].tobytes()); sb = json.loads(b['summary_json'].tobytes())
        demand_equal = sa['metadata']['demand_sha256'] == sb['metadata']['demand_sha256']
        physical_metrics = {name: dict(first=sa['metrics'][name], second=sb['metrics'][name],
                                      exactly_equal=sa['metrics'][name] == sb['metrics'][name])
                            for name in sa['metrics'] if not name.endswith('wall_seconds')}
        return dict(first=first['name'], second=second['name'], demand_equal=demand_equal,
            physical_arrays=checks, physical_metrics=physical_metrics,
            travel_times_exact=sa['travel_times'] == sb['travel_times'],
            termination_exact=(sa['completed'], sa['episode_truncated']) == (sb['completed'], sb['episode_truncated']),
            trajectories_exact=demand_equal and all(v['exactly_equal'] for v in checks.values())
                and sa['travel_times'] == sb['travel_times']
                and all(v['exactly_equal'] for v in physical_metrics.values()),
            backend_only_control_valid=versions_match,
            speedup_first_over_second=first['sumo_wall_seconds'] / second['sumo_wall_seconds'])

    try:
        for profiled in ([False, True] if args.profile_pair else [False]):
            pair = []
            for backend in args.backends:
                name = ('profile_' if profiled else 'wall_') + backend
                folder = output / name
                trainer.config = replace(config, run=replace(config.run, output_dir=folder),
                    simulation=replace(config.simulation, backend=backend))
                trainer.metrics = copy.deepcopy(metrics)
                trainer._next_environment_episode = args.demand_index
                trainer.candidates = NextHopCandidates(topology.network)
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                trainer.rng.bit_generator.state = copy.deepcopy(rng[3])
                profile = cProfile.Profile()
                started = time.perf_counter()
                print(json.dumps(dict(phase='start', name=name, utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))), flush=True)
                if profiled:
                    profile.enable()
                collected = trainer._run_episode(mode='online', episode_index=args.demand_index)
                if profiled:
                    profile.disable()
                elapsed = time.perf_counter() - started
                row = dict(name=name, backend=backend, profiled=profiled, wrapper_seconds=elapsed,
                    sumo_wall_seconds=collected.metrics['online_wall_seconds'], metrics=collected.metrics,
                    archive=str(folder / 'episodes' / 'online' / f'episode_{args.demand_index:06d}.npz'),
                    actor_unchanged=actor_hash() == identity['actor_sha256'])
                if not row['actor_unchanged']:
                    raise RuntimeError('frozen actor changed during collection')
                runs.append(row); pair.append(row)
                print(json.dumps(row), flush=True)
                if profiled:
                    profile.dump_stats(str(folder / 'profile.pstats'))
                    for sort in ('cumulative', 'tottime'):
                        stream = io.StringIO()
                        pstats.Stats(profile, stream=stream).sort_stats(sort).print_stats(80)
                        (folder / f'profile_{sort}.txt').write_text(stream.getvalue())
                (output / 'results.json').write_text(json.dumps(dict(protocol=identity, runs=runs,
                    comparisons=comparisons), indent=2) + '\n')
            if len(pair) == 2:
                comparisons.append(compare(*pair))
            (output / 'results.json').write_text(json.dumps(dict(protocol=identity, runs=runs,
                comparisons=comparisons), indent=2) + '\n')
            if len(pair) == 2:
                print(json.dumps(comparisons[-1]), flush=True)
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
