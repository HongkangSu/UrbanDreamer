"""Isolated, reproducible full-budget imagination profiling from a frozen checkpoint."""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--memory-fraction', type=float, default=.035)
    parser.add_argument('--passes', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-profile', action='store_true')
    args = parser.parse_args()
    if args.passes < 1 or args.threads < 1 or not 0 < args.memory_fraction <= 1:
        parser.error('passes and threads must be positive and memory fraction must be in (0, 1]')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    if args.source_dir:
        sys.path.insert(0, str(args.source_dir.resolve()))
    import torch
    from urban_dreamer.learning import training
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.checkpoint import load_checkpoint
    from urban_dreamer.workflow.config import load_config
    from urban_dreamer.workflow.pipeline import combined_input_hash
    from tools.train_codec_ready import SparseGraphCodec

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('profiling output must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        free, _ = torch.cuda.mem_get_info(args.device)
        if free < 1024 ** 3:
            raise RuntimeError('less than 1 GiB free; refusing profiling contention')
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    config = load_config(args.snapshot_dir / 'run_config.toml')
    config = replace(config, run=replace(config.run, output_dir=output, device=args.device))
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    checkpoint = load_checkpoint(output_dir=args.snapshot_dir,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(prepared))
    trainer = training.MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology)
    trainer.world_model.flow_projector.__class__ = SparseGraphCodec
    trainer.world_model.flow_projector.register_buffer('sparse_adjacency',
        trainer._tensor(topology.network.adjacency).to_sparse().coalesce(), persistent=False)
    trainer.load_state_dict(checkpoint.training_state)
    physical_work = dict(windows=0, simulated_seconds=0., active_vehicle_seconds=0.)
    original_advance = training.advance_imagined_window
    def counted_advance(*advance_args, **advance_kwargs):
        result = original_advance(*advance_args, **advance_kwargs)
        physical_work['windows'] += 1
        physical_work['simulated_seconds'] += result.end_time - advance_kwargs['start_time']
        physical_work['active_vehicle_seconds'] += result.active_vehicle_seconds
        return result
    training.advance_imagined_window = counted_advance
    optimizer_steps = {name: 0 for name in ('actor', 'latent_critic', 'event_critic')}
    for name in optimizer_steps:
        original = trainer.optimizers[name].step
        def counted_step(*step_args, _name=name, _step=original, **step_kwargs):
            result = _step(*step_args, **step_kwargs)
            optimizer_steps[_name] += 1
            return result
        trainer.optimizers[name].step = counted_step
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    identity = dict(checkpoint=str(checkpoint.path), checkpoint_sha256=hashlib.sha256(checkpoint.path.read_bytes()).hexdigest(),
        source_file=training.__file__, device=args.device,
        completed_online_episodes=checkpoint.next_unit, threads=args.threads,
        rollouts_per_pass=training.IMAGINED_ROLLOUTS_PER_PASS,
        horizon=training.IMAGINED_HORIZON, passes=args.passes,
        cprofile_enabled=not args.no_profile,
        imagination_workers=int(os.environ.get('URB_IMAGINATION_WORKERS', trainer.metrics.get('imagination_workers', 0))),
        started_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        note='Isolated optimizer updates, no live checkpoint writes; saved checkpoint RNG restored.')
    (output / 'protocol.json').write_text(json.dumps(identity, indent=2) + '\n')
    rows = []
    profiler = cProfile.Profile()
    try:
        for index in range(args.passes):
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            if not args.no_profile:
                profiler.enable()
            metrics = trainer._train_imagination_pass()
            parallel_work = getattr(trainer, '_last_parallel_work', None)
            if parallel_work is not None:
                physical_work['windows'] += parallel_work['windows']
                physical_work['simulated_seconds'] += parallel_work['seconds']
                physical_work['active_vehicle_seconds'] += parallel_work['active_vehicle_seconds']
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            if not args.no_profile:
                profiler.disable()
            row = dict(pass_index=index, seconds=time.perf_counter() - start,
                finished_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), **metrics)
            row['optimizer_steps'] = dict(optimizer_steps)
            row['physical_work'] = dict(physical_work)
            if args.device.startswith('cuda'):
                row['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
                row['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
            rows.append(row)
            print(json.dumps(row), flush=True)
            (output / 'results.json').write_text(json.dumps(dict(protocol=identity, passes=rows), indent=2) + '\n')
        if not args.no_profile:
            profiler.dump_stats(str(output / 'profile.pstats'))
            for sort in ('cumulative', 'tottime'):
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats(sort).print_stats(100)
                (output / f'profile_{sort}.txt').write_text(stream.getvalue())
    finally:
        training.advance_imagined_window = original_advance
        trainer.close()


if __name__ == '__main__':
    main()
