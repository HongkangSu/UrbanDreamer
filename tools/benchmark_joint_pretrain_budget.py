"""Bounded private production FM plus supervised-decoder throughput benchmark."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def profile(args):
    import torch
    from urban_dreamer.learning.training import MainlineTrainer
    from urban_dreamer.traffic.routing import validate_prepared_inputs
    from urban_dreamer.traffic.topology import load_traffic_topology
    from urban_dreamer.workflow.config import DataConfig, load_config
    from tools.train_codec_ready import SparseGraphCodec
    from tools.run_complete_pretrain import compile_pretrain_blocks
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < args.memory_fraction * total + args.admission_margin_mib * 1024**2:
        raise RuntimeError('insufficient free memory for bounded benchmark')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    config = load_config(args.config)
    config = replace(config,
        run=replace(config.run, device=args.device, output_dir=args.output_dir / 'private_unused'),
        data=DataConfig(pretrain_cache_manifest=args.cache_manifest, validation_cache_manifest=args.cache_manifest),
        codec=replace(config.codec, decoder_pretrain_enabled=True, decoder_online_enabled=True, decoder_allow_unready=True),
        method=replace(config.method, fm_training_mode='single_step', exposure_readout_supervision='observed_endpoint', fm_steps=100, fm_solver='euler'),
        resources=replace(config.resources, world_model_initialization_sequence_length=1,
            world_model_initialization_batch_size=args.batch, world_model_pretrain_precision='bf16',
            world_model_pretrain_compile_blocks=True, world_model_pretrain_space_chunk=64,
            world_model_pretrain_time_chunk=65536, world_model_pretrain_mlp_chunk=65536,
            world_model_pretrain_activation_checkpointing=True))
    prepared = validate_prepared_inputs(config)
    trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=load_traffic_topology(prepared))
    codec = trainer.world_model.flow_projector
    payload = torch.load(args.codec, map_location='cpu', weights_only=False)
    codec.load_state_dict(payload['codec'], strict=True)
    codec.__class__ = SparseGraphCodec
    codec.register_buffer('sparse_adjacency', trainer._tensor(trainer.network.adjacency).to_sparse().coalesce(), persistent=False)
    before = {name: value.detach().cpu().clone() for name, value in codec.named_parameters()}
    compiled = compile_pretrain_blocks(trainer)
    if args.activation_offload:
        from tools.pretrain_activation_offload import enable_pretrain_activation_offload
        enable_pretrain_activation_offload(trainer)
    if args.tail_execution_batch:
        from tools.pretrain_tail_execution import enable_pretrain_tail_execution
        enable_pretrain_tail_execution(trainer, max_batch=args.tail_execution_batch)
    entries = trainer.offline_data.manifest['entries']
    episode = max(range(len(entries)), key=lambda i: entries[i]['windows'])
    available = entries[episode]['windows']
    tails = args.tail_batches or {32: [16], 64: [32], 128: [64, 96]}[args.batch]
    if any(size < 1 or size >= args.batch for size in tails):
        raise ValueError('tail batches must be positive and smaller than the full batch')
    result = dict(batch_size=args.batch, status='running', compiled_blocks=compiled,
        precision='bf16', decoder_supervision=True, encoder_frozen=True,
        fm_training_mode='single_step', readout_supervision='observed_endpoint',
        tail_update_seconds={}, sample_loading_seconds={}, measurements={},
        memory_fraction=args.memory_fraction, fresh_dynamics=True, codec_source=str(args.codec),
        activation_offload=args.activation_offload,
        tail_execution_batch=args.tail_execution_batch,
        source_episode_windows=available, full_batch_repeated_starts=max(args.batch-available, 0),
        note='Full B128 repeats one valid start because real episodes have at most127 windows; throughput only, production retains tails without padding.')
    output = args.output_dir / f'batch_{args.batch}.json'
    write(output, result)
    for size in [args.batch] + tails:
        host = []
        for _ in range(3):
            started = time.perf_counter()
            batch = trainer.offline_data.batch_from_starts(episode, [index % available for index in range(size)], fm_only=True)
            host.append(time.perf_counter() - started)
        def update():
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            metrics = trainer._update_world(trainer.offline_data, initial=True, batches=[batch])
            torch.cuda.synchronize(device)
            return time.perf_counter() - started, metrics
        torch.cuda.reset_peak_memory_stats(device)
        warm = [update()[0] for _ in range(3)]
        warm_peak_allocated = torch.cuda.max_memory_allocated(device)/1024**2
        warm_peak_reserved = torch.cuda.max_memory_reserved(device)/1024**2
        torch.cuda.reset_peak_memory_stats(device)
        rows = [update() for _ in range(args.repeats)]
        times = [row[0] for row in rows]
        median = statistics.median(times)
        result['sample_loading_seconds'][str(size)] = dict(cold=host[0], warm_median=statistics.median(host[1:]))
        result['measurements'][str(size)] = dict(warmup_seconds=warm, steady_seconds=times, samples_per_second=size / median,
            warmup_peak_allocated_mib=warm_peak_allocated, warmup_peak_reserved_mib=warm_peak_reserved,
            metrics_last=rows[-1][1], peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2)
        result['peak_allocated_mib'] = max(result.get('peak_allocated_mib', 0), torch.cuda.max_memory_allocated(device)/1024**2)
        result['peak_reserved_mib'] = max(result.get('peak_reserved_mib', 0), torch.cuda.max_memory_reserved(device)/1024**2)
        if size == args.batch:
            result['full_update_seconds_median'] = median
        else:
            result['tail_update_seconds'][str(size)] = median
        write(output, result)
        print(f'batch={args.batch} actual={size} seconds={median:.4f} samples/s={size / median:.2f}', flush=True)
    changed = [name for name, value in codec.named_parameters() if not torch.equal(before[name], value.detach().cpu())]
    result.update(status='complete', changed_codec_parameters=changed,
        encoder_unchanged=all(name.startswith('decoder_') for name in changed),
        decoder_updated=bool(changed), decoder_gradients_finite=all(bool(torch.isfinite(value.grad).all())
            for name, value in codec.named_parameters() if name.startswith('decoder_') and value.grad is not None))
    if not result['encoder_unchanged'] or not result['decoder_updated'] or not result['decoder_gradients_finite']:
        raise RuntimeError('codec gradient/update invariant failed')
    write(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/ingolstadt_custom_offline.toml')
    parser.add_argument('--cache-manifest', type=Path, default=ROOT/'outputs/ingolstadt_noincident_pretrain_20261008/cache/cache_manifest.json')
    parser.add_argument('--codec', type=Path, default=ROOT/'outputs/ingolstadt_noincident_pretrain_20261008/decoder_calibration/calibration_latest.pt')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/ingolstadt_full_retrain_20261009/benchmark')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--memory-fraction', type=float, default=.30)
    parser.add_argument('--admission-margin-mib', type=int, default=512)
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--tail-execution-batch', type=int)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--batch', type=int, choices=[32,64,128])
    parser.add_argument('--tail-batches', nargs='+', type=int,
                        help='explicit production remainder sizes, e.g. 63 for 127-window episodes at batch64')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch:
        try:
            profile(args)
        except Exception as exc:
            path=args.output_dir/f'batch_{args.batch}.json'
            result=json.loads(path.read_text()) if path.exists() else dict(batch_size=args.batch)
            result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
            write(path, result)
            raise
    else:
        result=dict(status='running', profiles=[], benchmark_only=True, production_modified=False)
        write(args.output_dir/'results.json', result)
        for size in [32,64,128]:
            command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--batch', str(size)]
            subprocess.run(command, check=False)
            result['profiles'].append(json.loads((args.output_dir/f'batch_{size}.json').read_text()))
            write(args.output_dir/'results.json', result)
        result['status']='complete'
        write(args.output_dir/'results.json', result)

if __name__ == '__main__':
    main()
