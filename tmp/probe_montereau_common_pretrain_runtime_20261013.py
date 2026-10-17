"""Private consecutive B64/B63 updates using Montereau's production checkpoint.

Writes only benchmark artifacts. Never saves a training checkpoint or changes
production configuration, data, optimizer state, or running processes.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CITY = ROOT / 'outputs/two_city_urbandreamer_pretrain_supplement_20261012/montereau_fault_yonne'
SOURCE = CITY.parent / 'source_recoveryfix_20261012'
OUTPUT = CITY / 'benchmark_common_runtime_20261013'
sys.path.insert(0, str(SOURCE))


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cuda:0', 'cuda:1'], default='cuda:0')
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--activation-offload', action='store_true',
                        help='optional same-batch CPU offload comparison; default is disabled')
    args = parser.parse_args()
    if args.pairs < 2:
        parser.error('at least two consecutive full/tail pairs are required')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / ('result_offload.json' if args.activation_offload else 'result.json')
    result = dict(status='starting', benchmark_only=True, production_modified=False,
                  source_root=str(SOURCE), device=args.device, rows=[],
                  production_quality_revalidation=False, online_replay_loaded=False,
                  note='Private selected trajectories; reuses checkpoint held-out authorization without making a new quality claim.')

    def publish(phase):
        result['phase'] = phase
        result['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        write_json(result_path, result)
        print(json.dumps(dict(phase=phase, utc=result['updated_utc'], updates=len(result['rows']))), flush=True)

    trainer = None
    try:
        import torch
        from urban_dreamer.learning.training import MainlineTrainer
        from urban_dreamer.learning.pretrain_cache import CompactPretrainDataset
        from urban_dreamer.learning.offline_data import _identity
        from urban_dreamer.models.world_model import WORLD_MODEL_DIM
        from urban_dreamer.traffic.routing import validate_prepared_inputs
        from urban_dreamer.traffic.topology import load_traffic_topology
        from urban_dreamer.workflow.config import load_config
        from urban_dreamer.workflow.pipeline import combined_input_hash
        from tools.run_complete_pretrain import compile_pretrain_blocks
        from tools.train_codec_ready import SparseGraphCodec

        os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '2')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(OUTPUT / 'inductor_cache')
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(1.0, device)
        free, total = torch.cuda.mem_get_info(device)
        result.update(gpu_name=torch.cuda.get_device_name(device), free_before_mib=free / 1024**2,
                      total_mib=total / 1024**2, allocator_fraction=1.0)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 pretraining requires a supported CUDA device')

        base = load_config(CITY / 'world_model/run_config.toml')
        resources = base.resources
        expected = dict(world_model_initialization_batch_size=64,
            world_model_gradient_accumulation_steps=1, world_model_initialization_sequence_length=1,
            world_model_pretrain_precision='bf16', world_model_pretrain_compile_blocks=True,
            world_model_pretrain_space_chunk=64, world_model_pretrain_time_chunk=65536,
            world_model_pretrain_mlp_chunk=65536, world_model_pretrain_activation_checkpointing=True)
        if any(getattr(resources, key) != value for key, value in expected.items()):
            raise ValueError('production configuration no longer matches the common runtime probe')
        if (base.method.fm_training_mode != 'single_step'
                or base.method.exposure_readout_supervision != 'observed_endpoint'
                or base.run.world_model_initialization_epochs != 1):
            raise ValueError('production training semantics differ from the common runtime probe')
        result['profile'] = dict(batch_size=64, tail_size=63, accumulation_steps=1, epoch_budget=1,
            sequence_length=1, precision='bf16', compile_blocks=True, space_chunk=64,
            time_chunk=65536, mlp_chunk=65536, activation_checkpointing=True,
            activation_offload=args.activation_offload, tail_execution_batch=None, threads=4,
            fm_training_mode='single_step', readout_supervision='observed_endpoint')
        publish('copying_checkpoint')
        checkpoint_path = CITY / 'world_model/checkpoints/latest.pt'
        snapshot_path = OUTPUT / 'source_checkpoint.pt'
        shutil.copy2(checkpoint_path, snapshot_path)
        digest = hashlib.sha256()
        with snapshot_path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        payload = torch.load(snapshot_path, map_location='cpu', weights_only=True, mmap=True)
        if payload['config_fingerprint'] != base.semantic_fingerprint or payload['phase'] != 'pretrain':
            raise ValueError('snapshot checkpoint configuration or phase mismatch')
        state = payload['training_state']
        result.update(source_checkpoint=str(checkpoint_path), snapshot_sha256=digest.hexdigest(),
            source_config_fingerprint=payload['config_fingerprint'], source_input_hash=payload['input_hash'],
            source_world_model_updates=state['metrics']['world_model_updates'],
            source_epochs_completed=state['metrics']['world_model_initialization_epochs_completed'])

        manifest_path = Path(base.data.pretrain_cache_manifest)
        full_manifest = json.loads(manifest_path.read_text())
        if full_manifest['identity'] != _identity({k: v for k, v in full_manifest.items() if k != 'identity'}):
            raise ValueError('production cache manifest identity mismatch')
        candidates = sorted((entry for entry in full_manifest['entries']
                             if entry['split'] == 'train' and entry['windows'] >= 127),
                            key=lambda entry: entry['events'], reverse=True)
        selected = []
        for entry in candidates:
            if all(entry['demand_group'] != chosen['demand_group'] for chosen in selected):
                selected.append(entry)
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise ValueError('need two independent real training trajectories with 127 windows')
        selected_paths = {entry['path'] for entry in selected}
        subset = copy.deepcopy(full_manifest)
        subset['entries'] = copy.deepcopy(selected)
        subset['sources'] = [copy.deepcopy(source) for source in full_manifest['sources']
                             if source['path'] in selected_paths]
        subset.update(duplicate_files=0, train_episodes=2, validation_episodes=0)
        subset['identity'] = _identity({k: v for k, v in subset.items() if k != 'identity'})
        subset_path = OUTPUT / 'private_subset_manifest.json'
        write_json(subset_path, subset)
        result.update(production_cache_identity=full_manifest['identity'], private_cache_identity=subset['identity'],
            selected_episodes=[{key: entry[key] for key in ('path', 'cache_path', 'windows', 'events', 'demand_group')}
                               for entry in selected])
        config = replace(base, run=replace(base.run, device=args.device, output_dir=OUTPUT / 'private_unused'))
        prepared = validate_prepared_inputs(config)
        if payload['input_hash'] != combined_input_hash(prepared):
            raise ValueError('production checkpoint input hash mismatch')
        topology = load_traffic_topology(prepared)
        publish('verifying_two_selected_sources')
        offline = CompactPretrainDataset(config, prepared, topology, 1,
            cache_manifest_path=subset_path, split='train')
        # Retain the original held-out authorization. This metadata stand-in
        # prevents a full-data startup scan and cannot perform quality validation.
        validation_identity = state['replay']['codec_validation']['manifest_identity']
        validation = SimpleNamespace(manifest={'identity': validation_identity})
        publish('constructing_private_trainer')
        trainer = MainlineTrainer(config=config, prepared_inputs=prepared, topology=topology,
                                 _offline_data=offline, _codec_validation_data=validation)
        for name, model_state in state['models'].items():
            getattr(trainer, name).load_state_dict(model_state, strict=True)
        for name in ('world_model_initial', 'active_fraction', 'decoder_pretrain'):
            trainer.optimizers[name].load_state_dict(state['optimizers'][name])
        trainer.metrics = copy.deepcopy(state['metrics'])
        trainer._validate_codec_pretraining_ready()
        trainer._set_dynamics_codec_gradients()
        codec = trainer.world_model.flow_projector
        codec.__class__ = SparseGraphCodec
        codec.register_buffer('sparse_adjacency', trainer._tensor(trainer.network.adjacency).to_sparse().coalesce(),
                              persistent=False)
        result.update(edge_count=trainer.edge_count,
            source_quality_authorization_identity=validation_identity,
            loaded_optimizers=['world_model_initial', 'active_fraction', 'decoder_pretrain'],
            optimizer_learning_rates={name: [group['lr'] for group in trainer.optimizers[name].param_groups]
                                     for name in ('world_model_initial', 'active_fraction', 'decoder_pretrain')})
        result['compiled_blocks'] = compile_pretrain_blocks(trainer)
        if args.activation_offload:
            from tools.pretrain_activation_offload import enable_pretrain_activation_offload
            enable_pretrain_activation_offload(trainer)
        counts = dict(compiled_full=0, eager_tail=0)
        for block in trainer.world_model.dynamics.transformer.layers:
            original = block.forward
            full_shape = (64, 1, block.space.attn_mask.shape[-1], WORLD_MODEL_DIM)

            def tracked_forward(x, original=original, full_shape=full_shape):
                compiled = (tuple(x.shape) == full_shape and x.device.type == 'cuda'
                    and x.dtype in (torch.float32, torch.bfloat16) and torch.is_grad_enabled()
                    and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16)
                counts['compiled_full' if compiled else 'eager_tail'] += 1
                return original(x)

            block.forward = tracked_forward

        def snapshot(module, encoder_only=False):
            return {name: value.detach().cpu().clone() for name, value in module.named_parameters()
                    if not encoder_only or not name.startswith('decoder_')}

        encoder_before = snapshot(codec, encoder_only=True)
        torch.cuda.reset_peak_memory_stats(device)
        publish('consecutive_full_and_tail_updates')
        for index, size in enumerate([64, 63] * args.pairs):
            episode = (index // 2) % 2
            starts = list(range(64)) if size == 64 else list(range(64, 127))
            loading_start = time.perf_counter()
            batch = offline.batch_from_starts(episode, starts, fm_only=True)
            loading_seconds = time.perf_counter() - loading_start
            codec_before, dynamics_before = snapshot(codec), snapshot(trainer.world_model.dynamics)
            calls_before = dict(counts)
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            metrics = trainer._update_world(offline, initial=True, batches=[batch])
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            finite_metrics = all(math.isfinite(value) for value in metrics.values() if isinstance(value, (int, float)))
            gradients_finite = all(bool(torch.isfinite(p.grad).all()) for p in trainer.world_model.parameters()
                                   if p.grad is not None)
            parameters_finite = all(bool(torch.isfinite(p).all()) for p in trainer.world_model.parameters())
            encoder_unchanged = all(torch.equal(old, dict(codec.named_parameters())[name].detach().cpu())
                                    for name, old in encoder_before.items())
            encoder_frozen = all(not p.requires_grad and p.grad is None for name, p in codec.named_parameters()
                                 if not name.startswith('decoder_'))
            decoder_changed = [name for name, p in codec.named_parameters()
                               if not torch.equal(codec_before[name], p.detach().cpu())]
            dynamics_changed = sum(not torch.equal(dynamics_before[name], p.detach().cpu())
                                   for name, p in trainer.world_model.dynamics.named_parameters())
            dispatch = {key: counts[key] - calls_before[key] for key in counts}
            free, total = torch.cuda.mem_get_info(device)
            row = dict(index=index, batch_size=size, private_episode=episode,
                loading_seconds=loading_seconds, update_seconds=seconds, metrics=metrics,
                gradients_finite=gradients_finite, parameters_finite=parameters_finite,
                encoder_frozen=encoder_frozen, encoder_unchanged=encoder_unchanged,
                changed_decoder_parameters=decoder_changed, changed_dynamics_tensors=dynamics_changed,
                dispatch_calls=dispatch, allocated_mib=torch.cuda.memory_allocated(device) / 1024**2,
                reserved_mib=torch.cuda.memory_reserved(device) / 1024**2,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
                gpu_free_after_mib=free / 1024**2)
            result['rows'].append(row)
            publish('consecutive_full_and_tail_updates')
            expected_dispatch = 'compiled_full' if size == 64 else 'eager_tail'
            if not (finite_metrics and gradients_finite and parameters_finite and encoder_frozen
                    and encoder_unchanged and decoder_changed and dynamics_changed
                    and all(name.startswith('decoder_') for name in decoder_changed)
                    and dispatch[expected_dispatch] > 0):
                raise RuntimeError('gradient/update/dispatch invariant failed')
            del codec_before, dynamics_before, batch
        result.update(status='complete', private_optimizer_updates=len(result['rows']),
            full_updates=sum(row['batch_size'] == 64 for row in result['rows']),
            tail_updates=sum(row['batch_size'] == 63 for row in result['rows']),
            peak_allocated_mib=max(row['peak_allocated_mib'] for row in result['rows']),
            peak_reserved_mib=max(row['peak_reserved_mib'] for row in result['rows']),
            private_world_model_updates=trainer.metrics['world_model_updates'])
        publish('complete')
    except Exception as error:
        result.update(status='failed', error=str(error), traceback=traceback.format_exc())
        if trainer is not None:
            result.update(failed_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
                          failed_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2)
        publish('failed')
        raise
    finally:
        if trainer is not None:
            trainer.close()


if __name__ == '__main__':
    main()
