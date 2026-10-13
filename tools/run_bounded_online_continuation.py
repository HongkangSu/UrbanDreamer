"""Resume a bounded-critic run with argmax evaluation by default."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tools.run_online_pilot import (OnlinePilot, PilotTrainer, atomic_json,
    learning_budgets, sha256_file)
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.checkpoint import load_checkpoint, save_checkpoint, CheckpointPhase
from urban_dreamer.workflow.pipeline import combined_input_hash
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.learning.training import evaluation_protocol


def prepare_progress(output, checkpoint, protocol, config):
    """Keep training history; restart evaluation baseline for the changed behavior policy."""
    if (output / 'pilot_progress.json').exists():
        return
    state = checkpoint.training_state
    history = state['metrics'].get('online_pilot_progress')
    if history is None and checkpoint.next_unit == 0 and state['metrics'].get('online_episodes_completed', 0) == 0:
        history = dict(schema_version=1, online=[], before=[], after=[], milestones=[], phase='before_evaluation')
    if history is None or len(history['online']) != checkpoint.next_unit:
        raise ValueError('continuation requires checkpoint-authoritative episode history')
    atomic_json(output / 'pre_migration_pilot_progress.json', history)
    # The archived policy protocol may differ in sampling or guidance. Establish
    # a new paired baseline instead of mixing its evaluations with this run.
    progress = copy.deepcopy(history)
    progress.update(before=[], after=[], milestones=[], phase='before_evaluation',
                    critic_prior_migration_episode=checkpoint.next_unit)
    state['metrics']['online_pilot_progress'] = progress
    save_checkpoint(output_dir=output, config_fingerprint=config.semantic_fingerprint,
        input_hash=checkpoint.input_hash, phase=CheckpointPhase.ONLINE_TRAINING,
        next_unit=checkpoint.next_unit, training_state=state)
    atomic_json(output / 'pilot_protocol.json', protocol)
    atomic_json(output / 'pilot_progress.json', progress)


class StopAtBoundary(Exception):
    pass


class BoundaryPilot(OnlinePilot):
    stop_requested = False

    def periodic_evaluation(self):
        if self.stop_requested:
            raise StopAtBoundary()
        return super().periodic_evaluation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.25)
    parser.add_argument('--evaluation-every', type=int, default=20)
    parser.add_argument('--evaluation-temperature', type=float, default=.2)
    parser.add_argument('--evaluation-policy', choices=('sampled', 'argmax'), default='argmax')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if (args.threads < 1 or not 0 < args.gpu_memory_fraction <= 1 or args.evaluation_every < 0
            or not math.isfinite(args.evaluation_temperature) or args.evaluation_temperature <= 0):
        parser.error('invalid resource/evaluation settings')
    evaluation_temperature = None if args.evaluation_policy == 'argmax' else args.evaluation_temperature
    output = args.output_dir.resolve()
    manifest_path = ROOT / 'source_manifest.json'
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text())
        for name, digest in source_manifest['files'].items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError(f'frozen execution source changed: {name}')
    config = load_config(output / 'run_config.toml')
    if config.method.critic_value_mode != 'bounded_cohort_v1':
        raise ValueError('this continuation requires bounded critics')
    # Direct continuations use the same parallel rollout setup as queued runs.
    os.environ.setdefault('URB_IMAGINATION_WORKERS', '8')
    migration = json.loads((output / 'critic_migration.json').read_text())
    if sha256_file(output / 'source_checkpoint.pt') != migration['source_checkpoint_sha256']:
        raise ValueError('source checkpoint ancestry changed')
    torch.set_num_threads(args.threads)
    device = torch.device(config.run.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    prepared = validate_prepared_inputs(config)
    checkpoint = load_checkpoint(output_dir=output,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=combined_input_hash(prepared))
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING:
        raise ValueError('continuation checkpoint must be in online training')
    policy = evaluation_protocol(evaluation_temperature,
        prior_weight=config.method.sp_prior_evaluation_weight, demand_seed=config.method.evaluation_demand_seed)
    protocol = dict(schema_version=1, kind='bounded_online_continuation',
        source=dict(path=str(output / 'source_checkpoint.pt'), sha256=migration['source_checkpoint_sha256']),
        migration_episode=migration['next_unit'], config_fingerprint=config.semantic_fingerprint,
        budgets=learning_budgets(config), online_episode_budget=config.run.online_training_episodes,
        evaluation_every=args.evaluation_every,
        critic=dict(mode=config.method.critic_value_mode, target_tau=config.method.critic_target_tau,
                    terminal_aux_ratio=config.method.critic_terminal_aux_ratio),
        prior=dict(enabled=config.method.sp_prior_weight > 0.,
            cost_mode=config.method.sp_prior_cost_mode, weight=config.method.sp_prior_weight,
            evaluation_weight=config.method.sp_prior_evaluation_weight,
            speed_floor_mps=config.method.sp_prior_dynamic_speed_floor_mps,
            temperature_seconds=config.method.sp_prior_temperature,
            hold_episodes=config.method.sp_prior_hold_episodes, off_episode=config.method.sp_prior_off_episode),
        evaluation=policy, evaluation_policy=policy['archive_policy'],
        evaluation_baseline='new paired baseline with recorded evaluation policy; previous history archived',
        source_root=str(ROOT))
    prepare_progress(output, checkpoint, protocol, config)
    del checkpoint
    trainer = PilotTrainer(config=config, prepared_inputs=prepared, topology=load_traffic_topology(prepared),
                           pilot_log=output / 'pilot_metrics.jsonl')
    trainer.evaluation_temperature = evaluation_temperature
    trainer.world_model.flow_projector.__class__ = SparseGraphCodec
    trainer.world_model.flow_projector.register_buffer('sparse_adjacency',
        trainer._tensor(trainer.network.adjacency).to_sparse().coalesce(), persistent=False)
    runner = BoundaryPilot(trainer, output=output, input_hash=combined_input_hash(prepared), protocol=protocol,
        online_episodes=config.run.online_training_episodes, evaluation_every=args.evaluation_every)
    def request_stop(signum, frame):
        runner.stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    status = dict(pid=os.getpid(), phase='loading', updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                  migration_episode=migration['next_unit'], source_root=str(ROOT), command=sys.argv)
    atomic_json(output / 'runtime_status.json', status)
    try:
        if args.verify_only:
            runner.resume()
            trainer.validate_codec_ready()
            status['phase'] = 'verified'
        else:
            status['phase'] = 'online_running'
            atomic_json(output / 'runtime_status.json', status)
            runner.run()
            status['phase'] = 'completed'
    except StopAtBoundary:
        status['phase'] = 'stopped_at_checkpoint_boundary'
    except BaseException as error:
        status.update(phase='failed', error=repr(error))
        raise
    finally:
        trainer.close()
        status['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        atomic_json(output / 'runtime_status.json', status)


if __name__ == '__main__':
    main()
