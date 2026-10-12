"""Experimental component policy learning with genuine SUMO and actor/V/Q updates.

Component checkpoints are development-screen dynamics, not completed production
pretraining. Admission records this distinction and measures codec readiness.
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
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from tools.component_experiment_data import atomic_json, dataset, sha256
from tools import run_online_pilot as pilot
from tools.train_codec_ready import SparseGraphCodec
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.offline_data import _identity
from urban_dreamer.models.experimental_dynamics import VARIANTS, apply_variant, experimental_loss
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config


def tensor_digest(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
    return digest.hexdigest()


def validate_component_source(experiment_dir, checkpoint, expected_sha256, *, variant, seed):
    """Validate immutable campaign and explicit offline-checkpoint admission."""
    experiment_dir, checkpoint = Path(experiment_dir), Path(checkpoint)
    parent = json.loads((experiment_dir / 'protocol.json').read_text())
    if sha256(checkpoint) != expected_sha256:
        raise ValueError('component checkpoint SHA256 mismatch')
    for filename, key in [('run_config.toml', 'config_sha256'),
                          ('checkpoints/latest.pt', 'checkpoint_sha256')]:
        if sha256(experiment_dir / 'snapshot' / filename) != parent[key]:
            raise ValueError(f'campaign snapshot digest mismatch: {filename}')
    manifest = json.loads((experiment_dir / 'development_manifest.json').read_text())
    identity = _identity({k: v for k, v in manifest.items() if k != 'identity'})
    if manifest['identity'] != identity or identity != parent['development_manifest_identity']:
        raise ValueError('campaign development manifest identity mismatch')
    groups = {split: {entry['demand_group'] for entry in manifest['entries']
                      if entry['split'] == split} for split in ('train', 'validation')}
    if not all(groups.values()) or groups['train'] & groups['validation']:
        raise ValueError('component codec validation requires disjoint development demand groups')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    protocol = payload['protocol']
    if payload.get('schema_version') != 1 or protocol.get('parent') != parent:
        raise ValueError('component checkpoint parent/schema mismatch')
    if (payload['variant'] != variant or protocol['variant']['variant'] != variant
            or payload['seed'] != seed or protocol['training_seed'] != seed):
        raise ValueError('component checkpoint variant/seed mismatch')
    if (protocol['mode'] != 'single_step' or protocol['horizon'] != 1
            or not protocol['frozen_codec'] or not 0 < payload['update'] <= protocol['planned_updates']):
        raise ValueError('policy component admission requires a trained frozen-codec single-step checkpoint')
    extra = payload['model'].get('dynamics._extra_state')
    if variant != 'graph_fm' and extra != dict(variant=variant, seed=seed, version=1):
        raise ValueError('component serialized variant/seed mismatch')
    return payload, parent


def resolved_seeds(args):
    """Legacy --seed supplies both defaults; explicit overrides opt into v2."""
    component = getattr(args, 'component_seed', None)
    policy = getattr(args, 'policy_seed', None)
    return (args.seed if component is None else component,
            args.seed if policy is None else policy)


def seed_protocol(args):
    if getattr(args, 'component_seed', None) is None and getattr(args, 'policy_seed', None) is None:
        return None
    component, policy = resolved_seeds(args)
    return dict(schema_version=1, component_seed=component, policy_seed=policy,
        online_demand_seed=policy,
        policy_seed_scope='fresh policy/critics/readout, learner RNG, online demand; evaluation seed unchanged')


def runtime_config(source, args):
    _, policy_seed = resolved_seeds(args)
    config = replace(source,
        run=replace(source.run, output_dir=args.output_dir.resolve(), device=args.device, seed=policy_seed),
        resources=replace(source.resources, world_model_pretrain_precision='fp32',
            world_model_pretrain_compile_blocks=False, world_model_imagination_precision='fp32',
            world_model_imagination_compile=False, imagination_rollouts_per_pass=args.rollouts,
            imagination_passes_per_episode=args.passes, imagination_time_budget_seconds=0.),
        method=replace(source.method, critic_value_mode='bounded_cohort_v1', disagreement_weight=0.,
            fm_steps=args.steps, fm_training_mode='single_step', exposure_readout_supervision='observed_endpoint'),
        codec=replace(source.codec, decoder_pretrain_enabled=False, decoder_online_enabled=False,
            decoder_allow_unready=False, batch_size=4),
        data=replace(source.data,
            pretrain_cache_manifest=args.experiment_dir.resolve() / 'development_manifest.json',
            validation_cache_manifest=args.experiment_dir.resolve() / 'development_manifest.json'))
    return replace(config, semantic_fingerprint=_semantic_fingerprint(**{
        key: getattr(config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))


def city_executor_admission(config, prepared, topology, args):
    """Validate static controls against the campaign-bound network, fail closed."""
    from urban_dreamer.traffic.movement_controls import CityControlTable
    simulation = config.simulation
    if (simulation.traffic_model != 'meso' or not simulation.meso_junction_control
            or simulation.meso_lane_queue or simulation.meso_multi_queue
            or simulation.session_protocol != 'event' or simulation.observation_step_seconds != 1.):
        raise ValueError('city executor requires single-queue meso with junction control and 1 s event observations')
    # No additional files, TLS changes, alternate configuration, or unreviewed
    # SUMO switches may invalidate the table inferred from the bound net.xml.
    if tuple(simulation.sumo_extra_args) not in ((), ('--collision.action', 'warn')):
        raise ValueError('city executor rejects unverified SUMO arguments/control overrides')
    network_hash = sha256(prepared.network_xml)
    if network_hash != prepared.input_hashes['network_xml']:
        raise ValueError('city network digest differs from prepared input provenance')
    controls = CityControlTable.from_network(prepared.network_xml, topology.network.edge_ids,
        no_external_overrides=True, verified_road_only_no_trains=False)
    if any(m.support != 'exact_control' for m in controls.movements.values()):
        raise ValueError('city executor requires exact static control support; unsupported/rail approximations rejected')
    required = {(i, int(j)) for i, successors in enumerate(topology.network.adjacency_list) for j in successors}
    if required != set(controls.movements):
        raise ValueError('city controls do not exactly cover the registered routing movements')
    table = controls.preflight()
    parameters = dict(uncertain_order=args.city_queue_order == 'seeded_uncertain',
        discharge_headway_seconds=args.city_discharge_headway,
        receiving_headway_seconds=args.city_receiving_headway,
        conflict_clearance_seconds=args.city_conflict_clearance)
    if any(not math.isfinite(value) or value <= 0 for key, value in parameters.items() if key != 'uncertain_order'):
        raise ValueError('city executor headways/clearance must be finite and positive')
    protocol = dict(schema_version=1, name='city', network_xml=str(prepared.network_xml.resolve()),
        network_sha256=network_hash, input_hashes=prepared.input_hashes,
        control_table_sha256=hashlib.sha256((json.dumps(table, indent=2, sort_keys=True,
            allow_nan=False) + '\n').encode()).hexdigest(),
        control_support=table['support_counts'], movement_count=table['movement_count'],
        parameters=parameters, queue_order=args.city_queue_order,
        seed_rule='SHA256 of checkpointed CPU Torch RNG state, rollout index, auxiliary flag; no RNG draws consumed',
        lifecycle='fresh arbiter per rollout, persistent through contiguous windows, discarded at rollout end',
        approximation='one FIFO per external road; spatial initial order; declared seeded ties; constant headways and conflict clearance; not full SUMO physics',
        sumo_extra_args=list(simulation.sumo_extra_args),
        source_sha256={name: sha256(ROOT/name) for name in (
            'urban_dreamer/learning/training.py', 'urban_dreamer/learning/city_imagination.py',
            'urban_dreamer/learning/imagination.py', 'urban_dreamer/traffic/movement_controls.py')})
    return controls, table, protocol


class ComponentPilotTrainer(pilot.PilotTrainer):
    def configure_city_executor(self, controls, admission):
        self._city_controls = controls
        self._city_executor_admission = copy.deepcopy(admission)

    def _validate_city_execution(self):
        resources = self.config.resources
        if (int(os.environ.get('URB_IMAGINATION_WORKERS', '0')) != 0
                or resources.world_model_imagination_compile
                or resources.world_model_imagination_precision != 'fp32'):
            raise ValueError('city executor supports serial FP32 imagination without compilation only')

    def _train_imagination_pass(self):
        if hasattr(self, '_city_controls'):
            self._validate_city_execution()
        return super()._train_imagination_pass()

    def _create_imagination_executor(self, *, rollout_index, critic_only):
        if not hasattr(self, '_city_controls'):
            return super()._create_imagination_executor(rollout_index=rollout_index, critic_only=critic_only)
        from urban_dreamer.learning.city_imagination import ChronologicalJunctionArbiter
        self._validate_city_execution()
        # Hash, never advance, the checkpointed/forked stream so opting into a
        # city arbiter does not itself shift actor or world-model random draws.
        identity = torch.get_rng_state().numpy().tobytes() + json.dumps(
            [int(rollout_index), bool(critic_only)]).encode()
        seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big')
        arbiter = ChronologicalJunctionArbiter(self._city_controls, seed=seed,
            **self._city_executor_admission['parameters'])
        self.metrics['city_executor_rollouts'] = self.metrics.get('city_executor_rollouts', 0) + 1
        self.metrics['city_executor_aux_rollouts'] = self.metrics.get('city_executor_aux_rollouts', 0) + int(critic_only)
        return arbiter

    def _advance_imagination_window(self, executor, states, **kwargs):
        if not hasattr(self, '_city_controls'):
            return super()._advance_imagination_window(executor, states, **kwargs)
        from urban_dreamer.learning.city_imagination import advance_city_imagined_window
        previous = len(executor.audit)
        result = advance_city_imagined_window(states, junction_arbiter=executor, **kwargs)
        self.metrics['city_executor_windows'] = self.metrics.get('city_executor_windows', 0) + 1
        for event in executor.audit[previous:]:
            key = 'city_executor_' + event['event']
            self.metrics[key] = self.metrics.get(key, 0) + 1
            if event['event'] == 'queue_initialization' and event['kind'] == 'seeded_uncertain_order':
                self.metrics['city_executor_uncertain_queues'] = self.metrics.get('city_executor_uncertain_queues', 0) + 1
        return result

    def initialize_component(self, payload, admission):
        """Preserve paired policy/readout initialization and RNG across adapters."""
        if admission.get('imagination_executor') != getattr(self, '_city_executor_admission', None):
            raise ValueError('configured city executor differs from component admission')
        before_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        readout = copy.deepcopy(self.world_model.active_vehicle_fraction_readout.state_dict())
        try:
            apply_variant(self.world_model, payload['variant'], seed=payload['seed'])
            self.world_model.load_state_dict(payload['model'], strict=True)
            self.world_model.active_vehicle_fraction_readout.load_state_dict(readout, strict=True)
        finally:
            torch.set_rng_state(before_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
        codec = self.world_model.flow_projector
        codec.__class__ = SparseGraphCodec
        codec.register_buffer('sparse_adjacency', self._tensor(self.network.adjacency).to_sparse().coalesce(),
                              persistent=False)
        codec.requires_grad_(False)
        self.world_model.active_vehicle_fraction_readout.requires_grad_(True)
        self._world_parameters = list(self.world_model.dynamics.parameters())
        self.optimizers['world_model_initial'] = torch.optim.AdamW(self._world_parameters, lr=1e-4, weight_decay=1e-2)
        self.optimizers['world_model_online'] = torch.optim.Adam(self._world_parameters, lr=1e-5)
        self.metrics['component_admission'] = copy.deepcopy(admission)
        self.metrics['component_codec_sha256'] = tensor_digest(codec)

    def _world_loss(self, batch):
        self.validate_codec_ready()
        scoped = prepare_transition_targets(batch, capacities=self.network.capacities)
        inputs = batch.flow[:, :1].to(self.device)
        targets = scoped.transition_target_flow[:, :1].to(self.device)
        with torch.no_grad():
            z_inputs = self.world_model.encode(inputs[:, 0]).unsqueeze(1)
            z_targets = self.world_model.encode(targets[:, 0]).unsqueeze(1)
        loss, metrics, _ = experimental_loss(self.world_model, z_inputs=z_inputs, z_targets=z_targets,
            action=batch.action[:, :1].to(self.device), action_events=batch.action_events.for_step(0).to(self.device),
            prior_weight=self.config.method.prior_loss_weight, predict_endpoints=False)
        self.metrics.update({key + '_last': float(value) for key, value in metrics.items()})
        self.metrics['world_model_supervision_depth_last'] = 1.
        with torch.no_grad():
            raw_fraction = self.world_model.decode(z_targets)[..., 2].mean(dim=-1)
        fraction = self.world_model.active_vehicle_fraction_readout(z_targets, z_prev=z_inputs,
            raw_decoder_fraction=raw_fraction, transition_seconds=30.)
        auxiliary = F.mse_loss(fraction.float(), batch.active_vehicle_fraction[:, :1].to(self.device).float())
        return loss, auxiliary

    def assert_frozen_codec(self):
        if (any(p.requires_grad for p in self.world_model.flow_projector.parameters())
                or tensor_digest(self.world_model.flow_projector) != self.metrics['component_codec_sha256']):
            raise RuntimeError('component frozen codec changed')

    def state_dict(self):
        self.assert_frozen_codec()
        self.metrics['component_optimizer_steps'] = dict(self.pilot_optimizer_steps)
        return super().state_dict()

    def load_state_dict(self, state):
        if state['metrics'].get('component_admission') != self.metrics.get('component_admission'):
            raise ValueError('resumed component admission differs')
        super().load_state_dict(state)
        self.pilot_optimizer_steps = dict(self.metrics['component_optimizer_steps'])
        self.validate_codec_ready()
        self.assert_frozen_codec()


def strict_resume_audit(trainer, output, input_hash):
    """Reload the on-disk strict schema and prove optimizer, replay and RNG restore."""
    checkpoint = pilot.load_checkpoint(output_dir=output,
        expected_config_fingerprint=trainer.config.semantic_fingerprint, expected_input_hash=input_hash)
    trainer.load_state_dict(checkpoint.training_state)
    restored = trainer.state_dict()
    def equal(first, second):
        if torch.is_tensor(first):
            return torch.equal(first.cpu(), torch.as_tensor(second).cpu())
        if isinstance(first, dict):
            return set(first) == set(second) and all(equal(first[key], second[key]) for key in first)
        if isinstance(first, (list, tuple)):
            return len(first) == len(second) and all(equal(a, b) for a, b in zip(first, second))
        if hasattr(first, 'shape'):
            import numpy as np
            return bool(np.array_equal(first, second))
        return first == second
    if not equal(checkpoint.training_state, restored):
        raise RuntimeError('strict component resume did not restore complete training state')
    return dict(checkpoint_sha256=sha256(checkpoint.path), exact_state_restored=True,
        compared=['models', 'optimizers', 'replay', 'rng', 'metrics', 'next_environment_episode', 'reward_contract'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--seed', type=int, default=42,
        help='legacy shared seed; default for each seed not explicitly overridden')
    p.add_argument('--component-seed', type=int,
        help='frozen checkpoint training seed; opts into the version 2 seed protocol')
    p.add_argument('--policy-seed', type=int,
        help='fresh policy/readout, learner RNG and online demand seed; evaluation seed stays fixed')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--gpu-memory-fraction', type=float, default=.06)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--passes', type=int, default=1)
    p.add_argument('--rollouts', type=int, default=2)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--evaluation-episodes', type=int, default=1)
    p.add_argument('--imagination-executor', choices=('legacy', 'city'), default='legacy')
    p.add_argument('--city-queue-order', choices=('strict', 'seeded_uncertain'), default='strict',
        help='strict rejects tied seed positions; seeded_uncertain explicitly admits random tie ordering')
    p.add_argument('--city-discharge-headway', type=float, default=1.)
    p.add_argument('--city-receiving-headway', type=float, default=1.)
    p.add_argument('--city-conflict-clearance', type=float, default=1.)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    component_seed, policy_seed = resolved_seeds(args)
    if not all(0 <= value < 2**32 for value in (component_seed, policy_seed)):
        p.error('resolved component and policy seeds must be within [0, 2**32)')
    if min(args.episodes, args.passes, args.rollouts, args.steps, args.evaluation_episodes, args.threads) < 1:
        p.error('budgets must be positive')
    if not 0 < args.gpu_memory_fraction <= .06:
        p.error('component smoke GPU fraction must be within (0,.06]')
    if args.imagination_executor == 'legacy' and (args.city_queue_order != 'strict' or
            (args.city_discharge_headway, args.city_receiving_headway, args.city_conflict_clearance) != (1., 1., 1.)):
        p.error('city queue/headway options require --imagination-executor city')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        p.error('existing output requires explicit --resume')
    os.environ['URB_IMAGINATION_WORKERS'] = '0'
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        free, total = torch.cuda.mem_get_info(device)
        minimum_free = (4 if args.imagination_executor == 'city' else 2) * 1024**3
        if free < max(minimum_free, args.gpu_memory_fraction * total + 512 * 1024**2):
            raise RuntimeError(f'GPU admission requires at least {minimum_free // 1024**3} GiB free and allocation headroom')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    payload, parent = validate_component_source(args.experiment_dir, args.checkpoint,
        args.checkpoint_sha256, variant=args.variant, seed=component_seed)
    source = load_config(args.experiment_dir / 'snapshot/run_config.toml')
    config = runtime_config(source, args)
    prepared = pilot.validate_prepared_inputs(config)
    topology = pilot.load_traffic_topology(prepared)
    city = city_executor_admission(config, prepared, topology, args) if args.imagination_executor == 'city' else None
    train = dataset(args.experiment_dir, config, prepared, topology, split='train', horizon=1)
    validation = dataset(args.experiment_dir, config, prepared, topology, split='validation', horizon=1)
    if train.manifest['identity'] != payload['protocol']['training_manifest_identity']:
        raise ValueError('component source training manifest identity mismatch')
    admission = dict(kind='experimental_offline_component_development_screen',
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=args.checkpoint_sha256,
        offline_update=payload['update'], offline_planned_updates=payload['protocol']['planned_updates'],
        variant=args.variant, seed=component_seed, parent=parent,
        training_manifest_identity=train.manifest['identity'],
        validation_manifest_identity=validation.manifest['identity'],
        production_pretraining_complete_claim=False)
    protocol = dict(schema_version=1, source=admission, role='policy_smoke_not_production_budget',
        reward_contract=pilot.training.REWARD_CONTRACT,
        reward_source='explicit vehicle seconds; -(cumulative_mean_end-cumulative_mean_start)/60',
        fresh_modules=['actor', 'latent_critic', 'event_critic', 'target_latent_critic', 'active_vehicle_fraction_readout'],
        frozen_codec=True, readout_supervision='observed_endpoint_auxiliary_only',
        budgets=pilot.learning_budgets(config), evaluation_indices=list(range(args.evaluation_episodes)),
        runtime_config_fingerprint=config.semantic_fingerprint, fm_steps=args.steps,
        precision='fp32', imagination_workers=0, imagination_compile=False,
        device=args.device, threads=args.threads, gpu_memory_fraction=args.gpu_memory_fraction,
        runner_sha256=sha256(__file__), dynamics_sha256=sha256(ROOT/'urban_dreamer/models/experimental_dynamics.py'))
    seeds = seed_protocol(args)
    if seeds is not None:
        # Seed identity is also checkpointed in metrics via component_admission.
        # Legacy invocations retain their exact v1 field layout and fingerprint.
        admission['seeds'] = seeds
        protocol.update(schema_version=2, seeds=seeds)
    if city is not None:
        admission['imagination_executor'] = city[2]
        protocol.update(schema_version=3, imagination_executor=city[2])
    trainer = ComponentPilotTrainer(config=config, prepared_inputs=prepared, topology=topology,
        _offline_data=train, _codec_validation_data=validation, pilot_log=args.output_dir/'pilot_metrics.jsonl')
    input_hash = pilot.combined_input_hash(prepared)
    try:
        if city is not None:
            trainer.configure_city_executor(city[0], city[2])
        trainer.initialize_component(payload, admission)
        del payload
        # This real gate consumes the entire disjoint selected development split.
        # Preserve its measurements even when admission fails.
        if not args.resume:
            print(json.dumps(dict(phase='codec_validation_started', admission=admission)), flush=True)
            validation_result = trainer._validate_codec()
            print(json.dumps(dict(phase='codec_validation_finished', metrics=validation_result)), flush=True)
            trainer.validate_codec_ready()
        previous_indices = pilot.EVALUATION_INDICES
        pilot.EVALUATION_INDICES = tuple(protocol['evaluation_indices'])
        try:
            runner = pilot.OnlinePilot(trainer, output=args.output_dir, input_hash=input_hash,
                protocol=protocol, online_episodes=args.episodes)
            result = runner.run()
        finally:
            pilot.EVALUATION_INDICES = previous_indices
        trainer.assert_frozen_codec()
        audit = strict_resume_audit(trainer, args.output_dir, input_hash)
        audit.update(codec_fixed=True, codec_sha256=tensor_digest(trainer.world_model.flow_projector),
            optimizer_steps=dict(trainer.pilot_optimizer_steps),
            reward_contract=trainer.state_dict()['reward_contract'],
            paired_demand_hashes=[row['demand_hash'] for row in result['before']],
            world_model_updates=trainer.metrics['world_model_updates'])
        if city is not None:
            audit.update(imagination_executor=city[2], executor_counts={
                key: value for key, value in trainer.metrics.items() if key.startswith('city_executor_')})
        atomic_json(args.output_dir/'component_audit.json', audit)
        if city is not None:
            atomic_json(args.output_dir/'city_control_table.json', city[1])
        if sha256(args.checkpoint) != args.checkpoint_sha256:
            raise RuntimeError('offline source changed during isolated policy learning')
        print(json.dumps(dict(status='completed', audit=audit), indent=2), flush=True)
        return result
    except Exception as exc:
        atomic_json(args.output_dir.parent/(args.output_dir.name+'.failure.json'),
            dict(type=type(exc).__name__, error=str(exc), metrics=trainer.metrics))
        raise
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
