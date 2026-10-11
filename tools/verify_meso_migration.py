#!/usr/bin/env python3
"""Real, CPU-only mesoscopic migration smoke; never bypass codec readiness.

Production training constants and shipped configurations are untouched. This
entrypoint writes an explicit test configuration, bounded budget provenance,
real SUMO archives, held-out codec measurements, and a strict checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# The trainer checkpoints CUDA RNGs when CUDA is visible. Hide devices before
# importing torch so even checkpoint validation leaves production GPU jobs alone.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning.baselines import build_method
from urban_dreamer.learning.baselines.common import EpisodeRunner
from urban_dreamer.learning.cohort_targets import prepare_transition_targets
from urban_dreamer.learning.episode_archive import archive_physics_identity, read_episode_archive
from urban_dreamer.learning.offline_data import OfflineEpisodeDataset
from urban_dreamer.learning.training import MainlineTrainer
import urban_dreamer.learning.training as training
from urban_dreamer.models.world_model import independent_transition_fm_loss
from urban_dreamer.traffic.simulation_settings import physics_identity, sumo_model_args
from urban_dreamer.workflow.checkpoint import CheckpointPhase, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import load_config
from urban_dreamer.workflow.pipeline import combined_input_hash


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_config(config, path):
    # Snapshot values are JSON-compatible and the simple scalars/arrays used
    # here are also TOML-compatible; absent optional paths are omitted.
    text = '\n\n'.join('[' + section + ']\n' + '\n'.join(
        key + ' = ' + json.dumps(value) for key, value in values.items() if value is not None)
        for section, values in config.resolved_snapshot().items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + '\n')
    return load_config(path)


def runner_args(config_path, *, limit=16, seed=42, method='random'):
    return argparse.Namespace(config=config_path, device='cpu', threads=1,
        seed=seed, vehicle_limit=limit, method=method, session=None,
        backend=None, step_seconds=None, forbid_revisit=False,
        actor_lr=2e-4, critic_lr=3e-4, global_critic_lr=3e-4,
        sp_prior_weight=0., sp_prior_temperature=30.,
        sp_prior_hold_episodes=200, sp_prior_off_episode=1000)


def digest(module):
    result = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


@contextmanager
def online_test_budget():
    budget = dict(ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE=1,
        IMAGINED_PASSES_PER_EPISODE=1, IMAGINED_ROLLOUTS_PER_PASS=2,
        IMAGINED_HORIZON=2)
    originals = {name: getattr(training, name) for name in budget}
    try:
        for name, value in budget.items():
            setattr(training, name, value)
        yield dict(scope='TEST ONLY process-local overrides', production=originals, smoke=budget)
    finally:
        for name, value in originals.items():
            setattr(training, name, value)


def isolated_fm_gradient_check(trainer):
    """Operator evidence only; this is deliberately outside trainer lifecycle."""
    batch = next(trainer.offline_data.iter_epoch(1, trainer.rng))
    scoped = prepare_transition_targets(batch, capacities=trainer.network.capacities)
    with torch.no_grad():
        observed = trainer.world_model.encode(batch.flow[:, :1].flatten(0, 1)).unsqueeze(1)
        target = trainer.world_model.encode(scoped.transition_target_flow[:, :1].flatten(0, 1)).unsqueeze(1)
    before = digest(trainer.world_model.dynamics)
    optimizer = trainer.optimizers['world_model_initial']
    optimizer.zero_grad(set_to_none=True)
    loss, metrics, endpoint = independent_transition_fm_loss(trainer.world_model.dynamics,
        z_inputs=observed, z_targets=target, action=batch.action[:, :1],
        action_events=batch.action_events, steps=2, predict_endpoints=True)
    loss.backward()
    gradients = [parameter.grad for parameter in trainer.world_model.dynamics.parameters()
                 if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    norm = float(torch.nn.utils.clip_grad_norm_(trainer.world_model.dynamics.parameters(), 10.))
    assert norm > 0 and bool(torch.isfinite(endpoint).all())
    optimizer.step()
    assert before != digest(trainer.world_model.dynamics)
    trainer.offline_data.release_cache()
    return dict(scope='isolated operator check, not codec-gated lifecycle completion',
        loss=float(loss.detach()), grad_norm=norm,
        endpoint_shape=list(endpoint.shape), finite=True, parameters_changed=True,
        **{key: float(value) for key, value in metrics.items()})


def toy_validation(output, max_codec_updates, codec_learning_rate=1e-4):
    root = output / 'toy'
    base = load_config(ROOT / 'configs/toy_cross_bottleneck.toml')
    config = replace(base,
        run=replace(base.run, output_dir=root / 'mainline', device='cpu',
            codec_pretraining_epochs=max_codec_updates, world_model_initialization_epochs=1,
            online_training_episodes=1, evaluation_episodes=1),
        simulation=replace(base.simulation, demand_duration_seconds=60.,
            world_model_intervals_per_episode=8, training_completion_buffer_seconds=0,
            evaluation_horizon_seconds=240, episode_wall_timeout_seconds=120.),
        resources=replace(base.resources, world_model_initialization_batch_size=1,
            world_model_initialization_sequence_length=3,
            world_model_gradient_accumulation_steps=1, online_world_model_batch_size=1),
        data=replace(base.data, archive_roots=(root / 'archives',),
            validation_archive_roots=(), pretrain_cache_manifest=None, validation_cache_manifest=None),
        codec=replace(base.codec, learning_rate=codec_learning_rate, batch_size=32,
            max_updates=max_codec_updates, validation_every=100))
    config = write_config(config, root / 'smoke.toml')
    runner = EpisodeRunner(runner_args(root / 'smoke.toml'))
    collection = []
    for episode, policy in enumerate(('shortest', 'random', 'shortest', 'random'), start=1):
        path = root / 'archives' / policy / 'train' / f'episode_{episode:06d}.npz'
        _, _, metrics = runner.run_episode(policy, seed=episode,
            archive_path=path, archive_metadata=dict(phase='train', episode=episode))
        metadata = read_episode_archive(path)['metadata']
        assert archive_physics_identity(metadata) == physics_identity(config.simulation)
        collection.append(dict(path=str(path), **metrics))
        print(json.dumps(dict(stage='toy_archive', policy=policy, episode=episode,
            vehicles=metrics['vehicle_count'], events=metrics['event_count'])), flush=True)
    dataset = OfflineEpisodeDataset(config, runner.prepared, runner.topology, 3,
        manifest_path=root / 'strict_inventory.json')
    dataset.verify_integrity()
    trainer = MainlineTrainer(config=config, prepared_inputs=runner.prepared,
        topology=runner.topology)
    trainer.validate_pretraining_data()
    def demand_hashes(source):
        return sorted({read_episode_archive(Path(entry['path']))['metadata']['demand_sha256']
                       for entry in source.manifest['entries']})
    train_demands = demand_hashes(trainer.offline_data)
    heldout_demands = demand_hashes(trainer.codec_validation_data)
    assert not set(train_demands).intersection(heldout_demands)
    initial = trainer._validate_codec()
    history = []
    while not trainer.codec_ready() and trainer.metrics['codec_updates'] < max_codec_updates:
        result = trainer.initialize_codec_epoch(int(trainer.metrics['codec_epochs_completed']))
        history.append(result)
        if len(history) % 20 == 0 or result['codec_ready']:
            print(json.dumps(dict(stage='codec', **result)), flush=True)
            write_json(root / 'codec_progress.json', dict(initial=initial, history=history))
    write_json(root / 'codec_progress.json', dict(initial=initial, history=history))
    result = dict(strict_dataset=dataset.summary, archives=collection,
        physics_identity=physics_identity(config.simulation),
        sumo_model_args=sumo_model_args(config.simulation),
        codec_ready=trainer.codec_ready(), codec_updates=trainer.metrics['codec_updates'],
        codec_initial=initial, codec_final=history[-1] if history else initial,
        train_manifest_identity=trainer.offline_data.manifest['identity'],
        validation_manifest_identity=trainer.codec_validation_data.manifest['identity'],
        train_demand_sha256=train_demands, heldout_demand_sha256=heldout_demands,
        demand_groups_disjoint=True)
    if trainer.codec_ready():
        result['world_initialization'] = trainer.initialize_world_model_epoch(0)
        before = {name: digest(getattr(trainer, name)) for name in
                  ('world_model', 'actor', 'latent_critic', 'event_critic')}
        with online_test_budget() as budget:
            result['online_budget'] = budget
            result['online'] = trainer.train_online_episode(0)
        result['updated_models'] = {name: before[name] != digest(getattr(trainer, name)) for name in before}
        assert all(result['updated_models'].values()), result['updated_models']
        assert result['online']['imagined_events'] > 0
    else:
        try:
            trainer.train_online_episode(0)
        except ValueError as error:
            result['online_correctly_blocked'] = str(error)
        else:
            raise AssertionError('online training bypassed codec readiness')
        result['isolated_fm'] = isolated_fm_gradient_check(trainer)
        collected = trainer._run_episode(mode='online', episode_index=0)
        result['ungated_real_collection'] = collected.metrics
    first = trainer.evaluate_episode(0)
    first_path = config.run.output_dir / 'episodes' / 'evaluation' / 'episode_000000.npz'
    first_archive = read_episode_archive(first_path)
    state = trainer.state_dict()
    checkpoint = save_checkpoint(output_dir=config.run.output_dir,
        config_fingerprint=config.semantic_fingerprint,
        input_hash=combined_input_hash(runner.prepared),
        phase=CheckpointPhase.ONLINE_TRAINING if trainer.codec_ready() else CheckpointPhase.CODEC_PRETRAIN,
        next_unit=int(trainer.metrics['online_episodes_completed']), training_state=state)
    loaded = load_checkpoint(output_dir=config.run.output_dir,
        expected_config_fingerprint=config.semantic_fingerprint,
        expected_input_hash=combined_input_hash(runner.prepared))
    restored = MainlineTrainer(config=config, prepared_inputs=runner.prepared, topology=runner.topology)
    restored.load_state_dict(loaded.training_state)
    assert all(digest(getattr(trainer, name)) == digest(getattr(restored, name))
               for name in ('world_model', 'actor', 'latent_critic', 'event_critic'))
    second = restored.evaluate_episode(0)
    second_archive = read_episode_archive(first_path)
    assert first_archive['metadata']['demand_sha256'] == second_archive['metadata']['demand_sha256']
    assert first['evaluation_t_all'] == second['evaluation_t_all']
    assert np.array_equal(first_archive['flow'], second_archive['flow'])
    result.update(checkpoint=str(checkpoint), strict_checkpoint_restored=True,
        deterministic_evaluation=dict(first=first, repeated=second,
            same_demand=True, exact_flow_equal=True))
    write_json(root / 'report.json', result)
    return result


def city_validation(output):
    """Shared full shipped physics, demand duration and deadline, limited population."""
    from tools.train_toy_actor_critic import RealActorCritic
    root = output / 'ingolstadt'
    base = load_config(ROOT / 'configs/ingolstadt_custom.toml')
    config = replace(base, run=replace(base.run, device='cpu', output_dir=root / 'mainline'),
        data=replace(base.data, archive_roots=(root / 'empty_pretrain',),
            validation_archive_roots=(), pretrain_cache_manifest=None, validation_cache_manifest=None))
    (root / 'empty_pretrain').mkdir(parents=True, exist_ok=True)
    config = write_config(config, root / 'smoke.toml')
    args = runner_args(root / 'smoke.toml', seed=config.method.evaluation_demand_seed)
    runner = EpisodeRunner(args)
    trainer = MainlineTrainer(config=config, prepared_inputs=runner.prepared, topology=runner.topology)
    mainline = trainer.evaluate_episode(0)
    path = config.run.output_dir / 'episodes' / 'evaluation' / 'episode_000000.npz'
    archive = read_episode_archive(path)
    identity = archive_physics_identity(archive['metadata'])
    demands = [archive['metadata']['demand_sha256']]
    results = {'mainline': mainline}
    # These are fresh frozen actors for execution compatibility, not trained
    # policies or a micro checkpoint imported into a mesoscopic run.
    for method in ('shortest', 'random', 'gnac', 'dreamer', 'mbpo', 'dta'):
        if method in ('shortest', 'random'):
            select, on_demand = method, None
        else:
            args.method = method
            policy = build_method(method, args, runner)
            select = functools.partial(policy.select, explore=False)
            on_demand = getattr(policy, 'begin_episode', None)
        path = root / method / 'evaluation' / 'episode_000000.npz'
        _, _, metrics = runner.run_episode(select, seed=0, label=method,
            archive_path=path, archive_metadata=dict(phase='evaluation', method=method, episode=0),
            on_demand=on_demand)
        metadata = read_episode_archive(path)['metadata']
        assert archive_physics_identity(metadata) == identity
        demands.append(metadata['demand_sha256'])
        results[method] = metrics
        print(json.dumps(dict(stage='city_evaluation', method=method,
            events=metrics['event_count'], vehicles=metrics['vehicle_count'])), flush=True)
    ac = RealActorCritic(runner_args(root / 'smoke.toml', seed=config.method.evaluation_demand_seed,
                                   method='online_ac'))
    path = root / 'online_ac' / 'evaluation' / 'episode_000000.npz'
    _, metrics = ac.run_episode(policy='deterministic', seed=0,
        archive_path=path, archive_metadata=dict(phase='evaluation', method='online_ac', episode=0))
    metadata = read_episode_archive(path)['metadata']
    assert archive_physics_identity(metadata) == identity
    demands.append(metadata['demand_realization_sha256'])
    results['online_ac'] = metrics
    assert len(set(demands)) == 1, demands
    result = dict(physics_identity=identity, paired_demand_sha256=demands[0],
        settings=config.resolved_snapshot()['simulation'], reference_vehicle_limit=16,
        policy_provenance='fresh frozen initialization; no trained quality claim; no micro checkpoint resume',
        same_physics_all_methods=True, same_realized_demand_all_methods=True, evaluations=results)
    write_json(root / 'report.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/meso_migration_validation_20261006')
    parser.add_argument('--max-codec-updates', type=int, default=4000)
    parser.add_argument('--codec-learning-rate', type=float, default=1e-4,
                        help='Explicit verification budget; defaults to the production codec rate')
    parser.add_argument('--skip-city', action='store_true')
    args = parser.parse_args()
    if args.max_codec_updates < 1 or args.codec_learning_rate <= 0:
        parser.error('max-codec-updates and codec-learning-rate must be positive')
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite an existing verification run')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.monotonic()
    import libsumo
    report = dict(sumo_version=list(libsumo.getVersion()), device='cpu', threads=1,
        provenance=dict(scope='TEST ONLY', reference_vehicle_limit=16,
            independent_toy_archives=4, toy_demand_duration_seconds=60,
            toy_deadline_seconds=240, codec_max_updates=args.max_codec_updates,
            codec_learning_rate=args.codec_learning_rate, codec_gate_thresholds='unchanged shipped thresholds'))
    try:
        report['toy'] = toy_validation(output, args.max_codec_updates, args.codec_learning_rate)
        if not args.skip_city:
            report['ingolstadt'] = city_validation(output)
        report['status'] = 'passed' if report['toy']['codec_ready'] else 'codec_not_ready_within_budget'
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic() - started
        write_json(output / 'report.json', report)
    print(json.dumps(dict(status=report['status'], report=str(output / 'report.json'),
        wall_seconds=report['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
