"""Bounded real online actor-critic pilot with paired, seeded SUMO evaluation."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from urban_dreamer.learning import training
from urban_dreamer.learning.training import MainlineTrainer
from urban_dreamer.traffic.routing import validate_prepared_inputs
from urban_dreamer.traffic.topology import load_traffic_topology
from urban_dreamer.workflow.checkpoint import CheckpointPhase, latest_checkpoint_path, load_checkpoint, save_checkpoint
from urban_dreamer.workflow.config import _semantic_fingerprint, load_config
from urban_dreamer.workflow.pipeline import combined_input_hash
from tools.run_complete_pretrain import export_run_config
from tools.train_codec_ready import SparseGraphCodec

ONLINE_EPISODES = 5
EVALUATION_INDICES = (0, 1, 2)
BUDGETS = dict(world_model_updates_per_episode=4, imagination_passes_per_episode=16,
               rollouts_per_pass=8, macro_windows_per_rollout=8)
MODEL_NAMES = ('actor', 'latent_critic', 'event_critic')


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def assert_original_budgets():
    actual = dict(world_model_updates_per_episode=training.ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE,
        imagination_passes_per_episode=training.IMAGINED_PASSES_PER_EPISODE,
        rollouts_per_pass=training.IMAGINED_ROLLOUTS_PER_PASS,
        macro_windows_per_rollout=training.IMAGINED_HORIZON)
    if actual != BUDGETS:
        raise ValueError(f'pilot requires the original full online actor-critic budgets: {BUDGETS}; got {actual}')


def learning_budgets(config):
    budgets = training.resolved_imagination_budget(config)
    # Historical protocols omitted a disabled wall-time budget.
    if budgets['imagination_time_budget_seconds'] == 0:
        budgets.pop('imagination_time_budget_seconds')
    return budgets


def validate_pretrained_source(checkpoint, config):
    state = checkpoint.training_state
    metrics = state['metrics']
    if checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING or checkpoint.next_unit != 0:
        raise ValueError('source must be the completed-pretraining online-start checkpoint')
    if metrics['world_model_initialization_epochs_completed'] != config.run.world_model_initialization_epochs:
        raise ValueError('source has not completed every configured world-model initialization epoch')
    if (metrics.get('online_episodes_completed', 0) != 0 or state['next_environment_episode'] != 0
            or state['replay']['online']['episodes']):
        raise ValueError('baseline source must not contain prior online training or collection')


@contextmanager
def isolated_evaluation(trainer, directory):
    """Evaluation changes files only; preserve training RNG, counters and modes."""
    config = trainer.config
    actor_prior = (getattr(trainer.actor, 'prior_weight', 0.),
                   getattr(trainer.actor, 'prior_temperature_seconds', 30.))
    metrics = copy.deepcopy(trainer.metrics)
    environment_index = trainer._next_environment_episode
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    generator_rng = copy.deepcopy(trainer.rng.bit_generator.state)
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    modules = [getattr(trainer, name) for name in ('world_model', *MODEL_NAMES)]
    modes = [[(module, module.training) for module in model.modules()] for model in modules]
    trainer.config = replace(config, run=replace(config.run, output_dir=Path(directory)))
    for model in modules:
        model.eval()
    try:
        yield
    finally:
        trainer.config = config
        training.set_shortest_path_prior(trainer.actor, *actor_prior)
        trainer.metrics = metrics
        trainer._next_environment_episode = environment_index
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        trainer.rng.bit_generator.state = generator_rng
        torch.set_rng_state(torch_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        for subtree in modes:
            for module, was_training in subtree:
                module.training = was_training


def parameter_snapshot(trainer):
    return {name: [parameter.detach().cpu().clone() for parameter in getattr(trainer, name).parameters()]
            for name in MODEL_NAMES}


def parameter_changes(trainer, previous):
    result = {}
    for name in MODEL_NAMES:
        squared = maximum = 0.
        changed = 0
        for before, parameter in zip(previous[name], getattr(trainer, name).parameters()):
            difference = parameter.detach().cpu() - before
            squared += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            changed += int(bool((difference != 0).any()))
        result[name] = dict(l2_change=math.sqrt(squared), max_abs_change=maximum, changed_tensors=changed)
    return result


def evaluation_row(collected, index, demand_hash):
    metrics = dict(collected.metrics)
    population = collected.total_vehicle_count
    nonarrivals = int(metrics['evaluation_nonarrival_count'])
    episode_terminal = bool(collected.transitions[-1].terminal)
    complete = nonarrivals == 0
    return dict(episode_index=index, demand_hash=demand_hash, total_vehicle_count=population,
        arrived_count=population - nonarrivals, nonarrival_count=nonarrivals, complete=complete,
        episode_terminal=episode_terminal,
        observed_minutes_per_vehicle=collected.total_travel_minutes,
        travel_minutes_per_vehicle=collected.total_travel_minutes if complete else None,
        travel_time_is_censored=not complete, metrics=metrics)


def compare_paired(before, after):
    if len(before) != len(after):
        raise ValueError('paired evaluation membership differs')
    rows = []
    for first, second in zip(before, after):
        if first.get('engine_provenance') != second.get('engine_provenance'):
            raise ValueError('paired evaluation engine provenance differs')
        if (first['episode_index'], first['demand_hash'], first['total_vehicle_count']) != (
                second['episode_index'], second['demand_hash'], second['total_vehicle_count']):
            raise ValueError('paired evaluation does not share the identical realized demand')
        complete_pair = first['nonarrival_count'] == 0 and second['nonarrival_count'] == 0
        rows.append(dict(episode_index=first['episode_index'], demand_hash=first['demand_hash'],
            completion_rate_before=first['metrics']['evaluation_completion_rate'],
            completion_rate_after=second['metrics']['evaluation_completion_rate'],
            nonarrival_delta=second['nonarrival_count'] - first['nonarrival_count'],
            observed_minutes_delta=second['observed_minutes_per_vehicle'] - first['observed_minutes_per_vehicle'],
            completed_travel_minutes_delta=(second['travel_minutes_per_vehicle'] - first['travel_minutes_per_vehicle'])
                if complete_pair else None,
            censored_pair=not complete_pair,
            selection_score_delta=second['metrics']['evaluation_selection_score'] - first['metrics']['evaluation_selection_score']))
    population = sum(row['total_vehicle_count'] for row in before)
    return dict(pairs=rows, demand_vehicle_count=population,
        vehicle_weighted_completion_before=sum(row['arrived_count'] for row in before) / max(population, 1),
        vehicle_weighted_completion_after=sum(row['arrived_count'] for row in after) / max(population, 1),
        nonarrival_count_before=sum(row['nonarrival_count'] for row in before),
        nonarrival_count_after=sum(row['nonarrival_count'] for row in after),
        mean_paired_selection_score_delta=float(np.mean([row['selection_score_delta'] for row in rows])),
        note='Paired fixed demands; censored observed exposure is not completed travel time. No statistical or general traffic-performance claim.')


class PilotTrainer(MainlineTrainer):
    def __init__(self, *, pilot_log, **kwargs):
        super().__init__(**kwargs)
        self.pilot_log = Path(pilot_log)
        self.pilot_episode = -1
        self.pilot_pass = 0
        self.pilot_optimizer_steps = {name: 0 for name in MODEL_NAMES}
        for name in MODEL_NAMES:
            optimizer = self.optimizers[name]
            original = optimizer.step
            def step(*args, _name=name, _step=original, **kwargs):
                value = _step(*args, **kwargs)
                self.pilot_optimizer_steps[_name] += 1
                return value
            optimizer.step = step

    def log(self, record):
        record = dict(utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), **record)
        self.pilot_log.parent.mkdir(parents=True, exist_ok=True)
        with self.pilot_log.open('a') as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)

    def _run_episode(self, *, mode, episode_index):
        started = time.monotonic()
        self.log(dict(phase='real_episode_started', mode=mode, episode_index=episode_index))
        result = super()._run_episode(mode=mode, episode_index=episode_index)
        self.log(dict(phase='real_episode_finished', mode=mode, episode_index=episode_index,
            seconds=time.monotonic() - started, **result.metrics))
        return result

    def _update_world(self, *args, **kwargs):
        started = time.monotonic()
        result = super()._update_world(*args, **kwargs)
        self.log(dict(phase='online_world_model_update', episode_index=self.pilot_episode,
                      seconds=time.monotonic() - started, **result))
        return result

    def _train_imagination_pass(self):
        self.pilot_pass += 1
        started = time.monotonic()
        self.log(dict(phase='actor_critic_pass_started', episode_index=self.pilot_episode,
            pass_index=self.pilot_pass,
            rollouts=training.resolved_imagination_budget(self.config)['rollouts_per_pass'],
            horizon=training.IMAGINED_HORIZON))
        result = super()._train_imagination_pass()
        self.log(dict(phase='actor_critic_pass_finished', episode_index=self.pilot_episode,
            pass_index=self.pilot_pass, seconds=time.monotonic() - started,
            optimizer_steps=dict(self.pilot_optimizer_steps), **result))
        return result


class OnlinePilot:
    def __init__(self, trainer, *, output, input_hash, protocol, online_episodes=ONLINE_EPISODES,
                 evaluation_every=0, extend_run=False):
        self.trainer, self.output, self.input_hash = trainer, Path(output), input_hash
        self.online_episodes, self.evaluation_every, self.extend_run = online_episodes, evaluation_every, extend_run
        self.protocol = dict(protocol, online_episode_budget=online_episodes, evaluation_every=evaluation_every)
        self.budgets = self.protocol.get('budgets', BUDGETS)
        self.progress_path = self.output / 'pilot_progress.json'
        self.progress = dict(schema_version=1, before=[], online=[], after=[], milestones=[], phase='before_evaluation')

    def record_effective_budget(self):
        atomic_json(self.output / 'effective_run_budget.json', dict(
            authoritative_runner_online_episodes=self.online_episodes,
            source_config_online_training_episodes=self.trainer.config.run.online_training_episodes,
            evaluation_every=self.evaluation_every, learning_budgets=self.budgets,
            note='Dedicated runner controls the effective episode budget; source run_config is retained for strict model/checkpoint semantics.'))

    def preserve_completed_milestone(self):
        count = len(self.progress['online'])
        folder = self.output / 'milestones' / f'after_{count:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        result = self.output / 'pilot_result.json'
        if result.exists() and not (folder / 'result.json').exists():
            shutil.copy2(result, folder / 'result.json')
        evaluation = self.output / 'after_evaluation'
        if evaluation.exists() and not (folder / 'evaluation').exists():
            evaluation.replace(folder / 'evaluation')
        if not any(row['online_episodes'] == count for row in self.progress['milestones']):
            self.progress['milestones'].append(dict(online_episodes=count,
                phase='completed' if len(self.progress['after']) == len(EVALUATION_INDICES) else 'incomplete',
                evaluations=copy.deepcopy(self.progress['after']), result_path=str(folder / 'result.json')))
        if result.exists():
            result.unlink()
        self.progress['after'] = []
        self.progress['phase'] = 'online_training'

    def save(self, *, completed=False):
        state = self.trainer.state_dict()
        state['metrics'] = dict(state['metrics'], online_pilot_progress=copy.deepcopy(self.progress))
        save_checkpoint(output_dir=self.output, config_fingerprint=self.trainer.config.semantic_fingerprint,
            input_hash=self.input_hash, phase=CheckpointPhase.COMPLETED if completed else CheckpointPhase.ONLINE_TRAINING,
            next_unit=0 if completed else len(self.progress['online']), training_state=state)
        atomic_json(self.progress_path, self.progress)

    def resume(self):
        if self.progress_path.exists():
            saved = json.loads((self.output / 'pilot_protocol.json').read_text())
            previous_budget = saved.get('online_episode_budget', ONLINE_EPISODES)
            previous_every = saved.get('evaluation_every', 0)
            ignored = {'online_episode_budget', 'evaluation_every'}
            if {key: value for key, value in saved.items() if key not in ignored} != {
                    key: value for key, value in self.protocol.items() if key not in ignored}:
                raise ValueError('online pilot protocol/source identity differs from saved run')
            if self.online_episodes < previous_budget:
                raise ValueError('online episode budget cannot decrease on resume')
            changed = self.online_episodes != previous_budget or self.evaluation_every != previous_every
            if changed and not self.extend_run:
                raise ValueError('episode budget/evaluation interval change requires explicit --extend-run')
            self.progress = json.loads(self.progress_path.read_text())
            checkpoint = load_checkpoint(output_dir=self.output,
                expected_config_fingerprint=self.trainer.config.semantic_fingerprint,
                expected_input_hash=self.input_hash)
            self.trainer.load_state_dict(checkpoint.training_state)
            durable = checkpoint.training_state['metrics'].get('online_pilot_progress')
            if durable and (len(durable['online']) > len(self.progress['online'])
                            or checkpoint.phase is CheckpointPhase.COMPLETED):
                self.progress = durable
                atomic_json(self.progress_path, self.progress)
            elif durable and len(durable['online']) == len(self.progress['online']):
                # Periodic evaluation commits share the same training cursor.
                # The checkpoint is authoritative for those commits; legacy
                # before/after evaluations still use their isolated JSON rows.
                self.progress.setdefault('milestones', [])
                for committed in durable.get('milestones', []):
                    saved_milestone = next((row for row in self.progress['milestones']
                        if row['online_episodes'] == committed['online_episodes']), None)
                    if saved_milestone is None:
                        self.progress['milestones'].append(copy.deepcopy(committed))
                    elif (len(committed['evaluations']) > len(saved_milestone['evaluations'])
                          or committed['phase'] == 'completed'):
                        saved_milestone.update(copy.deepcopy(committed))
                atomic_json(self.progress_path, self.progress)
            if checkpoint.phase is not CheckpointPhase.COMPLETED and checkpoint.next_unit != len(self.progress['online']):
                raise ValueError('pilot checkpoint and per-episode progress differ')
            self.progress.setdefault('milestones', [])
            if len(self.progress['online']) < self.online_episodes and (
                    self.progress['phase'] == 'completed' or (changed and self.progress['after'])):
                self.preserve_completed_milestone()
                self.save()
            atomic_json(self.output / 'pilot_protocol.json', self.protocol)
            self.record_effective_budget()
            return
        if self.output.exists() and any(self.output.iterdir()):
            migration = (self.protocol.get('calibrated_noincident_initialization')
                         or self.protocol.get('meso_initialization'))
            if migration is None:
                raise ValueError('new online pilot requires a fresh output or matching pilot progress')
            checkpoint = load_checkpoint(output_dir=self.output,
                expected_config_fingerprint=self.trainer.config.semantic_fingerprint,
                expected_input_hash=self.input_hash)
            if (checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING or checkpoint.next_unit != 0
                    or sha256_file(checkpoint.path) != migration['target_initialization_sha256']):
                raise ValueError('fresh meso pilot must start at the verified initialization checkpoint')
            self.trainer.load_state_dict(checkpoint.training_state)
            self.trainer.validate_codec_ready()
            self.progress.update(evaluation_backend=self.trainer.config.simulation.backend,
                                 evaluation_engine_version=migration['target_engine_version'])
        atomic_json(self.output / 'pilot_protocol.json', self.protocol)
        export_run_config(self.trainer.config, self.output / 'run_config.toml')
        self.record_effective_budget()
        self.save()

    def evaluate(self, stage, *, rows=None, directory=None, durable=False):
        rows = self.progress[stage] if rows is None else rows
        for index in EVALUATION_INDICES[len(rows):]:
            destination = directory or self.output / (stage + '_evaluation')
            with isolated_evaluation(self.trainer, destination):
                collected = self.trainer._run_episode(mode='evaluation', episode_index=index)
            manifest = json.loads((destination / 'evaluation_demands' / f'episode_{index:04d}.json').read_text())
            row = evaluation_row(collected, index, manifest['demand_hash'])
            if self.progress.get('evaluation_engine_version'):
                row['engine_provenance'] = dict(backend=self.progress['evaluation_backend'],
                    version=self.progress['evaluation_engine_version'])
            rows.append(row)
            if durable:
                self.save()
            else:
                atomic_json(self.progress_path, self.progress)
        return rows

    def periodic_evaluation(self):
        count = len(self.progress['online'])
        if not self.evaluation_every or not count or count % self.evaluation_every or count == self.online_episodes:
            return
        milestone = next((row for row in self.progress['milestones'] if row['online_episodes'] == count), None)
        if milestone is None:
            milestone = dict(online_episodes=count, phase='evaluation', evaluations=[])
            self.progress['milestones'].append(milestone)
            self.save()
        if milestone['phase'] == 'completed':
            return
        directory = self.output / 'milestones' / f'after_{count:06d}' / 'evaluation'
        self.trainer.log(dict(phase='periodic_evaluation_started', online_episodes=count,
                              completed_demands=len(milestone['evaluations'])))
        self.evaluate('milestone', rows=milestone['evaluations'], directory=directory, durable=True)
        milestone['assessment'] = compare_paired(self.progress['before'], milestone['evaluations'])
        milestone['phase'] = 'completed'
        atomic_json(directory.parent / 'result.json', milestone)
        self.save()
        self.trainer.log(dict(phase='periodic_evaluation_completed', online_episodes=count,
                              assessment=milestone['assessment']))

    def run(self, *, baseline_only=False):
        self.resume()
        if self.progress['phase'] == 'completed':
            return json.loads((self.output / 'pilot_result.json').read_text())
        self.evaluate('before')
        self.progress['phase'] = 'online_training'
        atomic_json(self.progress_path, self.progress)
        if baseline_only:
            return dict(status='baseline_evaluated', before=self.progress['before'], budgets=self.budgets)
        self.periodic_evaluation()
        for index in range(len(self.progress['online']), self.online_episodes):
            previous = parameter_snapshot(self.trainer)
            self.trainer.pilot_episode, self.trainer.pilot_pass = index, 0
            steps = dict(self.trainer.pilot_optimizer_steps)
            started = time.monotonic()
            result = self.trainer.train_online_episode(index)
            delta = parameter_changes(self.trainer, previous)
            optimizer_steps = {name: self.trainer.pilot_optimizer_steps[name] - steps[name] for name in MODEL_NAMES}
            if result.get('online_vehicle_count', 1) > 0 and any(
                    optimizer_steps[name] == 0 or delta[name]['l2_change'] == 0 for name in MODEL_NAMES):
                raise RuntimeError('online episode did not update actor, macro V and event Q; refusing a WM-only success claim')
            self.progress['online'].append(dict(episode_index=index, seconds=time.monotonic() - started,
                metrics=result, optimizer_steps=optimizer_steps, parameter_changes=delta))
            self.save()
            self.trainer.log(dict(phase='online_episode_checkpoint', **self.progress['online'][-1]))
            self.periodic_evaluation()
        self.progress['phase'] = 'after_evaluation'
        atomic_json(self.progress_path, self.progress)
        self.evaluate('after')
        assessment = compare_paired(self.progress['before'], self.progress['after'])
        result = dict(status='completed', source=self.protocol['source'], budgets=self.budgets,
            effective_online_episode_budget=self.online_episodes,
            before=self.progress['before'], online=self.progress['online'], after=self.progress['after'],
            assessment=assessment, environment_episodes_consumed=self.trainer._next_environment_episode)
        atomic_json(self.output / 'pilot_result.json', result)
        self.progress['phase'] = 'completed'
        self.save(completed=True)
        return result


def validate_profile_continuation(output, config, input_hash, source_hash, args):
    """Resume a recorded profile change while retaining the frozen seed ancestry."""
    from tools.change_online_training_profile import changed_config, revised_protocol
    output = Path(output)
    record = json.loads((output / 'latest_profile_change.json').read_text())
    if (record.get('kind') != 'imagination_and_policy_prior_profile_change'
            or record.get('new_fingerprint') != config.semantic_fingerprint
            or record.get('model_optimizer_replay_rng_preserved') is not True):
        raise ValueError('online profile change identity or preservation proof differs')
    backup = Path(record['backup'])
    old = load_config(backup / 'run_config.toml')
    if (old.semantic_fingerprint != record['old_fingerprint']
            or sha256_file(backup / 'checkpoint.pt') != record['old_checkpoint_sha256']):
        raise ValueError('online profile change source ancestry differs')
    settings = record['after']
    expected = changed_config(old, rollouts=settings['rollouts'],
        evaluation_weight=settings['evaluation_prior'], hold=settings['hold'], off=settings['off'],
        precision=settings['imagination_precision'])
    if expected.semantic_fingerprint != config.semantic_fingerprint:
        raise ValueError('online runtime configuration differs from its recorded profile change')
    protocol = json.loads((output / 'pilot_protocol.json').read_text())
    original = json.loads((backup / 'pilot_protocol.json').read_text())
    if protocol != revised_protocol(original, config):
        raise ValueError('online profile continuation protocol differs from its recorded source')
    if protocol['source']['sha256'] != source_hash:
        raise ValueError('online profile continuation pretraining seed ancestry differs')
    if 'evaluation' in protocol:
        temperature = None if args.evaluation_policy == 'argmax' else args.evaluation_temperature
        evaluation = training.evaluation_protocol(temperature,
            prior_weight=config.method.sp_prior_evaluation_weight,
            demand_seed=config.method.evaluation_demand_seed)
        if protocol['evaluation'] != evaluation:
            raise ValueError('online profile continuation evaluation policy differs')
    elif args.evaluation_policy != 'argmax':
        raise ValueError('legacy online profile continuation requires its frozen argmax evaluation policy')
    for name, value in (('online_episode_budget', args.online_episodes),
                        ('evaluation_every', args.evaluation_every), ('threads', args.threads),
                        ('gpu_memory_fraction', args.gpu_memory_fraction), ('device', args.device)):
        if protocol[name] != value:
            raise ValueError(f'online profile continuation runner setting differs: {name}')
    checkpoint = load_checkpoint(output_dir=output,
        expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=input_hash)
    history = checkpoint.training_state['metrics'].get('online_pilot_progress', {})
    if (checkpoint.phase is not CheckpointPhase.ONLINE_TRAINING
            or checkpoint.next_unit < record['boundary']
            or len(history.get('online', [])) != checkpoint.next_unit):
        raise ValueError('online profile continuation checkpoint/history cursor differs')
    return protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'outputs/ingolstadt_bf16_pretrain_20261002')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ingolstadt_online_pilot_20261003')
    parser.add_argument('--device', choices=('cpu', 'cuda:0', 'cuda:1'), default='cuda:1')
    parser.add_argument('--run-config', type=Path, help='explicit migrated runtime configuration; source pretrain provenance stays separate')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=.06)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--online-episodes', type=int, default=ONLINE_EPISODES)
    parser.add_argument('--evaluation-every', type=int, default=0)
    parser.add_argument('--evaluation-temperature', type=float, default=.2)
    parser.add_argument('--evaluation-policy', choices=('sampled', 'argmax'), default='argmax')
    parser.add_argument('--extend-run', action='store_true',
        help='explicitly increase a saved run budget; all source/model/learning settings must match')
    args = parser.parse_args(argv)
    if args.threads < 1 or not math.isfinite(args.gpu_memory_fraction) or not 0 < args.gpu_memory_fraction <= 1:
        parser.error('positive threads and finite gpu fraction in (0,1] are required')
    if args.online_episodes < 1 or args.evaluation_every < 0:
        parser.error('online episodes must be positive and evaluation interval nonnegative')
    if not math.isfinite(args.evaluation_temperature) or args.evaluation_temperature <= 0:
        parser.error('evaluation temperature must be finite and positive')
    evaluation_temperature = None if args.evaluation_policy == 'argmax' else args.evaluation_temperature
    assert_original_budgets()
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda:'):
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        free, _ = torch.cuda.mem_get_info(device)
        if free < 2 * 1024 ** 3:
            raise RuntimeError(f'{args.device} has less than2GiB available; no silent CPU fallback')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    source_config = load_config(args.source_dir / 'run_config.toml')
    source_raw = torch.load(latest_checkpoint_path(args.source_dir), map_location='cpu', weights_only=True)
    # The frozen pretrainer used the old semantic payload. A current loader must
    # prove that exact identity before treating it as an initialization source.
    from tools.queue_pretrained_online import checkpoint_source_config, LEGACY_REWARD_CONTRACT
    source_config = checkpoint_source_config(source_config, source_raw)
    legacy_source = source_raw['training_state']['reward_contract'] == LEGACY_REWARD_CONTRACT
    del source_raw
    runtime_config = load_config(args.run_config) if args.run_config else source_config
    if legacy_source and not args.run_config:
        runtime_config = replace(runtime_config, method=replace(runtime_config.method, disagreement_weight=0.))
        runtime_config = replace(runtime_config, semantic_fingerprint=_semantic_fingerprint(**{
            key: getattr(runtime_config, key) for key in ('inputs', 'run', 'simulation', 'resources', 'method', 'data', 'codec')}))
    config = replace(runtime_config, run=replace(runtime_config.run, output_dir=args.output_dir.resolve(), device=args.device))
    entropy_path = args.output_dir / 'entropy_migration.json'
    entropy_migration = json.loads(entropy_path.read_text()) if args.run_config and entropy_path.is_file() else None
    entropy_mode = os.environ.get('URB_ACTOR_ENTROPY_MODE', config.method.actor_entropy_mode)
    if entropy_mode != config.method.actor_entropy_mode:
        raise ValueError('production entropy override must match the fingerprinted method.actor_entropy_mode')
    calibrated_path = args.output_dir / 'calibrated_noincident_initialization.json'
    calibrated_initialization = json.loads(calibrated_path.read_text()) if args.run_config and calibrated_path.is_file() else None
    prior_path = args.output_dir / 'policy_prior_migration.json'
    prior_migration = json.loads(prior_path.read_text()) if args.run_config and prior_path.is_file() else None
    meso_path = args.output_dir / 'meso_initialization.json'
    meso_initialization = json.loads(meso_path.read_text()) if args.run_config and meso_path.is_file() else None
    budget_path = args.output_dir / 'imagination_budget_migration.json'
    budget_migration = json.loads(budget_path.read_text()) if args.run_config and budget_path.is_file() else None
    decoder_path = args.output_dir / 'decoder_migration.json'
    decoder_migration = json.loads(decoder_path.read_text()) if args.run_config and decoder_path.is_file() else None
    profile_path = args.output_dir / 'latest_profile_change.json'
    profile_migration = json.loads(profile_path.read_text()) if args.run_config and profile_path.is_file() else None
    if args.run_config:
        if (meso_initialization is None and prior_migration is None and calibrated_initialization is None
                and entropy_migration is None and budget_migration is None and decoder_migration is None
                and profile_migration is None):
            from tools.migrate_online_backend import validate_backend_only_change
            validate_backend_only_change(source_config, config)
            if not (args.output_dir / 'backend_migration.json').is_file():
                raise ValueError('--run-config requires an explicit completed backend or meso initialization migration')
    prepared = validate_prepared_inputs(config)
    topology = load_traffic_topology(prepared)
    input_hash = combined_input_hash(prepared)
    source_checkpoint = load_checkpoint(output_dir=args.source_dir,
        expected_config_fingerprint=source_config.semantic_fingerprint, expected_input_hash=input_hash)
    validate_pretrained_source(source_checkpoint, source_config)
    source_hash = sha256_file(source_checkpoint.path)
    protocol = dict(schema_version=1, source=dict(path=str(source_checkpoint.path.resolve()), sha256=source_hash,
        config_fingerprint=source_config.semantic_fingerprint, input_hash=input_hash), budgets=learning_budgets(config),
        online_episode_budget=args.online_episodes, evaluation_every=args.evaluation_every,
        paired_evaluation_indices=list(EVALUATION_INDICES),
        evaluation_demand_seed=config.method.evaluation_demand_seed,
        session_protocol=config.simulation.session_protocol, observation_seconds=config.simulation.observation_step_seconds,
        backend=config.simulation.backend, device=args.device, threads=args.threads, gpu_memory_fraction=args.gpu_memory_fraction,
        config_semantics_unchanged=not bool(args.run_config), output_isolated=True)
    protocol['reward_contract'] = training.REWARD_CONTRACT
    protocol['global_reward'] = dict(formula='-(m(end)-m(start))/60',
        trip_start='actual_departure', normalization='cumulative_actual_started_count',
        unfinished='observed_time_only', individual_timeout_penalty=False,
        imagination_disagreement_weight=config.method.disagreement_weight)
    if legacy_source:
        protocol['config_semantics_unchanged'] = False
        protocol['runtime_config_fingerprint'] = config.semantic_fingerprint
    handoff_path = args.source_dir / 'handoff_manifest.json'
    if handoff_path.is_file():
        handoff = json.loads(handoff_path.read_text())
        if handoff.get('reward_contract') != training.REWARD_CONTRACT:
            raise ValueError('online source handoff reward contract differs from trainer')
        for name, digest in handoff['frozen_artifacts'].items():
            if sha256_file(args.source_dir / name) != digest:
                raise ValueError(f'online source handoff frozen artifact changed: {name}')
        protocol['pretrain_handoff'] = handoff
    if config.method.sp_prior_weight > 0 or config.method.sp_prior_evaluation_weight > 0:
        protocol['shortest_path_prior'] = dict(
            initial_weight=config.method.sp_prior_weight,
            temperature_seconds=config.method.sp_prior_temperature,
            hold_episodes=config.method.sp_prior_hold_episodes,
            off_episode=config.method.sp_prior_off_episode,
            schedule='cosine_global_one_based_episode',
            evaluation_weight=config.method.sp_prior_evaluation_weight,
            evaluation_policy=training.evaluation_protocol(evaluation_temperature,
                prior_weight=config.method.sp_prior_evaluation_weight)['archive_policy'])
    if entropy_mode == 'annealed':
        from tools.continue_mainline_entropy_annealing import entropy_protocol
        protocol['actor_entropy'] = entropy_protocol(config.method)
        protocol['runtime_config_fingerprint'] = config.semantic_fingerprint
    elif entropy_mode != 'legacy':
        protocol['actor_entropy'] = dict(mode=entropy_mode, base_coefficient=.002,
            coefficient='base/(episode_population*max(global_advantage_std,1))' if entropy_mode == 'population' else 'zero')
    if profile_migration is not None:
        protocol = validate_profile_continuation(args.output_dir, config, input_hash, source_hash, args)
    elif decoder_migration is not None:
        from tools.migrate_calibrated_decoder import validate_decoder_continuation, rewrite_protocol
        validate_decoder_continuation(args.output_dir, config, input_hash, source_hash)
        protocol = rewrite_protocol(json.loads((args.output_dir / 'source_protocol.json').read_text()),
                                    config, decoder_migration)
        if (args.online_episodes != protocol['online_episode_budget']
                or args.evaluation_every != protocol['evaluation_every']
                or args.threads != protocol['threads']
                or args.gpu_memory_fraction != protocol['gpu_memory_fraction']
                or args.device != protocol['device']):
            raise ValueError('decoder continuation runner settings differ from the saved protocol')
    elif budget_migration is not None:
        from tools.migrate_imagination_budget import validate_budget_continuation, rewrite_protocol
        validate_budget_continuation(args.output_dir, config, input_hash, source_hash)
        protocol = rewrite_protocol(json.loads((args.output_dir / 'source_protocol.json').read_text()),
                                    config, budget_migration)
        if (args.online_episodes != protocol['online_episode_budget']
                or args.evaluation_every != protocol['evaluation_every']
                or args.threads != protocol['threads']
                or args.gpu_memory_fraction != protocol['gpu_memory_fraction']
                or args.device != protocol['device']):
            raise ValueError('budget continuation runner settings differ from the saved protocol')
    elif entropy_migration is not None:
        if entropy_migration.get('kind') == 'fixed_entropy_annealing_continuation':
            from tools.continue_mainline_entropy_annealing import validate_continuation
            validate_continuation(args.output_dir, config, input_hash, source_hash)
        else:
            from tools.migrate_mainline_entropy import validate_entropy_continuation
            validate_entropy_continuation(args.output_dir, config, input_hash, source_hash)
        if args.online_episodes != config.run.online_training_episodes:
            raise ValueError('entropy continuation episode budget must match runtime configuration')
        protocol.update(runtime_config_fingerprint=config.semantic_fingerprint, entropy_migration=entropy_migration)
        if calibrated_initialization is not None:
            protocol['calibrated_noincident_initialization'] = calibrated_initialization
    elif calibrated_initialization is not None:
        from tools.migrate_calibrated_meso_noincident import validate_calibrated_noincident_manifest
        validate_calibrated_noincident_manifest(calibrated_initialization, config, args.output_dir, input_hash)
        if args.online_episodes != config.run.online_training_episodes:
            raise ValueError('calibrated meso runner episode budget must match runtime configuration')
        protocol.update(runtime_config_fingerprint=config.semantic_fingerprint,
                        calibrated_noincident_initialization=calibrated_initialization)
    elif prior_migration is not None:
        from tools.migrate_mainline_prior import validate_prior_continuation
        validate_prior_continuation(args.output_dir, config, input_hash, source_hash)
        if args.online_episodes != config.run.online_training_episodes:
            raise ValueError('prior continuation episode budget must match runtime configuration')
        protocol.update(runtime_config_fingerprint=config.semantic_fingerprint,
                        policy_prior_migration=prior_migration)
        if meso_initialization is not None:
            protocol['meso_initialization'] = meso_initialization
    elif meso_initialization is not None:
        from tools.migrate_pretrained_meso import validate_initialization_manifest
        validate_initialization_manifest(meso_initialization, source_checkpoint, source_config,
                                         config, input_hash, args.output_dir)
        if args.online_episodes != config.run.online_training_episodes:
            raise ValueError('meso runner episode budget must match the target run configuration')
        protocol.update(runtime_config_fingerprint=config.semantic_fingerprint,
                        meso_initialization=meso_initialization)
    elif args.run_config:
        migration = json.loads((args.output_dir / 'backend_migration.json').read_text())
        if (migration['target_config_fingerprint'] != config.semantic_fingerprint
                or migration['source_config_fingerprint'] != source_config.semantic_fingerprint
                or migration['target_output'] != str(args.output_dir.resolve())):
            raise ValueError('runtime configuration does not match the explicit backend migration')
        import libsumo
        if libsumo.__version__ != migration['target_engine_version']:
            raise ValueError('installed libsumo version differs from migrated engine provenance')
        protocol.update(runtime_config_fingerprint=config.semantic_fingerprint, backend_migration=migration)
    if profile_migration is None:
        protocol['evaluation'] = training.evaluation_protocol(evaluation_temperature,
            prior_weight=config.method.sp_prior_evaluation_weight, demand_seed=config.method.evaluation_demand_seed)
        protocol['evaluation_policy'] = protocol['evaluation']['archive_policy']
    trainer = PilotTrainer(config=config, prepared_inputs=prepared, topology=topology,
                           pilot_log=args.output_dir / 'pilot_metrics.jsonl')
    trainer.evaluation_temperature = evaluation_temperature
    trainer.world_model.flow_projector.__class__ = SparseGraphCodec
    trainer.world_model.flow_projector.register_buffer('sparse_adjacency',
        trainer._tensor(topology.network.adjacency).to_sparse().coalesce(), persistent=False)
    if profile_migration is not None or decoder_migration is not None or budget_migration is not None:
        continued = load_checkpoint(output_dir=args.output_dir,
            expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=input_hash)
        trainer.load_state_dict(continued.training_state)
    elif entropy_migration is not None:
        continued = load_checkpoint(output_dir=args.output_dir,
            expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=input_hash)
        trainer.load_state_dict(continued.training_state)
    elif calibrated_initialization is not None:
        initialized = torch.load(args.output_dir / 'initialization_checkpoint.pt', map_location='cpu', weights_only=True)
        if (initialized['config_fingerprint'] != config.semantic_fingerprint or initialized['input_hash'] != input_hash
                or initialized['phase'] != CheckpointPhase.ONLINE_TRAINING.value or initialized['next_unit'] != 0):
            raise ValueError('calibrated no-incident initialization checkpoint differs')
        trainer.load_state_dict(initialized['training_state'])
    elif prior_migration is not None:
        continued = load_checkpoint(output_dir=args.output_dir,
            expected_config_fingerprint=config.semantic_fingerprint, expected_input_hash=input_hash)
        trainer.load_state_dict(continued.training_state)
    elif meso_initialization is not None:
        initialized = torch.load(args.output_dir / 'initialization_checkpoint.pt', map_location='cpu', weights_only=True)
        if (initialized['config_fingerprint'] != config.semantic_fingerprint or initialized['input_hash'] != input_hash
                or initialized['phase'] != CheckpointPhase.ONLINE_TRAINING.value or initialized['next_unit'] != 0):
            raise ValueError('meso initialization checkpoint config/input/phase differs')
        trainer.load_state_dict(initialized['training_state'])
        from tools.migrate_pretrained_meso import validate_target_initialization
        validate_target_initialization(initialized['training_state'], source_checkpoint, meso_initialization)
    else:
        if legacy_source:
            from tools.queue_pretrained_online import migrate_pretrained_reward
            initial_state, migration = migrate_pretrained_reward(source_checkpoint.training_state)
            protocol['reward_migration'] = migration
            trainer.load_state_dict(initial_state)
        else:
            trainer.load_state_dict(source_checkpoint.training_state)
    trainer.validate_codec_ready()
    runner = OnlinePilot(trainer, output=args.output_dir, input_hash=input_hash, protocol=protocol,
        online_episodes=args.online_episodes, evaluation_every=args.evaluation_every, extend_run=args.extend_run)
    try:
        result = runner.run(baseline_only=args.baseline_only)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
        if sha256_file(source_checkpoint.path) != source_hash:
            raise RuntimeError('pretrained source changed during isolated online pilot')
    except Exception as error:
        atomic_json(args.output_dir.with_suffix('.failure.json'), dict(type=type(error).__name__, error=str(error),
            last_saved_online_episodes=len(runner.progress['online']),
            note='Budgets were not reduced; resume only from the last per-episode checkpoint.'))
        raise
    finally:
        trainer.close()


if __name__ == '__main__':
    main()
