from __future__ import annotations

import copy
import json
import math
import os
import hashlib
import random
import time
import tempfile
from contextlib import contextmanager, nullcontext
from itertools import islice
from functools import partial
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from ..workflow.config import MainlineConfig
from .entropy_schedule import entropy_schedule_profile
from .critic_budget import remaining_cost_budget
from ..traffic.demand import PoissonDemand, demand_identity
from ..traffic.routing import PreparedInputs
from ..traffic.topology import NextHopCandidates, TrafficTopology
from ..models.policy import (CentralizedLatentValueCritic, NextHopActor, lambda_return,
    attach_timeout_penalties, individual_actor_advantage, population_scaled_entropy,
    set_shortest_path_prior, shortest_path_prior_weight)
from ..models.world_model import FLOW_LATENT_DIM, UrbanDreamerWorldModel, independent_transition_fm_loss, recursive_fm_loss
from .observations import (build_actor_observations, prepare_actor_window, PreparedActorWindow,
                           selected_decisions)
from .sumo_sampling import SumoEpisodeSampler, resolve_session_factory
from .imagination import advance_imagined_window, clip_road_speed_fractions
from .replay import MacroTransition, ReplayBuffer
from .global_reward import REWARD_CONTRACT, REWARD_SCALE, causal_context, observed_mean_seconds
from .action_events import ActionEvent, pack_action_events
from .routing_evaluation_metrics import (DiversionCost, all_demand_censored_metrics,
    DIVERSION_COST_DEFINITION, ALL_DEMAND_CENSORED_DEFINITION)
from .episode_archive import EpisodeArchive
from .offline_data import OfflineEpisodeDataset
from .cohort_targets import prepare_transition_targets, prepare_cohort_batch
from .codec_training import (codec_train_step, codec_metrics, make_codec_optimizer,
                             codec_objective, decoder_parameters, make_decoder_optimizer)
from ..models.event_critic import (EventActionCritic, build_event_q_inputs,
    detached_counterfactual_baseline)

WORLD_MODEL_INTERVAL_SECONDS = 30
ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE = 4
IMAGINED_ROLLOUTS_PER_PASS = 8
IMAGINED_PASSES_PER_EPISODE = 16
IMAGINED_HORIZON = 8
ACTOR_ENTROPY_COEF = 0.002
GRAD_CLIP = 10.0


def evaluation_protocol(temperature: float | None = None, *, prior_weight: float = 0.,
                        demand_seed: int = 0) -> dict[str, Any]:
    """Describe runtime evaluation without changing pretrained config identity."""
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('evaluation temperature must be finite and positive')
    policy = 'argmax' if temperature is None else f'sampled_temperature_{temperature:g}'
    prefix = 'shortest_path_prior' if prior_weight > 0 else 'zero_prior'
    return dict(policy=policy, archive_policy=f'{prefix}_{policy}', temperature=temperature,
                prior_weight=prior_weight, action_seed_base=int(demand_seed) + 1000003,
                action_seed_rule='action_seed_base + evaluation_episode_index',
                training_rng_isolated=True)


@contextmanager
def isolated_evaluation_rng(trainer, episode_index: int):
    """Repeat paired policy sampling while preserving every training RNG stream."""
    seed = evaluation_protocol(getattr(trainer, 'evaluation_temperature', None),
        demand_seed=trainer.config.method.evaluation_demand_seed)['action_seed_base'] + episode_index
    python_state, numpy_state = random.getstate(), np.random.get_state()
    generator = trainer.rng
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        trainer.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        trainer.rng = generator
        torch.set_rng_state(torch_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)


def resolved_imagination_budget(config: MainlineConfig) -> dict[str, int | float]:
    """Resolve episode budgets without changing the contents of a full pass."""
    return dict(world_model_updates_per_episode=ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE,
                imagination_passes_per_episode=config.resources.imagination_passes_per_episode,
                imagination_time_budget_seconds=config.resources.imagination_time_budget_seconds,
                rollouts_per_pass=config.resources.imagination_rollouts_per_pass,
                macro_windows_per_rollout=IMAGINED_HORIZON)


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows if key in row]))
            for key in {key for row in rows for key in row}}


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, (list, tuple)) else value


@dataclass(frozen=True)
class CollectedEpisode:
    transitions: tuple[MacroTransition, ...]
    total_travel_minutes: float
    metrics: dict[str, float]
    total_vehicle_count: int
    arrival_times: dict[str, float] | None = None


class MainlineTrainer:
    """Collect physical event windows, then train macro dynamics and event policies."""
    def __init__(self, *, config: MainlineConfig, prepared_inputs: PreparedInputs,
                 topology: TrafficTopology, _session_factory: Callable[..., Any] | None = None,
                 _offline_data=None, _codec_validation_data=None) -> None:
        self.config, self.prepared_inputs, self.topology = config, prepared_inputs, topology
        self.device = torch.device(config.run.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('configured CUDA device is unavailable')
        random.seed(config.run.seed)
        np.random.seed(config.run.seed)
        torch.manual_seed(config.run.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.run.seed)
        self.rng = np.random.default_rng(config.run.seed)
        self.network = topology.network
        self.edge_count = self.network.edge_count
        self.demand = PoissonDemand.from_vehicles(topology.vehicles,
            duration_seconds=config.simulation.demand_duration_seconds)
        # Reference input count; physical and replay trajectories carry realized N.
        self.total_vehicle_count = len(topology.vehicles)
        self.candidates = NextHopCandidates(self.network,
            traffic_model=config.simulation.traffic_model)
        if getattr(config.simulation, 'routing_mode', 'next_hop') == 'current_to_dest':
            from ..traffic.route_candidates import RouteCandidates
            self.candidates = RouteCandidates(self.network, traffic_model=config.simulation.traffic_model)
        self.microstep_seconds = 1.0
        self._session_factory = _session_factory
        self._next_environment_episode = 0
        self.world_model = UrbanDreamerWorldModel(
            adjacency=self._tensor(self.network.adjacency),
            static_features=self._tensor(self.network.static_features),
            low_frequency_rank=config.method.low_frequency_rank,
            prior_noise_scale=config.method.prior_noise_scale, event_protocol=True,
            prior_filter=getattr(config.method, 'prior_filter', 'hard'),
            prior_filter_strength=getattr(config.method, 'prior_filter_strength', 10.0),
            prior_action_conditioned=getattr(config.method, 'prior_action_conditioned', True),
            prior_filter_input=getattr(config.method, 'prior_filter_input', False),
            prior_usage=getattr(config.method, 'prior_usage', 'source_reference'),
            codec_speed_positive=config.codec.speed_positive).to(self.device)
        self.actor = NextHopActor().to(self.device)
        self.evaluation_temperature = None
        bounded = self._bounded_critic()
        self.latent_critic = CentralizedLatentValueCritic(bounded=bounded).to(self.device)
        self.event_critic = EventActionCritic(bounded=bounded).to(self.device)
        if bounded:
            if config.method.disagreement_weight != 0:
                raise ValueError('bounded cohort critics require zero disagreement penalty')
            self.target_latent_critic = copy.deepcopy(self.latent_critic).eval().requires_grad_(False)
            self._critic_aux_rng = np.random.default_rng(config.run.seed + 91873)
            self._critic_aux_torch_cpu = torch.Generator().manual_seed(config.run.seed + 91873).get_state()
            self._critic_aux_torch_cuda = [torch.Generator(device=f'cuda:{i}').manual_seed(
                config.run.seed + 91873 + i).get_state() for i in range(torch.cuda.device_count())]
        self._world_parameters = list(self.world_model.dynamics.parameters())
        self.optimizers = {
            'world_model_initial': torch.optim.AdamW(self._world_parameters, lr=1e-4, weight_decay=1e-2),
            'world_model_online': torch.optim.Adam(self._world_parameters, lr=1e-5),
            'active_fraction': torch.optim.Adam(self.world_model.active_vehicle_fraction_readout.parameters(), lr=1e-3),
            'actor': torch.optim.Adam(self.actor.parameters(), lr=2e-4),
            'latent_critic': torch.optim.Adam(self.latent_critic.parameters(), lr=3e-4),
            'event_critic': torch.optim.Adam(self.event_critic.parameters(), lr=3e-4),
            'codec': make_codec_optimizer(self.world_model, lr=config.codec.learning_rate),
        }
        if config.codec.decoder_pretrain_enabled:
            self.optimizers['decoder_pretrain'] = make_decoder_optimizer(
                self.world_model, lr=config.codec.decoder_pretrain_learning_rate)
        if config.codec.decoder_online_enabled:
            self.optimizers['decoder_online'] = make_decoder_optimizer(
                self.world_model, lr=config.codec.decoder_online_learning_rate)
        self.online_replay = ReplayBuffer(sequence_length=config.method.supervision_horizon)
        if _offline_data is None:
            self.offline_data, self.codec_validation_data = self._pretraining_sources()
        else:
            self.offline_data, self.codec_validation_data = _offline_data, _codec_validation_data
        self.metrics = {key: 0. for key in (
            'codec_epochs_completed', 'codec_updates', 'codec_ready',
            'world_model_initialization_epochs_completed',
            'world_model_initialization_batches_completed', 'world_model_initialization_samples_seen',
            'world_model_initialization_updates',
            'world_model_epoch_samples', 'world_model_epoch_batches', 'world_model_epoch_updates',
            'world_model_epoch_loss_sum', 'world_model_epoch_readout_loss_sum',
            'online_episodes_completed', 'evaluation_episodes_completed',
            'evaluation_t_all_sum', 'evaluation_t_all_mean', 'evaluation_vehicle_minutes_sum',
            'evaluation_vehicle_count_sum', 'evaluation_t_all_vehicle_weighted',
            'world_model_updates', 'world_model_loss_last',
            'active_fraction_loss_last', 'actor_loss_last',
            'latent_critic_loss_last', 't_all_last')}
        if bounded:
            self.metrics.update(critic_value_mode=config.method.critic_value_mode,
                critic_target_tau=config.method.critic_target_tau,
                critic_terminal_aux_ratio=config.method.critic_terminal_aux_ratio)
        self._set_online_prior()
        if config.codec.decoder_pretrain_enabled:
            self.metrics['decoder_pretrain_updates'] = 0.
        if config.codec.decoder_online_enabled:
            self.metrics['decoder_online_updates'] = 0.

    def _bounded_critic(self):
        return getattr(self.config.method, 'critic_value_mode', 'legacy') == 'bounded_cohort_v1'

    def _value_prediction(self, latent, context, budgets=None, *, target=False):
        if not self._bounded_critic():
            return self.latent_critic(latent, context)
        model = self.target_latent_critic if target else self.latent_critic
        budget = torch.as_tensor(budgets, device=self.device, dtype=torch.float32)
        value = model(latent, context, budget=budget)
        if hasattr(self, '_critic_diagnostics'):
            with torch.no_grad():
                flat = value.detach().reshape(-1)
                positive = budget > 1e-6
                fraction = -flat[positive] / budget[positive]
                stats = self._critic_diagnostics
                stats['critic_predictions'] += flat.numel()
                stats['critic_nonfinite_predictions'] += int((~torch.isfinite(flat)).sum())
                stats['critic_bound_violations'] += int(((flat > 1e-6) | (flat < -budget - 1e-6)).sum())
                stats['critic_fraction_saturated'] += int(((fraction < .01) | (fraction > .99)).sum())
                stats['critic_terminal_predictions'] += int((budget == 0).sum())
                stats['critic_tiny_budget_predictions'] += int(((budget > 0) & ~positive).sum())
        return value

    def _validate_critic_targets(self, targets, budgets):
        if not self._bounded_critic():
            return
        budget = torch.as_tensor(budgets, dtype=torch.float32, device=targets.device).reshape(-1)
        target = targets.detach().float().reshape(-1)
        if budget.shape != target.shape:
            raise ValueError('bounded critic target/budget shape mismatch')
        tolerance = 1e-5 * budget.clamp_min(1.)
        invalid = (~torch.isfinite(target) | ~torch.isfinite(budget) | (budget < 0)
                   | (target > tolerance) | (target < -budget - tolerance))
        if hasattr(self, '_critic_diagnostics'):
            self._critic_diagnostics['critic_target_count'] += target.numel()
            self._critic_diagnostics['critic_target_bound_violations'] += int(invalid.sum())
        if bool(invalid.any()):
            raise ValueError('bounded critic target violates finite physical return bounds')

    @torch.no_grad()
    def _update_target_critic(self):
        if not self._bounded_critic():
            return
        tau = self.config.method.critic_target_tau
        for target, online in zip(self.target_latent_critic.parameters(), self.latent_critic.parameters()):
            target.lerp_(online, tau)
        for target, online in zip(self.target_latent_critic.buffers(), self.latent_critic.buffers()):
            if target.is_floating_point():
                target.lerp_(online, tau)
            else:
                target.copy_(online)

    def _terminal_aux_losses(self, rollout_count):
        if not self._bounded_critic():
            return []
        count = math.ceil(self.config.method.critic_terminal_aux_ratio * rollout_count)
        if not count:
            return []
        batch = self.online_replay.sample_terminal_seeds(batch_size=count, rng=self._critic_aux_rng,
            deadline=self._episode_end_time(), horizon_seconds=IMAGINED_HORIZON * WORLD_MODEL_INTERVAL_SECONDS)
        if batch is None:
            return []
        losses = []
        # Auxiliary actions and WM noise have a checkpointed private stream.
        # fork_rng restores the original CPU/CUDA streams even on exceptions.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.set_rng_state(self._critic_aux_torch_cpu)
            if torch.cuda.is_available() and self._critic_aux_torch_cuda:
                torch.cuda.set_rng_state_all(self._critic_aux_torch_cuda[:torch.cuda.device_count()])
            try:
                for index in range(len(batch.time)):
                    _, loss = self._imagine_from_seed(batch, index, rollout_index=index,
                        batched_events=True, copy_seed=False, critic_only=True)
                    losses.append(loss)
            finally:
                self._critic_aux_torch_cpu = torch.get_rng_state()
                self._critic_aux_torch_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        return losses

    def _set_online_prior(self):
        """Use the next global one-based round, matching the baseline schedule."""
        method = self.config.method
        weight = shortest_path_prior_weight(int(self.metrics['online_episodes_completed']) + 1,
            initial_weight=method.sp_prior_weight, hold_episodes=method.sp_prior_hold_episodes,
            off_episode=method.sp_prior_off_episode)
        set_shortest_path_prior(self.actor, weight, method.sp_prior_temperature)
        return weight

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _pretraining_sources(self):
        config = self.config
        length = config.resources.world_model_initialization_sequence_length
        if config.data.pretrain_cache_manifest is not None:
            from .pretrain_cache import CompactPretrainDataset
            training = CompactPretrainDataset(config, self.prepared_inputs, self.topology, length,
                cache_manifest_path=config.data.pretrain_cache_manifest, split='train')
            validation = CompactPretrainDataset(config, self.prepared_inputs, self.topology, length,
                cache_manifest_path=config.data.validation_cache_manifest or config.data.pretrain_cache_manifest,
                split='validation')
            train_groups = {entry['demand_group'] for entry in training.manifest['entries']}
            validation_groups = {entry['demand_group'] for entry in validation.manifest['entries']}
            if train_groups & validation_groups:
                raise ValueError('codec train and validation caches share realized demand identities')
            return training, validation
        # An existing split remains frozen across resume. New sources are split
        # by realized demand identity, shared across policies and directories.
        train_manifest = config.run.output_dir / 'offline_manifest.json'
        validation_manifest = config.run.output_dir / 'codec_validation_manifest.json'
        if train_manifest.is_file() and validation_manifest.is_file():
            val_config = (replace(config, data=replace(config.data, archive_roots=config.data.validation_archive_roots))
                          if config.data.validation_archive_roots else config)
            return (OfflineEpisodeDataset(config, self.prepared_inputs, self.topology, length),
                    OfflineEpisodeDataset(val_config, self.prepared_inputs, self.topology, length,
                                          manifest_path=validation_manifest))
        with tempfile.TemporaryDirectory(prefix='urban-pretrain-inventory-') as temporary:
            inventory = OfflineEpisodeDataset(config, self.prepared_inputs, self.topology, length,
                manifest_path=Path(temporary) / 'manifest.json')
        if not inventory.has_samples:
            return inventory, None
        def group(path):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(archive['summary_json'].tobytes().decode())['metadata']
            identity = metadata.get('demand_sha256', metadata.get('demand_realization_sha256'))
            if not identity:
                realization = metadata.get('realized_demand', metadata.get('demand_realization'))
                if realization is None:
                    raise ValueError('codec validation requires realized-demand identity')
                identity = hashlib.sha256(json.dumps(realization, sort_keys=True).encode()).hexdigest()
            return str(identity)
        files = [(entry['path'], group(entry['path'])) for entry in inventory.manifest['entries']]
        if config.data.validation_archive_roots:
            val_config = replace(config, data=replace(config.data, archive_roots=config.data.validation_archive_roots))
            validation = OfflineEpisodeDataset(val_config, self.prepared_inputs, self.topology, length,
                                              manifest_path=validation_manifest)
            validation_groups = {group(entry['path']) for entry in validation.manifest['entries']}
            if validation_groups.intersection(identity for _, identity in files):
                raise ValueError('codec train and validation sources share realized demand identities')
            paths = [path for path, _ in files]
        else:
            identities = sorted({identity for _, identity in files})
            if len(identities) < 2:
                raise ValueError('codec requires at least two independent demand identities for validation')
            rng = np.random.default_rng(config.run.seed)
            reserved = {identities[int(index)] for index in rng.permutation(len(identities))[:max(1, round(.15 * len(identities)))]}
            paths = [path for path, identity in files if identity not in reserved]
            validation = OfflineEpisodeDataset(config, self.prepared_inputs, self.topology, length,
                archive_paths=[path for path, identity in files if identity in reserved],
                manifest_path=validation_manifest)
        training = OfflineEpisodeDataset(config, self.prepared_inputs, self.topology, length,
                                         archive_paths=paths, manifest_path=train_manifest)
        return training, validation

    def _codec_batches(self, source, *, shuffle):
        if hasattr(source, 'iter_codec_batches'):
            for flow, next_flow in source.iter_codec_batches(self.config.codec.batch_size, self.rng, shuffle=shuffle):
                yield flow.to(self.device), next_flow.to(self.device)
            return
        indices = self.rng.permutation(len(source.manifest['entries'])) if shuffle else range(len(source.manifest['entries']))
        for index in indices:
            episode = source._episode(int(index))
            rows = episode['transitions']
            flow = torch.stack([rows[0].flow, *[row.next_flow for row in rows]])
            order = self.rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
            for offset in range(0, len(order), self.config.codec.batch_size):
                picks = order[offset:offset + self.config.codec.batch_size]
                yield flow[picks].to(self.device), flow[picks + 1].to(self.device)

    def _transition_codec_batches(self, source, *, shuffle):
        if hasattr(source, 'iter_transition_codec_batches'):
            for flow, next_flow in source.iter_transition_codec_batches(self.config.codec.batch_size, self.rng,
                                                                      shuffle=shuffle):
                yield flow.to(self.device), next_flow.to(self.device)
            return
        indices = self.rng.permutation(len(source.manifest['entries'])) if shuffle else range(len(source.manifest['entries']))
        for index in indices:
            episode = source._episode(int(index))
            replay = ReplayBuffer(source.sequence_length, max_episodes=1)
            replay.append_episode(episode['transitions'], total_travel_minutes=episode['total_travel_minutes'],
                                  total_vehicle_count=episode['total_vehicle_count'],
                                  arrival_times=episode.get('arrival_times'))
            rows = episode['transitions']
            full_flow = torch.stack([rows[0].flow, *[row.next_flow for row in rows]])
            count = max(1, len(rows) - source.sequence_length + 1)
            starts = np.unique(np.rint(np.linspace(0, count - 1, min(4, count))).astype(int))
            if shuffle:
                self.rng.shuffle(starts)
            scoped = prepare_transition_targets(replay._batch_from_picks([(0, int(start)) for start in starts]),
                                                 capacities=self.network.capacities)
            length = scoped.transition_target_flow.shape[1]
            anchors = np.unique(np.rint(np.linspace(0, length - 1, min(4, length))).astype(int))
            frames, following = [], []
            for ordinal, start in enumerate(starts):
                for anchor in anchors:
                    # Decoder speed uses an observed next boundary. Do not
                    # invent a following speed for an episode's final endpoint.
                    if int(start + anchor + 2) < len(full_flow):
                        frames.append(scoped.transition_target_flow[ordinal, anchor])
                        following.append(full_flow[int(start + anchor + 2)])
            if not frames:
                continue
            flow, next_flow = torch.stack(frames), torch.stack(following)
            for offset in range(0, len(flow), self.config.codec.batch_size):
                yield flow[offset:offset + self.config.codec.batch_size].to(self.device), next_flow[offset:offset + self.config.codec.batch_size].to(self.device)

    def _decoder_training_enabled(self):
        return self.config.codec.decoder_pretrain_enabled or self.config.codec.decoder_online_enabled

    def _decoder_revision(self):
        return self.metrics.get('decoder_pretrain_updates', 0.) + self.metrics.get('decoder_online_updates', 0.)

    def _set_dynamics_codec_gradients(self):
        """Keep the latent coordinates fixed while the supervised decoder learns."""
        self.world_model.flow_projector.requires_grad_(False)
        if self._decoder_training_enabled():
            for parameter in decoder_parameters(self.world_model):
                parameter.requires_grad_(True)

    def _codec_override_admitted(self):
        if not (self.config.codec.decoder_allow_unready and self._decoder_training_enabled()):
            return False
        source = self.codec_validation_data
        if source is None or not source.has_samples:
            raise ValueError('continuous decoder admission requires independent held-out demand trajectories')
        # Explicit deployment admission is independent of the quality result.
        # This small reduction never changes codec_ready or historical MAEs.
        finite = torch.stack([torch.isfinite(p).all() for p in self.world_model.flow_projector.parameters()]).all()
        if not bool(finite):
            raise ValueError('continuous decoder admission rejects nonfinite codec weights')
        return True

    def _codec_limits(self):
        limits = self.config.codec
        result = {'codec_count_mae_occupied': limits.occupied_vehicle_mae_max,
                'codec_count_mae_all': limits.vehicle_mae_max,
                'codec_count_mae_empty': limits.empty_vehicle_mae_max,
                'codec_total_relative_error': limits.total_relative_mae_max,
                'codec_speed_mae_mps': limits.speed_mae_mps_max}
        if self._decoder_training_enabled():
            result.update(codec_speed_mae_occupied_mps=limits.occupied_speed_mae_mps_max,
                          codec_speed_mae_slow_mps=limits.slow_speed_mae_mps_max)
        return result

    def codec_ready(self):
        source = self.codec_validation_data
        if not self.metrics.get('codec_ready') or source is None:
            return False
        if self.metrics.get('codec_validation_manifest_identity') != source.manifest['identity']:
            return False
        if self.metrics.get('codec_validation_updates') != self.metrics.get('codec_updates'):
            return False
        if (self._decoder_training_enabled()
                and self.metrics.get('codec_validation_decoder_updates', 0.)
                != self._decoder_revision()):
            return False
        for prefix in ('codec_validation_', 'codec_hybrid_validation_'):
            if self._decoder_training_enabled() and self.metrics.get(prefix + 'codec_slow_roads', 0) < self.config.codec.slow_validation_min_roads:
                return False
        return all(np.isfinite(self.metrics.get(prefix + key, np.nan)) and
                   0 <= self.metrics.get(prefix + key, np.inf) <= maximum
                   for prefix in ('codec_validation_', 'codec_hybrid_validation_')
                   for key, maximum in self._codec_limits().items()
                   if key != 'codec_speed_mae_slow_mps' or self.metrics.get(prefix + 'codec_slow_roads', 0) > 0)

    def validate_codec_ready(self):
        if self.codec_ready():
            if self.config.codec.decoder_online_enabled:
                self.metrics['decoder_online_authorized_manifest_identity'] = self.codec_validation_data.manifest['identity']
                self.metrics['decoder_online_authorized_codec_updates'] = self.metrics['codec_updates']
            return
        source = self.codec_validation_data
        admitted = (self.config.codec.decoder_online_enabled and source is not None
                    and source.has_samples
                    and self.metrics.get('decoder_online_authorized_manifest_identity') == source.manifest['identity']
                    and self.metrics.get('decoder_online_authorized_codec_updates') == self.metrics.get('codec_updates'))
        if not admitted and not self._codec_override_admitted():
            raise ValueError('codec quality gate has not passed; FM/online training is blocked')

    def _validate_codec_pretraining_ready(self):
        """A held-out foundation or explicit continuous admission permits decoder updates."""
        if self._codec_override_admitted():
            return
        if not self.config.codec.decoder_pretrain_enabled:
            self.validate_codec_ready()
            return
        if self.codec_ready():
            self.metrics['decoder_pretrain_authorized_manifest_identity'] = self.codec_validation_data.manifest['identity']
            self.metrics['decoder_pretrain_authorized_codec_updates'] = self.metrics['codec_updates']
            return
        source = self.codec_validation_data
        if (source is None or self.metrics.get('decoder_pretrain_authorized_manifest_identity') != source.manifest['identity']
                or self.metrics.get('decoder_pretrain_authorized_codec_updates') != self.metrics.get('codec_updates')):
            raise ValueError('decoder pretraining requires a passed initial held-out codec proof')

    def validate_codec_pretraining_ready(self):
        self._validate_codec_pretraining_ready()

    def _validate_codec(self):
        if self.codec_validation_data is None or not self.codec_validation_data.has_samples:
            raise ValueError('codec quality gate requires independent held-out demand trajectories')
        was_training = self.world_model.training
        self.world_model.eval()
        groups = {}
        try:
            for name, batches in (('raw', self._codec_batches(self.codec_validation_data, shuffle=False)),
                                  ('hybrid', self._transition_codec_batches(self.codec_validation_data, shuffle=False))):
                totals, counts = {}, {}
                for flow, next_flow in batches:
                    result = codec_metrics(self.world_model, flow, next_flow, self.network,
                                           self.config.codec) if self._decoder_training_enabled() else codec_metrics(self.world_model, flow, next_flow, self.network)
                    cells = flow.shape[0] * self.edge_count
                    occupied = int(result['codec_count_occupied_roads'])
                    empty = int(result['codec_count_empty_roads'])
                    weights = [('codec_count_mae_all', cells), ('codec_count_mae_occupied', occupied),
                                        ('codec_count_mae_empty', empty), ('codec_total_relative_error', flow.shape[0]),
                                        ('codec_speed_mae_mps', cells)]
                    if self._decoder_training_enabled():
                        weights.extend([('codec_speed_mae_occupied_mps', occupied),
                                        ('codec_speed_mae_slow_mps', int(result['codec_slow_roads']))])
                    for key, weight in weights:
                        totals[key] = totals.get(key, 0.) + float(result[key]) * weight
                        counts[key] = counts.get(key, 0) + weight
                if not counts or any(count == 0 for key, count in counts.items() if key != 'codec_speed_mae_slow_mps'):
                    raise ValueError('codec validation lacks required occupied/empty observations')
                groups[name] = {key: totals[key] / counts[key] if counts[key] else None for key in totals}
                if self._decoder_training_enabled():
                    groups[name]['codec_slow_roads'] = counts.get('codec_speed_mae_slow_mps', 0)
        finally:
            self.world_model.train(was_training)
            self.codec_validation_data.release_cache()
        ready = all(np.isfinite(metrics[key]) and metrics[key] <= maximum
                    for metrics in groups.values() for key, maximum in self._codec_limits().items()
                    if key != 'codec_speed_mae_slow_mps' or metrics.get('codec_slow_roads', 0) > 0)
        if self._decoder_training_enabled():
            ready = ready and all(metrics['codec_slow_roads'] >= self.config.codec.slow_validation_min_roads
                                  for metrics in groups.values())
        for name, metrics in groups.items():
            prefix = 'codec_validation_' if name == 'raw' else 'codec_hybrid_validation_'
            self.metrics.update({prefix + key: value for key, value in metrics.items()})
        self.metrics['codec_ready'] = float(ready)
        self.metrics['codec_validation_manifest_identity'] = self.codec_validation_data.manifest['identity']
        self.metrics['codec_validation_updates'] = self.metrics['codec_updates']
        if self._decoder_training_enabled():
            self.metrics['codec_validation_decoder_updates'] = self._decoder_revision()
        return {**groups['raw'], **{'hybrid_' + key: value for key, value in groups['hybrid'].items()},
                'codec_ready': ready}

    def initialize_codec_epoch(self, epoch_index):
        if self.codec_ready():
            self.metrics['codec_epochs_completed'] = max(self.metrics['codec_epochs_completed'], epoch_index + 1)
            return dict(epoch=epoch_index + 1, updates=self.metrics['codec_updates'], codec_ready=True, skipped=True)
        if epoch_index != int(self.metrics['codec_epochs_completed']):
            raise ValueError('codec must resume at the next epoch boundary')
        self.validate_pretraining_data()
        if self.codec_validation_data is None:
            raise ValueError('codec requires an independent validation source')
        self.world_model.flow_projector.requires_grad_(True)
        self.world_model.flow_projector.train()
        last = {}
        raw_batches = self._codec_batches(self.offline_data, shuffle=True)
        hybrid_batches = iter(self._transition_codec_batches(self.offline_data, shuffle=True))
        def mixed_batches():
            for ordinal, raw in enumerate(raw_batches):
                yield raw
                if ordinal % 3 == 0:
                    hybrid = next(hybrid_batches, None)
                    if hybrid is not None:
                        yield hybrid
        for flow, next_flow in mixed_batches():
            if int(self.metrics['codec_updates']) >= self.config.codec.max_updates:
                break
            result = codec_train_step(self.world_model, flow, next_flow, self.network,
                                      self.config.method, self.optimizers['codec'], self.config.codec)
            self.metrics['codec_updates'] += 1
            last = {key: float(value) for key, value in result.items()}
            if int(self.metrics['codec_updates']) % self.config.codec.validation_every == 0:
                last.update(self._validate_codec())
                if self.codec_ready():
                    break
        last.update(self._validate_codec())
        self.offline_data.release_cache()
        self.metrics['codec_epochs_completed'] += 1
        self.metrics.update(last)
        return dict(epoch=epoch_index + 1, updates=self.metrics['codec_updates'], **last)

    def _context(self, states: list[dict[str, Any]] | tuple, timestamp: float, *,
                 mode: str = 'online', started_count: int | None = None,
                 cumulative_vehicle_seconds: float = 0.) -> np.ndarray:
        active = sum(bool(state['departed']) and not bool(state['arrived']) for state in states)
        started = (sum(bool(state['departed']) for state in states)
                   if started_count is None else started_count)
        return causal_context(timestamp, self._episode_end_time(mode), active,
                              started, cumulative_vehicle_seconds)

    def _policy_features(self, flow: np.ndarray) -> np.ndarray:
        # Volume and speed are decoded fields in both physical and imagined windows.
        return np.asarray(flow[:, [2, 1]], dtype=np.float32)

    def _prior_observation_kwargs(self):
        mode = getattr(self.config.method, 'sp_prior_cost_mode', 'free_flow')
        return (dict(prior_cost_mode=mode, prior_speed_floor_mps=self.config.method.sp_prior_dynamic_speed_floor_mps,
                     prior_speed_cap_mps=self.config.simulation.vehicle_max_speed_mps)
                if mode == 'dynamic' else {})

    def _observations(self, states, flow, context, timestamp):
        policy_features = flow if isinstance(flow, PreparedActorWindow) else self._policy_features(flow)
        kwargs = dict(policy_edge_features=policy_features, vehicle_states=states,
                      candidates=self.candidates, episode_progress=float(context[0]),
                      active_vehicle_fraction=float(context[1]))
        return build_actor_observations(**kwargs, **self._prior_observation_kwargs())

    def _select(self, states, flow, timestamp, *, context, mode, actor=None, observations=None):
        if observations is None:
            states = sorted(states, key=lambda s: str(s['vehicle_id']))
            observations = self._observations(states, flow, context, timestamp)
        actor = self.actor if actor is None else actor
        device = next(actor.parameters()).device
        prior_costs = getattr(observations, 'prior_costs_seconds', None)
        prior_kwargs = ({'prior_costs_seconds': torch.as_tensor(prior_costs, dtype=torch.float32, device=device)}
                        if prior_costs is not None else {})
        with torch.no_grad():
            temperature = getattr(self, 'evaluation_temperature', None) if mode == 'evaluation' else 1.
            selection = actor.sample_action(torch.as_tensor(observations.features, dtype=torch.float32, device=device),
                torch.as_tensor(observations.mask, device=device),
                deterministic=mode == 'evaluation' and temperature is None,
                temperature=1. if temperature is None else temperature, **prior_kwargs)
        indices = selection.candidate_index.cpu().tolist()
        return selected_decisions(states, observations, indices, timestamp)

    def _episode_step_limit(self, mode):
        simulation = self.config.simulation
        if mode == 'evaluation':
            seconds = simulation.evaluation_horizon_seconds
        else:
            seconds = (simulation.world_model_intervals_per_episode * WORLD_MODEL_INTERVAL_SECONDS
                       + simulation.training_completion_buffer_seconds)
        return math.ceil(seconds / WORLD_MODEL_INTERVAL_SECONDS)

    def _episode_end_time(self, mode: str = 'online') -> float:
        # Real collection uses complete 30-second windows, including the drain buffer.
        return float(self._episode_step_limit(mode) * WORLD_MODEL_INTERVAL_SECONDS)

    def _run_episode(self, *, mode: str, episode_index: int) -> CollectedEpisode:
        if mode == 'evaluation':
            with isolated_evaluation_rng(self, episode_index):
                return self._collect_episode(mode=mode, episode_index=episode_index)
        return self._collect_episode(mode=mode, episode_index=episode_index)

    def _collect_episode(self, *, mode: str, episode_index: int) -> CollectedEpisode:
        if mode not in {'online', 'evaluation'}:
            raise ValueError(f'unsupported collection mode {mode!r}')
        evaluation = mode == 'evaluation'
        demand_seed = self.config.method.evaluation_demand_seed if evaluation else self.config.run.seed
        demand_index = episode_index if evaluation else self._next_environment_episode
        episode_topology = replace(self.topology, vehicles=self.demand.sample(
            seed=demand_seed, episode_index=demand_index))
        if evaluation:
            self._record_evaluation_demand(episode_topology.vehicles, episode_index)
        total_vehicle_count = len(episode_topology.vehicles)
        denominator = max(total_vehicle_count, 1)
        # A policy is fixed throughout collection. Keep small decisions beside
        # SUMO's CPU state, without moving the trainable actor/optimizer itself.
        collection_actor = (copy.deepcopy(self.actor).cpu().requires_grad_(False)
                            if self.device.type == 'cuda' or evaluation else self.actor)
        if evaluation:
            # Evaluation guidance is explicit and does not change the training schedule.
            set_shortest_path_prior(collection_actor, self.config.method.sp_prior_evaluation_weight,
                                    self.config.method.sp_prior_temperature)
        policy_protocol = (evaluation_protocol(getattr(self, 'evaluation_temperature', None),
            prior_weight=collection_actor.prior_weight, demand_seed=demand_seed) if evaluation else None)
        session_config = replace(self.config, run=replace(self.config.run, seed=demand_seed)) if evaluation else self.config
        session_kwargs = dict(config=session_config, prepared_inputs=self.prepared_inputs,
            topology=episode_topology, episode_index=demand_index)
        session = (self._session_factory(**session_kwargs) if self._session_factory is not None
                   else resolve_session_factory('event', 1., self.config.simulation.backend)(**session_kwargs))
        started = time.monotonic()
        transitions = []
        events = 0
        policy_decision_count = 0
        exposure = 0.
        diversion_cost = DiversionCost(self.candidates) if evaluation else None
        with SumoEpisodeSampler(session=session, candidates=self.candidates,
                deadline_seconds=self._episode_end_time(mode), total_vehicle_count=total_vehicle_count,
                context_mode='observation', sort_pending=True, pad_empty=False, advance_empty=True,
                max_windows=self._episode_step_limit(mode),
                wall_timeout_seconds=self.config.simulation.episode_wall_timeout_seconds,
                timeout_label=mode,
                observation_builder=partial(build_actor_observations, **self._prior_observation_kwargs())) as sampler:
            state = sampler.state
            from dataclasses import asdict
            archive = EpisodeArchive(dict(city=self.config.inputs.network_name,
                method='urban_dreamer', policy=mode, seed=demand_seed, episode=episode_index,
                phase='evaluation' if evaluation else 'train', collection_phase=mode,
                sp_prior_weight=collection_actor.prior_weight,
                sp_prior_temperature=collection_actor.prior_temperature_seconds,
                sp_prior_cost_mode=getattr(self.config.method, 'sp_prior_cost_mode', 'free_flow'),
                sp_prior_dynamic_speed_floor_mps=getattr(self.config.method, 'sp_prior_dynamic_speed_floor_mps', .5),
                sp_prior_schedule_episode=int(self.metrics['online_episodes_completed']) + 1,
                evaluation_policy=policy_protocol['archive_policy'] if evaluation else None,
                evaluation_temperature=policy_protocol['temperature'] if evaluation else None,
                **(dict(diversion_cost_definition=DIVERSION_COST_DEFINITION,
                        all_demand_censored_definition=ALL_DEMAND_CENSORED_DEFINITION) if evaluation else {}),
                evaluation_action_seed=(policy_protocol['action_seed_base'] + episode_index) if evaluation else None,
                input_hashes=self.prepared_inputs.input_hashes, edge_ids=list(self.network.edge_ids),
                total_vehicle_count=total_vehicle_count, session_protocol='event',
                **({'routing_mode': 'current_to_dest'}
                   if getattr(self.config.simulation, 'routing_mode', 'next_hop') == 'current_to_dest' else {}),
                step_seconds=1., backend=self.config.simulation.backend,
                config=session_config.resolved_snapshot(),
                realized_demand=[asdict(vehicle) for vehicle in episode_topology.vehicles],
                demand_sha256=demand_identity(episode_topology.vehicles),
                demand=dict(law='independent_poisson_per_od',
                            duration_seconds=self.demand.duration_seconds,
                            expected_vehicle_count=self.demand.expected_vehicle_count,
                            od_rates=[dict(origin_index=o, destination_index=d,
                                           rate_vehicles_per_second=rate)
                                      for o, d, rate in self.demand.od_rates])))
            archive.start(state)
            destinations = {str(v['vehicle_id']): int(v['destination_edge']) for v in state['vehicles']}
            start_inputs = {}
            def before_window(index, boundary, start_context):
                vehicles = copy.deepcopy(tuple(vehicle for vehicle in boundary['vehicles']
                    if vehicle['departed'] and not vehicle['arrived']))
                timestamp = float(boundary['time'])
                start_inputs.update(flow=np.asarray(boundary['flow'], dtype=np.float32),
                    vehicles=vehicles, time=timestamp,
                    context=np.array(start_context, dtype=np.float32, copy=True),
                    started_count=sampler.accounting.started_count,
                    cumulative_vehicle_seconds=sampler.accounting.cumulative_vehicle_seconds)
            def select_observed(pending, observations, frozen, now, index, observed_context):
                return self._select(pending, frozen, now, context=observed_context,
                    mode=mode, actor=collection_actor, observations=observations)
            for sampled in sampler.windows(select_observed, before_window=before_window):
                window, step = sampled.result, sampled.index
                conditioning = getattr(window, 'conditioning_decisions', None)
                if conditioning is None:
                    conditioning = window.decisions
                window_events = tuple(ActionEvent(str(decision.vehicle_id), events + index,
                    float(decision.decision_time), int(decision.current_edge), int(decision.next_edge),
                    destinations[str(decision.vehicle_id)]) for index, decision in enumerate(conditioning))
                archive.append(window)
                if diversion_cost is not None:
                    diversion_cost.add(conditioning, destinations)
                state = window.next_state
                exposure += window.active_vehicle_seconds
                events += len(conditioning)
                policy_decision_count += len(window.decisions)
                transitions.append(MacroTransition(flow=start_inputs['flow'], action=window.action_counts / denominator,
                    departure=window.departure_counts / denominator, context=start_inputs['context'], next_flow=state['flow'],
                    active_vehicle_fraction=float(np.clip(window.active_vehicle_seconds / (denominator * 30.), 0., 1.)),
                    vehicles=start_inputs['vehicles'], time=start_inputs['time'],
                    terminal=bool(session.done or window.end_time >= self._episode_end_time(mode) - 1e-9),
                    episode=episode_index, step_index=step, action_events=window_events,
                    started_count=start_inputs['started_count'],
                    cumulative_vehicle_seconds=start_inputs['cumulative_vehicle_seconds'],
                    next_started_count=sampler.accounting.started_count,
                    next_cumulative_vehicle_seconds=sampler.accounting.cumulative_vehicle_seconds,
                    global_reward=-REWARD_SCALE * (sampler.accounting.mean_seconds
                        - observed_mean_seconds(start_inputs['cumulative_vehicle_seconds'], start_inputs['started_count']))))
            complete = session.done
            if not transitions:
                raise RuntimeError('episode produced no macro transitions')
            transitions[-1] = replace(transitions[-1], next_vehicles=tuple(
                vehicle for vehicle in state['vehicles'] if vehicle['departed'] and not vehicle['arrived']))
            observed_arrivals = {str(vehicle['vehicle_id']): float(vehicle.get('arrival_time',
                vehicle['departure_time'] + session.travel_times[str(vehicle['vehicle_id'])]))
                                for vehicle in state['vehicles'] if vehicle['arrived']}
            durations = session.travel_times
            if complete and len(durations) != total_vehicle_count:
                raise RuntimeError('episode completed without all vehicle travel times')
            # Incomplete online trajectories report actual observed exposure, never fabricated arrivals.
            travel_minutes = sampler.accounting.mean_seconds / 60.
            unfinished = total_vehicle_count - len(durations)
            penalty = 0.
            metrics = {f'{mode}_sp_prior_weight': collection_actor.prior_weight,
                f'{mode}_sp_prior_temperature': collection_actor.prior_temperature_seconds,
                f'{mode}_sp_prior_schedule_episode': float(self.metrics['online_episodes_completed']) + 1.,
                f'{mode}_nonarrival_count': float(unfinished),
                f'{mode}_nonarrival_penalty_sum': penalty,
                f'{mode}_vehicle_timeout_penalty_sum': penalty,
                f'{mode}_global_reward_sum': -travel_minutes,
                f'{mode}_selection_score': -travel_minutes,
                f'{mode}_vehicle_count': float(total_vehicle_count),
                f'{mode}_vehicle_minutes': exposure / 60.,
                f'{mode}_started_vehicle_count': float(sampler.accounting.started_count),
                f'{mode}_cumulative_vehicle_seconds': exposure,
                f'{mode}_demand_seed': float(demand_seed),
                f'{mode}_demand_index': float(demand_index),
                f'{mode}_demand_duration_seconds': float(self.config.simulation.demand_duration_seconds),
                f'{mode}_time_limit_seconds': self._episode_end_time(mode),
                f'{mode}_completion_rate': len(durations) / total_vehicle_count if total_vehicle_count else 1.,
                f'{mode}_steps': float(len(transitions)), f'{mode}_events': float(events),
                f'{mode}_wall_seconds': time.monotonic() - started,
                f'{mode}_microstep_seconds': float(session.microstep_seconds),
                f'{mode}_microsteps': float(session.metrics["microsteps"]),
                f'{mode}_t_all': travel_minutes}
            if evaluation:
                evaluation_metrics = diversion_cost.metrics(sampler.accounting.started_count)
                evaluation_metrics.update(all_demand_censored_metrics(
                    episode_topology.vehicles,
                    [dict(vehicle, arrival_time=observed_arrivals[str(vehicle['vehicle_id'])])
                     if vehicle['arrived'] else vehicle for vehicle in state['vehicles']],
                    min(float(state['time']), self._episode_end_time(mode))))
                metrics.update({f'evaluation_{key}': value for key, value in evaluation_metrics.items()})
            if getattr(self.config.simulation, 'routing_mode', 'next_hop') == 'current_to_dest':
                metrics[f'{mode}_policy_decisions'] = float(policy_decision_count)
            archive.save(self.config.run.output_dir / 'episodes' / mode / f'episode_{episode_index:06d}.npz',
                metrics=metrics, travel_times=durations, completed=complete,
                deadline_seconds=self._episode_end_time(mode))
            self._next_environment_episode += 1
            return CollectedEpisode(tuple(transitions), travel_minutes, metrics, total_vehicle_count, observed_arrivals)

    def _record_evaluation_demand(self, vehicles, episode_index):
        from dataclasses import asdict
        directory = self.config.run.output_dir / 'evaluation_demands'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'episode_{episode_index:04d}.json'
        payload = {'schema_version': 1, 'seed': self.config.method.evaluation_demand_seed,
                   'episode_index': episode_index, 'demand_hash': demand_identity(vehicles),
                   'duration_seconds': self.config.simulation.demand_duration_seconds,
                   'od_rates': getattr(self.demand, 'od_rates', None),
                   'vehicles': [asdict(vehicle) for vehicle in vehicles]}
        if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise ValueError('evaluation demand manifest differs from the fixed evaluation set')
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')

    def validate_pretraining_data(self):
        if not self.offline_data.has_samples:
            raise ValueError('offline pretraining requires compatible training NPZ archives in [data].archive_roots; no eligible sequences found')

    def _world_loss(self, batch):
        if getattr(self, '_decoder_pretraining_active', False):
            self._validate_codec_pretraining_ready()
        else:
            self.validate_codec_ready()
        method = self.config.method
        recursive = method.fm_training_mode == 'recursive_3step'
        if recursive:
            depth = min(3, batch.action.shape[1])
            scoped = prepare_cohort_batch(batch, capacities=self.network.capacities)
            inputs = scoped.transition_flow[:, :depth].to(self.device)
            targets = scoped.transition_flow[:, 1:depth+1].to(self.device)
            action = scoped.transition_action[:, :depth].to(self.device)
            events = scoped.transition_action_events.to(self.device)
            fraction_targets = scoped.transition_active_vehicle_fraction[:, :depth].to(self.device)
        else:
            scoped = prepare_transition_targets(batch, capacities=self.network.capacities)
            depth = (1 if method.fm_training_mode == 'single_step' else
                     min(batch.action.shape[1], method.supervision_horizon,
                         1 + int(self.metrics['world_model_updates']) // method.supervision_increment_updates))
            inputs = batch.flow[:, :depth].to(self.device)
            targets = scoped.transition_target_flow[:, :depth].to(self.device)
            action = batch.action[:, :depth].to(self.device)
            events = batch.action_events.to(self.device)
            fraction_targets = batch.active_vehicle_fraction[:, :depth].to(self.device)
        batch_size = inputs.shape[0]
        with torch.no_grad():
            if recursive:
                # Adjacent input/target frames are the same cohort observations.
                # Encode H+1 distinct frames once instead of two overlapping Hs.
                frames = torch.cat([inputs[:, :1], targets], dim=1)
                encoded = self.world_model.encode(frames.reshape(-1, self.edge_count, 8)).reshape(
                    batch_size, depth + 1, self.edge_count, FLOW_LATENT_DIM)
                z_inputs, z_targets = encoded[:, :-1], encoded[:, 1:]
            else:
                z_inputs = self.world_model.encode(inputs.reshape(-1, self.edge_count, 8)).reshape(
                    batch_size, depth, self.edge_count, FLOW_LATENT_DIM)
                z_targets = self.world_model.encode(targets.reshape(-1, self.edge_count, 8)).reshape_as(z_inputs)
        precision = getattr(self, '_world_training_precision',
                            getattr(self.config.resources, 'world_model_pretrain_precision', 'fp32'))
        if precision == 'bf16' and self.device.type != 'cuda':
            raise ValueError('BF16 pretraining requires CUDA; use fp32 for CPU training')
        amp = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if precision == 'bf16' else nullcontext
        sampled_readout = getattr(method, 'exposure_readout_supervision', 'observed_endpoint') == 'sampled_endpoint'
        with amp():
            if recursive:
                loss, fm_metrics, generated = recursive_fm_loss(self.world_model.dynamics,
                    z_sequence=torch.cat([z_inputs[:, :1], z_targets], dim=1),
                    action=action, action_events=events,
                    steps=method.fm_steps, solver=method.fm_solver,
                    picard_iterations=method.picard_iterations, prior_weight=method.prior_loss_weight,
                    predict_endpoints=sampled_readout, checkpoint_interval=10)
                # Readout updates are separate from the FM graph. Recursion itself
                # retains gradients through every generated physical transition.
                predicted = generated.detach() if sampled_readout else z_targets
            else:
                loss, fm_metrics, predicted = independent_transition_fm_loss(self.world_model.dynamics,
                    z_inputs=z_inputs, z_targets=z_targets, action=action, action_events=events,
                    steps=method.fm_steps, solver=method.fm_solver,
                    picard_iterations=method.picard_iterations, prior_weight=method.prior_loss_weight,
                    predict_endpoints=sampled_readout)
        self.metrics.update({key + '_last': float(value) for key, value in fm_metrics.items()})
        self.metrics['world_model_supervision_depth_last'] = float(depth)
        # Single-window labels retain the full observed prior. Recursive labels
        # instead follow the fixed seed cohort, matching imagined vehicle membership.
        # Codec and physical labels remain FP32 even with AMP dynamics.
        with torch.no_grad():
            raw_fraction = self.world_model.decode(predicted.float())[..., 2].mean(dim=-1)
        readout_previous = (torch.cat([z_inputs[:, :1], predicted[:, :-1]], dim=1)
                            if recursive and sampled_readout else z_inputs)
        with amp():
            fraction = self.world_model.active_vehicle_fraction_readout(predicted,
                z_prev=readout_previous, raw_decoder_fraction=raw_fraction, transition_seconds=30.)
        readout_loss = F.mse_loss(fraction.float(), fraction_targets.float())
        return loss, readout_loss

    def _decoder_pretrain_loss(self, batch):
        """Only observed latents supervise the decoder; no FM graph or future gradients."""
        current = batch.flow[:, :-1].to(self.device)
        following = batch.flow[:, 1:].to(self.device)
        frames, next_frames = [current], [following]
        scoped = prepare_transition_targets(batch, capacities=self.network.capacities)
        hybrid = scoped.transition_target_flow.to(self.device)
        if hybrid.shape[1] > 1:
            frames.append(hybrid[:, :-1])
            next_frames.append(batch.flow[:, 2:].to(self.device))
        codec_following = getattr(batch, 'decoder_following_flow', None)
        codec_valid = getattr(batch, 'decoder_following_valid', None)
        if codec_following is not None and codec_valid is not None:
            valid = codec_valid.to(self.device).bool()
            if valid.shape != hybrid.shape[:2] or codec_following.shape != hybrid.shape:
                raise ValueError('decoder following frames must match the transition window layout')
            # Earlier windows are already paired above; the extra frame adds
            # only the final endpoint, including single-window FM batches.
            valid = valid[:, -1]
            if bool(valid.any()):
                frames.append(batch.flow.to(self.device)[:, -1][valid].unsqueeze(1))
                next_frames.append(codec_following.to(self.device)[:, -1][valid].unsqueeze(1))
                frames.append(hybrid[:, -1][valid].unsqueeze(1))
                next_frames.append(codec_following.to(self.device)[:, -1][valid].unsqueeze(1))
        flow = torch.cat([value.reshape(-1, self.edge_count, 8) for value in frames])
        next_flow = torch.cat([value.reshape(-1, self.edge_count, 8) for value in next_frames])
        return codec_objective(self.world_model, flow, next_flow, self.network,
                               self.config.method, codec=self.config.codec, decoder_only=True)

    def _update_world(self, replay, *, initial, batches=None):
        update_started = time.perf_counter()
        self._decoder_pretraining_active = initial and self.config.codec.decoder_pretrain_enabled
        if self._decoder_pretraining_active:
            self._validate_codec_pretraining_ready()
        else:
            self.validate_codec_ready()
        self._set_dynamics_codec_gradients()
        if not replay.has_samples:
            return {'world_model_loss': 0., 'active_fraction_loss': 0., 'world_model_update_skipped': 1.}
        self.world_model.train()
        self._world_training_precision = (self.config.resources.world_model_pretrain_precision if initial else 'fp32')
        if self._world_training_precision == 'bf16' and (self.device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
            raise ValueError('BF16 pretraining requires a CUDA device with BF16 support')
        if initial:
            resources = self.config.resources
            self.world_model.dynamics.set_pretraining_execution_profile(
                space_chunk=resources.world_model_pretrain_space_chunk,
                time_chunk=resources.world_model_pretrain_time_chunk,
                mlp_chunk=resources.world_model_pretrain_mlp_chunk,
                activation_checkpointing=resources.world_model_pretrain_activation_checkpointing)
        else:
            self.world_model.set_online_chunk_profile()
        optimizer = self.optimizers['world_model_initial' if initial else 'world_model_online']
        readout_optimizer = self.optimizers['active_fraction']
        optimizer.zero_grad(set_to_none=True)
        readout_optimizer.zero_grad(set_to_none=True)
        decoder_phase = 'decoder_pretrain' if initial else 'decoder_online'
        decoder_enabled = self.config.codec.decoder_pretrain_enabled if initial else self.config.codec.decoder_online_enabled
        decoder_optimizer = self.optimizers.get(decoder_phase) if decoder_enabled else None
        if decoder_optimizer is not None:
            decoder_optimizer.zero_grad(set_to_none=True)
            for parameter in decoder_parameters(self.world_model):
                parameter.requires_grad_(True)
        if batches is None:
            size = (self.config.resources.world_model_initialization_batch_size if initial
                    else self.config.resources.online_world_model_batch_size)
            batches = [replay.sample_batch(batch_size=size, rng=self.rng)]
        sample_count = sum(batch.flow.shape[0] for batch in batches)
        metrics = {'world_model_loss': 0., 'active_fraction_loss': 0.}
        for batch in batches:
            weight = batch.flow.shape[0] / sample_count
            loss, readout_loss = self._world_loss(batch)
            (loss * weight).backward()
            (readout_loss * weight).backward()
            metrics['world_model_loss'] += float(loss.detach()) * weight
            metrics['active_fraction_loss'] += float(readout_loss.detach()) * weight
            if decoder_optimizer is not None:
                decoder_loss, decoder_metrics = self._decoder_pretrain_loss(batch)
                if not bool(torch.isfinite(decoder_loss)):
                    raise FloatingPointError('nonfinite decoder pretraining loss')
                (decoder_loss * weight).backward()
                for key, value in decoder_metrics.items():
                    metrics[decoder_phase + '_' + key] = metrics.get(decoder_phase + '_' + key, 0.) + float(value) * weight
        reject_nonfinite = (initial and self.config.resources.world_model_pretrain_compile_blocks) or decoder_optimizer is not None
        torch.nn.utils.clip_grad_norm_(self._world_parameters, GRAD_CLIP,
                                       error_if_nonfinite=reject_nonfinite)
        torch.nn.utils.clip_grad_norm_(self.world_model.active_vehicle_fraction_readout.parameters(),
                                       GRAD_CLIP, error_if_nonfinite=reject_nonfinite)
        if decoder_optimizer is not None:
            norm = torch.nn.utils.clip_grad_norm_(decoder_parameters(self.world_model), GRAD_CLIP, error_if_nonfinite=True)
            metrics[decoder_phase + '_gradient_norm'] = float(norm)
        optimizer.step()
        readout_optimizer.step()
        if decoder_optimizer is not None:
            decoder_optimizer.step()
            self.metrics[decoder_phase + '_updates'] = self.metrics.get(decoder_phase + '_updates', 0.) + 1
            metrics[decoder_phase + '_updated'] = 1.
            self.metrics['codec_ready'] = 0.
        self.world_model.active_vehicle_fraction_readout.calibration_updates.add_(1)
        self.metrics['world_model_updates'] += 1
        self.metrics.update({key + '_last': value for key, value in metrics.items()})
        self._last_world_update_seconds = time.perf_counter() - update_started
        return metrics

    def initialize_world_model_epoch(self, epoch_index):
        self._validate_codec_pretraining_ready()
        if not 0 <= epoch_index < self.config.run.world_model_initialization_epochs:
            raise IndexError('world-model initialization exceeds configured epoch budget')
        if epoch_index != int(self.metrics['world_model_initialization_epochs_completed']):
            raise ValueError('world-model initialization must resume at the next epoch boundary')
        self.validate_pretraining_data()
        batches = self.offline_data.iter_epoch(self.config.resources.world_model_initialization_batch_size, self.rng)
        accumulation = self.config.resources.world_model_gradient_accumulation_steps
        while True:
            data_started = time.perf_counter()
            group = list(islice(batches, accumulation))
            self._last_pretrain_data_seconds = time.perf_counter() - data_started
            if not group:
                break
            self._last_pretrain_group_samples = sum(batch.flow.shape[0] for batch in group)
            metrics = self._update_world(self.offline_data, initial=True, batches=group)
            count = sum(batch.flow.shape[0] for batch in group)
            self.metrics['world_model_initialization_batches_completed'] += len(group)
            self.metrics['world_model_initialization_samples_seen'] += count
            self.metrics['world_model_initialization_updates'] += 1
            self.metrics['world_model_epoch_batches'] += len(group)
            self.metrics['world_model_epoch_samples'] += count
            self.metrics['world_model_epoch_updates'] += 1
            self.metrics['world_model_epoch_loss_sum'] += metrics['world_model_loss'] * count
            self.metrics['world_model_epoch_readout_loss_sum'] += metrics['active_fraction_loss'] * count
            progress_hook = getattr(self, '_pretrain_progress_hook', None)
            if progress_hook is not None:
                progress_hook(epoch_index)
        self.offline_data.release_cache()
        self.metrics['world_model_initialization_epochs_completed'] += 1
        if self.config.codec.decoder_pretrain_enabled:
            self._validate_codec()
        samples = self.metrics['world_model_epoch_samples']
        result = {'epoch': epoch_index + 1, 'microbatches': int(self.metrics['world_model_epoch_batches']),
                  'samples': int(samples), 'optimizer_updates': int(self.metrics['world_model_epoch_updates']),
                  'world_model_loss': self.metrics['world_model_epoch_loss_sum'] / max(samples, 1),
                  'active_fraction_loss': self.metrics['world_model_epoch_readout_loss_sum'] / max(samples, 1)}
        for key in ('world_model_epoch_samples', 'world_model_epoch_batches', 'world_model_epoch_updates',
                    'world_model_epoch_loss_sum', 'world_model_epoch_readout_loss_sum'):
            self.metrics[key] = 0.
        return result

    def train_online_episode(self, episode_index):
        self.validate_codec_ready()
        self._set_online_prior()
        collected = self._run_episode(mode='online', episode_index=episode_index)
        self.online_replay.append_episode(collected.transitions, total_travel_minutes=collected.total_travel_minutes,
                                         total_vehicle_count=collected.total_vehicle_count,
                                         arrival_times=collected.arrival_times)
        metrics = _mean_metrics([self._update_world(self.online_replay, initial=False)
                                for _ in range(ONLINE_WORLD_MODEL_UPDATES_PER_EPISODE)])
        metrics.update(self._train_episode_imagination())
        metrics.update(collected.metrics)
        self.metrics['online_episodes_completed'] += 1
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(metrics)
        return {'complete': True, **metrics}

    def _train_episode_imagination(self):
        budget = resolved_imagination_budget(self.config)
        requested = int(budget['imagination_passes_per_episode'])
        seconds = float(budget['imagination_time_budget_seconds'])
        rows = []
        start = time.perf_counter()
        elapsed = 0.
        estimate = float(self.metrics.get('imagination_pass_seconds_ema', 0.))
        stop_reason = 'pass_limit'
        for _ in range(requested):
            # Always finish at least one pass. Never truncate trajectories or
            # separate actor, V and Q updates to meet a wall-clock deadline.
            if rows and seconds > 0 and elapsed + estimate > seconds:
                stop_reason = ('budget_exceeded' if elapsed >= seconds
                               else 'insufficient_time_for_complete_pass')
                break
            pass_start = time.perf_counter()
            rows.append(self._train_imagination_pass())
            pass_seconds = time.perf_counter() - pass_start
            previous = float(self.metrics.get('imagination_pass_seconds_ema', pass_seconds))
            ema = .5 * previous + .5 * pass_seconds
            self.metrics['imagination_pass_seconds_ema'] = ema
            # A slow latest pass must not be hidden by a faster historical EMA.
            estimate = max(ema, pass_seconds)
            elapsed = time.perf_counter() - start
        if seconds > 0 and elapsed > seconds:
            stop_reason = 'budget_exceeded'
        result = _mean_metrics(rows)
        result.update(imagination_passes_requested=int(requested),
                      imagination_passes_completed=len(rows),
                      imagination_seconds=elapsed,
                      imagination_time_budget_seconds=seconds,
                      imagination_pass_seconds_ema=self.metrics['imagination_pass_seconds_ema'],
                      imagination_budget_stop_reason=stop_reason)
        return result

    def evaluate_episode(self, episode_index):
        collected = self._run_episode(mode='evaluation', episode_index=episode_index)
        self.metrics['evaluation_episodes_completed'] += 1
        self.metrics['evaluation_t_all_sum'] += collected.total_travel_minutes
        self.metrics['evaluation_t_all_mean'] = (self.metrics['evaluation_t_all_sum']
                                               / self.metrics['evaluation_episodes_completed'])
        self.metrics['evaluation_vehicle_minutes_sum'] += collected.total_travel_minutes * collected.total_vehicle_count
        self.metrics['evaluation_vehicle_count_sum'] += collected.total_vehicle_count
        self.metrics['evaluation_t_all_vehicle_weighted'] = (
            self.metrics['evaluation_vehicle_minutes_sum'] / max(self.metrics['evaluation_vehicle_count_sum'], 1))
        self.metrics['t_all_last'] = collected.total_travel_minutes
        self.metrics.update(collected.metrics)
        return {'complete': True, **collected.metrics}

    def _predict_with_disagreement(self, latent, action, action_events):
        method = self.config.method
        samples, variance, _ = self.world_model.latent_disagreement(
            latent, self._tensor(action).reshape(1, self.edge_count),
            action_events=action_events.to(self.device),
            n_samples=method.disagreement_samples,
            steps=method.fm_steps, solver=method.fm_solver,
            picard_iterations=method.picard_iterations)
        # Keep a sampled successor; averaging would change the transition law.
        return samples[0], variance.reshape(())

    def _create_imagination_executor(self, *, rollout_index, critic_only):
        """Optional rollout-local state for serial physical execution."""
        return None

    def _advance_imagination_window(self, executor, states, **kwargs):
        return advance_imagined_window(states, **kwargs)

    def _imagine_from_seed(self, batch, batch_index, *, rollout_index, batched_events=False, copy_seed=True, critic_only=False):
        # Imagined explicit lifecycles follow observed active seed vehicles.
        # Each learned transition removes only births within its own window.
        states = [state for state in batch.vehicles[batch_index] if state['departed'] and not state['arrived']]
        if copy_seed:
            states = copy.deepcopy(states)
        timestamp = float(batch.time[batch_index])
        deadline = self._episode_end_time()
        if batch.started_count is None or batch.cumulative_vehicle_seconds is None:
            raise ValueError('imagination requires causal cumulative reward accounting')
        started_count = int(batch.started_count[batch_index])
        cumulative_seconds = float(batch.cumulative_vehicle_seconds[batch_index])
        total_vehicle_count = int(batch.total_vehicle_count[batch_index])
        if total_vehicle_count <= 0:
            raise ValueError('active imagination seed requires a positive episode population')
        if started_count <= 0:
            raise ValueError('active imagination seed requires a positive started population')
        if not states or timestamp >= deadline:
            raise ValueError('imagination seed needs active vehicles before the episode deadline')
        executor = self._create_imagination_executor(rollout_index=rollout_index, critic_only=critic_only)
        with torch.no_grad():
            latent = self.world_model.encode(batch.flow[batch_index, 0].unsqueeze(0).to(self.device))
        groups = []
        event_count = 0
        values, next_values, rewards, terminals, physical_windows = [], [], [], [], []
        physical_rewards, risk_costs, accounting_windows = [], [], []
        value_budgets = []
        windows = min(IMAGINED_HORIZON, math.ceil((deadline - timestamp) / WORLD_MODEL_INTERVAL_SECONDS))
        for window_index in range(windows):
            window_start = timestamp
            group_start = len(groups)
            window_cumulative = cumulative_seconds
            context = self._context(states, timestamp, started_count=started_count,
                                    cumulative_vehicle_seconds=cumulative_seconds)
            with torch.no_grad():
                decoded = self.world_model.decode(latent)[0].cpu().numpy()
            actor_window = prepare_actor_window(self._policy_features(decoded), self.network.edge_count)
            budget = remaining_cost_budget(timestamp, deadline, sum(not state['arrived'] for state in states), started_count)
            value_budgets.append(budget)
            value = self._value_prediction(latent.detach(), self._tensor(context).unsqueeze(0), [budget]).reshape(())

            def decide(events, frozen_flow, now):
                nonlocal event_count
                event_cumulative = window_cumulative + sum(
                    max(0., min(float(state.get('arrival_time', now)), now) - window_start)
                    for state in states)
                event_context = self._context(states, now, started_count=started_count,
                                             cumulative_vehicle_seconds=event_cumulative)
                observations = self._observations(events, actor_window, event_context, now)
                sampling_actor = getattr(self, '_sampling_actor', self.actor)
                sampling_device = torch.device('cpu') if sampling_actor is not self.actor else self.device
                features = torch.as_tensor(observations.features, dtype=torch.float32, device=sampling_device)
                mask = torch.as_tensor(observations.mask, device=sampling_device)
                # Sampling needs no backward graph. Recompute log probabilities
                # as one window batch below, while actor parameters are fixed.
                with torch.no_grad():
                    # The tiny frozen policy runs beside CPU vehicle motion;
                    # gradients use the original actor in a GPU window batch.
                    prior_costs = getattr(observations, 'prior_costs_seconds', None)
                    prior_kwargs = ({'prior_costs_seconds': torch.as_tensor(prior_costs, dtype=torch.float32, device=sampling_device)}
                                    if prior_costs is not None else {})
                    logits = sampling_actor.forward_logits(features, mask, **prior_kwargs)
                    if not bool(torch.isfinite(logits[mask]).all()):
                        raise ValueError('legal actor logits must be finite')
                    # CPU and CUDA RNG states are both checkpointed. Keeping
                    # the categorical draw beside the scorer avoids a device
                    # round trip at each physical observation. The probability
                    # law is unchanged, but historical seeded paths can differ.
                    probabilities = logits.softmax(dim=-1)
                    selected = torch.multinomial(probabilities, 1).squeeze(-1)
                selected_host = selected.detach().cpu().tolist()
                decisions = selected_decisions(events, observations, selected_host, now)
                if critic_only:
                    event_count += len(events)
                    return decisions
                groups.append(dict(budget=remaining_cost_budget(now, deadline,
                    sum(not state['arrived'] for state in states), started_count), vehicle_id=[str(state['vehicle_id']) for state in events],
                    window=window_index, population=started_count, _actor_pending=True,
                    decision_time=float(now), event_index=list(range(event_count, event_count + len(events))),
                    _q_observations=features.detach(), _q_mask=mask,
                    _q_candidates=observations.edge_indices.copy(), _q_context=event_context.copy(),
                    _q_probabilities=probabilities.detach(), _q_selected=selected,
                    _q_prior_costs=prior_costs))
                event_count += len(events)
                return decisions

            speeds = clip_road_speed_fractions(decoded[:, 1],
                getattr(self.config.simulation, 'traffic_model', 'micro')) * self.network.speed_limits_mps
            speeds = np.minimum(speeds, self.config.simulation.vehicle_max_speed_mps)
            duration = min(WORLD_MODEL_INTERVAL_SECONDS, deadline - timestamp)
            result = self._advance_imagination_window(executor, states, candidates=self.candidates,
                edge_speeds_mps=speeds, frozen_flow=decoded, start_time=timestamp,
                interval_seconds=duration, total_vehicle_count=total_vehicle_count, decide=decide,
                observation_step_seconds=1., session_protocol='event', event_index_start=event_count,
                traffic_model=getattr(self.config.simulation, 'traffic_model', 'micro'),
                vehicle_max_speed_mps=self.config.simulation.vehicle_max_speed_mps,
                stop_when_empty=True)
            policy_events = (result.action_events if result.policy_events is None else result.policy_events)
            self._complete_window_event_q(groups[group_start:], policy_events, latent, window_start)
            if result.action_events:
                event_count = max(event_count, result.action_events[-1].event_index + 1)
            full_window = abs(result.end_time - (timestamp + WORLD_MODEL_INTERVAL_SECONDS)) <= 1e-7
            with torch.no_grad():
                # A terminal partial tail has physical cost but no fictitious
                # fixed-30-second model transition or endpoint risk charge.
                if full_window:
                    sparse = pack_action_events([[result.action_events]], window_starts=[[window_start]])
                    next_latent, disagreement = self._predict_with_disagreement(latent, result.action, sparse)
                    risk_cost = self.config.method.disagreement_weight * disagreement
                else:
                    next_latent, risk_cost = latent, latent.new_zeros(())
                next_cumulative = cumulative_seconds + result.active_vehicle_seconds
                physical_reward = latent.new_tensor(-REWARD_SCALE * (
                    observed_mean_seconds(next_cumulative, started_count)
                    - observed_mean_seconds(cumulative_seconds, started_count)))
                reward = physical_reward - risk_cost
                accounting_windows.append((cumulative_seconds, started_count))
                cumulative_seconds = next_cumulative
                physical_rewards.append(physical_reward)
                risk_costs.append(risk_cost)
            timestamp = result.end_time
            unfinished = sum(not bool(state['arrived']) for state in states)
            timed_out = timestamp >= deadline - 1e-9
            terminal = timed_out or unfinished == 0
            next_context = self._context(states, timestamp, started_count=started_count,
                                         cumulative_vehicle_seconds=cumulative_seconds)
            with torch.no_grad():
                next_budget = remaining_cost_budget(timestamp, deadline, unfinished, started_count)
                next_value = self._value_prediction(next_latent, self._tensor(next_context).unsqueeze(0),
                    [next_budget], target=True).reshape(())
            values.append(value)
            next_values.append(next_value)
            rewards.append(reward)
            terminals.append(terminal)
            physical_windows.append(result)
            latent = next_latent
            if terminal:
                break
        # Physical arrivals at the endpoint are accounted for; unfinished trips
        # contribute observed time only, without a terminal penalty.
        failed_ids = set()
        # Macro lambda returns retain the existing gamma=1 convention. Event
        # targets use full Monte Carlo reward-to-go with the same undiscounted
        # physical-time objective, plus a value bootstrap only at truncation.
        targets = lambda_return(torch.stack(rewards), torch.stack(next_values),
                                terminal=torch.tensor(terminals, device=self.device))
        values_tensor = torch.stack(values)
        self._validate_critic_targets(targets, value_budgets)
        critic_loss = F.mse_loss(values_tensor, targets.detach())
        bootstrap = next_values[-1] if not terminals[-1] else latent.new_zeros(())
        for row in groups:
            window = row['window']
            # Remove only physical time already elapsed before this action.
            # Endpoint risk depends on all actions in the macro window and is
            # therefore retained in full, even for a late event.
            elapsed = sum(segment.active_count * max(0., min(segment.end_time, row['decision_time'])
                - segment.start_time) for segment in physical_windows[window].cost_segments)
            start_cumulative, start_count = accounting_windows[window]
            event_mean = observed_mean_seconds(start_cumulative + elapsed, start_count)
            physical_suffix = -REWARD_SCALE * (observed_mean_seconds(cumulative_seconds, started_count) - event_mean)
            target = latent.new_tensor(physical_suffix) - torch.stack(risk_costs[window:]).sum() + bootstrap
            self._validate_critic_targets(target.reshape(1), [row['budget']])
            row['global_return'] = target.detach().expand(len(row['vehicle_id']))
            row['global_advantage'] = row['global_return'] - row['counterfactual_baseline']
            row['global_reward'] = physical_rewards[window]
            row['risk_cost'] = risk_costs[window]
            row['timeout_return'] = latent.new_zeros(len(row['vehicle_id']))
        if batched_events:
            return groups, critic_loss
        # Compatibility view for diagnostics: the training hot path keeps
        # simultaneous events in tensors, rather than scalar autograd nodes.
        rows = []
        vector_keys = ('log_prob', 'entropy', 'counterfactual_baseline', 'executed_q',
                       'global_return', 'global_advantage')
        for group in groups:
            for index, vehicle_id in enumerate(group['vehicle_id']):
                row = {key: group[key][index] for key in vector_keys}
                row.update(vehicle_id=vehicle_id, event_index=group['event_index'][index],
                    window=group['window'], decision_time=group['decision_time'],
                    global_reward=group['global_reward'])
                rows.append(row)
        attach_timeout_penalties(rows, failed_ids)
        return rows, critic_loss

    def _complete_window_event_q(self, groups, events, latent, window_start):
        """Batch fixed decision conditions before any policy/critic update.

        Each row keeps its sampling-time observation, context and policy
        probabilities. The Q builder masks every later event from that row;
        batching computation does not expose the future or refit the baseline.
        """
        if not groups:
            return
        width = max(group['_q_mask'].shape[1] for group in groups)
        sizes = [len(group['vehicle_id']) for group in groups]
        masks = torch.cat([F.pad(group['_q_mask'], (0, width - group['_q_mask'].shape[1]))
                           for group in groups]).to(self.device)
        observations = torch.cat([F.pad(group['_q_observations'],
            (0, 0, 0, width - group['_q_mask'].shape[1])) for group in groups]).to(self.device)
        candidates = np.concatenate([np.pad(group['_q_candidates'],
            ((0, 0), (0, width - group['_q_mask'].shape[1]))) for group in groups])
        contexts = np.concatenate([np.repeat(group['_q_context'][None], size, axis=0)
                                   for group, size in zip(groups, sizes)])
        sparse = pack_action_events([[tuple(events)]], window_starts=[[window_start]])
        features = build_event_q_inputs(self.world_model, z=latent.detach(), observations=observations,
            candidate_edges=torch.as_tensor(candidates, device=self.device), mask=masks,
            current_edges=torch.tensor([event.current_edge for event in events], device=self.device),
            destination_edges=torch.tensor([event.destination_edge for event in events], device=self.device),
            decision_times=torch.tensor([event.decision_time for event in events], device=self.device),
            event_indices=torch.tensor([event.event_index for event in events], device=self.device),
            context=self._tensor(contexts), events=sparse.to(self.device), window_start=window_start)
        q_kwargs = {}
        if self._bounded_critic():
            q_kwargs['budget'] = torch.tensor([group['budget'] for group, size in zip(groups, sizes)
                for _ in range(size)], dtype=torch.float32, device=self.device)
        # Every candidate is needed for the saved baseline, but only the
        # executed candidate contributes to the Q loss. Do not retain hidden
        # activations for unused candidates across the whole imagination pass.
        with torch.no_grad():
            q_values = self.event_critic(features, masks, **q_kwargs)
        probabilities = torch.cat([F.pad(group['_q_probabilities'],
            (0, width - group['_q_mask'].shape[1])) for group in groups]).to(self.device)
        baseline = detached_counterfactual_baseline(probabilities, q_values, masks)
        selected = torch.cat([group['_q_selected'] for group in groups]).to(self.device)
        if all(group.get('_actor_pending', False) for group in groups):
            actor_kwargs = {}
            if groups[0].get('_q_prior_costs') is not None:
                actor_kwargs['prior_costs_seconds'] = torch.as_tensor(np.concatenate([
                    np.pad(group['_q_prior_costs'], ((0, 0), (0, width - group['_q_mask'].shape[1])))
                    for group in groups]), dtype=torch.float32, device=self.device)
            distribution = torch.distributions.Categorical(logits=self.actor.forward_logits(observations, masks, **actor_kwargs))
            log_probs, entropies = distribution.log_prob(selected), distribution.entropy()
            for group, log_prob, entropy in zip(groups, log_probs.split(sizes), entropies.split(sizes)):
                group['log_prob'], group['entropy'] = log_prob, entropy
                del group['_actor_pending']
        executed_features = features[torch.arange(len(selected), device=self.device), selected]
        executed = self.event_critic(executed_features[:, None],
            torch.ones((len(selected), 1), dtype=torch.bool, device=self.device), **q_kwargs).squeeze(1)
        for group, values, saved in zip(groups, executed.split(sizes), baseline.split(sizes)):
            group['executed_q'] = values
            group['counterfactual_baseline'] = saved
            for key in tuple(group):
                if key.startswith('_q_'):
                    del group[key]

    def _train_imagination_pass(self):
        if self._decoder_training_enabled() and self._decoder_revision() > 0:
            self.validate_codec_ready()
        if not self.online_replay.has_active_seeds:
            return {'latent_critic_loss': 0., 'imagined_events': 0., 'actor_loss': 0.}
        self.world_model.eval()
        from .fm_inference import configure_imagination_inference
        self.metrics['world_model_imagination_execution_profile'] = configure_imagination_inference(self)
        rows, macro_losses = [], []
        if self._bounded_critic():
            self._critic_diagnostics = dict(critic_predictions=0, critic_nonfinite_predictions=0, critic_bound_violations=0,
                critic_fraction_saturated=0, critic_terminal_predictions=0, critic_tiny_budget_predictions=0,
                critic_target_count=0, critic_target_bound_violations=0)
        workers = int(os.environ.get('URB_IMAGINATION_WORKERS', self.metrics.get('imagination_workers', 0)))
        rollout_count = self.config.resources.imagination_rollouts_per_pass
        if not 0 <= workers <= rollout_count:
            raise ValueError('imagination workers must be between zero and the rollout count')
        if workers:
            from .parallel_imagination import collect_parallel_imagination
            rows, macro_losses = collect_parallel_imagination(self, workers=workers)
        else:
            if self.device.type == 'cuda':
                self._sampling_actor = copy.deepcopy(self.actor).to('cpu').requires_grad_(False)
            try:
                for rollout in range(rollout_count):
                    batch = self.online_replay.sample_active_seeds(batch_size=1, rng=self.rng, initial_only=True)
                    events, loss = self._imagine_from_seed(batch, 0, rollout_index=rollout,
                        batched_events=True, copy_seed=False)
                    rows.extend(events)
                    macro_losses.append(loss)
            finally:
                if hasattr(self, '_sampling_actor'):
                    del self._sampling_actor
        self.metrics['imagination_workers'] = workers
        auxiliary_losses = self._terminal_aux_losses(rollout_count)
        macro_losses.extend(auxiliary_losses)
        macro_loss = torch.stack(macro_losses).mean()
        # Macro learning is independent of whether this window contained policy events.
        self.optimizers['latent_critic'].zero_grad(set_to_none=True)
        macro_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.latent_critic.parameters(), GRAD_CLIP)
        self.optimizers['latent_critic'].step()
        self._update_target_critic()
        metrics = {'latent_critic_loss': float(macro_loss.detach()),
                   'imagined_events': float(sum(len(row['vehicle_id']) for row in rows)), 'actor_loss': 0.}
        if self._bounded_critic():
            metrics['critic_terminal_aux_rollouts'] = float(len(auxiliary_losses))
            metrics['critic_target_tau'] = self.config.method.critic_target_tau
            metrics.update(self._critic_diagnostics)
            del self._critic_diagnostics
        if rows:
            q_loss = F.mse_loss(torch.cat([row['executed_q'] for row in rows]),
                                torch.cat([row['global_return'] for row in rows]).detach())
            self.optimizers['event_critic'].zero_grad(set_to_none=True)
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.event_critic.parameters(), GRAD_CLIP)
            self.optimizers['event_critic'].step()
            shared_advantage = torch.cat([row['global_advantage'] for row in rows])
            advantage = individual_actor_advantage(shared_advantage, torch.zeros_like(shared_advantage))
            entropies = torch.cat([row['entropy'] for row in rows])
            entropy_mode = os.environ.get('URB_ACTOR_ENTROPY_MODE', self.config.method.actor_entropy_mode)
            entropy_profile = entropy_schedule_profile(
                replace(self.config.method, actor_entropy_mode=entropy_mode),
                completed_episodes=int(self.metrics.get('online_episodes_completed', 0)),
                total_episodes=max(1, int(self.config.run.online_training_episodes)))
            populations = (torch.cat([row['entropy'].new_full(row['entropy'].shape, row['population'])
                                      for row in rows]) if entropy_mode == 'population'
                           else torch.ones_like(entropies))
            entropy_bonus, entropy_stats = population_scaled_entropy(entropies, populations,
                torch.cat([row['global_advantage'] for row in rows]),
                base_coef=entropy_profile['coefficient'], mode=entropy_mode)
            actor_loss = -(torch.cat([row['log_prob'] for row in rows]) * advantage.detach()
                           + entropy_bonus).mean()
            self.optimizers['actor'].zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
            self.optimizers['actor'].step()
            metrics['actor_loss'] = float(actor_loss.detach())
            metrics.update({key: float(value) for key, value in entropy_stats.items()})
            metrics['policy_entropy_mean'] = float(entropies.detach().mean())
            metrics['entropy_schedule_progress'] = entropy_profile['progress']
            metrics['entropy_schedule_completed_episodes'] = entropy_profile['completed_episodes']
            metrics['entropy_schedule_total_episodes'] = entropy_profile['total_episodes']
            self.metrics['actor_entropy_mode'] = entropy_mode
            baselines = torch.cat([row['counterfactual_baseline'] for row in rows])
            global_advantage = torch.cat([row['global_advantage'] for row in rows])
            metrics.update(event_critic_loss=float(q_loss.detach()),
                counterfactual_baseline_mean=float(baselines.mean()),
                counterfactual_baseline_std=float(baselines.std(unbiased=False)),
                global_advantage_mean=float(global_advantage.mean()),
                global_advantage_std=float(global_advantage.std(unbiased=False)),
                actor_advantage_mean=float(advantage.mean()),
                actor_advantage_std=float(advantage.std(unbiased=False)))
        self.metrics.update({key + '_last': value for key, value in metrics.items() if key.endswith('_loss')})
        return metrics

    def state_dict(self):
        numpy_state = np.random.get_state()
        state = {'reward_contract': REWARD_CONTRACT,
            'models': {name: getattr(self, name).state_dict() for name in
                    ('world_model', 'actor', 'latent_critic', 'event_critic')},
            'optimizers': {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            'replay': {'online': self.online_replay.state_dict(), 'offline': self.offline_data.state_dict(),
                      'codec_validation': self.codec_validation_data.state_dict() if self.codec_validation_data is not None else None},
            'rng': {'python_random': random.getstate(), 'numpy_global': list(numpy_state),
                'numpy_generator': copy.deepcopy(self.rng.bit_generator.state), 'torch_cpu': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            'metrics': dict(self.metrics), 'next_environment_episode': self._next_environment_episode}
        if self._bounded_critic():
            state['models']['target_latent_critic'] = self.target_latent_critic.state_dict()
            state['rng'].update(critic_aux_numpy=copy.deepcopy(self._critic_aux_rng.bit_generator.state),
                critic_aux_torch_cpu=self._critic_aux_torch_cpu.clone(),
                critic_aux_torch_cuda=[value.clone() for value in self._critic_aux_torch_cuda])
        return state

    def load_state_dict(self, state):
        if set(state) != {'reward_contract', 'models', 'optimizers', 'replay', 'rng', 'metrics', 'next_environment_episode'}:
            raise ValueError('MainlineTrainer checkpoint state has an invalid schema')
        if state['reward_contract'] != REWARD_CONTRACT:
            raise ValueError('MainlineTrainer checkpoint reward contract requires causal cumulative mean global reward')
        model_names = {'world_model', 'actor', 'latent_critic', 'event_critic'}
        if self._bounded_critic():
            model_names.add('target_latent_critic')
            for key in ('critic_value_mode', 'critic_target_tau', 'critic_terminal_aux_ratio'):
                if state['metrics'].get(key) != getattr(self.config.method, key):
                    raise ValueError(f'bounded critic checkpoint contract mismatch: {key}')
            required_rng = {'critic_aux_numpy', 'critic_aux_torch_cpu', 'critic_aux_torch_cuda'}
            if not required_rng <= set(state['rng']):
                raise ValueError('bounded critic checkpoint requires auxiliary RNG state')
            if (self.device.type == 'cuda'
                    and len(state['rng']['critic_aux_torch_cuda']) != torch.cuda.device_count()):
                raise ValueError('bounded critic auxiliary RNG requires the checkpoint CUDA device count')
        elif state['metrics'].get('critic_value_mode', 'legacy') != 'legacy':
            raise ValueError('bounded critic checkpoint cannot load into legacy critics')
        if set(state['models']) != model_names:
            raise ValueError('MainlineTrainer model schema requires the event Q critic')
        if set(state['optimizers']) != set(self.optimizers):
            raise ValueError('MainlineTrainer optimizer schema requires global reward only')
        if set(state['replay']) != {'online', 'offline', 'codec_validation'}:
            raise ValueError('MainlineTrainer replay schema requires frozen offline manifests')
        self.offline_data.load_state_dict(state['replay']['offline'])
        validation = state['replay']['codec_validation']
        if (validation is None) != (self.codec_validation_data is None):
            raise ValueError('codec validation source differs from checkpoint')
        if validation is not None:
            self.codec_validation_data.load_state_dict(validation)
        for name in model_names:
            getattr(self, name).load_state_dict(state['models'][name], strict=True)
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][name])
        self.online_replay.load_state_dict(state['replay']['online'])
        self._next_environment_episode = int(state['next_environment_episode'])
        self.metrics = dict(state['metrics'])
        self._set_online_prior()
        admitted = (self.config.codec.decoder_pretrain_enabled
                    and self.metrics.get('decoder_pretrain_authorized_manifest_identity')
                    == getattr(self.codec_validation_data, 'manifest', {}).get('identity')
                    and self.metrics.get('decoder_pretrain_authorized_codec_updates')
                    == self.metrics.get('codec_updates'))
        online_admitted = (self.config.codec.decoder_online_enabled
            and self.codec_validation_data is not None
            and self.metrics.get('decoder_online_authorized_manifest_identity') == self.codec_validation_data.manifest['identity']
            and self.metrics.get('decoder_online_authorized_codec_updates') == self.metrics.get('codec_updates'))
        if self.codec_ready() or admitted or online_admitted or self._codec_override_admitted():
            self._set_dynamics_codec_gradients()
        else:
            self.world_model.flow_projector.requires_grad_(True)
        rng = state['rng']
        if self._bounded_critic():
            self._critic_aux_rng.bit_generator.state = copy.deepcopy(rng['critic_aux_numpy'])
            self._critic_aux_torch_cpu = torch.as_tensor(rng['critic_aux_torch_cpu'], dtype=torch.uint8, device='cpu').clone()
            self._critic_aux_torch_cuda = [torch.as_tensor(value, dtype=torch.uint8, device='cpu').clone()
                                          for value in rng['critic_aux_torch_cuda']]
        random.setstate(_tuple_tree(rng['python_random']))
        numpy_state = list(rng['numpy_global'])
        numpy_state[1] = np.asarray(numpy_state[1], dtype=np.uint32)
        np.random.set_state(tuple(numpy_state))
        self.rng.bit_generator.state = copy.deepcopy(rng['numpy_generator'])
        torch.set_rng_state(torch.as_tensor(rng['torch_cpu'], dtype=torch.uint8, device='cpu'))
        if torch.cuda.is_available() and rng['torch_cuda']:
            torch.cuda.set_rng_state_all([torch.as_tensor(s, dtype=torch.uint8, device='cpu') for s in rng['torch_cuda']])

    def summary(self):
        return {**self.metrics,
            'pretrain_epochs_completed': self.metrics['world_model_initialization_epochs_completed'],
            'pretrain_samples_seen': self.metrics['world_model_initialization_samples_seen'],
            'online_replay_episodes': len(self.online_replay.episodes),
            **{'offline_' + key: value for key, value in self.offline_data.summary.items()},
            'next_environment_episode': self._next_environment_episode}

    def close(self):
        # Sessions are scoped to _run_episode and released even when collection fails.
        if getattr(self, '_parallel_imagination_executor', None) is not None:
            from .parallel_imagination import close_parallel_imagination
            close_parallel_imagination(self)
