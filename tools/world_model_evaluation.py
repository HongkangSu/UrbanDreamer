"""Held-out traffic predictions with fixed stochastic trajectories.

The decoder's volume belongs to its boundary, but its speed describes the
following interval. Reconstruction therefore compares speed at t to the mean
of observed speeds at t and t+30. A rollout endpoint at t+30 needs observations
at both t+30 and t+60 to evaluate speed; an unavailable speed label is excluded
and its count is zero. Stochastic trajectories remain separate throughout the
recursion; decoded traffic and fraction probabilities are averaged, then scored.

Occupied-road errors use strictly positive *observed target* volume, so large
empty networks cannot dilute those metrics. Persistence retains the initial
observed road speed/volume and initial active fraction at every forecast horizon.
Endpoint reconstruction diagnostics use exactly the same forecast targets and
road masks, separating codec errors from rollout errors. Total-volume bias is
signed (prediction minus observation) in vehicles, including empty roads. The
negative raw speed fraction counts individual decoded stochastic samples before
their predictions are averaged; it does not clip speed or require a speed label.
With initial-cohort transition scope, forecasts and endpoint reconstructions use
fixed seed-membership volume, events and exposure. The raw reconstruction
metrics continue to evaluate the full observed environment for the codec.
With previous-state transition scope, every prediction starts from its own full
observed boundary. Only that window's endpoint volume excludes current-window
births; actions and exposure retain full traffic. These independent thirty-second
forecasts are identified by their window offset, never by a recursive horizon.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import random

import numpy as np
import torch

from urban_dreamer.learning.cohort_targets import prepare_cohort_batch, prepare_transition_targets


@contextmanager
def _fixed_evaluation_state(trainer, seed):
    model = trainer.world_model
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    rng = getattr(trainer, 'rng', None)
    generator_state = deepcopy(rng.bit_generator.state) if rng is not None else None
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            model.eval()
            with torch.no_grad():
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if rng is not None:
            rng.bit_generator.state = generator_state
        # Restoring flags individually preserves deliberately mixed module modes.
        for module, mode in modes:
            module.training = mode


def _repeat_events(events, samples):
    """Repeat one sparse physical window as independent stochastic batch rows."""
    columns = ('time_index', 'phase', 'current_edge', 'next_edge',
               'destination_edge', 'event_index')
    return replace(events,
        **{name: getattr(events, name).repeat(samples) for name in columns},
        batch_index=torch.cat([events.batch_index + sample * events.batch_size
                               for sample in range(samples)]),
        vehicle_ids=events.vehicle_ids * samples, batch_size=events.batch_size * samples)


class _Metrics:
    def __init__(self):
        self.sums, self.counts = defaultdict(float), defaultdict(int)

    def add(self, name, values, mask=None):
        values = values.detach().double()
        if mask is not None:
            values = values[mask]
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f'nonfinite held-out metric: {name}')
        self.sums[name] += float(values.sum())
        self.counts[name] += values.numel()

    def result(self):
        return {**{name: self.sums[name] / max(count, 1) for name, count in self.counts.items()},
                **{name + '_count': count for name, count in self.counts.items()}}


def _road_metrics(metrics, prefix, prediction, target, speed_target, limits, capacities):
    occupied = target[..., 2] > 0
    volume_error = (prediction[..., 2] - target[..., 2]).abs()
    metrics.add(prefix + '_volume_mae_normalized', volume_error)
    metrics.add(prefix + '_volume_mae_vehicles', volume_error * capacities)
    metrics.add(prefix + '_occupied_road_volume_mae_normalized', volume_error, occupied)
    metrics.add(prefix + '_occupied_road_volume_mae_vehicles', volume_error * capacities, occupied)
    metrics.add(prefix + '_total_volume_mae_vehicles',
                ((prediction[..., 2] - target[..., 2]) * capacities).sum(-1).abs())
    metrics.add(prefix + '_total_volume_bias_vehicles',
                ((prediction[..., 2] - target[..., 2]) * capacities).sum(-1))
    metrics.add(prefix + '_true_total_volume_mean_vehicles', (target[..., 2] * capacities).sum(-1))
    metrics.add(prefix + '_pred_total_volume_mean_vehicles', (prediction[..., 2] * capacities).sum(-1))
    if speed_target is None:
        speed_error = prediction[..., 1].new_empty((0, prediction.shape[-2]))
        speed_occupied = torch.zeros_like(speed_error, dtype=torch.bool)
    else:
        speed_error = (prediction[..., 1] - speed_target).abs()
        speed_occupied = occupied
    metrics.add(prefix + '_speed_mae_normalized', speed_error)
    metrics.add(prefix + '_speed_mae_mps', speed_error * limits)
    metrics.add(prefix + '_occupied_road_speed_mae_normalized', speed_error, speed_occupied)
    metrics.add(prefix + '_occupied_road_speed_mae_mps', speed_error * limits, speed_occupied)


def evaluate_world_model(trainer, batches, *, seed=31415, n_samples=2, horizon=3):
    """Evaluate fixed held-out batches without changing training or random state.

    Inputs are replay ``SequenceBatch`` instances with full thirty-second event
    windows. Legacy forecasts start at the first boundary and recurse. Previous-
    state supervision instead predicts at most ``horizon`` independent windows,
    each starting from its full observed input. The returned dictionary contains
    numeric MAEs and their element counts, including occupied-road counts.
    A zero count denotes an unavailable label, not a measured zero error.
    """
    for name, value in (('seed', seed), ('n_samples', n_samples), ('horizon', horizon)):
        if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'seed' else 1):
            raise ValueError(f'{name} must be a nonnegative seed or positive integer count')
    if horizon > 3:
        raise ValueError('held-out forecast horizon must be at most three 30-second windows')
    model, method, device = trainer.world_model, trainer.config.method, trainer.device
    independent = getattr(method, 'transition_volume_scope', 'full') == 'previous_state_excluding_current_births'
    limits = torch.as_tensor(trainer.network.speed_limits_mps, device=device, dtype=torch.float32)
    capacities = torch.as_tensor(trainer.network.capacities, device=device, dtype=torch.float32).clamp_min(1.)
    metrics, batch_count, sequence_count = _Metrics(), 0, 0
    with _fixed_evaluation_state(trainer, seed):
        for batch in batches:
            flow, action = batch.flow.to(device), batch.action.to(device)
            if flow.ndim != 4 or flow.shape[-1] != 8 or flow.shape[1] < 2:
                raise ValueError('held-out flow must have shape (B, T+1, E, 8) with at least one window')
            size, frames, edges, _ = flow.shape
            if size == 0 or edges == 0 or tuple(action.shape) != (size, frames - 1, edges):
                raise ValueError('held-out actions must align with nonempty flow windows')
            if limits.shape != (edges,) or capacities.shape != (edges,):
                raise ValueError('held-out road registry must match the trainer network')
            if not bool(torch.isfinite(flow).all()):
                raise ValueError('held-out observed flow must be finite')
            if batch.action_events is None:
                raise ValueError('held-out event evaluation requires explicit window events')
            events = batch.action_events.to(device)
            if (events.batch_size, events.time_steps) != (size, frames - 1):
                raise ValueError('held-out sparse events must align with physical windows')
            if batch.active_vehicle_fraction.shape != (size, frames - 1):
                raise ValueError('held-out active fractions must have one value per full window')
            encoded = model.encode(flow.reshape(-1, edges, 8)).reshape(size, frames, edges, -1)
            reconstructed = model.decode(encoded[:, :-1].reshape(-1, edges, encoded.shape[-1]))
            reconstructed = reconstructed.reshape(size, frames - 1, edges, -1)
            _road_metrics(metrics, 'reconstruction', reconstructed, flow[:, :-1],
                (flow[:, :-1, :, 1] + flow[:, 1:, :, 1]) / 2., limits, capacities)
            cohort_scope = getattr(method, 'transition_volume_scope', 'full') == 'initial_cohort'
            if independent:
                scoped = prepare_transition_targets(batch, capacities=capacities)
                target_flow = scoped.transition_target_flow.to(device)
                target_action, target_events = action, events
                target_fraction = batch.active_vehicle_fraction.to(device)
                forecast_encoded = model.encode(target_flow.reshape(-1, edges, 8)).reshape(size, frames - 1, edges, -1)
            elif cohort_scope:
                scoped = prepare_cohort_batch(batch, capacities=capacities)
                target_flow = scoped.transition_flow.to(device)
                target_action = scoped.transition_action.to(device)
                target_events = scoped.transition_action_events.to(device)
                target_fraction = scoped.transition_active_vehicle_fraction.to(device)
                forecast_encoded = model.encode(target_flow.reshape(-1, edges, 8)).reshape(size, frames, edges, -1)
            else:
                target_flow, target_action, target_events = flow, action, events
                target_fraction = batch.active_vehicle_fraction.to(device)
                forecast_encoded = encoded
            initial = forecast_encoded[:, 0]
            current = initial.repeat(n_samples, 1, 1)
            persistence = target_flow[:, 0]
            initial_fraction = batch.context[:, 0, 1].to(device)
            for step in range(min(horizon, frames - 1)):
                if independent:
                    current = encoded[:, step].repeat(n_samples, 1, 1)
                    persistence = flow[:, step]
                    initial_fraction = batch.context[:, step, 1].to(device)
                previous = current
                current = model.transition(current, target_action[:, step].repeat(n_samples, 1),
                    action_events=_repeat_events(target_events.for_step(step), n_samples),
                    steps=method.fm_steps, solver=method.fm_solver,
                    picard_iterations=method.picard_iterations)
                samples = model.decode(current).reshape(n_samples, size, edges, -1)
                prediction = samples.mean(dim=0)
                target = target_flow[:, step if independent else step + 1]
                following = flow if independent else target_flow
                speed_target = ((target[..., 1] + following[:, step + 2, :, 1]) / 2.
                                if step + 2 < frames else None)
                suffix = f'{30 * (step + 1)}s'
                forecast_prefix = f'window_{30 * step}_forecast30' if independent else 'rollout_' + suffix
                persistence_prefix = f'window_{30 * step}_persistence' if independent else 'persistence_' + suffix
                endpoint_prefix = f'window_{30 * step}_endpoint_reconstruction' if independent else 'endpoint_reconstruction_' + suffix
                _road_metrics(metrics, forecast_prefix, prediction, target, speed_target, limits, capacities)
                _road_metrics(metrics, persistence_prefix, persistence, target, speed_target, limits, capacities)
                endpoint_reconstruction = model.decode(forecast_encoded[:, step if independent else step + 1])
                _road_metrics(metrics, endpoint_prefix,
                              endpoint_reconstruction, target, speed_target, limits, capacities)
                metrics.add(forecast_prefix + '_negative_raw_speed_fraction', samples[..., 1] < 0)
                fraction_samples = model.active_vehicle_fraction_readout(current, z_prev=previous,
                    raw_decoder_fraction=samples[..., 2].mean(-1).reshape(-1), transition_seconds=30.)
                fraction = fraction_samples.reshape(n_samples, size).mean(0)
                fraction_target = target_fraction[:, step]
                metrics.add(forecast_prefix + '_active_fraction_mae', (fraction - fraction_target).abs())
                metrics.add(persistence_prefix + '_active_fraction_mae',
                            (initial_fraction - fraction_target).abs())
            batch_count += 1
            sequence_count += size
    if batch_count == 0:
        raise ValueError('held-out evaluation requires at least one batch')
    result = dict(metrics.result(), evaluated_batches=batch_count, evaluated_sequences=sequence_count,
                  stochastic_samples=n_samples, requested_horizon_windows=horizon)
    if independent:
        result.update(supervision='independent_observed_start_windows', forecast_window_seconds=30,
                      transition_volume_scope='previous_state_excluding_current_births')
    return result
