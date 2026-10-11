from copy import deepcopy
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools.world_model_evaluation import evaluate_world_model
from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.learning.replay import SequenceBatch


class OracleModel(nn.Module):
    def __init__(self, noise=0.):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.))
        self.child = nn.Linear(1, 1)
        self.noise = noise
        self.calls = []
        self.chunk_profile = (3, 7)

    def encode(self, flow):
        return torch.nn.functional.pad(flow, (0, 56)) + self.bias

    def decode(self, latent):
        values = latent[..., :4].clone()
        values[..., 1] += 1.
        return values

    def transition(self, z, action, *, action_events, **kwargs):
        self.calls.append((action_events, tuple(action.shape), kwargs))
        result = z.clone()
        result[..., 1] += 2.
        result[..., 2] += action
        if self.noise:
            # Exercise all global RNGs; real world-model sampling uses Torch.
            result[..., 2] += (torch.randn_like(action) + random.random() + np.random.random()) * self.noise
        return result

    def active_vehicle_fraction_readout(self, current, *, z_prev, raw_decoder_fraction, transition_seconds):
        assert transition_seconds == 30.
        assert current.shape == z_prev.shape
        return raw_decoder_fraction


def make_batch(frames=5, size=1):
    flow = torch.zeros(size, frames, 2, 8)
    flow[..., 1] = torch.arange(frames)[None, :, None] * 2.
    flow[:, :, 0, 2] = torch.arange(frames)[None] + 1.
    action = torch.tensor([1., 0.]).expand(size, frames - 1, 2).clone()
    windows = [[(ActionEvent(f'car{row}', step, step * 30. + 1., 0, 1, 1),)
                for step in range(frames - 1)] for row in range(size)]
    context = torch.zeros(size, frames - 1, 2)
    context[..., 1] = .5
    return SequenceBatch(flow=flow, action=action, departure=torch.zeros_like(action), context=context,
        active_vehicle_fraction=flow[:, 1:, :, 2].mean(-1),
        vehicles=tuple(() for _ in range(size)), time=torch.zeros(size),
        terminal=torch.zeros(size, frames - 1, dtype=torch.bool), total_travel_minutes=torch.ones(size),
        episode_complete=torch.zeros(size, dtype=torch.bool), episode=torch.arange(size),
        start_index=torch.zeros(size, dtype=torch.long), total_vehicle_count=torch.ones(size, dtype=torch.long),
        action_events=pack_action_events(windows))


def make_trainer(model=None):
    model = model or OracleModel()
    return SimpleNamespace(world_model=model, device=torch.device('cpu'),
        network=SimpleNamespace(speed_limits_mps=np.array([10., 20.]), capacities=np.array([5., 10.])),
        config=SimpleNamespace(method=SimpleNamespace(fm_steps=2, fm_solver='picard', picard_iterations=3)),
        rng=np.random.default_rng(7), metrics={'world_model_updates': 201, 'world_model_supervision_depth_last': 3},
        optimizers={'world': torch.optim.Adam(model.parameters())})


def test_labels_use_endpoint_volume_and_forward_interval_speed():
    trainer = make_trainer()
    actual = evaluate_world_model(trainer, [make_batch()], n_samples=2)
    for prefix in ('reconstruction', 'rollout_30s', 'rollout_60s', 'rollout_90s'):
        assert actual[prefix + '_volume_mae_normalized'] == 0.
        assert actual[prefix + '_speed_mae_normalized'] == 0.
        assert actual[prefix + '_occupied_road_speed_mae_mps'] == 0.
    for suffix in ('30s', '60s', '90s'):
        for field in ('volume_mae_normalized', 'occupied_road_volume_mae_vehicles',
                      'speed_mae_normalized', 'occupied_road_speed_mae_mps'):
            assert actual['endpoint_reconstruction_' + suffix + '_' + field] == actual['rollout_' + suffix + '_' + field]
    assert actual['rollout_90s_active_fraction_mae'] == 0.
    assert actual['persistence_30s_volume_mae_normalized'] == .5
    assert actual['persistence_30s_occupied_road_volume_mae_normalized'] == 1.
    assert actual['persistence_30s_volume_mae_vehicles'] == 2.5
    assert actual['persistence_30s_occupied_road_volume_mae_vehicles'] == 5.
    assert actual['persistence_30s_speed_mae_normalized'] == 3.
    assert actual['persistence_30s_speed_mae_mps'] == 45.
    assert actual['persistence_90s_occupied_road_volume_mae_vehicles'] == 15.
    assert actual['persistence_90s_speed_mae_normalized'] == 7.
    assert actual['evaluated_sequences'] == 1
    for step, (events, shape, options) in enumerate(trainer.world_model.calls):
        assert shape == (2, 2)
        assert events.batch_size == 2 and events.time_steps == 1
        assert events.event_index.tolist() == [step, step]
        assert events.batch_index.tolist() == [0, 1]
        assert events.time_index.tolist() == [0, 0]
        assert options == dict(steps=2, solver='picard', picard_iterations=3)


def test_unavailable_forward_speed_label_is_excluded_without_fabrication():
    actual = evaluate_world_model(make_trainer(), [make_batch(frames=4)])
    assert actual['rollout_90s_volume_mae_normalized_count'] == 2
    assert actual['rollout_90s_speed_mae_normalized_count'] == 0
    assert actual['rollout_90s_speed_mae_normalized'] == 0.
    assert actual['rollout_60s_speed_mae_normalized_count'] == 2
    assert actual['reconstruction_speed_mae_normalized_count'] == 6
    assert actual['endpoint_reconstruction_90s_speed_mae_normalized_count'] == 0
    assert actual['endpoint_reconstruction_90s_volume_mae_normalized_count'] == 2


def test_metric_reduction_weights_elements_and_exposes_empty_road_counts():
    first, second = make_batch(size=1), make_batch(size=3)
    second = replace(second, action=torch.zeros_like(second.action))
    actual = evaluate_world_model(make_trainer(), [first, second], horizon=1)
    assert actual['rollout_30s_volume_mae_normalized'] == .375
    assert actual['rollout_30s_occupied_road_volume_mae_normalized'] == .75
    assert actual['rollout_30s_volume_mae_normalized_count'] == 8
    assert actual['rollout_30s_occupied_road_volume_mae_normalized_count'] == 4
    assert actual['evaluated_batches'] == 2
    empty = make_batch()
    empty.flow[..., 2] = 0.
    empty.action.zero_()
    errors = evaluate_world_model(make_trainer(), [empty], horizon=1)
    assert errors['rollout_30s_occupied_road_speed_mae_mps_count'] == 0
    assert errors['rollout_30s_occupied_road_speed_mae_mps'] == 0.


def test_predictions_average_decoded_samples_not_latents():
    class NonlinearModel(OracleModel):
        def decode(self, latent):
            decoded = super().decode(latent)
            decoded[..., 2] = latent[..., 2].square()
            return decoded

        def transition(self, z, action, **kwargs):
            result = super().transition(z, action, **kwargs)
            result[..., 2] = torch.tensor([-1., 1.])[:, None]
            return result

    batch = make_batch(frames=2)
    batch.flow[:, 1, :, 2] = 1.
    actual = evaluate_world_model(make_trainer(NonlinearModel()), [batch], horizon=1)
    # Mean decoded (-1)^2 and 1^2 is 1. Decoding their mean would predict zero.
    assert actual['rollout_30s_volume_mae_normalized'] == 0.


def test_fixed_seed_preserves_random_flags_parameters_optimizers_and_trainer_state():
    trainer = make_trainer(OracleModel(noise=.3))
    trainer.world_model.train()
    trainer.world_model.child.eval()
    model_before = deepcopy(trainer.world_model.state_dict())
    optimizer_before = deepcopy(trainer.optimizers['world'].state_dict())
    metrics_before = dict(trainer.metrics)
    python_before, numpy_before = random.getstate(), np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    generator_before = deepcopy(trainer.rng.bit_generator.state)
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    def batches():
        trainer.rng.random()
        yield make_batch()
    first = evaluate_world_model(trainer, batches(), seed=73)
    second = evaluate_world_model(trainer, batches(), seed=73)
    assert first == second
    assert random.getstate() == python_before
    now_numpy = np.random.get_state()
    assert now_numpy[0] == numpy_before[0] and now_numpy[2:] == numpy_before[2:]
    np.testing.assert_array_equal(now_numpy[1], numpy_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert trainer.rng.bit_generator.state == generator_before
    assert trainer.metrics == metrics_before
    assert trainer.world_model.training and not trainer.world_model.child.training
    assert trainer.world_model.chunk_profile == (3, 7)
    for key, value in model_before.items():
        torch.testing.assert_close(trainer.world_model.state_dict()[key], value, rtol=0., atol=0.)
    assert trainer.optimizers['world'].state_dict() == optimizer_before
    assert all(p.grad is None for p in trainer.world_model.parameters())
    for before, after in zip(cuda_before, torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []):
        assert torch.equal(before, after)


def test_failures_restore_state_and_missing_events_are_explicit():
    trainer = make_trainer()
    trainer.world_model.train()
    before = torch.get_rng_state().clone()
    batch = replace(make_batch(), action_events=None)
    with pytest.raises(ValueError, match='explicit window events'):
        evaluate_world_model(trainer, [batch])
    assert trainer.world_model.training
    assert torch.equal(torch.get_rng_state(), before)
    with pytest.raises(ValueError, match='at least one batch'):
        evaluate_world_model(trainer, [])
    with pytest.raises(ValueError, match='at most three'):
        evaluate_world_model(trainer, [make_batch()], horizon=4)


def test_real_event_world_model_evaluates_three_windows_without_parameter_updates():
    from urban_dreamer.models.world_model import UrbanDreamerWorldModel
    model = UrbanDreamerWorldModel(adjacency=torch.tensor([[0., 1.], [1., 0.]]),
        static_features=torch.ones(2, 4), low_frequency_rank=1, event_protocol=True)
    trainer = make_trainer(model)
    batch = make_batch()
    batch.flow[..., 1] /= 8.
    batch.flow[..., 2] /= 5.
    batch.active_vehicle_fraction.div_(5.)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    actual = evaluate_world_model(trainer, [batch], n_samples=2, horizon=3)
    assert all(np.isfinite(value) for value in actual.values())
    assert actual['rollout_90s_speed_mae_normalized_count'] == 2
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0., atol=0.)


def test_endpoint_codec_diagnostics_and_signed_volume_bias_use_same_targets_and_units():
    class BiasedCodecAndDynamics(OracleModel):
        def decode(self, latent):
            prediction = super().decode(latent)
            prediction[..., 2] += .5
            return prediction

        def transition(self, z, action, **kwargs):
            prediction = super().transition(z, action, **kwargs)
            prediction[..., 2] += action
            return prediction

    actual = evaluate_world_model(make_trainer(BiasedCodecAndDynamics()), [make_batch()])
    assert actual['endpoint_reconstruction_30s_occupied_road_volume_mae_normalized'] == .5
    assert actual['rollout_30s_occupied_road_volume_mae_normalized'] == 1.5
    assert actual['endpoint_reconstruction_30s_total_volume_bias_vehicles'] == 7.5
    assert actual['rollout_30s_total_volume_bias_vehicles'] == 12.5
    assert actual['rollout_30s_true_total_volume_mean_vehicles'] == 10.
    assert actual['rollout_30s_pred_total_volume_mean_vehicles'] == 22.5
    assert actual['persistence_30s_total_volume_bias_vehicles'] == -5.
    assert actual['endpoint_reconstruction_90s_true_total_volume_mean_vehicles'] == 20.
    assert actual['rollout_90s_true_total_volume_mean_vehicles'] == 20.
    assert actual['endpoint_reconstruction_90s_speed_mae_normalized'] == 0.


def test_negative_raw_speed_fraction_counts_samples_before_averaging_without_labels():
    class MixedSampleSpeeds(OracleModel):
        def transition(self, z, action, **kwargs):
            result = super().transition(z, action, **kwargs)
            result[..., 1] = torch.tensor([-3., 3.])[:, None]
            return result

    actual = evaluate_world_model(make_trainer(MixedSampleSpeeds()), [make_batch(frames=2)], horizon=1)
    # Decoded sample speeds are -2 and 4, whose average is positive. The
    # diagnostic still identifies half the samples as physically negative.
    assert actual['rollout_30s_negative_raw_speed_fraction'] == .5
    assert actual['rollout_30s_negative_raw_speed_fraction_count'] == 4
    assert actual['rollout_30s_speed_mae_normalized_count'] == 0


def test_cohort_forecasts_scope_events_volume_and_exposure_but_codec_reconstructs_full_traffic():
    class RecordingModel(OracleModel):
        def __init__(self):
            super().__init__()
            self.encoded_flows = []
            self.actions = []

        def encode(self, flow):
            self.encoded_flows.append(flow.detach().clone())
            return super().encode(flow)

        def transition(self, z, action, **kwargs):
            self.actions.append(action.detach().clone())
            return super().transition(z, action, **kwargs)

    batch = make_batch(frames=4)
    batch.flow[..., 2] = 0.
    batch.flow[0, :2, 0, 2] = .2
    batch.flow[0, 1:, 1, 2] = .1
    def v(name, edge):
        return dict(vehicle_id=name, current_edge=edge, departed=True, arrived=False, in_junction=False)
    windows = [[(ActionEvent('old', 0, 5., 0, 1, 1), ActionEvent('baby', 1, 20., 0, 1, 1)),
                (ActionEvent('baby', 2, 35., 1, 0, 1),), ()]]
    batch = replace(batch, boundary_vehicles=(((v('old', 0),), (v('old', 0), v('baby', 1)),
                                               (v('baby', 1),), (v('baby', 1),)),),
                    arrival_times=({'old': 45.},), total_vehicle_count=torch.tensor([2]),
                    action_events=pack_action_events(windows))
    trainer = make_trainer(RecordingModel())
    trainer.config.method.transition_volume_scope = 'initial_cohort'
    actual = evaluate_world_model(trainer, [batch], horizon=2)
    raw_encoded, cohort_encoded = trainer.world_model.encoded_flows
    torch.testing.assert_close(raw_encoded[..., 2], batch.flow.reshape(-1, 2, 8)[..., 2])
    assert cohort_encoded[1, 1, 2] == 0.  # Baby road volume excluded at t30.
    assert cohort_encoded[2:, :, 2].eq(0).all()  # Old cohort arrived at t45.
    assert actual['rollout_30s_true_total_volume_mean_vehicles'] == pytest.approx(1.)
    assert actual['rollout_60s_true_total_volume_mean_vehicles'] == 0.
    assert actual['reconstruction_true_total_volume_mean_vehicles'] == pytest.approx(4 / 3)
    assert actual['endpoint_reconstruction_30s_true_total_volume_mean_vehicles'] == pytest.approx(1.)
    assert trainer.world_model.calls[0][0].vehicle_ids == ('old', 'old')
    assert trainer.world_model.calls[1][0].vehicle_ids == ()
    torch.testing.assert_close(trainer.world_model.actions[0], torch.tensor([[0., .5], [0., .5]]))
    assert trainer.world_model.actions[1].eq(0).all()
    # Fraction target is 30/(30*N)=.5, then 15/(30*N)=.25; the
    # persistence estimate remains initial observed fraction .5.
    assert actual['persistence_30s_active_fraction_mae'] == 0.
    assert actual['persistence_60s_active_fraction_mae'] == .25
    torch.testing.assert_close(cohort_encoded[..., 1], raw_encoded[..., 1])


def test_cohort_evaluation_requires_boundary_metadata():
    trainer = make_trainer()
    trainer.config.method.transition_volume_scope = 'initial_cohort'
    with pytest.raises(ValueError, match='boundary snapshots'):
        evaluate_world_model(trainer, [make_batch()])


def make_birth_windows():
    batch = make_batch(frames=4)
    batch.flow[..., 2] = 0.
    batch.flow[0, :, 0, 2] = torch.tensor([.2, .2, .4, .4])
    batch.flow[0, 1:, 1, 2] = .1
    def vehicle(name, edge):
        return dict(vehicle_id=name, current_edge=edge, departed=True, arrived=False, in_junction=False)
    a, b, c = vehicle('a', 0), vehicle('b', 1), vehicle('c', 0)
    events = pack_action_events([[
        (ActionEvent('a', 0, 5., 0, 1, 1), ActionEvent('b', 1, 20., 0, 1, 1)),
        (ActionEvent('b', 2, 35., 1, 0, 1), ActionEvent('c', 3, 40., 1, 0, 1)),
        (ActionEvent('c', 4, 65., 0, 1, 1),),
    ]])
    context = batch.context.clone()
    context[0, :, 1] = torch.tensor([1 / 3, 2 / 3, 1.])
    return replace(batch, boundary_vehicles=(((a,), (a, b), (a, b, c), (a, b, c)),),
        total_vehicle_count=torch.tensor([3]), action_events=events, context=context,
        active_vehicle_fraction=torch.tensor([[.6, .8, 1.]]))


def test_independent_windows_restore_previous_births_and_preserve_actions_and_exposure():
    class RecordingModel(OracleModel):
        def __init__(self):
            super().__init__()
            self.starts, self.actions, self.readout_starts = [], [], []

        def transition(self, z, action, **kwargs):
            self.starts.append(z.detach().clone())
            self.actions.append(action.detach().clone())
            return super().transition(z, action, **kwargs)

        def active_vehicle_fraction_readout(self, current, *, z_prev, **kwargs):
            self.readout_starts.append(z_prev.detach().clone())
            return current.new_zeros(current.shape[0])

    trainer, batch = make_trainer(RecordingModel()), make_birth_windows()
    trainer.config.method.transition_volume_scope = 'previous_state_excluding_current_births'
    original_flow, original_action = batch.flow.clone(), batch.action.clone()
    original_fraction = batch.active_vehicle_fraction.clone()
    actual = evaluate_world_model(trainer, [batch], n_samples=2, horizon=3)
    for step, expected_count in enumerate((1., 2., 3.)):
        window = f'window_{30 * step}'
        assert actual[window + '_forecast30_true_total_volume_mean_vehicles'] == pytest.approx(expected_count)
        assert actual[window + '_endpoint_reconstruction_volume_mae_vehicles'] == 0.
        assert actual[window + '_forecast30_active_fraction_mae'] == pytest.approx(float(original_fraction[0, step]))
        assert actual[window + '_persistence_active_fraction_mae'] == pytest.approx(
            abs(float(batch.context[0, step, 1] - original_fraction[0, step])))
        torch.testing.assert_close(trainer.world_model.starts[step][..., :8], original_flow[:, step].repeat(2, 1, 1))
        torch.testing.assert_close(trainer.world_model.readout_starts[step], trainer.world_model.starts[step])
        torch.testing.assert_close(trainer.world_model.actions[step], original_action[:, step].repeat(2, 1))
        expected_events = batch.action_events.for_step(step)
        assert trainer.world_model.calls[step][0].vehicle_ids == expected_events.vehicle_ids * 2
    # B is absent only from the first endpoint label, and is present in the
    # next observed input and endpoint. C is restored in the following window.
    assert trainer.world_model.starts[1][0, 1, 2] == pytest.approx(.1)
    assert trainer.world_model.starts[2][0, 0, 2] == pytest.approx(.4)
    torch.testing.assert_close(batch.flow, original_flow)
    torch.testing.assert_close(batch.action, original_action)
    torch.testing.assert_close(batch.active_vehicle_fraction, original_fraction)
    assert actual['supervision'] == 'independent_observed_start_windows'
    assert actual['forecast_window_seconds'] == 30
    assert not any(name.startswith('rollout_') or '90s' in name for name in actual)


def test_independent_window_speed_requires_real_following_boundary_and_explicit_metadata():
    trainer = make_trainer()
    trainer.config.method.transition_volume_scope = 'previous_state_excluding_current_births'
    actual = evaluate_world_model(trainer, [make_birth_windows()], horizon=3)
    assert actual['window_30_forecast30_speed_mae_normalized_count'] == 2
    assert actual['window_60_forecast30_speed_mae_normalized_count'] == 0
    assert actual['window_60_endpoint_reconstruction_speed_mae_normalized_count'] == 0
    assert actual['window_60_forecast30_volume_mae_normalized_count'] == 2
    with pytest.raises(ValueError, match='boundary snapshots'):
        evaluate_world_model(trainer, [make_batch()])
