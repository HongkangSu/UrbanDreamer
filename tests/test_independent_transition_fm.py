from dataclasses import replace
import copy

import pytest
import torch

from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
from urban_dreamer.models.world_model import Dynamics, independent_transition_fm_loss


@pytest.fixture
def case():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    adjacency = torch.ones(3, 3) - torch.eye(3)
    dynamics = Dynamics(adjacency=adjacency, static_features=torch.ones(3, 4), event_protocol=True)
    dynamics.eval()
    # The production head starts at zero, which deliberately blocks condition
    # gradients until its first update. Exercise the learned-head regime here.
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.01)
    torch.nn.init.normal_(dynamics.graph_prior.output.weight, std=.01)
    inputs, targets = torch.randn(2, 3, 3, 64), torch.randn(2, 3, 3, 64)
    windows = [[(ActionEvent(f'b{batch}-t{step}', batch * 3 + step,
                             step * 30. + 2., step % 3, (step + 1) % 3, 2),)
                for step in range(3)] for batch in range(2)]
    events = pack_action_events(windows)
    action = torch.rand(2, 3, 3)
    noise = torch.randn(2, 3, dynamics.graph_prior.rank, 64)
    tau = torch.tensor([[.2, .4, .6], [.3, .5, .7]])
    return dynamics, inputs, targets, action, events, noise, tau


@pytest.mark.parametrize('solver', ['euler', 'picard'])
def test_flattened_windows_match_separate_transitions_with_identical_noise(case, solver):
    dynamics, inputs, targets, action, events, noise, tau = case
    loss, metrics, prediction = independent_transition_fm_loss(dynamics,
        z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        noise=noise, flow_time=tau, steps=2, solver=solver, picard_iterations=2)
    losses, flow_losses, prior_losses = [], [], []
    expected = torch.empty_like(prediction)
    for batch in range(2):
        for step in range(3):
            selected = events.select((events.batch_index == batch) & (events.time_index == step))
            selected = replace(selected, batch_index=torch.zeros_like(selected.batch_index),
                               time_index=torch.zeros_like(selected.time_index), batch_size=1, time_steps=1)
            value, detail, endpoint = independent_transition_fm_loss(dynamics,
                z_inputs=inputs[batch:batch+1, step:step+1],
                z_targets=targets[batch:batch+1, step:step+1],
                action=action[batch:batch+1, step:step+1], action_events=selected,
                noise=noise[batch:batch+1, step:step+1], flow_time=tau[batch:batch+1, step:step+1],
                steps=2, solver=solver, picard_iterations=2)
            losses.append(value)
            flow_losses.append(detail['flow_mse'])
            prior_losses.append(detail['prior_mse'])
            expected[batch:batch+1, step:step+1] = endpoint
    torch.testing.assert_close(loss, torch.stack(losses).mean(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(metrics['flow_mse'], torch.stack(flow_losses).mean(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(metrics['prior_mse'], torch.stack(prior_losses).mean(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(prediction, expected, atol=2e-6, rtol=1e-5)
    assert not prediction.requires_grad


def test_each_window_condition_is_encoded_once_and_events_keep_their_window(case, monkeypatch):
    dynamics, inputs, targets, action, events, noise, tau = case
    condition_calls, solve_grad_modes = [], []
    encode, solve = dynamics.encode_condition, dynamics.solve
    def capture_encode(value, *, action_events):
        condition_calls.append((value.detach().clone(), action_events))
        return encode(value, action_events=action_events)
    def capture_solve(*args, **kwargs):
        solve_grad_modes.append(torch.is_grad_enabled())
        return solve(*args, **kwargs)
    monkeypatch.setattr(dynamics, 'encode_condition', capture_encode)
    monkeypatch.setattr(dynamics, 'solve', capture_solve)
    loss, _, prediction = independent_transition_fm_loss(dynamics,
        z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        noise=noise, flow_time=tau)
    assert len(condition_calls) == 1
    flat_action, flat_events = condition_calls[0]
    torch.testing.assert_close(flat_action, action.reshape(6, 1, 3))
    assert flat_events.batch_size == 6 and flat_events.time_steps == 1
    torch.testing.assert_close(flat_events.batch_index, events.batch_index * 3 + events.time_index)
    assert flat_events.time_index.eq(0).all()
    assert flat_events.vehicle_ids == events.vehicle_ids
    assert solve_grad_modes == [False]
    assert not prediction.requires_grad
    loss.backward()
    for parameter in (dynamics.flow_head.weight,
                      dynamics.event_encoder.edge_embedding.weight,
                      dynamics.event_encoder.event_mlp[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_future_inputs_targets_and_actions_do_not_change_earlier_predictions(case):
    dynamics, inputs, targets, action, events, noise, tau = case
    _, _, before = independent_transition_fm_loss(dynamics,
        z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        noise=noise, flow_time=tau)
    future_inputs, future_targets, future_actions = inputs.clone(), targets.clone(), action.clone()
    future_inputs[:, 1:] += 100.
    future_targets[:, 1:] -= 100.
    future_actions[:, 1:] *= 30.
    future_events = replace(events, destination_edge=torch.where(events.time_index > 0,
        torch.zeros_like(events.destination_edge), events.destination_edge))
    _, _, after = independent_transition_fm_loss(dynamics,
        z_inputs=future_inputs, z_targets=future_targets, action=future_actions,
        action_events=future_events, noise=noise, flow_time=tau)
    torch.testing.assert_close(before[:, 0], after[:, 0], atol=0., rtol=0.)
    assert not torch.allclose(before[:, 1:], after[:, 1:])


def test_frozen_codec_inputs_and_endpoint_targets_receive_no_gradient(case):
    dynamics, inputs, targets, action, events, noise, tau = case
    inputs.requires_grad_(True)
    targets.requires_grad_(True)
    loss, _, prediction = independent_transition_fm_loss(dynamics,
        z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        noise=noise, flow_time=tau)
    loss.backward()
    assert inputs.grad is None and targets.grad is None
    assert not prediction.requires_grad
    assert dynamics.flow_head.weight.grad is not None


def test_disabling_endpoint_solve_preserves_exact_fm_loss_and_parameter_gradients(case, monkeypatch):
    original, inputs, targets, action, events, noise, tau = case
    with_solve, without_solve = copy.deepcopy(original), copy.deepcopy(original)
    targets.requires_grad_(True)
    solved_loss, solved_metrics, _ = independent_transition_fm_loss(with_solve,
        z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        noise=noise, flow_time=tau, predict_endpoints=True)
    solved_loss.backward()
    solve_calls = []
    def forbidden_solve(*args, **kwargs):
        solve_calls.append(True)
        pytest.fail('posterior readout mode must not run an endpoint ODE solve')
    monkeypatch.setattr(without_solve, 'solve', forbidden_solve)
    posterior_loss, posterior_metrics, posterior = independent_transition_fm_loss(without_solve,
        z_inputs=inputs, z_targets=targets, action=action, action_events=events,
        noise=noise, flow_time=tau, predict_endpoints=False)
    posterior_loss.backward()
    assert solve_calls == []
    torch.testing.assert_close(posterior, targets.detach(), rtol=0., atol=0.)
    assert not posterior.requires_grad and targets.grad is None
    torch.testing.assert_close(solved_loss, posterior_loss, rtol=0., atol=0.)
    for key in solved_metrics:
        torch.testing.assert_close(solved_metrics[key], posterior_metrics[key], rtol=0., atol=0.)
    for name, parameter in with_solve.named_parameters():
        other = dict(without_solve.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0., atol=0., msg=name)


def test_shape_and_flow_time_validation_reject_misaligned_transitions(case):
    dynamics, inputs, targets, action, events, noise, tau = case
    with pytest.raises(ValueError):
        independent_transition_fm_loss(dynamics, z_inputs=inputs, z_targets=targets[:, :2],
            action=action, action_events=events)
    with pytest.raises(ValueError):
        independent_transition_fm_loss(dynamics, z_inputs=inputs, z_targets=targets,
            action=action, action_events=events, flow_time=torch.full_like(tau, float('nan')))
    with pytest.raises(ValueError, match='action_events'):
        independent_transition_fm_loss(dynamics, z_inputs=inputs, z_targets=targets, action=action)


def test_fast_pretraining_profile_matches_legacy_loss_endpoints_and_parameter_gradients(case):
    original, inputs, targets, action, events, noise, tau = case
    legacy, fast = copy.deepcopy(original), copy.deepcopy(original)
    legacy.set_initial_chunk_profile()
    fast.set_pretraining_execution_profile(space_chunk=32, time_chunk=8192,
        mlp_chunk=8192, activation_checkpointing=False)
    # Thirty-two independent windows expose the legacy attention chunk loops,
    # while keeping this numerical-equivalence check inexpensive on CPU.
    inputs = inputs[:1, :1].expand(32, -1, -1, -1).clone()
    targets = targets[:1, :1].expand_as(inputs).clone()
    action = action[:1, :1].expand(32, -1, -1).clone()
    noise = noise[:1, :1].expand(32, -1, -1, -1).clone()
    tau = tau[:1, :1].expand(32, -1).clone()
    window = (ActionEvent('v', 0, 2., 0, 1, 2),)
    events = pack_action_events([[window] for _ in range(32)])
    results = []
    for dynamics in (legacy, fast):
        loss, metrics, prediction = independent_transition_fm_loss(dynamics,
            z_inputs=inputs, z_targets=targets, action=action, action_events=events,
            noise=noise, flow_time=tau, steps=2)
        loss.backward()
        results.append((loss.detach(), metrics, prediction))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(results[0][2], results[1][2], rtol=1e-5, atol=2e-6)
    for name, parameter in legacy.named_parameters():
        fast_parameter = dict(fast.named_parameters())[name]
        if parameter.grad is None:
            assert fast_parameter.grad is None, name
        else:
            assert fast_parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, fast_parameter.grad,
                rtol=2e-4, atol=2e-6, msg=name)


def test_execution_profile_updates_every_layer_and_legacy_profiles_restore_checkpointing(case):
    dynamics = case[0]
    assert dynamics.transformer.activation_checkpointing is True
    dynamics.set_pretraining_execution_profile(space_chunk=32, time_chunk=8192,
        mlp_chunk=4096, activation_checkpointing=False)
    assert dynamics.transformer.activation_checkpointing is False
    for layer in dynamics.transformer.layers:
        assert layer.space.attention_batch_chunk_size == 32
        assert layer.time.attention_batch_chunk_size == 8192
        assert layer.mlp.token_chunk_size == 4096
    for restore in (dynamics.set_initial_chunk_profile, dynamics.set_online_chunk_profile):
        dynamics.set_pretraining_execution_profile(space_chunk=32, time_chunk=8192,
            mlp_chunk=4096, activation_checkpointing=False)
        restore()
        assert dynamics.transformer.activation_checkpointing is True


@pytest.mark.parametrize('name,value', [('space_chunk', 0), ('space_chunk', -1),
    ('time_chunk', True), ('time_chunk', 2.5), ('mlp_chunk', -100),
    ('activation_checkpointing', 1), ('activation_checkpointing', 'false')])
def test_execution_profile_rejects_invalid_values_before_changing_state(case, name, value):
    dynamics = case[0]
    profile = dict(space_chunk=32, time_chunk=8192, mlp_chunk=4096, activation_checkpointing=False)
    before = [(layer.space.attention_batch_chunk_size, layer.time.attention_batch_chunk_size,
               layer.mlp.token_chunk_size) for layer in dynamics.transformer.layers]
    profile[name] = value
    with pytest.raises(ValueError):
        dynamics.set_pretraining_execution_profile(**profile)
    assert dynamics.transformer.activation_checkpointing is True
    after = [(layer.space.attention_batch_chunk_size, layer.time.attention_batch_chunk_size,
              layer.mlp.token_chunk_size) for layer in dynamics.transformer.layers]
    assert before == after
