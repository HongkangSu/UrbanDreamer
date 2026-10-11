from __future__ import annotations

import importlib.util
import copy
from pathlib import Path

import pytest
import torch


def _load_world_model_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "urban_dreamer"
        / "models"
        / "world_model.py"
    )
    spec = importlib.util.spec_from_file_location("urban_dreamer_world_model_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


wm = _load_world_model_module()
ActiveVehicleFractionReadout = wm.ActiveVehicleFractionReadout
DynamicsOutput = wm.DynamicsOutput
UrbanDreamerWorldModel = wm.UrbanDreamerWorldModel
cfm_loss = wm.cfm_loss
decoder_bpr_supervision_loss = wm.decoder_bpr_supervision_loss
decoder_reconstruction_loss = wm.decoder_reconstruction_loss
episode_t_all_supervision_loss = wm.episode_t_all_supervision_loss
transition_decoded_supervision_loss = wm.transition_decoded_supervision_loss


def _reference_attention(module, value, *, causal=False, all_true_mask=False):
    batch, length, width = value.shape
    q, k, v = module.qkv(value).chunk(3, dim=-1)
    q, k, v = [item.reshape(batch, length, module.n_heads, module.head_dim).transpose(1, 2)
               for item in (q, k, v)]
    mask = torch.ones(batch, 1, length, length, dtype=torch.bool) if all_true_mask else None
    result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
        dropout_p=0., is_causal=causal)
    return module.output(result.transpose(1, 2).contiguous().reshape(batch, length, width))


@pytest.mark.parametrize('chunk', [0, 2])
def test_space_attention_matches_explicit_all_true_sdpa_outputs_and_gradients(chunk):
    torch.set_num_threads(1)
    torch.manual_seed(124)
    attention = wm.SpaceSelfAttention(7)
    attention.attention_batch_chunk_size = chunk
    reference = copy.deepcopy(attention.attention)
    x = torch.randn(2, 3, 7, wm.WORLD_MODEL_DIM, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)
    actual = attention(x)
    expected = _reference_attention(reference, x_reference.reshape(6, 7, wm.WORLD_MODEL_DIM),
                                    all_true_mask=True).reshape_as(actual)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=2e-5, atol=2e-6)
    for name, parameter in attention.attention.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                   rtol=2e-5, atol=2e-5, msg=name)


@pytest.mark.parametrize('time_steps', [1, 2])
def test_time_attention_matches_full_causal_sdpa_and_all_gradients(time_steps):
    torch.set_num_threads(1)
    torch.manual_seed(125)
    attention = wm.TimeSelfAttention()
    attention.attention_batch_chunk_size = 3
    reference = copy.deepcopy(attention.attention)
    x = torch.randn(2, time_steps, 5, wm.WORLD_MODEL_DIM, requires_grad=True)
    x_reference = x.detach().clone().requires_grad_(True)
    actual = attention(x)
    flat = x_reference.permute(0, 2, 1, 3).reshape(10, time_steps, wm.WORLD_MODEL_DIM)
    expected = _reference_attention(reference, flat, causal=True).reshape(
        2, 5, time_steps, wm.WORLD_MODEL_DIM).permute(0, 2, 1, 3)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(x.grad, x_reference.grad, rtol=2e-5, atol=2e-6)
    for name, parameter in attention.attention.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                   rtol=2e-5, atol=2e-5, msg=name)
    if time_steps == 1:
        assert attention.attention.qkv.weight.grad[:2 * wm.WORLD_MODEL_DIM].eq(0).all()
        assert attention.attention.qkv.bias.grad[:2 * wm.WORLD_MODEL_DIM].eq(0).all()
    else:
        changed = x.detach().clone()
        changed[:, 1] += 100.
        torch.testing.assert_close(attention(changed)[:, 0], actual.detach()[:, 0], rtol=0., atol=0.)


def test_event_aggregation_under_cpu_bfloat16_preserves_integer_counts_and_fp32_additive_features():
    from urban_dreamer.learning.action_events import ActionEvent, pack_action_events
    adjacency = torch.ones(3, 3) - torch.eye(3)
    encoder = wm.EventActionEncoder(adjacency=adjacency, static_features=torch.ones(3, 4))
    events = pack_action_events([[(ActionEvent('a', 0, 2., 0, 1, 2),
                                  ActionEvent('b', 1, 2., 0, 1, 2))]])
    expected_counts = encoder.aggregate(events).count
    with torch.autocast('cpu', dtype=torch.bfloat16):
        aggregate = encoder.aggregate(events)
        output = encoder.readout(aggregate)
        loss = output.float().square().mean() + aggregate.feature_sum.square().mean()
    assert aggregate.count.dtype == torch.long
    assert aggregate.feature_sum.dtype == torch.float32
    assert torch.equal(aggregate.count, expected_counts)
    loss.backward()
    for parameter in (encoder.edge_embedding.weight, encoder.event_mlp[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


class ConstantVelocityDynamics(torch.nn.Module):
    def __init__(self, value: float, *, edge_count: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(
        self,
        action: torch.Tensor,
        *,
        packed_latents: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> DynamicsOutput:
        del action, flow_time
        velocity = torch.ones_like(packed_latents) * self.weight
        return DynamicsOutput(latent_velocity=velocity)


class IdentityDecodeProjector(torch.nn.Module):
    def decode(self, z_pred_next: torch.Tensor) -> torch.Tensor:
        return z_pred_next


def _build_graph(edge_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    adjacency = torch.eye(edge_count, dtype=torch.float32)
    static_features = torch.linspace(0.1, 0.4, steps=edge_count * 4, dtype=torch.float32).view(edge_count, 4)
    return adjacency, static_features


def test_cfm_loss_vectorizes_all_explicit_transitions() -> None:
    dynamics = ConstantVelocityDynamics(0.0, edge_count=2)
    z_sequence = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 2, 64).clone()
    action = torch.zeros((1, 2, 2), dtype=torch.float32)

    loss, aux = cfm_loss(
        dynamics,
        z_sequence=z_sequence,
        action=action,
    )
    loss.backward()

    assert torch.isclose(loss, torch.tensor(4.0))
    assert torch.isclose(aux["flow_mse"], torch.tensor(4.0))
    assert torch.isclose(aux["weighted_flow_mse"], torch.tensor(4.0))
    assert dynamics.weight.grad is not None
    assert torch.isclose(dynamics.weight.grad, torch.tensor(-4.0))


def test_fixed_loss_helpers_preserve_fixed_public_weights() -> None:
    projector = IdentityDecodeProjector()
    decoded_dyn = torch.tensor([[[[0.0, 2.0, 1.0, 0.0]]]], dtype=torch.float32)
    flow_targets = torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]], dtype=torch.float32)

    recon_loss, recon_aux = decoder_reconstruction_loss(
        decoded_dyn=decoded_dyn,
        flow_targets=flow_targets,
    )
    torch.testing.assert_close(recon_loss, 0.2 * recon_aux["decoder_recon_weighted_mse"])

    transition_loss, transition_aux = transition_decoded_supervision_loss(
        z_pred_next=decoded_dyn,
        flow_projector=projector,  # type: ignore[arg-type]
        flow_target_next=flow_targets,
    )
    torch.testing.assert_close(
        transition_loss,
        transition_aux["pred_next_decoded_weighted_mse"],
    )

    bpr_loss, bpr_aux = decoder_bpr_supervision_loss(
        decoded_dyn=decoded_dyn,
        flow_targets=flow_targets,
        lengths=torch.tensor([2.0]),
        speed_limits=torch.tensor([1.0]),
        capacities=torch.tensor([10.0]),
    )
    torch.testing.assert_close(bpr_loss, 0.1 * bpr_aux["bpr_weighted_mse"])

    adjacency, _ = _build_graph(1)
    flow_projector = wm.FlowProjector(adjacency=adjacency)
    z_tokens = torch.randn(2, 2, 1, 64)
    reward_targets = torch.tensor([[-12.0, -10.0], [-15.0, -11.0]], dtype=torch.float32)
    t_all_loss, t_all_aux = episode_t_all_supervision_loss(
        flow_projector=flow_projector,
        z_tokens=z_tokens,
        reward_targets=reward_targets,
    )
    torch.testing.assert_close(t_all_loss, 0.2 * t_all_aux["t_all_pred_mse"])


def test_active_vehicle_readout_is_anchored_to_decoder_fraction() -> None:
    readout = ActiveVehicleFractionReadout()
    z_prev = torch.randn(2, 3, 64)
    z_next = torch.randn(2, 3, 64)

    prediction = readout(
        z_next,
        z_prev=z_prev,
        raw_decoder_fraction=torch.tensor([0.25, 0.75]),
        transition_seconds=torch.tensor([30.0, 45.0]),
    )

    assert torch.allclose(prediction, torch.tensor([0.25, 0.75]), atol=1e-6, rtol=0.0)


def test_world_model_fixed_architecture_chunk_profiles_and_nonpersistent_buffers() -> None:
    adjacency, static_features = _build_graph(2)
    model = UrbanDreamerWorldModel(adjacency=adjacency, static_features=static_features)

    assert model.flow_projector.input_proj.in_features == 8
    assert model.flow_projector.input_proj.out_features == 64
    assert len(model.flow_projector.gnn_layers) == 2
    assert isinstance(model.dynamics.action_encoder, wm.ActionGNNEncoder)
    assert len(model.dynamics.action_encoder.action_gnn) == 2
    assert model.dynamics.action_encoder.edge_conditioning.in_features == wm.ACTION_EDGE_DIM
    assert len(model.dynamics.transformer.layers) == 6
    assert tuple(model.dynamics.register_tokens.shape) == (4, 256)
    assert set(wm.__all__) == {
        "UrbanDreamerWorldModel",
        "cfm_loss",
        "cfm_predict_one_timestep",
        "recursive_fm_loss",
        "independent_transition_fm_loss",
        "decoder_bpr_supervision_loss",
        "decoder_reconstruction_loss",
        "episode_t_all_supervision_loss",
        "transition_decoded_supervision_loss",
    }

    keys = set(model.state_dict().keys())
    assert all("hdv" not in key and "depart" not in key for key in keys)
    assert "flow_projector.adjacency" not in keys
    assert "dynamics.action_encoder.adjacency" not in keys
    assert "dynamics.action_encoder.static_features" not in keys
    assert "dynamics.graph_prior.basis" in keys
    assert "dynamics.graph_prior.adjacency" not in keys

    first_layer = model.dynamics.transformer.layers[0]
    assert first_layer.space.attention_batch_chunk_size == 1
    assert first_layer.time.attention_batch_chunk_size == 256
    assert first_layer.mlp.token_chunk_size == 4096
    model.set_online_chunk_profile()
    assert first_layer.space.attention_batch_chunk_size == 16
    assert first_layer.time.attention_batch_chunk_size == 32768
    assert first_layer.mlp.token_chunk_size == 16384

    flow_features = torch.randn(1, 1, 2, 8, dtype=torch.float32)
    latent = model.encode(flow_features)
    decoded = model.decode(latent)
    output = model.dynamics(
        torch.zeros((1, 1, 2), dtype=torch.float32),
        packed_latents=latent,
        flow_time=torch.tensor([[0.25]], dtype=torch.float32),
    )

    assert tuple(latent.shape) == (1, 1, 2, 64)
    assert tuple(decoded.shape) == (1, 1, 2, 4)
    assert tuple(output.latent_velocity.shape) == (1, 1, 2, 64)
    assert tuple(model.flow_projector.predict_t_all_from_z(latent[:, 0]).shape) == (1,)


def test_cfm_conditions_each_source_target_pair_without_offset() -> None:
    class ConditionedDynamics(torch.nn.Module):
        def forward(self, action, *, packed_latents, flow_time):
            assert torch.equal(action[0, :, 0], torch.tensor([2., 5.]))
            return DynamicsOutput(action.unsqueeze(-1).expand_as(packed_latents))

    states = torch.tensor([0., 2., 7.]).view(1, 3, 1, 1).expand(1, 3, 1, 64)
    loss, _ = cfm_loss(
        ConditionedDynamics(), z_sequence=states,
        action=torch.tensor([[[2.], [5.]]]),
    )
    assert loss == 0


def test_active_fraction_readout_trains_on_interval_exposure() -> None:
    readout = ActiveVehicleFractionReadout()
    latent = torch.zeros(1, 2, 64)
    optimizer = torch.optim.SGD(readout.parameters(), lr=0.1)
    target = torch.tensor([0.75])  # 45 active vehicle-seconds / (2 vehicles * 30s).
    before = readout(latent, z_prev=latent, raw_decoder_fraction=0.25, transition_seconds=30.)
    loss = torch.nn.functional.mse_loss(before, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = readout(latent, z_prev=latent, raw_decoder_fraction=0.25, transition_seconds=30.)
    assert torch.abs(after - target) < torch.abs(before - target)
    assert (-after * 30. / 60.) < (-before * 30. / 60.)


def test_missing_window_speed_label_has_zero_gradient_but_other_fields_train():
    prediction = torch.tensor([[[[.3, .5, .4, .3], [.3, .5, .4, .3]]]], requires_grad=True)
    target = torch.tensor([[[[0., .8, 0., 0.], [0., 100., 0., 0.]]]])
    mask = torch.tensor([[[True, False]]])
    loss, _ = decoder_reconstruction_loss(decoded_dyn=prediction, flow_targets=target,
                                           speed_target_mask=mask)
    loss.backward()
    assert prediction.grad[0, 0, 0, 1] != 0
    assert prediction.grad[0, 0, 1, 1] == 0
    assert prediction.grad[0, 0, 1, 0] != 0
    changed = target.clone()
    changed[0, 0, 1, 1] = .1
    other, _ = decoder_reconstruction_loss(decoded_dyn=prediction, flow_targets=changed,
                                            speed_target_mask=mask)
    torch.testing.assert_close(loss, other)
    transition, _ = transition_decoded_supervision_loss(z_pred_next=prediction,
        flow_projector=IdentityDecodeProjector(), flow_target_next=target, speed_target_mask=mask)
    torch.testing.assert_close(transition, loss / wm.DECODER_RECON_WEIGHT)


def test_cfm_prediction_uses_only_action_and_latents():
    past = torch.zeros(1, 2, 3, 64)
    predicted = wm.cfm_predict_one_timestep(
        ConstantVelocityDynamics(2., edge_count=3),
        past_latents=past,
        action=torch.zeros(1, 1, 3),
    )
    torch.testing.assert_close(predicted, torch.full((1, 1, 3, 64), 2.))


def _small_dynamics(*, noise_scale=.1):
    adjacency = torch.tensor([[0., 1., 0.], [0., 0., 1.], [0., 0., 0.]])
    return wm.Dynamics(adjacency=adjacency, static_features=torch.zeros(3, 4),
                       low_frequency_rank=1, prior_noise_scale=noise_scale)


def test_prior_projector_is_orthogonal_low_spectrum_and_directional():
    dynamics = _small_dynamics()
    prior = dynamics.graph_prior
    basis = prior.basis
    projector = basis @ basis.T
    torch.testing.assert_close(projector, projector.T)
    torch.testing.assert_close(projector @ projector, projector, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(basis.T @ basis, torch.eye(1), atol=1e-6, rtol=1e-6)
    expected_zero_mode = torch.tensor([1., 2.**.5, 1.])
    torch.testing.assert_close(projector @ expected_zero_mode, expected_zero_mode, atol=1e-6, rtol=1e-6)
    assert not torch.equal(prior.adjacency, prior.adjacency.T)
    with torch.no_grad():
        prior.output.weight.copy_(torch.eye(64))
    z = torch.randn(2, 1, 3, 64)
    conditions = dynamics.encode_condition(torch.zeros(2, 1, 3))
    increment = prior(z, conditions)
    torch.testing.assert_close(increment, prior.project(increment), atol=1e-6, rtol=1e-6)


def test_prior_noise_has_graph_covariance_and_no_high_frequency_component():
    torch.manual_seed(713)
    prior = _small_dynamics(noise_scale=.3).graph_prior
    z = torch.zeros(12000, 1, 3, 64)
    perturbation = prior.sample_noise(z)
    torch.testing.assert_close(perturbation, prior.project(perturbation), atol=1e-6, rtol=1e-6)
    coordinate_noise = perturbation[:, 0, :, 0]
    covariance = coordinate_noise.T @ coordinate_noise / coordinate_noise.shape[0]
    torch.testing.assert_close(covariance, .3**2 * prior.basis @ prior.basis.T, atol=.002, rtol=.04)
    assert abs(coordinate_noise.mean()) < .005


def test_euler_and_converged_batched_picard_are_identical(monkeypatch):
    dynamics = _small_dynamics(noise_scale=0.)
    batch_shapes = []

    def field(action, *, packed_latents, flow_time, condition=None):
        batch_shapes.append(tuple(packed_latents.shape))
        return DynamicsOutput(.2 * packed_latents + flow_time[..., None, None])

    monkeypatch.setattr(dynamics, "forward", field)
    z = torch.randn(2, 3, 64)
    actions = torch.zeros(2, 3)
    euler = dynamics.transition(z, actions, steps=4)
    picard = dynamics.transition(z, actions, steps=4, solver="picard", picard_iterations=4)
    torch.testing.assert_close(euler, picard, atol=1e-6, rtol=1e-6)
    assert batch_shapes == [(2, 1, 3, 64)] * 4 + [(8, 1, 3, 64)] * 4


def test_solve_caches_only_invariant_conditions_and_refreshes_between_solves():
    dynamics = _small_dynamics()
    dynamics.transformer = torch.nn.Identity()
    counts = {"encoding": 0, "prior": 0}

    def encoded(*_):
        counts["encoding"] += 1

    def prior(*_):
        counts["prior"] += 1

    dynamics.action_encoder.register_forward_hook(encoded)
    dynamics.graph_prior.register_forward_hook(prior)
    z = torch.zeros(1, 3, 64)
    actions = torch.zeros(1, 3)
    dynamics.transition(z, actions, steps=4)
    assert counts == {"encoding": 1, "prior": 1}
    dynamics.latent_disagreement(z, actions + 1, n_samples=3, steps=4)
    assert counts == {"encoding": 2, "prior": 2}


def test_same_condition_disagreement_uses_unbiased_latent_variance_without_decode():
    dynamics = _small_dynamics(noise_scale=.2)
    dynamics.transformer = torch.nn.Identity()
    z = torch.zeros(1, 3, 64)
    noise = torch.tensor([-1., 0., 1.]).view(3, 1, 1, 1).expand(3, 1, 1, 64)
    samples, global_variance, edge_variance = dynamics.latent_disagreement(
        z, torch.zeros(1, 3), n_samples=3, noise=noise)
    torch.testing.assert_close(edge_variance, samples.var(0, unbiased=True).mean(-1))
    torch.testing.assert_close(global_variance, torch.tensor([.04 / 3]), atol=1e-6, rtol=1e-6)
    identical = noise[:1].expand_as(noise)
    _, zero, _ = dynamics.latent_disagreement(z, torch.zeros(1, 3),
                                             n_samples=3, noise=identical)
    torch.testing.assert_close(zero, torch.zeros_like(zero))
    with pytest.raises(ValueError, match="at least two"):
        dynamics.latent_disagreement(z, torch.zeros(1, 3), n_samples=1)


def test_recursive_fm_preserves_early_solver_gradients_and_detaches_targets(monkeypatch):
    dynamics = _small_dynamics(noise_scale=0.)
    velocity_weight = torch.nn.Parameter(torch.tensor(.2))
    velocities, sources = [], []

    def field(action, *, packed_latents, flow_time, condition=None):
        del flow_time, condition
        sources.append(packed_latents.detach().clone())
        velocity = velocity_weight * packed_latents + action[..., None]
        velocity.retain_grad()
        velocities.append(velocity)
        return DynamicsOutput(velocity)

    monkeypatch.setattr(dynamics, "forward", field)
    sequence = torch.tensor([1., 4., 9.]).view(1, 3, 1, 1).expand(1, 3, 3, 64).clone().requires_grad_()
    loss, metrics, predicted = wm.recursive_fm_loss(dynamics, z_sequence=sequence,
        action=torch.tensor([[[1., 1., 1.], [2., 2., 2.]]]),
        flow_time=torch.full((1, 2), .25), prior_weight=0., depth_weights=torch.tensor([0., 1.]), steps=1)
    loss.backward()
    # Call 0 = step-one FM, call 1 = step-one solve, call 2 = step-two FM.
    assert velocities[1].grad is not None and bool(velocities[1].grad.abs().sum() > 0)
    assert sequence.grad is not None and bool(sequence.grad[:, :1].abs().sum() > 0)
    torch.testing.assert_close(sequence.grad[:, 1:], torch.zeros_like(sequence.grad[:, 1:]))
    torch.testing.assert_close(sources[2], .75 * predicted[:, :1].detach() + .25 * sequence[:, 2:].detach())
    assert tuple(predicted.shape) == (1, 2, 3, 64)
    assert not metrics["flow_mse"].requires_grad
    # The objective is velocity regression. It does not compare solved endpoints.
    expected = (.2 * sources[2] + 2. - (sequence[:, 2:].detach() - predicted[:, :1].detach())).square().mean()
    torch.testing.assert_close(loss, expected)


def test_prior_parameters_belong_to_dynamics_optimizer_and_get_increment_loss_gradients():
    dynamics = _small_dynamics(noise_scale=0.)
    dynamics.transformer = torch.nn.Identity()
    sequence = torch.tensor([0., 1.]).view(1, 2, 1, 1).expand(1, 2, 3, 64)
    loss, _, _ = wm.recursive_fm_loss(dynamics, z_sequence=sequence, action=torch.zeros(1, 1, 3),
                                     steps=1, prior_weight=1.)
    loss.backward()
    assert dynamics.graph_prior.output.weight.grad is not None
    assert dynamics.graph_prior.output.weight.grad.abs().sum() > 0
    assert "graph_prior.output.weight" in dict(dynamics.named_parameters())


def test_checkpoint_preserves_basis_noise_coordinates_and_exact_transition():
    dynamics = _small_dynamics()
    dynamics.transformer = torch.nn.Identity()
    z = torch.randn(1, 3, 64)
    noise = torch.randn(1, 1, 64)
    action = torch.randn(1, 3)
    expected = dynamics.transition(z, action, noise=noise)
    restored = _small_dynamics()
    restored.transformer = torch.nn.Identity()
    with torch.no_grad():
        restored.graph_prior.basis.neg_()
    restored.load_state_dict(dynamics.state_dict())
    torch.testing.assert_close(restored.graph_prior.basis, dynamics.graph_prior.basis, atol=0, rtol=0)
    torch.testing.assert_close(restored.transition(z, action, noise=noise), expected, atol=0, rtol=0)


def test_latent_disagreement_preserves_physical_time_axis_for_batched_sequences():
    dynamics = _small_dynamics()
    dynamics.transformer = torch.nn.Identity()
    z = torch.zeros(2, 2, 3, 64)
    samples, global_variance, edge_variance = dynamics.latent_disagreement(
        z, torch.zeros(2, 2, 3), n_samples=3, solver="picard", steps=2,
        noise=torch.randn(3, 2, 2, 1, 64))
    assert tuple(samples.shape) == (3, 2, 2, 3, 64)
    assert tuple(global_variance.shape) == (2, 2)
    assert tuple(edge_variance.shape) == (2, 2, 3)


def test_large_graph_basis_is_bounded_orthogonal_and_independent_of_global_randomness():
    adjacency = torch.zeros(150, 150)
    adjacency[torch.arange(149), torch.arange(1, 150)] = 1.
    # Unrelated parameter initialization must not rotate the noise coefficient basis.
    first = wm.LowFrequencyGraphPrior(adjacency, rank=4)
    torch.randn(23)
    second = wm.LowFrequencyGraphPrior(adjacency, rank=4)
    assert tuple(first.basis.shape) == (150, 4)
    torch.testing.assert_close(first.basis, second.basis, atol=0, rtol=0)
    torch.testing.assert_close(first.basis.T @ first.basis, torch.eye(4), atol=1e-6, rtol=1e-6)
    projection = first.project(torch.randn(1, 150, 64))
    torch.testing.assert_close(first.project(projection), projection, atol=1e-6, rtol=1e-6)


def test_world_model_removes_global_condition_parameters_and_token_slot():
    dynamics = _small_dynamics()
    assert not hasattr(wm, "ACTION_CONTEXT_DIM")
    assert not hasattr(dynamics.action_encoder, "context_mlp")
    assert not hasattr(dynamics.action_encoder, "action_token")
    assert not hasattr(dynamics, "action_slice")
    assert set(wm.ConditionEncoding.__dataclass_fields__) == {"edge_tokens", "reference_tokens"}
    assert all("context" not in name and "action_token" not in name
               for name, _ in dynamics.named_parameters())
    assert dynamics.flow_time_slice == slice(0, 1)
    assert dynamics.spatial_slice == slice(1, 4)
    assert dynamics.register_slice == slice(4, 8)
    assert dynamics.transformer.layers[0].space.attn_mask.shape[-1] == 3 + 1 + wm.REGISTER_TOKENS

    class CaptureTokens(torch.nn.Module):
        def forward(self, tokens):
            self.tokens = tokens
            return tokens

    capture = CaptureTokens()
    dynamics.transformer = capture
    action = torch.randn(1, 1, 3)
    state = torch.randn(1, 1, 3, 64)
    time = torch.tensor([[.4]])
    condition = dynamics.encode_condition(action)
    output = dynamics(action, packed_latents=state, flow_time=time, condition=condition)
    expected = wm.add_sinusoidal_positions(torch.cat([
        dynamics.flow_time(time), dynamics.spatial_proj(state) + condition.edge_tokens,
        dynamics.register_tokens.view(1, 1, wm.REGISTER_TOKENS, wm.WORLD_MODEL_DIM),
    ], dim=2))
    torch.testing.assert_close(capture.tokens, expected)
    torch.testing.assert_close(output.latent_velocity, dynamics.flow_head(expected[:, :, 1:4]))


def test_world_model_apis_reject_removed_context_instead_of_ignoring_it():
    adjacency, static = _build_graph(3)
    model = UrbanDreamerWorldModel(adjacency=adjacency, static_features=static)
    dynamics = model.dynamics
    state = torch.zeros(1, 1, 3, 64)
    action = torch.zeros(1, 1, 3)
    condition = dynamics.encode_condition(action)
    calls = [
        lambda: dynamics.action_encoder(action, context=torch.zeros(1, 1, 2)),
        lambda: dynamics.encode_condition(action, context=torch.zeros(1, 1, 2)),
        lambda: dynamics(action, packed_latents=state, flow_time=torch.zeros(1, 1),
                         context=torch.zeros(1, 1, 2)),
        lambda: dynamics.solve(state, action, condition, context=torch.zeros(1, 1, 2)),
        lambda: dynamics.transition(state, action, context=torch.zeros(1, 1, 2)),
        lambda: dynamics.latent_disagreement(state, action, context=torch.zeros(1, 1, 2)),
        lambda: model.transition(state, action, context=torch.zeros(1, 1, 2)),
        lambda: model.latent_disagreement(state, action, context=torch.zeros(1, 1, 2)),
        lambda: wm.recursive_fm_loss(dynamics, z_sequence=state.expand(-1, 2, -1, -1),
                                    action=action, context=torch.zeros(1, 1, 2)),
        lambda: wm.cfm_loss(dynamics, z_sequence=state.expand(-1, 2, -1, -1),
                           action=action, context=torch.zeros(1, 1, 2)),
        lambda: wm.cfm_predict_one_timestep(dynamics, past_latents=state, action=action,
                                           context=torch.zeros(1, 1, 2)),
    ]
    for call in calls:
        with pytest.raises(TypeError, match="context"):
            call()


def test_edge_actions_and_fm_time_still_change_velocity_and_receive_gradients():
    torch.manual_seed(901)
    dynamics = _small_dynamics()
    # The production head starts at zero; open it to test actual conditioning paths.
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.02)
    state = torch.randn(1, 1, 3, 64)
    action = torch.tensor([[[.2, .5, .8]]], requires_grad=True)
    time = torch.tensor([[.2]], requires_grad=True)
    velocity = dynamics(action, packed_latents=state, flow_time=time).latent_velocity
    with torch.no_grad():
        changed_action = dynamics(action + .7, packed_latents=state, flow_time=time).latent_velocity
        changed_time = dynamics(action, packed_latents=state, flow_time=time + .6).latent_velocity
    assert not torch.allclose(velocity, changed_action, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(velocity, changed_time, atol=1e-6, rtol=1e-6)
    velocity.square().mean().backward()
    assert action.grad is not None and action.grad.abs().sum() > 0
    assert time.grad is not None and time.grad.abs().sum() > 0
    for name, parameter in dynamics.named_parameters():
        if name.startswith(("action_encoder.", "flow_time.", "transformer.")):
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name


def test_transition_cache_keeps_current_state_reference_and_action_gradients_live():
    torch.manual_seed(902)
    dynamics = _small_dynamics(noise_scale=0.)
    torch.nn.init.normal_(dynamics.flow_head.weight, std=.02)
    torch.nn.init.normal_(dynamics.graph_prior.output.weight, std=.02)
    state = torch.randn(1, 3, 64, requires_grad=True)
    action = torch.tensor([[.2, .5, .8]], requires_grad=True)
    predicted = dynamics.transition(state, action, steps=2)
    predicted.square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert action.grad is not None and action.grad.abs().sum() > 0
    # Every retained trainable branch participates, including the cached reference
    # projection and all normalization layers; no deleted-token-only branch survives.
    for name, parameter in dynamics.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
