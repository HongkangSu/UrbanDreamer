from __future__ import annotations

import importlib.util
import math
from pathlib import Path

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


class ConstantVelocityDynamics(torch.nn.Module):
    def __init__(self, value: float, *, edge_count: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))
        self.logits = torch.nn.Parameter(torch.zeros((edge_count,), dtype=torch.float32))

    def forward(
        self,
        action: torch.Tensor,
        *,
        depart: torch.Tensor,
        context: torch.Tensor,
        packed_latents: torch.Tensor,
        hdv_context: torch.Tensor,
        flow_time: torch.Tensor,
    ) -> DynamicsOutput:
        del action, depart, context, hdv_context, flow_time
        velocity = torch.ones_like(packed_latents) * self.weight
        logits = self.logits.view(1, 1, -1).expand(
            packed_latents.shape[0],
            packed_latents.shape[1],
            -1,
        )
        return DynamicsOutput(latent_velocity=velocity, hdv_edge_logits=logits)


class IdentityDecodeProjector(torch.nn.Module):
    def decode(self, z_pred_next: torch.Tensor) -> torch.Tensor:
        return z_pred_next


def _build_graph(edge_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    adjacency = torch.eye(edge_count, dtype=torch.float32)
    static_features = torch.linspace(0.1, 0.4, steps=edge_count * 4, dtype=torch.float32).view(edge_count, 4)
    return adjacency, static_features


def test_cfm_loss_vectorizes_all_t_minus_1_transitions_with_fixed_hdv_weight() -> None:
    dynamics = ConstantVelocityDynamics(0.0, edge_count=2)
    z_sequence = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, 2, 64).clone()
    action = torch.zeros((1, 2, 2), dtype=torch.float32)
    depart = torch.zeros_like(action)
    context = torch.zeros((1, 2, 3), dtype=torch.float32)
    hdv_context = torch.zeros_like(action)
    hdv_target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32)

    loss, aux = cfm_loss(
        dynamics,
        z_sequence=z_sequence,
        action=action,
        depart=depart,
        context=context,
        hdv_context=hdv_context,
        hdv_target=hdv_target,
    )
    loss.backward()

    expected_hdv_ce = math.log(2.0)
    expected_loss = 4.0 + (0.05 * expected_hdv_ce)
    assert torch.isclose(loss, torch.tensor(expected_loss))
    assert torch.isclose(aux["flow_mse"], torch.tensor(4.0))
    assert torch.isclose(aux["weighted_flow_mse"], torch.tensor(4.0))
    assert torch.isclose(aux["hdv_policy_ce"], torch.tensor(expected_hdv_ce))
    assert dynamics.weight.grad is not None
    assert torch.isclose(dynamics.weight.grad, torch.tensor(-4.0))
    assert dynamics.logits.grad is not None


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
    assert len(model.dynamics.action_encoder.action_gnn) == 2
    assert len(model.dynamics.transformer.layers) == 6
    assert tuple(model.dynamics.register_tokens.shape) == (4, 256)
    assert set(wm.__all__) == {
        "UrbanDreamerWorldModel",
        "cfm_loss",
        "cfm_predict_one_timestep",
        "decoder_bpr_supervision_loss",
        "decoder_reconstruction_loss",
        "episode_t_all_supervision_loss",
        "transition_decoded_supervision_loss",
    }

    keys = set(model.state_dict().keys())
    assert "flow_projector.adjacency" not in keys
    assert "dynamics.action_encoder.adjacency" not in keys
    assert "dynamics.action_encoder.static_features" not in keys

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
        depart=torch.zeros((1, 1, 2), dtype=torch.float32),
        context=torch.zeros((1, 1, 3), dtype=torch.float32),
        packed_latents=latent,
        hdv_context=torch.zeros((1, 1, 2), dtype=torch.float32),
        flow_time=torch.tensor([[0.25]], dtype=torch.float32),
    )

    assert tuple(latent.shape) == (1, 1, 2, 64)
    assert tuple(decoded.shape) == (1, 1, 2, 4)
    assert tuple(output.latent_velocity.shape) == (1, 1, 2, 64)
    assert tuple(output.hdv_edge_logits.shape) == (1, 1, 2)
    assert tuple(model.flow_projector.predict_t_all_from_z(latent[:, 0]).shape) == (1,)
