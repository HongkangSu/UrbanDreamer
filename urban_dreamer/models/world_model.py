from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


FLOW_FEATURE_DIM = 8
FLOW_DYNAMIC_DIM = 4
FLOW_LATENT_DIM = 64
WORLD_MODEL_DIM = 256
TRANSFORMER_DEPTH = 6
TRANSFORMER_HEADS = 4
REGISTER_TOKENS = 4
GRAPH_LAYERS = 2
ACTION_CONTEXT_DIM = 3
ACTION_EDGE_DIM = 64
FLOW_SPEED_INDEX = 1
FLOW_VOLUME_INDEX = 2

DECODER_RECON_WEIGHT = 0.2
TRANSITION_DECODED_WEIGHT = 1.0
BPR_SUPERVISION_WEIGHT = 0.1
T_ALL_SUPERVISION_WEIGHT = 0.2
HDV_POLICY_CE_WEIGHT = 0.05

INITIAL_SPACE_CHUNK = 1
INITIAL_TIME_CHUNK = 256
INITIAL_MLP_CHUNK = 4096
ONLINE_SPACE_CHUNK = 16
ONLINE_TIME_CHUNK = 32768
ONLINE_MLP_CHUNK = 16384


@dataclass(frozen=True)
class DynamicsOutput:
    latent_velocity: torch.Tensor
    hdv_edge_logits: torch.Tensor


class EdgeGNNLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.message = nn.Linear(width, width, bias=False)
        self.self_proj = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        message = torch.matmul(adjacency, self.message(hidden))
        update = F.gelu(self.self_proj(hidden) + message)
        return self.norm(hidden + update)


def decode_nonnegative_volume(raw_volume: torch.Tensor) -> torch.Tensor:
    return F.softplus(raw_volume, beta=10.0)


def _positive_pred_speed_for_loss(raw_speed: torch.Tensor) -> torch.Tensor:
    raw = raw_speed.float()
    return torch.where(raw >= 0.0, raw, F.softplus(raw, beta=1.0))


def _positive_pred_volume_for_loss(raw_volume: torch.Tensor) -> torch.Tensor:
    raw = raw_volume.float()
    return torch.where(raw >= 0.0, raw, F.softplus(raw, beta=10.0))


def _transform_dynamic_targets(
    pred_decoded: torch.Tensor,
    target_decoded: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = pred_decoded.float()
    target = target_decoded.float()

    pred_out = pred.clone()
    target_out = target.clone()

    pred_speed = _positive_pred_speed_for_loss(pred[..., FLOW_SPEED_INDEX]).clamp_min(0.01)
    target_speed = target[..., FLOW_SPEED_INDEX].clamp_min(0.1)
    pred_out[..., FLOW_SPEED_INDEX] = pred_speed.reciprocal() - 1.0
    target_out[..., FLOW_SPEED_INDEX] = target_speed.reciprocal() - 1.0

    pred_vol = _positive_pred_volume_for_loss(pred[..., FLOW_VOLUME_INDEX])
    target_vol = target[..., FLOW_VOLUME_INDEX].clamp_min(0.0)
    pred_out[..., FLOW_VOLUME_INDEX] = torch.log1p(pred_vol)
    target_out[..., FLOW_VOLUME_INDEX] = torch.log1p(target_vol)
    return pred_out, target_out


def _dynamic_weighted_mse(pred_decoded: torch.Tensor, target_decoded: torch.Tensor) -> torch.Tensor:
    pred_eval, target_eval = _transform_dynamic_targets(pred_decoded, target_decoded)
    err2 = (pred_eval - target_eval).pow(2)
    feature_weight = torch.ones((FLOW_DYNAMIC_DIM,), device=err2.device, dtype=err2.dtype)
    feature_weight[FLOW_SPEED_INDEX] = 4.0
    feature_weight[FLOW_VOLUME_INDEX] = 2.0
    return (err2 * feature_weight.view(1, *([1] * (err2.dim() - 2)), FLOW_DYNAMIC_DIM)).mean()


def _validate_transition_tensor(
    values: torch.Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    if tuple(values.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(values.shape)}")
    return values


class FlowProjector(nn.Module):
    def __init__(self, *, adjacency: torch.Tensor) -> None:
        super().__init__()
        if adjacency.dim() != 2 or int(adjacency.shape[0]) != int(adjacency.shape[1]):
            raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
        self.n_edges = int(adjacency.shape[0])
        self.register_buffer("adjacency", adjacency.float(), persistent=False)

        self.input_proj = nn.Linear(FLOW_FEATURE_DIM, FLOW_LATENT_DIM)
        self.gnn_layers = nn.ModuleList([EdgeGNNLayer(FLOW_LATENT_DIM) for _ in range(GRAPH_LAYERS)])
        self.decoder_shared = nn.Sequential(
            nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM),
            nn.SiLU(),
        )
        self.decoder_speed = nn.Linear(FLOW_LATENT_DIM, 1)
        self.decoder_volume = nn.Linear(FLOW_LATENT_DIM, 1)
        self.decoder_other = nn.Linear(FLOW_LATENT_DIM, FLOW_DYNAMIC_DIM - 2)
        pooled_dim = 3 * FLOW_LATENT_DIM
        self.t_all_head = nn.Sequential(
            nn.Linear(pooled_dim, FLOW_LATENT_DIM),
            nn.GELU(),
            nn.Linear(FLOW_LATENT_DIM, FLOW_LATENT_DIM // 2),
            nn.GELU(),
            nn.Linear(FLOW_LATENT_DIM // 2, 1),
        )

        nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.decoder_shared[0].weight, gain=1.0)
        nn.init.zeros_(self.decoder_shared[0].bias)
        nn.init.orthogonal_(self.decoder_speed.weight, gain=1.0)
        nn.init.zeros_(self.decoder_speed.bias)
        nn.init.orthogonal_(self.decoder_volume.weight, gain=1.0)
        nn.init.zeros_(self.decoder_volume.bias)
        nn.init.orthogonal_(self.decoder_other.weight, gain=1.0)
        nn.init.zeros_(self.decoder_other.bias)
        nn.init.zeros_(self.t_all_head[-1].weight)
        nn.init.constant_(self.t_all_head[-1].bias, 1.5)

    def forward(self, flow_features: torch.Tensor) -> torch.Tensor:
        if int(flow_features.shape[-1]) != FLOW_FEATURE_DIM:
            raise ValueError(
                f"flow_features last dim must be {FLOW_FEATURE_DIM}, got {int(flow_features.shape[-1])}"
            )
        hidden = self.input_proj(flow_features)
        adjacency = self.adjacency.to(device=hidden.device, dtype=hidden.dtype)
        for layer in self.gnn_layers:
            hidden = layer(hidden, adjacency)
        return hidden

    def decode(self, latent_edges: torch.Tensor) -> torch.Tensor:
        if int(latent_edges.shape[-1]) != FLOW_LATENT_DIM:
            raise ValueError(
                f"latent_edges last dim must be {FLOW_LATENT_DIM}, got {int(latent_edges.shape[-1])}"
            )
        hidden = self.decoder_shared(latent_edges)
        speed = self.decoder_speed(hidden)
        volume = decode_nonnegative_volume(self.decoder_volume(hidden))
        other = self.decoder_other(hidden)
        return torch.cat([other[..., :1], speed, volume, other[..., 1:2]], dim=-1)

    def predict_t_all_from_z(self, latent_edges: torch.Tensor) -> torch.Tensor:
        decoded = self.decode(latent_edges)
        vehicle_volume = decoded[..., FLOW_VOLUME_INDEX].clamp_min(0.0)
        weights = vehicle_volume / vehicle_volume.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.cat(
            [
                latent_edges.mean(dim=-2),
                latent_edges.max(dim=-2).values,
                (latent_edges * weights.unsqueeze(-1)).sum(dim=-2),
            ],
            dim=-1,
        )
        return F.softplus(self.t_all_head(pooled).squeeze(-1))


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(variance + self.eps))


class MLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = 4 * width
        self.input = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)
        self.token_chunk_size = 0

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.input(x).chunk(2, dim=-1)
        return self.output(up * F.silu(gate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.token_chunk_size)
        if chunk_size <= 0:
            return self._forward_chunk(x)
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        if int(flat.shape[0]) <= chunk_size:
            return self._forward_chunk(x)
        chunks = [
            self._forward_chunk(flat[start : start + chunk_size])
            for start in range(0, int(flat.shape[0]), chunk_size)
        ]
        return torch.cat(chunks, dim=0).reshape(shape)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        if width % n_heads != 0:
            raise ValueError(f"width={width} must be divisible by n_heads={n_heads}")
        self.width = int(width)
        self.n_heads = int(n_heads)
        self.head_dim = self.width // self.n_heads
        self.qkv = nn.Linear(self.width, 3 * self.width)
        self.output = nn.Linear(self.width, self.width)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)
        query = query.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


def sinusoid_table(size: int, width: int, *, device: torch.device) -> torch.Tensor:
    position = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    channel = torch.arange(width, device=device, dtype=torch.float32).unsqueeze(0)
    exponent = torch.floor(channel / 2.0)
    scale = torch.exp(-(2.0 * exponent) / max(1.0, float(width)) * math.log(10000.0))
    angle = position * scale
    return torch.where((channel % 2) == 0, torch.sin(angle), torch.cos(angle))


def add_sinusoidal_positions(tokens: torch.Tensor) -> torch.Tensor:
    batch, time_steps, token_count, width = tokens.shape
    position_t = sinusoid_table(time_steps, width, device=tokens.device)
    position_s = sinusoid_table(token_count, width, device=tokens.device)
    position = (position_t[None, :, None, :] + position_s[None, None, :, :]) * (1.0 / math.sqrt(width))
    return tokens + position.to(dtype=tokens.dtype)


class SpaceSelfAttention(nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.attention_batch_chunk_size = 0
        self.attention = MultiheadSelfAttention(WORLD_MODEL_DIM, TRANSFORMER_HEADS)
        mask = torch.ones((1, 1, token_count, token_count), dtype=torch.bool)
        self.register_buffer("attn_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, token_count, width = x.shape
        flat = x.reshape(batch * time_steps, token_count, width)
        chunk_size = int(self.attention_batch_chunk_size)
        if chunk_size <= 0 or int(flat.shape[0]) <= chunk_size:
            output = self.attention(
                flat,
                attn_mask=self.attn_mask.expand(batch * time_steps, 1, token_count, token_count),
            )
        else:
            output = torch.cat(
                [
                    self.attention(
                        flat[start : start + chunk_size],
                        attn_mask=self.attn_mask.expand(
                            min(chunk_size, int(flat.shape[0]) - start),
                            1,
                            token_count,
                            token_count,
                        ),
                    )
                    for start in range(0, int(flat.shape[0]), chunk_size)
                ],
                dim=0,
            )
        return output.reshape(batch, time_steps, token_count, width)


class TimeSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_batch_chunk_size = 0
        self.attention = MultiheadSelfAttention(WORLD_MODEL_DIM, TRANSFORMER_HEADS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time_steps, token_count, width = x.shape
        flat = x.permute(0, 2, 1, 3).contiguous().view(batch * token_count, time_steps, width)
        chunk_size = int(self.attention_batch_chunk_size)
        if chunk_size <= 0 or int(flat.shape[0]) <= chunk_size:
            output = self.attention(flat, is_causal=True)
        else:
            output = torch.cat(
                [
                    self.attention(flat[start : start + chunk_size], is_causal=True)
                    for start in range(0, int(flat.shape[0]), chunk_size)
                ],
                dim=0,
            )
        return output.view(batch, token_count, time_steps, width).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.norm_space = RMSNorm(WORLD_MODEL_DIM)
        self.space = SpaceSelfAttention(token_count)
        self.norm_time = RMSNorm(WORLD_MODEL_DIM)
        self.time = TimeSelfAttention()
        self.norm_mlp = RMSNorm(WORLD_MODEL_DIM)
        self.mlp = MLP(WORLD_MODEL_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.space(self.norm_space(x))
        x = x + self.time(self.norm_time(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(self, token_count: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([BlockCausalLayer(token_count) for _ in range(TRANSFORMER_DEPTH)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if torch.is_grad_enabled():
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x


class FlowTimeEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden = 2 * WORLD_MODEL_DIM
        self.network = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, WORLD_MODEL_DIM),
        )

    def forward(self, flow_time: torch.Tensor) -> torch.Tensor:
        return self.network(flow_time.unsqueeze(-1)).unsqueeze(2)


class ActionDepartGNNEncoder(nn.Module):
    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        static_features: torch.Tensor,
    ) -> None:
        super().__init__()
        if adjacency.dim() != 2 or int(adjacency.shape[0]) != int(adjacency.shape[1]):
            raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
        if static_features.dim() != 2 or int(static_features.shape[0]) != int(adjacency.shape[0]):
            raise ValueError(
                "static_features must have shape (E, S) with the same edge count as adjacency, "
                f"got {tuple(static_features.shape)}"
            )
        self.n_edges = int(adjacency.shape[0])
        self.n_static = int(static_features.shape[1])
        self.register_buffer("adjacency", adjacency.float(), persistent=False)
        self.register_buffer("static_features", static_features.float(), persistent=False)

        self.action_input = nn.Linear(1 + self.n_static, ACTION_EDGE_DIM)
        self.depart_input = nn.Linear(1 + self.n_static, ACTION_EDGE_DIM)
        self.hdv_input = nn.Linear(1 + self.n_static, ACTION_EDGE_DIM)
        self.action_gnn = nn.ModuleList([EdgeGNNLayer(ACTION_EDGE_DIM) for _ in range(GRAPH_LAYERS)])
        self.depart_gnn = nn.ModuleList([EdgeGNNLayer(ACTION_EDGE_DIM) for _ in range(GRAPH_LAYERS)])
        self.hdv_gnn = nn.ModuleList([EdgeGNNLayer(ACTION_EDGE_DIM) for _ in range(GRAPH_LAYERS)])
        self.context_mlp = nn.Sequential(
            nn.Linear(ACTION_CONTEXT_DIM, ACTION_EDGE_DIM),
            nn.SiLU(),
            nn.Linear(ACTION_EDGE_DIM, ACTION_EDGE_DIM),
        )
        self.action_token = nn.Linear(5 * ACTION_EDGE_DIM, WORLD_MODEL_DIM)
        self.edge_conditioning = nn.Linear(3 * ACTION_EDGE_DIM, WORLD_MODEL_DIM)

    def _augment_with_static(self, values: torch.Tensor) -> torch.Tensor:
        batch, time_steps, edge_count = values.shape
        static = self.static_features.to(device=values.device, dtype=values.dtype)
        static = static.view(1, 1, edge_count, self.n_static).expand(batch, time_steps, edge_count, self.n_static)
        return torch.cat([values.unsqueeze(-1), static], dim=-1)

    def _run_stack(self, projection: nn.Linear, layers: nn.ModuleList, values: torch.Tensor) -> torch.Tensor:
        hidden = projection(self._augment_with_static(values))
        adjacency = self.adjacency.to(device=hidden.device, dtype=hidden.dtype)
        for layer in layers:
            hidden = layer(hidden, adjacency)
        return hidden

    def forward(
        self,
        action: torch.Tensor,
        *,
        depart: torch.Tensor,
        context: torch.Tensor,
        hdv_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(action.shape) != tuple(depart.shape) or tuple(action.shape) != tuple(hdv_context.shape):
            raise ValueError("action, depart, and hdv_context must share shape (B, T, E)")
        if int(action.shape[-1]) != self.n_edges:
            raise ValueError(f"expected {self.n_edges} edge-aligned actions, got {int(action.shape[-1])}")
        if context.dim() != 3 or tuple(context.shape[:2]) != tuple(action.shape[:2]) or int(context.shape[-1]) != ACTION_CONTEXT_DIM:
            raise ValueError(
                f"context must have shape (B, T, {ACTION_CONTEXT_DIM}), got {tuple(context.shape)}"
            )

        action_hidden = self._run_stack(self.action_input, self.action_gnn, action)
        depart_hidden = self._run_stack(self.depart_input, self.depart_gnn, depart)
        hdv_hidden = self._run_stack(self.hdv_input, self.hdv_gnn, hdv_context)
        context_hidden = self.context_mlp(context)

        pooled = torch.cat(
            [
                action_hidden.mean(dim=2),
                action_hidden.max(dim=2).values,
                depart_hidden.mean(dim=2),
                hdv_hidden.mean(dim=2),
                context_hidden,
            ],
            dim=-1,
        )
        action_token = self.action_token(pooled).unsqueeze(2)
        edge_tokens = self.edge_conditioning(torch.cat([action_hidden, depart_hidden, hdv_hidden], dim=-1))
        return action_token, edge_tokens


class Dynamics(nn.Module):
    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        static_features: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_edges = int(adjacency.shape[0])
        self.action_encoder = ActionDepartGNNEncoder(adjacency=adjacency, static_features=static_features)
        self.spatial_proj = nn.Linear(FLOW_LATENT_DIM, WORLD_MODEL_DIM)
        self.flow_time = FlowTimeEmbedder()
        self.register_tokens = nn.Parameter(torch.empty(REGISTER_TOKENS, WORLD_MODEL_DIM))
        nn.init.normal_(self.register_tokens, std=0.02)

        self.edge_action_proj = nn.Linear(2, WORLD_MODEL_DIM)
        self.hdv_context_edge_proj = nn.Linear(2, WORLD_MODEL_DIM)
        nn.init.zeros_(self.edge_action_proj.weight)
        nn.init.zeros_(self.edge_action_proj.bias)
        nn.init.normal_(self.hdv_context_edge_proj.weight, std=0.02)
        nn.init.zeros_(self.hdv_context_edge_proj.bias)

        self.action_slice = slice(0, 1)
        self.flow_time_slice = slice(1, 2)
        self.spatial_slice = slice(2, 2 + self.n_edges)
        self.register_slice = slice(2 + self.n_edges, 2 + self.n_edges + REGISTER_TOKENS)
        self.transformer = BlockCausalTransformer(2 + self.n_edges + REGISTER_TOKENS)

        self.flow_head = nn.Linear(WORLD_MODEL_DIM, FLOW_LATENT_DIM)
        self.hdv_policy_head = nn.Sequential(
            nn.Linear(WORLD_MODEL_DIM, WORLD_MODEL_DIM),
            nn.GELU(),
            nn.Linear(WORLD_MODEL_DIM, 1),
        )
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.hdv_policy_head[-1].weight)
        nn.init.zeros_(self.hdv_policy_head[-1].bias)

        self.set_initial_chunk_profile()

    def _set_chunk_profile(self, *, space_chunk: int, time_chunk: int, mlp_chunk: int) -> None:
        for layer in self.transformer.layers:
            layer.space.attention_batch_chunk_size = int(space_chunk)
            layer.time.attention_batch_chunk_size = int(time_chunk)
            layer.mlp.token_chunk_size = int(mlp_chunk)

    def set_initial_chunk_profile(self) -> None:
        self._set_chunk_profile(
            space_chunk=INITIAL_SPACE_CHUNK,
            time_chunk=INITIAL_TIME_CHUNK,
            mlp_chunk=INITIAL_MLP_CHUNK,
        )

    def set_online_chunk_profile(self) -> None:
        self._set_chunk_profile(
            space_chunk=ONLINE_SPACE_CHUNK,
            time_chunk=ONLINE_TIME_CHUNK,
            mlp_chunk=ONLINE_MLP_CHUNK,
        )

    def _edge_action_features(self, edge_actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_actions = edge_actions.float()
        mask_values = (edge_actions.abs() > 1e-6).to(edge_actions.dtype)
        mask = mask_values.unsqueeze(-1)
        active_edge_count = mask_values.sum(dim=-1, keepdim=True).clamp_min(1.0)
        features = torch.stack(
            [mask_values, edge_actions * active_edge_count],
            dim=-1,
        )
        return mask, features

    def _inject_actions(
        self,
        spatial_tokens: torch.Tensor,
        edge_actions: torch.Tensor,
        projection: nn.Linear,
    ) -> torch.Tensor:
        mask, features = self._edge_action_features(edge_actions.to(device=spatial_tokens.device, dtype=spatial_tokens.dtype))
        injected = projection(features.to(dtype=spatial_tokens.dtype))
        return spatial_tokens + injected * mask

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
        if packed_latents.dim() != 4 or tuple(packed_latents.shape[-2:]) != (self.n_edges, FLOW_LATENT_DIM):
            raise ValueError(
                "packed_latents must have shape (B, T, E, 64), "
                f"got {tuple(packed_latents.shape)}"
            )
        batch, time_steps, _, _ = packed_latents.shape
        expected_edges = (batch, time_steps, self.n_edges)
        _validate_transition_tensor(action, name="action", expected_shape=expected_edges)
        _validate_transition_tensor(depart, name="depart", expected_shape=expected_edges)
        _validate_transition_tensor(hdv_context, name="hdv_context", expected_shape=expected_edges)
        if tuple(context.shape) != (batch, time_steps, ACTION_CONTEXT_DIM):
            raise ValueError(
                f"context must have shape {(batch, time_steps, ACTION_CONTEXT_DIM)}, got {tuple(context.shape)}"
            )
        if tuple(flow_time.shape) != (batch, time_steps):
            raise ValueError(f"flow_time must have shape {(batch, time_steps)}, got {tuple(flow_time.shape)}")

        spatial_tokens = self.spatial_proj(packed_latents)
        action_token, graph_edge_tokens = self.action_encoder(
            action,
            depart=depart,
            context=context,
            hdv_context=hdv_context,
        )
        spatial_tokens = self._inject_actions(spatial_tokens, action, self.edge_action_proj)
        spatial_tokens = self._inject_actions(spatial_tokens, hdv_context, self.hdv_context_edge_proj)
        spatial_tokens = spatial_tokens + graph_edge_tokens

        register_tokens = self.register_tokens.view(1, 1, REGISTER_TOKENS, WORLD_MODEL_DIM)
        register_tokens = register_tokens.expand(batch, time_steps, -1, -1)
        time_token = self.flow_time(flow_time.to(dtype=spatial_tokens.dtype))
        tokens = torch.cat([action_token, time_token, spatial_tokens, register_tokens], dim=2)
        hidden = self.transformer(add_sinusoidal_positions(tokens))
        spatial_hidden = hidden[:, :, self.spatial_slice, :]
        return DynamicsOutput(
            latent_velocity=self.flow_head(spatial_hidden),
            hdv_edge_logits=self.hdv_policy_head(spatial_hidden).squeeze(-1),
        )


class ActiveVehicleFractionReadout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fraction_eps = 1e-3
        hidden = FLOW_LATENT_DIM // 2
        pooled_dim = 4 * FLOW_LATENT_DIM
        self.latent_norm = nn.LayerNorm(pooled_dim)
        self.correction_head = nn.Sequential(
            nn.Linear(pooled_dim + 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("global_logit_bias", torch.zeros((), dtype=torch.float32))
        self.register_buffer("calibration_updates", torch.zeros((), dtype=torch.long))
        nn.init.orthogonal_(self.correction_head[0].weight)
        nn.init.zeros_(self.correction_head[0].bias)
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)

    def corrected_logits(
        self,
        z_next: torch.Tensor,
        *,
        z_prev: torch.Tensor,
        raw_decoder_fraction: torch.Tensor | float,
        transition_seconds: torch.Tensor | float,
    ) -> torch.Tensor:
        if z_next.dim() < 2 or int(z_next.shape[-1]) != FLOW_LATENT_DIM:
            raise ValueError(
                "ActiveVehicleFractionReadout expects (..., E, 64), "
                f"got {tuple(z_next.shape)}"
            )
        if tuple(z_prev.shape) != tuple(z_next.shape):
            raise ValueError("z_prev must match z_next shape")

        leading_shape = z_next.shape[:-2]
        latent_features = self.latent_norm(
            torch.cat(
                [
                    z_prev.mean(dim=-2),
                    z_prev.max(dim=-2).values,
                    z_next.mean(dim=-2),
                    z_next.max(dim=-2).values,
                ],
                dim=-1,
            )
        )
        duration = torch.as_tensor(transition_seconds, device=z_next.device, dtype=z_next.dtype)
        while duration.dim() < len(leading_shape):
            duration = duration.unsqueeze(-1)
        duration = torch.broadcast_to(duration, leading_shape)
        if bool(torch.any(~torch.isfinite(duration)).item()) or bool(torch.any(duration <= 0.0).item()):
            raise ValueError("transition_seconds must be finite and positive")

        raw = torch.as_tensor(raw_decoder_fraction, device=z_next.device, dtype=z_next.dtype)
        while raw.dim() < len(leading_shape):
            raw = raw.unsqueeze(-1)
        raw = torch.broadcast_to(raw, leading_shape)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=0.0).clamp(min=0.0, max=1e6)
        base = raw.clamp(min=self.fraction_eps, max=1.0 - self.fraction_eps)
        raw_features = torch.stack([base, torch.log1p(raw)], dim=-1)
        correction = self.correction_head(
            torch.cat([latent_features, raw_features, torch.log1p(duration).unsqueeze(-1)], dim=-1)
        ).squeeze(-1)
        return torch.logit(base) + self.global_logit_bias.to(device=z_next.device, dtype=z_next.dtype) + correction

    def forward(
        self,
        z_next: torch.Tensor,
        *,
        z_prev: torch.Tensor,
        raw_decoder_fraction: torch.Tensor | float,
        transition_seconds: torch.Tensor | float,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.corrected_logits(
                z_next,
                z_prev=z_prev,
                raw_decoder_fraction=raw_decoder_fraction,
                transition_seconds=transition_seconds,
            )
        )


class UrbanDreamerWorldModel(nn.Module):
    def __init__(self, *, adjacency: torch.Tensor, static_features: torch.Tensor) -> None:
        super().__init__()
        self.flow_projector = FlowProjector(adjacency=adjacency)
        self.dynamics = Dynamics(adjacency=adjacency, static_features=static_features)
        self.active_vehicle_fraction_readout = ActiveVehicleFractionReadout()

    def encode(self, flow_features: torch.Tensor) -> torch.Tensor:
        return self.flow_projector(flow_features)

    def decode(self, latent_edges: torch.Tensor) -> torch.Tensor:
        return self.flow_projector.decode(latent_edges)

    def set_initial_chunk_profile(self) -> None:
        self.dynamics.set_initial_chunk_profile()

    def set_online_chunk_profile(self) -> None:
        self.dynamics.set_online_chunk_profile()


def decoder_reconstruction_loss(
    *,
    decoded_dyn: torch.Tensor,
    flow_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_dyn = flow_targets[..., :FLOW_DYNAMIC_DIM].to(dtype=decoded_dyn.dtype)
    weighted_mse = _dynamic_weighted_mse(decoded_dyn, target_dyn)
    loss = DECODER_RECON_WEIGHT * weighted_mse
    return loss, {
        "decoder_recon_weighted_mse": weighted_mse.detach(),
        "decoder_recon_loss": loss.detach(),
    }


def transition_decoded_supervision_loss(
    *,
    z_pred_next: torch.Tensor,
    flow_projector: FlowProjector | nn.Module,
    flow_target_next: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_next_decoded = flow_projector.decode(z_pred_next)  # type: ignore[attr-defined]
    target_dyn = flow_target_next[..., :FLOW_DYNAMIC_DIM].to(dtype=pred_next_decoded.dtype)
    weighted_mse = _dynamic_weighted_mse(pred_next_decoded, target_dyn)
    loss = TRANSITION_DECODED_WEIGHT * weighted_mse
    return loss, {
        "pred_next_decoded_weighted_mse": weighted_mse.detach(),
        "pred_next_decoded_supervision_loss": loss.detach(),
    }


def decoder_bpr_supervision_loss(
    *,
    decoded_dyn: torch.Tensor,
    flow_targets: torch.Tensor,
    lengths: torch.Tensor,
    speed_limits: torch.Tensor,
    capacities: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_dyn = flow_targets[..., :FLOW_DYNAMIC_DIM].to(dtype=decoded_dyn.dtype)
    cap = capacities.to(device=decoded_dyn.device, dtype=torch.float32).clamp_min(1.0).view(1, *([1] * (decoded_dyn.dim() - 2)), -1)
    t0 = (
        lengths.to(device=decoded_dyn.device, dtype=torch.float32)
        / speed_limits.to(device=decoded_dyn.device, dtype=torch.float32).clamp_min(0.1)
    ).view(1, *([1] * (decoded_dyn.dim() - 2)), -1)
    pred_volume = _positive_pred_volume_for_loss(decoded_dyn[..., FLOW_VOLUME_INDEX]) * cap
    true_volume = target_dyn[..., FLOW_VOLUME_INDEX].float().clamp_min(0.0) * cap
    pred_time = t0 * (1.0 + 0.15 * (pred_volume / cap).pow(4.0))
    true_time = t0 * (1.0 + 0.15 * (true_volume / cap).pow(4.0))
    pred_eval = pred_time / t0.clamp_min(1e-6) - 1.0
    true_eval = true_time / t0.clamp_min(1e-6) - 1.0
    weighted_mse = (pred_eval - true_eval).pow(2).mean()
    loss = BPR_SUPERVISION_WEIGHT * weighted_mse
    return loss, {
        "bpr_weighted_mse": weighted_mse.detach(),
        "bpr_supervision_loss": loss.detach(),
    }


def episode_t_all_supervision_loss(
    *,
    flow_projector: FlowProjector,
    z_tokens: torch.Tensor,
    reward_targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = z_tokens.device
    if reward_targets.dim() not in {1, 2}:
        raise ValueError(
            "reward_targets must have shape (B,) or (B, T), "
            f"got {tuple(reward_targets.shape)}"
        )
    reward_last = (
        reward_targets[:, -1].to(device=device, dtype=torch.float32)
        if reward_targets.dim() == 2
        else reward_targets.to(device=device, dtype=torch.float32)
    )
    target_t_all = -reward_last
    valid = torch.isfinite(target_t_all) & (target_t_all > 0.0) & (reward_last < 0.0)
    if not bool(valid.all().item()):
        raise ValueError(
            "reward_targets must contain finite negative total-travel-minute "
            "rewards"
        )

    pred_t_all = flow_projector.predict_t_all_from_z(z_tokens[:, -1])
    target_t_all = target_t_all.float()
    scale = target_t_all.mean().clamp_min(1e-6)
    mse = ((pred_t_all.float() / scale) - (target_t_all / scale)).pow(2).mean()
    loss = T_ALL_SUPERVISION_WEIGHT * mse
    return loss, {
        "t_all_supervision_loss": loss.detach(),
        "t_all_pred_mse": mse.detach(),
    }


def cfm_loss(
    dynamics: Dynamics | nn.Module,
    *,
    z_sequence: torch.Tensor,
    action: torch.Tensor,
    depart: torch.Tensor,
    context: torch.Tensor,
    hdv_context: torch.Tensor,
    hdv_target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if z_sequence.dim() != 4 or int(z_sequence.shape[-1]) != FLOW_LATENT_DIM or int(z_sequence.shape[1]) < 2:
        raise ValueError(
            "z_sequence must have shape (B, T, E, 64) with T >= 2, "
            f"got {tuple(z_sequence.shape)}"
        )
    batch, total_steps, edge_count, _ = z_sequence.shape
    transition_steps = total_steps - 1
    edge_shape = (batch, transition_steps, edge_count)
    _validate_transition_tensor(action, name="action", expected_shape=edge_shape)
    _validate_transition_tensor(depart, name="depart", expected_shape=edge_shape)
    _validate_transition_tensor(hdv_context, name="hdv_context", expected_shape=edge_shape)
    _validate_transition_tensor(hdv_target, name="hdv_target", expected_shape=edge_shape)
    if tuple(context.shape) != (batch, transition_steps, ACTION_CONTEXT_DIM):
        raise ValueError(
            f"context must have shape {(batch, transition_steps, ACTION_CONTEXT_DIM)}, got {tuple(context.shape)}"
        )

    source_z = z_sequence[:, :-1].contiguous()
    target_z = z_sequence[:, 1:].contiguous()
    flow_time = torch.rand(batch, transition_steps, device=z_sequence.device, dtype=torch.float32)
    x_t = (1.0 - flow_time[..., None, None]) * source_z + flow_time[..., None, None] * target_z
    target_velocity = target_z - source_z

    output = dynamics(
        action,
        depart=depart,
        context=context,
        packed_latents=x_t,
        hdv_context=hdv_context,
        flow_time=flow_time,
    )
    error2 = (output.latent_velocity.float() - target_velocity.float()).pow(2)
    flow_mse = error2.mean()

    target = hdv_target.to(device=z_sequence.device, dtype=torch.float32).clamp_min(0.0)
    target_sum = target.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(target, 1.0 / float(target.shape[-1]))
    target = torch.where(target_sum > 1e-8, target / target_sum.clamp_min(1e-8), uniform)
    hdv_ce = -(target * F.log_softmax(output.hdv_edge_logits.float(), dim=-1)).sum(dim=-1).mean()

    loss = flow_mse + HDV_POLICY_CE_WEIGHT * hdv_ce
    return loss, {
        "flow_mse": flow_mse.detach(),
        "weighted_flow_mse": flow_mse.detach(),
        "hdv_policy_ce": hdv_ce.detach(),
        "t_mean": flow_time.mean().detach(),
    }


def cfm_predict_one_timestep(
    dynamics: Dynamics | nn.Module,
    *,
    past_latents: torch.Tensor,
    action: torch.Tensor,
    depart: torch.Tensor,
    context: torch.Tensor,
    hdv_context: torch.Tensor,
) -> torch.Tensor:
    if past_latents.dim() != 4 or int(past_latents.shape[-1]) != FLOW_LATENT_DIM or int(past_latents.shape[1]) < 1:
        raise ValueError(
            "past_latents must have shape (B, T, E, 64) with T >= 1, "
            f"got {tuple(past_latents.shape)}"
        )
    current = past_latents[:, -1:].contiguous()
    batch, _, edge_count, _ = current.shape
    edge_shape = (batch, 1, edge_count)
    _validate_transition_tensor(action, name="action", expected_shape=edge_shape)
    _validate_transition_tensor(depart, name="depart", expected_shape=edge_shape)
    _validate_transition_tensor(hdv_context, name="hdv_context", expected_shape=edge_shape)
    if tuple(context.shape) != (batch, 1, ACTION_CONTEXT_DIM):
        raise ValueError(f"context must have shape {(batch, 1, ACTION_CONTEXT_DIM)}, got {tuple(context.shape)}")

    velocity = dynamics(
        action,
        depart=depart,
        context=context,
        packed_latents=current,
        hdv_context=hdv_context,
        flow_time=torch.zeros((batch, 1), device=current.device, dtype=torch.float32),
    ).latent_velocity
    return current + velocity


__all__ = [
    "UrbanDreamerWorldModel",
    "cfm_loss",
    "cfm_predict_one_timestep",
    "decoder_bpr_supervision_loss",
    "decoder_reconstruction_loss",
    "episode_t_all_supervision_loss",
    "transition_decoded_supervision_loss",
]
