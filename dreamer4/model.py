# model.py
import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeGNNLayer(nn.Module):
    """Single message-passing layer on edge line graph."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_msg = nn.Linear(d_model, d_model, bias=False)
        self.W_self = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """h: (*, E, d), adj: (E, E) -> (*, E, d)"""
        msg = torch.matmul(adj, self.W_msg(h))
        out = F.gelu(self.W_self(h) + msg)
        return self.norm(out + h)


class Modality(IntEnum):
    LATENT = -1
    IMAGE = 0
    ACTION = 1
    PROPRIO = 2
    REGISTER = 3
    SPATIAL = 4
    FLOW_TIME = 5
    SHORTCUT_SIGNAL = 5
    SHORTCUT_STEP = 6
    AGENT = 7


@dataclass(frozen=True)
class TokenLayout:
    n_latents: int
    segments: Tuple[Tuple[Modality, int], ...]

    def S(self) -> int:
        return self.n_latents + sum(n for _, n in self.segments)

    def modality_ids(self) -> torch.Tensor:
        parts = []
        if self.n_latents > 0:
            parts.append(torch.full((self.n_latents,), int(Modality.LATENT), dtype=torch.int32))
        for m, n in self.segments:
            if n > 0:
                parts.append(torch.full((n,), int(m), dtype=torch.int32))
        return torch.cat(parts, dim=0) if parts else torch.zeros((0,), dtype=torch.int32)

    def slices(self) -> Dict[Modality, slice]:
        idx = 0
        out: Dict[Modality, slice] = {}
        if self.n_latents > 0:
            out[Modality.LATENT] = slice(idx, idx + self.n_latents)
            idx += self.n_latents
        for m, n in self.segments:
            if n > 0 and m not in out:
                out[m] = slice(idx, idx + n)
            idx += n
        return out



def sinusoid_table(n: int, d: int, base: float = 10000.0, device=None) -> torch.Tensor:
    # fp32 by construction
    pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # (n,1)
    i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)  # (1,d)
    k   = torch.floor(i / 2.0)
    # stable: exp(log(base) * exponent)
    div = torch.exp(-(2.0 * k) / max(1.0, float(d)) * math.log(base))
    ang = pos * div
    return torch.where((i % 2) == 0, torch.sin(ang), torch.cos(ang))  # (n,d) fp32


def add_sinusoidal_positions(tokens_btSd: torch.Tensor) -> torch.Tensor:
    B, T, S, D = tokens_btSd.shape
    device = tokens_btSd.device
    pos_t = sinusoid_table(T, D, device=device)  # fp32
    pos_s = sinusoid_table(S, D, device=device)  # fp32
    pos = (pos_t[None, :, None, :] + pos_s[None, None, :, :]) * (1.0 / math.sqrt(D))
    return tokens_btSd + pos.to(dtype=tokens_btSd.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * (self.scale / torch.sqrt(var + self.eps))


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc_in = nn.Linear(d_model, 2 * hidden)
        self.fc_out = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc_in(x).chunk(2, dim=-1)
        h = u * F.silu(v)
        h = self.drop(h)
        y = self.fc_out(h)
        y = self.drop(y)
        return y


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x_nld: torch.Tensor, *, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        """
        x: (N,L,D)
        attn_mask: bool, True means "allowed to attend" (for torch SDPA), broadcastable to (N,1,L,L) or (N,H,L,L)
        """
        N, L, D = x_nld.shape
        q, k, v = self.qkv(x_nld).chunk(3, dim=-1)

        q = q.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)

        drop = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        return self.out(y)


class SpaceSelfAttentionModality(nn.Module):
    def __init__(self, d_model: int, n_heads: int, modality_ids: torch.Tensor, n_latents: int, mode: str, dropout: float):
        super().__init__()
        self.n_latents = int(n_latents)
        self.mode = mode
        self.register_buffer("modality_ids", modality_ids.to(torch.int32), persistent=False)

        S = int(self.modality_ids.numel())
        allow = self._build_allow(S)                               # (S,S) True=allowed
        attn_mask = allow.unsqueeze(0).unsqueeze(0)                # (1,1,S,S) True=allowed (PyTorch SDPA bool mask)
        self.register_buffer("attn_mask", attn_mask, persistent=False)

        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def _build_allow(self, S: int) -> torch.Tensor:
        device = self.modality_ids.device
        q_idx = torch.arange(S, device=device).unsqueeze(1)  # (S,1)
        k_idx = torch.arange(S, device=device).unsqueeze(0)  # (1,S)

        is_q_lat = q_idx < self.n_latents
        is_k_lat = k_idx < self.n_latents

        q_mod = self.modality_ids[q_idx]
        k_mod = self.modality_ids[k_idx]
        same_mod = (q_mod == k_mod)

        if self.mode == "encoder":
            allow_lat_q = torch.ones((S, S), dtype=torch.bool, device=device)
            allow_nonlat_q = same_mod
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "decoder":
            allow_lat_q = is_k_lat
            allow_nonlat_q = same_mod | is_k_lat
            return torch.where(is_q_lat, allow_lat_q, allow_nonlat_q)

        if self.mode == "wm_agent":
            # full mixing across modalities
            return torch.ones((S, S), dtype=torch.bool, device=device)

        if self.mode == "wm_agent_isolated":
            # non-agent tokens: can attend to everything EXCEPT agent tokens
            # agent tokens: attend only to agent tokens (keeps them inert in pretrain)
            is_q_agent = (q_mod == int(Modality.AGENT))
            is_k_agent = (k_mod == int(Modality.AGENT))

            allow = torch.ones((S, S), dtype=torch.bool, device=device)

            # non-agent queries cannot see agent keys
            allow_non_agent_q = ~is_q_agent
            allow = torch.where(allow_non_agent_q, ~is_k_agent, allow)

            # agent queries only see agent keys
            allow = torch.where(is_q_agent, is_k_agent, allow)
            return allow

        raise ValueError(f"Unsupported mode for tokenizer/wm: {self.mode}")

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        x = x_btSd.reshape(B * T, S, D)
        mask = self.attn_mask.expand(B * T, 1, S, S)
        y = self.attn(x, attn_mask=mask, is_causal=False)
        return y.reshape(B, T, S, D)


class TimeSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, latents_only: bool, n_latents: int):
        super().__init__()
        self.latents_only = bool(latents_only)
        self.n_latents = int(n_latents)
        self.attn = MultiheadSelfAttention(d_model, n_heads, dropout=dropout)

    def forward(self, x_btSd: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x_btSd.shape
        if self.latents_only:
            L = self.n_latents
            lat = x_btSd[:, :, :L, :]  # (B,T,L,D)
            lat_nld = lat.permute(0, 2, 1, 3).contiguous().view(B * L, T, D)
            out = self.attn(lat_nld, is_causal=True)
            out = out.view(B, L, T, D).permute(0, 2, 1, 3).contiguous()
            x = x_btSd.clone()
            x[:, :, :L, :] = out
            return x
        else:
            x_nld = x_btSd.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
            out = self.attn(x_nld, is_causal=True)
            return out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()


class BlockCausalLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        layer_index: int,
        time_every: int,
        latents_only_time: bool,
    ):
        super().__init__()
        self.do_time = ((layer_index + 1) % time_every == 0)

        self.norm1 = RMSNorm(d_model)
        self.space = SpaceSelfAttentionModality(d_model, n_heads, modality_ids, n_latents, space_mode, dropout)
        self.drop1 = nn.Dropout(dropout)

        if self.do_time:
            self.norm2 = RMSNorm(d_model)
            self.time = TimeSelfAttention(d_model, n_heads, dropout, latents_only_time, n_latents)
            self.drop2 = nn.Dropout(dropout)

        self.norm3 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.space(self.norm1(x)))
        if self.do_time:
            x = x + self.drop2(self.time(self.norm2(x)))
        x = x + self.drop3(self.mlp(self.norm3(x)))
        return x


class BlockCausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        depth: int,
        n_latents: int,
        modality_ids: torch.Tensor,
        space_mode: str,
        dropout: float,
        mlp_ratio: float,
        time_every: int,
        latents_only_time: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockCausalLayer(
                d_model=d_model, n_heads=n_heads, n_latents=n_latents,
                modality_ids=modality_ids, space_mode=space_mode,
                dropout=dropout, mlp_ratio=mlp_ratio,
                layer_index=i, time_every=time_every,
                latents_only_time=latents_only_time,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x




class ActionEncoder(nn.Module):
    """
    Continuous actions in [-1,1], shape (B,T,A) -> token (B,T,1,D).
    If actions is None (unlabeled pretrain), emits a learned base token.
    """
    def __init__(self, d_model: int, action_dim: int = 16, hidden_mult: float = 2.0):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)

        hidden = int(self.d_model * hidden_mult)
        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)

        self.fc1 = nn.Linear(self.action_dim, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)

        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        actions: Optional[torch.Tensor],                 # (B,T,A) or None
        *,
        batch_time_shape: Optional[Tuple[int,int]] = None,
        act_mask: Optional[torch.Tensor] = None,         # (B,T,A) or (A,)
        as_tokens: bool = True,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
        else:
            x = actions
            if act_mask is not None:
                x = x * act_mask
            x = x.clamp(-1, 1)
            out = self.fc2(F.silu(self.fc1(x))) + self.base.view(1, 1, -1)

        return out[:, :, None, :] if as_tokens else out


class EdgeActionGNNLayer(nn.Module):
    """
    Message passing layer over edge line graph.
    """
    def __init__(self, d_edge: int):
        super().__init__()
        self.W_msg = nn.Linear(d_edge, d_edge, bias=False)
        self.W_self = nn.Linear(d_edge, d_edge)
        self.norm = nn.LayerNorm(d_edge)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B*T, E, d_edge), adj: (E, E)
        msg = torch.matmul(adj, self.W_msg(h))
        out = F.gelu(self.W_self(h) + msg)
        return self.norm(out + h)


class EdgeGNNActionEncoder(nn.Module):
    """
    Graph-aware action encoder for edge-level actions.

    Expects actions of shape (B,T,E), where E == n_edges.
    Uses edge graph message passing and pools to a single action token.
    """
    def __init__(
        self,
        d_model: int,
        n_edges: int,
        n_static: int = 0,
        d_edge: int = 64,
        n_gnn_layers: int = 2,
        hidden_mult: float = 2.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_edges = int(n_edges)
        self.n_static = int(max(0, n_static))
        self.d_edge = int(d_edge)
        self.n_gnn_layers = int(max(0, n_gnn_layers))
        self.action_dim = self.n_edges

        self.base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.base, std=0.02)

        in_dim = 1 + self.n_static
        self.input_proj = nn.Linear(in_dim, self.d_edge)
        self.gnn_layers = nn.ModuleList([EdgeActionGNNLayer(self.d_edge) for _ in range(self.n_gnn_layers)])

        hidden = int(self.d_model * float(hidden_mult))
        self.fc1 = nn.Linear(self.d_edge, hidden)
        self.fc2 = nn.Linear(hidden, self.d_model)

        nn.init.orthogonal_(self.input_proj.weight, gain=1.0)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

        self.register_buffer("_adj", torch.empty(0), persistent=False)
        self.register_buffer("_static_features", torch.empty(0), persistent=False)
        self._has_adj = False
        self._has_static = False

    def set_adjacency(self, adj: torch.Tensor) -> None:
        if adj.dim() != 2:
            raise ValueError(f"adjacency must be 2D (E,E), got {tuple(adj.shape)}")
        if adj.shape[0] != self.n_edges or adj.shape[1] != self.n_edges:
            raise ValueError(
                f"adjacency shape mismatch: expected ({self.n_edges}, {self.n_edges}), got {tuple(adj.shape)}"
            )
        self._adj = adj
        self._has_adj = True

    def set_static_features(self, static_features: torch.Tensor) -> None:
        if static_features.dim() != 2:
            raise ValueError(
                f"static_features must be 2D (E, n_static), got {tuple(static_features.shape)}"
            )
        if static_features.shape[0] != self.n_edges:
            raise ValueError(
                f"static_features edge dim mismatch: expected {self.n_edges}, got {int(static_features.shape[0])}"
            )
        if self.n_static == 0:
            self.n_static = int(static_features.shape[1])
        if static_features.shape[1] != self.n_static:
            raise ValueError(
                f"static_features feature dim mismatch: expected {self.n_static}, got {int(static_features.shape[1])}"
            )
        self._static_features = static_features
        self._has_static = True

    def forward(
        self,
        actions: Optional[torch.Tensor],
        *,
        batch_time_shape: Optional[Tuple[int, int]] = None,
        act_mask: Optional[torch.Tensor] = None,
        as_tokens: bool = True,
    ) -> torch.Tensor:
        if actions is None:
            assert batch_time_shape is not None
            B, T = batch_time_shape
            out = self.base.view(1, 1, -1).expand(B, T, -1)
            return out[:, :, None, :] if as_tokens else out

        if actions.dim() != 3:
            raise ValueError(f"EdgeGNNActionEncoder expects actions (B,T,E), got {tuple(actions.shape)}")
        if int(actions.shape[-1]) != self.n_edges:
            raise ValueError(
                f"EdgeGNNActionEncoder expects action_dim=n_edges={self.n_edges}, got {int(actions.shape[-1])}"
            )

        x = actions
        if act_mask is not None:
            x = x * act_mask
        x = x.clamp(-1, 1)

        B, T, E = x.shape
        x_edge = x.unsqueeze(-1)  # (B,T,E,1)
        if self.n_static > 0:
            if not self._has_static:
                raise RuntimeError("EdgeGNNActionEncoder requires static_features when n_static > 0")
            sf = self._static_features.to(device=x.device, dtype=x.dtype).view(1, 1, E, self.n_static)
            sf = sf.expand(B, T, E, self.n_static)
            x_in = torch.cat([x_edge, sf], dim=-1)
        else:
            x_in = x_edge

        h = self.input_proj(x_in)  # (B,T,E,d_edge)
        if self.n_gnn_layers > 0 and self._has_adj:
            adj = self._adj.to(device=h.device, dtype=h.dtype)
            h_bt = h.reshape(B * T, E, self.d_edge)
            for layer in self.gnn_layers:
                h_bt = layer(h_bt, adj)
            h = h_bt.view(B, T, E, self.d_edge)

        pooled = h.mean(dim=2)  # (B,T,d_edge)
        out = self.fc2(F.silu(self.fc1(pooled))) + self.base.view(1, 1, -1)
        return out[:, :, None, :] if as_tokens else out


class ActionDepartGNNEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_edges: int,
        n_static: int,
        d_edge: int = 64,
        n_gnn_layers: int = 2,
        context_dim: int = 3,
        context_hidden: int = 64,
        use_hdv_context: bool = False,
    ):
        super().__init__()
        self.n_edges = int(n_edges)
        self.n_static = int(max(0, n_static))
        self.d_model = int(d_model)
        self.context_dim = int(max(0, context_dim))
        self.use_hdv_context = bool(use_hdv_context)
        self.action_in = nn.Linear(1 + self.n_static, d_edge)
        self.depart_in = nn.Linear(1 + self.n_static, d_edge)
        self.action_gnn = nn.ModuleList([EdgeGNNLayer(d_edge) for _ in range(n_gnn_layers)])
        self.depart_gnn = nn.ModuleList([EdgeGNNLayer(d_edge) for _ in range(n_gnn_layers)])
        self.ctx_mlp = nn.Sequential(
            nn.Linear(self.context_dim, context_hidden),
            nn.SiLU(),
            nn.Linear(context_hidden, context_hidden),
        )
        self.out = nn.Linear(4 * d_edge + context_hidden, d_model)
        if self.use_hdv_context:
            self.hdv_context_in = nn.Linear(1 + self.n_static, d_edge)
            self.hdv_context_gnn = nn.ModuleList([EdgeGNNLayer(d_edge) for _ in range(n_gnn_layers)])
            self.hdv_context_out = nn.Linear(2 * d_edge, d_model)
            nn.init.normal_(self.hdv_context_out.weight, std=0.02)
            nn.init.zeros_(self.hdv_context_out.bias)
        else:
            self.hdv_context_in = None
            self.hdv_context_gnn = None
            self.hdv_context_out = None
        self.register_buffer("_static_features", torch.empty(0), persistent=False)
        self.register_buffer("_adj", torch.empty(0), persistent=False)
        self._has_static = False
        self._has_adj = False

    def set_static_features(self, static_features: torch.Tensor) -> None:
        if static_features.dim() != 2:
            raise ValueError(
                f"static_features must be 2D (E, n_static), got {tuple(static_features.shape)}"
            )
        if int(static_features.shape[0]) != self.n_edges:
            raise ValueError(
                f"static_features edge dim mismatch: expected {self.n_edges}, got {int(static_features.shape[0])}"
            )
        if int(static_features.shape[1]) != self.n_static:
            raise ValueError(
                f"static_features feature dim mismatch: expected {self.n_static}, got {int(static_features.shape[1])}"
            )
        self._static_features = static_features
        self._has_static = True

    def set_adjacency(self, adj: torch.Tensor) -> None:
        if adj.dim() != 2:
            raise ValueError(f"adjacency must be 2D (E,E), got {tuple(adj.shape)}")
        if adj.shape != (self.n_edges, self.n_edges):
            raise ValueError(
                f"adjacency shape mismatch: expected ({self.n_edges}, {self.n_edges}), got {tuple(adj.shape)}"
            )
        self._adj = adj
        self._has_adj = True

    def forward(
        self,
        action: torch.Tensor,
        *,
        depart: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        hdv_context: Optional[torch.Tensor] = None,
        as_tokens: bool = True,
        **_: object,
    ) -> torch.Tensor:
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if action.dim() != 3 or int(action.shape[-1]) != self.n_edges:
            raise ValueError(f"ActionDepartGNNEncoder expects action shape (B,T,{self.n_edges}), got {tuple(action.shape)}")
        if depart is None:
            depart = torch.zeros_like(action)
        elif depart.dim() == 2:
            depart = depart.unsqueeze(1)
        if depart.shape != action.shape:
            raise ValueError(f"depart shape must match action shape {tuple(action.shape)}, got {tuple(depart.shape)}")
        if context is None:
            context = torch.zeros(
                action.shape[0], action.shape[1], self.context_dim, device=action.device, dtype=action.dtype
            )
        elif context.dim() == 2:
            context = context.unsqueeze(1)
        if context.dim() != 3 or int(context.shape[0]) != int(action.shape[0]) or int(context.shape[1]) != int(action.shape[1]):
            raise ValueError(
                f"context must have shape (B,T,{self.context_dim}), got {tuple(context.shape)}"
            )
        if int(context.shape[-1]) != self.context_dim:
            raise ValueError(
                f"context feature dim mismatch: expected {self.context_dim}, got {int(context.shape[-1])}"
            )
        if self.use_hdv_context:
            if hdv_context is None:
                hdv_context = torch.zeros_like(action)
            elif hdv_context.dim() == 2:
                hdv_context = hdv_context.unsqueeze(1)
            if hdv_context.shape != action.shape:
                raise ValueError(
                    f"hdv_context shape must match action shape {tuple(action.shape)}, got {tuple(hdv_context.shape)}"
                )
        if self.n_static > 0 and not self._has_static:
            raise RuntimeError("ActionDepartGNNEncoder requires static_features when n_static > 0")
        if not self._has_adj:
            raise RuntimeError("ActionDepartGNNEncoder requires adjacency before forward")

        bsz, seq, n_edges = action.shape
        if self.n_static > 0:
            static = self._static_features.to(device=action.device, dtype=action.dtype)
            static = static.view(1, 1, n_edges, self.n_static).expand(bsz, seq, n_edges, self.n_static)
            action_in = torch.cat([action.unsqueeze(-1), static], dim=-1)
            depart_in = torch.cat([depart.unsqueeze(-1), static], dim=-1)
            hdv_in = (
                torch.cat([hdv_context.unsqueeze(-1), static], dim=-1)
                if self.use_hdv_context and hdv_context is not None else None
            )
        else:
            action_in = action.unsqueeze(-1)
            depart_in = depart.unsqueeze(-1)
            hdv_in = hdv_context.unsqueeze(-1) if self.use_hdv_context and hdv_context is not None else None

        action_h = self.action_in(action_in)
        depart_h = self.depart_in(depart_in)
        adj = self._adj.to(device=action.device, dtype=action.dtype)
        for layer in self.action_gnn:
            action_h = layer(action_h, adj)
        for layer in self.depart_gnn:
            depart_h = layer(depart_h, adj)

        pooled_parts = [
            action_h.mean(dim=2),
            action_h.max(dim=2).values,
            depart_h.mean(dim=2),
            depart_h.max(dim=2).values,
            self.ctx_mlp(context),
        ]
        out = self.out(torch.cat(pooled_parts, dim=-1))
        if self.use_hdv_context and hdv_in is not None and self.hdv_context_in is not None:
            hdv_h = self.hdv_context_in(hdv_in)
            assert self.hdv_context_gnn is not None
            for layer in self.hdv_context_gnn:
                hdv_h = layer(hdv_h, adj)
            hdv_pooled = torch.cat(
                [
                    hdv_h.mean(dim=2),
                    hdv_h.max(dim=2).values,
                ],
                dim=-1,
            )
            assert self.hdv_context_out is not None
            out = out + self.hdv_context_out(hdv_pooled)
        return out[:, :, None, :] if as_tokens else out


class FlowTimeEmbedder(nn.Module):
    """Embed continuous CFM time t in [0, 1] into d_model token."""

    def __init__(self, d_model: int, hidden_mult: float = 2.0):
        super().__init__()
        hidden = int(d_model * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, t: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
        if t.shape != (B, T):
            raise ValueError(f"flow_time shape mismatch: expected {(B, T)}, got {tuple(t.shape)}")
        emb = self.net(t.unsqueeze(-1))
        return emb.unsqueeze(2)


def route_indices_to_fracs(
    route_indices: torch.Tensor,
    n_routes: int,
) -> torch.Tensor:
    """
    Convert per-vehicle discrete route indices to a route-fraction vector.

    Args:
        route_indices: (..., N_veh) long, values in [0, n_routes)
    Returns:
        (..., K) float, sums to 1 along the last dimension
    """
    one_hot = F.one_hot(route_indices.long(), num_classes=n_routes).float()
    return one_hot.mean(dim=-2)


def combine_av_hdv_fracs(
    av_indices: Optional[torch.Tensor],   # (B, T, N_av) long  or  (B, T)
    hdv_indices: Optional[torch.Tensor],  # (B, T, N_hdv) long or None
    n_routes: int,
) -> Optional[torch.Tensor]:             # (B, T, K) float  or None
    """
    Merge AV and HDV per-vehicle route indices into a single joint-fraction vector.

    The world model is agnostic to vehicle identity; only the aggregate
    route distribution matters for predicting traffic physics.
    Returns None when both inputs are None.
    """
    if av_indices is None and hdv_indices is None:
        return None

    parts: list[torch.Tensor] = []
    for idx in (av_indices, hdv_indices):
        if idx is None:
            continue
        if idx.dim() == 2:          # (B, T) → (B, T, 1)
            idx = idx.unsqueeze(-1)
        parts.append(idx.long())

    all_indices = torch.cat(parts, dim=-1)  # (B, T, N_av + N_hdv)
    return route_indices_to_fracs(all_indices, n_routes)


class TaskEmbedder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_agent: int = 1,
        use_ids: bool = True,
        n_tasks: int = 128,
        d_task: int = 64,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.n_agent = int(n_agent)
        self.use_ids = bool(use_ids)
        self.n_tasks = int(n_tasks)
        self.d_task = int(d_task)

        if self.use_ids:
            self.task_table = nn.Embedding(self.n_tasks, self.d_model)
        else:
            self.task_proj = nn.Linear(self.d_task, self.d_model)

        self.agent_base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.agent_base, std=0.02)

    def forward(self, task: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
        if self.use_ids:
            emb = self.task_table(task.to(torch.long))  # (B,D)
        else:
            emb = self.task_proj(task)                  # (B,D)

        x = emb + self.agent_base.view(1, -1)          # (B,D)
        return x[:, None, None, :].expand(B, T, self.n_agent, self.d_model)


class EdgeActionTokenizer(nn.Module):
    """
    Dreamer-style tokenizer for edge-indexed continuous actions.

    Each edge is an action type with a learned embedding. The scalar action
    value modulates that edge embedding, and active edge tokens are pooled into
    one global action token for the transformer ACTION slot.
    """

    def __init__(
        self,
        *,
        n_edges: int,
        d_model: int,
        eps: float = 1e-6,
        value_norm: str = "raw",
    ):
        super().__init__()
        self.n_edges = int(n_edges)
        self.d_model = int(d_model)
        self.eps = float(eps)
        self.value_norm = str(value_norm).strip().lower()
        if self.value_norm not in ("raw", "active_count"):
            raise ValueError(
                f"Unsupported edge action value_norm={value_norm!r}; expected 'raw' or 'active_count'"
            )

        self.edge_embedding = nn.Embedding(self.n_edges, self.d_model)
        self.action_type_embedding = nn.Parameter(torch.empty(self.d_model))
        self.global_action_base = nn.Parameter(torch.empty(self.d_model))
        nn.init.normal_(self.edge_embedding.weight, std=0.02)
        nn.init.normal_(self.action_type_embedding, std=0.02)
        nn.init.normal_(self.global_action_base, std=0.02)

    def _mask_and_values(self, edge_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_actions = edge_actions.to(torch.float32)
        mask_values = (edge_actions.abs() > self.eps).to(edge_actions.dtype)
        if self.value_norm == "active_count":
            active_count = mask_values.sum(dim=-1, keepdim=True).clamp_min(1.0)
            values = edge_actions * active_count
        else:
            values = edge_actions
        return mask_values, values

    def forward(
        self,
        edge_actions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, E = edge_actions.shape
        if int(E) != self.n_edges:
            raise ValueError(f"edge action dim mismatch: expected {self.n_edges}, got {E}")

        mask_values, values = self._mask_and_values(edge_actions)
        edge_ids = torch.arange(self.n_edges, device=edge_actions.device)
        edge_emb = self.edge_embedding(edge_ids).to(dtype=dtype).view(1, 1, self.n_edges, self.d_model)
        values = values.to(dtype=dtype).unsqueeze(-1)
        active_mask = mask_values.to(dtype=dtype).unsqueeze(-1)

        edge_tokens = edge_emb * values + self.action_type_embedding.to(dtype=dtype).view(1, 1, 1, -1)
        edge_tokens = edge_tokens * active_mask
        denom = active_mask.sum(dim=2, keepdim=True).clamp_min(1.0).sqrt()
        global_token = edge_tokens.sum(dim=2, keepdim=True) / denom
        global_token = global_token + self.global_action_base.to(dtype=dtype).view(1, 1, 1, -1)
        return global_token, edge_tokens, active_mask


class Dynamics(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_bottleneck: int,
        d_spatial: int,
        n_spatial: int,
        n_register: int,
        n_agent: int,
        n_heads: int,
        depth: int,
        k_max: int,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        time_every: int = 4,
        space_mode: str = "wm_agent_isolated",  # or "wm_agent"
        action_encoder: Optional[nn.Module] = None,  # Custom action encoder
        use_cfm: bool = True,
        hdv_policy_dim: int = 0,
        hdv_policy_mode: str = "edge",
        edge_action_injection: bool = False,
        edge_action_injection_eps: float = 1e-6,
        edge_action_injection_scale: float = 1.0,
        edge_action_value_norm: str = "raw",
        edge_action_id_embedding: bool = False,
        edge_action_tokenizer: bool = False,
        edge_action_tokenizer_fuse_encoder: bool = False,
        edge_action_output_injection: bool = False,
        edge_action_output_from_injection: bool = False,
        action_reconstruction: bool = False,
    ):
        super().__init__()
        assert d_spatial % d_bottleneck == 0, "expected packing: d_spatial = d_bottleneck * packing_factor"
        self.d_model = int(d_model)
        self.d_spatial = int(d_spatial)
        self.n_spatial = int(n_spatial)
        self.n_register = int(n_register)
        self.n_agent = int(n_agent)
        self.k_max = int(k_max)
        self.use_cfm = bool(use_cfm)
        self.edge_action_injection = bool(edge_action_injection)
        self.edge_action_injection_eps = float(edge_action_injection_eps)
        self.edge_action_injection_scale = float(edge_action_injection_scale)
        self.edge_action_value_norm = str(edge_action_value_norm).strip().lower()
        if self.edge_action_value_norm not in ("raw", "active_count"):
            raise ValueError(
                f"Unsupported edge_action_value_norm={edge_action_value_norm!r}; "
                "expected 'raw' or 'active_count'"
            )
        self.edge_action_id_embedding_enabled = bool(edge_action_id_embedding)
        self.edge_action_tokenizer_enabled = bool(edge_action_tokenizer)
        self.edge_action_tokenizer_fuse_encoder = bool(edge_action_tokenizer_fuse_encoder)
        self.edge_action_output_injection = bool(edge_action_output_injection)
        self.edge_action_output_from_injection = bool(edge_action_output_from_injection)
        if not self.use_cfm:
            raise ValueError("Dynamics now supports CFM only (use_cfm must be True).")
        self.hdv_policy_dim = int(hdv_policy_dim)
        self.hdv_policy_mode = str(hdv_policy_mode).lower()
        if self.hdv_policy_mode != "edge":
            raise ValueError("Dynamics now supports edge-level HDV head only (hdv_policy_mode='edge').")

        self.spatial_proj = nn.Linear(self.d_spatial, self.d_model)
        self.register_tokens = nn.Parameter(torch.empty(self.n_register, self.d_model))
        nn.init.normal_(self.register_tokens, std=0.02)

        # Use custom action encoder if provided, otherwise default ActionEncoder
        if action_encoder is not None:
            self.action_encoder = action_encoder
        else:
            self.action_encoder = ActionEncoder(d_model=self.d_model, action_dim=16)

        if self.edge_action_tokenizer_enabled:
            self.edge_action_tokenizer = EdgeActionTokenizer(
                n_edges=self.n_spatial,
                d_model=self.d_model,
                eps=self.edge_action_injection_eps,
                value_norm=self.edge_action_value_norm,
            )
        else:
            self.edge_action_tokenizer = None

        edge_action_input_dim = 2 if self.edge_action_value_norm == "active_count" else 1
        if self.edge_action_injection and self.edge_action_tokenizer is None:
            self.edge_action_proj = nn.Linear(edge_action_input_dim, self.d_model)
            nn.init.zeros_(self.edge_action_proj.weight)
            nn.init.zeros_(self.edge_action_proj.bias)
            if self.edge_action_id_embedding_enabled:
                self.edge_action_id_embedding = nn.Embedding(self.n_spatial, self.d_model)
                nn.init.normal_(self.edge_action_id_embedding.weight, std=0.02)
            else:
                self.edge_action_id_embedding = None
        else:
            self.edge_action_proj = None
            self.edge_action_id_embedding = None
        self.use_hdv_context = bool(getattr(self.action_encoder, "use_hdv_context", False))
        if self.use_hdv_context:
            self.hdv_context_edge_proj = nn.Linear(edge_action_input_dim, self.d_model)
            nn.init.normal_(self.hdv_context_edge_proj.weight, std=0.02)
            nn.init.zeros_(self.hdv_context_edge_proj.bias)
        else:
            self.hdv_context_edge_proj = None

        self.time_embedder = FlowTimeEmbedder(self.d_model)
        segments = [
            (Modality.ACTION, 1),
            (Modality.FLOW_TIME, 1),
            (Modality.SPATIAL, self.n_spatial),
            (Modality.REGISTER, self.n_register),
        ]
        if self.n_agent > 0:
            segments.append((Modality.AGENT, self.n_agent))

        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.action_slice = sl[Modality.ACTION]
        self.spatial_slice = sl[Modality.SPATIAL]
        self.agent_slice = sl.get(Modality.AGENT, slice(0, 0))
        modality_ids = self.layout.modality_ids()

        self.transformer = BlockCausalTransformer(
            d_model=self.d_model,
            n_heads=int(n_heads),
            depth=int(depth),
            n_latents=0,
            modality_ids=modality_ids,
            space_mode=space_mode,
            dropout=float(dropout),
            mlp_ratio=float(mlp_ratio),
            time_every=int(time_every),
            latents_only_time=False,
        )

        self.flow_x_head = nn.Linear(self.d_model, self.d_spatial)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)
        if self.edge_action_output_injection and (
            self.edge_action_tokenizer_enabled or self.edge_action_output_from_injection
        ):
            self.edge_action_output_proj = nn.Linear(self.d_model, self.d_spatial)
            nn.init.zeros_(self.edge_action_output_proj.weight)
            nn.init.zeros_(self.edge_action_output_proj.bias)
        else:
            self.edge_action_output_proj = None
        if bool(action_reconstruction):
            self.action_recon_head = nn.Linear(self.d_model, self.n_spatial * 2)
            nn.init.normal_(self.action_recon_head.weight, std=0.02)
            nn.init.zeros_(self.action_recon_head.bias)
        else:
            self.action_recon_head = None
        if self.hdv_policy_dim > 0:
            if self.hdv_policy_dim != self.n_spatial:
                raise ValueError(
                    "edge HDV policy head requires hdv_policy_dim == n_spatial "
                    f"({self.n_spatial}), got {self.hdv_policy_dim}"
                )
            # Per-edge HDV distribution decoder from latent spatial tokens.
            self.hdv_policy_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 1),
            )
            nn.init.zeros_(self.hdv_policy_head[-1].weight)
            nn.init.zeros_(self.hdv_policy_head[-1].bias)
        else:
            self.hdv_policy_head = None

    def forward(
        self,
        actions: Optional[torch.Tensor],          # (B,T,16) or None
        step_idxs: Optional[torch.Tensor],        # unused (kept for API compatibility)
        signal_idxs: Optional[torch.Tensor],      # unused (kept for API compatibility)
        packed_enc_tokens: torch.Tensor,          # (B,T,n_spatial,d_spatial)
        *,
        act_mask: Optional[torch.Tensor] = None,  # (B,T,16) or (16,) or None
        agent_tokens: Optional[torch.Tensor] = None,
        flow_time: Optional[torch.Tensor] = None,  # required (B,T) in CFM mode
        return_hdv_policy: bool = False,
        return_action_recon: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
        B, T = packed_enc_tokens.shape[:2]

        spatial_tokens = self.spatial_proj(packed_enc_tokens)  # (B,T,n_spatial,d_model)

        edge_action_tokens = None
        edge_action_values = self._edge_action_values(
            actions,
            batch_time_shape=(B, T),
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        tokenizer_action_tokens = None
        if self.edge_action_tokenizer is not None and edge_action_values is not None:
            tokenizer_action_tokens, edge_action_tokens, _ = self.edge_action_tokenizer(
                edge_action_values,
                dtype=spatial_tokens.dtype,
            )
        if tokenizer_action_tokens is not None and not self.edge_action_tokenizer_fuse_encoder:
            action_tokens = tokenizer_action_tokens
        else:
            if isinstance(actions, tuple):
                hdv_context = actions[3] if len(actions) > 3 else None
                action_tokens = self.action_encoder(
                    actions[0],
                    depart=actions[1],
                    context=actions[2],
                    hdv_context=hdv_context,
                    as_tokens=True,
                )
            else:
                action_tokens = self.action_encoder(
                    actions,
                    batch_time_shape=(B, T),
                    act_mask=act_mask,
                    as_tokens=True,
                )  # (B,T,1,d_model)
            if tokenizer_action_tokens is not None:
                # Preserve the full encoder-conditioned ACTION token (e.g. HDV
                # departures and context) while still injecting edge-indexed
                # tokenizer identity into the transformer and optional output
                # residual path.
                action_tokens = action_tokens + tokenizer_action_tokens
        if edge_action_tokens is None and self.edge_action_output_from_injection:
            edge_action_tokens = self._edge_action_projected_tokens(
                actions,
                batch_time_shape=spatial_tokens.shape[:2],
                device=spatial_tokens.device,
                dtype=spatial_tokens.dtype,
            )
        spatial_tokens = self._inject_edge_actions(spatial_tokens, actions, edge_action_tokens=edge_action_tokens)
        spatial_tokens = self._inject_hdv_context(spatial_tokens, actions)

        reg = self.register_tokens.view(1, 1, self.n_register, self.d_model).expand(B, T, -1, -1)

        if flow_time is None:
            raise ValueError("flow_time is required in CFM-only Dynamics")
        time_tok = self.time_embedder(flow_time.to(spatial_tokens.dtype), B=B, T=T)
        if self.n_agent > 0:
            if agent_tokens is None:
                agent_tokens = torch.zeros(
                    (B, T, self.n_agent, self.d_model),
                    device=spatial_tokens.device,
                    dtype=spatial_tokens.dtype,
                )
            toks = [action_tokens, time_tok, spatial_tokens, reg, agent_tokens]
        else:
            toks = [action_tokens, time_tok, spatial_tokens, reg]

        tokens = torch.cat(toks, dim=2)  # (B,T,S,D)
        tokens = add_sinusoidal_positions(tokens)
        x = self.transformer(tokens)

        spatial_out = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial_out)  # (B,T,n_spatial,d_spatial)
        if self.edge_action_output_proj is not None and edge_action_tokens is not None:
            # Direct per-edge action residual; zero-initialized so it is opt-in and
            # learned only when action-conditioned losses need this path.
            x1_hat = x1_hat + self.edge_action_output_proj(edge_action_tokens) * self.edge_action_injection_scale
        action_recon = None
        if return_action_recon and self.action_recon_head is not None:
            action_hidden = x[:, :, self.action_slice, :].mean(dim=2)
            action_recon = self.action_recon_head(action_hidden).view(B, T, self.n_spatial, 2)

        h_t = None
        if self.n_agent > 0:
            h_t = x[:, :, self.agent_slice, :]   # (B,T,n_agent,d_model)

        hdv_logits = None
        if return_hdv_policy and self.hdv_policy_head is not None:
            # Edge-level distribution from latent spatial tokens (B,T,E,D)->(B,T,E).
            hdv_logits = self.hdv_policy_head(spatial_out).squeeze(-1)

        if return_hdv_policy and return_action_recon:
            return x1_hat, h_t, hdv_logits, action_recon
        if return_hdv_policy:
            return x1_hat, h_t, hdv_logits
        if return_action_recon:
            return x1_hat, h_t, action_recon
        return x1_hat, h_t

    def _edge_action_values(
        self,
        actions: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        *,
        batch_time_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if isinstance(actions, tuple):
            edge_actions = actions[0]
        else:
            edge_actions = actions
        if edge_actions is None:
            return None
        B, T = batch_time_shape
        if edge_actions.dim() == 2:
            edge_actions = edge_actions.unsqueeze(1)
        if edge_actions.dim() != 3:
            return None
        if int(edge_actions.shape[0]) not in (1, B):
            return None
        if int(edge_actions.shape[0]) == 1 and B > 1:
            edge_actions = edge_actions.expand(B, -1, -1)
        action_T = int(edge_actions.shape[1])
        if action_T == 1 and T > 1:
            edge_actions = edge_actions.expand(-1, T, -1)
        elif action_T > T:
            edge_actions = edge_actions[:, -T:, :]
        elif action_T != T:
            return None
        if int(edge_actions.shape[-1]) < self.n_spatial:
            return None
        return edge_actions[..., : self.n_spatial].to(device=device, dtype=dtype)

    def _inject_edge_actions(
        self,
        spatial_tokens: torch.Tensor,
        actions: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        *,
        edge_action_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.edge_action_injection:
            return spatial_tokens
        if edge_action_tokens is not None:
            return spatial_tokens + edge_action_tokens * self.edge_action_injection_scale
        projected = self._edge_action_projected_tokens(
            actions,
            batch_time_shape=spatial_tokens.shape[:2],
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        if projected is None:
            return spatial_tokens
        return spatial_tokens + projected * self.edge_action_injection_scale

    def _inject_hdv_context(
        self,
        spatial_tokens: torch.Tensor,
        actions: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
    ) -> torch.Tensor:
        if self.hdv_context_edge_proj is None or not isinstance(actions, tuple) or len(actions) <= 3:
            return spatial_tokens
        hdv_context = actions[3]
        if hdv_context is None:
            return spatial_tokens
        edge_context = self._edge_action_values(
            hdv_context,
            batch_time_shape=spatial_tokens.shape[:2],
            device=spatial_tokens.device,
            dtype=spatial_tokens.dtype,
        )
        if edge_context is None:
            return spatial_tokens
        mask, features = self._edge_action_features(edge_context, dtype=spatial_tokens.dtype)
        return spatial_tokens + self.hdv_context_edge_proj(features) * mask * self.edge_action_injection_scale

    def _edge_action_projected_tokens(
        self,
        actions: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        *,
        batch_time_shape: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.edge_action_proj is None:
            return None
        edge_actions = self._edge_action_values(
            actions,
            batch_time_shape=batch_time_shape,
            device=device,
            dtype=dtype,
        )
        if edge_actions is None:
            return None
        mask, action_features = self._edge_action_features(edge_actions, dtype=dtype)
        injected = self.edge_action_proj(action_features)
        if self.edge_action_id_embedding is not None:
            edge_ids = torch.arange(self.n_spatial, device=device)
            injected = injected + self.edge_action_id_embedding(edge_ids).view(1, 1, self.n_spatial, self.d_model)
        return injected * mask

    def _edge_action_features(self, edge_actions: torch.Tensor, *, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_actions = edge_actions.to(dtype=dtype)
        mask_values = (edge_actions.abs() > self.edge_action_injection_eps).to(dtype)
        mask = mask_values.unsqueeze(-1)
        if self.edge_action_value_norm == "active_count":
            active_count = mask_values.sum(dim=-1, keepdim=True).clamp_min(1.0)
            action_features = torch.stack([mask_values, edge_actions * active_count], dim=-1)
        else:
            action_features = edge_actions.unsqueeze(-1)
        return mask, action_features
