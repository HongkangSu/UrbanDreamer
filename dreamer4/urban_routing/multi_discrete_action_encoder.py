# multi_discrete_action_encoder.py
"""
Multi-Discrete Action Encoder for Urban Routing

This module provides an action encoder that handles multi-discrete action spaces,
where multiple decision points (junctions) can make simultaneous decisions.
"""

import torch
import torch.nn as nn
from typing import Optional


class MultiDiscreteActionEncoder(nn.Module):
    """
    Encoder for multi-discrete action spaces.

    Handles scenarios where multiple independent discrete actions occur simultaneously,
    such as multiple traffic junctions making routing decisions.

    Args:
        d_model: Model dimension (must match dynamics model)
        n_action_spaces: Number of independent action spaces (e.g., number of junctions)
        max_choices_per_space: Maximum number of choices per action space
        hidden_mult: Hidden layer size multiplier
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_action_spaces: int,
        max_choices_per_space: int = 4,
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_action_spaces = n_action_spaces
        self.max_choices = max_choices_per_space

        # Calculate embedding dimension per action space
        # Use a fixed embedding size and project to d_model
        self.embed_dim_per_space = 16  # Fixed embedding size per action

        # Embedding for each discrete choice
        # We use a single embedding layer that can handle all possible choices
        # Including -1 for "no action" padding
        self.action_embedding = nn.Embedding(
            num_embeddings=max_choices_per_space + 1,  # +1 for padding (-1 -> 0)
            embedding_dim=self.embed_dim_per_space,
            padding_idx=0  # Map -1 to index 0 (padding)
        )

        # Project concatenated embeddings to d_model
        self.input_proj = nn.Linear(n_action_spaces * self.embed_dim_per_space, d_model)

        # Fusion layer to combine all action embeddings
        hidden_dim = int(d_model * hidden_mult)
        self.fusion = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        actions: Optional[torch.Tensor],  # (B, T, n_action_spaces) or None
        *,
        batch_time_shape: Optional[tuple] = None,
        act_mask: Optional[torch.Tensor] = None,  # (B, T, n_action_spaces) or None
        as_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Encode multi-discrete actions.

        Args:
            actions: Integer tensor of shape (B, T, n_action_spaces) with values in [0, max_choices)
                    or -1 for no action. Can be None for unlabeled pretraining.
            batch_time_shape: Tuple (B, T) required when actions is None
            act_mask: Optional mask for actions
            as_tokens: If True, return shape (B, T, 1, d_model), else (B, T, d_model)

        Returns:
            Encoded actions of shape (B, T, 1, d_model) if as_tokens else (B, T, d_model)
        """
        # Handle None actions (unlabeled pretraining)
        if actions is None:
            assert batch_time_shape is not None, "batch_time_shape required when actions is None"
            B, T = batch_time_shape
            # Return a learned base token
            base_token = torch.zeros(B, T, self.d_model, device=self.input_proj.weight.device)
            return base_token[:, :, None, :] if as_tokens else base_token

        B, T, N = actions.shape
        assert N == self.n_action_spaces, \
            f"Expected {self.n_action_spaces} action spaces, got {N}"

        # Shift actions: -1 -> 0 (padding), 0 -> 1, 1 -> 2, etc.
        actions_shifted = actions + 1  # (B, T, N)
        actions_shifted = actions_shifted.clamp(min=0, max=self.max_choices)

        # Embed each action
        embedded = self.action_embedding(actions_shifted)  # (B, T, N, embed_dim_per_space)

        # Flatten and concatenate all action embeddings
        embedded_flat = embedded.reshape(B, T, -1)  # (B, T, N * embed_dim_per_space)

        # Apply action mask if provided
        if act_mask is not None:
            # Expand mask to match embedding dimension
            mask_expanded = act_mask.unsqueeze(-1)  # (B, T, N, 1)
            mask_expanded = mask_expanded.expand(-1, -1, -1, self.embed_dim_per_space)
            mask_flat = mask_expanded.reshape(B, T, -1)  # (B, T, N * embed_dim_per_space)
            embedded_flat = embedded_flat * mask_flat

        # Project to d_model
        projected = self.input_proj(embedded_flat)  # (B, T, d_model)

        # Fusion layer
        fused = self.fusion(projected)  # (B, T, d_model)

        # Residual connection + norm
        output = self.norm(projected + fused)

        # Return with or without token dimension
        return output[:, :, None, :] if as_tokens else output

    def encode_single_timestep(
        self,
        action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a single timestep action.

        Args:
            action: (B, n_action_spaces) or (n_action_spaces,)
            action_mask: Optional (B, n_action_spaces) or (n_action_spaces,)

        Returns:
            (B, 1, d_model) or (1, d_model)
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)  # (1, N)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)  # (1, N)
            squeeze_output = True
        else:
            squeeze_output = False

        # Add time dimension
        action = action.unsqueeze(1)  # (B, 1, N)
        if action_mask is not None:
            action_mask = action_mask.unsqueeze(1)  # (B, 1, N)

        # Encode
        encoded = self(action, action_mask)  # (B, 1, 1, d_model)
        encoded = encoded.squeeze(2)  # (B, 1, d_model)

        if squeeze_output:
            encoded = encoded.squeeze(0)  # (1, d_model)

        return encoded


# Example usage and testing
if __name__ == "__main__":
    print("MultiDiscreteActionEncoder Test")
    print("=" * 60)

    # Configuration
    d_model = 256
    n_junctions = 50
    max_choices = 4
    batch_size = 4
    seq_length = 16

    # Create encoder
    encoder = MultiDiscreteActionEncoder(
        d_model=d_model,
        n_action_spaces=n_junctions,
        max_choices_per_space=max_choices
    )

    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Input shape: (B={batch_size}, T={seq_length}, N={n_junctions})")
    print(f"Output shape: (B={batch_size}, T={seq_length}, 1, d_model={d_model})")
    print()

    # Test 1: Random actions with some padding
    print("Test 1: Random multi-discrete actions")
    actions = torch.randint(-1, max_choices, (batch_size, seq_length, n_junctions))
    print(f"  Action range: [{actions.min()}, {actions.max()}]")
    print(f"  Padding ratio: {(actions == -1).float().mean():.2%}")

    encoded = encoder(actions)
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Encoded stats: mean={encoded.mean():.4f}, std={encoded.std():.4f}")
    print()

    # Test 2: With action mask
    print("Test 2: With action mask")
    action_mask = (actions != -1).float()
    encoded_masked = encoder(actions, action_mask)
    print(f"  Mask ratio: {action_mask.mean():.2%}")
    print(f"  Encoded shape: {encoded_masked.shape}")
    print(f"  Encoded stats: mean={encoded_masked.mean():.4f}, std={encoded_masked.std():.4f}")
    print()

    # Test 3: Single timestep
    print("Test 3: Single timestep encoding")
    single_action = torch.randint(-1, max_choices, (n_junctions,))
    single_encoded = encoder.encode_single_timestep(single_action)
    print(f"  Single action shape: {single_action.shape}")
    print(f"  Encoded shape: {single_encoded.shape}")
    print()

    # Test 4: All padding (no actions)
    print("Test 4: All padding (no actions)")
    padding_actions = torch.full((batch_size, seq_length, n_junctions), -1)
    padding_encoded = encoder(padding_actions)
    print(f"  Encoded shape: {padding_encoded.shape}")
    print(f"  Should be close to zero: {padding_encoded.abs().mean():.6f}")
    print()

    # Test 5: Gradient flow
    print("Test 5: Gradient flow test")
    actions_grad = torch.randint(-1, max_choices, (2, 4, n_junctions))
    encoded_grad = encoder(actions_grad)
    loss = encoded_grad.mean()
    loss.backward()
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Embedding grad norm: {encoder.action_embedding.weight.grad.norm():.6f}")
    print()

    print("=" * 60)
    print("All tests passed!")
