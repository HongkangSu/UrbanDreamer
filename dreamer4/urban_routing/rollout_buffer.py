"""
Rollout Buffer for managing historical states during world model rollout.

This module provides a buffer that maintains a sliding window of past latent states
for autoregressive prediction in the world model.
"""

import torch
from typing import Optional, List
from collections import deque


class RolloutBuffer:
    """
    Buffer for managing historical latent states during world model rollout.

    Maintains a sliding window of past packed latent representations to enable
    autoregressive prediction in the dynamics model.

    Args:
        max_length: Maximum number of historical states to keep
        n_spatial: Number of spatial tokens
        d_spatial: Dimension of each spatial token
        device: Device to store tensors on
        dtype: Data type for tensors
    """

    def __init__(
        self,
        max_length: int = 16,
        n_spatial: Optional[int] = None,
        d_spatial: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.max_length = max_length
        self.n_spatial = n_spatial
        self.d_spatial = d_spatial
        self.device = device
        self.dtype = dtype

        # Use deque for efficient append and popleft operations
        self.buffer: deque = deque(maxlen=max_length)

    def add(self, z_packed: torch.Tensor):
        """
        Add a new latent state to the buffer.

        Args:
            z_packed: Packed latent state of shape (n_spatial, d_spatial) or
                     (1, n_spatial, d_spatial)
        """
        # Ensure 2D: (n_spatial, d_spatial)
        if z_packed.dim() == 3:
            z_packed = z_packed.squeeze(0)

        assert z_packed.dim() == 2, f"Expected 2D tensor, got {z_packed.dim()}D"

        # Update shape info if not set
        if self.n_spatial is None:
            self.n_spatial = z_packed.shape[0]
        if self.d_spatial is None:
            self.d_spatial = z_packed.shape[1]

        # Validate shape
        assert z_packed.shape == (self.n_spatial, self.d_spatial), \
            f"Shape mismatch: expected ({self.n_spatial}, {self.d_spatial}), got {z_packed.shape}"

        # Add to buffer (will auto-remove oldest if at max_length)
        self.buffer.append(z_packed.detach().clone())

    def get_past_packed(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get all historical states as a batched tensor.

        Args:
            batch_size: Batch size to expand to

        Returns:
            Tensor of shape (B, t, n_spatial, d_spatial) where t is the number
            of states in the buffer
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty. Add at least one state first.")

        # Stack all states: (t, n_spatial, d_spatial)
        stacked = torch.stack(list(self.buffer), dim=0)

        # Add batch dimension and expand: (1, t, n_spatial, d_spatial) -> (B, t, n_spatial, d_spatial)
        batched = stacked.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return batched.to(device=self.device, dtype=self.dtype)

    def get_latest(self) -> Optional[torch.Tensor]:
        """
        Get the most recent latent state.

        Returns:
            Latest state of shape (n_spatial, d_spatial) or None if buffer is empty
        """
        if len(self.buffer) == 0:
            return None
        return self.buffer[-1]

    def reset(self):
        """Clear the buffer."""
        self.buffer.clear()

    def __len__(self) -> int:
        """Return the number of states in the buffer."""
        return len(self.buffer)

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return len(self.buffer) >= self.max_length

    def clone(self) -> 'RolloutBuffer':
        """Create a deep copy of this buffer."""
        new_buffer = RolloutBuffer(
            max_length=self.max_length,
            n_spatial=self.n_spatial,
            d_spatial=self.d_spatial,
            device=self.device,
            dtype=self.dtype,
        )
        for state in self.buffer:
            new_buffer.add(state.clone())
        return new_buffer

    def __repr__(self) -> str:
        return (f"RolloutBuffer(length={len(self)}/{self.max_length}, "
                f"shape=({self.n_spatial}, {self.d_spatial}), "
                f"device={self.device})")


class BatchedRolloutBuffer:
    """
    Batched version of RolloutBuffer for handling multiple rollouts simultaneously.

    Useful for parallel evaluation or when rolling out multiple trajectories.

    Args:
        batch_size: Number of independent buffers
        max_length: Maximum length for each buffer
        n_spatial: Number of spatial tokens
        d_spatial: Dimension of each spatial token
        device: Device to store tensors on
        dtype: Data type for tensors
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int = 16,
        n_spatial: Optional[int] = None,
        d_spatial: Optional[int] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.batch_size = batch_size
        self.buffers = [
            RolloutBuffer(
                max_length=max_length,
                n_spatial=n_spatial,
                d_spatial=d_spatial,
                device=device,
                dtype=dtype,
            )
            for _ in range(batch_size)
        ]

    def add(self, z_packed_batch: torch.Tensor):
        """
        Add batched latent states to buffers.

        Args:
            z_packed_batch: Batched states of shape (B, n_spatial, d_spatial)
        """
        assert z_packed_batch.shape[0] == self.batch_size, \
            f"Batch size mismatch: expected {self.batch_size}, got {z_packed_batch.shape[0]}"

        for i, buffer in enumerate(self.buffers):
            buffer.add(z_packed_batch[i])

    def get_past_packed(self) -> torch.Tensor:
        """
        Get all historical states as a batched tensor.

        Returns:
            Tensor of shape (B, t, n_spatial, d_spatial)
        """
        # Get from first buffer with batch_size=1, then stack
        all_past = [buf.get_past_packed(batch_size=1) for buf in self.buffers]
        return torch.cat(all_past, dim=0)  # (B, t, n_spatial, d_spatial)

    def reset(self):
        """Clear all buffers."""
        for buffer in self.buffers:
            buffer.reset()

    def reset_index(self, idx: int):
        """Clear a specific buffer by index."""
        self.buffers[idx].reset()

    def __len__(self) -> int:
        """Return length of the first buffer (assumes all same length)."""
        return len(self.buffers[0]) if self.buffers else 0

    def __repr__(self) -> str:
        return f"BatchedRolloutBuffer(batch_size={self.batch_size}, {self.buffers[0]})"
