#!/usr/bin/env python3
"""
Traffic-based Reward Model

Implements reward functions based on traffic metrics from rendered bird's eye views
or latent states. Optimizes for global travel time minimization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class TrafficMetricsRewardModel(nn.Module):
    """
    Reward model based on traffic metrics extracted from bird's eye view images.

    Optimization objective: Minimize global travel time

    Key metrics:
    - Throughput: Number of fast-moving vehicles (green/cyan pixels)
    - Congestion: Number of stopped/slow vehicles (red/orange pixels)
    - Average speed: Overall traffic flow speed
    - Waiting time: Vehicles that are stopped

    Reward = throughput_weight * throughput - congestion_weight * congestion
    """

    def __init__(
        self,
        throughput_weight: float = 1.0,
        congestion_weight: float = 0.5,
        use_latent: bool = False,
        latent_dim: int = 32768,  # 1024 * 32
    ):
        """
        Args:
            throughput_weight: Weight for throughput reward (encourage fast movement)
            congestion_weight: Weight for congestion penalty (discourage traffic jams)
            use_latent: If True, predict metrics from latent states instead of images
            latent_dim: Dimension of latent state (for learned predictor)
        """
        super().__init__()

        self.throughput_weight = throughput_weight
        self.congestion_weight = congestion_weight
        self.use_latent = use_latent

        if use_latent:
            # Learned predictor from latent states to traffic metrics
            # Input: (z_current, z_next) concatenated
            # Output: (throughput, congestion, avg_speed)
            self.metric_predictor = nn.Sequential(
                nn.Linear(latent_dim * 2, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3),  # [throughput, congestion, avg_speed]
            )

        # Color thresholds for traffic metrics (BGR format)
        # Based on TrafficRenderer color scheme
        self.stopped_colors = [
            (255, 0, 0),      # Red: stopped vehicles
        ]
        self.slow_colors = [
            (255, 165, 0),    # Orange: slow vehicles
            (255, 255, 0),    # Yellow: medium speed
        ]
        self.fast_colors = [
            (0, 255, 0),      # Green: fast vehicles
            (0, 128, 255),    # Cyan: very fast vehicles
        ]

        # Junction congestion colors (green -> yellow -> red)
        self.congested_junction_colors = [
            (255, 255, 0),    # Yellow: medium congestion
            (255, 0, 0),      # Red: high congestion
        ]

    def forward(
        self,
        z_current: Optional[torch.Tensor] = None,
        z_next: Optional[torch.Tensor] = None,
        img_current: Optional[np.ndarray] = None,
        img_next: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute reward based on traffic state transition.

        Args:
            z_current: Current latent state (n_spatial, d_spatial)
            z_next: Next latent state (n_spatial, d_spatial)
            img_current: Current bird's eye view (H, W, 3) uint8
            img_next: Next bird's eye view (H, W, 3) uint8

        Returns:
            Scalar reward (higher = better traffic flow)
        """
        if self.use_latent and z_current is not None and z_next is not None:
            # Use learned predictor
            reward = self._compute_reward_from_latent(z_current, z_next)
        elif img_current is not None and img_next is not None:
            # Use image-based metrics
            reward = self._compute_reward_from_images(img_current, img_next)
        else:
            raise ValueError("Must provide either latent states or images")

        return reward

    def _compute_reward_from_latent(
        self,
        z_current: torch.Tensor,
        z_next: torch.Tensor,
    ) -> float:
        """Predict reward from latent states using learned model."""
        # Flatten latent states
        z_curr_flat = z_current.flatten().unsqueeze(0)  # (1, latent_dim)
        z_next_flat = z_next.flatten().unsqueeze(0)

        # Concatenate
        z_cat = torch.cat([z_curr_flat, z_next_flat], dim=1)  # (1, latent_dim * 2)

        # Predict metrics
        with torch.no_grad():
            metrics = self.metric_predictor(z_cat)  # (1, 3)

        throughput = metrics[0, 0].item()
        congestion = metrics[0, 1].item()
        avg_speed = metrics[0, 2].item()

        # Compute reward
        reward = (
            self.throughput_weight * throughput
            - self.congestion_weight * congestion
            + 0.1 * avg_speed  # Small bonus for higher average speed
        )

        return reward

    def _compute_reward_from_images(
        self,
        img_current: np.ndarray,
        img_next: np.ndarray,
    ) -> float:
        """
        Compute reward from bird's eye view images.

        Key insight: We want to maximize vehicles moving quickly,
        minimize stopped/slow vehicles, and reduce junction congestion.
        """
        # Extract metrics from next state
        metrics_next = self._extract_traffic_metrics(img_next)

        # Also check improvement from current state
        metrics_current = self._extract_traffic_metrics(img_current)

        # Reward components
        # 1. Throughput: vehicles moving fast
        throughput = metrics_next['fast_vehicle_count']

        # 2. Congestion penalty: stopped/slow vehicles
        congestion = metrics_next['stopped_count'] + 0.5 * metrics_next['slow_count']

        # 3. Junction congestion penalty
        junction_congestion = metrics_next['congested_junction_count']

        # 4. Improvement bonus: traffic getting better over time
        throughput_improvement = (
            metrics_next['fast_vehicle_count'] - metrics_current['fast_vehicle_count']
        )
        congestion_reduction = (
            metrics_current['stopped_count'] - metrics_next['stopped_count']
        )

        # Compute final reward
        reward = (
            self.throughput_weight * throughput
            - self.congestion_weight * congestion
            - 0.3 * junction_congestion
            + 0.2 * throughput_improvement  # Bonus for increasing throughput
            + 0.2 * congestion_reduction     # Bonus for reducing congestion
        )

        return reward

    def _extract_traffic_metrics(self, img: np.ndarray) -> Dict[str, int]:
        """
        Extract traffic metrics from bird's eye view image.

        Returns:
            Dict with keys:
                - stopped_count: Number of stopped vehicle pixels
                - slow_count: Number of slow vehicle pixels
                - fast_vehicle_count: Number of fast-moving vehicle pixels
                - congested_junction_count: Number of congested junction pixels
        """
        metrics = {
            'stopped_count': 0,
            'slow_count': 0,
            'fast_vehicle_count': 0,
            'congested_junction_count': 0,
        }

        # Convert to BGR for OpenCV color matching
        # (TrafficRenderer uses BGR format internally)
        if img.shape[2] == 3:
            img_bgr = img[..., ::-1]  # RGB -> BGR
        else:
            img_bgr = img

        # Count stopped vehicles (red pixels)
        for color in self.stopped_colors:
            mask = self._color_mask(img_bgr, color, tolerance=30)
            metrics['stopped_count'] += np.sum(mask)

        # Count slow vehicles (orange/yellow pixels)
        for color in self.slow_colors:
            mask = self._color_mask(img_bgr, color, tolerance=30)
            metrics['slow_count'] += np.sum(mask)

        # Count fast vehicles (green/cyan pixels)
        for color in self.fast_colors:
            mask = self._color_mask(img_bgr, color, tolerance=30)
            metrics['fast_vehicle_count'] += np.sum(mask)

        # Count congested junctions (yellow/red pixels that are larger blobs)
        for color in self.congested_junction_colors:
            mask = self._color_mask(img_bgr, color, tolerance=30)
            metrics['congested_junction_count'] += np.sum(mask)

        return metrics

    def _color_mask(
        self,
        img: np.ndarray,
        target_color: Tuple[int, int, int],
        tolerance: int = 30
    ) -> np.ndarray:
        """
        Create binary mask for pixels matching target color within tolerance.

        Args:
            img: Input image (H, W, 3) uint8
            target_color: Target BGR color tuple
            tolerance: Color matching tolerance

        Returns:
            Binary mask (H, W) bool
        """
        b, g, r = target_color

        # Create masks for each channel
        b_mask = np.abs(img[..., 0].astype(int) - b) <= tolerance
        g_mask = np.abs(img[..., 1].astype(int) - g) <= tolerance
        r_mask = np.abs(img[..., 2].astype(int) - r) <= tolerance

        # Combine masks
        mask = b_mask & g_mask & r_mask

        return mask

    def train_from_sumo_data(
        self,
        trajectories: list,
        num_epochs: int = 100,
        lr: float = 1e-3,
    ):
        """
        Train the learned metric predictor from SUMO ground truth data.

        Args:
            trajectories: List of dicts with keys:
                - z_current: Latent state
                - z_next: Next latent state
                - img_current: Current image
                - img_next: Next image
            num_epochs: Training epochs
            lr: Learning rate
        """
        if not self.use_latent:
            raise ValueError("Cannot train image-based reward model")

        optimizer = torch.optim.Adam(self.metric_predictor.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        print(f"Training reward model on {len(trajectories)} samples...")

        for epoch in range(num_epochs):
            total_loss = 0.0

            for traj in trajectories:
                z_current = traj['z_current']
                z_next = traj['z_next']
                img_next = traj['img_next']

                # Extract ground truth metrics from image
                gt_metrics = self._extract_traffic_metrics(img_next)
                gt_tensor = torch.tensor([
                    gt_metrics['fast_vehicle_count'],
                    gt_metrics['stopped_count'] + gt_metrics['slow_count'],
                    gt_metrics['fast_vehicle_count'] - gt_metrics['stopped_count'],  # avg_speed proxy
                ], dtype=torch.float32)

                # Predict metrics
                z_curr_flat = z_current.flatten().unsqueeze(0)
                z_next_flat = z_next.flatten().unsqueeze(0)
                z_cat = torch.cat([z_curr_flat, z_next_flat], dim=1)

                pred_metrics = self.metric_predictor(z_cat)

                # Compute loss
                loss = loss_fn(pred_metrics.squeeze(0), gt_tensor)

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(trajectories)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        print("✓ Reward model training complete!")


class SimpleTravelTimeReward:
    """
    Simplified reward model that directly estimates travel time reduction.

    Does not require neural network - uses heuristics from image statistics.
    """

    def __init__(self):
        self.prev_vehicle_positions = None

    def __call__(
        self,
        img_current: np.ndarray,
        img_next: np.ndarray,
    ) -> float:
        """
        Estimate reward based on average vehicle movement.

        Higher movement = lower travel time = higher reward
        """
        # Simple heuristic: count pixels of different colors
        # More green/cyan (fast) = good
        # More red/orange (slow) = bad

        # Count fast-moving vehicles (green, cyan hues)
        fast_mask = (
            (img_next[..., 1] > 200) |  # Green channel high
            (img_next[..., 2] > 200)    # Blue channel high (cyan)
        )
        num_fast = np.sum(fast_mask)

        # Count slow/stopped vehicles (red, orange hues)
        slow_mask = (
            (img_next[..., 0] > 200) &  # Red channel high
            (img_next[..., 1] < 100)    # Green channel low
        )
        num_slow = np.sum(slow_mask)

        # Compute reward
        # Goal: maximize fast vehicles, minimize slow vehicles
        reward = (num_fast / 1000.0) - (num_slow / 500.0)

        return reward
